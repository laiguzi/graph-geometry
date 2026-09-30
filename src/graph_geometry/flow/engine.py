"""Ricci flow engine (orchestrator).

Ported from ``ricciflow_sim/core/ricciflow.py``, delegating curvature to
graph_geometry's curvature layer (:func:`graph_geometry.curvature.curvature`) and
weight updates to a pluggable flow equation.

Semantics (per Bai-Li-Liu-Lai 2509.19989): only ``weight`` evolves; ``distance``
stays FIXED (used for shortest paths / the OT cost matrix).

Committed states
----------------
An iteration produces exactly **one** committed state, and every observable ---
snapshot, convergence value, ``self.G``, the progress callback and the iteration
GEXF --- is derived from that one graph. This is what makes the outputs of a run
describe the same instant of the trajectory rather than three different ones.
The order below is normative and pinned by ``tests/test_flow_committed_state.py``:

1. Compute the evolving-quantity-weighted mean curvature of the committed state.
2. Update the evolving quantity on every edge of a *candidate* copy.
3. Recompute curvature on the candidate (the **pre-surgery candidate**).
4. First early-stop checkpoint, on that candidate.
5. If it converged, commit it and stop; scheduled surgery does not run --- an
   already converged graph must not be changed by surgery.
6. Otherwise, if no surgery is scheduled this iteration, commit the candidate.
7. Otherwise apply surgery to the evolving attribute.
8. If surgery removed every edge, commit the empty graph and stop as
   ``exhausted`` --- not ``converged``: exhaustion is not a mathematical result.
9. Otherwise recompute curvature (the topology changed) and commit. If the
   curvature is undefined on the post-surgery topology (for example a digraph
   that is no longer strongly connected), nothing is committed for this
   iteration and the run stops as ``undefined``, keeping every earlier state.

If the curvature solver itself fails on a candidate (step 3 or 9) -- typically a
flow that has driven the evolving metric so far out of scale that one graph
holds edge lengths beyond double precision's range -- that candidate is not
committed either and the run stops as ``numerical``, again keeping every
earlier state.
10. Second early-stop checkpoint, on the committed post-surgery state: surgery
    can itself produce a converged graph and that must be detectable at once.

Curvature is recomputed *before* each commit, so a failure there leaves the
iteration uncommitted rather than half-recorded.

Surgery is scheduled by the number of completed updates: the strategy's
``should_apply(n)`` is asked after the ``n``-th update (``n >= 1``), so
``interval=5`` operates on the result of the 5th update and the post-surgery
graph is committed state 5.

Curvature configuration
-----------------------
A named curvature (or a :class:`~graph_geometry.curvature.CurvatureRequest`) is
resolved **once** per run against the prepared input graph, with the run's
:class:`~graph_geometry.graph.semantics.GraphSemantics`. Every later evaluation
re-uses that record and repeats only the dynamic checks -- loops, attribute
values, reachability -- because surgery can change the topology. The resolved
record and its warnings are attached to :class:`FlowResult`. If the flow evolves
an attribute the definition does not consume, the run continues but warns that
the curvature may be invariant under the flow.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import networkx as nx

from ..curvature.errors import (
    CurvatureConfigurationError,
    CurvatureDomainError,
    CurvatureNumericalError,
    CurvatureWarning,
)
from ..curvature.evaluator import _resolve, curvature as legacy_curvature, evaluate_curvature
from ..curvature.model import CurvatureRequest, ResolvedCurvature
from ..curvature.registry import get_curvature_spec
from ..curvature.validate import validate_edge_attributes, validate_simple_graph
from ..graph.semantics import GraphSemantics
from .equations import FlowSpec, resolve_flow
from .surgery import SurgerySpec, resolve_surgery

__all__ = ["RicciFlow", "FlowResult", "FlowDivergenceError", "TERMINATION_REASONS"]


class FlowDivergenceError(CurvatureNumericalError, ValueError):
    """The first update already drives the evolving quantity non-positive.

    Raised only at iteration 0, where nothing beyond the input was committed;
    later divergence ends the run with ``termination_reason="diverged"`` instead.
    A :class:`ValueError` too, as before it was typed.
    """

#: Why a run stopped. ``converged`` and ``exhausted`` are deliberately distinct:
#: a graph whose edges were all removed by surgery has not converged, it has run
#: out of graph, and reporting that as convergence would be a false result.
#: ``diverged`` is the same kind of distinction: the update would have driven the
#: evolving quantity to zero or below, so the run stops with every committed
#: state still valid, rather than raising and discarding them. A diverging
#: pairing is a real comparative result -- normalized weight flow contracts under
#: Ollivier and Lin-Lu-Yau but diverges under the node-weighted Forman convention
#: on the same graph -- so the states that show it happening are worth keeping.
#: ``undefined`` is the topological counterpart: surgery produced a graph on
#: which the curvature is not defined (e.g. a digraph that lost strong
#: connectivity under an OT definition), so that iteration is not committed and
#: the states before it are returned. ``numerical`` is the solver-side
#: counterpart: the curvature solver failed on a candidate state (for example an
#: unnormalized flow that collapsed the metric until edge lengths in one graph
#: differ by ~1e20, beyond what a double-precision LP can resolve).
TERMINATION_REASONS = (
    "converged", "exhausted", "degenerate", "diverged", "undefined", "numerical",
    "iterations",
)


def _graph_info(G: nx.Graph) -> str:
    kind = "Directed" if G.is_directed() else "Undirected"
    return f"{kind} graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"


@dataclass(frozen=True)
class FlowResult:
    """The outcome of one run: committed states plus why it stopped.

    Unpacks as the historical ``(snapshots, convergence)`` pair, so
    ``snapshots, convergence = flow.run(...)`` keeps working.

    Attributes
    ----------
    snapshots :
        One graph per committed state: index 0 is the initial state (carrying
        ``ricciCurvature`` and ``original_RC``), then one per completed
        iteration.
    convergence :
        Curvature spread ``max(kappa) - min(kappa)`` of ``snapshots[i]``, at
        index ``i - 1``. An ``exhausted`` final iteration contributes a snapshot
        but **no** convergence value: spread is undefined on an empty edge set,
        and encoding it as ``0.0`` or ``NaN`` would read as convergence. That is
        the one case where ``len(convergence) != len(snapshots) - 1``.
    termination_reason :
        One of :data:`TERMINATION_REASONS`. ``iterations`` means the requested
        count was exhausted; ``degenerate`` means the evolving quantity summed
        to zero while edges remained, so no update was defined; ``diverged``
        means an update after the first committed iteration would have made the
        evolving quantity non-positive or non-finite, so that candidate was not
        committed (on the very first update this raises
        :class:`FlowDivergenceError` instead, as there is no valid state to keep);
        ``undefined`` means the curvature is not defined on the topology a
        surgery produced, and ``numerical`` that the curvature solver failed on
        a candidate state; in both cases that iteration was not committed.
    iterations_completed :
        Number of committed iterations, i.e. ``len(snapshots) - 1``.
    diagnosis :
        Empty unless the run ended abnormally. For ``diverged`` it names the
        iteration, the step, the largest step that would have survived, and the
        recent spread trend; for ``undefined`` it carries the domain error; for
        ``numerical`` the solver error and the scale of the evolving quantity.
    resolved :
        The resolved curvature configuration every committed state was computed
        with, or ``None`` when the curvature was a raw ``CurvatureConfig`` or
        callable.
    warnings :
        Resolution warnings plus flow-level caveats (for example an evolving
        attribute the curvature does not consume).
    evolve, weight_attr, distance_attr :
        Which role the flow evolved (``"weight"`` or ``"distance"``) and the
        edge attributes holding the two roles, so a reader of the snapshots can
        tell an interaction weight from a metric distance. ``None`` on a result
        built by hand without them; :attr:`evolving_attr` is then ``None`` too.
    """

    snapshots: list = field(default_factory=list)
    convergence: list = field(default_factory=list)
    termination_reason: str = "iterations"
    iterations_completed: int = 0
    diagnosis: str = ""
    resolved: Optional[ResolvedCurvature] = None
    warnings: tuple = ()
    evolve: Optional[str] = None
    weight_attr: Optional[str] = None
    distance_attr: Optional[str] = None

    @property
    def evolving_attr(self) -> Optional[str]:
        """The edge attribute the flow updated, or ``None`` when not recorded."""
        if self.evolve == "weight":
            return self.weight_attr
        if self.evolve == "distance":
            return self.distance_attr
        return None

    @property
    def coupled(self) -> bool:
        """True when one attribute plays both roles, so evolving it moves both."""
        return self.weight_attr is not None and self.weight_attr == self.distance_attr

    def to_dict(self) -> dict:
        """JSON-safe record of the run (the snapshots themselves are exported separately).

        ``convergence`` is given as ``[[committed state index, spread], ...]``,
        so an exhausted final state (which has no spread) cannot shift the
        alignment.
        """
        return {
            "termination_reason": self.termination_reason,
            "iterations_completed": self.iterations_completed,
            "convergence": [[i + 1, value] for i, value in enumerate(self.convergence)],
            "diagnosis": self.diagnosis,
            "resolved": self.resolved.to_dict() if self.resolved is not None else None,
            "warnings": list(self.warnings),
            "evolve": self.evolve,
            "evolving_attr": self.evolving_attr,
            "coupled": self.coupled,
        }

    def __iter__(self):
        yield self.snapshots
        yield self.convergence

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index):
        return (self.snapshots, self.convergence)[index]


class RicciFlow:
    """Run Ricci flow on a graph, returning in-memory snapshots.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
        Input graph (copied; the caller's graph is not mutated).
    curvature : str | CurvatureRequest | CurvatureConfig | callable
        A registered name or a :class:`CurvatureRequest` is resolved once per
        run and re-evaluated each iteration. A raw ``CurvatureConfig`` or legacy
        compute callable is still accepted and has no resolved record.
    flow_equation : str | callable
        Update rule: ``"normalized"`` | ``"unnormalized"`` | ``"additive"``, a
        callable ``(K, q, step, K_avg) -> delta``, or a free-form math/LaTeX
        expression string (see :class:`~graph_geometry.flow.ExpressionFlow`).
    weight, distance : str
        Edge attribute names for the two quantities. Set them to the **same**
        name (e.g. ``distance="weight"``) to *couple* the metric to the weight --
        one quantity then drives both the transition kernel and the shortest-path
        / OT metric, and evolving it moves both.
    evolve : str
        Which quantity the flow updates: ``"weight"`` (default; Bai-Li-Liu-Lai,
        distance fixed) or ``"distance"`` (classic metric flow, weight fixed).
        The other stays fixed. For ``"distance"`` prefer multiplicative equations
        (normalized/unnormalized) so distances stay positive.
    proc : int
        Parallel processes for curvature computation.
    semantics : GraphSemantics, optional
        Attribute roles. When given, its ``weight_attr``/``distance_attr`` are
        the flow's weight and distance; passing a different ``weight`` or
        ``distance`` as well is a configuration error.
    **curvature_params
        Parameters of a *named* curvature (e.g. ``kernel``, ``beta``,
        ``alpha``); with a :class:`CurvatureRequest` put them in the request.
    """

    def __init__(
        self,
        G: nx.Graph,
        curvature: Any = "lin_lu_yau",
        flow_equation: FlowSpec = "normalized",
        weight: str = "weight",
        distance: str = "distance",
        evolve: str = "weight",
        proc: int = 1,
        semantics: Optional[GraphSemantics] = None,
        **curvature_params: Any,
    ):
        if evolve not in ("weight", "distance"):
            raise CurvatureConfigurationError(
                f"evolve must be 'weight' or 'distance', got {evolve!r}"
            )
        if semantics is not None:
            if not isinstance(semantics, GraphSemantics):
                raise CurvatureConfigurationError(
                    f"semantics must be a GraphSemantics; got {type(semantics).__name__}."
                )
            for given, default, field_name in (
                (weight, "weight", "weight_attr"), (distance, "distance", "distance_attr"),
            ):
                if given != default and given != getattr(semantics, field_name):
                    raise CurvatureConfigurationError(
                        f"{given!r} contradicts semantics.{field_name}="
                        f"{getattr(semantics, field_name)!r}; set the role once."
                    )
            weight, distance = semantics.weight_attr, semantics.distance_attr
        if isinstance(curvature, CurvatureRequest) and curvature_params:
            raise CurvatureConfigurationError(
                "pass curvature parameters inside the CurvatureRequest, not as keywords."
            )
        if isinstance(curvature, (str, CurvatureRequest)) and semantics is None:
            semantics = GraphSemantics(
                weight_attr=weight,
                distance_attr=distance,
                node_weight_attr=curvature_params.pop("node_weight_attr", None),
            )
        self.G = G.copy()
        self.curvature = curvature
        self.curvature_params = curvature_params
        self.semantics = semantics
        self.flow_equation = resolve_flow(flow_equation)
        self.weight = weight
        self.distance = distance
        self.evolve = evolve
        self.proc = proc
        self.termination_reason: Optional[str] = None
        self.resolved: Optional[ResolvedCurvature] = None
        self.warnings: tuple = ()

    @property
    def _evolving_attr(self) -> str:
        """The edge attribute the flow updates (the other stays fixed)."""
        return self.weight if self.evolve == "weight" else self.distance

    def _resolve_curvature(self) -> None:
        """Resolve a named curvature once, and warn if the flow cannot move it."""
        self.resolved = None
        self.warnings = ()
        if not isinstance(self.curvature, (str, CurvatureRequest)):
            return
        if isinstance(self.curvature, CurvatureRequest):
            request, legacy = self.curvature, False
        else:
            request = CurvatureRequest(self.curvature, dict(self.curvature_params))
            legacy = True
        self.resolved = _resolve(self.G, request, self.semantics, "edges", legacy=legacy)
        notes = list(self.resolved.warnings)

        evolving = {"edge_weight" if self.evolve == "weight" else "edge_distance"}
        if self.semantics.coupled:
            evolving = {"edge_weight", "edge_distance"}
        consumes = get_curvature_spec(self.resolved.method).capabilities.consumes
        if not evolving & consumes:
            note = (
                f"the flow evolves {self._evolving_attr!r} "
                f"({', '.join(sorted(evolving))}), which {self.resolved.method!r} does "
                f"not consume ({', '.join(sorted(consumes))}): its curvature may be "
                f"invariant under the flow, so only surgery can change it."
            )
            warnings.warn(note, CurvatureWarning, stacklevel=3)
            notes.append(note)
        self.warnings = tuple(notes)

    def _compute_curvature(self, G: nx.Graph) -> None:
        """Recompute curvature, writing ``ricciCurvature`` onto ``G``."""
        if self.resolved is not None:
            evaluate_curvature(G, self.resolved, proc=self.proc, annotate=True)
            return
        legacy_curvature(
            G,
            self.curvature,
            weight=self.weight,
            distance=self.distance,
            proc=self.proc,
            **self.curvature_params,
        )

    def _largest_safe_step(self, k_avg: float, step: float, q_attr: str) -> float:
        """Largest fraction of ``step`` keeping every evolving value positive.

        Found by bisection rather than algebra, so it holds for any registered
        flow equation, including a user callable or a parsed expression.
        """

        def survives(scale: float) -> bool:
            for u, v in self.G.edges():
                q = self.G[u][v][q_attr]
                dq = self.flow_equation(
                    self.G[u][v]["ricciCurvature"], q, step * scale, k_avg
                )
                # written so that NaN fails: ``NaN <= 0`` is False, and the
                # old test certified a NaN update as safe
                if not (math.isfinite(q + dq) and q + dq > 0):
                    return False
            return True

        if not survives(0.0):
            # not even a vanishing step is valid: the equation itself returns a
            # non-finite or non-positive value, and no step size fixes that
            return 0.0
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if survives(mid):
                lo = mid
            else:
                hi = mid
        return step * lo

    def _divergence_diagnosis(
        self, exc: Exception, i: int, k_avg: float, step: float,
        q_attr: str, convergence: list,
    ) -> str:
        """Turn a bare 'non-positive weight' into a diagnosis of the run.

        The raw message reports the symptom -- one edge went negative -- which
        tells a user nothing about what to change. The step that would have been
        safe *at this iteration* is computable, and the spread history says
        whether a smaller step would actually help or merely postpone the same
        failure.
        """
        safe = self._largest_safe_step(k_avg, step, q_attr)
        lines = [
            f"Ricci flow diverged at iteration {i}: the update with step={step:g} "
            f"would make an edge's {q_attr} non-positive or non-finite, which is "
            f"not a valid {q_attr}.",
            f"  underlying check: {exc}",
        ]
        if safe == 0.0:
            lines.append(
                "  No positive step keeps every value finite and positive: the "
                "flow equation itself returns an invalid update here, so a "
                "smaller step will not help. Check the flow equation."
            )
            return "\n".join(lines)
        lines.append(
            f"  largest step that stays positive and finite at this iteration: {safe:.3g}"
        )
        recent = [c for c in convergence[-5:] if c is not None]
        if len(recent) >= 2:
            lines.append(
                f"  curvature spread over the last {len(recent)} iterations: "
                + " -> ".join(f"{c:.4g}" for c in recent)
            )
        if len(recent) >= 2 and recent[-1] > recent[0]:
            lines.append(
                "  The spread is growing, so this curvature and flow equation are "
                "diverging on this graph: a smaller step will postpone the failure, "
                "not prevent it. Try another flow equation, evolve 'distance' "
                "instead of 'weight', or use a curvature whose flow contracts here."
            )
        else:
            lines.append(f"  Re-run with step below {safe:.3g}.")
        return "\n".join(lines)

    def _validate_evolving_quantity(self, G: nx.Graph, *, where: str) -> None:
        """Require a finite positive evolving weight/metric before a solve."""
        q_attr = self._evolving_attr
        validate_edge_attributes(
            G,
            weight=self.weight if q_attr == self.weight else None,
            distance=self.distance if q_attr == self.distance else None,
            where=where,
        )

    def _numerical_diagnosis(
        self, exc: Exception, i: int, stage: str, q_attr: str
    ) -> str:
        """Explain a solver failure on a candidate, with the scale that caused it."""
        values = [
            self.G[u][v][q_attr] for u, v in self.G.edges() if q_attr in self.G[u][v]
        ]
        lines = [
            f"the curvature solver failed on the {stage} of update {i + 1}; that "
            f"state was not committed and the run stopped with every earlier "
            f"state intact.",
            f"  underlying error: {exc}",
        ]
        if values:
            low, high = min(values), max(values)
            lines.append(
                f"  last committed {q_attr}: [{low:.3g}, {high:.3g}]"
                + (f", ratio {high / low:.3g}" if low > 0 else "")
            )
            if low > 0 and high / low > 1e12:
                lines.append(
                    f"  The {q_attr} values of one graph span more than twelve "
                    f"orders of magnitude, near the resolution of double "
                    f"precision: the flow has collapsed or blown up the metric. "
                    f"A normalized flow equation keeps the total fixed."
                )
        return "\n".join(lines)

    def _curvature_label(self) -> str:
        if self.resolved is not None:
            return repr(self.resolved.method)
        return "the curvature"

    @staticmethod
    def _spread(G: nx.Graph) -> Optional[float]:
        """Curvature spread ``max - min``, or ``None`` on an empty edge set."""
        rc = nx.get_edge_attributes(G, "ricciCurvature")
        if not rc:
            return None
        return max(rc.values()) - min(rc.values())

    def run(
        self,
        iterations: int = 50,
        step: float = 0.01,
        delta: float = 1e-6,
        surgery: SurgerySpec = None,
        early_stop: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> FlowResult:
        """Run the flow and return a :class:`FlowResult`.

        The result unpacks as ``(snapshots, convergence)`` for compatibility and
        additionally carries ``termination_reason`` and ``iterations_completed``.

        ``early_stop`` (default True) halts when the spread drops below
        ``delta``; set False for fixed-length research runs. Both early-stop
        checkpoints of the committed-state order are skipped when it is False.

        The input is copied but nothing in it is silently discarded: a
        multigraph or a self-loop is refused with a ``CurvatureDomainError``, as
        in standalone evaluation. If an OT method encounters unreachable directed
        transport pairs it raises a clear ``CurvatureDomainError``; methods such
        as ``forman_directed`` and ``eidi_jost`` that support the topology
        continue on the full input graph.

        With ``save_dir``, ``origin.gexf`` is the raw input as the run received
        it and ``<i>.gexf`` is committed state ``i`` --- the same graph as
        ``snapshots[i]``, matching :func:`graph_geometry.io.save_gexf_snapshots`.
        """
        surgery_strategy = resolve_surgery(surgery)
        # the same simple-graph policy as standalone evaluation, checked before
        # the internal copy is touched: a self-loop is refused, never dropped
        validate_simple_graph(self.G, "Ricci flow")

        if verbose:
            print(f"Ricci flow on {_graph_info(self.G)}")

        if self.G.number_of_edges() == 0:
            raise CurvatureDomainError(
                "Ricci flow requires at least one non-self-loop edge; curvature "
                "flow is undefined on an empty edge set."
            )

        # ensure weight/distance exist (structure-only inputs get unit defaults;
        # source-provided values are preserved) unless the semantics say a
        # missing value is an error
        unit_weight = self.semantics is None or self.semantics.missing_weight == "unit"
        unit_distance = self.semantics is None or self.semantics.missing_distance == "unit"
        for u, v in self.G.edges():
            if unit_weight:
                self.G[u][v].setdefault(self.weight, 1.0)
            if unit_distance:
                self.G[u][v].setdefault(self.distance, 1.0)
        self._validate_evolving_quantity(self.G, where="Ricci flow input")
        self._resolve_curvature()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            nx.write_gexf(self.G, os.path.join(save_dir, "origin.gexf"))

        snapshots: list = []
        convergence: list[float] = []

        def commit(G: nx.Graph, index: int, *, record_spread: bool = True):
            """Record exactly one state; every observable derives from ``G``."""
            self.G = G
            snapshots.append(G.copy())
            spread = self._spread(G)
            if record_spread and spread is not None:
                convergence.append(spread)
            if save_dir:
                nx.write_gexf(G, os.path.join(save_dir, f"{index}.gexf"))
            return spread

        def report(i: int, spread: Optional[float], q_attr: str) -> None:
            if verbose:
                rc = nx.get_edge_attributes(self.G, "ricciCurvature")
                q = nx.get_edge_attributes(self.G, q_attr)
                if rc:
                    print(
                        f"  iter {i:4d} | RC diff: {spread:.6f} | "
                        f"RC [{min(rc.values()):.4f}, {max(rc.values()):.4f}] | "
                        f"{q_attr} [{min(q.values()):.4f}, {max(q.values()):.4f}]"
                    )
                else:
                    print(f"  iter {i:4d} | no edges remain")
            # the callback is typed for a float; an exhausted state has no spread
            if progress_callback and spread is not None:
                progress_callback(i, iterations, spread)

        # initial curvature, then the initial committed state (no spread value:
        # convergence[i - 1] belongs to snapshots[i], so index 0 contributes none)
        self._compute_curvature(self.G)
        for u, v in self.G.edges():
            self.G[u][v]["original_RC"] = self.G[u][v]["ricciCurvature"]
        commit(self.G, 0, record_spread=False)

        q_attr = self._evolving_attr  # the attribute that evolves (weight or distance)
        reason = "iterations"
        diagnosis = ""

        for i in range(iterations):
            # 1. weighted mean curvature of the committed state (keeps the
            #    normalized flow volume-preserving in whichever quantity evolves)
            total_q = sum(self.G[u][v][q_attr] for u, v in self.G.edges())
            if total_q == 0:
                reason = "exhausted" if self.G.number_of_edges() == 0 else "degenerate"
                if verbose:
                    print(f"  Stopping at iteration {i}: no {q_attr} remaining.")
                break
            sum_kq = sum(
                self.G[u][v]["ricciCurvature"] * self.G[u][v][q_attr]
                for u, v in self.G.edges()
            )
            k_avg = sum_kq / total_q

            # 2-3. update a candidate copy, then recompute curvature on it
            candidate = self.G.copy()
            for u, v in candidate.edges():
                dq = self.flow_equation(
                    candidate[u][v]["ricciCurvature"], candidate[u][v][q_attr], step, k_avg
                )
                candidate[u][v][q_attr] += dq
            try:
                self._validate_evolving_quantity(candidate, where="Ricci flow update")
            except ValueError as exc:
                detail = self._divergence_diagnosis(
                    exc, i, k_avg, step, q_attr, convergence
                )
                if i == 0:
                    # Nothing was ever committed beyond the initial state, so
                    # there is no partial result to hand back and nothing to
                    # learn from it: the configuration itself is unusable (a
                    # flow equation that inverts the quantity, a step far past
                    # any valid range). Raise, as an invalid argument would.
                    raise FlowDivergenceError(detail) from exc
                # Past the first iteration the committed states are valid and
                # are exactly what shows the divergence, so they are returned
                # rather than lost to an exception. The offending candidate is
                # never committed.
                reason = "diverged"
                diagnosis = detail
                if verbose:
                    print(f"  {diagnosis}")
                break
            try:
                self._compute_curvature(candidate)
            except CurvatureNumericalError as exc:
                # The solver could not evaluate this candidate. The committed
                # states are valid, so they are returned; the candidate is not.
                reason = "numerical"
                diagnosis = self._numerical_diagnosis(exc, i, "updated graph", q_attr)
                if verbose:
                    print(f"  {diagnosis}")
                break

            # 4-5. first checkpoint: a converged graph is not operated on
            spread = self._spread(candidate)
            if early_stop and spread is not None and spread < delta:
                commit(candidate, i + 1)
                report(i, spread, q_attr)
                reason = "converged"
                if verbose:
                    print(f"  Converged at iteration {i}.")
                break

            # 6. no surgery this iteration: the candidate is the committed state.
            #    Surgery is scheduled by completed updates (1-based): this is
            #    update i + 1, so interval=5 operates after the 5th update.
            if not surgery_strategy.should_apply(i + 1):
                commit(candidate, i + 1)
                report(i, spread, q_attr)
                continue

            # 7-8. surgery ranks the same quantity the selected flow evolves
            operated = surgery_strategy.apply(candidate, q_attr)
            if operated.number_of_edges() == 0:
                commit(operated, i + 1, record_spread=False)
                report(i, None, q_attr)
                reason = "exhausted"
                if verbose:
                    print(f"  Surgery removed every edge at iteration {i}.")
                break

            # 9. surgery changed the topology, so every stored curvature belongs
            #    to a graph that no longer exists (measured drift up to 0.19 on a
            #    barbell graph). Recompute before committing, so a failure here
            #    leaves the iteration uncommitted rather than half-recorded.
            try:
                self._compute_curvature(operated)
            except CurvatureDomainError as exc:
                # The curvature is not defined on the topology surgery produced
                # (e.g. a digraph that lost strong connectivity under an OT
                # definition). The committed states before it are valid, so
                # they are returned rather than lost to the exception, as for
                # ``diverged``; this iteration is not committed.
                reason = "undefined"
                diagnosis = (
                    f"surgery after update {i + 1} produced a graph on which "
                    f"{self._curvature_label()} is undefined; the iteration was "
                    f"not committed.\n  underlying error: {exc}"
                )
                if verbose:
                    print(f"  {diagnosis}")
                break
            except CurvatureNumericalError as exc:
                reason = "numerical"
                diagnosis = self._numerical_diagnosis(exc, i, "post-surgery graph", q_attr)
                if verbose:
                    print(f"  {diagnosis}")
                break
            spread = commit(operated, i + 1)
            report(i, spread, q_attr)

            # 10. second checkpoint: surgery may itself have produced convergence
            if early_stop and spread is not None and spread < delta:
                reason = "converged"
                if verbose:
                    print(f"  Converged after surgery at iteration {i}.")
                break

        self.termination_reason = reason
        return FlowResult(
            snapshots=snapshots,
            convergence=convergence,
            termination_reason=reason,
            iterations_completed=len(snapshots) - 1,
            diagnosis=diagnosis,
            resolved=self.resolved,
            warnings=self.warnings,
            evolve=self.evolve,
            weight_attr=self.weight,
            distance_attr=self.distance,
        )
