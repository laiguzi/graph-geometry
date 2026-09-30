"""Run several curvature configurations on one graph and tabulate the results.

:func:`compare_curvature` evaluates *k* :class:`CurvatureRequest` objects over
the same graph, each on its own copy. Requests are identified by their
**labels**, so two configurations of one definition can be compared::

    import graph_geometry as gg

    cmp = gg.compare_curvature(
        G,
        requests=[
            gg.CurvatureRequest("ollivier", {"alpha": 0.0, "kernel": "out"}, label="ORC a=0"),
            gg.CurvatureRequest("ollivier", {"alpha": 0.5, "kernel": "out"}, label="ORC a=0.5"),
        ],
        semantics=gg.GraphSemantics(),
        proc=4,
    )
    cmp.summary()      # per label: n, min, median, max, %negative, failure kind
    cmp.agreement()    # pairwise: spearman, pearson, sign agreement
    cmp.to_csv("curvature_comparison.csv")

Two behaviours to know about:

**A request that fails is recorded, not raised.** A definition can be undefined
on a given graph: ``lin_lu_yau`` on a digraph that is not strongly connected has
unreachable transport pairs, which is the correct answer rather than a bug. Each
run records either a :class:`CurvatureResult` or a :class:`CurvatureFailure`
whose ``kind`` is one of ``configuration``, ``inapplicable``, ``invalid_input``,
``numerical_failure``, ``contract_failure`` or ``internal_failure``; the rest of
the comparison still runs.

**Pairwise statistics use the common support.** They are computed over the edges
where both runs produced a value, and report how many that was.

The historical form ``compare_curvature(G, ["forman_node_weighted", "lin_lu_yau"],
params={...})`` remains available: each name becomes a request labelled by that
name.
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from statistics import median
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Union

import networkx as nx
import numpy as np

from .curvature.errors import CurvatureConfigurationError, failure_kind
from .curvature.evaluator import _resolve, evaluate_curvature
from .curvature.model import CurvatureFailure, CurvatureRequest, CurvatureResult
from .curvature.registry import list_curvatures, lookup_curvature
from .graph.semantics import GraphSemantics

__all__ = [
    "CurvatureRun",
    "CurvatureComparison",
    "compare_curvature",
    "FlowRun",
    "FlowComparison",
    "compare_flow",
    "DEFAULT_COMPARISON_STATUSES",
]

#: Tiers compared when no request is given. Experimental definitions never
#: enter a default comparison.
DEFAULT_COMPARISON_STATUSES = frozenset({"core", "specialized"})


def _rankdata(values: np.ndarray, tol: float = 0.0) -> np.ndarray:
    """Average-tie ranks (the one piece of scipy.stats this module would need).

    Values within ``tol`` of the smallest value of their tie group share that
    group's average rank; ``tol=0.0`` ties exactly equal values only, as
    ``scipy.stats.rankdata`` does. A group is anchored at its smallest value
    rather than grown pairwise, so a slowly increasing sequence is not chained
    into one tie.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] - values[order[start]] <= tol:
            stop += 1
        # 1-based positions start+1 .. stop share their mean
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    return ranks


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, or nan when either side is constant."""
    if len(a) < 2:
        return float("nan")
    sa, sb = a.std(), b.std()
    if sa < 1e-15 or sb < 1e-15:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


#: Curvature magnitudes at or below this are treated as exactly zero when a
#: *sign* is taken. The OT solvers do not share an error floor: ``ollivier``
#: goes through POT's ``emd2`` and returns exact 0.0 on a flat edge, while
#: ``lin_lu_yau`` goes through a CVXPY solve. With the default LP solver it is
#: exact (max 6.7e-16), but with a conic solver such as SCS it leaves a residual
#: around 1e-5 (measured: mean +2.3e-7, max 2.6e-5). Taking ``np.sign``
#: of that residual made two methods that are mathematically identical on a
#: flat graph look like they never agreed: on ``C8``, where every edge is truly
#: 0, ``sign_agreement`` was 0.00 instead of 1.00, and on a 4x4 grid
#: ``pct_negative`` was 16.7% instead of 0%.
#:
#: The value is the tolerance for CVXPY-derived values, i.e. it is set by the
#: weakest solver a caller may select, not by the default one. A curvature
#: smaller than this is not distinguishable from solver noise, so it is not
#: given a sign.
SIGN_TOLERANCE = 1e-4


def _tolerant_sign(x: np.ndarray, tol: float) -> np.ndarray:
    """``np.sign``, with everything inside +/-``tol`` collapsed to zero."""
    return np.sign(np.where(np.abs(x) <= tol, 0.0, x))


@dataclass(frozen=True)
class CurvatureRun:
    """One request of a comparison and what it produced."""

    label: str
    request: CurvatureRequest
    outcome: Union[CurvatureResult, CurvatureFailure]

    @property
    def ok(self) -> bool:
        return isinstance(self.outcome, CurvatureResult)

    def to_dict(self) -> dict:
        outcome = self.outcome
        return {
            "label": self.label,
            "request": self.request.to_dict(),
            "ok": self.ok,
            "resolved": outcome.resolved.to_dict() if outcome.resolved else None,
            "failure": None if self.ok else outcome.to_dict(),
        }


@dataclass(frozen=True)
class CurvatureComparison:
    """Curvature from several requests over one graph. See :func:`compare_curvature`."""

    runs: tuple[CurvatureRun, ...]
    n_edges: int = 0

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Mapping],
        *,
        errors: Optional[Mapping[str, str]] = None,
        n_edges: int = 0,
        methods: Optional[Iterable[str]] = None,
    ) -> "CurvatureComparison":
        """Build a comparison from precomputed ``{label: {edge: kappa}}`` values.

        For statistics on values computed elsewhere; runs carry no resolved
        record. ``errors`` maps labels to messages recorded as
        ``internal_failure``.
        """
        errors = dict(errors or {})
        labels = list(methods) if methods is not None else [*values, *errors]
        runs = []
        for label in labels:
            if label in errors:
                outcome = CurvatureFailure(
                    kind="internal_failure", exception_type="Error",
                    message=str(errors[label]),
                )
            else:
                edge_values = dict(values[label])
                outcome = CurvatureResult(
                    values=edge_values, incident_node_means={}, resolved=None
                )
            runs.append(CurvatureRun(label, CurvatureRequest(label, label=label), outcome))
        return cls(runs=tuple(runs), n_edges=n_edges)

    # ── compatibility views (read-only, keyed by label) ──────────────────

    @property
    def labels(self) -> list[str]:
        return [run.label for run in self.runs]

    @property
    def methods(self) -> list[str]:
        """Request labels, in request order (historical name)."""
        return self.labels

    @property
    def values(self) -> Mapping[str, dict]:
        """``{label: {edge: kappa}}`` for successful runs (read-only view)."""
        return MappingProxyType(
            {run.label: dict(run.outcome.values) for run in self.runs if run.ok}
        )

    @property
    def errors(self) -> Mapping[str, str]:
        """``{label: "ExceptionType: message"}`` for failed runs (read-only view)."""
        return MappingProxyType(
            {run.label: str(run.outcome) for run in self.runs if not run.ok}
        )

    @property
    def failures(self) -> Mapping[str, CurvatureFailure]:
        return MappingProxyType(
            {run.label: run.outcome for run in self.runs if not run.ok}
        )

    def run(self, label: str) -> CurvatureRun:
        for run in self.runs:
            if run.label == label:
                return run
        raise KeyError(label)

    # ── per-run view ─────────────────────────────────────────────────────

    def summary(self, tol: float = SIGN_TOLERANCE) -> list[dict]:
        """One row per request label: applicability, distribution, failure kind.

        ``computable`` and ``of`` count the edges with a value and the edges in
        the graph. Because every successful evaluation must satisfy the
        exact-edge result contract, coverage is all or nothing: a successful run
        has ``computable == of``, and a failed one has ``computable == 0`` with
        its ``failure_kind`` and ``error`` set. The two fields are kept for
        compatibility; read them as "did this definition apply", not as a
        partial-coverage ratio.

        ``pct_negative`` counts values below ``-tol`` rather than below 0, so a
        flat edge reported as -1e-6 by a conic solver is not miscounted as
        negatively curved. Pass ``tol=0.0`` for the raw sign.
        """
        rows = []
        for run in self.runs:
            base = {
                "method": run.label,
                "definition": (
                    run.outcome.resolved.method if run.outcome.resolved else run.request.method
                ),
            }
            if not run.ok:
                rows.append({**base, "computable": 0, "of": self.n_edges,
                             "min": None, "median": None, "max": None,
                             "mean": None, "pct_negative": None,
                             "failure_kind": run.outcome.kind,
                             "error": str(run.outcome)})
                continue
            vals = list(run.outcome.values.values())
            if not vals:
                rows.append({**base, "computable": 0, "of": self.n_edges,
                             "min": None, "median": None, "max": None,
                             "mean": None, "pct_negative": None,
                             "failure_kind": "", "error": ""})
                continue
            negative = sum(1 for v in vals if v < -tol)
            rows.append({
                **base,
                "computable": len(vals),
                "of": self.n_edges,
                "min": min(vals), "median": median(vals), "max": max(vals),
                "mean": sum(vals) / len(vals),
                "pct_negative": 100.0 * negative / len(vals),
                "failure_kind": "",
                "error": "",
            })
        return rows

    # ── pairwise view ────────────────────────────────────────────────────

    def agreement(self, tol: float = SIGN_TOLERANCE) -> list[dict]:
        """Pairwise statistics over the edges where both runs succeeded.

        ``sign_agreement`` compares signs *after* collapsing magnitudes within
        ``tol`` to zero, so two definitions that are both mathematically flat on
        an edge agree there whatever residue their solvers leave. ``spearman``
        applies the same tolerance to ranks: values within ``tol`` of each other
        are tied. Without it, roundoff decides the order of mathematically equal
        values, and two definitions that are exactly proportional on a graph
        with few distinct values were reported with Spearman 0.91 instead of 1.
        Pass ``tol=0.0`` for the raw sign and exact-equality ties.
        """
        values = self.values
        ok = [run.label for run in self.runs if run.ok]
        rows = []
        for i, a in enumerate(ok):
            for b in ok[i + 1:]:
                shared = sorted(set(values[a]) & set(values[b]), key=str)
                xa = np.array([values[a][e] for e in shared], dtype=float)
                xb = np.array([values[b][e] for e in shared], dtype=float)
                if len(shared) == 0:
                    rows.append({"method_a": a, "method_b": b, "n": 0,
                                 "spearman": float("nan"), "pearson": float("nan"),
                                 "sign_agreement": float("nan")})
                    continue
                same_sign = float(
                    np.mean(_tolerant_sign(xa, tol) == _tolerant_sign(xb, tol))
                )
                rows.append({
                    "method_a": a, "method_b": b, "n": len(shared),
                    "spearman": _corr(_rankdata(xa, tol), _rankdata(xb, tol)),
                    "pearson": _corr(xa, xb),
                    "sign_agreement": same_sign,
                })
        return rows

    # ── raw view / export ────────────────────────────────────────────────

    def rows(self) -> list[dict]:
        """Tidy long format: one row per (edge, label)."""
        out = []
        for run in self.runs:
            if not run.ok:
                continue
            for (u, v), value in run.outcome.values.items():
                out.append({"source": u, "target": v, "method": run.label, "kappa": value})
        return out

    def wide(self) -> dict:
        """``{edge: {label: kappa}}`` -- missing where a run could not compute."""
        table: dict = {}
        for run in self.runs:
            if not run.ok:
                continue
            for edge, value in run.outcome.values.items():
                table.setdefault(edge, {})[run.label] = value
        return table

    def csv_text(self) -> str:
        """The wide table as CSV text: one row per edge, one column per label."""
        ok = [run.label for run in self.runs if run.ok]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["source", "target", *ok])
        for (u, v), per_label in self.wide().items():
            writer.writerow([u, v, *(per_label.get(m, "") for m in ok)])
        return buffer.getvalue()

    def to_csv(self, path: str) -> str:
        """Write :meth:`csv_text` to ``path`` and return the path."""
        with open(path, "w", newline="") as fh:
            fh.write(self.csv_text())
        return path

    def to_dict(self) -> dict:
        """JSON-safe provenance: every request, its resolution and outcome kind."""
        return {"n_edges": self.n_edges, "runs": [run.to_dict() for run in self.runs]}


def _legacy_requests(methods, params) -> list[CurvatureRequest]:
    if methods is None:
        names = [s.name for s in list_curvatures(statuses=DEFAULT_COMPARISON_STATUSES)]
    elif isinstance(methods, str):
        names = [methods]
    else:
        names = list(methods)
    for name in names:
        if isinstance(name, CurvatureRequest):
            continue
        lookup_curvature(name)  # unknown names fail before any computation
    extra: dict = params or {}
    return [
        name if isinstance(name, CurvatureRequest)
        else CurvatureRequest(name, dict(extra.get(name, {})), label=name)
        for name in names
    ]


def compare_curvature(
    G: nx.Graph,
    methods: Optional[Iterable[Union[str, CurvatureRequest]]] = None,
    *,
    requests: Optional[Iterable[CurvatureRequest]] = None,
    semantics: Optional[GraphSemantics] = None,
    proc: int = 1,
    weight: str = "weight",
    distance: str = "distance",
    params: Optional[dict] = None,
) -> CurvatureComparison:
    """Evaluate several curvature requests on ``G`` and collect the results.

    Parameters
    ----------
    requests:
        The configurations to compare. Labels (``request.label`` or, when
        absent, the method name) must be unique.
    semantics:
        Attribute roles shared by every request. Defaults to
        ``GraphSemantics(weight_attr=weight, distance_attr=distance)``.
    proc:
        Worker processes for each evaluation.
    methods, params:
        Historical form: method names (default: every core and specialized
        definition) and per-name keyword parameters.

    ``G`` is never mutated: each request runs on its own copy. Unknown method
    names and duplicate labels raise
    :class:`~graph_geometry.curvature.errors.CurvatureConfigurationError` before
    anything is computed; every other failure is recorded on its run.
    """
    if requests is not None:
        if methods is not None or params is not None:
            raise CurvatureConfigurationError(
                "pass either requests= or the historical methods=/params=, not both."
            )
        request_list = list(requests)
        for request in request_list:
            if not isinstance(request, CurvatureRequest):
                raise CurvatureConfigurationError(
                    f"requests must contain CurvatureRequest objects; got {request!r}."
                )
        legacy = False
    else:
        request_list = _legacy_requests(methods, params)
        legacy = True

    labels = [request.display_label for request in request_list]
    duplicated = sorted({label for label in labels if labels.count(label) > 1})
    if duplicated:
        raise CurvatureConfigurationError(
            f"comparison labels must be unique; duplicated: {duplicated}. Give "
            f"each CurvatureRequest a distinct label."
        )
    if semantics is None:
        semantics = GraphSemantics(weight_attr=weight, distance_attr=distance)

    runs = []
    for request in request_list:
        H = G.copy()
        resolved = None
        try:
            resolved = _resolve(H, request, semantics, "edges", legacy=legacy)
            outcome = evaluate_curvature(H, resolved, proc=proc)
        except Exception as exc:  # noqa: BLE001 - recorded with its failure kind
            outcome = CurvatureFailure(
                kind=failure_kind(exc),
                exception_type=type(exc).__name__,
                message=str(exc),
                resolved=resolved,
            )
        runs.append(CurvatureRun(label=request.display_label, request=request, outcome=outcome))

    return CurvatureComparison(runs=tuple(runs), n_edges=G.number_of_edges())


@dataclass(frozen=True)
class FlowRun:
    """One flow equation of a flow comparison and what it produced."""

    label: str
    equation: Any
    outcome: Any          # FlowResult or CurvatureFailure

    @property
    def ok(self) -> bool:
        return not isinstance(self.outcome, CurvatureFailure)


def _initial_spread(result) -> Optional[float]:
    if not result.snapshots:
        return None
    rc = nx.get_edge_attributes(result.snapshots[0], "ricciCurvature")
    return (max(rc.values()) - min(rc.values())) if rc else None


@dataclass(frozen=True)
class FlowComparison(Mapping):
    """One curvature evolved under several flow equations. See :func:`compare_flow`.

    Also a read-only mapping ``label -> {"convergence", "snapshots", "result"}``
    (or ``{"error": ...}``), the form :func:`compare_flow` returned before.
    """

    runs: tuple[FlowRun, ...]

    # ── historical mapping view ──────────────────────────────────────────

    def __getitem__(self, label: str) -> dict:
        run = self.run(label)
        if not run.ok:
            return {"error": str(run.outcome), "failure_kind": run.outcome.kind}
        return {"convergence": list(run.outcome.convergence),
                "snapshots": run.outcome.snapshots, "result": run.outcome}

    def __iter__(self):
        return iter(self.labels)

    def __len__(self) -> int:
        return len(self.runs)

    # ── views ────────────────────────────────────────────────────────────

    @property
    def labels(self) -> list[str]:
        return [run.label for run in self.runs]

    def run(self, label: str) -> FlowRun:
        for run in self.runs:
            if run.label == label:
                return run
        raise KeyError(label)

    @property
    def results(self) -> Mapping:
        return MappingProxyType({r.label: r.outcome for r in self.runs if r.ok})

    @property
    def failures(self) -> Mapping:
        return MappingProxyType({r.label: r.outcome for r in self.runs if not r.ok})

    def summary(self) -> list[dict]:
        """One row per equation: how the run stopped and how far the spread moved.

        ``spread_ratio`` is final over initial curvature spread; below 1 the flow
        contracted curvature differences, above 1 it amplified them.
        """
        rows = []
        for run in self.runs:
            if not run.ok:
                rows.append({"equation": run.label, "termination_reason": None,
                             "iterations_completed": 0, "initial_spread": None,
                             "final_spread": None, "spread_ratio": None,
                             "final_edges": None, "failure_kind": run.outcome.kind,
                             "error": str(run.outcome)})
                continue
            result = run.outcome
            initial = _initial_spread(result)
            final = result.convergence[-1] if result.convergence else initial
            rows.append({
                "equation": run.label,
                "termination_reason": result.termination_reason,
                "iterations_completed": result.iterations_completed,
                "initial_spread": initial,
                "final_spread": final,
                "spread_ratio": (final / initial) if initial and final is not None else None,
                "final_edges": result.snapshots[-1].number_of_edges() if result.snapshots else None,
                "failure_kind": "",
                "error": "",
            })
        return rows

    def convergence_rows(self) -> list[dict]:
        """Long format ``{"equation", "iteration", "spread"}`` (iteration = committed state)."""
        return [
            {"equation": run.label, "iteration": i, "spread": value}
            for run in self.runs if run.ok
            for i, value in enumerate(run.outcome.convergence, start=1)
        ]

    def csv_text(self) -> str:
        """Convergence side by side: one row per iteration, one column per equation."""
        ok = [run for run in self.runs if run.ok]
        length = max((len(run.outcome.convergence) for run in ok), default=0)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(["iteration", *(run.label for run in ok)])
        for i in range(length):
            writer.writerow([i + 1, *(
                run.outcome.convergence[i] if i < len(run.outcome.convergence) else ""
                for run in ok
            )])
        return buffer.getvalue()

    def to_csv(self, path: str) -> str:
        with open(path, "w", newline="") as fh:
            fh.write(self.csv_text())
        return path

    def to_dict(self) -> dict:
        return {"runs": [
            {"label": run.label, "equation": run.equation if isinstance(run.equation, str)
             else repr(run.equation), "ok": run.ok,
             "result": run.outcome.to_dict()}
            for run in self.runs
        ]}


def compare_flow(
    G: nx.Graph,
    equations: Mapping,
    *,
    curvature: Any = "lin_lu_yau",
    semantics: Optional[GraphSemantics] = None,
    evolve: str = "weight",
    proc: int = 1,
    iterations: int = 20,
    step: float = 0.05,
    **run_kwargs: Any,
) -> FlowComparison:
    """Evolve one curvature under several flow equations on copies of ``G``.

    ``equations`` maps a label to anything :func:`~graph_geometry.flow.resolve_flow`
    accepts -- a preset name, a callable, or a math/LaTeX expression string.
    ``curvature`` is a name or a :class:`CurvatureRequest`; ``semantics``,
    ``evolve``, ``proc`` and ``run_kwargs`` (``delta``, ``surgery``,
    ``early_stop``, ...) are shared, so the equation is the only thing that varies.

    A failing equation is recorded with its failure kind -- ``configuration``
    for an invalid equation, ``numerical_failure`` when the first step already
    diverges, and the curvature's own kinds -- and the others still run.
    """
    from .flow.equations import resolve_flow
    from .simulator import RicciFlowSimulator

    runs = []
    for label, spec in equations.items():
        try:
            equation = resolve_flow(spec)
        except Exception as exc:  # noqa: BLE001 - recorded as a configuration failure
            runs.append(FlowRun(label, spec, CurvatureFailure(
                kind="configuration", exception_type=type(exc).__name__, message=str(exc),
            )))
            continue
        try:
            sim = RicciFlowSimulator(
                G.copy(), curvature=curvature, flow_equation=equation,
                evolve=evolve, proc=proc, semantics=semantics,
            )
            sim.run(iterations=iterations, step=step, **run_kwargs)
            outcome = sim.result
        except Exception as exc:  # noqa: BLE001 - recorded with its failure kind
            outcome = CurvatureFailure(
                kind=failure_kind(exc), exception_type=type(exc).__name__, message=str(exc),
            )
        runs.append(FlowRun(label, spec, outcome))
    return FlowComparison(runs=tuple(runs))
