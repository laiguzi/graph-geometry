"""The shared endpoint-pair optimal-transport engine.

:class:`TransportCurvatureEngine` runs any :class:`~graph_geometry.curvature.transport.OTCurvatureConfig`
-- Ollivier, Lin-Lu-Yau and Eidi-Jost alike -- on edges or ordered node pairs.
It owns everything the definitions share: attribute validation, shortest-path
lengths, per-node measure caching, cost-matrix construction, batching over
``proc`` and normalisation. It returns raw values and never annotates the graph;
the registry path applies the result contract in the evaluator.

:class:`CurvatureConfig` (the four legacy slots: node kernel, distribution,
``(x, y, cost) -> float`` solver, curvature-from-OT) and :class:`RicciCurvature`
remain as compatibility adapters over the engine for one deprecation cycle.
:func:`write_curvature` / :func:`write_node_means` are the annotation helpers
for code that drives formulas directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Iterable, Optional, Union

import networkx as nx

from ..graph.semantics import GraphSemantics
from .errors import CurvatureConfigurationError, CurvatureContractError, CurvatureDomainError
from .transport import LegacyTransportSolver, OTCurvatureConfig, TransportResult
from .types import CurvatureFromOT, DistributionFn, OTSolverFn, TransitionKernelFn
from .validate import validate_edge_attributes

__all__ = [
    "TransportCurvatureEngine",
    "incident_means",
    "CurvatureConfig",
    "RicciCurvature",
    "write_node_means",
    "write_curvature",
]


def incident_means(G: nx.Graph, value_of) -> dict:
    """Mean of each node's incident edge values; nodes without values are omitted.

    ``value_of(u, v)`` returns the value of stored edge ``(u, v)`` or ``None``.
    Outgoing edges are summed before incoming ones on a digraph. The single
    implementation behind :func:`write_node_means` and the evaluator's
    ``incident_node_means``, so the two agree bit for bit.
    """
    means: dict = {}
    directed = G.is_directed()
    for n in G.nodes():
        total = 0.0
        count = 0
        incident = (
            [(n, nbr) for nbr in G.successors(n)] + [(nbr, n) for nbr in G.predecessors(n)]
            if directed else [(n, nbr) for nbr in G.neighbors(n)]
        )
        for u, v in incident:
            value = value_of(u, v)
            if value is not None:
                total += value
                count += 1
        if count > 0:
            means[n] = total / count
    return means


def write_node_means(G: nx.Graph, attr: str = "ricciCurvature") -> None:
    """Write each node's ``attr`` as the mean of its incident edge ``attr`` values.

    Nodes with no incident edge value are left unset (and a value left over from
    an earlier topology, e.g. before surgery, is cleared).
    """
    means = incident_means(G, lambda u, v: G[u][v].get(attr))
    for n in G.nodes():
        G.nodes[n].pop(attr, None)
    for n, value in means.items():
        G.nodes[n][attr] = value


def write_curvature(G: nx.Graph, edge_values: dict, attr: str = "ricciCurvature") -> dict:
    """Write edge values + node means onto ``G`` and return the edge dict (floats)."""
    result = {e: float(v) for e, v in edge_values.items()}
    for (u, v), value in result.items():
        G[u][v][attr] = value
    write_node_means(G, attr)
    return result


#: For solvers that opt into conditioning, use a power-of-two scale when the
#: binary exponent of ``d(x, y)`` is outside [-_SCALE_BAND, _SCALE_BAND].
#: Other solvers always receive the original costs.
_SCALE_BAND = 8


def _cost_scale(distance: float) -> float:
    exponent = math.frexp(distance)[1] - 1          # 2**exponent <= d < 2**(exponent+1)
    if -_SCALE_BAND <= exponent <= _SCALE_BAND:
        return 1.0
    return math.ldexp(1.0, exponent)


#: Why the pool pickles per chunk rather than once per worker.
#:
#: ``pool.map(self._batch, batches)`` pickles the engine -- including the
#: all-pairs distance dict and the cached measures -- for every chunk, which
#: looks wasteful. Sending it instead through a pool ``initializer`` was measured
#: and is *slower*: at 700 nodes and 244,650 pairs the per-chunk form reached
#: 2.46x at ``proc=4`` and 2.69x at ``proc=8``, against 2.03x and 1.95x for the
#: initializer. Per-chunk transfer overlaps with computation and lets the pool
#: load-balance, while the initializer serialises the same bytes at worker
#: start-up. Do not "optimise" this back without re-measuring.


class TransportCurvatureEngine:
    """Evaluate a transport curvature on edges or ordered pairs.

    Parameters
    ----------
    G :
        The graph. It is read, never annotated.
    config :
        The four components -- kernel ``K_c``, distribution ``D_c``, solver
        ``S_c`` and curvature function ``F_c`` -- plus the ground cost.
    semantics :
        Attribute roles; ``distance_attr`` is the metric behind shortest paths
        and the ground cost, ``weight_attr`` is read by the measure builder.
    proc :
        Worker processes for the per-pair solves. Output is identical for every
        ``proc``.
    """

    def __init__(
        self,
        G: nx.Graph,
        config: OTCurvatureConfig,
        semantics: Optional[GraphSemantics] = None,
        proc: int = 1,
    ):
        if not isinstance(config, OTCurvatureConfig):
            raise CurvatureConfigurationError(
                f"config must be an OTCurvatureConfig; got {type(config).__name__}."
            )
        self.G = G
        self.config = config
        self.semantics = semantics if semantics is not None else GraphSemantics()
        self.proc = proc
        self.lengths: dict = {}
        self.measures: Any = None

    @property
    def where(self) -> str:
        return f"OT curvature ({self.config.name})"

    # ── preparation ────────────────────────────────────────────────────

    def shortest_path_lengths(self) -> dict:
        """Shortest-path lengths under the distance attribute.

        Only the lengths are built. An earlier version called
        ``all_pairs_dijkstra_path``, materialised every path as a list, and
        re-derived each length by summing along it -- the number Dijkstra had
        already computed. The paths were never read, and because ``proc > 1``
        pickles the engine they were shipped to every worker too. Dropping them
        measured 2.6x faster and 3x less memory on the lengths alone.
        """
        return dict(
            nx.all_pairs_dijkstra_path_length(self.G, weight=self.semantics.distance_attr)
        )

    def _prepare(self, endpoints: Optional[set] = None) -> None:
        validate_edge_attributes(
            self.G, distance=self.semantics.distance_attr, where=self.where
        )
        builder = self.config.measure_builder
        validate = getattr(builder, "validate", None)
        if validate is not None:
            validate(self.G, self.semantics, self.where)
        self.lengths = self.shortest_path_lengths()
        bind = getattr(builder, "bind", None)
        if bind is not None:
            self.measures = bind(self.G, self.semantics, nodes=endpoints)
        else:
            self.measures = _UnboundMeasures(builder, self.G, self.semantics)

    # ── one pair ───────────────────────────────────────────────────────

    def _evaluate(self, source: Any, target: Any, details: bool = False):
        if source == target:
            # self-loop: curvature defined as 0 (the evaluator rejects loops
            # before a registered definition gets here)
            return (0.0, None) if details else 0.0
        row = self.lengths.get(source, {})
        if target not in row:
            raise CurvatureDomainError(
                f"{self.where}: node {target!r} is unreachable from {source!r}, so "
                f"d(x, y) is undefined on this pair (the graph is not strongly "
                f"connected)."
            )
        distance = row[target]
        degenerate = self.config.degenerate_distance
        if degenerate is not None and distance < degenerate:
            return (0.0, None) if details else 0.0
        mu, nu = self.measures(source, target)
        cost = self.config.ground_cost(mu, nu, lengths=self.lengths)
        # Conditioning is a solver capability: regularisation and other
        # nonlinear cost dependence must be transformed by the solver itself.
        # Missing/declined support preserves the original problem verbatim.
        solver = self.config.solver
        scale = _cost_scale(distance)
        if scale != 1.0:
            rescale = getattr(solver, "for_cost_scale", None)
            if rescale is not None and not callable(rescale):
                raise CurvatureContractError("solver.for_cost_scale must be callable.")
            scaled_solver = rescale(scale) if rescale is not None else None
            if scaled_solver is None:
                scale = 1.0
            elif not callable(scaled_solver):
                raise CurvatureContractError(
                    "solver.for_cost_scale must return a callable or None."
                )
            else:
                solver = scaled_solver
        result = solver(
            mu.mass, nu.mass, cost if scale == 1.0 else cost / scale,
            return_plan=self.config.return_plan or details,
        )
        if not isinstance(result, TransportResult):
            result = TransportResult(cost=result)
        if scale != 1.0:
            result = TransportResult(cost=result.cost * scale, plan=result.plan)
        value = self.config.curvature(result.cost, distance)
        if details:
            return value, (mu, nu, cost, result)
        return value

    def _batch(self, pair_batch):
        return [((s, t), self._evaluate(s, t)) for s, t in pair_batch]

    def _detail_batch(self, pair_batch):
        return [((s, t), self._evaluate(s, t, details=True)) for s, t in pair_batch]

    def _batch_size(self, n_pairs: int) -> int:
        """Pairs per work unit, derived from ``proc``.

        Previously a fixed 2000, which made every graph under 2000 edges a
        single batch: ``pool.map`` then gave all the work to one worker and
        ``proc > 1`` gave no speedup. Several chunks per worker keeps one slow
        pair from stalling a core.
        """
        if self.proc > 1 and n_pairs > 1:
            n_chunks = min(n_pairs, self.proc * 4)
            return math.ceil(n_pairs / n_chunks)
        return 2000

    def _run(self, pairs: list, worker) -> list:
        batch_size = self._batch_size(len(pairs))
        batches = [pairs[i:i + batch_size] for i in range(0, len(pairs), batch_size)]
        if self.proc > 1 and len(batches) > 1:
            with Pool(processes=self.proc) as pool:
                result = pool.map(worker, batches)
        else:
            result = [worker(b) for b in batches]
        return [item for sublist in result for item in sublist]

    # ── public API ─────────────────────────────────────────────────────

    def _check_pairs(self, pairs: Iterable) -> list:
        pairs = [tuple(pair) for pair in pairs]
        for x, y in pairs:
            for node in (x, y):
                if node not in self.G:
                    raise CurvatureConfigurationError(
                        f"{self.where}: node {node!r} is not in the graph."
                    )
            if x == y:
                raise CurvatureConfigurationError(
                    f"{self.where}: curvature needs two distinct nodes, got "
                    f"({x!r}, {x!r}); kappa is defined through d(x, y) > 0."
                )
        return pairs

    def compute_edges(self) -> dict:
        """``{(u, v): kappa}`` on every edge, in ``G.edges()`` order."""
        edges = list(self.G.edges())
        self._prepare()
        return dict(self._run(edges, self._batch))

    def compute_pairs(self, pairs: Iterable) -> dict:
        """``{(x, y): kappa}`` on the given ordered pairs, in request order."""
        pairs = self._check_pairs(pairs)
        self._prepare({node for pair in pairs for node in pair})
        return dict(self._run(pairs, self._batch))

    def transport_details(self, pairs: Optional[Iterable] = None) -> dict:
        """``{(x, y): (kappa, (mu_x, mu_y, cost_matrix, TransportResult))}``.

        The transport plan is always requested. ``pairs=None`` means every edge.
        A diagnostic view over the same orchestration as :meth:`compute_edges`.
        """
        if pairs is None:
            pairs = list(self.G.edges())
            self._prepare()
        else:
            pairs = self._check_pairs(pairs)
            self._prepare({node for pair in pairs for node in pair})
        return dict(self._run(pairs, self._detail_batch))


@dataclass(frozen=True)
class _UnboundMeasures:
    """Adapter for a pair builder without ``bind``: build per pair, no caching."""

    builder: Any
    G: nx.Graph
    semantics: GraphSemantics

    def __call__(self, source, target):
        return self.builder(self.G, source, target, semantics=self.semantics)


# ── compatibility adapters ──────────────────────────────────────────────────


@dataclass
class CurvatureConfig:
    """Legacy bundle of the four OT callables (node kernel form).

    Kept for one deprecation cycle; :meth:`to_ot_config` converts it into the
    endpoint-pair form the engine runs. New built-ins construct
    :class:`~graph_geometry.curvature.transport.OTCurvatureConfig` directly.
    """

    name: str
    transition_kernel: TransitionKernelFn
    distribution: DistributionFn
    ot_solver: OTSolverFn
    curvature_from_ot: CurvatureFromOT

    def __repr__(self) -> str:
        return f"CurvatureConfig(name={self.name!r})"

    def to_ot_config(self) -> OTCurvatureConfig:
        return OTCurvatureConfig(
            name=self.name,
            kernel=self.transition_kernel,
            distribution=self.distribution,
            solver=LegacyTransportSolver(self.ot_solver),
            curvature=self.curvature_from_ot,
        )


class RicciCurvature:
    """Compatibility front end: run a (legacy or endpoint-pair) OT config.

    ``weight``/``distance`` name the edge attributes; missing values count as 1.
    """

    def __init__(
        self,
        G: nx.Graph,
        config: Union[CurvatureConfig, OTCurvatureConfig],
        weight: str = "weight",
        distance: str = "distance",
        proc: int = 1,
    ):
        self.G = G
        self.config = config
        self.weight = weight
        self.distance = distance
        self.proc = proc
        ot_config = config.to_ot_config() if isinstance(config, CurvatureConfig) else config
        self.engine = TransportCurvatureEngine(
            G, ot_config, GraphSemantics(weight_attr=weight, distance_attr=distance), proc
        )

    def _all_pairs_shortest_path(self) -> dict:
        return self.engine.shortest_path_lengths()

    def _batch_size(self, n_edges: int) -> int:
        return self.engine._batch_size(n_edges)

    def compute_pairs(self, pairs) -> dict:
        """Curvature on arbitrary node pairs; does **not** touch the graph.

        Optimal-transport curvature is defined for any two distinct nodes with
        ``d(x, y) > 0`` -- an edge is not required, only the two measures and
        the metric between their supports.
        """
        return self.engine.compute_pairs(pairs)

    def compute_edge_values(self) -> dict:
        """Curvature on every edge; does **not** touch the graph."""
        return self.engine.compute_edges()

    def compute_ricci_curvature(self) -> dict:
        """Compute curvature, write ``ricciCurvature`` attrs, return edge dict.

        Low-level convenience for code that drives the engine directly (for
        example a plugin swapping one :class:`CurvatureConfig` slot). The
        registry path leaves annotation to the evaluator.
        """
        return write_curvature(self.G, self.compute_edge_values())
