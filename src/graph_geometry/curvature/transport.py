"""Endpoint-pair optimal transport: measures, ground costs and solvers.

Every transport curvature on an ordered pair ``(x, y)`` is the same pipeline::

    endpoint-pair measure builder   -> (mu_x, mu_y), each owning its support
        -> ground-cost builder      -> ordered distances supp(mu_x) x supp(mu_y)
        -> transport solver         -> cost (and optionally the plan)
        -> curvature normalisation  -> kappa(x, y) from the cost and d(x, y)

The measure at each endpoint is a kernel followed by a distribution,
``mu_z = D(K(z), z)``, and the two endpoints may use different kernels:
Ollivier and Lin-Lu-Yau apply one rule at both, while Eidi-Jost reads the
in-neighbourhood at ``x`` and the out-neighbourhood at ``y`` with no idleness.
All three are configurations of :class:`KernelPairMeasureBuilder`. An empty
selected neighbourhood falls back to the Dirac measure at the endpoint.

A :class:`DiscreteMeasure` owns its support and its mass together, so the
alignment of support order, probability vector and cost-matrix rows is
structural rather than a convention two lists have to keep. The orchestration --
shortest-path lengths, measure caching, batching over ``proc`` -- lives in
:class:`~graph_geometry.curvature.base.TransportCurvatureEngine`.

These are internal numerical protocols; the public curvature contract is in
:mod:`~graph_geometry.curvature.model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol

import networkx as nx
import numpy as np

from ..graph.semantics import GraphSemantics
from .errors import CurvatureContractError, CurvatureDomainError, CurvatureInputError
from .kernels import check_kernel_domain
from .types import DistributionFn, OTSolverFn, TransitionKernelFn
from .validate import validate_edge_attributes

__all__ = [
    "MASS_TOLERANCE",
    "DiscreteMeasure",
    "TransportResult",
    "PairMeasureBuilder",
    "GroundCostBuilder",
    "TransportSolver",
    "CostScalableTransportSolver",
    "OTCurvatureConfig",
    "BoundPairMeasures",
    "EndpointKernel",
    "NodeKernel",
    "DirectionalKernel",
    "NoIdlenessDistribution",
    "directional_neighbourhood_mass",
    "as_endpoint_kernel",
    "KernelPairMeasureBuilder",
    "InOutPairMeasureBuilder",
    "ShortestPathGroundCost",
    "EMDTransportSolver",
    "LegacyTransportSolver",
]

#: Allowed deviation of a measure's total mass from 1.
MASS_TOLERANCE = 1e-9


@dataclass(frozen=True)
class DiscreteMeasure:
    """A probability measure: a support and its masses, validated together."""

    support: tuple
    mass: np.ndarray

    def __post_init__(self) -> None:
        support = tuple(self.support)
        mass = np.array(self.mass, dtype=float)
        if mass.ndim != 1:
            raise CurvatureContractError(
                f"a measure's mass must be one-dimensional; got shape {mass.shape}."
            )
        if len(support) == 0:
            raise CurvatureContractError("a measure needs a non-empty support.")
        if len(support) != mass.shape[0]:
            raise CurvatureContractError(
                f"measure support has {len(support)} node(s) but {mass.shape[0]} "
                f"mass value(s); support and mass must be aligned."
            )
        if len(set(support)) != len(support):
            raise CurvatureContractError(f"measure support repeats a node: {support!r}.")
        if not np.all(np.isfinite(mass)):
            raise CurvatureContractError(f"measure mass must be finite; got {mass!r}.")
        if np.any(mass < 0.0):
            raise CurvatureContractError(f"measure mass must be non-negative; got {mass!r}.")
        total = float(mass.sum())
        if abs(total - 1.0) > MASS_TOLERANCE:
            raise CurvatureContractError(
                f"measure mass must total 1; got {total!r} on support {support!r}."
            )
        mass.setflags(write=False)
        object.__setattr__(self, "support", support)
        object.__setattr__(self, "mass", mass)

    @classmethod
    def dirac(cls, node: Any) -> "DiscreteMeasure":
        """All mass on ``node``."""
        return cls((node,), np.array([1.0]))

    def as_dict(self) -> dict:
        return {node: float(m) for node, m in zip(self.support, self.mass)}


@dataclass(frozen=True)
class TransportResult:
    """A transport cost, with the optimal plan when one was requested."""

    cost: float
    plan: Optional[np.ndarray] = None


# ── protocols ───────────────────────────────────────────────────────────────


class PairMeasureBuilder(Protocol):
    def __call__(
        self,
        G: nx.Graph,
        source: object,
        target: object,
        *,
        semantics: GraphSemantics,
    ) -> tuple[DiscreteMeasure, DiscreteMeasure]: ...


class GroundCostBuilder(Protocol):
    def __call__(
        self,
        source_measure: DiscreteMeasure,
        target_measure: DiscreteMeasure,
        *,
        lengths: Mapping[object, Mapping[object, float]],
    ) -> np.ndarray: ...


class TransportSolver(Protocol):
    """Solve using the original ground-cost units by default.

    A solver may additionally implement ``for_cost_scale(scale)`` to opt into
    numerical conditioning; see :class:`CostScalableTransportSolver`.
    """

    def __call__(
        self,
        source_mass: np.ndarray,
        target_mass: np.ndarray,
        cost_matrix: np.ndarray,
        *,
        return_plan: bool = False,
    ) -> TransportResult: ...


class CostScalableTransportSolver(TransportSolver, Protocol):
    """Optional, explicit support for dividing the cost matrix by ``scale``.

    ``for_cost_scale`` receives a finite positive scale and returns a solver
    configured for ``C / scale``, without modifying the original solver. Its
    returned cost must equal the original cost divided by ``scale`` and its
    transport plan must solve the original problem. The engine restores the
    cost's original units; it does not rescale the plan.

    Homogeneous exact solvers can return ``self``. Regularised solvers must
    also transform parameters in cost units (e.g. ``reg / scale`` for
    Sinkhorn). Returning ``None`` declines scaling for this call; omitting
    the method declines it altogether. A method returning another callable
    does not need to inherit this protocol.
    """

    def for_cost_scale(self, scale: float) -> Optional[TransportSolver]: ...


# ── endpoint measures: kernel -> distribution, per endpoint role ────────────
#
# The measure at an endpoint is built in two steps, as in the four-component
# decomposition of the transport family:
#
#     kernel        nu_z = K(z)       a probability measure on the neighbourhood
#                                     the kernel selects, or None when it is empty
#     distribution  mu_z = D(nu_z, z) idleness / self-mass; None -> delta_z
#
# The generalisation is that the two endpoints of an ordered pair (x, y) may use
# different kernels. Ollivier and Lin-Lu-Yau use one rule at both endpoints;
# Eidi-Jost uses the in-neighbourhood at x and the out-neighbourhood at y, with
# no idleness. An empty selected neighbourhood always falls back to delta_z.


@dataclass(frozen=True)
class BoundPairMeasures:
    """Measures cached once per node for one graph, selected by endpoint role."""

    source_measures: Mapping
    target_measures: Mapping

    def __call__(self, source: object, target: object) -> tuple[DiscreteMeasure, DiscreteMeasure]:
        return self.source_measures[source], self.target_measures[target]


class EndpointKernel(Protocol):
    """``nu_z``: a measure on the selected neighbourhood of ``z``, or ``None``."""

    def validate(self, G: nx.Graph, semantics: GraphSemantics, where: str) -> None: ...

    def prepare(self, G: nx.Graph) -> Any: ...

    def neighbourhood_measure(
        self, G: nx.Graph, node: object, *, semantics: GraphSemantics, context: Any
    ) -> Optional[DiscreteMeasure]: ...


@dataclass
class NodeKernel:
    """Adapt a node transition kernel ``(node, G, neighbors, weight_attr) -> array``.

    The support is the node's full neighbourhood (the union of predecessors and
    successors on a digraph), with a zero entry for every neighbour the rule
    gives no mass -- the support Ollivier and Lin-Lu-Yau have always used. A
    node the rule cannot leave (isolated, or a sink under ``out`` / a source
    under ``in``, reported as an all-zero vector) has an empty selected
    neighbourhood.

    The support is listed in the graph's own node order (``G.nodes()``), not by
    comparing labels: NetworkX accepts any hashable label, and a graph mixing
    ``0``, ``"a"`` and ``("t", 1)`` has no natural order in Python 3. The order
    is deterministic for a given graph, and labels are never converted, so node
    identity is unchanged.
    """

    kernel: TransitionKernelFn

    def validate(self, G: nx.Graph, semantics: GraphSemantics, where: str) -> None:
        validate_edge_attributes(G, weight=semantics.weight_attr, where=where)
        check_kernel_domain(self.kernel, G, where=where)

    def prepare(self, G: nx.Graph) -> dict:
        position = {node: index for index, node in enumerate(G.nodes())}.__getitem__
        if G.is_directed():
            return {
                node: sorted(set(G.predecessors(node)) | set(G.successors(node)), key=position)
                for node in G.nodes()
            }
        return {node: sorted(G.neighbors(node), key=position) for node in G.nodes()}

    def neighbourhood_measure(self, G, node, *, semantics, context):
        nbrs = context[node]
        if not nbrs:
            return None
        probs = np.asarray(self.kernel(node, G, context, semantics.weight_attr), dtype=float)
        if probs.shape != (len(nbrs),):
            if probs.shape == (1,) and probs[0] == 1.0:
                return None  # the historical "stay at the node" form
            raise CurvatureContractError(
                f"transition kernel {type(self.kernel).__name__} returned "
                f"{probs.shape[0]} value(s) for node {node!r}, which has "
                f"{len(nbrs)} neighbour(s)."
            )
        if not probs.any():
            return None
        return DiscreteMeasure(tuple(nbrs), probs)


def directional_neighbourhood_mass(
    G: nx.DiGraph, node: Any, direction: str, weight: str = "weight"
) -> Optional[dict]:
    """Weight-proportional mass on the in- or out-neighbours, ``None`` if there are none.

    All-zero weights give the uniform measure. Negative weights raise: optimal
    transport needs a probability measure, and a negative weight gives negative
    mass that can still sum to 1 and slip past a mass check.
    """
    if direction == "in":
        nbrs = list(G.predecessors(node))
        w = {z: float(G[z][node].get(weight, 1.0)) for z in nbrs}
    elif direction == "out":
        nbrs = list(G.successors(node))
        w = {z: float(G[node][z].get(weight, 1.0)) for z in nbrs}
    else:
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")

    negative = {z: v for z, v in w.items() if v < 0}
    if negative:
        raise CurvatureInputError(
            f"negative {weight!r} on edges incident to {node!r}: {negative}. "
            "Optimal transport needs a probability measure, and negative "
            "weights give negative mass. For a signed graph, either use |w| and "
            "keep the sign as a separate attribute, split into sign layers, or "
            "use the signed double cover — see graph_geometry.curvature.directed."
        )
    if not nbrs:
        return None
    total = sum(w.values())
    if total <= 0:                    # all weights zero -> uniform
        return {z: 1.0 / len(nbrs) for z in nbrs}
    return {z: v / total for z, v in w.items()}


@dataclass
class DirectionalKernel:
    """The in- or out-neighbourhood measure, support in adjacency order.

    Unlike :class:`NodeKernel` the support holds only the selected neighbours.
    Zero weights are allowed (all-zero gives the uniform measure); negative
    weights are rejected.
    """

    direction: str

    def __post_init__(self) -> None:
        if self.direction not in ("in", "out"):
            raise ValueError(f"direction must be 'in' or 'out', got {self.direction!r}")

    def validate(self, G: nx.Graph, semantics: GraphSemantics, where: str) -> None:
        if not G.is_directed():
            raise CurvatureDomainError(
                f"{where} reads the {self.direction}-neighbourhood, so it needs a "
                f"directed graph; got an undirected graph."
            )
        # Sign is left to the measure, whose message names the node and points
        # at the signed double cover.
        validate_edge_attributes(
            G, weight=semantics.weight_attr, where=where, positive_weight=False
        )

    def prepare(self, G: nx.Graph) -> None:
        return None

    def neighbourhood_measure(self, G, node, *, semantics, context):
        mass = directional_neighbourhood_mass(G, node, self.direction, semantics.weight_attr)
        if mass is None:
            return None
        return DiscreteMeasure(tuple(mass), np.array([mass[u] for u in mass]))


@dataclass
class NoIdlenessDistribution:
    """``mu_z = nu_z`` with no self-mass; an empty neighbourhood gives ``delta_z``."""

    def measure(self, nu: Optional[DiscreteMeasure], node: Any) -> DiscreteMeasure:
        return DiscreteMeasure.dirac(node) if nu is None else nu

    def __call__(self, kernel: np.ndarray) -> list:
        return list(kernel)


def as_endpoint_kernel(kernel: Any) -> Any:
    """An endpoint kernel as-is; a node transition kernel wrapped in :class:`NodeKernel`."""
    if kernel is None:
        return None
    if hasattr(kernel, "neighbourhood_measure"):
        return kernel
    return NodeKernel(kernel)


@dataclass
class KernelPairMeasureBuilder:
    """Endpoint measures ``mu_z = D(K(z), z)``, with a kernel per endpoint role.

    ``kernel`` builds the measure at the source ``x``; ``target_kernel`` (default:
    the same kernel) builds it at the target ``y``. Either may be a node
    transition kernel (wrapped in :class:`NodeKernel`) or an endpoint kernel such
    as :class:`DirectionalKernel`.

    A distribution with a ``measure(nu, node)`` method maps the neighbourhood
    measure itself. Otherwise the historical convention applies: the
    distribution maps the neighbourhood masses to a list with the self-mass
    appended, and the endpoint takes the final support position -- the order the
    signed Kantorovich-Rubinstein program relies on. Either way an empty selected
    neighbourhood gives the Dirac measure at the endpoint.
    """

    kernel: Any
    distribution: DistributionFn
    target_kernel: Any = None

    def __post_init__(self) -> None:
        self._source = as_endpoint_kernel(self.kernel)
        self._target = (
            self._source if self.target_kernel is None
            else as_endpoint_kernel(self.target_kernel)
        )

    @property
    def symmetric(self) -> bool:
        return self._target is self._source

    def _kernels(self) -> tuple:
        return (self._source,) if self.symmetric else (self._source, self._target)

    def validate(self, G: nx.Graph, semantics: GraphSemantics, where: str) -> None:
        for kernel in self._kernels():
            kernel.validate(G, semantics, where)

    def _measure(self, kernel, G, node, semantics, context) -> DiscreteMeasure:
        nu = kernel.neighbourhood_measure(G, node, semantics=semantics, context=context)
        distribute = getattr(self.distribution, "measure", None)
        if distribute is not None:
            return distribute(nu, node)
        if nu is None:
            return DiscreteMeasure.dirac(node)
        return DiscreteMeasure(nu.support + (node,), np.array(self.distribution(nu.mass)))

    def _role_measures(self, kernel, G, semantics, nodes) -> dict:
        context = kernel.prepare(G)
        return {
            node: self._measure(kernel, G, node, semantics, context)
            for node in G.nodes()
            if nodes is None or node in nodes
        }

    def bind(
        self, G: nx.Graph, semantics: GraphSemantics, nodes: Optional[set] = None
    ) -> BoundPairMeasures:
        """Compute each needed node's measure once per role (``nodes=None``: all)."""
        source = self._role_measures(self._source, G, semantics, nodes)
        if self.symmetric:
            return BoundPairMeasures(source, source)
        return BoundPairMeasures(source, self._role_measures(self._target, G, semantics, nodes))

    def __call__(self, G, source, target, *, semantics):
        return (
            self._measure(self._source, G, source, semantics, self._source.prepare(G)),
            self._measure(self._target, G, target, semantics, self._target.prepare(G)),
        )


def InOutPairMeasureBuilder() -> KernelPairMeasureBuilder:  # noqa: N802 - kept class-like name
    """Eidi-Jost endpoint measures: in-neighbourhood at ``x``, out at ``y``, no idleness."""
    return KernelPairMeasureBuilder(
        kernel=DirectionalKernel("in"),
        distribution=NoIdlenessDistribution(),
        target_kernel=DirectionalKernel("out"),
    )


# ── ground cost ─────────────────────────────────────────────────────────────


@dataclass
class ShortestPathGroundCost:
    """Ordered shortest-path distances from the first support to the second.

    Never symmetrised: on a digraph row ``u`` and column ``v`` hold ``d(u, v)``.
    An unreachable pair makes the transport cost undefined, which is a domain
    error, not a numerical one.
    """

    def __call__(self, source_measure, target_measure, *, lengths) -> np.ndarray:
        cost = np.zeros((len(source_measure.support), len(target_measure.support)))
        for i, u in enumerate(source_measure.support):
            row = lengths.get(u, {})
            for j, v in enumerate(target_measure.support):
                if v not in row:
                    raise CurvatureDomainError(
                        f"OT curvature needs the graph to be (strongly) connected "
                        f"between the two measures: node {v!r} is unreachable from "
                        f"{u!r}, so the transport cost is undefined. Select a "
                        f"strongly connected component explicitly, or use a "
                        f"definition that supports this topology (e.g. "
                        f"method='eidi_jost' on a digraph)."
                    )
                cost[i][j] = row[v]
        return cost


# ── solvers ─────────────────────────────────────────────────────────────────


@dataclass
class EMDTransportSolver:
    """Exact earth mover's distance (POT). The plan is computed only on request."""

    def for_cost_scale(self, scale: float) -> EMDTransportSolver:
        """EMD is homogeneous in the cost; its minimisers are unchanged."""
        return self

    def __call__(self, source_mass, target_mass, cost_matrix, *, return_plan=False):
        import ot

        a = np.array(source_mass, dtype=float)
        b = np.array(target_mass, dtype=float)
        M = np.array(cost_matrix, dtype=float)
        if return_plan:
            plan = ot.emd(a, b, M)
            return TransportResult(cost=float((plan * M).sum()), plan=plan)
        return TransportResult(cost=float(ot.emd2(a, b, M)))


@dataclass
class LegacyTransportSolver:
    """Adapt a ``(x, y, cost) -> float`` solver from a :class:`CurvatureConfig`.

    Scaling is enabled only if the wrapped solver implements
    ``for_cost_scale`` with the same contract as the modern solver protocol.
    """

    solver: OTSolverFn

    def for_cost_scale(self, scale: float) -> Optional[LegacyTransportSolver]:
        rescale = getattr(self.solver, "for_cost_scale", None)
        if rescale is None:
            return None
        if not callable(rescale):
            raise CurvatureContractError("solver.for_cost_scale must be callable.")
        solver = rescale(scale)
        if solver is None:
            return None
        if not callable(solver):
            raise CurvatureContractError("solver.for_cost_scale must return a callable or None.")
        return LegacyTransportSolver(solver)

    def __call__(self, source_mass, target_mass, cost_matrix, *, return_plan=False):
        cost = self.solver(
            np.array(source_mass, dtype=float), np.array(target_mass, dtype=float),
            np.array(cost_matrix, dtype=float),
        )
        return TransportResult(cost=cost)


# ── the four components, as one configuration ───────────────────────────────


@dataclass(frozen=True)
class OTCurvatureConfig:
    """A transport curvature as its four components.

    =========  ================  ==================================================
    component  field             role
    =========  ================  ==================================================
    ``K_c``    ``kernel``        neighbourhood measure at the source (and at the
                                 target unless ``target_kernel`` is given)
    ``D_c``    ``distribution``  idleness / self-mass: ``mu_z = D(K(z), z)``
    ``S_c``    ``solver``        transport between the two endpoint measures
    ``F_c``    ``curvature``     ``(transport cost, d(x, y)) -> kappa``
    =========  ================  ==================================================

    Further fields: ``target_kernel`` (a different K_c at the target, as in
    Eidi-Jost); ``ground_cost`` (the ordered cost between supports, shortest-path
    distance by default); ``measures`` (an advanced replacement for
    ``kernel``/``distribution``/``target_kernel`` by any pair measure builder);
    ``return_plan`` (keep the transport plan); ``degenerate_distance`` (below
    this ``d(x, y)`` the curvature is defined as 0; ``None``, the default and
    what every shipped definition uses, disables it. The historical absolute
    1e-7 guard made kappa depend on the unit of the metric).

    Cost conditioning is opt-in on ``solver`` through ``for_cost_scale``;
    otherwise the solver always receives the original cost matrix.
    """

    name: str
    kernel: Any = None
    distribution: Any = None
    solver: Any = None
    curvature: Optional[Callable[[float, float], float]] = None
    target_kernel: Any = None
    ground_cost: Any = field(default_factory=lambda: ShortestPathGroundCost())
    measures: Any = None
    return_plan: bool = False
    degenerate_distance: Optional[float] = None

    def __post_init__(self) -> None:
        from .errors import CurvatureConfigurationError

        missing = [slot for slot in ("solver", "curvature") if getattr(self, slot) is None]
        if missing:
            raise CurvatureConfigurationError(f"{self.name}: missing component(s) {missing}.")
        if self.measures is not None:
            given = [s for s in ("kernel", "distribution", "target_kernel")
                     if getattr(self, s) is not None]
            if given:
                raise CurvatureConfigurationError(
                    f"{self.name}: 'measures' replaces {given}; give one or the other."
                )
            builder = self.measures
        else:
            if self.kernel is None or self.distribution is None:
                raise CurvatureConfigurationError(
                    f"{self.name}: a transport curvature needs a kernel (K_c) and a "
                    f"distribution (D_c), or a pair measure builder as 'measures'."
                )
            builder = KernelPairMeasureBuilder(
                self.kernel, self.distribution, target_kernel=self.target_kernel
            )
        object.__setattr__(self, "_measure_builder", builder)

    @property
    def measure_builder(self) -> Any:
        """The pair measure builder the engine runs: K_c and D_c at each endpoint."""
        return self._measure_builder
