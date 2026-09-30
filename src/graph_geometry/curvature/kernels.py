"""Transition kernels and beta strategies.

Ported from ``ricciflow_sim/core/curvature.py``, but implemented as **picklable
callable classes** rather than closures, so a :class:`CurvatureConfig` built from
them can be shipped to worker processes (``proc > 1``).

A transition kernel is a callable ``(node, G, neighbors, weight_attr) -> np.ndarray``
returning transition probabilities over ``neighbors[node]`` (same order).

A beta strategy is a callable ``(node, G) -> float`` giving the in/out mixing
weight for directed graphs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Union

import numpy as np

from .errors import CurvatureDomainError

__all__ = [
    "EPSILON",
    "BetaSpec",
    "BetaConstant",
    "BetaDegreeProportional",
    "BetaWeightProportional",
    "BetaDict",
    "BetaNodeAttr",
    "check_beta",
    "resolve_beta",
    "MixedKernel",
    "UndirectedKernel",
    "OutKernel",
    "InKernel",
    "AutoKernel",
    "resolve_kernel",
    "kernel_domain",
    "check_kernel_domain",
]

#: Historical absolute threshold, kept only as a public name. It is no longer
#: used as a cut-off: comparing a weight sum (or a distance) against a fixed
#: 1e-7 made the curvature depend on the unit of measurement -- scaling every
#: weight by 1e-8 switched the kernel to the uniform measure and changed kappa,
#: and scaling every distance by 1e-8 set kappa to 0. Weights and distances are
#: validated finite and strictly positive, so no threshold is needed.
EPSILON = 1e-7

BetaSpec = Union[float, int, Mapping, str, Callable]


# ── Beta strategies (Bai-Li-Liu-Lai 2509.19989) ─────────────────────────────


def check_beta(value: Any, *, context: str = "beta") -> float:
    """Return ``value`` as a float, rejecting anything outside ``[0, 1]``.

    ``P = beta*P_out + (1-beta)*P_in`` is a convex combination, so it is only a
    probability measure for ``beta`` in ``[0, 1]``. Outside that range the
    result can carry **negative** mass while still summing to 1 (e.g.
    ``beta=-0.5`` gives ``[1.25, -0.25]``), which slips past a mass check and
    reaches the OT solver as a signed measure. Mirrors the ``alpha`` validation
    in :class:`~graph_geometry.curvature.ot_ricci.OllivierDistribution`.
    """
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{context} must be a number, got {value!r}")
    try:
        beta = float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{context} must be a number, got {value!r}") from e
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"{context} must be in [0, 1], got {beta}")
    return beta


@dataclass
class BetaConstant:
    """Global constant beta for all nodes."""

    value: float = 0.8

    def __post_init__(self):
        self.value = check_beta(self.value)

    def __call__(self, node: Any, G: Any) -> float:
        return float(self.value)


@dataclass
class BetaDegreeProportional:
    """beta(x) = deg_out(x) / deg(x) — each incident edge gets equal attention.

    From Bai-Li-Liu-Lai (2509.19989), Remark 3.2. Falls back to 0.5 for
    isolated nodes. In ``[0, 1]`` by construction.
    """

    def __call__(self, node: Any, G: Any) -> float:
        out_deg = G.out_degree(node)
        total_deg = G.in_degree(node) + out_deg
        return out_deg / total_deg if total_deg > 0 else 0.5


@dataclass
class BetaWeightProportional:
    """beta(x) = W_out(x) / (W_out(x) + W_in(x)) — the weighted analogue.

    Where :class:`BetaDegreeProportional` gives every incident *edge* equal
    attention, this gives every unit of incident *weight* equal attention. On a
    weighted digraph the two differ sharply: nodes with equal in/out degree but
    lopsided weights are indistinguishable to the degree version.

    In ``[0, 1]`` by construction (weights are assumed non-negative). Falls back
    to 0.5 for isolated nodes and for nodes whose incident weight is 0.
    """

    weight: str = "weight"

    def __call__(self, node: Any, G: Any) -> float:
        w_out = sum(G[node][t].get(self.weight, 1.0) for t in G.successors(node))
        w_in = sum(G[s][node].get(self.weight, 1.0) for s in G.predecessors(node))
        total = w_out + w_in
        return w_out / total if total > 0 else 0.5


@dataclass
class BetaDict:
    """Per-node beta from a mapping, with a default for missing nodes.

    Any ``collections.abc.Mapping`` is accepted, including the
    :class:`~graph_geometry.curvature.model.FrozenMapping` a resolved record
    stores in place of the caller's ``dict``.
    """

    mapping: Mapping
    default: float = 0.5

    def __post_init__(self):
        for node, value in self.mapping.items():
            check_beta(value, context=f"beta for node {node!r}")
        self.default = check_beta(self.default, context="beta default")

    def __call__(self, node: Any, G: Any) -> float:
        return float(self.mapping.get(node, self.default))


@dataclass
class BetaNodeAttr:
    """Per-node beta read from a **node attribute** on the graph.

    Lets the balancing factor travel with the graph rather than the call site,
    so it survives file formats. Two entry points already exist:

    - edge-list ``N <node> <value>`` lines, via
      ``load.from_edgelist(path, node_attr="beta")``;
    - GEXF node attributes, which ``nx.read_gexf`` restores verbatim.

    Nodes without the attribute fall back to ``default``. Values are validated
    on read, so a malformed file fails with a clear error rather than silently
    producing a signed measure.
    """

    attr: str = "beta"
    default: float = 0.5

    def __post_init__(self):
        self.default = check_beta(self.default, context="beta default")

    def __call__(self, node: Any, G: Any) -> float:
        if node not in G.nodes or self.attr not in G.nodes[node]:
            return float(self.default)
        return check_beta(
            G.nodes[node][self.attr],
            context=f"node {node!r} attribute {self.attr!r}",
        )


#: Named beta strategies. Aliases are grouped so the canonical name is first.
_BETA_STRATEGIES = {
    "degree_proportional": BetaDegreeProportional,
    "adaptive": BetaDegreeProportional,
    "balanced": BetaDegreeProportional,
    "weight_proportional": BetaWeightProportional,
    "node_attr": BetaNodeAttr,
}


def resolve_beta(spec: BetaSpec) -> Callable[[Any, Any], float]:
    """Coerce a beta spec into a ``(node, G) -> float`` callable.

    Accepts (superset of the three forks):

    - ``float``/``int`` -> :class:`BetaConstant` (validated to ``[0, 1]``);
    - a mapping (``dict``, or the frozen copy held by a resolved record) ->
      :class:`BetaDict` (every value validated);
    - ``"degree_proportional"`` / ``"adaptive"`` / ``"balanced"`` ->
      :class:`BetaDegreeProportional`;
    - ``"weight_proportional"`` -> :class:`BetaWeightProportional`;
    - ``"node_attr"`` -> :class:`BetaNodeAttr` (reads ``G.nodes[n]["beta"]``);
    - a callable, evaluated and range-checked by :class:`MixedKernel`; it must
      also be picklable for ``proc > 1``.
    """
    if callable(spec):
        return spec
    if isinstance(spec, bool):
        raise TypeError(f"Invalid beta spec: {spec!r}")
    if isinstance(spec, (int, float)):
        return BetaConstant(check_beta(spec))
    if isinstance(spec, Mapping):
        return BetaDict(spec)
    if isinstance(spec, str):
        cls = _BETA_STRATEGIES.get(spec.lower())
        if cls is not None:
            return cls()
        raise ValueError(
            f"Unknown beta strategy {spec!r}. Use a float in [0, 1], a dict, a "
            f"callable, or one of {sorted(_BETA_STRATEGIES)}."
        )
    raise TypeError(f"Invalid beta spec: {spec!r}")


# ── Transition kernels ──────────────────────────────────────────────────────


def _gamma(G: Any, i: Any, j: Any, weight_attr: str) -> float:
    return G[i][j].get(weight_attr, 1.0)


@dataclass
class MixedKernel:
    """Mixed in/out kernel: P = beta * P_out + (1 - beta) * P_in (directed).

    ``beta`` is clamped to 0 at sinks (no out-edges) and 1 at sources (no
    in-edges) so the result is always a probability measure; see
    :meth:`__call__`.
    """

    #: Graphs this kernel is defined on: see :func:`check_kernel_domain`.
    domain = "directed"

    beta: BetaSpec = 0.8

    def __post_init__(self):
        self._beta_fn = resolve_beta(self.beta)

    def __call__(self, node, G, neighbors, weight_attr) -> np.ndarray:
        beta_val = check_beta(
            self._beta_fn(node, G), context=f"beta returned for node {node!r}"
        )
        in_nbrs = list(G.predecessors(node))
        out_nbrs = list(G.successors(node))

        if not in_nbrs and not out_nbrs:
            return np.array([1.0])

        in_sum = sum(_gamma(G, n, node, weight_attr) for n in in_nbrs)
        if in_sum > 0:
            in_probs = {n: _gamma(G, n, node, weight_attr) / in_sum for n in in_nbrs}
        elif in_nbrs:
            in_probs = {n: 1.0 / len(in_nbrs) for n in in_nbrs}
        else:
            in_probs = {}

        out_sum = sum(_gamma(G, node, n, weight_attr) for n in out_nbrs)
        if out_sum > 0:
            out_probs = {n: _gamma(G, node, n, weight_attr) / out_sum for n in out_nbrs}
        elif out_nbrs:
            out_probs = {n: 1.0 / len(out_nbrs) for n in out_nbrs}
        else:
            out_probs = {}

        # Clamp beta when one side of the mix is empty. P_out and P_in are each
        # probability measures, so P = beta*P_out + (1-beta)*P_in has total mass 1
        # only when BOTH exist. At a sink (no out-edges) the beta*P_out term is
        # absent and the raw mix would sum to (1 - beta); at a source (no in-edges)
        # it would sum to beta. A balancing factor has nothing to balance in those
        # cases, so the walk collapses to the side that exists. This matches
        # BetaDegreeProportional, where beta = deg_out/deg is already 0 at a sink
        # and 1 at a source. Nodes with both directions are unaffected.
        if not out_nbrs:
            beta_val = 0.0
        elif not in_nbrs:
            beta_val = 1.0

        result = []
        for nbr in neighbors[node]:
            p = 0.0
            if nbr in out_probs:
                p += beta_val * out_probs[nbr]
            if nbr in in_probs:
                p += (1 - beta_val) * in_probs[nbr]
            result.append(p)
        return np.array(result)


@dataclass
class UndirectedKernel:
    """Standard random walk: P(x->y) = w(x,y) / sum_z w(x,z)."""

    #: Graphs this kernel is defined on: see :func:`check_kernel_domain`.
    domain = "undirected"

    def __call__(self, node, G, neighbors, weight_attr) -> np.ndarray:
        nbrs = neighbors[node]
        if not nbrs:
            return np.array([1.0])
        total = sum(_gamma(G, node, n, weight_attr) for n in nbrs)
        if total > 0:
            result = [_gamma(G, node, n, weight_attr) / total for n in nbrs]
        else:
            result = [1.0 / len(nbrs)] * len(nbrs)
        return np.array(result)


@dataclass
class OutKernel:
    """Out-walk kernel (Yamada 2016): outgoing edges only.

    The result is aligned with ``neighbors[node]``. A node with neighbours but
    no out-edge (a sink) gets an all-zero vector -- an empty selected
    neighbourhood, whose endpoint measure falls back to the Dirac mass on the
    node. (The earlier ``[1.0]`` for that case was indistinguishable from "all
    mass on the only neighbour".) An isolated node still gets ``[1.0]``.
    """

    #: Graphs this kernel is defined on: see :func:`check_kernel_domain`.
    domain = "directed"

    def __call__(self, node, G, neighbors, weight_attr) -> np.ndarray:
        successors = list(G.successors(node))
        out_nbrs = set(successors)
        if not out_nbrs:
            return np.zeros(len(neighbors[node])) if neighbors[node] else np.array([1.0])
        # summed in adjacency order: a set of string labels iterates in
        # per-process hash order, which changed the last bits between runs
        out_sum = sum(_gamma(G, node, n, weight_attr) for n in successors)
        result = []
        for nbr in neighbors[node]:
            if nbr in out_nbrs and out_sum > 0:
                result.append(_gamma(G, node, nbr, weight_attr) / out_sum)
            else:
                result.append(0.0)
        return np.array(result)


@dataclass
class InKernel:
    """In-walk kernel: incoming edges only.

    Aligned with ``neighbors[node]``; a source (neighbours but no in-edge) gets
    an all-zero vector, an isolated node ``[1.0]``. See :class:`OutKernel`.
    """

    #: Graphs this kernel is defined on: see :func:`check_kernel_domain`.
    domain = "directed"

    def __call__(self, node, G, neighbors, weight_attr) -> np.ndarray:
        predecessors = list(G.predecessors(node))
        in_nbrs = set(predecessors)
        if not in_nbrs:
            return np.zeros(len(neighbors[node])) if neighbors[node] else np.array([1.0])
        in_sum = sum(_gamma(G, n, node, weight_attr) for n in predecessors)
        result = []
        for nbr in neighbors[node]:
            if nbr in in_nbrs and in_sum > 0:
                result.append(_gamma(G, nbr, node, weight_attr) / in_sum)
            else:
                result.append(0.0)
        return np.array(result)


@dataclass
class AutoKernel:
    """UndirectedKernel for undirected graphs, MixedKernel(beta) for directed."""

    #: Graphs this kernel is defined on: see :func:`check_kernel_domain`.
    domain = "any"

    beta: BetaSpec = 0.8

    def __post_init__(self):
        self._directed = MixedKernel(self.beta)
        self._undirected = UndirectedKernel()

    def __call__(self, node, G, neighbors, weight_attr) -> np.ndarray:
        if G.is_directed():
            return self._directed(node, G, neighbors, weight_attr)
        return self._undirected(node, G, neighbors, weight_attr)


def kernel_domain(kernel: Any) -> str:
    """The graph class a kernel is defined on: ``directed``/``undirected``/``any``.

    A kernel with no declaration is treated as ``"any"``, so a user-supplied
    callable is never rejected on the strength of metadata it did not provide.
    """
    return getattr(kernel, "domain", "any")


def check_kernel_domain(kernel: Any, G: Any, *, where: str = "curvature") -> None:
    """Reject a kernel/graph mismatch with a domain error, not a leaked internal.

    Before this check, the mismatch surfaced from deep inside the walk: an out,
    in or mixed kernel on an undirected graph raised ``AttributeError: 'Graph'
    object has no attribute 'successors'``, and the undirected kernel on a
    digraph raised a bare ``KeyError`` naming a node. Both read as crashes rather
    than as "this kernel is not defined on this graph", and neither says what to
    do about it.

    The domain of an optimal-transport curvature is a property of its
    *configuration*, not of its name: ``ollivier`` accepts either graph class
    with ``kernel="auto"`` and only directed graphs with ``kernel="out"``.
    """
    domain = kernel_domain(kernel)
    if domain == "any":
        return
    directed = G.is_directed()
    if domain == "directed" and not directed:
        raise CurvatureDomainError(
            f"{where}: the {type(kernel).__name__} is defined on directed graphs "
            f"only, but the input is undirected. Use kernel='auto' (or "
            f"'undirected'), or pass a directed graph."
        )
    if domain == "undirected" and directed:
        raise CurvatureDomainError(
            f"{where}: the {type(kernel).__name__} is defined on undirected "
            f"graphs only, but the input is directed. Use kernel='auto' to pick "
            f"the mixed kernel, or an explicit 'out'/'in'/'mixed' kernel, or "
            f"convert with G.to_undirected() so the choice is recorded."
        )


def resolve_kernel(kernel: str = "auto", beta: BetaSpec = 0.8):
    """Resolve a kernel name to a kernel instance.

    ``"auto"`` | ``"undirected"`` | ``"out"`` | ``"in"`` | ``"mixed"``.
    ``beta`` applies to ``"mixed"`` and ``"auto"`` (directed).
    """
    if kernel == "auto":
        return AutoKernel(beta)
    if kernel == "undirected":
        return UndirectedKernel()
    if kernel == "out":
        return OutKernel()
    if kernel == "in":
        return InKernel()
    if kernel == "mixed":
        return MixedKernel(beta)
    raise ValueError(
        f"Unknown kernel {kernel!r}. Choose from: "
        "'auto', 'undirected', 'out', 'in', 'mixed'."
    )
