"""Eidi-Jost in-out Ollivier curvature for directed graphs (no connectivity assumption).

Eidi, M. & Jost, J. (2020). *Ollivier Ricci curvature of directed hypergraphs.*
Scientific Reports 10, 12466 (arXiv:1907.04727). Implemented here for the graph
case (``n = m = 1`` of their Definition 3.2; weighted per Definition 7.1 with
unit vertex weights).

Why this exists alongside :func:`~graph_geometry.curvature.lin_lu_yau`
----------------------------------------------------------------------
The OT engine in ``base.py`` supports the measure on the **undirected**
neighbourhood (``predecessors | successors``) while the cost matrix uses
**directed** shortest paths. A pure sink inside that support gives
``d(sink, ·) = inf``, so the transport has no finite plan and curvature is
genuinely undefined -- not a numerical failure. Yamada's out-out convention
(arXiv:1602.07779) has the same problem and is usually stated with a strong
connectivity assumption, which, quoting Eidi-Jost, "does not hold in many real
directed networks".

Their fix is to change the **directions the measures are read from**, not the
metric. For an edge ``e: x -> y``:

    mu_x_in   lives on the IN-neighbours of x   (on x itself when x is a source)
    mu_y_out  lives on the OUT-neighbours of y  (on y itself when y is a sink)

    kappa(e) = 1 - W_1(mu_x_in, mu_y_out) / d(x, y)

Finiteness is then structural: for any ``u`` in ``supp(mu_x_in)`` and any ``v``
in ``supp(mu_y_out)`` the directed path ``u -> x -> y -> v`` exists, hence

    d(u, v) <= 3   always, on any digraph.

A sink ``y`` is absorbed by the definition -- the hole sits on ``y`` itself, at
distance 2 from any ``u``. No strong connectivity, no infinities, no fallbacks.
Of the four possible conventions only in-out has this guarantee; in-in, out-in
and out-out can all strand mass.

Trade-off
---------
There is **no balancing factor** here, and one cannot be added: mixing any
out-out component back in reintroduces the unreachable pairs that the in-out
choice exists to avoid. Use this for whole-graph geometry on a digraph that is
not strongly connected, and ``lin_lu_yau`` with ``kernel="mixed"`` on a strongly
connected subgraph when the balancing factor is the object of study.

The metric
----------
``distance`` defaults to 1 on every edge, which is the hop metric the source
definition uses; there ``d(x, y) = 1`` and the formula above reduces to the
paper's ``kappa = 1 - W_1``. A non-unit ``distance`` may be supplied, and the
division by ``d(x, y)`` makes the result scale-invariant: multiplying every
distance by a positive constant leaves ``kappa`` unchanged.

Two properties below hold **only for a uniform metric** (every edge the same
length, the default included), because ``d(u, v) <= 3`` counts hops:

- the bound ``kappa`` in ``[-2, 1]``;
- the combinatorial form ``kappa = mu_0 - mu_2 - 2*mu_3``, where ``mu_i`` is the
  mass moved at distance ``i`` and :func:`directed_curvature_profile` returns
  that decomposition. ``mu_0`` is mass held in place by directed 3-cycles, so
  the profile says *why* an edge is curved the way it is.

Under a non-uniform metric both generalise: the reachability guarantee still
gives a finite ``W_1``, but the floor becomes
``1 - 3 * max_route_length / d(x, y)``, which is below ``-2`` when routes are
long relative to the edge. Interpret the profile buckets as distances, not hops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from ..graph.semantics import GraphSemantics
from .transport import (
    DirectionalKernel,
    EMDTransportSolver,
    NoIdlenessDistribution,
    OTCurvatureConfig,
    directional_neighbourhood_mass,
)
from .validate import validate_signs

__all__ = [
    "directed_curvature_profile",
    "directed_in_out_measure",
    "eidi_jost",
    "eidi_jost_ot_config",
    "signed_double_cover",
]


def signed_double_cover(
    G: nx.DiGraph,
    *,
    sign: str = "sign",
    weight: str = "weight",
    distance: str = "distance",
    positive: str = "promotion",
) -> nx.DiGraph:
    """Zaslavsky's signed double cover — turn a signed digraph into an unsigned one.

    OT curvature cannot consume negative weights (see
    :func:`directed_in_out_measure`). The double cover sidesteps that by moving
    the sign **into the topology**: every node ``v`` becomes ``(v, +)`` and
    ``(v, -)``, and an edge ``u -> v`` lifts to

    - positive edge: ``(u,+) -> (v,+)`` and ``(u,-) -> (v,-)``   (sign preserved)
    - negative edge: ``(u,+) -> (v,-)`` and ``(u,-) -> (v,+)``   (sheets swap)

    Weights become ``|w|`` and are all non-negative, so any curvature method
    applies. Sign information is not discarded — it is encoded in *which sheet*
    a path lands on, and curvature on the cover reflects the composition of
    signs along paths, i.e. structural balance.

    The classical fact that makes this meaningful: a signed graph is **balanced**
    iff its double cover splits into two disconnected copies of the base graph
    (Zaslavsky's switching lemma), and unbalanced iff the cover stays connected.
    So curvature on the cover is sensitive to balance in a way that computing on
    ``|w|`` alone is not.

    Costs: 2× the nodes and edges, and every edge appears twice (the two lifts
    of one base edge are distinct edges of the cover, generally with different
    curvature). Interpreting a base edge means looking at both lifts.

    ``sign`` is the edge attribute holding the sign; an edge counts as positive
    when it equals ``positive``, and negative otherwise. Nodes are relabelled to
    ``(node, +1)`` / ``(node, -1)`` tuples.
    """
    if not G.is_directed():
        raise ValueError("signed_double_cover expects a directed graph")
    validate_signs(G, sign=sign, positive=positive, where="signed_double_cover")

    H = nx.DiGraph()
    for n, data in G.nodes(data=True):
        for s in (1, -1):
            H.add_node((n, s), **data)

    for u, v, data in G.edges(data=True):
        w = abs(float(data.get(weight, 1.0)))
        d = float(data.get(distance, 1.0))
        is_positive = data.get(sign, positive) == positive
        for s in (1, -1):
            t = s if is_positive else -s
            H.add_edge((u, s), (v, t), **{weight: w, distance: d,
                                          sign: data.get(sign, positive),
                                          "base_edge": f"{u}->{v}"})
    return H


def directed_in_out_measure(
    G: nx.DiGraph, node: Any, direction: str, weight: str = "weight"
) -> dict:
    """``mu_x_in`` (``direction="in"``) or ``mu_y_out`` (``direction="out"``).

    Mass is split across the incident edges in that direction in proportion to
    their weight. A node with no edge in the required direction is a source
    (for ``"in"``) or a sink (for ``"out"``) and keeps all the mass itself --
    the clause that makes sinks harmless.

    Raises on **negative weights**: optimal transport is defined between
    *probability* measures, and a negative weight yields negative mass. Such a
    "measure" can still sum to 1 (e.g. weights 3 and -1 give ``{a: 1.5,
    b: -0.5}``), so it passes a mass check and reaches the solver as a signed
    measure, producing numbers that mean nothing. See the module docstring for
    what to do with a signed graph instead.
    """
    mass = directional_neighbourhood_mass(G, node, direction, weight)
    return {node: 1.0} if mass is None else mass


@dataclass(frozen=True)
class _InOutCurvature:
    """``kappa = 1 - W_1 / d(x, y)`` (picklable)."""

    def __call__(self, cost: float, distance: float) -> float:
        return 1.0 - float(cost) / distance


def eidi_jost_ot_config() -> OTCurvatureConfig:
    """Endpoint-pair configuration of Eidi-Jost curvature on the shared engine.

    Eidi-Jost is an ordinary configuration of the transport family: the
    in-neighbourhood kernel at the source, the out-neighbourhood kernel at the
    target, no idleness (an empty neighbourhood gives the Dirac measure), the
    ordered shortest-path ground cost, exact EMD with the plan retained (it backs
    :func:`directed_curvature_profile`), and ``1 - W_1 / d(x, y)``.
    """
    return OTCurvatureConfig(
        name="Eidi-Jost",
        kernel=DirectionalKernel("in"),
        target_kernel=DirectionalKernel("out"),
        distribution=NoIdlenessDistribution(),
        solver=EMDTransportSolver(),
        curvature=_InOutCurvature(),
        return_plan=True,
    )


def _engine(G, weight: str, distance: str, proc: int):
    from .base import TransportCurvatureEngine  # the engine imports this package's kernels

    return TransportCurvatureEngine(
        G, eidi_jost_ot_config(),
        GraphSemantics(weight_attr=weight, distance_attr=distance), proc,
    )


def directed_curvature_profile(
    G: nx.DiGraph, *, weight: str = "weight", distance: str = "distance",
    pairs=None, proc: int = 1,
) -> tuple[dict, dict]:
    """Return ``(edge_curvature, mass_profile)``.

    ``mass_profile[(u, v)]`` maps transport distance -> mass moved at that
    distance (the paper's ``mu_i``). Does not write graph attributes. This is a
    diagnostic view over the shared engine's transport plans, not a second
    implementation.

    ``pairs`` is a low-level diagnostic hook. The published construction, and
    the finiteness guarantee ``u -> x -> y -> v``, hold only when ``(x, y)`` is
    an edge; the registry therefore exposes Eidi-Jost on edges only. A
    non-edge pair whose transport is undefined raises
    :class:`~graph_geometry.curvature.errors.CurvatureDomainError`.
    """
    details = _engine(G, weight, distance, proc).transport_details(pairs)
    kappa: dict = {}
    profile: dict = {}
    for pair, (value, (_mu, _nu, cost, result)) in details.items():
        kappa[pair] = value
        buckets: dict = {}
        plan = result.plan
        rows, cols = (plan > 1e-12).nonzero()
        for i, j in zip(rows, cols):
            key = round(float(cost[i][j]), 6)
            buckets[key] = buckets.get(key, 0.0) + float(plan[i][j])
        profile[pair] = buckets
    return kappa, profile


def eidi_jost(
    G: nx.Graph,
    *,
    weight: str = "weight",
    distance: str = "distance",
    proc: int = 1,
) -> dict:
    """Eidi-Jost in-out directed Ollivier curvature on every edge (raw values).

    Works on any directed graph, including one that is not strongly connected.
    Returns ``{(u, v): kappa}`` and does not annotate ``G``; use
    ``compute_curvature(G, "eidi_jost")`` for the result contract.
    """
    return _engine(G, weight, distance, proc).compute_edges()
