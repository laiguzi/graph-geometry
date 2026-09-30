"""Node-weighted Forman-Ricci curvature on undirected graphs.

Registered as the *core* definitions ``forman_node_weighted`` and
``augmented_forman_node_weighted``; the former names ``forman_sreejith`` and
``augmented_forman_sreejith`` remain deprecated aliases with identical numbers.
The functions here return raw ``{edge: value}`` mappings and do not annotate
the graph.

For an undirected edge ``e = (u, v)`` with edge weight ``w_e`` and node weights
``m_u``, ``m_v``, both are configurations of the combinatorial template
(:mod:`~graph_geometry.curvature.combinatorial`):

    node-weighted(e) = m_u + m_v - sum_{z in {u,v}} sum_{e' in delta(z)\\e} m_z sqrt(w_e / w_e')

    augmented(e)     = m_u + m_v - sum_{z} sum_{e' in delta(z)\\(e u R_z(e))} m_z sqrt(w_e / w_e')
                       + sum_{triangles f through e} w_e^2 / w_f

where ``R_z(e)`` are the triangle-closing edges at ``z`` (Forman 2003; Sreejith
et al. 2016; Samal et al. 2018). These equal GraphRicciCurvature's
``w_e * (m_u/w_e + m_v/w_e - sum m_z / sqrt(w_e w_e'))`` and
``w_e * (sum_f w_e/w_f + ...)`` term by term (to floating-point rounding).

With ``node_weight="one"`` and ``face_weight=1`` these reproduce
GraphRicciCurvature's ``FormanRicci(method="1d")`` and
``FormanRicci(method="augmented")`` respectively; the augmented form under unit
weights is ``4 - deg(u) - deg(v) + 3*|triangles|``, which is also what
curvature-filtrations (SCOTT) calls ``forman_curvature`` on an unweighted graph.
With ``node_weight="inverse_degree"`` the plain form reproduces pynetflow's
``Forman_ricci``. See ``paper/compare_with_libraries.py`` for the executed
cross-checks. The numbers quoted against GraphRicciCurvature in the paper's
related-work section are pinned in ``tests/test_forman_variants.py``.

The endpoint term ``A_c``, the charge ``B_c`` and the face term ``C_c`` defined
here are shared with the directed definitions in
:mod:`~graph_geometry.curvature.forman_directed`, which substitute only the
incidence selector ``I_c`` and the face selector ``F_c``.

Both variants here are defined for undirected graphs only. A directed graph is
rejected rather than silently symmetrised (GraphRicciCurvature converts and logs
an informational message); use ``forman_directed``,
``augmented_forman_directed`` or ``eidi_jost`` for directed input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Union

import networkx as nx

from ..graph.semantics import GraphSemantics
from .combinatorial import (
    CombinatorialContext,
    CombinatorialCurvatureConfig,
    CombinatorialCurvatureEngine,
)
from .errors import (
    CurvatureConfigurationError,
    CurvatureDomainError,
    CurvatureInputError,
)
from .validate import validate_edge_attributes

__all__ = [
    "forman_node_weighted_config",
    "augmented_forman_node_weighted_config",
    "forman_node_weighted",
    "augmented_forman_node_weighted",
    "forman_sreejith",
    "augmented_forman_sreejith",
    "resolve_node_weight",
    "NODE_WEIGHT_SCHEMES",
    "NodeWeights",
    "endpoint_term",
    "charge_term",
    "TriangleFaceTerm",
    "edge_weight",
]

NodeWeightSpec = Union[str, float, int, Mapping, Callable[[Any, nx.Graph], float]]

#: Named node-weight schemes. Forman's node weights are part of the *definition*
#: of the curvature, not a property of the graph, so they are named here rather
#: than being read from an attribute by default.
_NAMED_SCHEMES = {
    # Forman's default in the network literature: every node carries weight 1.
    "one": lambda n, G, weight: 1.0,
    # deg(n) -- used when the node weight is meant to track connectivity.
    "degree": lambda n, G, weight: float(G.degree(n)),
    # 1/deg(n) -- pynetflow's convention.
    "inverse_degree": lambda n, G, weight: 1.0 / max(G.degree(n), 1),
    # sum of incident edge weights -- node weight tracking edge strength.
    "incident_weight": lambda n, G, weight: sum(
        G[n][m].get(weight, 1.0) for m in G.neighbors(n)
    ),
}


def resolve_node_weight(
    spec: NodeWeightSpec, G: nx.Graph, weight: str, *, node_weight_attr: str | None = None
) -> dict:
    """Return ``{node: w_node}`` for a node-weight ``spec``.

    ``spec`` may be a named scheme (``"one" | "degree" | "inverse_degree" |
    "incident_weight"``), a number applied to every node, a mapping, or a
    callable ``(node, G) -> float``. The declarative schema accepts the named
    schemes only; the other forms remain for direct Python use.

    ``node_weight_attr`` overrides all of it by reading a node attribute. Every
    node must then carry a value: a missing one is an error rather than a
    silent fallback to ``spec``, because a half-attributed graph would mix two
    node-weight conventions in one result.

    Node weights supplied by the caller (attribute, number, mapping, callable)
    must be finite and strictly positive on every node -- the same rule the
    registry applies to ``node_weight_attr``. A zero or negative value used to
    be accepted and silently produced, e.g., all-zero curvature.
    """
    if node_weight_attr is not None:
        missing = [n for n in G.nodes() if node_weight_attr not in G.nodes[n]]
        if missing:
            raise CurvatureInputError(
                f"node_weight_attr={node_weight_attr!r} is missing on "
                f"{len(missing)} node(s), e.g. {missing[:3]}."
            )
        return _positive_node_weights(
            {n: G.nodes[n][node_weight_attr] for n in G.nodes()},
            f"node_weight_attr={node_weight_attr!r}", CurvatureInputError,
        )
    if isinstance(spec, str):
        try:
            fn = _NAMED_SCHEMES[spec]
        except KeyError:
            raise CurvatureConfigurationError(
                f"Unknown node_weight scheme {spec!r}. "
                f"Known: {sorted(_NAMED_SCHEMES)}; or pass a number, mapping or callable."
            ) from None
        return {n: float(fn(n, G, weight)) for n in G.nodes()}
    if isinstance(spec, Mapping):
        missing = [n for n in G.nodes() if n not in spec]
        if missing:
            raise CurvatureConfigurationError(
                f"node_weight mapping is missing {len(missing)} node(s), "
                f"e.g. {missing[:3]}."
            )
        return _positive_node_weights(
            {n: spec[n] for n in G.nodes()}, "node_weight mapping",
            CurvatureConfigurationError,
        )
    if callable(spec):
        return _positive_node_weights(
            {n: spec(n, G) for n in G.nodes()}, "node_weight callable",
            CurvatureConfigurationError,
        )
    return _positive_node_weights(
        {n: spec for n in G.nodes()}, "node_weight", CurvatureConfigurationError
    )


def _positive_node_weights(raw: dict, source: str, error: type) -> dict:
    """``{node: float}``, rejecting non-numeric, non-finite and non-positive values."""
    out = {}
    for n, value in raw.items():
        if isinstance(value, bool):
            raise error(f"{source} gives node {n!r} the non-numeric value {value!r}.")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise error(f"{source} gives node {n!r} the non-numeric value {value!r}.") from None
        if not math.isfinite(number) or number <= 0.0:
            raise error(
                f"{source} gives node {n!r} the value {value!r}; Forman node "
                f"weights must be finite and strictly positive."
            )
        out[n] = number
    return out


#: Named node-weight schemes accepted by the declarative schema.
NODE_WEIGHT_SCHEMES = tuple(_NAMED_SCHEMES)


def _require_undirected(G: nx.Graph, where: str) -> None:
    if G.is_directed():
        raise CurvatureDomainError(
            f"{where} is defined for undirected graphs only; got a directed graph. "
            f"Use method='forman_directed' or 'augmented_forman_directed' "
            f"(Saucan et al. 2019), or method='eidi_jost', "
            f"or symmetrise explicitly with G.to_undirected() so the convention is "
            f"recorded in the experiment rather than applied silently."
        )


def edge_weight(G: nx.Graph, u: Any, v: Any, weight: str) -> float:
    """``w_e`` of a stored edge.

    Weights are validated finite and strictly positive before any term is
    evaluated, so no floor is applied; the former ``1e-10`` floor changed the
    value of every edge lighter than it.
    """
    return float(G[u][v].get(weight, 1.0))


# ── template components ─────────────────────────────────────────────────────


def _validate(G: nx.Graph, semantics: GraphSemantics, where: str) -> None:
    _require_undirected(G, where)
    validate_edge_attributes(G, weight=semantics.weight_attr, where=where)


@dataclass(frozen=True)
class NodeWeights:
    """``prepare``: resolve Forman's node weights ``m`` once per graph."""

    node_weight: Any = "one"

    def __call__(self, ctx: CombinatorialContext) -> None:
        ctx.data["m"] = resolve_node_weight(
            self.node_weight, ctx.G, ctx.weight,
            node_weight_attr=ctx.semantics.node_weight_attr,
        )


def endpoint_term(ctx: CombinatorialContext, edge: tuple) -> float:
    u, v = edge
    return ctx.data["m"][u] + ctx.data["m"][v]


def _incident_edges(ctx: CombinatorialContext, z: Any, edge: tuple):
    """delta(z) without e."""
    u, v = edge
    other = v if z == u else u
    return ((z, m) for m in ctx.G.neighbors(z) if m != other)


def _non_triangle_edges(ctx: CombinatorialContext, z: Any, edge: tuple):
    """delta(z) without e and without the edges that close a triangle with e."""
    u, v = edge
    other = v if z == u else u
    face = set(_triangles(ctx, edge))
    return ((z, m) for m in ctx.G.neighbors(z) if m != other and m not in face)


def charge_term(ctx: CombinatorialContext, edge: tuple, incident: tuple, z: Any) -> float:
    """m_z * sqrt(w_e / w_e')."""
    G, weight = ctx.G, ctx.weight
    return ctx.data["m"][z] * math.sqrt(edge_weight(G, *edge, weight) / edge_weight(G, *incident, weight))


def _triangles(ctx: CombinatorialContext, edge: tuple) -> list:
    """Third vertices of the triangles through e, in u's adjacency order.

    A list, not a set, so float sums do not follow per-process hash order.
    """
    u, v = edge
    at_v = set(ctx.G.neighbors(v))
    return [m for m in ctx.G.neighbors(u) if m != v and m != u and m in at_v]


@dataclass(frozen=True)
class TriangleFaceTerm:
    """C_c(e, f) = w_e * w_e / w_f for a triangular face of weight ``face_weight``."""

    face_weight: float = 1.0

    def __call__(self, ctx: CombinatorialContext, edge: tuple, face: Any) -> float:
        w_e = edge_weight(ctx.G, *edge, ctx.weight)
        return w_e * (w_e / self.face_weight)


def forman_node_weighted_config(node_weight: NodeWeightSpec = "one") -> CombinatorialCurvatureConfig:
    """Template configuration of node-weighted Forman curvature.

    ``A_c = m_x + m_y``, ``S_c(z; e) = delta(z) \\ {e}``,
    ``B_c = m_z sqrt(w_e / w_e')``, no faces. The node weights come from
    ``GraphSemantics.node_weight_attr`` when it is set, otherwise from
    ``node_weight``.
    """
    return CombinatorialCurvatureConfig(
        name="forman_node_weighted",
        endpoint=endpoint_term, incidence=_incident_edges, charge=charge_term,
        prepare=NodeWeights(node_weight), validate=_validate,
    )


def augmented_forman_node_weighted_config(
    node_weight: NodeWeightSpec = "one", face_weight: float = 1.0
) -> CombinatorialCurvatureConfig:
    """Template configuration of node-weighted Forman curvature with triangles.

    As :func:`forman_node_weighted_config`, except that triangle-closing edges
    leave ``S_c`` and every triangle ``f`` through ``e`` contributes
    ``C_c(e, f) = w_e^2 / w_f`` with ``w_f = face_weight``.
    """
    if face_weight <= 0:
        raise CurvatureConfigurationError(f"face_weight must be positive, got {face_weight}")
    return CombinatorialCurvatureConfig(
        name="augmented_forman_node_weighted",
        endpoint=endpoint_term, incidence=_non_triangle_edges, charge=charge_term,
        faces=_triangles, face_term=TriangleFaceTerm(face_weight),
        prepare=NodeWeights(node_weight), validate=_validate,
    )


# ── raw functions ───────────────────────────────────────────────────────────


def forman_node_weighted(
    G: nx.Graph,
    *,
    weight: str = "weight",
    node_weight: NodeWeightSpec = "one",
    node_weight_attr: str | None = None,
) -> dict:
    """Node-weighted Forman-Ricci curvature, no triangle faces (raw values).

    ``node_weight`` selects Forman's node weights (default ``"one"``). See the
    module docstring for the formula.
    """
    semantics = GraphSemantics(weight_attr=weight, node_weight_attr=node_weight_attr)
    return CombinatorialCurvatureEngine(
        G, forman_node_weighted_config(node_weight), semantics
    ).compute_edges()


def augmented_forman_node_weighted(
    G: nx.Graph,
    *,
    weight: str = "weight",
    node_weight: NodeWeightSpec = "one",
    node_weight_attr: str | None = None,
    face_weight: float = 1.0,
) -> dict:
    """Node-weighted Forman-Ricci curvature with triangular faces.

    Faces are the triangles through the edge, each of weight ``face_weight``.
    Neighbours that close a triangle drop out of the star sums, which is what
    distinguishes this from :func:`forman_node_weighted`. Under unit weights
    the result is ``4 - deg(u) - deg(v) + 3*|triangles|``.
    """
    semantics = GraphSemantics(weight_attr=weight, node_weight_attr=node_weight_attr)
    return CombinatorialCurvatureEngine(
        G, augmented_forman_node_weighted_config(node_weight, face_weight), semantics
    ).compute_edges()


#: Former names of the same raw functions.
forman_sreejith = forman_node_weighted
augmented_forman_sreejith = augmented_forman_node_weighted
