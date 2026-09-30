"""Directed Forman-Ricci curvature (Sreejith et al. 2016; Saucan et al. 2019).

Registered as the *core* definitions ``forman_directed`` and
``augmented_forman_directed``. Both are configurations of the combinatorial
template (:mod:`~graph_geometry.curvature.combinatorial`) that reuse the
endpoint term, the charge and the face term of the undirected node-weighted
form (:mod:`~graph_geometry.curvature.forman_variants`) and substitute only the
incidence selector ``I_c`` and the face selector ``F_c``.

For a directed edge ``e = (x, y)`` with edge weights ``w``, node weights ``m``
and ``delta^-(z)`` / ``delta^+(z)`` the in- and out-edges of ``z``::

    kappa(e)  = m_x + m_y - sum_{e' in delta^-(x)} m_x sqrt(w_e / w_e')
                          - sum_{e' in delta^+(y)} m_y sqrt(w_e / w_e')

    kappa#(e) = kappa(e) restricted to edges outside the faces of e
                + sum_{t in F(e)} w_e^2 / w_t

Only the edges *entering the tail* and *leaving the head* are charged, so that
each one continues a path in the direction of ``e`` (Saucan et al. 2019, Eq. 5;
first stated in Sreejith et al. 2016; the same convention appears in Weber et
al. 2017). Three consequences follow from that choice:

- ``e`` itself is excluded automatically: it leaves ``x`` and enters ``y``, so
  it is in neither ``delta^-(x)`` nor ``delta^+(y)``.
- a reciprocal edge ``(y, x)`` *is* charged, once at each endpoint, since it
  enters ``x`` and leaves ``y``. The published definition excludes self-loops
  only; the 2016 preprint additionally excluded the reciprocal edge.
- with unit weights the value is ``2 - in(x) - out(y)``, which is Leal et al.
  (2021)'s ``F(-> e ->)``, the flow through ``e``.

``F(e)`` in the augmented form collects the **feed-forward triangles** through
``e`` -- ``a -> b``, ``b -> c``, ``a -> c`` -- which is the directed face chosen
by Saucan et al. (2019, Fig. 1(b)); a directed 3-cycle is not a face. ``e`` may
sit in any of the three positions of such a triangle, so the sum is over every
feed-forward triangle containing ``e`` (their Eq. 12, ``sum_{e < t}``). The
charged edges that lie in one of those faces leave the sums, exactly as
triangle-closing edges do in the undirected augmented form. Only ``e`` as a
first or second leg removes a charged edge: when ``e`` is the shortcut, the
other two edges of the face leave ``x`` and enter ``y``, and neither is charged.
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from ..graph.semantics import GraphSemantics
from .combinatorial import (
    CombinatorialContext,
    CombinatorialCurvatureConfig,
    CombinatorialCurvatureEngine,
)
from .errors import CurvatureConfigurationError, CurvatureDomainError
from .forman_variants import (
    NodeWeights,
    NodeWeightSpec,
    TriangleFaceTerm,
    charge_term,
    endpoint_term,
)
from .validate import validate_edge_attributes

__all__ = [
    "forman_directed_config",
    "augmented_forman_directed_config",
    "forman_directed",
    "augmented_forman_directed",
    "feed_forward_faces",
]


def _require_directed(G: nx.Graph, where: str) -> None:
    if not G.is_directed():
        raise CurvatureDomainError(
            f"{where} is defined for directed graphs only; got an undirected graph. "
            f"Its incidence selector reads the in-edges of the tail and the "
            f"out-edges of the head, which an undirected graph does not "
            f"distinguish. Use method='forman_node_weighted' or "
            f"'augmented_forman_node_weighted' for undirected input."
        )


def _validate(G: nx.Graph, semantics: GraphSemantics, where: str) -> None:
    _require_directed(G, where)
    validate_edge_attributes(G, weight=semantics.weight_attr, where=where)


# ── template components ─────────────────────────────────────────────────────


def _in_out_star(ctx: CombinatorialContext, z: Any, edge: tuple):
    """``delta^-(x)`` at the tail and ``delta^+(y)`` at the head, without loops.

    ``e`` needs no exclusion here: it leaves the tail and enters the head.
    """
    G = ctx.G
    x, y = edge
    if z == x:
        return [(s, x) for s in G.predecessors(x) if s != x]
    return [(y, t) for t in G.successors(y) if t != y]


def feed_forward_faces(G: nx.Graph, edge: tuple) -> list:
    """The feed-forward triangles containing ``edge``, as their two other arcs.

    A face is returned as ``((a, b), (c, d))`` in adjacency order, so the sums
    over faces do not follow Python's per-process hash order. ``e`` occurs in
    each of the three positions of a feed-forward triangle ``x -> q -> y`` plus
    shortcut, hence the three passes.
    """
    x, y = edge
    if x == y:
        return []
    faces = []
    for q in G.successors(x):  # e is the shortcut: x -> q -> y
        if q not in (x, y) and G.has_edge(q, y):
            faces.append(((x, q), (q, y)))
    for q in G.successors(y):  # e is the first leg: x -> y -> q, shortcut x -> q
        if q not in (x, y) and G.has_edge(x, q):
            faces.append(((y, q), (x, q)))
    for q in G.predecessors(x):  # e is the second leg: q -> x -> y, shortcut q -> y
        if q not in (x, y) and G.has_edge(q, y):
            faces.append(((q, x), (q, y)))
    return faces


def _faces(ctx: CombinatorialContext, edge: tuple) -> list:
    """``F_c(e)``, cached per edge: the engine asks for it once per star as well."""
    cache = ctx.data.setdefault("faces", {})
    if edge not in cache:
        cache[edge] = feed_forward_faces(ctx.G, edge)
    return cache[edge]


def _star_outside_faces(ctx: CombinatorialContext, z: Any, edge: tuple):
    """``I_c`` of the augmented form: the star minus the edges lying in a face."""
    in_a_face = {arc for face in _faces(ctx, edge) for arc in face}
    return [arc for arc in _in_out_star(ctx, z, edge) if arc not in in_a_face]


def forman_directed_config(
    node_weight: NodeWeightSpec = "one",
) -> CombinatorialCurvatureConfig:
    """Template configuration of directed Forman curvature.

    ``A_c = m_x + m_y``, ``I_c(x; e) = delta^-(x)``, ``I_c(y; e) = delta^+(y)``,
    ``B_c = m_z sqrt(w_e / w_e')``, no faces. The node weights come from
    ``GraphSemantics.node_weight_attr`` when it is set, otherwise from
    ``node_weight``.
    """
    return CombinatorialCurvatureConfig(
        name="forman_directed",
        endpoint=endpoint_term, incidence=_in_out_star, charge=charge_term,
        prepare=NodeWeights(node_weight), validate=_validate,
    )


def augmented_forman_directed_config(
    node_weight: NodeWeightSpec = "one", face_weight: float = 1.0
) -> CombinatorialCurvatureConfig:
    """Template configuration of directed augmented Forman curvature.

    As :func:`forman_directed_config`, except that the feed-forward triangles
    through ``e`` are faces of weight ``face_weight``: each contributes
    ``C_c(e, t) = w_e^2 / w_t``, and a charged edge lying in one of them leaves
    ``I_c``.
    """
    if face_weight <= 0:
        raise CurvatureConfigurationError(
            f"face_weight must be positive, got {face_weight}"
        )
    return CombinatorialCurvatureConfig(
        name="augmented_forman_directed",
        endpoint=endpoint_term, incidence=_star_outside_faces, charge=charge_term,
        faces=_faces, face_term=TriangleFaceTerm(face_weight),
        prepare=NodeWeights(node_weight), validate=_validate,
    )


# ── raw functions ───────────────────────────────────────────────────────────


def forman_directed(
    G: nx.DiGraph,
    *,
    weight: str = "weight",
    node_weight: NodeWeightSpec = "one",
    node_weight_attr: str | None = None,
) -> dict:
    """Directed Forman-Ricci curvature (raw values).

    ``2 - in(x) - out(y)`` under unit weights. See the module docstring.
    """
    semantics = GraphSemantics(weight_attr=weight, node_weight_attr=node_weight_attr)
    return CombinatorialCurvatureEngine(
        G, forman_directed_config(node_weight), semantics
    ).compute_edges()


def augmented_forman_directed(
    G: nx.DiGraph,
    *,
    weight: str = "weight",
    node_weight: NodeWeightSpec = "one",
    node_weight_attr: str | None = None,
    face_weight: float = 1.0,
) -> dict:
    """Directed augmented Forman-Ricci curvature, with feed-forward faces."""
    semantics = GraphSemantics(weight_attr=weight, node_weight_attr=node_weight_attr)
    return CombinatorialCurvatureEngine(
        G, augmented_forman_directed_config(node_weight, face_weight), semantics
    ).compute_edges()
