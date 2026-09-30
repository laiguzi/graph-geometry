"""Graph configuration: node/edge factor spec + ``apply_config``.

``GraphConfig`` bundles the edge-weight, edge-distance and node-factor strategies
(plus the attribute names to write and an RNG seed). ``apply_config`` resolves
those strategies and writes the attributes onto a graph.

This generalizes ``directed_ricciflow``'s ``GraphDataLoader.assign_attributes``,
which only supported ``fixed``/``random`` and used the global ``np.random``. Here
strategies also include ``degree`` and callables, and randomness is seedable and
reproducible.

Semantics preserved from the original:
- attributes are only written when **absent**, so source-provided weights (e.g.
  weighted edge lists) are not clobbered. Pass ``overwrite=True`` to force.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import networkx as nx

from .factors import (
    FactorSpec,
    as_rng,
    resolve_edge_factor,
    resolve_node_factor,
)
from .semantics import GraphSemantics

__all__ = ["GraphConfig", "apply_config"]


@dataclass
class GraphConfig:
    """Specification of node/edge factors to assign to a graph.

    Parameters
    ----------
    edge_weight, edge_distance, node_factor : FactorSpec
        Strategy for each attribute: ``"fixed"`` | ``"degree"`` | ``"random"``,
        a constant number, or a callable (see :mod:`graph_geometry.graph.factors`).
    weight_attr, distance_attr, node_attr : str
        Attribute names to write. Defaults match the three forks
        (``"weight"``/``"distance"`` on edges, ``"weight"`` on nodes).
    sign_attr : str
        Edge attribute holding the sign, for the signed constructions
        (:func:`~graph_geometry.curvature.signed_double_cover`). Declared here
        with the other semantic components rather than passed ad hoc, but never
        *written*: a sign is data that arrives with the graph, not a factor to
        be generated, so no ``edge_sign`` strategy exists.
    seed : int | None
        Seed for the ``"random"`` strategy. ``None`` is nondeterministic.
    overwrite : bool
        If ``False`` (default), only write attributes that are absent, matching
        the legacy ``assign_attributes`` behavior. If ``True``, overwrite.
    """

    edge_weight: FactorSpec = "fixed"
    edge_distance: FactorSpec = "fixed"
    node_factor: FactorSpec = "fixed"
    weight_attr: str = "weight"
    distance_attr: str = "distance"
    node_attr: str = "weight"
    sign_attr: str = "sign"
    seed: int | None = None
    overwrite: bool = False

    def to_semantics(self, *, node_weight: bool = False) -> GraphSemantics:
        """The :class:`GraphSemantics` that reads what this config writes.

        ``GraphConfig`` *assigns* attributes; ``GraphSemantics`` *interprets*
        them. The written node factor becomes the Forman node weight only when
        ``node_weight=True``: a node factor is not a node weight by default, and
        silently promoting it would change which convention a Forman run uses.
        """
        return GraphSemantics(
            weight_attr=self.weight_attr,
            distance_attr=self.distance_attr,
            node_weight_attr=self.node_attr if node_weight else None,
            sign_attr=self.sign_attr,
        )


def apply_config(
    G: nx.Graph,
    cfg: GraphConfig | None = None,
    *,
    copy: bool = False,
    **overrides: object,
) -> nx.Graph:
    """Write the configured edge/node factors onto ``G`` and return it.

    Parameters
    ----------
    G : networkx graph
        Graph to annotate. Directed and undirected are both supported.
    cfg : GraphConfig, optional
        The configuration. If omitted, one is built from ``**overrides``.
    copy : bool
        If ``True``, annotate and return a copy, leaving ``G`` untouched.
        Default ``False`` (annotate in place). Either way the graph is returned.
    **overrides
        Field overrides applied on top of ``cfg`` (or used to build a fresh
        ``GraphConfig`` when ``cfg`` is ``None``), e.g.
        ``apply_config(G, edge_weight="degree", seed=0)``.

    Notes
    -----
    Edge attributes are assigned before node attributes; when a strategy is
    ``"random"`` the draw order is edge-weight, edge-distance per edge (in graph
    edge order), then node factors (in node order), all from one seeded
    generator — so a given ``(graph, seed)`` is fully reproducible.
    """
    if cfg is None:
        cfg = GraphConfig(**overrides)  # type: ignore[arg-type]
    elif overrides:
        cfg = replace(cfg, **overrides)  # type: ignore[arg-type]

    if copy:
        G = G.copy()

    if G.is_multigraph():
        raise TypeError(
            f"GraphConfig does not support multigraphs (got {type(G).__name__}): "
            "collapse parallel edges before assigning one weight or distance "
            "per node pair."
        )

    rng = as_rng(cfg.seed)
    edge_weight = resolve_edge_factor(cfg.edge_weight, rng)
    edge_distance = resolve_edge_factor(cfg.edge_distance, rng)
    node_factor = resolve_node_factor(cfg.node_factor, rng)

    for u, v in G.edges():
        data = G[u][v]
        if cfg.overwrite or cfg.weight_attr not in data:
            data[cfg.weight_attr] = edge_weight(u, v, G)
        if cfg.overwrite or cfg.distance_attr not in data:
            data[cfg.distance_attr] = edge_distance(u, v, G)

    for n in G.nodes():
        node = G.nodes[n]
        if cfg.overwrite or cfg.node_attr not in node:
            node[cfg.node_attr] = node_factor(n, G)

    return G
