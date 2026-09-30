"""Input validation shared by the built-in curvature methods.

Rationale: without these checks a ``NaN`` weight silently produced an ordinary
looking curvature (0.5), an infinite weight surfaced as an assertion from
inside POT, a zero ``distance`` returned 0.0, and a multigraph failed with a
NetworkX ``AtlasView`` ``TypeError``. Every method now fails early, naming the
attribute and the offending edge, before any solver runs.

Missing attributes are **not** an error: they fall back to 1.0, matching the
convention already used by the transition kernels and the flow engine.
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx

from .errors import CurvatureDomainError, CurvatureInputError

__all__ = [
    "reject_multigraph",
    "reject_self_loops",
    "validate_simple_graph",
    "validate_edge_attributes",
    "validate_signs",
]


def reject_multigraph(G: nx.Graph, where: str) -> None:
    """Raise a :class:`CurvatureDomainError` for Multi(Di)Graph input."""
    if G.is_multigraph():
        raise CurvatureDomainError(
            f"{where} does not support multigraphs (got {type(G).__name__}): "
            f"parallel edges have no single weight or distance. Collapse them "
            f"first, e.g. nx.Graph(G) or nx.DiGraph(G)."
        )


def reject_self_loops(G: nx.Graph, where: str) -> None:
    """Raise a :class:`CurvatureDomainError` if ``G`` has any self-loop.

    No shipped definition assigns a loop a curvature, and silently dropping one
    would change the input without a record, so removal is left to the caller.
    """
    loops = list(nx.selfloop_edges(G))
    if loops:
        raise CurvatureDomainError(
            f"{where} requires a graph without self-loops; found {len(loops)} "
            f"(e.g. {loops[0]!r}). Remove them explicitly, e.g. "
            f"G.remove_edges_from(nx.selfloop_edges(G)), so the preprocessing is "
            f"recorded."
        )


def validate_simple_graph(G: nx.Graph, where: str) -> None:
    """The one simple-graph policy: no multigraphs and no self-loops.

    Shared by standalone evaluation and the flow engine, so both refuse the same
    inputs with the same error instead of one of them repairing the graph.
    """
    reject_multigraph(G, where)
    reject_self_loops(G, where)


def _check_finite(value: Any, attr: str, u: Any, v: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CurvatureInputError(
            f"{where}: edge ({u!r}, {v!r}) has non-numeric {attr}={value!r}."
        )
    value = float(value)
    if not math.isfinite(value):
        raise CurvatureInputError(
            f"{where}: edge ({u!r}, {v!r}) has non-finite {attr}={value!r}. "
            f"Curvature is undefined for NaN or infinite edge attributes."
        )
    return value


def validate_edge_attributes(
    G: nx.Graph,
    *,
    weight: str | None = None,
    distance: str | None = None,
    where: str = "curvature",
    positive_weight: bool = True,
) -> None:
    """Check every edge's ``weight``/``distance`` before any solver runs.

    ``weight``/``distance`` name the attributes to check; pass ``None`` to skip
    one. ``positive_weight=False`` leaves the weight-domain check to the
    definition. For example, Eidi-Jost permits an all-zero neighbourhood, which
    becomes a uniform measure, but rejects negative weights. ``distance`` must
    always be strictly positive, since it is the metric behind the shortest
    paths and the divisor in ``kappa = 1 - W1/d``.
    """
    reject_multigraph(G, where)
    for u, v, data in G.edges(data=True):
        if weight is not None:
            w = _check_finite(data.get(weight, 1.0), weight, u, v, where)
            if positive_weight and w <= 0.0:
                raise CurvatureInputError(
                    f"{where}: edge ({u!r}, {v!r}) has {weight}={w!r}. The "
                    f"transition kernel needs strictly positive weights to form "
                    f"a probability measure; a non-positive weight yields a "
                    f"signed measure and a meaningless transport cost."
                )
        if distance is not None:
            d = _check_finite(data.get(distance, 1.0), distance, u, v, where)
            if d <= 0.0:
                raise CurvatureInputError(
                    f"{where}: edge ({u!r}, {v!r}) has {distance}={d!r}. The "
                    f"metric must be strictly positive; kappa = 1 - W1/d is "
                    f"undefined for a zero or negative edge length."
                )


def validate_signs(
    G: nx.Graph,
    *,
    sign: str = "sign",
    positive: Any = "promotion",
    where: str = "signed operation",
) -> tuple:
    """Check a signed graph's sign labels before a sign-consuming operation.

    A sign here is a *label*, not a number: an edge counts as positive when its
    ``sign`` attribute equals ``positive`` and as negative otherwise. That rule
    fails silently in two ways, which this function turns into errors:

    - the data draws a two-way distinction but neither label is ``positive``
      (``"activation"``/``"inhibition"`` read against ``positive="promotion"``),
      so both collapse to negative and the result is uniformly wrong rather than
      visibly broken. A *single* label is left alone: an all-positive or
      all-negative graph is a legitimate signed graph, not a typo;
    - the attribute is absent, so every edge falls back to positive and a signed
      construction quietly degenerates into an unsigned one.

    Returns the observed labels as a sorted tuple.
    """
    labels = {data[sign] for _, _, data in G.edges(data=True) if sign in data}
    missing = sum(1 for _, _, data in G.edges(data=True) if sign not in data)

    if not labels:
        raise ValueError(
            f"{where}: no edge carries a {sign!r} attribute, so every edge would "
            f"default to positive and the sign would have no effect. Set the sign "
            f"attribute, or pass sign=<the attribute your graph actually uses>."
        )
    if missing:
        raise ValueError(
            f"{where}: {missing} of {G.number_of_edges()} edges have no {sign!r} "
            f"attribute. Those edges would silently be read as positive; assign "
            f"every edge a sign so the convention is explicit."
        )
    if len(labels) == 2 and positive not in labels:
        # A single label is legitimate -- an all-positive or an all-negative
        # graph is a real signed graph. Two labels neither of which is
        # ``positive`` is not: the distinction the data draws would collapse,
        # with both values read as negative.
        raise ValueError(
            f"{where}: {sign!r} takes two values "
            f"{tuple(sorted(map(str, labels)))} and positive={positive!r} is "
            f"neither, so both would be read as negative and the distinction "
            f"lost. Pass the label your data uses for a positive edge."
        )
    if len(labels) > 2:
        raise ValueError(
            f"{where}: a sign is binary, but {sign!r} takes "
            f"{len(labels)} distinct values {tuple(sorted(map(str, labels)))}. "
            f"Everything other than positive={positive!r} is read as negative."
        )
    return tuple(sorted(map(str, labels)))
