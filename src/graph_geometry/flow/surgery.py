"""Graph surgery strategies applied during Ricci flow.

Ported from ``ricciflow_sim/core/surgery.py``. Two usage modes:
  1. Functions: ``surgery(G, "weight", 0.02)``, ``surgery_n(G, "weight", 1)``.
  2. Strategy objects: ``IntervalSurgery(surgery, interval=10, portion=0.02)``.

``resolve_surgery`` accepts ``None`` | a strategy | a dict
(``{"name", "portion", "interval"}``). ``portion`` means a *fraction* of edges
for ``surgery`` and a *count* of edges for ``surgery_n``, so its default depends
on the name (:data:`DEFAULT_PORTION`); :data:`DEFAULT_INTERVAL` is shared by the
Python API and experiment configs. The functions here do not print (the flow
engine reports); the cut behaviour matches the source exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Union

import networkx as nx

__all__ = [
    "DEFAULT_INTERVAL",
    "DEFAULT_PORTION",
    "SurgerySpec",
    "no_surgery",
    "surgery",
    "surgery_n",
    "NoSurgery",
    "IntervalSurgery",
    "resolve_surgery",
]


# ── surgery functions (signature: (G, weight, portion) -> G) ─────────────────


def no_surgery(G_origin: nx.Graph, *args, **kwargs) -> nx.Graph:
    """No surgery — return the graph unchanged."""
    return G_origin


def _cut_count(n_edges: int, cut_proportion: float) -> int:
    """``ceil(n_edges * cut_proportion)``, robust to binary rounding.

    The ceiling is the source's convention (it kept ``floor(n * (1 - p))``
    edges), so any positive proportion cuts at least one edge. Computing it as
    ``int(n * (1 - p))`` in floating point cut one edge too many whenever
    ``n * p`` is a whole number that ``1 - p`` cannot represent exactly (e.g.
    ``n=5, p=0.8`` removed all 5 edges instead of 4).
    """
    exact = n_edges * cut_proportion
    nearest = round(exact)
    if math.isclose(exact, nearest, rel_tol=1e-9, abs_tol=1e-12):
        return int(nearest)
    return math.ceil(exact)


def surgery(G_origin: nx.Graph, weight: str = "weight", cut_proportion: float = 0.03) -> nx.Graph:
    """Remove the ``ceil(cut_proportion * |E|)`` heaviest edges by ``weight`` (new graph).

    Any positive proportion therefore removes at least one edge, even when
    ``cut_proportion * |E| < 1`` (e.g. 0.02 on a graph with fewer than 50
    edges). Ties are broken by edge order.
    """
    if not 0 <= cut_proportion <= 1:
        raise ValueError(f"cut_proportion must be in [0, 1], got {cut_proportion}")
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)
    sorted_edges = sorted(w.items(), key=lambda x: x[1])
    n_cut = _cut_count(len(sorted_edges), cut_proportion)
    to_cut = [e for (e, _) in sorted_edges[len(sorted_edges) - n_cut:]]
    G.remove_edges_from(to_cut)
    return G


def surgery_n(G_origin: nx.Graph, weight: str = "weight", cut_n: int = 1) -> nx.Graph:
    """Remove the top ``cut_n`` edges by weight (new graph).

    ``cut_n`` is a whole number of edges. A fractional value used to be
    truncated, and one below 1 (e.g. the fraction 0.02 meant for
    :func:`surgery`) made the slice ``[-0:]`` remove *every* edge.
    """
    if isinstance(cut_n, bool) or not isinstance(cut_n, (int, float)) or (
        isinstance(cut_n, float) and not cut_n.is_integer()
    ):
        raise ValueError(
            f"cut_n must be a whole number of edges, got {cut_n!r}; use surgery() "
            f"with a fraction to cut a proportion."
        )
    cut_n = int(cut_n)
    G = G_origin.copy()
    w = nx.get_edge_attributes(G, weight)
    if not 0 <= cut_n <= G.number_of_edges():
        raise ValueError(
            f"cut_n must be in [0, {G.number_of_edges()}], got {cut_n}"
        )
    sorted_edges = sorted(w.items(), key=lambda x: x[1])
    to_cut = [e for (e, _) in sorted_edges[-cut_n:]] if cut_n else []
    G.remove_edges_from(to_cut)
    return G


# ── strategy wrappers ────────────────────────────────────────────────────────


class NoSurgery:
    """Null surgery — never modifies the graph.

    Strategy protocol: the flow engine calls ``should_apply(n)`` after the
    ``n``-th completed update (``n >= 1``) and, when it returns True, operates
    on that update's result with ``apply(G, attribute)``.
    """

    def should_apply(self, iteration: int) -> bool:
        return False

    def apply(self, G: nx.Graph, weight: str) -> nx.Graph:
        return G

    def __repr__(self) -> str:
        return "NoSurgery()"


@dataclass
class IntervalSurgery:
    """Apply a surgery function after every ``interval``-th update.

    ``interval=5`` operates after updates 5, 10, 15, ... (``should_apply`` is
    given the 1-based count of completed updates).

    ``fn`` has signature ``(G, weight, portion) -> G`` (e.g. :func:`surgery` or
    :func:`surgery_n`); ``portion`` is passed as its third argument.
    """

    fn: Callable
    interval: int = 10
    portion: float = 0.02

    def __post_init__(self) -> None:
        if isinstance(self.interval, bool) or not isinstance(self.interval, int) or self.interval < 1:
            raise ValueError(f"surgery interval must be a positive integer, got {self.interval!r}")

    def should_apply(self, iteration: int) -> bool:
        return iteration >= 1 and iteration % self.interval == 0

    def apply(self, G: nx.Graph, weight: str) -> nx.Graph:
        return self.fn(G, weight, self.portion)


SurgerySpec = Union[None, dict, NoSurgery, IntervalSurgery]

_FN_MAP = {"surgery": surgery, "surgery_n": surgery_n}

#: Iterations between surgeries when a spec does not say.
DEFAULT_INTERVAL = 30

#: ``portion`` when a spec does not say: a fraction for ``surgery``, a count for
#: ``surgery_n``. One shared default for both was wrong for one of them.
DEFAULT_PORTION = {"surgery": 0.02, "surgery_n": 1}

_SPEC_KEYS = {"name", "portion", "interval"}


def resolve_surgery(surgery_config: SurgerySpec):
    """Convert a surgery spec into a strategy object.

    - ``None`` -> :class:`NoSurgery`
    - a strategy object (has ``.should_apply``) -> returned as-is
    - a dict ``{"name", "portion", "interval"}`` -> :class:`IntervalSurgery`;
      missing ``portion``/``interval`` take :data:`DEFAULT_PORTION` (per name)
      and :data:`DEFAULT_INTERVAL`; unknown keys are rejected
    """
    if surgery_config is None:
        return NoSurgery()
    if hasattr(surgery_config, "should_apply"):
        return surgery_config
    if not isinstance(surgery_config, dict):
        raise TypeError(
            f"surgery must be None, a strategy object, or a dict; got {surgery_config!r}"
        )

    unknown = sorted(set(surgery_config) - _SPEC_KEYS)
    if unknown:
        raise ValueError(f"Unknown surgery key(s) {unknown}; expected {sorted(_SPEC_KEYS)}.")
    name = surgery_config.get("name", "no_surgery")
    if name == "no_surgery":
        return NoSurgery()
    fn = _FN_MAP.get(name)
    if fn is None:
        raise ValueError(f"Unknown surgery name: {name!r}. Use 'surgery' or 'surgery_n'.")
    portion = surgery_config.get("portion", DEFAULT_PORTION[name])
    if name == "surgery_n":
        if isinstance(portion, bool) or not isinstance(portion, (int, float)) or (
            isinstance(portion, float) and not portion.is_integer()
        ) or portion < 0:
            raise ValueError(
                f"surgery_n portion is a whole number of edges, got {portion!r}"
            )
    elif isinstance(portion, bool) or not isinstance(portion, (int, float)) or not 0 <= portion <= 1:
        raise ValueError(f"surgery portion is a fraction in [0, 1], got {portion!r}")
    return IntervalSurgery(
        fn=fn,
        interval=surgery_config.get("interval", DEFAULT_INTERVAL),
        portion=portion,
    )
