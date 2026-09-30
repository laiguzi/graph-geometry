"""Edge trajectories: per-edge values across the committed states of a flow.

The data layer behind :func:`graph_geometry.viz.plot_edge_trajectories` and the
Streamlit trajectory view. It depends on neither matplotlib nor Streamlit.

Alignment
---------
Every committed state of a :class:`~graph_geometry.flow.FlowResult` carries the
``ricciCurvature`` computed *on that state* (the engine recomputes curvature
before each commit, including after surgery). Trajectories therefore read
``kappa_e^t`` and ``q_e^t`` from the same graph ``snapshots[t]``: nothing is
recomputed and no curvature history is kept beyond the snapshots the result
already holds, so extraction adds no transport solves and no extra graph copies.

Edges and surgery
-----------------
The edge set is the union over all states in order of first appearance -- for a
flow, the initial graph's edges. An edge removed by surgery is reported as
``present=False`` with ``None`` values from that state on; it is never filled
with zero or with its last value, so a plotted line stops where the edge ends.
Undirected edges are identified regardless of orientation and reported in the
orientation of their first appearance; directed edges keep their orientation.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

__all__ = ["EdgeTrajectories", "extract_edge_trajectories", "TRAJECTORY_FIELDS"]

CURVATURE_ATTR = "ricciCurvature"

#: Column order of :meth:`EdgeTrajectories.rows` (and of the CSV export).
TRAJECTORY_FIELDS = (
    "iteration", "source", "target", "edge", "group", "present",
    "curvature", "evolving_quantity", "evolving_attribute", "evolving_role",
)

_QUANTITIES = {
    "curvature": "curvature",
    "evolving_quantity": "evolving_quantity",
    "evolving": "evolving_quantity",
}


def _quantity(name: str) -> str:
    try:
        return _QUANTITIES[name]
    except KeyError:
        raise ValueError(
            f"quantity must be 'curvature' or 'evolving_quantity', got {name!r}."
        ) from None


@dataclass(frozen=True)
class EdgeTrajectories:
    """Snapshot-aligned per-edge series, with optional edge groups.

    Attributes
    ----------
    edges :
        Edge identities ``(u, v)``, in order of first appearance.
    directed :
        Whether ``(u, v)`` and ``(v, u)`` are different edges.
    n_states :
        Number of committed states; index ``t`` of every series is state ``t``
        (``t = 0`` is the initial graph).
    curvature, evolving_quantity :
        ``{edge: (value or None, ...)}``; ``None`` where the edge is absent or
        the attribute is missing.
    present :
        ``{edge: (bool, ...)}``: whether the edge exists in state ``t``.
    groups :
        ``{edge: group name}`` for the grouped edges only.
    evolving_attr, evolve, coupled :
        The attribute that evolved, its role (``"weight"``, ``"distance"`` or
        ``None`` when unknown) and whether it also plays the other role.
    """

    edges: tuple
    directed: bool
    n_states: int
    curvature: Mapping
    evolving_quantity: Mapping
    present: Mapping
    groups: Mapping
    evolving_attr: Optional[str] = None
    evolve: Optional[str] = None
    coupled: bool = False

    # ── identities ──────────────────────────────────────────────────────

    def canonical_edge(self, edge) -> tuple:
        """The stored identity of ``edge``; ``ValueError`` if it never occurs."""
        return _canonicalize(edge, self._index(), self.directed, "trajectories")

    def _index(self) -> dict:
        return {_identity(e, self.directed): e for e in self.edges}

    def edge_label(self, edge) -> str:
        """Readable, unambiguous label: ``'a' → 'b'`` or ``0 — 1``."""
        return self._label(self.canonical_edge(edge))

    def _label(self, edge) -> str:
        u, v = edge
        return f"{u!r} {'→' if self.directed else '—'} {v!r}"

    # ── series ──────────────────────────────────────────────────────────

    def series(self, quantity: str = "curvature") -> dict:
        """``{edge: [value or None per state]}`` for one quantity."""
        return {e: list(v) for e, v in getattr(self, _quantity(quantity)).items()}

    def quantity_label(self, quantity: str = "curvature") -> str:
        """Axis label naming the quantity by its role, not its attribute name."""
        if _quantity(quantity) == "curvature":
            return "edge curvature"
        if self.evolve == "weight":
            return "edge weight (= distance)" if self.coupled else "edge weight"
        if self.evolve == "distance":
            return "edge distance (= weight)" if self.coupled else "edge distance"
        if self.evolving_attr is not None:
            return f"evolving edge quantity ({self.evolving_attr})"
        return "evolving edge quantity"

    def group_members(self) -> dict:
        """``{group: (edges...)}`` in order of first appearance among ``edges``."""
        members: dict = {}
        for e in self.edges:
            if e in self.groups:
                members.setdefault(self.groups[e], []).append(e)
        return {g: tuple(es) for g, es in members.items()}

    def aggregate(self, quantity: str = "curvature", how: str = "mean",
                  members: Optional[Mapping] = None) -> dict:
        """Per-group summary per state over the edges **present** in that state.

        ``members`` defaults to :meth:`group_members`. A group that has lost
        edges to surgery is summarised over its remaining edges (so the summary
        can jump when an edge leaves); a state where none remain gives ``None``.
        ``how`` is ``"mean"``, ``"median"``, ``"min"`` or ``"max"``.
        """
        reducers = {
            "mean": lambda xs: math.fsum(xs) / len(xs),
            "median": _median,
            "min": min,
            "max": max,
        }
        if how not in reducers:
            raise ValueError(f"how must be one of {sorted(reducers)}, got {how!r}.")
        reduce = reducers[how]
        data = getattr(self, _quantity(quantity))
        members = self.group_members() if members is None else members
        out = {}
        for group, edges in members.items():
            row = []
            for t in range(self.n_states):
                values = [data[e][t] for e in edges if data[e][t] is not None]
                row.append(reduce(values) if values else None)
            out[group] = row
        return out

    # ── export ──────────────────────────────────────────────────────────

    def rows(self) -> list[dict]:
        """Tidy records, one per ``(state, edge)``, keyed by :data:`TRAJECTORY_FIELDS`."""
        rows = []
        labels = {e: self._label(e) for e in self.edges}
        for t in range(self.n_states):
            for e in self.edges:
                u, v = e
                rows.append({
                    "iteration": t,
                    "source": u,
                    "target": v,
                    "edge": labels[e],
                    "group": self.groups.get(e),
                    "present": self.present[e][t],
                    "curvature": self.curvature[e][t],
                    "evolving_quantity": self.evolving_quantity[e][t],
                    "evolving_attribute": self.evolving_attr,
                    "evolving_role": self.evolve,
                })
        return rows

    def to_dict(self) -> dict:
        """JSON-safe summary: metadata plus :meth:`rows` (node ids as given)."""
        return {
            "directed": self.directed,
            "n_states": self.n_states,
            "evolving_attr": self.evolving_attr,
            "evolve": self.evolve,
            "coupled": self.coupled,
            "rows": self.rows(),
        }


# ── extraction ──────────────────────────────────────────────────────────────


def extract_edge_trajectories(
    flow: Any,
    *,
    edges: Optional[Iterable] = None,
    groups: Optional[Mapping] = None,
    evolving_attr: Optional[str] = None,
    evolve: Optional[str] = None,
    curvature_attr: str = CURVATURE_ATTR,
) -> EdgeTrajectories:
    """Read per-edge curvature and evolving-quantity series from a flow.

    Parameters
    ----------
    flow :
        A :class:`~graph_geometry.flow.FlowResult`, an object with a ``result``
        attribute holding one (e.g. a run :class:`RicciFlowSimulator`), or a
        sequence of graph states.
    edges :
        Restrict to these edges (either orientation for an undirected flow).
        Each must occur in some state, otherwise ``ValueError``. Default: all.
    groups :
        ``{edge: group name}``. Every key must be an edge of the initial state
        (``ValueError`` otherwise); edges not in the mapping are ungrouped.
        Keys outside ``edges`` are validated but not reported.
    evolving_attr, evolve :
        The evolving attribute and its role (``"weight"``/``"distance"``).
        Taken from the ``FlowResult`` when omitted; required for a bare
        sequence of states if the evolving quantity is wanted.
    curvature_attr :
        Edge attribute holding curvature on each state.

    The flow is only read: no graph or attribute is modified.
    """
    result = getattr(flow, "result", None) if not _is_flow_result(flow) else flow
    if result is not None and _is_flow_result(result):
        snapshots = list(result.snapshots)
        if evolve is None and evolving_attr in (None, result.evolving_attr):
            evolve = result.evolve
        evolving_attr = evolving_attr or result.evolving_attr
        coupled = bool(result.coupled) and evolving_attr == result.evolving_attr
    else:
        if isinstance(flow, nx.Graph):
            raise TypeError("pass a FlowResult or a sequence of graph states, not one graph.")
        snapshots = list(flow)
        coupled = False
    if evolve not in (None, "weight", "distance"):
        raise ValueError(f"evolve must be 'weight', 'distance' or None, got {evolve!r}.")
    if not snapshots:
        raise ValueError("the flow has no committed states.")
    if not all(isinstance(G, nx.Graph) for G in snapshots):
        raise TypeError("every flow state must be a networkx graph.")
    directed = snapshots[0].is_directed()
    if any(G.is_directed() != directed for G in snapshots):
        raise ValueError("every flow state must have the same graph direction.")
    if any(G.is_multigraph() for G in snapshots):
        raise ValueError("edge trajectories are defined for simple graphs, not multigraphs.")

    index: dict = {}
    for G in snapshots:
        for e in G.edges():
            index.setdefault(_identity(e, directed), tuple(e))

    if edges is None:
        selected = list(index.values())
    else:
        selected, seen = [], set()
        for edge in edges:
            e = _canonicalize(edge, index, directed, "any flow state")
            if e not in seen:
                seen.add(e)
                selected.append(e)

    grouped: dict = {}
    if groups is not None:
        initial = {_identity(e, directed): tuple(e) for e in snapshots[0].edges()}
        chosen = set(selected)
        for edge, name in groups.items():
            e = _canonicalize(edge, initial, directed, "the initial state")
            e = index[_identity(e, directed)]
            if e in grouped and grouped[e] != name:
                raise ValueError(f"edge {edge!r} is mapped to two groups: "
                                 f"{grouped[e]!r} and {name!r}.")
            if e in chosen:
                grouped[e] = name

    curvature, evolving, present = {}, {}, {}
    for e in selected:
        u, v = e
        k_row, q_row, p_row = [], [], []
        for G in snapshots:
            has = G.has_edge(u, v)
            data = G.edges[u, v] if has else {}
            p_row.append(has)
            k_row.append(_value(data.get(curvature_attr)))
            q_row.append(_value(data.get(evolving_attr)) if evolving_attr else None)
        curvature[e], evolving[e], present[e] = tuple(k_row), tuple(q_row), tuple(p_row)

    return EdgeTrajectories(
        edges=tuple(selected),
        directed=directed,
        n_states=len(snapshots),
        curvature=curvature,
        evolving_quantity=evolving,
        present=present,
        groups=grouped,
        evolving_attr=evolving_attr,
        evolve=evolve,
        coupled=coupled,
    )


# ── helpers ─────────────────────────────────────────────────────────────────


def _is_flow_result(obj) -> bool:
    from .engine import FlowResult

    return isinstance(obj, FlowResult)


def _identity(edge, directed: bool) -> Hashable:
    u, v = edge
    return (u, v) if directed else frozenset((u, v))


def _canonicalize(edge, index: Mapping, directed: bool, where: str) -> tuple:
    try:
        u, v = edge
    except (TypeError, ValueError):
        raise ValueError(f"an edge must be a pair of nodes, got {edge!r}.") from None
    key = _identity((u, v), directed)
    if key not in index:
        kind = "directed edge" if directed else "edge"
        raise ValueError(f"{kind} {edge!r} does not occur in {where}.")
    return index[key]


def _value(x) -> Optional[float]:
    return None if x is None else float(x)


def _median(xs):
    xs = sorted(xs)
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2
