"""The combinatorial curvature template and its engine.

Every combinatorial definition in the package evaluates, for an edge
``e = (x, y)``, one closed expression over local incidences::

    kappa_c(e) = A_c(e)
                 - sum_{z in {x, y}} sum_{e' in S_c(z; e)} B_c(e, e', z)
                 + sum_{f in F_c(e)} C_c(e, f)

with five interchangeable components:

=========  ==========================================  ===========================
component  role                                        signature
=========  ==========================================  ===========================
``A_c``    endpoint contribution                       ``(ctx, e) -> float``
``S_c``    incident edges charged at endpoint ``z``    ``(ctx, z, e) -> edges``
``B_c``    charge for one incident edge                ``(ctx, e, e', z) -> float``
``F_c``    higher-order cells containing ``e``         ``(ctx, e) -> cells``
``C_c``    contribution of one cell                    ``(ctx, e, f) -> float``
=========  ==========================================  ===========================

A definition is a :class:`CombinatorialCurvatureConfig`; the
:class:`CombinatorialCurvatureEngine` validates the graph, builds a per-graph
context once (weights, node weights, neighbour sets) and assembles the terms in
that fixed order. It returns raw values and never annotates the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import networkx as nx

from ..graph.semantics import GraphSemantics
from .errors import CurvatureConfigurationError, CurvatureContractError

__all__ = [
    "CombinatorialContext",
    "CombinatorialCurvatureConfig",
    "CombinatorialCurvatureEngine",
]


@dataclass
class CombinatorialContext:
    """Per-graph data shared by the components of one evaluation.

    ``data`` holds whatever ``prepare`` precomputes (node weights, neighbour
    sets, floored edge weights), so components stay pure functions of the
    context and their arguments.
    """

    G: nx.Graph
    semantics: GraphSemantics
    data: dict = field(default_factory=dict)

    @property
    def weight(self) -> str:
        return self.semantics.weight_attr


def _no_preparation(ctx: CombinatorialContext) -> None:
    return None


@dataclass(frozen=True)
class CombinatorialCurvatureConfig:
    """The template components of one combinatorial definition.

    ``faces`` and ``face_term`` are optional together (no higher-order cells).
    ``prepare(ctx)`` fills ``ctx.data`` once per graph; ``validate(G, semantics,
    where)`` rejects graphs the definition does not apply to.
    """

    name: str
    endpoint: Callable[[CombinatorialContext, tuple], float]
    incidence: Callable[[CombinatorialContext, Any, tuple], Iterable]
    charge: Callable[[CombinatorialContext, tuple, tuple, Any], float]
    faces: Optional[Callable[[CombinatorialContext, tuple], Iterable]] = None
    face_term: Optional[Callable[[CombinatorialContext, tuple, Any], float]] = None
    prepare: Callable[[CombinatorialContext], None] = _no_preparation
    validate: Optional[Callable[[nx.Graph, GraphSemantics, str], None]] = None

    def __post_init__(self) -> None:
        for slot in ("endpoint", "incidence", "charge", "prepare"):
            if not callable(getattr(self, slot)):
                raise CurvatureConfigurationError(
                    f"{self.name}: component {slot!r} must be callable."
                )
        if (self.faces is None) != (self.face_term is None):
            raise CurvatureConfigurationError(
                f"{self.name}: 'faces' (F_c) and 'face_term' (C_c) are given together "
                f"or not at all."
            )


class CombinatorialCurvatureEngine:
    """Assemble ``A_c - sum B_c + sum C_c`` on every edge of ``G``."""

    def __init__(
        self,
        G: nx.Graph,
        config: CombinatorialCurvatureConfig,
        semantics: Optional[GraphSemantics] = None,
    ):
        if not isinstance(config, CombinatorialCurvatureConfig):
            raise CurvatureConfigurationError(
                f"config must be a CombinatorialCurvatureConfig; got {type(config).__name__}."
            )
        self.G = G
        self.config = config
        self.semantics = semantics if semantics is not None else GraphSemantics()
        self.context: Optional[CombinatorialContext] = None

    def _prepare(self) -> CombinatorialContext:
        if self.config.validate is not None:
            self.config.validate(self.G, self.semantics, self.config.name)
        ctx = CombinatorialContext(self.G, self.semantics)
        self.config.prepare(ctx)
        self.context = ctx
        return ctx

    def _terms(self, ctx: CombinatorialContext, u: Any, v: Any) -> tuple:
        config = self.config
        edge = (u, v)
        endpoint = config.endpoint(ctx, edge)
        charges = []
        for z in (u, v):
            total = 0.0
            for incident in config.incidence(ctx, z, edge):
                total += config.charge(ctx, edge, incident, z)
            charges.append(total)
        cells = None
        if config.faces is not None:
            cells = sum(config.face_term(ctx, edge, f) for f in config.faces(ctx, edge))
        return endpoint, charges, cells

    @staticmethod
    def _assemble(endpoint: float, charges: list, cells: Optional[float]) -> float:
        value = endpoint - charges[0] - charges[1]
        if cells is not None:
            value = value + cells
        return value

    def compute_edges(self) -> dict:
        """``{(u, v): kappa}`` on every edge, in ``G.edges()`` order."""
        ctx = self._prepare()
        values = {}
        for u, v in self.G.edges():
            value = self._assemble(*self._terms(ctx, u, v))
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CurvatureContractError(
                    f"{self.config.name}: the template produced {value!r} on ({u!r}, {v!r})."
                )
            values[(u, v)] = value
        return values

    def edge_terms(self) -> dict:
        """``{(u, v): {"A": ..., "B_x": ..., "B_y": ..., "C": ..., "kappa": ...}}``.

        A diagnostic view of the same assembly: which term drives each value.
        ``C`` is ``None`` for a definition without higher-order cells.
        """
        ctx = self._prepare()
        out = {}
        for u, v in self.G.edges():
            endpoint, charges, cells = self._terms(ctx, u, v)
            out[(u, v)] = {
                "A": endpoint, "B_x": charges[0], "B_y": charges[1], "C": cells,
                "kappa": self._assemble(endpoint, charges, cells),
            }
        return out
