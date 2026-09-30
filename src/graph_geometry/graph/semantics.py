"""How existing graph attributes are *interpreted*.

:class:`~graph_geometry.graph.config.GraphConfig` creates and assigns
attributes; :class:`GraphSemantics` says which attribute plays which role when
a curvature reads the graph. The two are deliberately separate: a loaded file
arrives with attributes nobody generated, and it is their interpretation --
strength versus metric, which node attribute is a Forman node weight -- that has
to be recorded with a result.

This module imports nothing from the curvature layer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

__all__ = ["GraphSemantics", "MISSING_POLICIES"]

#: What to do when an edge lacks a consumed weight or distance attribute.
MISSING_POLICIES = ("unit", "error")


def _attr_name(value: Any, field: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"GraphSemantics.{field} must be a non-empty attribute name"
            f"{' or None' if optional else ''}; got {value!r}."
        )


@dataclass(frozen=True)
class GraphSemantics:
    """Attribute roles a curvature computation reads.

    Parameters
    ----------
    weight_attr :
        Edge attribute holding connection strength (transition kernels, Forman
        edge weights).
    distance_attr :
        Edge attribute holding the metric (shortest paths, transport cost).
        Setting it equal to ``weight_attr`` couples the two roles explicitly.
    node_weight_attr :
        Node attribute holding Forman node weights. ``None`` lets a
        definition's own ``node_weight`` parameter select a named scheme; when
        set, every node must carry a value.
    sign_attr :
        Edge attribute holding a sign label, consumed only by explicit signed
        preprocessing such as the signed double cover. No curvature reads it.
    beta_attr :
        Node attribute holding a per-node balancing factor, read by the mixed
        transport kernel only under ``beta_strategy="node_attr"``; every node
        must then carry a value in ``[0, 1]``.
    missing_weight, missing_distance :
        ``"unit"`` treats an absent attribute as 1; ``"error"`` rejects it.
    """

    weight_attr: str = "weight"
    distance_attr: str = "distance"
    node_weight_attr: str | None = None
    sign_attr: str | None = "sign"
    beta_attr: str = "beta"
    missing_weight: Literal["unit", "error"] = "unit"
    missing_distance: Literal["unit", "error"] = "unit"

    def __post_init__(self) -> None:
        _attr_name(self.weight_attr, "weight_attr")
        _attr_name(self.distance_attr, "distance_attr")
        _attr_name(self.node_weight_attr, "node_weight_attr", optional=True)
        _attr_name(self.sign_attr, "sign_attr", optional=True)
        _attr_name(self.beta_attr, "beta_attr")
        for field in ("missing_weight", "missing_distance"):
            if getattr(self, field) not in MISSING_POLICIES:
                raise ValueError(
                    f"GraphSemantics.{field} must be one of {MISSING_POLICIES}; "
                    f"got {getattr(self, field)!r}."
                )

    @property
    def coupled(self) -> bool:
        """True when one attribute is both the weight and the distance."""
        return self.weight_attr == self.distance_attr

    def to_dict(self) -> dict:
        """JSON-safe representation for experiment provenance."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "GraphSemantics":
        """Build from a mapping, rejecting unknown keys."""
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"Unknown semantics field(s) {unknown}; expected a subset of "
                f"{sorted(known)}."
            )
        return cls(**data)
