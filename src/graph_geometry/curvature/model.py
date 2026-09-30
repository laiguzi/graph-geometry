"""Curvature metadata, requests, resolved records and results.

These records are the public contract between the registry, the evaluator and
every consumer (comparison, flow, experiments, the UI). They are immutable:
input mappings are copied on construction and their values frozen recursively
(:func:`freeze_value`), so a record attached to a result cannot change after the
run that produced it, even when the caller later mutates a nested mapping or
list it passed in. Each has a JSON-safe ``to_dict()`` for experiment provenance.

One limit is deliberate. Python-only compatibility values that are arbitrary
objects -- a per-node ``beta`` or ``node_weight`` callable, a kernel instance --
are kept as they are: a record can hold them, but it cannot make the object
itself immutable. Such values are also why those forms cannot be persisted in an
experiment file.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Literal, Protocol

import networkx as nx

from ..graph.semantics import GraphSemantics
from .errors import CurvatureConfigurationError

__all__ = [
    "GraphKind",
    "CurvatureScope",
    "CurvatureFamily",
    "CurvatureStatus",
    "SemanticChannel",
    "FailureKind",
    "GRAPH_KINDS",
    "CURVATURE_SCOPES",
    "CURVATURE_STATUSES",
    "SEMANTIC_CHANNELS",
    "FAILURE_KINDS",
    "FrozenMapping",
    "freeze_value",
    "CurvatureCapabilities",
    "ParameterSpec",
    "CurvatureRequest",
    "ResolvedCurvature",
    "CurvatureResult",
    "CurvatureFailure",
    "EdgeCompute",
    "PairCompute",
    "CurvatureResolver",
    "graph_kind_of",
    "json_safe",
]

GraphKind = Literal["directed", "undirected"]
CurvatureScope = Literal["edges", "pairs"]
CurvatureFamily = Literal["transport", "combinatorial"]
CurvatureStatus = Literal["core", "specialized", "experimental"]
SemanticChannel = Literal["topology", "edge_weight", "edge_distance", "node_weight"]
FailureKind = Literal[
    "configuration", "inapplicable", "invalid_input",
    "numerical_failure", "contract_failure", "internal_failure",
]

GRAPH_KINDS: tuple[str, ...] = ("directed", "undirected")
CURVATURE_SCOPES: tuple[str, ...] = ("edges", "pairs")
CURVATURE_FAMILIES: tuple[str, ...] = ("transport", "combinatorial")
CURVATURE_STATUSES: tuple[str, ...] = ("core", "specialized", "experimental")
SEMANTIC_CHANNELS: tuple[str, ...] = (
    "topology", "edge_weight", "edge_distance", "node_weight",
)
FAILURE_KINDS: tuple[str, ...] = (
    "configuration", "inapplicable", "invalid_input",
    "numerical_failure", "contract_failure", "internal_failure",
)
PARAMETER_KINDS = ("float", "integer", "boolean", "choice", "string")


def graph_kind_of(G: nx.Graph) -> str:
    """The graph's actual kind; never declared by a caller."""
    return "directed" if G.is_directed() else "undirected"


def json_safe(value: Any) -> Any:
    """Best-effort JSON-safe copy: containers recurse, unknown objects -> repr."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, Real):
        value = float(value) if not isinstance(value, int) else int(value)
        if isinstance(value, float) and not math.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        items = [json_safe(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    return repr(value)


_SCALARS = (str, bytes, int, float, complex, bool, type(None))


def freeze_value(value: Any) -> Any:
    """An immutable copy of a record value, built recursively.

    ``Mapping`` becomes :class:`FrozenMapping`, ``list``/``tuple`` become
    ``tuple`` and ``set``/``frozenset`` become ``frozenset``, with their contents
    frozen the same way. Mapping *keys* are kept exactly as given, since they may
    be graph nodes or edges whose identity matters. Scalars, callables and any
    other object are returned unchanged (see the module docstring).
    """
    if isinstance(value, _SCALARS) or isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item) for item in value)
    return value


class FrozenMapping(Mapping):
    """A read-only, picklable, recursively frozen copy of a mapping.

    Values are frozen with :func:`freeze_value`; keys are kept as given. It is
    hashable whenever its frozen values are.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping | None = None):
        object.__setattr__(
            self, "_data", {key: freeze_value(value) for key, value in dict(data or {}).items()}
        )

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __setattr__(self, name, value):
        raise AttributeError("FrozenMapping is read-only")

    def __reduce__(self):
        return (FrozenMapping, (self._data,))

    def __hash__(self):
        return hash(tuple(sorted(self._data.items(), key=repr)))


def _freeze(obj: Any, name: str, value: Any) -> None:
    object.__setattr__(obj, name, value)


# ── metadata ────────────────────────────────────────────────────────────────


def _check_members(value: Any, allowed: tuple[str, ...], what: str) -> frozenset:
    members = frozenset(value)
    unknown = sorted(members - set(allowed))
    if unknown:
        raise CurvatureConfigurationError(
            f"Unknown {what} {unknown}; allowed: {list(allowed)}."
        )
    return members


@dataclass(frozen=True)
class CurvatureCapabilities:
    """What a curvature definition applies to and what it reads."""

    family: CurvatureFamily
    status: CurvatureStatus
    graph_kinds: frozenset[GraphKind]
    scopes: frozenset[CurvatureScope]
    consumes: frozenset[SemanticChannel]

    def __post_init__(self) -> None:
        if self.family not in CURVATURE_FAMILIES:
            raise CurvatureConfigurationError(
                f"family must be one of {list(CURVATURE_FAMILIES)}; got {self.family!r}."
            )
        if self.status not in CURVATURE_STATUSES:
            raise CurvatureConfigurationError(
                f"status must be one of {list(CURVATURE_STATUSES)}; got {self.status!r}."
            )
        _freeze(self, "graph_kinds", _check_members(self.graph_kinds, GRAPH_KINDS, "graph kinds"))
        _freeze(self, "scopes", _check_members(self.scopes, CURVATURE_SCOPES, "scopes"))
        _freeze(self, "consumes", _check_members(self.consumes, SEMANTIC_CHANNELS, "semantic channels"))
        if not self.graph_kinds:
            raise CurvatureConfigurationError("graph_kinds must not be empty.")
        if "edges" not in self.scopes:
            raise CurvatureConfigurationError(
                "every curvature definition must support the 'edges' scope."
            )

    def to_dict(self) -> dict:
        return {
            "family": self.family,
            "status": self.status,
            "graph_kinds": sorted(self.graph_kinds),
            "scopes": sorted(self.scopes),
            "consumes": sorted(self.consumes),
        }


@dataclass(frozen=True)
class ParameterSpec:
    """One mathematical request parameter: type, default, validation, activation.

    ``active_when`` maps another parameter's name to the values under which this
    one is active, e.g. ``{"kernel": ("mixed",)}``; it is evaluated against the
    *resolved* parameters, so an ``auto`` kernel counts as what it resolves to.
    ``graph_kinds`` restricts activation to some graph kinds.

    Helper fields beyond the published contract:

    - ``minimum_inclusive`` / ``maximum_inclusive`` express open bounds
      (``face_weight`` must be strictly positive);
    - ``inactive_when_semantics`` names :class:`GraphSemantics` fields that,
      when set, deactivate the parameter (``node_weight`` yields to
      ``node_weight_attr``);
    - ``legacy_validator`` accepts values outside the schema kind from the
      compatibility wrappers only, for one deprecation cycle. It returns the
      validated value or raises.
    """

    name: str
    kind: Literal["float", "integer", "boolean", "choice", "string"]
    default: object
    choices: tuple[object, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    description: str = ""
    graph_kinds: frozenset[GraphKind] = frozenset({"directed", "undirected"})
    active_when: Mapping[str, tuple[object, ...]] = field(default_factory=dict)
    minimum_inclusive: bool = True
    maximum_inclusive: bool = True
    inactive_when_semantics: tuple[str, ...] = ()
    legacy_validator: Callable[[Any], Any] | None = field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise CurvatureConfigurationError(
                f"parameter name must be an identifier; got {self.name!r}."
            )
        if self.kind not in PARAMETER_KINDS:
            raise CurvatureConfigurationError(
                f"parameter {self.name!r}: kind must be one of "
                f"{list(PARAMETER_KINDS)}; got {self.kind!r}."
            )
        if self.kind == "choice" and not self.choices:
            raise CurvatureConfigurationError(
                f"choice parameter {self.name!r} needs at least one choice."
            )
        _freeze(self, "choices", tuple(self.choices))
        _freeze(self, "graph_kinds", _check_members(self.graph_kinds, GRAPH_KINDS, "graph kinds"))
        _freeze(self, "active_when", FrozenMapping(
            {k: tuple(v) for k, v in dict(self.active_when).items()}
        ))
        _freeze(self, "inactive_when_semantics", tuple(self.inactive_when_semantics))
        for semantic in self.inactive_when_semantics:
            if semantic not in GraphSemantics.__dataclass_fields__:
                raise CurvatureConfigurationError(
                    f"parameter {self.name!r}: unknown semantics field {semantic!r}."
                )
        # the default must satisfy the parameter's own schema
        self.validate(self.default, where="default")

    def validate(self, value: Any, *, where: str = "request") -> Any:
        """Return ``value`` coerced to the parameter kind, or raise."""
        name = f"parameter {self.name!r}"
        bad = CurvatureConfigurationError
        if self.kind == "boolean":
            if not isinstance(value, bool):
                raise bad(f"{name} must be a boolean; got {value!r} ({where}).")
            return value
        if self.kind == "string":
            if not isinstance(value, str):
                raise bad(f"{name} must be a string; got {value!r} ({where}).")
            return value
        if self.kind == "choice":
            for choice in self.choices:
                if type(choice) is type(value) and choice == value:
                    return choice
            raise bad(
                f"{name} must be one of {list(self.choices)}; got {value!r} ({where})."
            )
        # numeric kinds
        if isinstance(value, bool) or not isinstance(value, Real):
            raise bad(f"{name} must be a number; got {value!r} ({where}).")
        if self.kind == "integer":
            if isinstance(value, float) and not value.is_integer():
                raise bad(f"{name} must be an integer; got {value!r} ({where}).")
            value = int(value)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise bad(f"{name} must be finite; got {value!r} ({where}).")
        if self.minimum is not None:
            below = value < self.minimum if self.minimum_inclusive else value <= self.minimum
            if below:
                op = ">=" if self.minimum_inclusive else ">"
                raise bad(f"{name} must be {op} {self.minimum}; got {value!r} ({where}).")
        if self.maximum is not None:
            above = value > self.maximum if self.maximum_inclusive else value >= self.maximum
            if above:
                op = "<=" if self.maximum_inclusive else "<"
                raise bad(f"{name} must be {op} {self.maximum}; got {value!r} ({where}).")
        return value

    def is_active(
        self,
        graph_kind: str,
        parameters: Mapping[str, Any],
        semantics: GraphSemantics | None = None,
    ) -> bool:
        """Whether the parameter takes part in this resolved configuration."""
        if graph_kind not in self.graph_kinds:
            return False
        for other, allowed in self.active_when.items():
            if parameters.get(other) not in allowed:
                return False
        if semantics is not None:
            for semantic in self.inactive_when_semantics:
                if getattr(semantics, semantic) is not None:
                    return False
        return True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "default": json_safe(self.default),
            "choices": json_safe(list(self.choices)),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "minimum_inclusive": self.minimum_inclusive,
            "maximum_inclusive": self.maximum_inclusive,
            "description": self.description,
            "graph_kinds": sorted(self.graph_kinds),
            "active_when": json_safe(dict(self.active_when)),
            "inactive_when_semantics": list(self.inactive_when_semantics),
        }


# ── requests, resolved records, results ─────────────────────────────────────


@dataclass(frozen=True)
class CurvatureRequest:
    """What to compute: a method name, its parameters, and a display label."""

    method: str
    parameters: Mapping[str, object] = field(default_factory=dict)
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise CurvatureConfigurationError(
                f"CurvatureRequest.method must be a non-empty name; got {self.method!r}."
            )
        if not isinstance(self.parameters, Mapping):
            raise CurvatureConfigurationError(
                f"CurvatureRequest.parameters must be a mapping; got "
                f"{type(self.parameters).__name__}."
            )
        if self.label is not None and (not isinstance(self.label, str) or not self.label):
            raise CurvatureConfigurationError(
                f"CurvatureRequest.label must be a non-empty string or None; got {self.label!r}."
            )
        _freeze(self, "parameters", FrozenMapping(self.parameters))

    @property
    def display_label(self) -> str:
        """``label`` when given, otherwise the requested method name."""
        return self.label or self.method

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "parameters": json_safe(dict(self.parameters)),
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CurvatureRequest":
        return cls(
            method=data["method"],
            parameters=dict(data.get("parameters") or {}),
            label=data.get("label"),
        )


@dataclass(frozen=True)
class ResolvedCurvature:
    """A request after resolution: every convention explicit, nothing ``auto``."""

    requested_method: str
    method: str
    label: str
    graph_kind: GraphKind
    scope: CurvatureScope
    direction_convention: str
    parameters: Mapping[str, object]
    semantics: GraphSemantics
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _freeze(self, "parameters", FrozenMapping(self.parameters))
        _freeze(self, "warnings", tuple(self.warnings))

    def to_dict(self) -> dict:
        return {
            "requested_method": self.requested_method,
            "method": self.method,
            "label": self.label,
            "graph_kind": self.graph_kind,
            "scope": self.scope,
            "direction_convention": self.direction_convention,
            "parameters": json_safe(dict(self.parameters)),
            "semantics": self.semantics.to_dict(),
            "warnings": list(self.warnings),
        }


def _node_key(node: Any) -> Any:
    return node if isinstance(node, (str, int, float, bool)) or node is None else repr(node)


@dataclass(frozen=True)
class CurvatureResult:
    """Values of one successful evaluation, with the record that produced them.

    ``incident_node_means`` is a *summary* of the edge values -- the mean over
    each node's incident edges -- not a separately defined node curvature. It is
    empty for a pair evaluation.
    """

    values: Mapping[tuple[object, object], float]
    incident_node_means: Mapping[object, float]
    resolved: ResolvedCurvature | None

    def __post_init__(self) -> None:
        _freeze(self, "values", FrozenMapping(self.values))
        _freeze(self, "incident_node_means", FrozenMapping(self.incident_node_means))

    def rows(self) -> list[dict]:
        """``[{"source", "target", "kappa"}, ...]`` in result order (a value table)."""
        return [{"source": u, "target": v, "kappa": value} for (u, v), value in self.values.items()]

    def to_dict(self) -> dict:
        return {
            "values": [
                [_node_key(u), _node_key(v), value] for (u, v), value in self.values.items()
            ],
            "incident_node_means": [
                [_node_key(n), value] for n, value in self.incident_node_means.items()
            ],
            "resolved": self.resolved.to_dict() if self.resolved else None,
        }


@dataclass(frozen=True)
class CurvatureFailure:
    """A recorded failure: its kind, the exception, and how far resolution got."""

    kind: FailureKind
    exception_type: str
    message: str
    resolved: ResolvedCurvature | None = None

    def __post_init__(self) -> None:
        if self.kind not in FAILURE_KINDS:
            raise CurvatureConfigurationError(
                f"failure kind must be one of {list(FAILURE_KINDS)}; got {self.kind!r}."
            )

    def __str__(self) -> str:
        return f"{self.exception_type}: {self.message}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "exception_type": self.exception_type,
            "message": self.message,
            "resolved": self.resolved.to_dict() if self.resolved else None,
        }


# ── callable protocols ──────────────────────────────────────────────────────


class EdgeCompute(Protocol):
    """Raw edge values; must not annotate ``G``."""

    def __call__(
        self,
        G: nx.Graph,
        *,
        semantics: GraphSemantics,
        proc: int,
        **parameters: object,
    ) -> Mapping[tuple[object, object], float]: ...


class PairCompute(Protocol):
    """Raw values on requested ordered pairs; must not annotate ``G``."""

    def __call__(
        self,
        G: nx.Graph,
        pairs: tuple[tuple[object, object], ...],
        *,
        semantics: GraphSemantics,
        proc: int,
        **parameters: object,
    ) -> Mapping[tuple[object, object], float]: ...


class CurvatureResolver(Protocol):
    """Returns effective parameters, direction convention, warnings."""

    def __call__(
        self,
        *,
        graph_kind: GraphKind,
        scope: CurvatureScope,
        parameters: Mapping[str, object],
    ) -> tuple[Mapping[str, object], str, tuple[str, ...]]: ...
