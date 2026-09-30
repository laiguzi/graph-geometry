"""The curvature registry: canonical specifications, aliases and queries.

The registry's unit is one **mathematical curvature definition**. Graph kind,
direction convention, attribute roles, parameters and solver are configuration
of that definition, never additional entries. This module stores and queries
specifications only; it performs no computation and contains no formula. The
shipped definitions are registered in :mod:`~graph_geometry.curvature.builtins`
and evaluated by :mod:`~graph_geometry.curvature.evaluator`.

Canonical names and aliases live in separate maps, so a deprecated alias is
never listed as a second curvature.
"""

from __future__ import annotations

import inspect
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx

from .errors import CurvatureConfigurationError, CurvatureDeprecationWarning
from .model import (
    CURVATURE_STATUSES,
    CurvatureCapabilities,
    CurvatureResolver,
    EdgeCompute,
    PairCompute,
    ParameterSpec,
)

__all__ = [
    "CurvatureSpec",
    "CurvatureAlias",
    "CURVATURE_REGISTRY",
    "CURVATURE_ALIASES",
    "CURVATURE_REMOVED",
    "PAIR_CAPABLE",
    "register_curvature",
    "register_curvature_alias",
    "register_removed_curvature",
    "unregister_curvature",
    "get_curvature_spec",
    "lookup_curvature",
    "list_curvatures",
]


@dataclass(frozen=True)
class CurvatureSpec:
    """A registered curvature definition and its complete metadata."""

    name: str
    compute_edges: EdgeCompute
    capabilities: CurvatureCapabilities
    parameters: tuple[ParameterSpec, ...] = ()
    compute_pairs: PairCompute | None = None
    resolver: CurvatureResolver | None = None
    label: str = ""
    color: str = ""
    description: str = ""
    citation: str = ""
    metadata_complete: bool = True

    @property
    def compute(self) -> Callable:
        """Deprecated name of :attr:`compute_edges`."""
        return self.compute_edges

    @property
    def status(self) -> str:
        return self.capabilities.status

    def parameter(self, name: str) -> ParameterSpec | None:
        return next((p for p in self.parameters if p.name == name), None)

    def to_dict(self) -> dict:
        """JSON-safe metadata (the callables are omitted)."""
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "citation": self.citation,
            "metadata_complete": self.metadata_complete,
            "capabilities": self.capabilities.to_dict(),
            "parameters": [p.to_dict() for p in self.parameters],
        }


@dataclass(frozen=True)
class CurvatureAlias:
    """A compatibility name resolving to a canonical definition."""

    name: str
    target: str
    deprecated: bool = True
    message: str = ""


#: Canonical definitions, in registration order.
CURVATURE_REGISTRY: "OrderedDict[str, CurvatureSpec]" = OrderedDict()

#: Compatibility names -> canonical names.
CURVATURE_ALIASES: "OrderedDict[str, CurvatureAlias]" = OrderedDict()

#: Deprecated snapshot of the pair-capable built-ins. Internal code queries
#: ``list_curvatures(scope="pairs")`` instead; this constant is kept only for
#: one compatibility cycle and does not track plugins.
PAIR_CAPABLE = ("lin_lu_yau", "ollivier")


# ── legacy plugin support ───────────────────────────────────────────────────

_LEGACY_HIDDEN = {"G", "weight", "distance", "proc"}


class LegacyEdgeCompute:
    """Adapt a pre-metadata ``compute(G, *, weight, distance, proc, **kw)``.

    The legacy callable may annotate the graph it receives, so it runs on a
    copy: the evaluator guarantees ``G`` is not mutated unless annotation was
    requested.
    """

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", type(fn).__name__)
        self.__doc__ = getattr(fn, "__doc__", None)

    def __call__(self, G: nx.Graph, *, semantics, proc: int, **parameters: Any):
        return self.fn(
            G.copy(),
            weight=semantics.weight_attr,
            distance=semantics.distance_attr,
            proc=proc,
            **parameters,
        )


def _infer_legacy_parameters(fn: Callable[..., Any]) -> tuple[ParameterSpec, ...]:
    """Parameter schema read once, at registration, from a legacy signature.

    Only keyword parameters with a boolean, integer, float or string default
    become parameters; the UI and validator then read the schema, never the
    signature.
    """
    specs = []
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return ()
    for name, p in signature.parameters.items():
        if name in _LEGACY_HIDDEN or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
            continue
        default = p.default
        if default is inspect.Parameter.empty:
            continue
        if isinstance(default, bool):
            kind = "boolean"
        elif isinstance(default, int):
            kind = "integer"
        elif isinstance(default, float):
            kind = "float"
        elif isinstance(default, str):
            kind = "string"
        else:
            continue
        specs.append(ParameterSpec(name=name, kind=kind, default=default))
    return tuple(specs)


_LEGACY_CAPABILITIES = CurvatureCapabilities(
    family="combinatorial",
    status="experimental",
    graph_kinds=frozenset({"directed", "undirected"}),
    scopes=frozenset({"edges"}),
    # Unknown, so assume everything is read: the flow's invariance warning must
    # not fire falsely for a plugin that never declared what it consumes.
    consumes=frozenset({"topology", "edge_weight", "edge_distance"}),
)


# ── registration ────────────────────────────────────────────────────────────


def _check_name(name: Any) -> str:
    if not isinstance(name, str) or not name:
        raise CurvatureConfigurationError(
            f"curvature name must be a non-empty string; got {name!r}."
        )
    if name in CURVATURE_REGISTRY or name in CURVATURE_ALIASES:
        raise CurvatureConfigurationError(
            f"curvature name {name!r} is already registered. Unregister it first "
            f"with graph_geometry.curvature.unregister_curvature({name!r})."
        )
    return name


def _check_spec(spec: CurvatureSpec) -> None:
    caps = spec.capabilities
    if "pairs" in caps.scopes and spec.compute_pairs is None:
        raise CurvatureConfigurationError(
            f"{spec.name!r} advertises the 'pairs' scope but has no compute_pairs."
        )
    if spec.compute_pairs is not None and "pairs" not in caps.scopes:
        raise CurvatureConfigurationError(
            f"{spec.name!r} supplies compute_pairs without the 'pairs' scope."
        )
    names = [p.name for p in spec.parameters]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise CurvatureConfigurationError(
            f"{spec.name!r} declares parameter(s) {duplicated} more than once."
        )
    reserved = sorted(set(names) & {"G", "semantics", "proc", "pairs"})
    if reserved:
        raise CurvatureConfigurationError(
            f"{spec.name!r}: {reserved} are execution/semantics fields, not "
            f"mathematical parameters."
        )
    for p in spec.parameters:
        unknown = sorted(set(p.active_when) - set(names))
        if unknown:
            raise CurvatureConfigurationError(
                f"{spec.name!r}: parameter {p.name!r} is activated by undeclared "
                f"parameter(s) {unknown}."
            )


def register_curvature(
    name: str,
    *,
    capabilities: CurvatureCapabilities | None = None,
    parameters: tuple[ParameterSpec, ...] = (),
    compute_pairs: PairCompute | None = None,
    resolver: CurvatureResolver | None = None,
    label: str = "",
    color: str = "",
    description: str = "",
    citation: str = "",
) -> Callable[[EdgeCompute], EdgeCompute]:
    """Decorator registering an edge-compute callable under ``name``.

    A definition with ``capabilities`` uses the metadata contract::

        @register_curvature(
            "my_kappa",
            capabilities=CurvatureCapabilities(
                family="combinatorial", status="experimental",
                graph_kinds=frozenset({"undirected"}),
                scopes=frozenset({"edges"}),
                consumes=frozenset({"topology", "edge_weight"}),
            ),
            parameters=(ParameterSpec("scale", "float", 1.0, minimum=0.0),),
        )
        def my_kappa(G, *, semantics, proc, scale):
            return {(u, v): scale for u, v in G.edges()}

    It returns raw values and does not annotate ``G``.

    **Legacy form.** Omitting ``capabilities`` registers a pre-metadata
    ``compute(G, *, weight, distance, proc, **params)``. It stays executable for
    one deprecation cycle, is marked ``experimental`` with incomplete metadata,
    has its parameter schema read once from its signature, and emits a
    :class:`~graph_geometry.curvature.errors.CurvatureDeprecationWarning`.
    """
    _check_name(name)

    def decorator(fn: EdgeCompute) -> EdgeCompute:
        _check_name(name)
        summary = description or (getattr(fn, "__doc__", None) or "").strip().split("\n")[0]
        if capabilities is None:
            warnings.warn(
                f"curvature {name!r} was registered without capabilities; it is "
                f"treated as an experimental definition with incomplete metadata. "
                f"Declare capabilities= and parameters= (see "
                f"examples/plugin_balanced_forman.py).",
                CurvatureDeprecationWarning,
                stacklevel=2,
            )
            spec = CurvatureSpec(
                name=name,
                compute_edges=LegacyEdgeCompute(fn),
                capabilities=_LEGACY_CAPABILITIES,
                parameters=tuple(parameters) or _infer_legacy_parameters(fn),
                compute_pairs=None,
                resolver=resolver,
                label=label or name,
                color=color,
                description=summary,
                citation=citation,
                metadata_complete=False,
            )
        else:
            if not isinstance(capabilities, CurvatureCapabilities):
                raise CurvatureConfigurationError(
                    f"capabilities must be a CurvatureCapabilities; got "
                    f"{type(capabilities).__name__}."
                )
            spec = CurvatureSpec(
                name=name,
                compute_edges=fn,
                capabilities=capabilities,
                parameters=tuple(parameters),
                compute_pairs=compute_pairs,
                resolver=resolver,
                label=label or name,
                color=color,
                description=summary,
                citation=citation,
                metadata_complete=bool(citation and summary),
            )
        _check_spec(spec)
        CURVATURE_REGISTRY[name] = spec
        return fn

    return decorator


def register_curvature_alias(name: str, target: str, *, message: str) -> None:
    """Register a deprecated compatibility name for a canonical definition."""
    _check_name(name)
    if target not in CURVATURE_REGISTRY:
        raise CurvatureConfigurationError(
            f"alias {name!r} points at {target!r}, which is not a registered "
            f"canonical curvature."
        )
    CURVATURE_ALIASES[name] = CurvatureAlias(
        name=name, target=target, deprecated=True, message=message
    )


#: Names that once resolved to a definition and have been withdrawn, mapped to
#: the message raised in their place. A withdrawn name is never quietly pointed
#: at another definition: the same call would then return different numbers
#: without saying so.
CURVATURE_REMOVED: dict[str, str] = {}


def register_removed_curvature(name: str, *, message: str) -> None:
    """Record ``name`` as withdrawn, so using it fails with ``message``."""
    if name in CURVATURE_REGISTRY or name in CURVATURE_ALIASES:
        raise CurvatureConfigurationError(
            f"{name!r} is still registered; unregister it before recording it as "
            f"removed."
        )
    CURVATURE_REMOVED[name] = message


def unregister_curvature(name: str) -> None:
    """Remove a canonical definition (and aliases pointing at it) or an alias."""
    if name in CURVATURE_ALIASES:
        del CURVATURE_ALIASES[name]
        return
    CURVATURE_REGISTRY.pop(name, None)
    for alias in [a for a, rec in CURVATURE_ALIASES.items() if rec.target == name]:
        del CURVATURE_ALIASES[alias]


# ── queries ─────────────────────────────────────────────────────────────────


def lookup_curvature(name: str) -> tuple[CurvatureSpec, CurvatureAlias | None]:
    """Resolve ``name`` (canonical or alias) to its spec and the alias used.

    Emits no warning; the evaluator warns when an alias is used for a run.
    """
    if not isinstance(name, str):
        raise CurvatureConfigurationError(f"curvature name must be a string; got {name!r}.")
    for key in (name, name.lower()):
        if key in CURVATURE_REGISTRY:
            return CURVATURE_REGISTRY[key], None
        if key in CURVATURE_ALIASES:
            alias = CURVATURE_ALIASES[key]
            if alias.target not in CURVATURE_REGISTRY:
                raise CurvatureConfigurationError(
                    f"alias {key!r} points at {alias.target!r}, which is no longer "
                    f"registered."
                )
            return CURVATURE_REGISTRY[alias.target], alias
    for key in (name, name.lower()):
        if key in CURVATURE_REMOVED:
            raise CurvatureConfigurationError(CURVATURE_REMOVED[key])
    raise CurvatureConfigurationError(
        f"Unknown curvature method {name!r}. Registered: {list(CURVATURE_REGISTRY)}"
        + (f"; aliases: {list(CURVATURE_ALIASES)}." if CURVATURE_ALIASES else ".")
    )


def get_curvature_spec(name: str) -> CurvatureSpec:
    """The canonical spec for ``name`` (aliases resolve silently)."""
    return lookup_curvature(name)[0]


def list_curvatures(
    *,
    graph_kind: str | None = None,
    scope: str | None = None,
    statuses: frozenset[str] | None = None,
) -> tuple[CurvatureSpec, ...]:
    """Canonical specs, filtered by capability; aliases are never listed."""
    if statuses is not None:
        unknown = sorted(set(statuses) - set(CURVATURE_STATUSES))
        if unknown:
            raise CurvatureConfigurationError(
                f"Unknown status(es) {unknown}; allowed: {list(CURVATURE_STATUSES)}."
            )
    out = []
    for spec in CURVATURE_REGISTRY.values():
        caps = spec.capabilities
        if graph_kind is not None and graph_kind not in caps.graph_kinds:
            continue
        if scope is not None and scope not in caps.scopes:
            continue
        if statuses is not None and caps.status not in statuses:
            continue
        out.append(spec)
    return tuple(out)
