"""Resolution, validation and the result contract.

Every curvature evaluation goes through two steps:

1. :func:`resolve_curvature` turns a request into a
   :class:`~graph_geometry.curvature.model.ResolvedCurvature`: aliases resolve
   to a canonical name, the graph kind is read from the graph and checked
   against the definition's capabilities, the scope is checked, one set of
   defaults is merged from the parameter schema, unknown / inactive /
   out-of-range / wrong-type parameters are rejected, and ``kernel="auto"`` is
   replaced by the convention it stands for. Nothing numerical happens here.
2. :func:`evaluate_curvature` runs the definition under that record: it rejects
   multigraphs and self-loops, validates the semantic channels the definition
   consumes, calls the formula, and enforces exactly one finite real per edge
   (or per requested pair). It never mutates the graph unless annotation was
   requested.

:func:`compute_curvature` and :func:`compute_pair_curvature` are the two steps
together. :func:`curvature` and :func:`curvature_pairs` are the compatibility
wrappers with the historical call forms.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import networkx as nx

from ..graph.semantics import GraphSemantics
from .base import CurvatureConfig, RicciCurvature, incident_means, write_curvature
from .errors import (
    CurvatureConfigurationError,
    CurvatureContractError,
    CurvatureDeprecationWarning,
    CurvatureDomainError,
    CurvatureInputError,
    CurvatureWarning,
)
from .model import (
    CURVATURE_SCOPES,
    CurvatureRequest,
    CurvatureResult,
    FrozenMapping,
    ResolvedCurvature,
    graph_kind_of,
)
from .registry import get_curvature_spec, list_curvatures, lookup_curvature
from .validate import validate_edge_attributes, validate_simple_graph

__all__ = [
    "resolve_curvature",
    "evaluate_curvature",
    "compute_curvature",
    "compute_pair_curvature",
    "curvature",
    "curvature_pairs",
    "incident_node_means",
]

ANNOTATION_ATTR = "ricciCurvature"


# ── resolution ──────────────────────────────────────────────────────────────


def _coerce_request(request: Any) -> CurvatureRequest:
    if isinstance(request, CurvatureRequest):
        return request
    if isinstance(request, str):
        return CurvatureRequest(request)
    raise CurvatureConfigurationError(
        f"request must be a method name or a CurvatureRequest; got {request!r}."
    )


def _coerce_semantics(semantics: Any) -> GraphSemantics:
    if semantics is None:
        return GraphSemantics()
    if isinstance(semantics, GraphSemantics):
        return semantics
    raise CurvatureConfigurationError(
        f"semantics must be a GraphSemantics or None; got {type(semantics).__name__}."
    )


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


def _check_graph(G: Any) -> None:
    if not isinstance(G, nx.Graph):
        raise TypeError(f"expected a NetworkX graph; got {type(G).__name__}.")


def _resolve(
    G: nx.Graph,
    request: Any,
    semantics: Any = None,
    scope: str = "edges",
    *,
    legacy: bool = False,
    emit: bool = True,
) -> ResolvedCurvature:
    """Shared resolution; ``legacy`` admits the compatibility value forms."""
    _check_graph(G)
    request = _coerce_request(request)
    semantics = _coerce_semantics(semantics)
    if scope not in CURVATURE_SCOPES:
        raise CurvatureConfigurationError(
            f"scope must be one of {list(CURVATURE_SCOPES)}; got {scope!r}."
        )

    spec, alias = lookup_curvature(request.method)
    notes: list[tuple[str, type[Warning]]] = []
    if alias is not None and alias.deprecated:
        notes.append((alias.message or f"{alias.name!r} is deprecated; use "
                      f"{alias.target!r}.", CurvatureDeprecationWarning))
    if not spec.metadata_complete and spec.capabilities.status == "experimental":
        notes.append((f"{spec.name!r} is experimental and its metadata is "
                      f"incomplete.", CurvatureWarning))

    caps = spec.capabilities
    graph_kind = graph_kind_of(G)
    if graph_kind not in caps.graph_kinds:
        supported = " or ".join(sorted(caps.graph_kinds))
        conversion = "G.to_directed()" if graph_kind == "undirected" else "G.to_undirected()"
        raise CurvatureDomainError(
            f"{spec.name!r} is defined for {supported} graphs only; got "
            f"{_article(graph_kind)} {graph_kind} graph. No conversion is applied implicitly: convert "
            f"explicitly (e.g. {conversion}) so the choice is recorded, or "
            f"choose a definition supporting {graph_kind} graphs: "
            f"{[s.name for s in list_curvatures(graph_kind=graph_kind)]}."
        )
    if scope not in caps.scopes:
        pair_capable = [s.name for s in list_curvatures(scope="pairs")]
        raise CurvatureDomainError(
            f"{request.method!r} is defined on edges only, so it has no value on "
            f"an arbitrary node pair: its construction is assembled from the "
            f"edge itself (for a combinatorial curvature, the stars of its "
            f"endpoints and the faces containing it; for Eidi-Jost, in-out "
            f"measures joined through the edge). Pair-capable methods: "
            f"{pair_capable}."
        )

    supplied = dict(request.parameters)
    declared = {p.name: p for p in spec.parameters}
    unknown = sorted(set(supplied) - set(declared))
    if unknown:
        raise CurvatureConfigurationError(
            f"{spec.name!r} has no parameter(s) {unknown}; declared: "
            f"{sorted(declared) or 'none'}."
        )

    merged: dict = {}
    legacy_forms: list[str] = []
    for p in spec.parameters:
        if p.name not in supplied:
            merged[p.name] = p.default
            continue
        try:
            merged[p.name] = p.validate(supplied[p.name])
        except CurvatureConfigurationError:
            if not (legacy and p.legacy_validator is not None):
                raise
            merged[p.name] = p.legacy_validator(supplied[p.name])
            legacy_forms.append(p.name)

    if spec.resolver is not None:
        effective, convention, resolver_notes = spec.resolver(
            graph_kind=graph_kind, scope=scope, parameters=FrozenMapping(merged)
        )
        effective = dict(effective)
        notes.extend((note, CurvatureWarning) for note in resolver_notes)
    else:
        effective, convention = merged, graph_kind
    for name in legacy_forms:
        if effective.get(name) is merged[name]:
            # still the Python object itself: not translated into the schema
            notes.append((
                f"parameter {name!r}={supplied[name]!r} is a Python-only "
                f"compatibility form outside the declarative schema; it cannot be "
                f"persisted in an experiment.", CurvatureDeprecationWarning,
            ))
    if set(effective) != set(merged):
        raise CurvatureContractError(
            f"the resolver of {spec.name!r} changed the parameter set "
            f"({sorted(merged)} -> {sorted(effective)})."
        )
    left_auto = sorted(
        name for name, value in effective.items() if isinstance(value, str) and value == "auto"
    )
    if left_auto:
        if spec.metadata_complete or spec.resolver is not None:
            raise CurvatureContractError(
                f"the resolver of {spec.name!r} left {left_auto} as 'auto'; a resolved "
                f"record must name the actual convention."
            )
        # A legacy plugin declares no resolver, so nothing can say what its
        # 'auto' means; record that instead of guessing.
        notes.append((
            f"{spec.name!r} is a legacy plugin without a resolver; {left_auto} "
            f"remain 'auto' and the convention it applied is not recorded.",
            CurvatureWarning,
        ))

    for p in spec.parameters:
        if p.is_active(graph_kind, effective, semantics):
            continue
        reason = _inactive_reason(p, graph_kind, effective, semantics)
        translated = p.name in supplied and legacy and (
            type(supplied[p.name]) is not type(effective[p.name])
            or supplied[p.name] != effective[p.name]
        )
        if translated:
            # The resolver rewrote a compatibility spelling into another
            # parameter (e.g. beta='degree_proportional' -> beta_strategy) and
            # said so; the placeholder it left is dropped without a second note.
            del effective[p.name]
            continue
        if p.name in supplied:
            if not legacy:
                raise CurvatureConfigurationError(
                    f"parameter {p.name!r} of {spec.name!r} is inactive here "
                    f"({reason}); remove it so the recorded configuration is the "
                    f"one that ran."
                )
            # The historical call forms silently ignored such values; they are
            # still ignored, but no longer silently, and never recorded.
            notes.append((
                f"parameter {p.name!r}={supplied[p.name]!r} of {spec.name!r} is "
                f"inactive here ({reason}) and was ignored; the strict API rejects it.",
                CurvatureDeprecationWarning,
            ))
        del effective[p.name]

    if emit:
        for message, category in notes:
            warnings.warn(message, category, stacklevel=4)

    return ResolvedCurvature(
        requested_method=request.method,
        method=spec.name,
        label=request.display_label,
        graph_kind=graph_kind,
        scope=scope,
        direction_convention=convention,
        parameters=effective,
        semantics=semantics,
        warnings=tuple(message for message, _ in notes),
    )


def _inactive_reason(p, graph_kind, parameters, semantics) -> str:
    if graph_kind not in p.graph_kinds:
        return f"it applies to {' or '.join(sorted(p.graph_kinds))} graphs only"
    for other, allowed in p.active_when.items():
        if parameters.get(other) not in allowed:
            return f"it requires {other} in {list(allowed)}, got {parameters.get(other)!r}"
    for field in p.inactive_when_semantics:
        if getattr(semantics, field) is not None:
            return f"GraphSemantics.{field} is set"
    return "inactive"


def resolve_curvature(
    G: nx.Graph,
    request: str | CurvatureRequest,
    *,
    semantics: GraphSemantics | None = None,
    scope: str = "edges",
) -> ResolvedCurvature:
    """Resolve ``request`` against ``G`` without any numerical work."""
    return _resolve(G, request, semantics, scope)


# ── validation and the result contract ──────────────────────────────────────


def _check_proc(proc: Any) -> int:
    if isinstance(proc, bool) or not isinstance(proc, int) or proc < 1:
        raise CurvatureConfigurationError(f"proc must be a positive integer; got {proc!r}.")
    return proc


def _validate_channels(G: nx.Graph, consumes, semantics: GraphSemantics, where: str) -> None:
    """Validate only the semantic channels the definition consumes."""
    weight = semantics.weight_attr if "edge_weight" in consumes else None
    distance = semantics.distance_attr if "edge_distance" in consumes else None
    for attr, policy, role in (
        (weight, semantics.missing_weight, "weight"),
        (distance, semantics.missing_distance, "distance"),
    ):
        if attr is None or policy != "error":
            continue
        missing = [(u, v) for u, v, data in G.edges(data=True) if attr not in data]
        if missing:
            raise CurvatureInputError(
                f"{where}: {len(missing)} edge(s) have no {role} attribute {attr!r} "
                f"(e.g. {missing[0]!r}) and missing_{role}='error'."
            )
    if weight is not None or distance is not None:
        # Finiteness for both; positivity for the metric. Positivity of weights
        # is definition-specific and stays with the formula.
        validate_edge_attributes(
            G, weight=weight, distance=distance, where=where, positive_weight=False
        )
    if "node_weight" in consumes and semantics.node_weight_attr is not None:
        attr = semantics.node_weight_attr
        for n, data in G.nodes(data=True):
            if attr not in data:
                raise CurvatureInputError(
                    f"{where}: node {n!r} has no node weight attribute {attr!r}; "
                    f"with node_weight_attr set, every node must carry a value."
                )
            value = data[attr]
            if isinstance(value, bool) or not isinstance(value, Real):
                raise CurvatureInputError(
                    f"{where}: node {n!r} has non-numeric {attr}={value!r}."
                )
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise CurvatureInputError(
                    f"{where}: node {n!r} has {attr}={value!r}; node weights must "
                    f"be finite and strictly positive."
                )


def _finite_real(value: Any, key: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise CurvatureContractError(
            f"{where} returned non-numeric curvature {value!r} for {key!r}; every "
            f"value must be a finite real number."
        )
    value = float(value)
    if not math.isfinite(value):
        raise CurvatureContractError(
            f"{where} returned non-finite curvature {value!r} for {key!r}."
        )
    return value


def _edge_contract(G: nx.Graph, result: Any, where: str) -> dict:
    """Exactly one finite real per graph edge, keyed in ``G.edges()`` orientation."""
    if not isinstance(result, Mapping):
        raise CurvatureContractError(
            f"{where} must return a mapping from graph edges to finite real "
            f"curvature values; got {type(result).__name__}."
        )
    remaining = dict(result)
    values: dict = {}
    missing: list = []
    for u, v in G.edges():
        edge = (u, v)
        if edge in remaining:
            value = remaining.pop(edge)
        elif not G.is_directed() and (v, u) in remaining:
            value = remaining.pop((v, u))
        else:
            missing.append(edge)
            continue
        values[edge] = _finite_real(value, edge, where)
    if missing or remaining:
        details = []
        if missing:
            details.append(f"missing {missing[:3]!r}")
        if remaining:
            details.append(f"unexpected {list(remaining)[:3]!r}")
        raise CurvatureContractError(
            f"{where} result must cover exactly the graph's {G.number_of_edges()} "
            f"edges ({'; '.join(details)})."
        )
    return values


def _pair_contract(pairs: tuple, result: Any, where: str) -> dict:
    """Exactly one finite real per requested pair, in the requested orientation."""
    if not isinstance(result, Mapping):
        raise CurvatureContractError(
            f"{where} must return a mapping from the requested pairs to finite "
            f"real curvature values; got {type(result).__name__}."
        )
    remaining = dict(result)
    values: dict = {}
    missing: list = []
    for pair in pairs:
        if pair in remaining:
            values[pair] = _finite_real(remaining.pop(pair), pair, where)
        else:
            missing.append(pair)
    if missing or remaining:
        details = []
        if missing:
            details.append(f"missing {missing[:3]!r}")
        if remaining:
            details.append(f"unexpected {list(remaining)[:3]!r}")
        raise CurvatureContractError(
            f"{where} result must cover exactly the {len(pairs)} requested pairs "
            f"({'; '.join(details)})."
        )
    return values


def incident_node_means(G: nx.Graph, values: Mapping) -> dict:
    """Mean of each node's incident edge values; nodes without edges are omitted.

    A summary of the edge result, not a separately defined node curvature. The
    same implementation as :func:`~graph_geometry.curvature.base.write_node_means`
    (undirected results may be keyed in either orientation).
    """
    if G.is_directed():
        return incident_means(G, lambda u, v: values.get((u, v)))
    return incident_means(
        G, lambda u, v: values[(u, v)] if (u, v) in values else values.get((v, u))
    )


def _annotate(G: nx.Graph, values: Mapping, means: Mapping, attr: str) -> None:
    for (u, v), value in values.items():
        G[u][v][attr] = value
    # Recalculation may follow surgery: clear values on nodes that became isolated.
    for n in G.nodes():
        G.nodes[n].pop(attr, None)
    for n, value in means.items():
        G.nodes[n][attr] = value


def _materialise_pairs(G: nx.Graph, pairs: Iterable) -> tuple:
    out = []
    seen = set()
    for pair in pairs:
        try:
            x, y = pair
        except (TypeError, ValueError):
            raise CurvatureConfigurationError(
                f"each pair must be a 2-tuple of nodes; got {pair!r}."
            ) from None
        for node in (x, y):
            if node not in G:
                raise CurvatureConfigurationError(f"node {node!r} is not in the graph.")
        if x == y:
            raise CurvatureConfigurationError(
                f"curvature needs two distinct nodes, got ({x!r}, {x!r}); kappa is "
                f"defined through d(x, y) > 0."
            )
        if (x, y) in seen:
            raise CurvatureConfigurationError(f"pair {(x, y)!r} is requested twice.")
        seen.add((x, y))
        out.append((x, y))
    return tuple(out)


def evaluate_curvature(
    G: nx.Graph,
    resolved: ResolvedCurvature,
    *,
    proc: int = 1,
    annotate: bool = False,
    annotation_attr: str = ANNOTATION_ATTR,
    pairs: Iterable | None = None,
) -> CurvatureResult:
    """Evaluate an already resolved configuration on ``G``.

    Static resolution happens once; this re-checks what can change between
    evaluations -- graph kind, loops, attribute values, reachability -- so a flow
    can re-evaluate one record after every topology change.
    """
    _check_graph(G)
    if not isinstance(resolved, ResolvedCurvature):
        raise CurvatureConfigurationError(
            f"resolved must be a ResolvedCurvature; got {type(resolved).__name__}."
        )
    proc = _check_proc(proc)
    spec = get_curvature_spec(resolved.method)
    where = resolved.method
    if graph_kind_of(G) != resolved.graph_kind:
        raise CurvatureDomainError(
            f"{where} was resolved for {_article(resolved.graph_kind)} "
            f"{resolved.graph_kind} graph; got {_article(graph_kind_of(G))} "
            f"{graph_kind_of(G)} graph."
        )
    validate_simple_graph(G, where)
    _validate_channels(G, spec.capabilities.consumes, resolved.semantics, where)

    parameters = dict(resolved.parameters)
    if resolved.scope == "edges":
        if pairs is not None:
            raise CurvatureConfigurationError("pairs are given but the scope is 'edges'.")
        raw = spec.compute_edges(G, semantics=resolved.semantics, proc=proc, **parameters)
        values = _edge_contract(G, raw, where)
        means = incident_node_means(G, values)
        if annotate:
            _annotate(G, values, means, annotation_attr)
        return CurvatureResult(values=values, incident_node_means=means, resolved=resolved)

    if pairs is None:
        raise CurvatureConfigurationError("scope 'pairs' requires the pairs to evaluate.")
    if annotate:
        raise CurvatureConfigurationError(
            "a pair result cannot annotate the graph: a non-adjacent pair has no "
            "edge to carry the attribute."
        )
    pairs = _materialise_pairs(G, pairs)
    raw = spec.compute_pairs(G, pairs, semantics=resolved.semantics, proc=proc, **parameters)
    values = _pair_contract(pairs, raw, where)
    return CurvatureResult(values=values, incident_node_means={}, resolved=resolved)


def compute_curvature(
    G: nx.Graph,
    request: str | CurvatureRequest = "lin_lu_yau",
    *,
    semantics: GraphSemantics | None = None,
    proc: int = 1,
    annotate: bool = False,
    annotation_attr: str = ANNOTATION_ATTR,
) -> CurvatureResult:
    """Resolve and evaluate ``request`` on every edge of ``G``.

    ``G`` is not modified unless ``annotate=True``, which writes the edge values
    and the incident node means to ``annotation_attr``.
    """
    resolved = _resolve(G, request, semantics, "edges")
    return evaluate_curvature(
        G, resolved, proc=proc, annotate=annotate, annotation_attr=annotation_attr
    )


def compute_pair_curvature(
    G: nx.Graph,
    request: str | CurvatureRequest,
    pairs: Iterable[tuple[object, object]],
    *,
    semantics: GraphSemantics | None = None,
    proc: int = 1,
) -> CurvatureResult:
    """Resolve and evaluate ``request`` on the given node pairs; writes nothing.

    ``pairs`` is materialised once; results keep the requested orientation.
    """
    resolved = _resolve(G, request, semantics, "pairs")
    return evaluate_curvature(G, resolved, proc=proc, pairs=pairs)


# ── compatibility wrappers ──────────────────────────────────────────────────


def _legacy_semantics(weight: str, distance: str, params: dict) -> GraphSemantics:
    return GraphSemantics(
        weight_attr=weight,
        distance_attr=distance,
        node_weight_attr=params.pop("node_weight_attr", None),
    )


def curvature(
    G: nx.Graph,
    method: Any = "lin_lu_yau",
    *,
    weight: str = "weight",
    distance: str = "distance",
    proc: int = 1,
    **params: Any,
) -> dict:
    """Compatibility wrapper: compute, annotate ``G``, return ``{edge: kappa}``.

    ``method`` may be a registered name or alias (``**params`` become the
    request parameters; ``node_weight_attr`` maps to
    :class:`~graph_geometry.graph.semantics.GraphSemantics`), a
    :class:`CurvatureRequest`, a :class:`CurvatureConfig` run directly through
    the transport engine, or a callable with the legacy compute signature.
    New code should call :func:`compute_curvature`.
    """
    _check_graph(G)
    if isinstance(method, (str, CurvatureRequest)):
        if isinstance(method, CurvatureRequest) and params:
            raise CurvatureConfigurationError(
                "pass parameters inside the CurvatureRequest, not as keywords."
            )
        semantics = _legacy_semantics(weight, distance, params)
        request = method if isinstance(method, CurvatureRequest) else CurvatureRequest(method, params)
        resolved = _resolve(G, request, semantics, "edges", legacy=True)
        return dict(evaluate_curvature(G, resolved, proc=proc, annotate=True).values)

    if isinstance(method, CurvatureConfig):
        _check_proc(proc)
        validate_simple_graph(G, method.name)
        raw = RicciCurvature(G, method, weight, distance, proc).compute_edge_values()
        return write_curvature(G, _edge_contract(G, raw, method.name))

    if callable(method):
        where = getattr(method, "__name__", type(method).__name__)
        validate_simple_graph(G, where)
        raw = method(G, weight=weight, distance=distance, proc=proc, **params)
        return write_curvature(G, _edge_contract(G, raw, where))

    raise TypeError(
        f"method must be a name, CurvatureRequest, CurvatureConfig, or callable; "
        f"got {method!r}"
    )


def _all_pairs(G: nx.Graph) -> list:
    nodes = list(G.nodes())
    if G.is_directed():
        return [(x, y) for x in nodes for y in nodes if x != y]
    return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]


def curvature_pairs(
    G: nx.Graph,
    method: Any = "lin_lu_yau",
    *,
    pairs=None,
    weight: str = "weight",
    distance: str = "distance",
    proc: int = 1,
    **params: Any,
) -> dict:
    """Compatibility wrapper for pair curvature -- ``{(u, v): kappa}``, writes nothing.

    ``pairs`` defaults to every distinct pair: unordered on an undirected graph,
    ordered on a digraph. Accepts pair-capable names (see
    ``list_curvatures(scope="pairs")``), a :class:`CurvatureRequest`, or a
    :class:`CurvatureConfig`. New code should call :func:`compute_pair_curvature`.
    """
    _check_graph(G)
    if pairs is None:
        pairs = _all_pairs(G)

    if isinstance(method, (str, CurvatureRequest)):
        if isinstance(method, CurvatureRequest) and params:
            raise CurvatureConfigurationError(
                "pass parameters inside the CurvatureRequest, not as keywords."
            )
        semantics = _legacy_semantics(weight, distance, params)
        request = method if isinstance(method, CurvatureRequest) else CurvatureRequest(method, params)
        resolved = _resolve(G, request, semantics, "pairs", legacy=True)
        return dict(evaluate_curvature(G, resolved, proc=proc, pairs=pairs).values)

    if isinstance(method, CurvatureConfig):
        _check_proc(proc)
        pairs = _materialise_pairs(G, pairs)
        raw = RicciCurvature(G, method, weight, distance, proc).compute_pairs(pairs)
        return _pair_contract(pairs, raw, method.name)

    raise TypeError(
        f"method must be a pair-capable name, a CurvatureRequest or a "
        f"CurvatureConfig; got {method!r}."
    )
