"""Adapters and complete metadata for the shipped curvature definitions.

Each entry is one mathematical definition. Graph kind, direction convention,
parameters and solver are its configuration. The formulas live in the formula
modules (``base``/``ot_ricci``, ``directed``, ``forman_variants``,
``forman_directed``);
this module only adapts them to the registry contract and declares what they
apply to.

Catalogue
---------
============================== =========== ============= ===================== ===========
canonical name                 tier        family        graph kinds           scope
============================== =========== ============= ===================== ===========
lin_lu_yau                     core        transport     directed, undirected  edges, pairs
ollivier                       core        transport     directed, undirected  edges, pairs
eidi_jost                      core        transport     directed              edges
forman_node_weighted           core        combinatorial undirected            edges
augmented_forman_node_weighted core        combinatorial undirected            edges
forman_directed                core        combinatorial directed               edges
augmented_forman_directed      core        combinatorial directed               edges
============================== =========== ============= ===================== ===========

Deprecated aliases keep their **existing numerical behaviour**:
``forman_sreejith`` and ``augmented_forman_sreejith`` are the node-weighted
undirected definitions. The names ``forman`` and ``augmented_forman`` were the
incident-weight convention inherited from ``my_ricci_aging``; that convention
matched no published definition. On an unweighted graph its plain value was the
constant 2, while its augmented value was ``2 + |T(e)|``. It was removed rather
than renamed, and the two names now raise with the replacements to use
(:data:`~graph_geometry.curvature.registry.REMOVED_CURVATURES`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any

from .base import TransportCurvatureEngine
from .directed import eidi_jost_ot_config
from .errors import CurvatureConfigurationError, CurvatureDomainError, CurvatureInputError
from .combinatorial import CombinatorialCurvatureEngine
from .forman_directed import (
    augmented_forman_directed_config,
    forman_directed_config,
)
from .forman_variants import (
    NODE_WEIGHT_SCHEMES,
    augmented_forman_node_weighted_config,
    forman_node_weighted_config,
)
from .kernels import (
    BetaDegreeProportional,
    BetaNodeAttr,
    BetaWeightProportional,
    resolve_beta,
)
from .model import CurvatureCapabilities, ParameterSpec
from .ot_ricci import (
    LLY_SOLVER_PREFERENCE,
    lin_lu_yau_ot_config,
    ollivier_ot_config,
    resolve_lly_solver,
)
from .registry import (
    register_curvature,
    register_curvature_alias,
    register_removed_curvature,
)

__all__ = ["KERNEL_CHOICES", "DEFAULT_BETA", "BETA_STRATEGIES", "LLY_SOLVER_CHOICES", "beta_spec"]

#: Transition-kernel conventions of the transport definitions.
KERNEL_CHOICES = ("auto", "undirected", "out", "in", "mixed")

#: The one balancing-factor default shared by Python, UI and configuration.
DEFAULT_BETA = 0.8

#: How the mixed kernel's balancing factor is chosen at each node.
#: ``constant`` uses the ``beta`` parameter everywhere; the others derive a
#: per-node beta(x) from the graph (Bai-Li-Liu-Lai 2509.19989, Remark 3.2).
BETA_STRATEGIES = ("constant", "degree_proportional", "weight_proportional", "node_attr")

#: Historical spellings of the strategies, accepted as ``beta=<name>`` from Python.
_BETA_STRATEGY_ALIASES = {
    "degree_proportional": "degree_proportional",
    "adaptive": "degree_proportional",
    "balanced": "degree_proportional",
    "weight_proportional": "weight_proportional",
    "node_attr": "node_attr",
}

LLY_SOLVER_CHOICES = (None, *LLY_SOLVER_PREFERENCE)

_CITE_OLLIVIER = (
    "Ollivier, Y. (2009). Ricci curvature of Markov chains on metric spaces. "
    "J. Funct. Anal. 256(3), 810-864."
)
_CITE_LLY = (
    "Lin, Y., Lu, L. & Yau, S.-T. (2011). Ricci curvature of graphs. Tohoku "
    "Math. J. 63(4), 605-627."
)
_CITE_MIXED = (
    "Directed mixed kernel: Bai, S., Li, R., Liu, S. & Lai, X. (2025), "
    "arXiv:2509.19989."
)
_CITE_EIDI_JOST = (
    "Eidi, M. & Jost, J. (2020). Ollivier Ricci curvature of directed "
    "hypergraphs. Sci. Rep. 10, 12466."
)
_CITE_FORMAN = (
    "Forman, R. (2003). Bochner's method for cell complexes and combinatorial "
    "Ricci curvature. Discrete Comput. Geom. 29, 323-374"
)
_CITE_SREEJITH = (
    "Sreejith, R. P., Mohanraj, K., Jost, J., Saucan, E. & Samal, A. (2016). "
    "Forman curvature for complex networks. J. Stat. Mech. 063206"
)
_CITE_SAMAL = (
    "Samal, A. et al. (2018). Comparative analysis of two discretizations of "
    "Ricci curvature for complex networks. Sci. Rep. 8, 8650"
)
_CITE_SREEJITH_DIRECTED = (
    "Sreejith, R. P., Jost, J., Saucan, E. & Samal, A. (2016). Forman curvature "
    "for directed networks. arXiv:1605.04662"
)
_CITE_SAUCAN_DIRECTED = (
    "Saucan, E., Sreejith, R. P., Vivek-Ananth, R. P., Jost, J. & Samal, A. "
    "(2019). Discrete Ricci curvatures for directed networks. Chaos Solitons "
    "Fractals 118, 347-360"
)


# ── legacy value forms accepted only by the compatibility wrappers ──────────


def _legacy_beta(value: Any) -> Any:
    """Per-node beta from Python: a strategy name, a mapping or a callable.

    A strategy name is translated by the resolver into ``beta_strategy`` and is
    fully declarative. A mapping or callable is validated exactly as the kernel
    will use it but cannot be persisted in an experiment.
    """
    if isinstance(value, bool) or isinstance(value, Real):
        raise CurvatureConfigurationError(
            f"parameter 'beta' must be in [0, 1]; got {value!r}."
        )
    if isinstance(value, str):
        if value.lower() not in _BETA_STRATEGY_ALIASES:
            raise CurvatureConfigurationError(
                f"parameter 'beta' names no strategy {value!r}; use a float in "
                f"[0, 1] or beta_strategy in {list(BETA_STRATEGIES)}."
            )
        return value
    try:
        resolve_beta(value)
    except (TypeError, ValueError) as exc:
        raise CurvatureConfigurationError(f"parameter 'beta': {exc}") from exc
    return value


def _legacy_node_weight(value: Any) -> Any:
    """A number, mapping or callable node weight from direct Python calls."""
    if isinstance(value, bool):
        raise CurvatureConfigurationError(
            f"parameter 'node_weight' must be one of {list(NODE_WEIGHT_SCHEMES)}; "
            f"got {value!r}."
        )
    if isinstance(value, Real):
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise CurvatureConfigurationError(
                f"parameter 'node_weight' must be finite and strictly positive; "
                f"got {value!r}."
            )
        return value
    if isinstance(value, Mapping) or callable(value):
        return value
    raise CurvatureConfigurationError(
        f"parameter 'node_weight' must be one of {list(NODE_WEIGHT_SCHEMES)}; "
        f"got {value!r}."
    )


# ── parameter schemas ───────────────────────────────────────────────────────

_ALPHA = ParameterSpec(
    name="alpha", kind="float", default=0.5, minimum=0.0, maximum=1.0,
    description="Idleness: probability mass kept at the node itself.",
)
_KERNEL = ParameterSpec(
    name="kernel", kind="choice", default="auto", choices=KERNEL_CHOICES,
    description=(
        "Neighbourhood convention: undirected random walk, or on a digraph out, "
        "in or mixed. 'auto' is a Python convenience resolved from the graph "
        "kind; persisted directed experiments must name out, in or mixed."
    ),
)
_BETA_STRATEGY = ParameterSpec(
    name="beta_strategy", kind="choice", default="constant", choices=BETA_STRATEGIES,
    description=(
        "How the mixed kernel's balancing factor beta(x) is chosen: 'constant' "
        "uses beta; 'degree_proportional' deg_out(x)/deg(x); "
        "'weight_proportional' W_out(x)/(W_out(x)+W_in(x)); 'node_attr' reads "
        "GraphSemantics.beta_attr on every node."
    ),
    graph_kinds=frozenset({"directed"}),
    active_when={"kernel": ("mixed",)},
)
_BETA = ParameterSpec(
    name="beta", kind="float", default=DEFAULT_BETA, minimum=0.0, maximum=1.0,
    description="Mixed-kernel balancing factor: beta*P_out + (1-beta)*P_in.",
    graph_kinds=frozenset({"directed"}),
    active_when={"kernel": ("mixed",), "beta_strategy": ("constant",)},
    legacy_validator=_legacy_beta,
)
_SOLVER = ParameterSpec(
    name="solver", kind="choice", default=None, choices=LLY_SOLVER_CHOICES,
    description=(
        "CVXPY solver for the signed Kantorovich-Rubinstein LP; None selects the "
        "first installed of " + ", ".join(LLY_SOLVER_PREFERENCE) + "."
    ),
)
_NODE_WEIGHT = ParameterSpec(
    name="node_weight", kind="choice", default="one", choices=NODE_WEIGHT_SCHEMES,
    description="Forman node-weight scheme; inactive when node_weight_attr is set.",
    inactive_when_semantics=("node_weight_attr",),
    legacy_validator=_legacy_node_weight,
)
_FACE_WEIGHT = ParameterSpec(
    name="face_weight", kind="float", default=1.0, minimum=0.0,
    minimum_inclusive=False,
    description="Weight of every triangular face (strictly positive).",
)


# ── resolvers ───────────────────────────────────────────────────────────────


def _resolve_kernel(graph_kind: str, parameters: dict) -> tuple[str, tuple[str, ...]]:
    kernel = parameters["kernel"]
    notes: tuple[str, ...] = ()
    if kernel == "auto":
        if graph_kind == "undirected":
            kernel = "undirected"
        else:
            kernel = "mixed"
            notes = (
                "kernel='auto' on a directed graph resolved to the mixed kernel "
                f"(beta_strategy={parameters.get('beta_strategy', 'constant')!r}, "
                f"beta={parameters.get('beta', DEFAULT_BETA)!r}); name kernel='out', "
                "'in' or 'mixed' explicitly for reproducible work.",
            )
    if graph_kind == "undirected" and kernel in ("out", "in", "mixed"):
        raise CurvatureDomainError(
            f"kernel={kernel!r} is defined on directed graphs only, but the input "
            f"is undirected. Use kernel='undirected' (or 'auto'), or pass a "
            f"directed graph."
        )
    if graph_kind == "directed" and kernel == "undirected":
        raise CurvatureDomainError(
            "kernel='undirected' is defined on undirected graphs only, but the "
            "input is directed. Name an explicit 'out', 'in' or 'mixed' kernel, "
            "or convert with G.to_undirected() so the choice is recorded."
        )
    return kernel, notes


def _translate_beta_spelling(parameters: dict) -> tuple[str, ...]:
    """``beta='degree_proportional'`` -> ``beta_strategy='degree_proportional'``.

    Only the historical Python spelling reaches here as a string (the schema
    rejects it otherwise); the translation makes it a declarative, persistable
    configuration.
    """
    beta = parameters.get("beta")
    if not isinstance(beta, str) or parameters.get("beta_strategy") != "constant":
        return ()
    strategy = _BETA_STRATEGY_ALIASES[beta.lower()]
    parameters["beta_strategy"] = strategy
    parameters["beta"] = DEFAULT_BETA
    return (
        f"beta={beta!r} is the compatibility spelling of beta_strategy={strategy!r}; "
        f"it is recorded as the latter.",
    )


def _transport_resolver(*, graph_kind, scope, parameters):
    effective = dict(parameters)
    notes = _translate_beta_spelling(effective)
    kernel, kernel_notes = _resolve_kernel(graph_kind, effective)
    effective["kernel"] = kernel
    return effective, kernel, notes + kernel_notes


def _lly_resolver(*, graph_kind, scope, parameters):
    effective, convention, notes = _transport_resolver(
        graph_kind=graph_kind, scope=scope, parameters=parameters
    )
    try:
        effective["solver"] = resolve_lly_solver(effective["solver"])
    except RuntimeError as exc:  # nothing installed at all
        raise CurvatureConfigurationError(str(exc)) from exc
    return effective, convention, notes


def _fixed_convention(directed: str, undirected: str):
    def resolver(*, graph_kind, scope, parameters):
        return dict(parameters), directed if graph_kind == "directed" else undirected, ()

    return resolver


# ── transport family ────────────────────────────────────────────────────────

_TRANSPORT_BOTH = CurvatureCapabilities(
    family="transport",
    status="core",
    graph_kinds=frozenset({"directed", "undirected"}),
    scopes=frozenset({"edges", "pairs"}),
    consumes=frozenset({"topology", "edge_weight", "edge_distance"}),
)


def _engine(G, config, semantics, proc) -> TransportCurvatureEngine:
    return TransportCurvatureEngine(G, config, semantics, proc)


def beta_spec(G, semantics, beta_strategy: str = "constant", beta: Any = DEFAULT_BETA):
    """The balancing factor the mixed kernel receives for a resolved configuration.

    ``constant`` passes ``beta`` through (a float, or a Python-only per-node
    object from the compatibility wrappers). ``node_attr`` requires the
    attribute on every node: a partly annotated graph would silently mix a node
    value with a fallback.
    """
    if beta_strategy == "constant":
        return beta
    if beta_strategy == "degree_proportional":
        return BetaDegreeProportional()
    if beta_strategy == "weight_proportional":
        return BetaWeightProportional(weight=semantics.weight_attr)
    if beta_strategy == "node_attr":
        attr = semantics.beta_attr
        missing = [n for n in G.nodes() if attr not in G.nodes[n]]
        if missing:
            raise CurvatureInputError(
                f"beta_strategy='node_attr' reads node attribute {attr!r}, which "
                f"{len(missing)} node(s) lack (e.g. {missing[:3]}). Set it on every "
                f"node, or name another attribute with GraphSemantics(beta_attr=...)."
            )
        return BetaNodeAttr(attr=attr)
    raise CurvatureConfigurationError(f"unknown beta_strategy {beta_strategy!r}.")


def _lly_pairs(G, pairs, *, semantics, proc, kernel, solver,
               beta_strategy="constant", beta=DEFAULT_BETA):
    config = lin_lu_yau_ot_config(
        kernel=kernel, beta=beta_spec(G, semantics, beta_strategy, beta), solver=solver
    )
    return _engine(G, config, semantics, proc).compute_pairs(pairs)


@register_curvature(
    "lin_lu_yau",
    capabilities=_TRANSPORT_BOTH,
    parameters=(_KERNEL, _BETA_STRATEGY, _BETA, _SOLVER),
    compute_pairs=_lly_pairs,
    resolver=_lly_resolver,
    label="Lin-Lu-Yau",
    color="#1f77b4",
    description="Lin-Lu-Yau Ricci curvature (signed Kantorovich-Rubinstein LP, CVXPY).",
    citation=f"{_CITE_LLY} {_CITE_MIXED}",
)
def _lly_edges(G, *, semantics, proc, kernel, solver,
               beta_strategy="constant", beta=DEFAULT_BETA):
    config = lin_lu_yau_ot_config(
        kernel=kernel, beta=beta_spec(G, semantics, beta_strategy, beta), solver=solver
    )
    return _engine(G, config, semantics, proc).compute_edges()


def _ollivier_pairs(G, pairs, *, semantics, proc, alpha, kernel,
                    beta_strategy="constant", beta=DEFAULT_BETA):
    config = ollivier_ot_config(
        alpha=alpha, kernel=kernel, beta=beta_spec(G, semantics, beta_strategy, beta)
    )
    return _engine(G, config, semantics, proc).compute_pairs(pairs)


@register_curvature(
    "ollivier",
    capabilities=_TRANSPORT_BOTH,
    parameters=(_ALPHA, _KERNEL, _BETA_STRATEGY, _BETA),
    compute_pairs=_ollivier_pairs,
    resolver=_transport_resolver,
    label="Ollivier",
    color="#ff7f0e",
    description="Ollivier-Ricci curvature with idleness alpha (exact EMD, POT).",
    citation=f"{_CITE_OLLIVIER} {_CITE_MIXED}",
)
def _ollivier_edges(G, *, semantics, proc, alpha, kernel,
                    beta_strategy="constant", beta=DEFAULT_BETA):
    config = ollivier_ot_config(
        alpha=alpha, kernel=kernel, beta=beta_spec(G, semantics, beta_strategy, beta)
    )
    return _engine(G, config, semantics, proc).compute_edges()


@register_curvature(
    "eidi_jost",
    capabilities=CurvatureCapabilities(
        family="transport",
        status="core",
        graph_kinds=frozenset({"directed"}),
        # The in-out supports are joined through the edge (x, y) itself; an
        # arbitrary ordered non-edge has no such guarantee and lies outside the
        # validated construction.
        scopes=frozenset({"edges"}),
        consumes=frozenset({"topology", "edge_weight", "edge_distance"}),
    ),
    resolver=_fixed_convention("in_out", "in_out"),
    label="Eidi-Jost (in-out)",
    color="#2ca02c",
    description="Directed Ollivier curvature via in-out measures; no strong connectivity needed.",
    citation=_CITE_EIDI_JOST,
)
def _eidi_jost_edges(G, *, semantics, proc):
    return _engine(G, eidi_jost_ot_config(), semantics, proc).compute_edges()


# ── combinatorial family ────────────────────────────────────────────────────


@register_curvature(
    "forman_node_weighted",
    capabilities=CurvatureCapabilities(
        family="combinatorial",
        status="core",
        graph_kinds=frozenset({"undirected"}),
        scopes=frozenset({"edges"}),
        consumes=frozenset({"topology", "edge_weight", "node_weight"}),
    ),
    parameters=(_NODE_WEIGHT,),
    resolver=_fixed_convention("undirected_incidence", "undirected_incidence"),
    label="Forman-Ricci (node-weighted)",
    color="#8c564b",
    description="Forman 2003 / Sreejith et al. 2016 with explicit node weights.",
    citation=f"{_CITE_FORMAN}; {_CITE_SREEJITH}.",
)
def _forman_node_weighted_edges(G, *, semantics, proc, node_weight="one"):
    return CombinatorialCurvatureEngine(
        G, forman_node_weighted_config(node_weight), semantics
    ).compute_edges()


@register_curvature(
    "augmented_forman_node_weighted",
    capabilities=CurvatureCapabilities(
        family="combinatorial",
        status="core",
        graph_kinds=frozenset({"undirected"}),
        scopes=frozenset({"edges"}),
        consumes=frozenset({"topology", "edge_weight", "node_weight"}),
    ),
    parameters=(_NODE_WEIGHT, _FACE_WEIGHT),
    resolver=_fixed_convention(
        "undirected_incidence_triangles", "undirected_incidence_triangles"
    ),
    label="Augmented Forman-Ricci (node-weighted)",
    color="#9467bd",
    description="Node-weighted Forman with triangular faces (Samal et al. 2018).",
    citation=f"{_CITE_FORMAN}; {_CITE_SREEJITH}; {_CITE_SAMAL}.",
)
def _augmented_forman_node_weighted_edges(
    G, *, semantics, proc, face_weight, node_weight="one"
):
    return CombinatorialCurvatureEngine(
        G, augmented_forman_node_weighted_config(node_weight, face_weight), semantics
    ).compute_edges()


_DIRECTED_CAPABILITIES = CurvatureCapabilities(
    family="combinatorial",
    status="core",
    graph_kinds=frozenset({"directed"}),
    scopes=frozenset({"edges"}),
    consumes=frozenset({"topology", "edge_weight", "node_weight"}),
)


@register_curvature(
    "forman_directed",
    capabilities=_DIRECTED_CAPABILITIES,
    parameters=(_NODE_WEIGHT,),
    resolver=_fixed_convention("in_tail_out_head", "in_tail_out_head"),
    label="Forman-Ricci (directed)",
    color="#d62728",
    description=(
        "Directed Forman-Ricci: the in-edges of the tail and the out-edges of "
        "the head are charged (Saucan et al. 2019, Eq. 5)."
    ),
    citation=f"{_CITE_FORMAN}; {_CITE_SREEJITH_DIRECTED}; {_CITE_SAUCAN_DIRECTED}.",
)
def _forman_directed_edges(G, *, semantics, proc, node_weight="one"):
    return CombinatorialCurvatureEngine(
        G, forman_directed_config(node_weight), semantics
    ).compute_edges()


@register_curvature(
    "augmented_forman_directed",
    capabilities=_DIRECTED_CAPABILITIES,
    parameters=(_NODE_WEIGHT, _FACE_WEIGHT),
    resolver=_fixed_convention(
        "in_tail_out_head_feed_forward_faces", "in_tail_out_head_feed_forward_faces"
    ),
    label="Augmented Forman-Ricci (directed)",
    color="#e377c2",
    description=(
        "Directed Forman-Ricci with the feed-forward triangles through the edge "
        "as faces (Saucan et al. 2019, Eq. 12)."
    ),
    citation=f"{_CITE_FORMAN}; {_CITE_SREEJITH_DIRECTED}; {_CITE_SAUCAN_DIRECTED}.",
)
def _augmented_forman_directed_edges(
    G, *, semantics, proc, face_weight, node_weight="one"
):
    return CombinatorialCurvatureEngine(
        G, augmented_forman_directed_config(node_weight, face_weight), semantics
    ).compute_edges()


# ── deprecated names, pinned to their existing numbers ──────────────────────

register_curvature_alias(
    "forman_sreejith", "forman_node_weighted",
    message="'forman_sreejith' is deprecated; use 'forman_node_weighted' (same values).",
)
register_curvature_alias(
    "augmented_forman_sreejith", "augmented_forman_node_weighted",
    message=(
        "'augmented_forman_sreejith' is deprecated; use "
        "'augmented_forman_node_weighted' (same values)."
    ),
)


# ── withdrawn names ─────────────────────────────────────────────────────────

_REMOVED_INCIDENT = (
    "{name!r} was the incident-weight Forman convention inherited from "
    "my_ricci_aging: the endpoint term was the sum of the incident edge weights "
    "W(u)+W(v) and the charges carried no node factor. It matches no published "
    "definition -- on an unweighted graph it is {unweighted}, where Forman's own "
    "formula gives {forman_unweighted} -- so it has been removed rather than "
    "renamed. Use {directed!r} for directed input (Saucan et al. 2019) or "
    "{undirected!r} for undirected input (Sreejith et al. 2016 / Samal et al. "
    "2018); both differ numerically."
)

_PLAIN = dict(
    unweighted="the constant 2 on every edge",
    forman_unweighted="4 - deg(u) - deg(v)",
    directed="forman_directed",
    undirected="forman_node_weighted",
)
_AUGMENTED = dict(
    unweighted="2 + |T(e)|, one per triangle through the edge",
    forman_unweighted="4 - deg(u) - deg(v) + 3|T(e)|",
    directed="augmented_forman_directed",
    undirected="augmented_forman_node_weighted",
)

for _name, _behaviour in (
    ("forman", _PLAIN),
    ("forman_incident_weight", _PLAIN),
    ("augmented_forman", _AUGMENTED),
    ("augmented_forman_incident_weight", _AUGMENTED),
):
    register_removed_curvature(
        _name, message=_REMOVED_INCIDENT.format(name=_name, **_behaviour)
    )
