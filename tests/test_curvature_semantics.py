"""Curvature semantics/API redesign: the acceptance criteria, executable.

One registry entry is one mathematical definition; graph kind, direction
convention, attribute roles and parameters are its configuration, resolved and
recorded before any numerical work.
"""

import json
import pickle
import warnings

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.curvature import write_node_means

CANONICAL = {
    "lin_lu_yau": ("core", "transport", {"directed", "undirected"}, {"edges", "pairs"}),
    "ollivier": ("core", "transport", {"directed", "undirected"}, {"edges", "pairs"}),
    "eidi_jost": ("core", "transport", {"directed"}, {"edges"}),
    "forman_node_weighted": ("core", "combinatorial", {"undirected"}, {"edges"}),
    "augmented_forman_node_weighted": ("core", "combinatorial", {"undirected"}, {"edges"}),
    "forman_directed": ("core", "combinatorial", {"directed"}, {"edges"}),
    "augmented_forman_directed": ("core", "combinatorial", {"directed"}, {"edges"}),
}

ALIASES = {
    "forman_sreejith": "forman_node_weighted",
    "augmented_forman_sreejith": "augmented_forman_node_weighted",
}


def _unit(G):
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    return G


def _weighted_undirected():
    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = float(G.degree(u) + G.degree(v))
        G[u][v]["distance"] = 1.0
    return G


def _weighted_directed():
    G = nx.DiGraph()
    for i, (u, v) in enumerate([(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0), (1, 3), (3, 1)]):
        G.add_edge(u, v, weight=0.5 + i % 3, distance=1.0 + (i % 2))
    return G


# ── metadata ────────────────────────────────────────────────────────────────


def test_the_built_in_catalogue_is_exactly_the_specified_one():
    assert list(gg.CURVATURE_REGISTRY) == list(CANONICAL)
    assert dict((a, r.target) for a, r in gg.CURVATURE_ALIASES.items()) == ALIASES


@pytest.mark.parametrize("name", list(CANONICAL))
def test_every_canonical_definition_has_complete_metadata(name):
    spec = gg.CURVATURE_REGISTRY[name]
    status, family, kinds, scopes = CANONICAL[name]
    caps = spec.capabilities
    assert spec.metadata_complete
    assert (caps.status, caps.family) == (status, family)
    assert caps.graph_kinds == frozenset(kinds)
    assert caps.scopes == frozenset(scopes)
    assert "topology" in caps.consumes
    assert spec.citation and spec.label and spec.description and spec.color
    assert json.loads(json.dumps(spec.to_dict())) == spec.to_dict()


def test_parameter_schema_is_the_authoritative_table():
    def schema(name):
        return {p.name: (p.kind, p.default) for p in gg.get_curvature_spec(name).parameters}

    assert schema("ollivier") == {
        "alpha": ("float", 0.5), "kernel": ("choice", "auto"),
        "beta_strategy": ("choice", "constant"), "beta": ("float", 0.8),
    }
    assert schema("lin_lu_yau") == {
        "kernel": ("choice", "auto"), "beta_strategy": ("choice", "constant"),
        "beta": ("float", 0.8), "solver": ("choice", None),
    }
    assert schema("eidi_jost") == {}
    assert schema("forman_directed") == {"node_weight": ("choice", "one")}
    assert schema("augmented_forman_directed") == {
        "node_weight": ("choice", "one"), "face_weight": ("float", 1.0),
    }
    assert schema("forman_node_weighted") == {"node_weight": ("choice", "one")}
    assert schema("augmented_forman_node_weighted") == {
        "node_weight": ("choice", "one"), "face_weight": ("float", 1.0),
    }
    reserved = {"proc", "weight", "distance", "semantics", "annotate", "label"}
    for spec in gg.list_curvatures():
        assert not reserved & {p.name for p in spec.parameters}


def test_one_beta_default_everywhere():
    from graph_geometry.curvature import kernels

    assert gg.get_curvature_spec("ollivier").parameter("beta").default == 0.8
    assert gg.get_curvature_spec("lin_lu_yau").parameter("beta").default == 0.8
    assert kernels.MixedKernel().beta == 0.8
    assert kernels.AutoKernel().beta == 0.8


def test_aliases_are_not_listed_as_curvatures():
    names = {spec.name for spec in gg.list_curvatures()}
    assert not names & set(ALIASES)


def test_capability_queries_filter():
    assert {s.name for s in gg.list_curvatures(graph_kind="directed")} == {
        n for n, (_, _, kinds, _) in CANONICAL.items() if "directed" in kinds
    }
    assert {s.name for s in gg.list_curvatures(scope="pairs")} == {"lin_lu_yau", "ollivier"}
    assert {s.name for s in gg.list_curvatures(statuses=frozenset({"core"}))} == set(CANONICAL)
    assert gg.list_curvatures(statuses=frozenset({"specialized"})) == ()


# ── resolution ──────────────────────────────────────────────────────────────


def test_a_capability_holds_both_kinds_but_a_run_resolves_one():
    spec = gg.get_curvature_spec("ollivier")
    assert spec.capabilities.graph_kinds == frozenset({"directed", "undirected"})
    und = gg.resolve_curvature(_unit(nx.cycle_graph(5)), "ollivier")
    dig = gg.resolve_curvature(_weighted_directed(), gg.CurvatureRequest("ollivier", {"kernel": "out"}))
    assert (und.graph_kind, und.direction_convention) == ("undirected", "undirected")
    assert (dig.graph_kind, dig.direction_convention) == ("directed", "out")
    assert "beta" not in und.parameters and "beta" not in dig.parameters


def test_auto_kernel_never_survives_resolution():
    with pytest.warns(gg.CurvatureWarning, match="kernel='auto' on a directed graph"):
        resolved = gg.resolve_curvature(_weighted_directed(), "lin_lu_yau")
    assert resolved.parameters["kernel"] == "mixed"
    assert resolved.parameters["beta"] == 0.8
    assert "auto" not in json.dumps(resolved.to_dict()["parameters"])
    assert resolved.warnings


def test_auto_on_an_undirected_graph_is_not_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resolved = gg.resolve_curvature(_unit(nx.cycle_graph(4)), "ollivier")
    assert resolved.parameters["kernel"] == "undirected"


def test_auto_equals_the_default_mixed_convention_numerically():
    G = _weighted_directed()
    with pytest.warns(gg.CurvatureWarning):
        auto = gg.compute_curvature(G, "lin_lu_yau").values
    mixed = gg.compute_curvature(G, gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed"})).values
    assert auto == mixed


def test_the_solver_actually_used_is_recorded():
    resolved = gg.resolve_curvature(_unit(nx.cycle_graph(4)), "lin_lu_yau")
    assert resolved.parameters["solver"] in ("SCIPY", "CLARABEL", "ECOS", "SCS")


@pytest.mark.parametrize(
    "parameters, match",
    [
        ({"alpah": 0.5}, "no parameter"),                  # misspelling is not swallowed
        ({"alpha": 1.5}, "<= 1.0"),
        ({"alpha": True}, "must be a number"),
        ({"alpha": float("nan")}, "finite"),
        ({"kernel": "sideways"}, "must be one of"),
        ({"kernel": "out", "beta": 0.3}, "inactive"),       # beta only for mixed
    ],
)
def test_strict_resolution_rejects_bad_parameters(parameters, match):
    with pytest.raises(gg.CurvatureConfigurationError, match=match):
        gg.resolve_curvature(_weighted_directed(), gg.CurvatureRequest("ollivier", parameters))


# ── balancing-factor strategies ─────────────────────────────────────────────


def _asymmetric_weighted():
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0), (1, 3)])
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + (u + v) % 3
        G[u][v]["distance"] = 1.0
    return G


def test_beta_strategy_is_a_declarative_choice_active_only_for_mixed():
    spec = gg.get_curvature_spec("ollivier").parameter("beta_strategy")
    assert spec.choices == ("constant", "degree_proportional", "weight_proportional", "node_attr")
    G = _asymmetric_weighted()
    out = gg.resolve_curvature(G, gg.CurvatureRequest("ollivier", {"kernel": "out"}))
    assert "beta_strategy" not in out.parameters and "beta" not in out.parameters
    strategy = gg.resolve_curvature(
        G, gg.CurvatureRequest("ollivier", {"kernel": "mixed", "beta_strategy": "degree_proportional"})
    )
    assert strategy.parameters["beta_strategy"] == "degree_proportional"
    assert "beta" not in strategy.parameters                    # beta applies to constant only
    with pytest.raises(gg.CurvatureConfigurationError, match="inactive"):
        gg.resolve_curvature(G, gg.CurvatureRequest(
            "ollivier", {"kernel": "mixed", "beta_strategy": "weight_proportional", "beta": 0.3}))
    undirected = _unit(nx.cycle_graph(4))
    with pytest.raises(gg.CurvatureConfigurationError, match="inactive"):
        gg.resolve_curvature(undirected, gg.CurvatureRequest(
            "ollivier", {"beta_strategy": "degree_proportional"}))


@pytest.mark.parametrize(
    "strategy, legacy",
    [
        ("degree_proportional", "degree_proportional"),
        ("weight_proportional", "weight_proportional"),
    ],
)
def test_each_strategy_reproduces_the_python_strategy_object(strategy, legacy):
    from graph_geometry.curvature import kernels

    G = _asymmetric_weighted()
    for method in ("ollivier", "lin_lu_yau"):
        declared = gg.compute_curvature(
            G, gg.CurvatureRequest(method, {"kernel": "mixed", "beta_strategy": strategy})
        ).values
        objects = {"degree_proportional": kernels.BetaDegreeProportional(),
                   "weight_proportional": kernels.BetaWeightProportional()}
        with pytest.warns(gg.CurvatureDeprecationWarning, match="Python-only"):
            direct = gg.curvature(G.copy(), method, kernel="mixed", beta=objects[legacy])
        assert dict(declared) == direct


def test_node_attr_strategy_reads_the_semantics_attribute_on_every_node():
    G = _asymmetric_weighted()
    nx.set_node_attributes(G, {0: 0.9, 1: 0.2, 2: 0.6, 3: 0.4}, "bal")
    semantics = gg.GraphSemantics(beta_attr="bal")
    request = gg.CurvatureRequest("ollivier", {"kernel": "mixed", "beta_strategy": "node_attr"})
    declared = gg.compute_curvature(G, request, semantics=semantics).values
    mapping = gg.compute_curvature(G, gg.CurvatureRequest("ollivier", {"kernel": "mixed"})).values
    assert dict(declared) != dict(mapping)
    from graph_geometry.curvature import kernels
    with pytest.warns(gg.CurvatureDeprecationWarning):
        direct = gg.curvature(G.copy(), "ollivier", kernel="mixed",
                              beta=kernels.BetaNodeAttr(attr="bal"))
    assert dict(declared) == direct
    del G.nodes[3]["bal"]
    with pytest.raises(gg.CurvatureInputError, match="lack"):
        gg.compute_curvature(G, request, semantics=semantics)


def test_historical_beta_strategy_spelling_is_translated_not_python_only():
    G = _asymmetric_weighted()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        kappa = gg.curvature(G.copy(), "lin_lu_yau", kernel="mixed", beta="adaptive")
    messages = [str(w.message) for w in caught]
    assert any("compatibility spelling of beta_strategy='degree_proportional'" in m
               for m in messages)
    assert not any("Python-only" in m for m in messages)
    declared = gg.compute_curvature(G, gg.CurvatureRequest(
        "lin_lu_yau", {"kernel": "mixed", "beta_strategy": "degree_proportional"})).values
    assert kappa == dict(declared)
    with pytest.raises(gg.CurvatureConfigurationError):     # strict API: beta is a float
        gg.resolve_curvature(G, gg.CurvatureRequest(
            "lin_lu_yau", {"kernel": "mixed", "beta": "adaptive"}))


def test_face_weight_must_be_strictly_positive_in_the_schema():
    with pytest.raises(gg.CurvatureConfigurationError, match="> 0.0"):
        gg.resolve_curvature(
            _unit(nx.complete_graph(3)),
            gg.CurvatureRequest("augmented_forman_node_weighted", {"face_weight": 0.0}),
        )


def test_node_weight_is_inactive_when_the_semantics_name_a_node_attribute():
    G = _unit(nx.path_graph(3))
    nx.set_node_attributes(G, 2.0, "w_node")
    semantics = gg.GraphSemantics(node_weight_attr="w_node")
    resolved = gg.resolve_curvature(G, "forman_node_weighted", semantics=semantics)
    assert "node_weight" not in resolved.parameters
    with pytest.raises(gg.CurvatureConfigurationError, match="node_weight_attr is set"):
        gg.resolve_curvature(
            G, gg.CurvatureRequest("forman_node_weighted", {"node_weight": "degree"}),
            semantics=semantics,
        )


def test_missing_node_weight_values_are_errors():
    G = _unit(nx.path_graph(3))
    G.nodes[0]["w_node"] = 2.0
    with pytest.raises(gg.CurvatureInputError, match="node 1"):
        gg.compute_curvature(
            G, "forman_node_weighted", semantics=gg.GraphSemantics(node_weight_attr="w_node")
        )


def test_resolution_does_no_numerical_work():
    calls = []

    @gg.register_curvature(
        "counting_kappa",
        capabilities=gg.CurvatureCapabilities(
            family="combinatorial", status="experimental",
            graph_kinds=frozenset({"undirected"}), scopes=frozenset({"edges"}),
            consumes=frozenset({"topology"}),
        ),
    )
    def counting(G, *, semantics, proc):
        calls.append(1)
        return {e: 0.0 for e in G.edges()}

    try:
        gg.resolve_curvature(nx.path_graph(3), "counting_kappa")
        assert not calls
        with pytest.raises(gg.CurvatureDomainError):
            gg.compute_curvature(nx.DiGraph([(0, 1)]), "counting_kappa")
        assert not calls
    finally:
        gg.unregister_curvature("counting_kappa")


def test_eidi_jost_rejects_pair_scope_before_numerical_work():
    with pytest.raises(gg.CurvatureDomainError, match="edges only"):
        gg.resolve_curvature(_weighted_directed(), "eidi_jost", scope="pairs")


def test_no_hidden_symmetrisation():
    with pytest.raises(gg.CurvatureDomainError, match="undirected graphs only"):
        gg.compute_curvature(_weighted_directed(), "forman_node_weighted")
    with pytest.raises(gg.CurvatureDomainError, match="directed graphs only"):
        gg.compute_curvature(_unit(nx.cycle_graph(4)), "eidi_jost")


# ── aliases keep their existing numbers ─────────────────────────────────────


@pytest.mark.parametrize(
    "alias, graph",
    [
        ("forman_sreejith", _weighted_undirected),
        ("augmented_forman_sreejith", _weighted_undirected),
    ],
)
def test_alias_and_canonical_name_are_numerically_identical(alias, graph):
    with pytest.warns(gg.CurvatureDeprecationWarning, match=alias):
        old = gg.compute_curvature(graph(), alias)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # canonical: no warning
        new = gg.compute_curvature(graph(), ALIASES[alias])
    assert dict(old.values) == dict(new.values)
    assert old.resolved.requested_method == alias
    assert old.resolved.method == new.resolved.method == ALIASES[alias]


@pytest.mark.parametrize(
    "withdrawn, replacement",
    [
        ("forman", "forman_directed"),
        ("augmented_forman", "augmented_forman_directed"),
        ("forman_incident_weight", "forman_node_weighted"),
        ("augmented_forman_incident_weight", "augmented_forman_node_weighted"),
    ],
)
def test_withdrawn_incident_weight_names_name_their_replacements(withdrawn, replacement):
    """The incident-weight convention was removed, not renamed or redirected.

    Redirecting the name would have returned different numbers under the same
    call, so the name fails and says which definitions replace it.
    """
    G = _unit(nx.star_graph(4))                             # centre degree 4, leaves 1
    with pytest.raises(gg.CurvatureConfigurationError, match=replacement):
        gg.compute_curvature(G, withdrawn)
    # the node-weighted formula is unaffected: 4 - deg(u) - deg(v) = 4 - 4 - 1
    assert set(gg.compute_curvature(G, "forman_node_weighted").values.values()) == {-1.0}


# ── evaluation and the result contract ──────────────────────────────────────


def test_compute_does_not_mutate_unless_asked():
    G = _unit(nx.cycle_graph(5))
    before = nx.to_dict_of_dicts(G), dict(G.nodes(data=True))
    result = gg.compute_curvature(G, "ollivier")
    assert (nx.to_dict_of_dicts(G), dict(G.nodes(data=True))) == before

    gg.compute_curvature(G, "ollivier", annotate=True, annotation_attr="orc")
    assert all(G[u][v]["orc"] == result.values[(u, v)] for u, v in G.edges())
    assert all(G.nodes[n]["orc"] == result.incident_node_means[n] for n in G.nodes())


def test_legacy_plugins_do_not_mutate_the_graph_either():
    with pytest.warns(gg.CurvatureDeprecationWarning):
        @gg.register_curvature("legacy_writer")
        def legacy_writer(G, *, weight="weight", distance="distance", proc=1):
            for u, v in G.edges():
                G[u][v]["ricciCurvature"] = 1.0
            return {e: 1.0 for e in G.edges()}

    try:
        G = nx.path_graph(3)
        with pytest.warns(gg.CurvatureWarning, match="experimental"):
            gg.compute_curvature(G, "legacy_writer")
        assert all("ricciCurvature" not in d for _, _, d in G.edges(data=True))
        assert gg.get_curvature_spec("legacy_writer").capabilities.status == "experimental"
        assert not gg.get_curvature_spec("legacy_writer").metadata_complete
    finally:
        gg.unregister_curvature("legacy_writer")


@pytest.mark.parametrize(
    "make, method",
    [(_weighted_undirected, "forman_node_weighted"), (_weighted_directed, "forman_directed")],
)
def test_incident_node_means_match_the_historical_writer_bit_for_bit(make, method):
    # Same graph object: G.copy() may reorder an undirected adjacency, and the
    # mean's summation order follows the adjacency order.
    G = make()
    result = gg.compute_curvature(G, method)
    for (u, v), value in result.values.items():
        G[u][v]["ricciCurvature"] = value
    write_node_means(G)
    assert dict(result.incident_node_means) == {
        n: d["ricciCurvature"] for n, d in G.nodes(data=True) if "ricciCurvature" in d
    }


def test_results_are_immutable_and_picklable():
    result = gg.compute_curvature(_unit(nx.cycle_graph(4)), "ollivier")
    with pytest.raises(TypeError):
        result.values[(0, 1)] = 9.0
    with pytest.raises(TypeError):
        result.resolved.parameters["alpha"] = 0.1
    restored = pickle.loads(pickle.dumps(result))
    assert dict(restored.values) == dict(result.values)


def test_request_parameters_are_copied_on_construction():
    params = {"alpha": 0.25}
    request = gg.CurvatureRequest("ollivier", params)
    params["alpha"] = 0.75
    assert request.parameters["alpha"] == 0.25


def test_nested_request_parameters_are_frozen_on_construction():
    """The outer copy used to share the caller's nested dict and list."""
    beta = {0: 0.2, 1: 0.7}
    extra = [1, 2]
    request = gg.CurvatureRequest("ollivier", {"beta": beta, "extra": extra, "tags": {"a"}})
    beta[0] = 0.9
    extra.append(3)
    assert dict(request.parameters["beta"]) == {0: 0.2, 1: 0.7}
    assert request.parameters["extra"] == (1, 2)
    assert request.parameters["tags"] == frozenset({"a"})


def test_mutating_a_beta_mapping_after_the_flow_does_not_rewrite_its_record():
    beta = {0: 0.2, 1: 0.7, 2: 0.5, 3: 0.4}
    result = gg.RicciFlow(
        _weighted_directed(), curvature="ollivier", kernel="mixed", beta=beta
    ).run(iterations=2, step=0.01, early_stop=False)
    beta[0] = 0.9
    del beta[3]
    assert dict(result.resolved.parameters["beta"]) == {0: 0.2, 1: 0.7, 2: 0.5, 3: 0.4}


def test_nested_frozen_mappings_refuse_mutation():
    record = gg.CurvatureRequest("ollivier", {"beta": {0: 0.2}, "outer": {"inner": {"x": [1]}}})
    nested = record.parameters["outer"]["inner"]
    with pytest.raises(TypeError):
        record.parameters["beta"][0] = 0.5
    with pytest.raises(TypeError):
        nested["x"] = 2
    with pytest.raises(AttributeError):
        nested._data = {}
    assert not hasattr(nested, "update") and not hasattr(nested, "pop")
    with pytest.raises(AttributeError):
        nested["x"].append(2)                               # a tuple now


def test_frozen_records_round_trip_through_pickle_hash_and_json():
    beta = {0: 0.2, 1: 0.7, 2: 0.5, 3: 0.4}
    request = gg.CurvatureRequest(
        "ollivier", {"kernel": "mixed", "beta": beta, "outer": {"seq": [1, (2, [3])]}}
    )
    restored = pickle.loads(pickle.dumps(request))
    assert restored == request
    assert hash(restored.parameters) == hash(request.parameters)
    assert restored.parameters["outer"]["seq"] == (1, (2, (3,)))
    assert json.loads(json.dumps(request.to_dict()))["parameters"] == {
        "kernel": "mixed",
        "beta": {"0": 0.2, "1": 0.7, "2": 0.5, "3": 0.4},
        "outer": {"seq": [1, [2, [3]]]},
    }
    result = gg.RicciFlow(
        _weighted_directed(), curvature="ollivier", kernel="mixed", beta=beta
    ).run(iterations=1, step=0.01, early_stop=False)
    assert pickle.loads(pickle.dumps(result.resolved)) == result.resolved
    json.dumps(result.to_dict())


def test_mapping_keys_are_kept_as_given():
    """Keys may be graph nodes; freezing must not turn a tuple node into anything else."""
    request = gg.CurvatureRequest("ollivier", {"beta": {("a", 1): 0.3, 0: 0.6}})
    assert set(request.parameters["beta"]) == {("a", 1), 0}


def test_pair_results_keep_the_requested_orientation_and_have_no_node_means():
    D = _unit(nx.DiGraph([(i, (i + 1) % 4) for i in range(4)]))
    pairs = [(2, 0), (0, 2), (3, 1)]
    result = gg.compute_pair_curvature(D, gg.CurvatureRequest("ollivier", {"kernel": "out"}), pairs)
    assert list(result.values) == pairs
    assert dict(result.incident_node_means) == {}
    assert result.resolved.scope == "pairs"


def test_every_model_serialises_to_json():
    G = _unit(nx.cycle_graph(4))
    result = gg.compute_curvature(G, gg.CurvatureRequest("ollivier", {"alpha": 0.0}, label="lazy"))
    failure = gg.CurvatureFailure("inapplicable", "CurvatureDomainError", "no")
    for record in (result, result.resolved, failure, gg.GraphSemantics(),
                   gg.get_curvature_spec("ollivier").capabilities,
                   gg.CurvatureRequest("ollivier", {"alpha": 0.0})):
        payload = record.to_dict()
        assert json.loads(json.dumps(payload)) == payload


def test_proc_does_not_change_values():
    G = _unit(nx.barabasi_albert_graph(30, 2, seed=4))
    serial = gg.compute_curvature(G, "ollivier", proc=1).values
    parallel = gg.compute_curvature(G, "ollivier", proc=3).values
    assert dict(serial) == dict(parallel)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_proc_is_validated(bad):
    with pytest.raises(gg.CurvatureConfigurationError, match="proc"):
        gg.compute_curvature(_unit(nx.cycle_graph(4)), "ollivier", proc=bad)


def test_missing_attribute_policy_error():
    G = nx.cycle_graph(4)
    with pytest.raises(gg.CurvatureInputError, match="missing_weight='error'"):
        gg.compute_curvature(G, "ollivier", semantics=gg.GraphSemantics(missing_weight="error"))
    # a definition that does not consume distance does not validate it
    gg.compute_curvature(
        _unit(G), "forman_node_weighted",
        semantics=gg.GraphSemantics(distance_attr="absent", missing_distance="error"),
    )


# ── registry rules ──────────────────────────────────────────────────────────


_CAPS = gg.CurvatureCapabilities(
    family="combinatorial", status="experimental",
    graph_kinds=frozenset({"undirected"}), scopes=frozenset({"edges"}),
    consumes=frozenset({"topology"}),
)


def _noop(G, *, semantics, proc):
    return {e: 0.0 for e in G.edges()}


def test_duplicate_names_are_rejected():
    with pytest.raises(gg.CurvatureConfigurationError, match="already registered"):
        gg.register_curvature("ollivier", capabilities=_CAPS)(_noop)
    with pytest.raises(gg.CurvatureConfigurationError, match="already registered"):
        gg.register_curvature("forman_node_weighted", capabilities=_CAPS)(_noop)


def test_dangling_alias_is_rejected():
    with pytest.raises(gg.CurvatureConfigurationError, match="not a registered"):
        gg.register_curvature_alias("nowhere_alias", "nowhere", message="x")


def test_pair_scope_requires_a_pair_callable_and_vice_versa():
    pairs_caps = gg.CurvatureCapabilities(
        family="transport", status="experimental",
        graph_kinds=frozenset({"undirected"}), scopes=frozenset({"edges", "pairs"}),
        consumes=frozenset({"topology"}),
    )
    with pytest.raises(gg.CurvatureConfigurationError, match="no compute_pairs"):
        gg.register_curvature("bad_pairs_a", capabilities=pairs_caps)(_noop)
    with pytest.raises(gg.CurvatureConfigurationError, match="without the 'pairs' scope"):
        gg.register_curvature("bad_pairs_b", capabilities=_CAPS, compute_pairs=_noop)(_noop)
    assert "bad_pairs_a" not in gg.CURVATURE_REGISTRY
    assert "bad_pairs_b" not in gg.CURVATURE_REGISTRY


# ── comparison ──────────────────────────────────────────────────────────────


def test_two_configurations_of_one_method_compare_under_distinct_labels():
    G = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]))
    cmp = gg.compare_curvature(
        G,
        requests=[
            gg.CurvatureRequest("ollivier", {"alpha": 0.0, "kernel": "out"}, label="ORC a=0"),
            gg.CurvatureRequest("ollivier", {"alpha": 0.5, "kernel": "out"}, label="ORC a=0.5"),
        ],
        semantics=gg.GraphSemantics(),
        proc=2,
    )
    assert cmp.labels == ["ORC a=0", "ORC a=0.5"]
    assert {row["definition"] for row in cmp.summary()} == {"ollivier"}
    assert cmp.csv_text().splitlines()[0] == "source,target,ORC a=0,ORC a=0.5"
    assert cmp.values["ORC a=0"] != cmp.values["ORC a=0.5"]
    assert json.loads(json.dumps(cmp.to_dict()))


def test_duplicate_comparison_labels_are_configuration_errors():
    with pytest.raises(gg.CurvatureConfigurationError, match="unique"):
        gg.compare_curvature(
            nx.cycle_graph(4),
            requests=[gg.CurvatureRequest("ollivier"), gg.CurvatureRequest("ollivier")],
        )


def test_comparison_views_are_read_only():
    cmp = gg.compare_curvature(_unit(nx.cycle_graph(4)), ["forman_node_weighted"])
    with pytest.raises(TypeError):
        cmp.values["x"] = {}
    with pytest.raises(TypeError):
        cmp.errors["x"] = "y"


def test_default_comparison_excludes_experimental_definitions():
    with pytest.warns(gg.CurvatureDeprecationWarning):
        @gg.register_curvature("experimental_kappa")
        def experimental_kappa(G, *, weight="weight", distance="distance", proc=1):
            return {e: 0.0 for e in G.edges()}

    try:
        cmp = gg.compare_curvature(_unit(nx.cycle_graph(4)))
        assert "experimental_kappa" not in cmp.labels
        assert set(cmp.labels) == set(CANONICAL)
    finally:
        gg.unregister_curvature("experimental_kappa")


def test_comparison_records_all_six_failure_kinds():
    def plugin(name, body):
        gg.register_curvature(
            name,
            capabilities=gg.CurvatureCapabilities(
                family="combinatorial", status="experimental",
                graph_kinds=frozenset({"directed", "undirected"}),
                scopes=frozenset({"edges"}),
                consumes=frozenset({"topology", "edge_weight"}),
            ),
        )(body)

    def numerical(G, *, semantics, proc):
        raise gg.CurvatureNumericalError("solver diverged")

    def partial(G, *, semantics, proc):
        return {next(iter(G.edges())): 0.0}

    def broken(G, *, semantics, proc):
        return 1 / 0

    plugin("fk_numerical", numerical)
    plugin("fk_contract", partial)
    plugin("fk_internal", broken)
    try:
        G = _unit(nx.cycle_graph(4))
        G[0][1]["weight"] = float("nan")
        H = _unit(nx.cycle_graph(4))
        requests = [
            gg.CurvatureRequest("ollivier", {"alpha": 2.0}, label="configuration"),
            gg.CurvatureRequest("eidi_jost", label="inapplicable"),
            gg.CurvatureRequest("fk_numerical", label="numerical_failure"),
            gg.CurvatureRequest("fk_contract", label="contract_failure"),
            gg.CurvatureRequest("fk_internal", label="internal_failure"),
        ]
        cmp = gg.compare_curvature(H, requests=requests)
        kinds = {label: failure.kind for label, failure in cmp.failures.items()}
        assert kinds == {r.label: r.label for r in requests}
        assert cmp.failures["internal_failure"].exception_type == "ZeroDivisionError"
        assert cmp.failures["configuration"].resolved is None
        assert cmp.failures["numerical_failure"].resolved.method == "fk_numerical"

        bad_input = gg.compare_curvature(G, requests=[gg.CurvatureRequest("ollivier")])
        assert bad_input.failures["ollivier"].kind == "invalid_input"
    finally:
        for name in ("fk_numerical", "fk_contract", "fk_internal"):
            gg.unregister_curvature(name)


def test_direct_calls_raise_the_typed_error_and_keep_internal_tracebacks():
    with pytest.raises(gg.CurvatureConfigurationError):
        gg.compute_curvature(nx.cycle_graph(4), "does_not_exist")

    @gg.register_curvature("raises_internal", capabilities=_CAPS)
    def raises_internal(G, *, semantics, proc):
        raise KeyError("programming error")

    try:
        with pytest.raises(KeyError):                     # not relabelled
            gg.compute_curvature(nx.cycle_graph(4), "raises_internal")
    finally:
        gg.unregister_curvature("raises_internal")


# ── flow consumes the resolved record ───────────────────────────────────────


def test_flow_resolves_once_and_records_the_configuration():
    G = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]))
    request = gg.CurvatureRequest("ollivier", {"kernel": "mixed", "beta": 0.6})
    result = gg.RicciFlow(G, curvature=request, semantics=gg.GraphSemantics()).run(
        iterations=3, step=0.05, early_stop=False
    )
    assert result.resolved.method == "ollivier"
    assert result.resolved.parameters["beta"] == 0.6
    # the snapshots are exactly what the recorded configuration computes
    fresh = gg.compute_curvature(result.snapshots[-1], request).values
    for (u, v), value in fresh.items():
        assert result.snapshots[-1][u][v]["ricciCurvature"] == pytest.approx(value, abs=1e-12)


def test_flow_request_and_named_call_are_identical():
    G = _unit(nx.cycle_graph(6))
    by_name = gg.RicciFlow(G, curvature="ollivier", alpha=0.25).run(iterations=4, step=0.05)
    by_request = gg.RicciFlow(
        G, curvature=gg.CurvatureRequest("ollivier", {"alpha": 0.25})
    ).run(iterations=4, step=0.05)
    assert by_name.convergence == by_request.convergence


def test_flow_warns_when_the_evolving_channel_is_not_consumed():
    G = _unit(nx.cycle_graph(5))
    with pytest.warns(gg.CurvatureWarning, match="invariant under the flow"):
        result = gg.RicciFlow(G, curvature="forman_node_weighted", evolve="distance").run(
            iterations=2, step=0.05, early_stop=False
        )
    assert any("invariant" in w for w in result.warnings)


def test_flow_semantics_and_contradicting_attribute_names_are_rejected():
    with pytest.raises(gg.CurvatureConfigurationError, match="contradicts"):
        gg.RicciFlow(nx.cycle_graph(4), weight="w", semantics=gg.GraphSemantics(weight_attr="strength"))


# ── graph layer ─────────────────────────────────────────────────────────────


def test_graph_config_hands_its_attribute_names_to_semantics():
    cfg = gg.GraphConfig(weight_attr="strength", distance_attr="length", node_attr="mass")
    assert cfg.to_semantics() == gg.GraphSemantics(
        weight_attr="strength", distance_attr="length", node_weight_attr=None
    )
    assert cfg.to_semantics(node_weight=True).node_weight_attr == "mass"


def test_semantics_are_validated():
    with pytest.raises(ValueError, match="missing_weight"):
        gg.GraphSemantics(missing_weight="zero")
    with pytest.raises(ValueError, match="Unknown semantics"):
        gg.GraphSemantics.from_dict({"weight": "w"})


def test_loaders_warn_when_the_direction_is_implicit():
    with pytest.warns(FutureWarning, match="directed"):
        G = gg.load.from_edges([(0, 1)])
    assert G.is_directed()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not gg.load.from_edges([(0, 1)], directed=False).is_directed()
