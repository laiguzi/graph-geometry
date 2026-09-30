"""Registry, dispatch, kernel/beta resolvers, and custom-curvature tests."""

import networkx as nx
import numpy as np
import pytest

import graph_geometry as gg
from graph_geometry.curvature import kernels


def test_builtin_methods_registered():
    assert set(gg.CURVATURE_REGISTRY) >= {
        "lin_lu_yau", "ollivier", "eidi_jost",
        "forman_node_weighted", "augmented_forman_node_weighted",
        "forman_directed", "augmented_forman_directed",
    }
    assert set(gg.CURVATURE_ALIASES) >= {"forman_sreejith", "augmented_forman_sreejith"}
    spec = gg.CURVATURE_REGISTRY["lin_lu_yau"]
    assert spec.label == "Lin-Lu-Yau"
    assert spec.color  # metadata present for UI reuse


def test_builtin_labels_and_colors_are_unique():
    """Regression: forman and lin_lu_yau both registered #1f77b4, so every
    consumer of ``spec.color`` -- the UI selector, the comparison histogram,
    the benchmark figure -- drew two different methods identically.
    """
    for field in ("color", "label"):
        values = [getattr(s, field) for s in gg.CURVATURE_REGISTRY.values()
                  if getattr(s, field)]
        duplicated = {v for v in values if values.count(v) > 1}
        assert not duplicated, f"duplicate {field}: {duplicated}"


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        gg.curvature(nx.complete_graph(3), method="nope")


def test_bad_method_type_raises():
    with pytest.raises(TypeError):
        gg.curvature(nx.complete_graph(3), method=123)


def test_register_custom_curvature():
    @gg.register_curvature("const_one", label="ConstOne")
    def _const_one(G, *, weight="weight", distance="distance", proc=1, **kw):
        vals = {(u, v): 1.0 for u, v in G.edges()}
        for (u, v), val in vals.items():
            G[u][v]["ricciCurvature"] = val
        return vals

    try:
        G = nx.complete_graph(3)
        r = gg.curvature(G, method="const_one")
        assert all(v == 1.0 for v in r.values())
        assert "const_one" in gg.CURVATURE_REGISTRY
        assert gg.CURVATURE_REGISTRY["const_one"].label == "ConstOne"
    finally:
        gg.CURVATURE_REGISTRY.pop("const_one", None)


def test_callable_method_dispatch():
    def my_fn(G, *, weight="weight", distance="distance", proc=1, **kw):
        return {(u, v): 42.0 for u, v in G.edges()}

    G = nx.complete_graph(3)
    r = gg.curvature(G, method=my_fn)
    assert all(v == 42.0 for v in r.values())
    assert all(G[u][v]["ricciCurvature"] == 42.0 for u, v in G.edges())
    assert all(G.nodes[n]["ricciCurvature"] == 42.0 for n in G.nodes())


@pytest.mark.parametrize(
    "bad_result, error",
    [
        ({(0, 1): 1.0}, ValueError),
        ({(0, 1): 1.0, (0, 2): 1.0, (1, 2): float("nan")}, ValueError),
        ({(0, 1): 1.0, (0, 2): 1.0, (1, 2): True}, gg.CurvatureContractError),
    ],
)
def test_callable_result_contract_is_enforced(bad_result, error):
    def malformed(G, **_kwargs):
        return bad_result

    with pytest.raises(error):
        gg.curvature(nx.complete_graph(3), method=malformed)


def test_callable_entry_point_rejects_multigraph_before_plugin_runs():
    called = False

    def plugin(G, **_kwargs):
        nonlocal called
        called = True
        return {}

    with pytest.raises(gg.CurvatureDomainError, match="does not support multigraphs"):
        gg.curvature(nx.MultiGraph([(0, 1)]), method=plugin)
    assert not called


def test_recalculation_clears_curvature_from_newly_isolated_nodes():
    G = nx.path_graph(3)
    gg.curvature(G, method="forman_node_weighted")
    G.remove_edge(1, 2)
    gg.curvature(G, method="forman_node_weighted")
    assert "ricciCurvature" not in G.nodes[2]


# ── kernel / beta resolvers ─────────────────────────────────────────────────


def test_resolve_beta_variants():
    assert kernels.resolve_beta(0.7)(None, None) == 0.7
    assert isinstance(kernels.resolve_beta("adaptive"), kernels.BetaDegreeProportional)
    bd = kernels.resolve_beta({"x": 0.9})
    assert bd("x", None) == 0.9
    assert bd("missing", None) == 0.5  # default


def test_resolve_beta_degree_proportional():
    G = nx.DiGraph([(0, 1), (0, 2), (1, 0)])  # node 0: out 2, in 1 -> 2/3
    beta = kernels.BetaDegreeProportional()
    assert beta(0, G) == pytest.approx(2 / 3)


def test_resolve_beta_rejects_bool_and_unknown():
    with pytest.raises(TypeError):
        kernels.resolve_beta(True)
    with pytest.raises(ValueError):
        kernels.resolve_beta("bogus")


# ── beta range validation ───────────────────────────────────────────────────
#
# P = beta*P_out + (1-beta)*P_in is a convex combination, so beta outside
# [0, 1] yields NEGATIVE mass that still sums to 1 (beta=-0.5 -> [1.25, -0.25])
# and therefore passes a mass check on its way into the OT solver.


@pytest.mark.parametrize("bad", [1.5, -0.5, 42.0, -1e-9, 1.0000001])
def test_beta_outside_unit_interval_is_rejected(bad):
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        kernels.resolve_beta(bad)


@pytest.mark.parametrize("ok", [0.0, 0.5, 1.0, 0, 1])
def test_beta_endpoints_are_allowed(ok):
    assert kernels.resolve_beta(ok)(None, None) == pytest.approx(float(ok))


def test_beta_dict_validates_values_and_default():
    with pytest.raises(ValueError, match="node 'x'"):
        kernels.resolve_beta({"x": 1.5})
    with pytest.raises(ValueError, match="default"):
        kernels.BetaDict({"x": 0.5}, default=-0.2)


def test_beta_validation_matches_alpha_validation():
    from graph_geometry.curvature.ot_ricci import OllivierDistribution

    for factory in (lambda: kernels.resolve_beta(1.5), lambda: OllivierDistribution(1.5)):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
            factory()


@pytest.mark.parametrize("bad", [2.0, -0.1, True, float("nan"), float("inf")])
def test_callable_beta_is_range_checked_when_evaluated(bad):
    G = nx.DiGraph([(0, 1), (1, 0)])
    with pytest.raises((TypeError, ValueError), match="beta returned for node"):
        gg.curvature(
            G,
            method="ollivier",
            kernel="mixed",
            beta=lambda _node, _graph: bad,
        )


@pytest.mark.parametrize("bad", [True, False, np.bool_(True)])
def test_alpha_rejects_boolean_values(bad):
    from graph_geometry.curvature.ot_ricci import OllivierDistribution

    with pytest.raises(TypeError, match="alpha must be a number"):
        OllivierDistribution(bad)


# ── BetaWeightProportional ──────────────────────────────────────────────────


def test_beta_weight_proportional_uses_weight_not_degree():
    # equal in/out degree, lopsided weights -> degree says 0.5, weight says 0.9
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=9.0)
    G.add_edge(2, 0, weight=1.0)
    assert kernels.BetaDegreeProportional()(0, G) == pytest.approx(0.5)
    assert kernels.BetaWeightProportional()(0, G) == pytest.approx(0.9)


def test_beta_weight_proportional_endpoints_and_isolated():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=2.0)          # 0 is a pure source, 1 a pure sink
    G.add_node("alone")
    beta = kernels.BetaWeightProportional()
    assert beta(0, G) == pytest.approx(1.0)
    assert beta(1, G) == pytest.approx(0.0)
    assert beta("alone", G) == pytest.approx(0.5)


def test_beta_weight_proportional_stays_in_unit_interval():
    G = nx.gnp_random_graph(25, 0.3, seed=7, directed=True)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 10.0 ** (i % 7 - 3)   # weights spanning 1e-3..1e3
    beta = kernels.BetaWeightProportional()
    assert all(0.0 <= beta(n, G) <= 1.0 for n in G.nodes())


def test_beta_weight_proportional_by_name():
    assert isinstance(
        kernels.resolve_beta("weight_proportional"), kernels.BetaWeightProportional
    )


# ── BetaNodeAttr ────────────────────────────────────────────────────────────


def test_beta_node_attr_reads_the_graph():
    G = nx.DiGraph([(0, 1), (1, 0)])
    G.nodes[0]["beta"] = 0.9
    beta = kernels.BetaNodeAttr()
    assert beta(0, G) == pytest.approx(0.9)
    assert beta(1, G) == pytest.approx(0.5)      # missing -> default


def test_beta_node_attr_validates_and_names_the_node():
    G = nx.DiGraph([(0, 1), (1, 0)])
    G.nodes[0]["beta"] = 7.0
    with pytest.raises(ValueError, match="node 0 attribute 'beta'"):
        kernels.BetaNodeAttr()(0, G)


def test_beta_node_attr_from_edgelist_N_lines(tmp_path):
    """The `N <node> <value>` edge-list line can now drive beta."""
    path = tmp_path / "g.edgelist"
    path.write_text("0 1\n1 0\n1 2\n2 1\nN 0 0.9\nN 1 0.1\n")

    G = gg.load.from_edgelist(str(path), directed=True, node_attr="beta")
    beta = kernels.BetaNodeAttr()
    assert beta("0", G) == pytest.approx(0.9)
    assert beta("1", G) == pytest.approx(0.1)
    assert beta("2", G) == pytest.approx(0.5)    # no N line -> default


def test_beta_node_attr_survives_gexf_round_trip(tmp_path):
    G = nx.DiGraph([("a b", "c:d"), ("c:d", "a b")])   # names GEXF must preserve
    G.nodes["a b"]["beta"] = 0.75
    path = tmp_path / "g.gexf"
    nx.write_gexf(G, path)

    H = nx.read_gexf(path)
    assert kernels.BetaNodeAttr()("a b", H) == pytest.approx(0.75)


def test_beta_node_attr_drives_curvature_end_to_end():
    # Asymmetric and strongly connected (the examples/balancing_factor.py graph).
    # A fully bidirectional graph with unit weights has P_out == P_in, so beta
    # provably cannot change anything there -- it makes a useless fixture.
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0), (1, 3)])
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    G.nodes[0]["beta"] = 0.95

    from_attr = gg.curvature(G.copy(), method="lin_lu_yau", kernel="mixed",
                             beta=kernels.BetaNodeAttr())
    uniform = gg.curvature(G.copy(), method="lin_lu_yau", kernel="mixed", beta=0.5)
    equivalent = gg.curvature(G.copy(), method="lin_lu_yau", kernel="mixed",
                              beta={0: 0.95})

    assert any(abs(from_attr[e] - uniform[e]) > 1e-9 for e in from_attr)
    assert all(from_attr[e] == pytest.approx(equivalent[e]) for e in from_attr)


def test_a_legacy_per_node_beta_mapping_still_computes_through_the_frozen_record():
    """The record now holds a FrozenMapping, which the kernel must accept as a map."""
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0)])
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    uniform = gg.curvature(G.copy(), method="ollivier", kernel="mixed", beta=0.3)
    as_map = gg.curvature(G.copy(), method="ollivier", kernel="mixed",
                          beta={n: 0.3 for n in G})
    assert as_map == uniform
    mixed = {0: 0.1, 1: 0.9, 2: 0.5, 3: 0.3}
    by_map = gg.curvature(G.copy(), method="ollivier", kernel="mixed", beta=mixed)
    by_callable = gg.curvature(G.copy(), method="ollivier", kernel="mixed",
                               beta=lambda node, _graph: mixed[node])
    assert by_map == by_callable
    by_map_parallel = gg.curvature(G.copy(), method="ollivier", kernel="mixed",
                                   beta=mixed, proc=2)
    assert by_map_parallel == pytest.approx(by_map, abs=1e-12)


def test_resolve_beta_accepts_any_mapping():
    from graph_geometry.curvature.model import FrozenMapping

    strategy = kernels.resolve_beta(FrozenMapping({0: 0.25}))
    assert isinstance(strategy, kernels.BetaDict)
    assert strategy(0, None) == 0.25 and strategy(7, None) == 0.5


def test_new_beta_strategies_are_picklable():
    import pickle

    for beta in ("weight_proportional", "node_attr"):
        cfg = gg.lin_lu_yau(kernel="mixed", beta=beta)
        assert pickle.loads(pickle.dumps(cfg)).name == "Lin-Lu-Yau"


def test_resolve_kernel_names_and_unknown():
    for name in ("auto", "undirected", "out", "in", "mixed"):
        assert kernels.resolve_kernel(name) is not None
    with pytest.raises(ValueError):
        kernels.resolve_kernel("bogus")


def test_kernel_classes_are_picklable():
    import pickle

    cfg = gg.lin_lu_yau(kernel="mixed", beta="adaptive")
    restored = pickle.loads(pickle.dumps(cfg))
    assert restored.name == "Lin-Lu-Yau"


def test_out_kernel_probabilities_sum_to_one_on_directed():
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    k = kernels.OutKernel()
    neighbors = {n: sorted(set(G.predecessors(n)) | set(G.successors(n))) for n in G.nodes()}
    probs = k(0, G, neighbors, "weight")  # node 0 has out-neighbors 1,2
    assert probs.sum() == pytest.approx(1.0)


# ── MixedKernel mass conservation at sinks / sources ────────────────────────
#
# P = beta*P_out + (1-beta)*P_in only has mass 1 when both sides exist. At a
# sink the beta*P_out term is absent (raw mix sums to 1-beta), at a source the
# (1-beta)*P_in term is (raw mix sums to beta). MixedKernel clamps beta in
# those cases. Graphs where every node has both directions are unaffected —
# which is why the parity goldens never caught this.


def _neighbors(G):
    return {n: sorted(set(G.predecessors(n)) | set(G.successors(n))) for n in G.nodes()}


def _unit_weights(G):
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


@pytest.mark.parametrize("beta", [0.0, 0.2, 0.5, 0.8, 1.0, "degree_proportional"])
def test_mixed_kernel_sums_to_one_with_sinks_and_sources(beta):
    # 0 is a pure source, 3 and 4 are pure sinks, 1 and 2 have both directions.
    G = _unit_weights(nx.DiGraph([(0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 4)]))
    k = kernels.MixedKernel(beta)
    nbrs = _neighbors(G)
    for n in G.nodes():
        assert k(n, G, nbrs, "weight").sum() == pytest.approx(1.0), f"node {n}"


def test_mixed_kernel_collapses_to_the_available_side():
    G = _unit_weights(nx.DiGraph([(0, 1), (1, 2)]))
    nbrs = _neighbors(G)
    k = kernels.MixedKernel(0.8)
    # node 0 is a source -> pure out-walk, all mass on its single successor
    assert k(0, G, nbrs, "weight") == pytest.approx([1.0])
    # node 2 is a sink -> pure in-walk, all mass on its single predecessor
    assert k(2, G, nbrs, "weight") == pytest.approx([1.0])


def test_mixed_kernel_unchanged_when_both_directions_exist():
    # node 0 has out {1, 2} and in {1}: beta must still apply verbatim
    G = _unit_weights(nx.DiGraph([(0, 1), (0, 2), (1, 0)]))
    nbrs = _neighbors(G)
    probs = kernels.MixedKernel(0.8)(0, G, nbrs, "weight")
    # neighbors[0] == [1, 2]; P(1) = .8*.5 + .2*1, P(2) = .8*.5
    assert probs == pytest.approx([0.8 * 0.5 + 0.2, 0.8 * 0.5])
    assert probs.sum() == pytest.approx(1.0)


def test_auto_kernel_inherits_the_clamp_on_directed():
    G = _unit_weights(nx.DiGraph([(0, 1), (1, 2)]))
    k = kernels.AutoKernel(0.3)
    nbrs = _neighbors(G)
    for n in G.nodes():
        assert k(n, G, nbrs, "weight").sum() == pytest.approx(1.0)


def test_isolated_node_still_returns_unit_mass():
    G = nx.DiGraph()
    G.add_node("lonely")
    k = kernels.MixedKernel(0.8)
    assert k("lonely", G, {"lonely": []}, "weight") == pytest.approx([1.0])


# ── kernel domains ───────────────────────────────────────────────────────────
# The domain of an OT curvature belongs to its *configuration*, not its name:
# ollivier accepts either graph class with kernel="auto" and only directed
# graphs with kernel="out". These used to surface as leaked internals --
# AttributeError('Graph' object has no attribute 'successors') for a directed
# kernel on an undirected graph, and a bare KeyError for the reverse.


def _unit(G):
    for u, v in G.edges():
        G[u][v].update(weight=1.0, distance=1.0)
    return G


@pytest.mark.parametrize("kernel", ["out", "in", "mixed"])
def test_directed_kernel_on_undirected_graph_is_a_domain_error(kernel):
    G = _unit(nx.cycle_graph(5))
    with pytest.raises(ValueError, match="defined on directed graphs only"):
        gg.curvature(G, method="ollivier", kernel=kernel)


def test_undirected_kernel_on_directed_graph_is_a_domain_error():
    D = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
    with pytest.raises(ValueError, match="defined on undirected graphs only"):
        gg.curvature(D, method="ollivier", kernel="undirected")


@pytest.mark.parametrize("graph", [nx.cycle_graph(5),
                                   nx.DiGraph([(0, 1), (1, 2), (2, 0)])])
def test_auto_kernel_accepts_either_graph_class(graph):
    assert gg.curvature(_unit(graph), method="ollivier", kernel="auto")


def test_kernel_domains_are_declared_not_inferred():
    from graph_geometry.curvature import kernels

    assert kernels.kernel_domain(kernels.OutKernel()) == "directed"
    assert kernels.kernel_domain(kernels.InKernel()) == "directed"
    assert kernels.kernel_domain(kernels.UndirectedKernel()) == "undirected"
    assert kernels.kernel_domain(kernels.AutoKernel()) == "any"
    # a user-supplied callable declares nothing and is never rejected for it
    assert kernels.kernel_domain(lambda *a, **k: None) == "any"
