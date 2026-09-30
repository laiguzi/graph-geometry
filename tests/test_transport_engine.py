"""The shared endpoint-pair OT engine behind Ollivier, Lin-Lu-Yau and Eidi-Jost.

endpoint-pair measures -> ordered ground cost -> solver -> normalisation. A
measure owns its support and mass together, and Eidi-Jost reads different
neighbourhoods at its two endpoints through the same orchestration.
"""

import math

import networkx as nx
import numpy as np
import pytest

import graph_geometry as gg
from graph_geometry.curvature import directed
from graph_geometry.curvature.base import TransportCurvatureEngine
from graph_geometry.curvature.directed import eidi_jost_ot_config
from graph_geometry.curvature.ot_ricci import lin_lu_yau_ot_config, ollivier_ot_config
from graph_geometry.curvature.transport import (
    DirectionalKernel,
    DiscreteMeasure,
    EMDTransportSolver,
    InOutPairMeasureBuilder,
    KernelPairMeasureBuilder,
    NoIdlenessDistribution,
    NodeKernel,
    OTCurvatureConfig,
)


def _unit(G):
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    return G


def _weighted_digraph():
    G = nx.DiGraph()
    edges = [(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0), (1, 3), (3, 1), (4, 0), (2, 5)]
    for i, (u, v) in enumerate(edges):
        G.add_edge(u, v, weight=0.5 + i % 3, distance=1.0 + (i % 2) * 0.5)
    return G


# ── measures ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "support, mass, match",
    [
        ((0, 1), [0.5], "aligned"),
        ((), [], "non-empty"),
        ((0, 0), [0.5, 0.5], "repeats"),
        ((0, 1), [0.7, 0.7], "total 1"),
        ((0, 1), [1.5, -0.5], "non-negative"),
        ((0, 1), [np.nan, 1.0], "finite"),
        ((0,), [[1.0]], "one-dimensional"),
    ],
)
def test_a_measure_validates_support_and_mass_together(support, mass, match):
    with pytest.raises(gg.CurvatureContractError, match=match):
        DiscreteMeasure(support, np.array(mass))


def test_a_measure_is_immutable():
    mu = DiscreteMeasure((0, 1), np.array([0.25, 0.75]))
    with pytest.raises(ValueError):
        mu.mass[0] = 1.0
    assert DiscreteMeasure.dirac("x").as_dict() == {"x": 1.0}


def test_kernel_measures_put_the_endpoint_last():
    G = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    builder = KernelPairMeasureBuilder(*_ollivier_slots(G))
    mu, nu = builder(G, 0, 2, semantics=gg.GraphSemantics())
    assert mu.support[-1] == 0 and nu.support[-1] == 2
    assert mu.support[:-1] == (1, 2)                    # sorted union neighbourhood
    assert mu.mass.sum() == pytest.approx(1.0)


def _ollivier_slots(G):
    config = gg.ollivier(alpha=0.5, kernel="mixed")
    return config.transition_kernel, config.distribution


def test_in_out_measures_read_different_neighbourhoods_at_the_two_endpoints():
    G = _unit(nx.DiGraph([("a", "x"), ("b", "x"), ("x", "y"), ("y", "c")]))
    mu, nu = InOutPairMeasureBuilder()(G, "x", "y", semantics=gg.GraphSemantics())
    assert mu.as_dict() == {"a": 0.5, "b": 0.5}         # in-neighbours of x
    assert nu.as_dict() == {"c": 1.0}                    # out-neighbours of y
    source, sink = InOutPairMeasureBuilder()(G, "a", "c", semantics=gg.GraphSemantics())
    assert source.as_dict() == {"a": 1.0} and sink.as_dict() == {"c": 1.0}   # Dirac fallback


def test_an_empty_selected_neighbourhood_falls_back_to_the_dirac_measure():
    """Out at a sink / in at a source: mu_z = delta_z, for every distribution.

    The out kernel used to return ``[1.0]`` there, which read as 'all mass on
    the first neighbour' whenever the sink had exactly one neighbour.
    """
    G = _unit(nx.DiGraph([(0, 1), (1, 0), (1, 2)]))      # 2 is a sink with one neighbour
    semantics = gg.GraphSemantics()
    for config in (gg.lin_lu_yau(kernel="out"), gg.ollivier(alpha=0.3, kernel="out")):
        builder = KernelPairMeasureBuilder(config.transition_kernel, config.distribution)
        _, nu = builder(G, 1, 2, semantics=semantics)
        assert nu.as_dict() == {2: 1.0}
    source, _ = KernelPairMeasureBuilder(
        gg.ollivier(kernel="in").transition_kernel, gg.ollivier().distribution
    )(G, 2, 0, semantics=semantics)
    assert source.support[-1] == 2 and source.mass.sum() == pytest.approx(1.0)
    # mixed collapses to the side that exists (the clamp), not to a Dirac
    mixed = KernelPairMeasureBuilder(gg.ollivier(kernel="mixed").transition_kernel,
                                     gg.ollivier().distribution)
    _, nu = mixed(G, 1, 2, semantics=semantics)
    assert nu.as_dict() == pytest.approx({1: 0.5, 2: 0.5})


def test_lin_lu_yau_with_a_dirac_endpoint_is_the_idleness_limit():
    """J = d(x,y) - sum_z nu_x(z) d(z,y); equals lim kappa_alpha/(1-alpha) of Ollivier."""
    G = _unit(nx.DiGraph([(0, 1), (1, 0), (0, 2), (1, 2)]))   # 2 is a sink
    lly = gg.compute_pair_curvature(
        G, gg.CurvatureRequest("lin_lu_yau", {"kernel": "out"}), [(0, 2), (1, 2)]
    ).values
    # nu_0 = {1: 1/2, 2: 1/2}: 1 - (1/2 * d(1,2) + 1/2 * d(2,2)) / d(0,2) = 1/2
    assert lly[(0, 2)] == pytest.approx(0.5, abs=1e-9)
    alpha = 0.999
    ollivier = gg.compute_pair_curvature(
        G, gg.CurvatureRequest("ollivier", {"kernel": "out", "alpha": alpha}), [(0, 2)]
    ).values[(0, 2)]
    assert ollivier / (1 - alpha) == pytest.approx(lly[(0, 2)], abs=1e-6)


def test_eidi_jost_is_a_kernel_and_distribution_configuration():
    """In-kernel at the source, out-kernel at the target, no idleness."""
    config = eidi_jost_ot_config()
    assert config.kernel == DirectionalKernel("in")
    assert config.target_kernel == DirectionalKernel("out")
    assert isinstance(config.distribution, NoIdlenessDistribution)
    builder = config.measure_builder
    assert isinstance(builder, KernelPairMeasureBuilder) and not builder.symmetric
    ollivier = ollivier_ot_config(kernel="out").measure_builder
    assert ollivier.symmetric and isinstance(ollivier._source, NodeKernel)


def test_a_transport_config_is_the_four_figure_components():
    """K_c kernel, D_c distribution, S_c solver, F_c curvature -- named as drawn."""
    from graph_geometry.curvature.ot_ricci import LLYCurvature, LLYDistribution, SignedKRTransportSolver

    config = lin_lu_yau_ot_config(kernel="mixed")
    assert isinstance(config.distribution, LLYDistribution)
    assert isinstance(config.solver, SignedKRTransportSolver)
    assert isinstance(config.curvature, LLYCurvature)
    assert config.kernel is not None and config.target_kernel is None
    with pytest.raises(gg.CurvatureConfigurationError, match="missing component"):
        OTCurvatureConfig(name="incomplete", kernel=config.kernel,
                          distribution=config.distribution, solver=config.solver)
    with pytest.raises(gg.CurvatureConfigurationError, match="replaces"):
        OTCurvatureConfig(name="ambiguous", kernel=config.kernel, solver=config.solver,
                          curvature=config.curvature, measures=config.measure_builder)


def test_a_kernel_pair_can_mix_rules_across_endpoints():
    """The generalisation is not Eidi-Jost-specific: any rule at either endpoint."""
    G = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]))
    config = OTCurvatureConfig(
        name="out-in Ollivier",
        kernel=DirectionalKernel("out"),
        target_kernel=DirectionalKernel("in"),
        distribution=NoIdlenessDistribution(),
        solver=EMDTransportSolver(),
        curvature=eidi_jost_ot_config().curvature,
    )
    values = TransportCurvatureEngine(G, config).compute_edges()
    assert set(values) == set(G.edges())


# ── the solver returns a plan only on request ───────────────────────────────


def test_emd_plan_is_computed_only_when_requested():
    a = np.array([0.5, 0.5])
    b = np.array([0.25, 0.75])
    M = np.array([[0.0, 1.0], [1.0, 0.0]])
    without = EMDTransportSolver()(a, b, M)
    with_plan = EMDTransportSolver()(a, b, M, return_plan=True)
    assert without.plan is None
    assert with_plan.plan.shape == (2, 2)
    assert with_plan.cost == pytest.approx(without.cost)
    assert with_plan.plan.sum(axis=1) == pytest.approx(a)


def test_lly_reports_its_objective_without_a_plan():
    G = _unit(nx.cycle_graph(5))
    engine = TransportCurvatureEngine(G, lin_lu_yau_ot_config(kernel="undirected"))
    (_, (_, _, _, result)), = engine.transport_details([(0, 1)]).values()
    assert result.plan is None


# ── one orchestration for the three transport definitions ──────────────────


def test_every_transport_built_in_runs_on_the_shared_engine(monkeypatch):
    calls = []
    original = TransportCurvatureEngine.compute_edges

    def spy(self):
        calls.append(self.config.name)
        return original(self)

    monkeypatch.setattr(TransportCurvatureEngine, "compute_edges", spy)
    G = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 1), (1, 0)]))
    gg.compute_curvature(G, gg.CurvatureRequest("ollivier", {"kernel": "out"}))
    gg.compute_curvature(G, gg.CurvatureRequest("lin_lu_yau", {"kernel": "in"}))
    gg.compute_curvature(G, "eidi_jost")
    assert calls == ["Ollivier(alpha=0.5)", "Lin-Lu-Yau", "Eidi-Jost"]


@pytest.mark.parametrize("kernel", ["out", "in", "mixed"])
def test_legacy_config_and_endpoint_pair_config_agree_exactly(kernel):
    G = nx.DiGraph(_weighted_digraph().subgraph([0, 1, 2, 3]))
    legacy = gg.RicciCurvature(G, gg.ollivier(alpha=0.3, kernel=kernel)).compute_edge_values()
    pair_form = TransportCurvatureEngine(
        G, ollivier_ot_config(alpha=0.3, kernel=kernel)
    ).compute_edges()
    assert legacy == pair_form


def test_eidi_jost_parallel_output_is_identical_to_serial():
    G = nx.gnp_random_graph(24, 0.15, seed=11, directed=True)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 0.5 + i % 4
        G[u][v]["distance"] = 1.0 + (i % 3) * 0.5
    serial = gg.compute_curvature(G, "eidi_jost", proc=1).values
    parallel = gg.compute_curvature(G, "eidi_jost", proc=3).values
    assert dict(serial) == dict(parallel)
    assert directed.eidi_jost(G, proc=3) == dict(serial)


def test_the_profile_is_a_view_of_the_same_transport():
    G = _weighted_digraph()
    kappa, profile = directed.directed_curvature_profile(G)
    assert kappa == dict(gg.compute_curvature(G, "eidi_jost").values)
    for edge, buckets in profile.items():
        assert sum(buckets.values()) == pytest.approx(1.0)


def test_eidi_jost_metric_scaling_is_invariant_on_the_shared_engine():
    G = _weighted_digraph()
    H = G.copy()
    for u, v in H.edges():
        H[u][v]["distance"] *= 10.0
    base = gg.compute_curvature(G, "eidi_jost").values
    scaled = gg.compute_curvature(H, "eidi_jost").values
    for edge, value in base.items():
        assert scaled[edge] == pytest.approx(value, abs=1e-12)


def test_an_unreachable_non_edge_pair_is_a_domain_error_not_a_key_error():
    G = _unit(nx.DiGraph([(0, 1), (1, 2)]))
    with pytest.raises(gg.CurvatureDomainError, match="unreachable"):
        directed.directed_curvature_profile(G, pairs=[(2, 0)])


def test_engine_rejects_a_legacy_config_object():
    with pytest.raises(gg.CurvatureConfigurationError, match="OTCurvatureConfig"):
        TransportCurvatureEngine(nx.path_graph(3), gg.ollivier())


def test_eidi_jost_config_requests_the_plan_and_has_no_distance_guard():
    config = eidi_jost_ot_config()
    assert isinstance(config, OTCurvatureConfig)
    assert config.return_plan is True
    assert config.degenerate_distance is None


# ── arbitrary hashable node labels ──────────────────────────────────────────

_MIXED = {0: 0, 1: "a", 2: ("t", 1), 3: 2.5, 4: frozenset({"x"}), 5: "b"}


def _mixed_undirected():
    G = nx.Graph()
    for u, v in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (3, 4), (4, 5), (5, 3)]:
        G.add_edge(u, v, weight=1.0 + (u + v) % 3, distance=1.0)
    return G


def _mixed_directed():
    D = nx.DiGraph()
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 2), (1, 0), (4, 3)]:
        D.add_edge(u, v, weight=1.0 + (u + v) % 3, distance=1.0)
    assert nx.is_strongly_connected(D)
    return D


def _requests(G):
    if G.is_directed():
        return [gg.CurvatureRequest("ollivier", {"alpha": 0.5, "kernel": "mixed"}),
                gg.CurvatureRequest("lin_lu_yau", {"kernel": "out"})]
    return [gg.CurvatureRequest("ollivier", {"alpha": 0.5}),
            gg.CurvatureRequest("lin_lu_yau")]


@pytest.mark.parametrize("make", [_mixed_undirected, _mixed_directed],
                         ids=["undirected", "strongly-connected-digraph"])
def test_mixed_hashable_labels_are_supported(make):
    """Labels 0, "a", ("t", 1), 2.5 and a frozenset have no natural order.

    The support used to be ``sorted`` by label, which raised ``TypeError``.
    """
    G = nx.relabel_nodes(make(), _MIXED)
    for request in _requests(G):
        values = gg.compute_curvature(G, request).values
        assert set(values) == set(G.edges())
        assert all(math.isfinite(v) for v in values.values())


@pytest.mark.parametrize("make", [_mixed_undirected, _mixed_directed],
                         ids=["undirected", "strongly-connected-digraph"])
def test_relabelling_does_not_change_the_numbers(make):
    """The support follows node order, so labels never affect the result.

    Integer labels and mixed labels in the same insertion order give the same
    transport problems, hence bit-identical curvature on every edge.
    """
    G = make()
    H = nx.relabel_nodes(G, _MIXED)
    for request in _requests(G):
        plain = gg.compute_curvature(G, request).values
        mixed = gg.compute_curvature(H, request).values
        assert {(_MIXED[u], _MIXED[v]): k for (u, v), k in plain.items()} == dict(mixed)


def test_support_follows_graph_node_order_not_label_order():
    from graph_geometry.curvature.kernels import UndirectedKernel

    G = nx.Graph()
    G.add_edges_from([("z", "b"), ("z", "a"), ("z", "c")])   # nodes: z, b, a, c
    assert NodeKernel(UndirectedKernel()).prepare(G)["z"] == ["b", "a", "c"]
    D = nx.DiGraph([("z", "b"), ("a", "z"), ("z", "a"), ("c", "z")])  # z, b, a, c
    assert NodeKernel(UndirectedKernel()).prepare(D)["z"] == ["b", "a", "c"]
