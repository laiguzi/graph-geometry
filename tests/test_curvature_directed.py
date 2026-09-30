"""Eidi-Jost in-out directed curvature.

The value of this definition is that it needs no connectivity assumption, so
the tests are built from the paper's own analytic results rather than from a
golden dump: Eidi & Jost (2020), *Ollivier Ricci curvature of directed
hypergraphs*, Sci. Reports 10, 12466 (arXiv:1907.04727), specialised to graphs
(n = m = 1).
"""

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.curvature import directed


def mk(edges, **attrs):
    G = nx.DiGraph()
    for u, v in edges:
        G.add_edge(u, v, weight=attrs.get("weight", 1.0), distance=attrs.get("distance", 1.0))
    return G


def kappa(G):
    k, _ = directed.directed_curvature_profile(G)
    return k


# ── the paper's analytic values ─────────────────────────────────────────────


def test_theorem_6_8_source_sink_bipartition_is_ricci_flat():
    """Thm 6.8: vertices split into pure sources A and pure sinks B => kappa = 0."""
    G = mk([("a1", "b1"), ("a1", "b2"), ("a2", "b1"), ("a2", "b2"), ("a2", "b3")])
    assert all(v == pytest.approx(0.0, abs=1e-9) for v in kappa(G).values())


def test_theorem_6_4_three_block_cycle_is_ricci_one():
    """Thm 6.4: A->B->C->A with all cross edges present => kappa = 1."""
    A, B, C = ["a1", "a2"], ["b1", "b2"], ["c1", "c2"]
    edges = ([(u, v) for u in A for v in B]
             + [(u, v) for u in B for v in C]
             + [(u, v) for u in C for v in A])
    assert all(v == pytest.approx(1.0, abs=1e-9) for v in kappa(mk(edges)).values())


def test_theorem_6_13_four_block_chain_is_ricci_minus_two():
    """Thm 6.13: A->B->C->D fully connected => the middle edges have kappa = -2."""
    A, B, C, D = ["a"], ["b1", "b2"], ["c1", "c2"], ["d"]
    edges = ([(u, v) for u in A for v in B]
             + [(u, v) for u in B for v in C]
             + [(u, v) for u in C for v in D])
    k = kappa(mk(edges))
    middle = [k[(u, v)] for u, v in k if u in B and v in C]
    assert middle and all(v == pytest.approx(-2.0, abs=1e-9) for v in middle)


@pytest.mark.parametrize(
    "edges, edge, expected",
    [
        # Thm 4.8 for graphs: kappa = -2 + k + k', k = [x is a source], k' = [y is a sink]
        ([("s", "x"), ("x", "y"), ("y", "t")], ("x", "y"), -2.0),   # k=0, k'=0
        ([("x", "y"), ("y", "t")], ("x", "y"), -1.0),               # k=1, k'=0
        ([("s", "x"), ("x", "y")], ("x", "y"), -1.0),               # k=0, k'=1
        ([("x", "y")], ("x", "y"), 0.0),                            # k=1, k'=1
    ],
)
def test_theorem_4_8_directed_tree_values(edges, edge, expected):
    assert kappa(mk(edges))[edge] == pytest.approx(expected, abs=1e-9)


# ── the property that motivates the definition ──────────────────────────────


def test_works_on_a_graph_that_is_not_strongly_connected():
    """The whole point: hubs fanning out to pure sinks, no strong connectivity."""
    hubs = ["h1", "h2", "h3"]
    sinks = [f"s{i}" for i in range(6)]
    edges = ([(a, b) for a in hubs for b in hubs if a != b]      # strongly connected core
             + [(h, s) for h in hubs for s in sinks])            # one-way fan-out
    G = mk(edges)
    assert not nx.is_strongly_connected(G)
    assert sum(1 for n in G if G.out_degree(n) == 0) == len(sinks)

    k = kappa(G)
    assert len(k) == G.number_of_edges()
    assert all(-2.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in k.values())


def test_lin_lu_yau_fails_where_eidi_jost_succeeds():
    """Contrast: the same graph is genuinely undefined for the OT engine."""
    G = mk([("h1", "h2"), ("h2", "h1"), ("h1", "s"), ("h2", "s")])
    with pytest.raises(ValueError, match="strongly.*connected"):
        gg.curvature(G.copy(), method="lin_lu_yau")
    assert len(gg.curvature(G.copy(), method="eidi_jost")) == G.number_of_edges()


def test_transport_distance_never_exceeds_three():
    """d(u, v) <= 3 is structural: u -> x -> y -> v always exists."""
    G = mk([("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("b", "e"), ("c", "f")])
    _, profile = directed.directed_curvature_profile(G)
    assert profile
    for edge, buckets in profile.items():
        assert max(buckets) <= 3.0 + 1e-9, f"{edge}: {buckets}"


def test_kappa_equals_mu0_minus_mu2_minus_2mu3():
    """The paper's combinatorial form of Eidi-Jost curvature.

    Because ``d <= 3`` is structural and the distances here are unit, the
    transport mass falls in buckets 0..3 and

        kappa = mu_0 - mu_2 - 2 mu_3

    (mu_1 cancels: mass moved one step is curvature-neutral). This is the
    identity quoted in the software paper's validation table, checked here
    rather than only in an application write-up.
    """
    G = nx.gnp_random_graph(14, 0.3, seed=5, directed=True)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0

    kappa_map, profile = directed.directed_curvature_profile(G)
    assert len(kappa_map) > 20                      # a non-trivial sample
    for edge, buckets in profile.items():
        assert set(buckets) <= {0.0, 1.0, 2.0, 3.0}, buckets
        combinatorial = (buckets.get(0.0, 0.0)
                         - buckets.get(2.0, 0.0)
                         - 2.0 * buckets.get(3.0, 0.0))
        assert kappa_map[edge] == pytest.approx(combinatorial, abs=1e-12), edge


def test_curvature_is_bounded_in_minus_two_one():
    G = nx.gnp_random_graph(20, 0.2, seed=3, directed=True)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    assert all(-2.0 - 1e-9 <= v <= 1.0 + 1e-9 for v in kappa(G).values())


# ── measures, weighting, and the result contract ────────────────────────────


def test_measure_puts_all_mass_on_the_node_at_a_sink_or_source():
    G = mk([("x", "y")])
    assert directed.directed_in_out_measure(G, "x", "in") == {"x": 1.0}     # source
    assert directed.directed_in_out_measure(G, "y", "out") == {"y": 1.0}    # sink


def test_measure_is_weight_proportional():
    G = nx.DiGraph()
    G.add_edge("a", "x", weight=3.0)
    G.add_edge("b", "x", weight=1.0)
    mu = directed.directed_in_out_measure(G, "x", "in")
    assert mu == pytest.approx({"a": 0.75, "b": 0.25})


def test_measure_rejects_bad_direction():
    with pytest.raises(ValueError, match="direction"):
        directed.directed_in_out_measure(mk([("x", "y")]), "x", "sideways")


def test_registered_and_honours_the_result_contract():
    assert "eidi_jost" in gg.CURVATURE_REGISTRY
    G = mk([("a", "b"), ("b", "c"), ("c", "a")])
    k = gg.curvature(G, method="eidi_jost")
    assert set(k) == set(G.edges())
    for u, v in G.edges():
        assert G[u][v]["ricciCurvature"] == pytest.approx(k[(u, v)])
    for n in G.nodes():
        assert "ricciCurvature" in G.nodes[n]


def test_undirected_input_raises_a_clear_error():
    G = nx.Graph([(0, 1), (1, 2)])
    with pytest.raises(gg.CurvatureDomainError, match="directed graphs only"):
        gg.curvature(G, method="eidi_jost")


# ── signed graphs ───────────────────────────────────────────────────────────
#
# OT is defined between probability measures. A negative weight gives negative
# mass, and the result can still sum to 1 (weights 3 and -1 -> {a: 1.5,
# b: -0.5}), so it passes a mass check and reaches the solver as a signed
# measure. Reject it instead.


def test_negative_weights_are_rejected():
    G = nx.DiGraph()
    G.add_edge("a", "x", weight=3.0, distance=1.0)
    G.add_edge("b", "x", weight=-1.0, distance=1.0)
    with pytest.raises(ValueError, match="negative 'weight'"):
        directed.directed_in_out_measure(G, "x", "in")


def test_negative_weights_would_otherwise_give_a_signed_measure():
    """Pin the exact failure mode the guard prevents."""
    G = nx.DiGraph()
    G.add_edge("a", "x", weight=3.0)
    G.add_edge("b", "x", weight=-1.0)
    raw = {"a": 3.0 / 2.0, "b": -1.0 / 2.0}
    assert sum(raw.values()) == pytest.approx(1.0)   # sums to 1 ...
    assert min(raw.values()) < 0                     # ... but is not a measure


def test_all_zero_weights_fall_back_to_uniform():
    G = nx.DiGraph()
    G.add_edge("a", "x", weight=0.0)
    G.add_edge("b", "x", weight=0.0)
    assert directed.directed_in_out_measure(G, "x", "in") == pytest.approx({"a": 0.5, "b": 0.5})


def signed(edges):
    G = nx.DiGraph()
    for u, v, s in edges:
        G.add_edge(u, v, weight=1.0, distance=1.0, sign=s)
    return G


P, N = "promotion", "inhibition"


@pytest.mark.parametrize(
    "edges, balanced",
    [
        ([("a", "b", P), ("b", "c", P), ("c", "a", P)], True),    # no negatives
        ([("a", "b", N), ("b", "c", N), ("c", "a", P)], True),    # even # negatives
        ([("a", "b", P), ("b", "c", P), ("c", "a", N)], False),   # odd  # negatives
        ([("a", "b", N), ("b", "c", N), ("c", "a", N)], False),
    ],
)
def test_double_cover_detects_structural_balance(edges, balanced):
    """Zaslavsky: balanced iff the cover splits into two copies of the base."""
    C = directed.signed_double_cover(signed(edges))
    components = nx.number_connected_components(C.to_undirected())
    assert components == (2 if balanced else 1)


def test_double_cover_doubles_nodes_and_edges_with_absolute_weights():
    G = signed([("a", "b", P), ("b", "c", N)])
    G["b"]["c"]["weight"] = -4.0          # sign may also be carried in the weight
    C = directed.signed_double_cover(G)
    assert C.number_of_nodes() == 2 * G.number_of_nodes()
    assert C.number_of_edges() == 2 * G.number_of_edges()
    assert all(d["weight"] >= 0 for _, _, d in C.edges(data=True))


def test_double_cover_lifts_have_equal_curvature():
    """The sheet swap (v, s) -> (v, -s) is always an automorphism.

    An edge (u,s)->(v,t) exists iff t = s*sgn(e); applying the swap gives
    (u,-s)->(v,-t) with -t = (-s)*sgn(e), which is again an edge. So any
    isomorphism-invariant curvature takes the same value on both lifts of a
    base edge — the second lift carries no extra information.
    """
    import collections

    rnd = __import__("random").Random(0)
    G0 = nx.gnp_random_graph(10, 0.3, seed=1, directed=True)
    G = nx.DiGraph()
    for u, v in G0.edges():
        G.add_edge(u, v, weight=rnd.uniform(0.2, 3.0), distance=1.0,
                   sign=N if rnd.random() < 0.4 else P)

    C = directed.signed_double_cover(G)
    kappa, _ = directed.directed_curvature_profile(C)
    by_base = collections.defaultdict(list)
    for (a, b), k in kappa.items():
        by_base[C[a][b]["base_edge"]].append(k)

    assert by_base
    for base, lifts in by_base.items():
        assert len(lifts) == 2, base
        assert lifts[0] == pytest.approx(lifts[1], abs=1e-12), base


def test_double_cover_curvature_differs_from_ignoring_the_sign():
    """The cover is not the same as computing on |w| — sign enters the geometry."""
    import statistics as st

    G = signed([("a", "b", P), ("b", "c", N), ("c", "a", P),
                ("a", "c", N), ("c", "b", P), ("b", "a", N)])
    base, _ = directed.directed_curvature_profile(G)
    cover, _ = directed.directed_curvature_profile(directed.signed_double_cover(G))
    assert st.median(base.values()) != pytest.approx(st.median(cover.values()))


def test_double_cover_rejects_undirected():
    with pytest.raises(ValueError, match="directed"):
        directed.signed_double_cover(nx.Graph([(0, 1)]))


# ── the metric under Eidi-Jost ───────────────────────────────────────────────
# Regression: kappa was 1 - W1 with no division by d(x,y), so a non-unit
# distance rescaled the curvature instead of leaving it fixed. On C4 with every
# distance d it read 1 - 3d, giving -29 at d = 10 and breaking the [-2, 1]
# bound. See RicciFlow-Computation.md, "Eidi-Jost normalises by d(x,y)".


def _uniform_cycle(n, dist=1.0, weight=1.0):
    G = nx.DiGraph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n, weight=weight, distance=dist)
    return G


@pytest.mark.parametrize("n,expected", [(3, 1.0), (4, -2.0), (5, -2.0), (7, -2.0)])
def test_unit_metric_results_are_unchanged(n, expected):
    """The default metric must still reproduce the paper's kappa = 1 - W1."""
    kappa = gg.curvature(_uniform_cycle(n), method="eidi_jost")
    assert all(v == pytest.approx(expected, abs=1e-12) for v in kappa.values())


@pytest.mark.parametrize("scale", [0.5, 2.0, 10.0, 137.0])
def test_uniform_metric_scaling_leaves_curvature_unchanged(scale):
    """Scale invariance: kappa = 1 - W1/d(x,y) is homogeneous of degree 0."""
    base = gg.curvature(_uniform_cycle(5), method="eidi_jost")
    scaled = gg.curvature(_uniform_cycle(5, dist=scale), method="eidi_jost")
    for edge, value in base.items():
        assert scaled[edge] == pytest.approx(value, abs=1e-12)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 8])
@pytest.mark.parametrize("scale", [1.0, 4.0])
def test_bound_holds_for_any_uniform_metric(n, scale):
    """kappa in [-2, 1] is a hop-metric result; it survives uniform rescaling."""
    kappa = gg.curvature(_uniform_cycle(n, dist=scale), method="eidi_jost")
    assert all(-2.0 - 1e-12 <= v <= 1.0 + 1e-12 for v in kappa.values())


def test_non_uniform_metric_may_leave_the_bound_and_that_is_documented():
    """A short edge among long routes legitimately falls below -2.

    The [-2, 1] bound comes from d(u,v) <= 3 *hops*. With a non-uniform metric
    the floor becomes 1 - 3*max_route_length/d(x,y), so this is a property of
    the generalisation, not a regression. Pinned so the docstring stays honest.
    """
    G = nx.DiGraph()
    for i in range(4):
        G.add_edge(i, (i + 1) % 4, weight=1.0, distance=100.0)
    G[0][1]["distance"] = 0.01
    kappa = gg.curvature(G, method="eidi_jost")
    assert kappa[(0, 1)] < -2.0


# ── sign labels are validated before a signed construction runs ──────────────
# The rule "positive when the label equals `positive`, negative otherwise" fails
# silently in both directions: a missing attribute reads as positive, and a
# vocabulary mismatch reads as negative. See validate_signs.


def test_double_cover_rejects_an_unsigned_graph():
    G = nx.DiGraph([("a", "b"), ("b", "c")])
    with pytest.raises(ValueError, match="no edge carries"):
        directed.signed_double_cover(G)


def test_double_cover_rejects_a_partly_signed_graph():
    G = signed([("a", "b", P), ("b", "c", N)])
    G.add_edge("c", "a")                      # no sign: would read as positive
    with pytest.raises(ValueError, match="have no .* attribute"):
        directed.signed_double_cover(G)


def test_double_cover_rejects_a_vocabulary_mismatch():
    """Both labels reading as negative loses the distinction the data draws."""
    G = signed([("a", "b", P), ("b", "c", N)])
    with pytest.raises(ValueError, match="neither"):
        directed.signed_double_cover(G, positive="activation")


def test_double_cover_accepts_a_single_label():
    """All-positive and all-negative are legitimate signed graphs, not typos."""
    for label in (P, N):
        G = signed([("a", "b", label), ("b", "c", label)])
        cover = directed.signed_double_cover(G)
        assert cover.number_of_edges() == 2 * G.number_of_edges()


def test_more_than_two_sign_labels_is_rejected():
    G = signed([("a", "b", P), ("b", "c", N)])
    G.add_edge("c", "a", sign="unknown", weight=1.0, distance=1.0)
    with pytest.raises(ValueError, match="binary"):
        directed.signed_double_cover(G)


def test_signed_double_cover_is_exported_at_top_level():
    """The paper cites it in Limitations and uses it in the idopNetwork run."""
    import graph_geometry as gg

    assert gg.signed_double_cover is directed.signed_double_cover
