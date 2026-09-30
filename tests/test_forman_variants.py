"""Node-weighted Forman variants: closed forms and other libraries' conventions.

The executed cross-library runs live in ``paper/compare_with_libraries.py``;
these are the hand-checkable anchors that keep the formulas pinned without
needing the third-party packages installed. The reference values are the ones
those runs reproduced to machine precision:

- ``node_weight="one"``           -> GraphRicciCurvature ``FormanRicci("1d")``
- augmented, ``node_weight="one"`` -> GraphRicciCurvature ``FormanRicci("augmented")``
  and SCOTT's ``forman_curvature`` on an unweighted graph
- ``node_weight="inverse_degree"`` -> pynetflow's ``Forman_ricci``
"""

import math

import networkx as nx
import pytest

import graph_geometry as gg


def _unit(G):
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    return G


# ── closed forms under unit weights ─────────────────────────────────────────


def test_forman_sreejith_unit_is_four_minus_degrees():
    G = _unit(nx.karate_club_graph())
    r = gg.curvature(G, method="forman_sreejith")
    for u, v in G.edges():
        assert r[(u, v)] == pytest.approx(4 - G.degree(u) - G.degree(v))


def test_augmented_unit_adds_three_per_triangle():
    G = _unit(nx.karate_club_graph())
    r = gg.curvature(G, method="augmented_forman_sreejith")
    for u, v in G.edges():
        tri = len(set(G[u]) & set(G[v]))
        assert r[(u, v)] == pytest.approx(4 - G.degree(u) - G.degree(v) + 3 * tri)


def test_augmented_equals_plain_on_triangle_free_graph():
    G = _unit(nx.cycle_graph(6))
    assert (gg.curvature(G.copy(), method="forman_sreejith")
            == gg.curvature(G.copy(), method="augmented_forman_sreejith"))


def test_weighted_single_edge_by_hand():
    # path 0-1-2 with w(0,1)=4, w(1,2)=9, node weights 1.
    # F(0,1) = w_e*(1/w_e + 1/w_e - 1/sqrt(w_e*9)) = 2 - 4/6 = 4/3
    G = nx.Graph()
    G.add_edge(0, 1, weight=4.0, distance=4.0)
    G.add_edge(1, 2, weight=9.0, distance=9.0)
    r = gg.curvature(G, method="forman_sreejith")
    assert r[(0, 1)] == pytest.approx(2.0 - 4.0 / math.sqrt(4.0 * 9.0))


def test_node_weight_scales_the_whole_side():
    """w_u multiplies its own star sum, not only the leading term."""
    G = _unit(nx.star_graph(4))  # centre 0, leaves 1..4
    base = gg.curvature(G.copy(), method="forman_sreejith", node_weight="one")
    scaled = gg.curvature(G.copy(), method="forman_sreejith", node_weight=2.0)
    for e, value in base.items():
        assert scaled[e] == pytest.approx(2.0 * value)


def test_inverse_degree_scheme_matches_manual_weights():
    G = _unit(nx.karate_club_graph())
    named = gg.curvature(G.copy(), method="forman_sreejith",
                         node_weight="inverse_degree")
    mapping = {n: 1.0 / G.degree(n) for n in G.nodes()}
    explicit = gg.curvature(G.copy(), method="forman_sreejith", node_weight=mapping)
    for e in named:
        assert named[e] == pytest.approx(explicit[e])


def test_node_weight_attr_overrides_scheme():
    G = _unit(nx.path_graph(3))
    nx.set_node_attributes(G, {0: 3.0, 1: 3.0, 2: 3.0}, "w_node")
    from_attr = gg.curvature(G.copy(), method="forman_sreejith",
                             node_weight_attr="w_node")
    from_value = gg.curvature(G.copy(), method="forman_sreejith", node_weight=3.0)
    for e in from_attr:
        assert from_attr[e] == pytest.approx(from_value[e])


# ── contract + error paths ──────────────────────────────────────────────────


@pytest.mark.parametrize("method", ["forman_sreejith", "augmented_forman_sreejith"])
def test_registered_and_honours_result_contract(method):
    assert method in gg.CURVATURE_ALIASES
    assert gg.get_curvature_spec(method).name in gg.CURVATURE_REGISTRY
    G = _unit(nx.karate_club_graph())
    r = gg.curvature(G, method=method)
    assert set(r) == {(u, v) for u, v in G.edges()}
    assert all("ricciCurvature" in G[u][v] for u, v in G.edges())
    for n in G.nodes():
        incident = [G[n][m]["ricciCurvature"] for m in G.neighbors(n)]
        assert G.nodes[n]["ricciCurvature"] == pytest.approx(
            sum(incident) / len(incident)
        )


@pytest.mark.parametrize("method", ["forman_sreejith", "augmented_forman_sreejith"])
def test_directed_input_is_rejected_not_symmetrised(method):
    D = nx.DiGraph()
    D.add_edge(0, 1, weight=1.0, distance=1.0)
    with pytest.raises(ValueError, match="undirected"):
        gg.curvature(D, method=method)


def test_unknown_node_weight_scheme_names_the_known_ones():
    G = _unit(nx.path_graph(3))
    with pytest.raises(ValueError, match="inverse_degree"):
        gg.curvature(G, method="forman_sreejith", node_weight="nope")


def test_node_weight_mapping_must_cover_every_node():
    G = _unit(nx.path_graph(3))
    with pytest.raises(ValueError, match="missing"):
        gg.curvature(G, method="forman_sreejith", node_weight={0: 1.0, 1: 1.0})


def test_face_weight_must_be_positive():
    G = _unit(nx.complete_graph(3))
    with pytest.raises(ValueError, match="face_weight"):
        gg.curvature(G, method="augmented_forman_sreejith", face_weight=0.0)


def test_face_weight_scales_the_triangle_term():
    G = _unit(nx.complete_graph(3))
    # every neighbour closes the triangle, so the star sums vanish and only
    # w_u/w_e + w_v/w_e = 2 and the face term |faces| * w_e/w_f survive:
    # face_weight=1 gives 2 + 1 = 3 (= 4 - 2 - 2 + 3), face_weight=2 gives 2.5.
    assert all(v == pytest.approx(3.0) for v in
               gg.curvature(G.copy(), method="augmented_forman_sreejith").values())
    r = gg.curvature(G.copy(), method="augmented_forman_sreejith", face_weight=2.0)
    assert all(v == pytest.approx(2.5) for v in r.values())


# ── evidence for the related-work section ───────────────────────────────────
# These pin the numbers quoted against GraphRicciCurvature. They were verified
# by running GRC 0.5.3.2 and 0.6.1 in a separate interpreter (it pins an old
# scipy) against this convention: max|difference| was 0.0 on unweighted karate
# and 7.1e-15 degree-weighted, and both GRC releases agreed byte-for-byte. GRC
# cannot be imported here, so the reproduced values are pinned instead.


def test_karate_range_matches_the_published_comparison():
    """Quoted in the related-work table: [-25, -2] over 78 edges."""
    G = nx.karate_club_graph()
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    kappa = gg.curvature(G, method="forman_sreejith")
    assert len(kappa) == 78
    assert min(kappa.values()) == pytest.approx(-25.0)
    assert max(kappa.values()) == pytest.approx(-2.0)


@pytest.mark.parametrize("node_weight", [
    -1, 0, 0.0, float("nan"), float("inf"),
    {0: -1.0, 1: 1.0, 2: 1.0, 3: 1.0},
    {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0},
    lambda n, G: -1.0,
    lambda n, G: float("nan"),
])
def test_non_positive_or_non_finite_node_weight_is_rejected(node_weight):
    """These used to be accepted and silently gave all-zero curvature."""
    G = nx.cycle_graph(4)
    with pytest.raises(gg.CurvatureConfigurationError, match="node_weight"):
        gg.curvature(G, method="forman_node_weighted", node_weight=node_weight)


def test_raw_function_node_weight_attr_must_be_positive():
    from graph_geometry.curvature.forman_variants import forman_node_weighted

    G = nx.cycle_graph(4)
    nx.set_node_attributes(G, {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}, "m")
    with pytest.raises(gg.CurvatureInputError, match="strictly positive"):
        forman_node_weighted(G, node_weight_attr="m")


def test_forman_does_not_floor_small_weights():
    """The plain node-weighted form depends only on weight ratios, so a common
    factor must not change it; the old 1e-10 floor collapsed it."""
    def build(scale):
        G = nx.path_graph(4)
        G.add_edge(1, 3)
        for (u, v), w in {(0, 1): 1, (1, 2): 3, (2, 3): 1, (1, 3): 2}.items():
            G[u][v]["weight"] = w * scale
        return G

    base = gg.compute_curvature(build(1.0), "forman_node_weighted").values
    tiny = gg.compute_curvature(build(1e-12), "forman_node_weighted").values
    for e in base:
        assert tiny[e] == pytest.approx(base[e], rel=1e-12)
