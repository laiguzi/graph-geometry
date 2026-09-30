"""Curvature on node pairs, adjacent or not.

`kappa = 1 - W_1(mu_x, mu_y)/d(x, y)` needs two measures and the metric between
their supports, not an edge, so the optimal-transport family is defined on any
two distinct nodes. The combinatorial family is not: Forman is built from the
edge's own incidence structure. These tests pin both halves of that split.
"""

import itertools

import networkx as nx
import pytest

import graph_geometry as gg

#: Verified against an independent scipy-HiGHS transportation LP.
P5_ALL_PAIRS = {
    (0, 1): 0.5, (0, 2): 0.25, (0, 3): 1 / 6, (0, 4): 0.25,
    (1, 2): 0.0, (1, 3): 0.0, (1, 4): 1 / 6,
    (2, 3): 0.0, (2, 4): 0.25,
    (3, 4): 0.5,
}


def _p5():
    G = nx.path_graph(5)
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    return G


def test_every_pair_of_p5_matches_the_independent_lp():
    kappa = gg.curvature_pairs(_p5(), "ollivier", alpha=0.5)
    assert len(kappa) == 10
    for pair, expected in P5_ALL_PAIRS.items():
        assert kappa[pair] == pytest.approx(expected, abs=1e-9)


def test_the_edge_subset_agrees_with_the_ordinary_edge_call():
    """Restricting to edges must reproduce `curvature` exactly."""
    G = nx.karate_club_graph()
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    edges = list(G.edges())
    on_edges = gg.curvature(G.copy(), method="ollivier")
    on_pairs = gg.curvature_pairs(G.copy(), "ollivier", pairs=edges)
    for edge in edges:
        assert on_pairs[edge] == pytest.approx(on_edges[edge], abs=1e-12)


def test_non_adjacent_pairs_are_reachable_and_finite():
    G = _p5()
    far = [(0, 4), (0, 3), (1, 4)]
    kappa = gg.curvature_pairs(G, "ollivier", pairs=far)
    assert set(kappa) == set(far)
    assert all(abs(v) < 10 for v in kappa.values())


def test_the_graph_is_not_mutated():
    """A non-adjacent pair has no edge to carry the attribute, so nothing is written."""
    G = _p5()
    gg.curvature_pairs(G, "ollivier")
    for _, _, data in G.edges(data=True):
        assert "ricciCurvature" not in data
    for _, data in G.nodes(data=True):
        assert "ricciCurvature" not in data


def test_lin_lu_yau_also_works_on_pairs():
    kappa = gg.curvature_pairs(_p5(), "lin_lu_yau")
    assert kappa[(0, 1)] == pytest.approx(1.0, abs=1e-4)
    assert kappa[(0, 2)] == pytest.approx(0.5, abs=1e-4)


def test_a_curvature_config_is_accepted_directly():
    kappa = gg.curvature_pairs(_p5(), gg.ollivier(alpha=0.25))
    assert len(kappa) == 10


# ── the combinatorial family has no value off an edge ───────────────────────


@pytest.mark.parametrize(
    "method", ["forman_node_weighted", "augmented_forman_node_weighted", "forman_sreejith"]
)
def test_combinatorial_methods_refuse_pairs_and_say_why(method):
    with pytest.raises(ValueError, match="edges only"):
        gg.curvature_pairs(_p5(), method)


def test_the_refusal_names_the_pair_capable_methods():
    with pytest.raises(ValueError) as excinfo:
        gg.curvature_pairs(_p5(), "forman_node_weighted")
    message = str(excinfo.value)
    for name in gg.PAIR_CAPABLE:
        assert name in message


# ── directed ────────────────────────────────────────────────────────────────


def _directed_c4():
    D = nx.DiGraph([(i, (i + 1) % 4) for i in range(4)])
    for _, _, data in D.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    return D


def test_directed_default_pairs_are_ordered():
    kappa = gg.curvature_pairs(_directed_c4(), "ollivier", kernel="out")
    assert len(kappa) == 4 * 3          # ordered, not 6 unordered
    assert (0, 1) in kappa and (1, 0) in kappa


def test_eidi_jost_is_edge_only_and_refuses_pairs_before_any_solve():
    """In-out supports are joined through the edge; a non-edge has no such path."""
    with pytest.raises(gg.CurvatureDomainError, match="edges only"):
        gg.curvature_pairs(_directed_c4(), "eidi_jost")
    with pytest.raises(gg.CurvatureDomainError, match="edges only"):
        gg.compute_pair_curvature(_directed_c4(), "eidi_jost", [(0, 1)])


def test_eidi_jost_rejects_an_undirected_graph_on_pairs_too():
    with pytest.raises(ValueError, match="directed"):
        gg.curvature_pairs(_p5(), "eidi_jost")


def test_pair_capability_comes_from_the_registry():
    capable = {spec.name for spec in gg.list_curvatures(scope="pairs")}
    assert capable == {"lin_lu_yau", "ollivier"}
    assert set(gg.PAIR_CAPABLE) == capable


# ── argument checking ───────────────────────────────────────────────────────


def test_a_self_pair_is_rejected():
    with pytest.raises(ValueError, match="distinct"):
        gg.curvature_pairs(_p5(), "ollivier", pairs=[(1, 1)])


def test_an_unknown_node_is_rejected():
    with pytest.raises(ValueError, match="not in the graph"):
        gg.curvature_pairs(_p5(), "ollivier", pairs=[(0, 99)])


def test_unknown_method_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown curvature method"):
        gg.curvature_pairs(_p5(), "nope")


def test_all_pairs_covers_exactly_the_combinations():
    G = _p5()
    kappa = gg.curvature_pairs(G, "ollivier")
    assert set(kappa) == set(itertools.combinations(sorted(G.nodes()), 2))


# ── parallelism ─────────────────────────────────────────────────────────────


def test_proc_is_accepted_and_does_not_change_the_values():
    """Parallel execution must be an optimisation, never a semantic change."""
    G = nx.barabasi_albert_graph(40, 3, seed=1)
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    serial = gg.curvature_pairs(G, "ollivier", proc=1)
    parallel = gg.curvature_pairs(G, "ollivier", proc=4)
    assert set(serial) == set(parallel)
    for pair, value in serial.items():
        assert parallel[pair] == pytest.approx(value, abs=1e-12)


def test_shortest_path_lengths_are_the_dijkstra_lengths():
    """The engine used to re-derive lengths by summing along materialised paths.

    Dropping that must not change a single number: the sum of edge distances
    along a shortest path is the shortest-path length.
    """
    from graph_geometry.curvature.base import RicciCurvature

    G = nx.barabasi_albert_graph(60, 2, seed=3)
    for u, v, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 0.5 + (u + v) % 4
    engine = RicciCurvature(G, gg.ollivier(alpha=0.5), "weight", "distance", 1)
    lengths = engine._all_pairs_shortest_path()
    reference = dict(nx.all_pairs_dijkstra_path_length(G, weight="distance"))
    assert set(lengths) == set(reference)
    for source, targets in reference.items():
        for target, expected in targets.items():
            assert lengths[source][target] == pytest.approx(expected, abs=1e-12)
