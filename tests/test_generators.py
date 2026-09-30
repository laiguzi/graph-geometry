"""Generator tests: structure (node/edge counts, connectivity)."""

import networkx as nx
import pytest

from graph_geometry import generate
from graph_geometry.graph.generators import GENERATORS


def test_cycle():
    G = generate.cycle(5)
    assert G.number_of_nodes() == 5
    assert G.number_of_edges() == 5


def test_complete():
    G = generate.complete(6)
    assert G.number_of_nodes() == 6
    assert G.number_of_edges() == 6 * 5 // 2


def test_petersen():
    G = generate.petersen()
    assert G.number_of_nodes() == 10
    assert G.number_of_edges() == 15


def test_sbm_sizes_and_seed_determinism():
    kw = dict(sizes=[8, 8], probs=[[0.9, 0.05], [0.05, 0.9]], seed=0)
    G1 = generate.sbm(**kw)
    G2 = generate.sbm(**kw)
    assert G1.number_of_nodes() == 16
    assert sorted(G1.edges()) == sorted(G2.edges())


def test_gab_structure():
    # a=2, b=1 -> internal a'=3, b'=2: two K3 cliques + a bridging clique on
    # their first nodes {0, 3}. Nodes 0..5.
    G = generate.gab(2, 1)
    assert G.number_of_nodes() == 6
    assert G.has_edge(0, 3)  # bridging edge between cliques' first nodes


def test_petersen_complete_bridge():
    G = generate.petersen_complete()
    assert G.number_of_nodes() == 16  # petersen 10 + complete 6
    assert G.has_edge(9, 10)  # single bridge
    # petersen(15) + complete(15) + 1 bridge
    assert G.number_of_edges() == 15 + 15 + 1


def test_complete_complete_bridge():
    G = generate.complete_complete(4, 5)
    assert G.number_of_nodes() == 9
    assert G.has_edge(3, 4)  # bridge (n1-1, n1)
    assert G.number_of_edges() == (4 * 3 // 2) + (5 * 4 // 2) + 1


def test_lfr_has_community_idx():
    # Parameters from NetworkX's own LFR docs (converge quickly); LFR is
    # stochastic, so skip rather than fail if a build can't satisfy them.
    try:
        G = generate.lfr(
            n=250, tau1=3, tau2=1.5, mu=0.1, average_degree=5, min_community=20, seed=10
        )
    except nx.ExceededMaxIterations:  # pragma: no cover - depends on RNG/version
        pytest.skip("LFR benchmark did not converge for these parameters")
    assert G.number_of_nodes() == 250
    assert all("community_idx" in G.nodes[n] for n in G.nodes())
    # the flattened per-node `community` frozensets should be gone
    assert all("community" not in G.nodes[n] for n in G.nodes())


def test_registry_matches_functions():
    assert GENERATORS["cycle"] is generate.cycle
    assert set(GENERATORS) == {
        "gab", "sbm", "lfr", "cycle", "petersen",
        "complete", "petersen_complete", "complete_complete",
    }


def test_generators_return_unweighted_graphs():
    # generators produce structure only; no weight/distance attrs
    G = generate.complete(4)
    assert all(not d for *_e, d in G.edges(data=True))
