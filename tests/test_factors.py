"""Factor strategy tests: correctness + seed determinism."""

import networkx as nx
import numpy as np
import pytest

from graph_geometry.graph.factors import (
    as_rng,
    resolve_edge_factor,
    resolve_node_factor,
)


def _triangle(directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


def test_fixed_edge_and_node_are_one():
    G = _triangle()
    ef = resolve_edge_factor("fixed")
    nf = resolve_node_factor("fixed")
    assert all(ef(u, v, G) == 1.0 for u, v in G.edges())
    assert all(nf(n, G) == 1.0 for n in G.nodes())


def test_numeric_spec_is_constant():
    G = _triangle()
    ef = resolve_edge_factor(0.5)
    nf = resolve_node_factor(2)
    assert all(ef(u, v, G) == 0.5 for u, v in G.edges())
    assert all(nf(n, G) == 2.0 for n in G.nodes())


def test_degree_node_matches_networkx_degree():
    G = nx.star_graph(4)  # center node 0 has degree 4, leaves degree 1
    nf = resolve_node_factor("degree")
    assert nf(0, G) == 4.0
    assert all(nf(leaf, G) == 1.0 for leaf in (1, 2, 3, 4))


def test_degree_edge_is_endpoint_degree_sum():
    G = nx.star_graph(4)
    ef = resolve_edge_factor("degree")
    # every edge connects center (deg 4) to a leaf (deg 1) -> 5
    assert all(ef(u, v, G) == 5.0 for u, v in G.edges())


def test_degree_uses_total_degree_on_directed():
    G = nx.DiGraph()
    G.add_edges_from([(0, 1), (0, 2)])  # node 0: out 2, in 0 -> total 2
    nf = resolve_node_factor("degree")
    assert nf(0, G) == 2.0


def test_random_is_reproducible_with_same_seed():
    G = _triangle()
    f1 = resolve_edge_factor("random", as_rng(0))
    f2 = resolve_edge_factor("random", as_rng(0))
    vals1 = [f1(u, v, G) for u, v in G.edges()]
    vals2 = [f2(u, v, G) for u, v in G.edges()]
    assert vals1 == vals2
    assert all(0.0 <= x <= 1.0 for x in vals1)


def test_random_differs_across_seeds():
    G = _triangle()
    f1 = resolve_edge_factor("random", as_rng(0))
    f2 = resolve_edge_factor("random", as_rng(1))
    vals1 = [f1(u, v, G) for u, v in G.edges()]
    vals2 = [f2(u, v, G) for u, v in G.edges()]
    assert vals1 != vals2


def test_callable_spec_passthrough():
    G = _triangle()
    ef = resolve_edge_factor(lambda u, v, g: float(u + v))
    assert ef(1, 2, G) == 3.0


def test_dict_spec_edge_explicit_assignments():
    G = _triangle(directed=True)
    ef = resolve_edge_factor({(0, 1): 0.5, (1, 2): 0.8, (2, 0): 1.5})
    assert ef(0, 1, G) == 0.5
    assert ef(2, 0, G) == 1.5
    with pytest.raises(ValueError):  # missing edge -> clear error
        ef(1, 0, G)


def test_dict_spec_edge_undirected_reversed_key():
    G = _triangle(directed=False)
    ef = resolve_edge_factor({(1, 0): 0.7})  # stored reversed
    assert ef(0, 1, G) == 0.7  # (v, u) fallback on undirected


def test_dict_spec_node_explicit_assignments():
    G = _triangle()
    nf = resolve_node_factor({0: 2.0, 1: 3.0, 2: 4.0})
    assert nf(1, G) == 3.0
    with pytest.raises(ValueError):
        nf(99, G)


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        resolve_edge_factor("bogus")
    with pytest.raises(ValueError):
        resolve_node_factor("bogus")


def test_bool_spec_rejected():
    # bool is an int subclass; reject to avoid silent True==1.0 surprises
    with pytest.raises(TypeError):
        resolve_edge_factor(True)
    with pytest.raises(TypeError):
        resolve_node_factor(False)


def test_as_rng_passthrough_and_construction():
    gen = np.random.default_rng(3)
    assert as_rng(gen) is gen
    assert isinstance(as_rng(3), np.random.Generator)
