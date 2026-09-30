"""GraphConfig / apply_config tests."""

import networkx as nx
import pytest

from graph_geometry import GraphConfig, apply_config


def _triangle(directed=False):
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    return G


def test_apply_config_writes_edge_and_node_attrs():
    G = _triangle()
    apply_config(G, GraphConfig(edge_weight="fixed", edge_distance=2.0, node_factor="degree"))
    for u, v in G.edges():
        assert G[u][v]["weight"] == 1.0
        assert G[u][v]["distance"] == 2.0
    for n in G.nodes():
        assert G.nodes[n]["weight"] == float(G.degree(n))


def test_apply_config_kwargs_without_cfg():
    G = _triangle()
    apply_config(G, edge_weight=0.7)
    assert all(G[u][v]["weight"] == 0.7 for u, v in G.edges())


def test_apply_config_returns_graph_and_mutates_in_place():
    G = _triangle()
    out = apply_config(G, edge_weight="fixed")
    assert out is G
    assert "weight" in next(iter(G.edges(data=True)))[2]


def test_copy_leaves_original_untouched():
    G = _triangle()
    out = apply_config(G, edge_weight="fixed", copy=True)
    assert out is not G
    assert all("weight" not in d for *_e, d in G.edges(data=True))
    assert all("weight" in d for *_e, d in out.edges(data=True))


def test_overwrite_false_preserves_existing_source_weights():
    G = _triangle()
    G[0][1]["weight"] = 99.0
    apply_config(G, edge_weight="fixed")  # overwrite=False default
    assert G[0][1]["weight"] == 99.0
    assert G[1][2]["weight"] == 1.0


def test_overwrite_true_clobbers():
    G = _triangle()
    G[0][1]["weight"] = 99.0
    apply_config(G, edge_weight="fixed", overwrite=True)
    assert G[0][1]["weight"] == 1.0


def test_seed_determinism_end_to_end():
    G1, G2 = _triangle(), _triangle()
    apply_config(G1, edge_weight="random", edge_distance="random", node_factor="random", seed=42)
    apply_config(G2, edge_weight="random", edge_distance="random", node_factor="random", seed=42)
    assert nx.get_edge_attributes(G1, "weight") == nx.get_edge_attributes(G2, "weight")
    assert nx.get_edge_attributes(G1, "distance") == nx.get_edge_attributes(G2, "distance")
    assert nx.get_node_attributes(G1, "weight") == nx.get_node_attributes(G2, "weight")


def test_different_seed_changes_random_values():
    G1, G2 = _triangle(), _triangle()
    apply_config(G1, edge_weight="random", seed=1)
    apply_config(G2, edge_weight="random", seed=2)
    assert nx.get_edge_attributes(G1, "weight") != nx.get_edge_attributes(G2, "weight")


def test_directed_graph_supported():
    G = _triangle(directed=True)
    apply_config(G, edge_weight="degree", node_factor="degree")
    # directed degree is total (in+out); each triangle node has total degree 2
    assert all(G.nodes[n]["weight"] == 2.0 for n in G.nodes())


def test_custom_attr_names():
    G = _triangle()
    cfg = GraphConfig(weight_attr="w", distance_attr="d", node_attr="nf", node_factor=3.0)
    apply_config(G, cfg)
    assert all("w" in G[u][v] and "d" in G[u][v] for u, v in G.edges())
    assert all(G.nodes[n]["nf"] == 3.0 for n in G.nodes())


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        apply_config(_triangle(), edge_weight="nope")


def test_explicit_user_assignments_end_to_end():
    # the data_load.py scenario: user gives specific edge weights/distances
    # and node weights; apply_config writes them all
    G = _triangle(directed=True)
    apply_config(
        G,
        edge_weight={(0, 1): 0.5, (1, 2): 0.8, (2, 0): 1.5},
        edge_distance={(0, 1): 2.0, (1, 2): 1.0, (2, 0): 3.0},
        node_factor={0: 10.0, 1: 20.0, 2: 30.0},
    )
    assert G[0][1]["weight"] == 0.5 and G[0][1]["distance"] == 2.0
    assert G[2][0]["weight"] == 1.5 and G[2][0]["distance"] == 3.0
    assert G.nodes[1]["weight"] == 20.0


def test_partial_dict_with_source_attrs_and_no_overwrite():
    # source file provided some weights; the dict only needs to cover the rest
    G = _triangle(directed=True)
    G[0][1]["weight"] = 99.0  # came from the file
    apply_config(G, edge_weight={(1, 2): 0.8, (2, 0): 1.5})  # overwrite=False
    assert G[0][1]["weight"] == 99.0  # untouched
    assert G[1][2]["weight"] == 0.8


@pytest.mark.parametrize("G", [nx.MultiGraph([(0, 1)]), nx.MultiDiGraph([(0, 1)])])
def test_apply_config_rejects_multigraphs_clearly(G):
    with pytest.raises(TypeError, match="does not support multigraphs"):
        apply_config(G)


def test_graph_config_declares_where_the_sign_lives():
    """Sign is a declared semantic component, but never a generated factor."""
    import dataclasses

    cfg = GraphConfig(sign_attr="interaction")
    assert cfg.sign_attr == "interaction"
    fields = {f.name for f in dataclasses.fields(GraphConfig)}
    assert "edge_sign" not in fields, "a sign is data, not a factor to generate"


def test_apply_config_never_writes_a_sign():
    G = nx.DiGraph([(0, 1), (1, 2)])
    apply_config(G, edge_weight="fixed")
    assert all("sign" not in d for _, _, d in G.edges(data=True))
