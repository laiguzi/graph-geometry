"""Loader tests: construction, weight preservation, round-trips."""

import networkx as nx
import pytest

from graph_geometry import load


def test_from_edges_directed_default():
    G = load.from_edges([(0, 1), (1, 2), (2, 0)])
    assert G.is_directed()
    assert G.number_of_edges() == 3
    assert not G.has_edge(1, 0)  # directed: reverse absent


def test_from_edges_undirected():
    G = load.from_edges([(0, 1), (1, 2)], directed=False)
    assert not G.is_directed()
    assert G.has_edge(1, 0)


def test_from_edges_preserves_triple_weight():
    G = load.from_edges([(0, 1, 0.5), (1, 2, 0.8)])
    assert G[0][1]["weight"] == 0.5
    assert G[1][2]["weight"] == 0.8


def test_from_edges_no_weight_for_pairs():
    # structure only; factor assignment is apply_config's job
    G = load.from_edges([(0, 1), (1, 2)])
    assert "weight" not in G[0][1]


def test_from_edges_four_tuple_weight_and_distance():
    G = load.from_edges([(0, 1, 0.5, 2.0), (1, 2, 0.8, 1.5)])
    assert G[0][1]["weight"] == 0.5 and G[0][1]["distance"] == 2.0
    assert G[1][2]["weight"] == 0.8 and G[1][2]["distance"] == 1.5


def test_from_edges_dict_payload():
    G = load.from_edges([(0, 1, {"weight": 0.3, "distance": 4.0, "label": "a"})])
    assert G[0][1]["weight"] == 0.3
    assert G[0][1]["distance"] == 4.0
    assert G[0][1]["label"] == "a"


def test_from_edges_bad_tuple():
    with pytest.raises(ValueError):
        load.from_edges([(0, 1, 2, 3, 4)])


def test_from_edgelist_multi_columns(tmp_path):
    p = tmp_path / "gwd.edgelist"
    p.write_text("0 1 0.5 2.0\n1 2 0.8 1.5\n")
    G = load.from_edgelist(str(p), directed=True, columns=("weight", "distance"))
    assert G["0"]["1"]["weight"] == 0.5
    assert G["0"]["1"]["distance"] == 2.0
    assert G["1"]["2"]["distance"] == 1.5


def test_from_edgelist_node_weight_lines(tmp_path):
    # 'N n w' lines assign node weights; mixable with weighted edges + comments
    p = tmp_path / "gn.edgelist"
    p.write_text(
        "# a comment\n"
        "N 1 2.5\n"
        "N 2 3.0\n"
        "1 2 0.5\n"
        "2 3 0.8\n"
        "N 9 7.0\n"          # isolated node, declared via N line only
    )
    G = load.from_edgelist(str(p), directed=True, weighted=True)
    assert G.nodes["1"]["weight"] == 2.5
    assert G.nodes["2"]["weight"] == 3.0
    assert G["1"]["2"]["weight"] == 0.5
    assert "9" in G.nodes and G.nodes["9"]["weight"] == 7.0
    assert G.degree("9") == 0  # isolated but present


def test_node_lines_with_columns_and_dataset_route(tmp_path):
    # full user format through from_dataset("mygraph"): edges w+d, node weights
    d = tmp_path / "mygraph"
    d.mkdir()
    (d / "mygraph_custom.edgelist").write_text(
        "N 1 10.0\nN 2 20.0\nN 3 30.0\n"
        "1 2 0.5 2.0\n2 3 0.8 1.0\n3 1 1.5 3.0\n"
    )
    G = load.from_dataset(
        "mygraph", tmp_path, variant="custom", columns=("weight", "distance")
    )
    assert G.nodes["2"]["weight"] == 20.0
    assert G["1"]["2"]["weight"] == 0.5 and G["1"]["2"]["distance"] == 2.0


def test_node_lines_not_clobbered_by_apply_config(tmp_path):
    import graph_geometry as gg

    p = tmp_path / "g.edgelist"
    p.write_text("N 1 5.0\n1 2\n2 3\n")
    G = load.from_edgelist(str(p), directed=True)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", node_factor="fixed")
    assert G.nodes["1"]["weight"] == 5.0   # file value preserved (overwrite=False)
    assert G.nodes["2"]["weight"] == 1.0   # filled by the fixed strategy


def test_malformed_node_line_raises(tmp_path):
    p = tmp_path / "bad.edgelist"
    p.write_text("N 1\n1 2\n")
    with pytest.raises(ValueError, match="node line"):
        load.from_edgelist(str(p))


def test_from_networkx_copies_by_default():
    src = nx.karate_club_graph()
    G = load.from_networkx(src)
    assert G is not src
    G.add_node("sentinel")
    assert "sentinel" not in src  # independent copy


def test_from_networkx_no_copy():
    src = nx.path_graph(3)
    G = load.from_networkx(src, copy=False)
    assert G is src


def test_from_edgelist_roundtrip(tmp_path):
    p = tmp_path / "g.edgelist"
    p.write_text("0 1\n1 2\n2 0\n")
    G = load.from_edgelist(str(p), directed=True)
    assert G.is_directed()
    assert G.number_of_edges() == 3


def test_from_edgelist_weighted(tmp_path):
    p = tmp_path / "gw.edgelist"
    p.write_text("0 1 0.5\n1 2 0.8\n")
    G = load.from_edgelist(str(p), directed=False, weighted=True)
    assert G["0"]["1"]["weight"] == 0.5


def test_from_gexf_roundtrip(tmp_path):
    src = nx.DiGraph()
    src.add_edge("a", "b", weight=2.0)
    p = tmp_path / "g.gexf"
    nx.write_gexf(src, str(p))
    G = load.from_gexf(str(p))
    assert G.is_directed()
    assert G["a"]["b"]["weight"] == 2.0


# ── tab-separated files and node ids containing spaces ──────────────────────
# From a real upload: an E. coli regulatory network (1,212 nodes) failed with
# "Failed to convert weight data Gene to type <class 'float'>". The target
# `Phantom Gene` contains a space, so whitespace splitting produced three
# tokens and the third was handed to NetworkX as a weight.


def test_tab_delimited_ids_may_contain_spaces(tmp_path):
    path = tmp_path / "spaced.edgelist"
    path.write_text("ArcA\tPhantom Gene\nArcA\tacrA\n")
    G = load.from_edgelist(str(path), directed=True, delimiter="\t")
    assert "Phantom Gene" in G
    assert G.number_of_nodes() == 3
    assert ("ArcA", "Phantom Gene") in G.edges()


def test_whitespace_splitting_still_the_default(tmp_path):
    """Without a delimiter, NetworkX's whitespace convention is unchanged."""
    path = tmp_path / "plain.edgelist"
    path.write_text("a b\nb c\n")
    G = load.from_edgelist(str(path), directed=True)
    assert set(G.edges()) == {("a", "b"), ("b", "c")}


def test_tab_delimiter_with_weights(tmp_path):
    path = tmp_path / "w.edgelist"
    path.write_text("Phantom Gene\tacrA\t2.5\nacrA\tacrB\t1.5\n")
    G = load.from_edgelist(str(path), directed=True, weighted=True, delimiter="\t")
    assert G["Phantom Gene"]["acrA"]["weight"] == pytest.approx(2.5)


def test_tab_delimited_node_lines(tmp_path):
    path = tmp_path / "n.edgelist"
    path.write_text("N\tPhantom Gene\t3.0\nPhantom Gene\tacrA\n")
    G = load.from_edgelist(str(path), directed=True, delimiter="\t")
    assert G.nodes["Phantom Gene"]["weight"] == pytest.approx(3.0)


def test_trailing_delimiter_does_not_create_an_empty_field(tmp_path):
    """The real file ends every row with a tab."""
    path = tmp_path / "t.edgelist"
    path.write_text("AcrR\tacrA\t\nAcrR\tacrB\t\n")
    G = load.from_edgelist(str(path), directed=True, delimiter="\t")
    assert G.number_of_edges() == 2
    assert all(not d for _, _, d in G.edges(data=True))
