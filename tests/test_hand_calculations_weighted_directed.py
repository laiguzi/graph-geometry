"""Hand-computable anchors beyond unweighted undirected graphs."""

import math

import networkx as nx
import pytest

import graph_geometry as gg


def test_weighted_nonunit_metric_path_ollivier_and_lly():
    """A three-node path checks both transition weights and metric lengths."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0, distance=2.0)
    G.add_edge(1, 2, weight=1.0, distance=3.0)

    ollivier = gg.curvature(G.copy(), method="ollivier", alpha=0.5)
    lly = gg.curvature(G.copy(), method="lin_lu_yau")

    # For edge 01, surplus 1/8 moves from 0 to 2 over distance 5:
    # W1=5/8 and kappa=1-(5/8)/2=11/16. Edge 12 is analogous.
    assert ollivier[(0, 1)] == pytest.approx(11 / 16, abs=1e-12)
    assert ollivier[(1, 2)] == pytest.approx(3 / 8, abs=1e-12)
    assert lly[(0, 1)] == pytest.approx(11 / 8, abs=1e-4)
    assert lly[(1, 2)] == pytest.approx(3 / 4, abs=1e-4)


def test_weighted_path_node_weighted_forman_by_direct_sum():
    """Undirected, weights 4 and 1: F(0,1) = 2 - sqrt(4/1) and F(1,2) = 2 - sqrt(1/4)."""
    G = nx.Graph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 2, weight=1.0)

    values = gg.curvature(G, method="forman_node_weighted")

    assert values[(0, 1)] == pytest.approx(2.0 - 2.0, abs=1e-12)
    assert values[(1, 2)] == pytest.approx(2.0 - 0.5, abs=1e-12)


def test_weighted_directed_forman_by_direct_sum():
    """Directed: only the in-edges of the tail and the out-edges of the head.

    0 -> 1 with w=4; 2 -> 0 (w=9) enters the tail, 1 -> 3 (w=1) leaves the head,
    and 0 -> 4 (w=1) leaves the tail, so it is not charged at all.
    """
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(2, 0, weight=9.0)
    G.add_edge(1, 3, weight=1.0)
    G.add_edge(0, 4, weight=1.0)

    values = gg.curvature(G, method="forman_directed")

    assert values[(0, 1)] == pytest.approx(2.0 - math.sqrt(4 / 9) - math.sqrt(4 / 1), abs=1e-12)
    # 2 -> 0: nothing enters 2, and 0 -> 1 and 0 -> 4 leave the head
    assert values[(2, 0)] == pytest.approx(2.0 - math.sqrt(9 / 4) - math.sqrt(9 / 1), abs=1e-12)


def test_weighted_feed_forward_triangle_augmented_forman_by_direct_sum():
    """The face term is w_e^2 / phi, and a face edge leaves the charged sum.

    0 -> 1 -> 2 with the shortcut 0 -> 2 is one feed-forward triangle, which is
    a face of all three of its edges (Saucan et al. 2019, Eq. 12).
    """
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 2, weight=1.0)
    G.add_edge(0, 2, weight=9.0)

    base = gg.curvature(G.copy(), method="forman_directed")
    augmented = gg.curvature(G.copy(), method="augmented_forman_directed")

    # (0,1): nothing enters 0; 1 -> 2 leaves the head and is charged, sqrt(4/1)
    assert base[(0, 1)] == pytest.approx(2.0 - 2.0, abs=1e-12)
    # and in the augmented form 1 -> 2 lies in the face, so only w_e^2 = 16 is left
    assert augmented[(0, 1)] == pytest.approx(2.0 + 16.0, abs=1e-12)
    # (0,2) is the shortcut: no charged edges either way, one face of weight 1
    assert base[(0, 2)] == pytest.approx(2.0, abs=1e-12)
    assert augmented[(0, 2)] == pytest.approx(2.0 + 81.0, abs=1e-12)


def test_weighted_directed_eidi_jost_transport_plan():
    """Weighted in/out measures with shared support give kappa=-1/2."""
    G = nx.DiGraph()
    G.add_edge("z", "x", weight=3.0, distance=1.0)
    G.add_edge("a", "x", weight=1.0, distance=1.0)
    G.add_edge("x", "y", weight=1.0, distance=1.0)
    G.add_edge("y", "z", weight=1.0, distance=1.0)
    G.add_edge("y", "b", weight=1.0, distance=1.0)

    # On x->y, mu_x^in={z:3/4,a:1/4}, mu_y^out={z:1/2,b:1/2}.
    # Half matches at z; the remaining mass travels distance 3, so W1=3/2.
    assert not nx.is_strongly_connected(G)
    values = gg.curvature(G, method="eidi_jost")
    assert values[("x", "y")] == pytest.approx(-0.5, abs=1e-12)
