"""Directed Forman-Ricci curvature (Saucan et al. 2019, Eq. 5 and Eq. 12).

    kappa(x,y)  = m_x + m_y - sum_{e' in delta^-(x)} m_x sqrt(w_e/w_e')
                            - sum_{e' in delta^+(y)} m_y sqrt(w_e/w_e')

Only edges entering the tail and leaving the head are charged, so ``e`` drops
out by itself, a reciprocal edge is charged at both endpoints, and the
unweighted value is ``2 - in(x) - out(y)`` (Leal et al. 2021's flow through the
arc). The augmented form adds the feed-forward triangles through the edge as
faces of weight ``phi``, each worth ``w_e^2 / phi``, and drops the charged edges
that lie in one of them.
"""

import math

import networkx as nx
import pytest

import graph_geometry as gg


def _unit(G):
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
    return G


# ── the incidence selector ──────────────────────────────────────────────────


def test_unweighted_value_is_two_minus_in_tail_minus_out_head():
    D = _unit(nx.gnp_random_graph(25, 0.2, seed=11, directed=True))
    values = gg.compute_curvature(D, "forman_directed").values
    for x, y in D.edges():
        assert values[(x, y)] == pytest.approx(2 - D.in_degree(x) - D.out_degree(y))


def test_a_single_edge_has_no_charged_neighbours():
    """e leaves the tail and enters the head, so it is in neither star."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=2.0)
    assert gg.compute_curvature(G, "forman_directed").values[(0, 1)] == pytest.approx(2.0)


def test_the_reciprocal_edge_is_charged_at_both_endpoints():
    """Published convention: only self-loops are excluded (the 2016 preprint
    also dropped the reciprocal edge; the 2019 paper does not)."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 0, weight=1.0)
    value = gg.compute_curvature(G, "forman_directed").values[(0, 1)]
    # (1,0) enters 0 and leaves 1: charged twice, each sqrt(4/1) = 2
    assert value == pytest.approx(2.0 - 2.0 - 2.0)


def test_self_loops_are_refused_before_the_definition_sees_them():
    """The papers ignore self-loops; the package refuses them for every
    definition, so skipping them in the star selector only keeps the raw
    function (which bypasses the dispatcher) faithful to the formula."""
    from graph_geometry.curvature.forman_directed import forman_directed

    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1.0)
    G.add_edge(0, 0, weight=1.0)
    G.add_edge(1, 1, weight=1.0)
    with pytest.raises(gg.CurvatureDomainError, match="self-loops"):
        gg.compute_curvature(G, "forman_directed")
    assert forman_directed(G)[(0, 1)] == pytest.approx(2.0)


def test_out_edges_of_the_tail_and_in_edges_of_the_head_are_not_charged():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(0, 2, weight=1.0)    # leaves the tail
    G.add_edge(3, 1, weight=1.0)    # enters the head
    assert gg.compute_curvature(G, "forman_directed").values[(0, 1)] == pytest.approx(2.0)


def test_node_weights_multiply_their_own_endpoint():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(2, 0, weight=1.0)    # in-edge of the tail
    G.add_edge(1, 3, weight=4.0)    # out-edge of the head
    for u in G:
        G.nodes[u]["m"] = float(u) + 1.0
    request = gg.CurvatureRequest("forman_directed", {})
    value = gg.compute_curvature(
        G, request, semantics=gg.GraphSemantics(node_weight_attr="m")
    ).values[(0, 1)]
    # m_0 = 1, m_1 = 2: 1 + 2 - 1*sqrt(4/1) - 2*sqrt(4/4)
    assert value == pytest.approx(1.0 + 2.0 - 2.0 - 2.0)


# ── the face selector: feed-forward triangles ───────────────────────────────


def _ffl():
    """x -> q -> y with the shortcut x -> y."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=1.0)    # x -> q
    G.add_edge(1, 2, weight=1.0)    # q -> y
    G.add_edge(0, 2, weight=1.0)    # x -> y, the shortcut
    return G


def test_every_edge_of_a_feed_forward_triangle_carries_the_face():
    """Eq. 12 sums over every face containing e, in any of its three positions."""
    values = gg.compute_curvature(_ffl(), "augmented_forman_directed").values
    assert set(values.values()) == {3.0}
    plain = gg.compute_curvature(_ffl(), "forman_directed").values
    assert dict(plain) == {(0, 1): 1.0, (1, 2): 1.0, (0, 2): 2.0}


def test_a_directed_three_cycle_has_no_faces():
    """A cycle is not the chosen directed simplex (Saucan et al. 2019, Fig. 1(b))."""
    C = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
    assert dict(gg.compute_curvature(C, "augmented_forman_directed").values) == dict(
        gg.compute_curvature(C, "forman_directed").values
    )


def _edge_terms(G, method):
    from graph_geometry.curvature.combinatorial import CombinatorialCurvatureEngine
    from graph_geometry.curvature.forman_directed import (
        augmented_forman_directed_config,
        forman_directed_config,
    )

    config = (augmented_forman_directed_config() if method.startswith("augmented")
              else forman_directed_config())
    return CombinatorialCurvatureEngine(G, config).edge_terms()


def test_charged_edges_inside_a_face_leave_the_sums():
    """Only a leg of the triangle drops a charge; the shortcut's face edges are
    an out-edge of the tail and an in-edge of the head, neither of them charged."""
    G = _ffl()
    terms = _edge_terms(G, "augmented_forman_directed")
    assert terms[(0, 1)]["B_y"] == 0.0     # (1,2) leaves the head but lies in the face
    assert terms[(1, 2)]["B_x"] == 0.0     # (0,1) enters the tail but lies in the face
    assert terms[(0, 2)]["C"] == 1.0       # the shortcut still carries the face


def test_face_weight_divides_the_face_term():
    G = _ffl()
    G[0][2]["weight"] = 3.0
    for phi in (1.0, 2.0, 4.5):
        values = gg.compute_curvature(G, gg.CurvatureRequest(
            "augmented_forman_directed", {"face_weight": phi})).values
        # the shortcut (0,2): no charged neighbours, one face of weight phi
        assert values[(0, 2)] == pytest.approx(2.0 + 9.0 / phi)


def test_weighted_feed_forward_triangle_by_hand():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)    # x -> q
    G.add_edge(1, 2, weight=2.0)    # q -> y
    G.add_edge(0, 2, weight=9.0)    # x -> y
    G.add_edge(3, 0, weight=1.0)    # in-edge of node 0
    plain = gg.compute_curvature(G, "forman_directed").values
    augmented = gg.compute_curvature(G, "augmented_forman_directed").values
    # (0,2): delta^-(0) = {(3,0)}, delta^+(2) = {} -> 2 - sqrt(9/1)
    assert plain[(0, 2)] == pytest.approx(2.0 - 3.0)
    # its face is ((0,1), (1,2)); neither is charged, so only C is added
    assert augmented[(0, 2)] == pytest.approx(2.0 - 3.0 + 81.0)
    # (0,1) is the first leg: delta^+(1) = {(1,2)} is in the face and leaves
    assert plain[(0, 1)] == pytest.approx(2.0 - math.sqrt(4 / 1) - math.sqrt(4 / 2))
    assert augmented[(0, 1)] == pytest.approx(2.0 - math.sqrt(4 / 1) + 16.0)


# ── the definitions are directed-only ───────────────────────────────────────


@pytest.mark.parametrize("method", ["forman_directed", "augmented_forman_directed"])
def test_undirected_input_is_refused_with_the_undirected_definitions_named(method):
    with pytest.raises(gg.CurvatureDomainError, match="forman_node_weighted"):
        gg.compute_curvature(_unit(nx.cycle_graph(4)), method)


def test_the_result_contract_is_the_usual_one():
    G = _unit(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
    result = gg.compute_curvature(G, "forman_directed")
    assert set(result.values) == set(G.edges())
    assert set(result.incident_node_means) == set(G.nodes())
    assert all("ricciCurvature" not in d for _, _, d in G.edges(data=True))
