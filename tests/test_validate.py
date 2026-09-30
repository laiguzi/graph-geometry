"""Input validation: the checks that stop a bad graph before any solver runs.

Regression tests for the defects recorded in RicciFlow-Computation.md under
"Curvature input validation is centralized and mandatory". Each case below used
to either return a plausible-looking number, or fail with an exception from
inside NetworkX / POT that named nothing useful.
"""

import math

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.curvature.validate import (
    reject_multigraph,
    validate_edge_attributes,
    validate_simple_graph,
)

OT_METHODS = ("lin_lu_yau", "ollivier")
COMBINATORIAL = ("forman_node_weighted", "augmented_forman_node_weighted")
ALL_BUILTINS = OT_METHODS + COMBINATORIAL


def _two_edge_graph(**first_edge):
    G = nx.Graph()
    G.add_edge(0, 1, **first_edge)
    G.add_edge(1, 2, weight=1.0, distance=1.0)
    return G


# ── non-finite and non-numeric attributes ────────────────────────────────────


@pytest.mark.parametrize("method", ALL_BUILTINS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_weight_rejected(method, bad):
    """A NaN weight used to come back as an ordinary 0.5; inf hit a POT assert."""
    G = _two_edge_graph(weight=bad, distance=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        gg.curvature(G, method=method)


@pytest.mark.parametrize("method", OT_METHODS)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_distance_rejected(method, bad):
    """A NaN distance used to be returned as a NaN curvature."""
    G = _two_edge_graph(weight=1.0, distance=bad)
    with pytest.raises(ValueError, match="non-finite"):
        gg.curvature(G, method=method)


@pytest.mark.parametrize("method", ALL_BUILTINS)
def test_bool_weight_rejected(method):
    G = _two_edge_graph(weight=True, distance=1.0)
    with pytest.raises(gg.CurvatureInputError, match="non-numeric"):
        gg.curvature(G, method=method)


# ── invalid measures and invalid metrics ─────────────────────────────────────


@pytest.mark.parametrize("method", ALL_BUILTINS)
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_weight_rejected(method, bad):
    """A non-positive weight gives a signed 'measure', not a probability one."""
    G = _two_edge_graph(weight=bad, distance=1.0)
    with pytest.raises(ValueError, match="weight"):
        gg.curvature(G, method=method)


@pytest.mark.parametrize("method", OT_METHODS)
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_distance_rejected(method, bad):
    """kappa = 1 - W1/d is undefined at d <= 0; zero used to return 0.0 silently."""
    G = _two_edge_graph(weight=1.0, distance=bad)
    with pytest.raises(ValueError, match="distance"):
        gg.curvature(G, method=method)


def test_error_names_the_attribute_and_the_edge():
    G = nx.Graph()
    G.add_edge("a", "b", weight=1.0, distance=1.0)
    G.add_edge("b", "c", weight=float("nan"), distance=1.0)
    with pytest.raises(ValueError) as excinfo:
        gg.curvature(G, method="ollivier")
    message = str(excinfo.value)
    assert "weight" in message
    assert "'b'" in message and "'c'" in message


# ── missing attributes are not an error ──────────────────────────────────────


def test_missing_attributes_default_to_one():
    """Every reader in the package uses .get(attr, 1.0); so does the metric now."""
    kappa = gg.curvature(nx.cycle_graph(5), method="ollivier")
    assert kappa[(0, 1)] == pytest.approx(0.25)


def test_extreme_but_valid_weights_are_accepted():
    for w in (1e-300, 1e300):
        G = _two_edge_graph(weight=w, distance=1.0)
        values = gg.curvature(G, method="ollivier")
        assert all(math.isfinite(v) for v in values.values())


# ── multigraphs ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("method", OT_METHODS)
@pytest.mark.parametrize(
    "graph, combinatorial",
    [
        (nx.MultiGraph([(0, 1), (0, 1), (1, 2)]), "forman_node_weighted"),
        (nx.MultiDiGraph([(0, 1), (1, 0), (1, 2), (2, 1)]), "forman_directed"),
    ],
    ids=["multigraph", "multidigraph"],
)
def test_multigraph_rejected_with_a_clear_error(method, graph, combinatorial):
    """Used to fail with KeyError('distance') or an AtlasView TypeError."""
    G = graph.copy()
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    for name in (method, combinatorial):
        with pytest.raises(gg.CurvatureDomainError, match="multigraph"):
            gg.curvature(G.copy(), method=name)


def test_eidi_jost_rejects_multigraphs_too():
    G = nx.MultiDiGraph([(0, 1), (1, 2), (2, 0)])
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
        data["distance"] = 1.0
    with pytest.raises(gg.CurvatureDomainError, match="multigraph"):
        gg.curvature(G, method="eidi_jost")


# ── Eidi-Jost keeps its own, better signed-graph message ─────────────────────


def test_eidi_jost_signed_weight_message_is_not_pre_empted():
    """validate_edge_attributes must not shadow directed_in_out_measure's error."""
    G = nx.DiGraph()
    for u, v, w in ((0, 1, 3.0), (1, 2, -1.0), (2, 0, 1.0)):
        G.add_edge(u, v, weight=w, distance=1.0)
    with pytest.raises(ValueError, match="double cover|signed"):
        gg.curvature(G, method="eidi_jost")


def test_eidi_jost_still_rejects_non_finite_weights():
    G = nx.DiGraph()
    for u, v, w in ((0, 1, float("nan")), (1, 2, 1.0), (2, 0, 1.0)):
        G.add_edge(u, v, weight=w, distance=1.0)
    with pytest.raises(ValueError, match="non-finite"):
        gg.curvature(G, method="eidi_jost")


# ── the helpers directly ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "graph, message",
    [
        (nx.Graph([(0, 1), (1, 1)]), "self-loops"),
        (nx.DiGraph([(0, 0)]), "self-loops"),
        (nx.MultiDiGraph([(0, 1), (0, 1)]), "multigraphs"),
    ],
    ids=["graph-loop", "digraph-loop-only", "multidigraph"],
)
def test_simple_graph_validator_refuses_loops_and_multigraphs(graph, message):
    with pytest.raises(gg.CurvatureDomainError, match=message):
        validate_simple_graph(graph, "here")


@pytest.mark.parametrize("graph", [nx.Graph(), nx.path_graph(3), nx.DiGraph([(0, 1), (1, 0)])])
def test_simple_graph_validator_accepts_simple_graphs_including_empty(graph):
    """An empty graph is simple; whether it is usable is the caller's decision."""
    validate_simple_graph(graph, "here")


def test_reject_multigraph_passes_simple_graphs():
    reject_multigraph(nx.Graph([(0, 1)]), "unit test")
    reject_multigraph(nx.DiGraph([(0, 1)]), "unit test")


def test_validate_can_skip_an_attribute():
    """distance=None means 'do not check the metric' (Forman does not need one)."""
    G = _two_edge_graph(weight=1.0, distance=-5.0)
    validate_edge_attributes(G, weight="weight", distance=None, where="unit test")
    with pytest.raises(ValueError, match="distance"):
        validate_edge_attributes(
            G, weight="weight", distance="distance", where="unit test"
        )


def test_validate_allows_signed_weights_when_asked():
    G = _two_edge_graph(weight=-2.0, distance=1.0)
    validate_edge_attributes(
        G, weight="weight", where="unit test", positive_weight=False
    )
