"""The combinatorial template kappa_c = A_c - sum B_c + sum C_c, as an engine.

Every shipped combinatorial definition is a configuration of the same five
components (endpoint term, incidence selector, charge, face selector, face
term), assembled by one engine -- the architecture drawn for the combinatorial
family, not just a way of writing the formulas. The undirected and directed
definitions share A_c, B_c and C_c and differ only in I_c and F_c.
"""

import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.curvature.combinatorial import (
    CombinatorialCurvatureConfig,
    CombinatorialCurvatureEngine,
)
from graph_geometry.curvature.forman_directed import (
    augmented_forman_directed_config,
    forman_directed_config,
)
from graph_geometry.curvature.forman_variants import (
    augmented_forman_node_weighted_config,
    forman_node_weighted_config,
)

SRC = Path(__file__).resolve().parents[1] / "src"


def _unit(G):
    for _, _, data in G.edges(data=True):
        data["weight"] = 1.0
    return G


def _weighted(G):
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 0.5 + i % 4
    return G


def _digraph():
    """A digraph with reciprocal edges and feed-forward triangles."""
    return nx.gnp_random_graph(18, 0.25, seed=5, directed=True)


CONFIGS = {
    "forman_node_weighted": forman_node_weighted_config,
    "augmented_forman_node_weighted": augmented_forman_node_weighted_config,
    "forman_directed": forman_directed_config,
    "augmented_forman_directed": augmented_forman_directed_config,
}

#: Each definition declares the graph kind it applies to; the template is the
#: same either way.
GRAPHS = {
    "forman_node_weighted": nx.karate_club_graph,
    "augmented_forman_node_weighted": nx.karate_club_graph,
    "forman_directed": _digraph,
    "augmented_forman_directed": _digraph,
}


@pytest.mark.parametrize("name", list(CONFIGS))
def test_every_combinatorial_built_in_runs_on_the_template_engine(name, monkeypatch):
    calls = []
    original = CombinatorialCurvatureEngine.compute_edges

    def spy(self):
        calls.append(self.config.name)
        return original(self)

    monkeypatch.setattr(CombinatorialCurvatureEngine, "compute_edges", spy)
    gg.compute_curvature(_unit(GRAPHS[name]()), name)
    assert calls == [name]


@pytest.mark.parametrize("name", list(CONFIGS))
def test_the_value_is_exactly_the_assembled_terms(name):
    G = _weighted(GRAPHS[name]())
    terms = CombinatorialCurvatureEngine(G, CONFIGS[name]()).edge_terms()
    values = gg.compute_curvature(G, name).values
    for edge, t in terms.items():
        expected = t["A"] - t["B_x"] - t["B_y"] + (t["C"] or 0.0)
        assert t["kappa"] == values[edge]
        assert t["kappa"] == pytest.approx(expected, abs=1e-12)
        assert (t["C"] is None) == name.startswith("forman")


# ── hand calculations through the template ──────────────────────────────────


def test_unit_weight_identities():
    """Undirected: 4 - deg - deg (+3 per triangle). Directed: 2 - in - out."""
    G = _unit(nx.karate_club_graph())
    deg = dict(G.degree())
    tri = {(u, v): len(set(G[u]) & set(G[v])) for u, v in G.edges()}
    nf = gg.compute_curvature(G, "forman_node_weighted").values
    naf = gg.compute_curvature(G, "augmented_forman_node_weighted").values
    for u, v in G.edges():
        assert nf[(u, v)] == pytest.approx(4 - deg[u] - deg[v])
        assert naf[(u, v)] == pytest.approx(4 - deg[u] - deg[v] + 3 * tri[(u, v)])

    D = _unit(_digraph())
    df = gg.compute_curvature(D, "forman_directed").values
    for x, y in D.edges():
        assert df[(x, y)] == pytest.approx(2 - D.in_degree(x) - D.out_degree(y))


def test_weighted_triangle_face_term_is_w_e_squared_over_face_weight():
    """NAF on a weighted triangle: A = m_u + m_v, no charged edges, C = w_e^2 / w_f.

    Every neighbour of an endpoint closes the triangle, so I_c is empty.
    """
    G = nx.Graph()
    G.add_edge(0, 1, weight=3.0)
    G.add_edge(1, 2, weight=2.0)
    G.add_edge(0, 2, weight=5.0)
    for face_weight in (1.0, 2.0):
        values = gg.compute_curvature(G, gg.CurvatureRequest(
            "augmented_forman_node_weighted", {"face_weight": face_weight})).values
        assert values[(0, 1)] == pytest.approx(2.0 + 9.0 / face_weight)
        assert values[(0, 2)] == pytest.approx(2.0 + 25.0 / face_weight)


def test_directed_incidence_by_hand_on_a_weighted_edge():
    """kappa(0->1) = m_0 + m_1 - sum_{delta^-(0)} sqrt(w_e/w') - sum_{delta^+(1)} sqrt(w_e/w')."""
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=4.0)
    G.add_edge(1, 0, weight=1.0)    # reciprocal: charged at both endpoints
    G.add_edge(2, 0, weight=9.0)    # in-edge of the tail
    G.add_edge(0, 3, weight=2.0)    # out-edge of the tail: not charged
    G.add_edge(1, 4, weight=5.0)    # out-edge of the head
    tail = math.sqrt(4 / 1) + math.sqrt(4 / 9)      # (1,0) and (2,0)
    head = math.sqrt(4 / 1) + math.sqrt(4 / 5)      # (1,0) and (1,4)
    value = gg.compute_curvature(G, "forman_directed").values[(0, 1)]
    assert value == pytest.approx(2.0 - tail - head)


# ── determinism ─────────────────────────────────────────────────────────────


def test_results_do_not_depend_on_the_python_hash_seed():
    """Sums over sets of string labels used to follow per-process hash order."""
    script = textwrap.dedent("""
        import hashlib, networkx as nx, graph_geometry as gg, warnings
        warnings.simplefilter("ignore")
        D = nx.gnp_random_graph(30, 0.2, seed=2, directed=True)
        D = nx.relabel_nodes(D, {n: f"gene_{n}" for n in D})
        for i, (u, v) in enumerate(D.edges()):
            D[u][v]["weight"] = 0.3 + (i % 7) * 0.37
        U = nx.Graph(D)
        C = D.subgraph(max(nx.strongly_connected_components(D), key=len)).copy()
        out = []
        out += gg.compute_curvature(D, "augmented_forman_directed").values.values()
        out += gg.compute_curvature(U, "augmented_forman_node_weighted").values.values()
        out += gg.compute_curvature(C, gg.CurvatureRequest("ollivier", {"kernel": "out"})).values.values()
        print(hashlib.sha256(repr(list(out)).encode()).hexdigest())
    """)
    digests = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(SRC)}
        run = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env=env, check=True)
        digests.add(run.stdout.strip())
    assert len(digests) == 1


# ── a new definition is a new configuration ─────────────────────────────────


def test_a_plugin_can_be_a_template_configuration():
    """Degree-gap curvature: A = 2, charge 1 per incident edge, no faces."""
    config = CombinatorialCurvatureConfig(
        name="degree_gap_template",
        endpoint=lambda ctx, e: 2.0,
        incidence=lambda ctx, z, e: [(z, m) for m in ctx.G.neighbors(z) if m not in e],
        charge=lambda ctx, e, e2, z: 1.0,
    )

    @gg.register_curvature(
        "degree_gap_template",
        capabilities=gg.CurvatureCapabilities(
            family="combinatorial", status="experimental",
            graph_kinds=frozenset({"undirected"}), scopes=frozenset({"edges"}),
            consumes=frozenset({"topology"}),
        ),
    )
    def degree_gap(G, *, semantics, proc):
        return CombinatorialCurvatureEngine(G, config, semantics).compute_edges()

    try:
        G = nx.path_graph(4)
        values = gg.compute_curvature(G, "degree_gap_template").values
        assert dict(values) == {(0, 1): 1.0, (1, 2): 0.0, (2, 3): 1.0}
    finally:
        gg.unregister_curvature("degree_gap_template")


def test_faces_and_face_term_come_together():
    with pytest.raises(gg.CurvatureConfigurationError, match="together"):
        CombinatorialCurvatureConfig(
            name="half", endpoint=lambda c, e: 0.0, incidence=lambda c, z, e: (),
            charge=lambda c, e, f, z: 0.0, faces=lambda c, e: (),
        )
