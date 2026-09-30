"""RicciFlowSimulator facade tests."""

import networkx as nx
import pytest

import graph_geometry as gg


def _assign(G):
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    return G


def test_from_edges_runs_and_exposes_results():
    sim = gg.RicciFlowSimulator.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)])
    sim.run(iterations=15, step=0.05)
    assert len(sim.convergence) == 15
    assert len(sim.snapshots) == 16
    assert sim.initial_graph is not None and sim.result_graph is not None
    assert sim.snapshot_at(0) is sim.snapshots[0]


def test_accessors_none_before_run():
    sim = gg.RicciFlowSimulator.from_edges([(0, 1), (1, 2), (2, 0)])
    assert sim.initial_graph is None
    assert sim.result_graph is None
    with pytest.raises(RuntimeError):
        sim.snapshot_at(0)


def test_from_networkx_does_not_mutate_source():
    src = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    gg.RicciFlowSimulator.from_networkx(src).run(iterations=5, step=0.05)
    assert "ricciCurvature" not in src[0][1]


def test_facade_matches_engine_exactly():
    def mk():
        return _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))

    se, ce = gg.RicciFlow(mk(), curvature="lin_lu_yau").run(iterations=25, step=0.05)
    sim = gg.RicciFlowSimulator(mk(), curvature="lin_lu_yau").run(iterations=25, step=0.05)
    assert ce == pytest.approx(sim.convergence, abs=1e-12)
    for u, v in se[-1].edges():
        assert se[-1][u][v]["weight"] == pytest.approx(sim.result_graph[u][v]["weight"], abs=1e-12)


def test_structure_only_edges_get_defaults():
    # from_edges returns structure only; the engine fills weight/distance
    sim = gg.RicciFlowSimulator.from_edges([(0, 1), (1, 2), (2, 0)]).run(iterations=3, step=0.05)
    assert all("weight" in sim.result_graph[u][v] for u, v in sim.result_graph.edges())


def test_from_edgelist_routes_loader_columns_instead_of_flow_kwargs(tmp_path):
    path = tmp_path / "weighted.edgelist"
    path.write_text("a b 2.5 0.4\nb a 1.5 0.8\n")
    sim = gg.RicciFlowSimulator.from_edgelist(
        str(path), directed=True, columns=("weight", "distance")
    )
    assert sim._G0["a"]["b"]["weight"] == pytest.approx(2.5)
    assert sim._G0["a"]["b"]["distance"] == pytest.approx(0.4)


def test_summary_runs(capsys):
    sim = gg.RicciFlowSimulator.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)]).run(iterations=5, step=0.05)
    sim.summary()
    out = capsys.readouterr().out
    assert "Ricci Flow Simulation Summary" in out


@pytest.mark.parametrize("interval", [1, 2])
def test_summary_reports_undefined_final_spread_after_exhaustion(capsys, interval):
    G = nx.barbell_graph(4, 0)
    sim = gg.RicciFlowSimulator.from_networkx(G, curvature="ollivier").run(
        iterations=3, early_stop=False,
        surgery={"name": "surgery_n", "portion": G.number_of_edges(), "interval": interval},
    )
    assert sim.termination_reason == "exhausted"
    assert sim.result_graph.number_of_edges() == 0
    assert len(sim.convergence) == interval - 1

    sim.summary()
    out = capsys.readouterr().out
    assert "Final RC difference  : undefined" in out
    assert "RC  initial :" in out
    assert "RC  final   : undefined" in out
    assert "w   final   : undefined" in out


@pytest.mark.parametrize("evolve, semantics, symbol, attr", [
    ("weight", gg.GraphSemantics(), "w", "weight"),
    ("distance", gg.GraphSemantics(), "d", "distance"),
    ("distance", gg.GraphSemantics(weight_attr="strength", distance_attr="length"), "d", "length"),
    ("distance", gg.GraphSemantics(weight_attr="weight", distance_attr="weight"), "d", "weight"),
])
def test_summary_uses_the_recorded_evolving_quantity(capsys, evolve, semantics, symbol, attr):
    sim = gg.RicciFlowSimulator.from_networkx(
        nx.barbell_graph(4, 0), curvature="ollivier", evolve=evolve, semantics=semantics,
    ).run(iterations=3, step=0.1, early_stop=False)
    values = list(nx.get_edge_attributes(sim.result_graph, attr).values())
    assert min(values) < max(values)
    assert sim.result.evolving_attr == attr

    sim.summary()
    out = capsys.readouterr().out
    assert f"Evolving quantity    : {evolve} ({attr})" in out
    assert f"{symbol}   final   : [{min(values):.4f}, {max(values):.4f}]" in out
    if evolve == "distance":
        assert "w   final" not in out


def test_summary_uses_initial_state_when_no_updates_were_committed(capsys):
    sim = gg.RicciFlowSimulator.from_networkx(
        nx.barbell_graph(4, 0), curvature="ollivier",
    ).run(iterations=0)
    assert sim.convergence == []
    rc = list(nx.get_edge_attributes(sim.result_graph, "ricciCurvature").values())
    assert max(rc) > min(rc)

    sim.summary()
    out = capsys.readouterr().out
    assert f"Final RC difference  : {max(rc) - min(rc):.6f}" in out
