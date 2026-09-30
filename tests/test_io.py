"""Snapshot IO / SnapshotSeries tests."""

import csv
import json

import networkx as nx

import graph_geometry as gg
from graph_geometry import io


def _series():
    sim = gg.RicciFlowSimulator.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)]).run(
        iterations=6, step=0.05
    )
    return gg.SnapshotSeries(sim.snapshots, sim.convergence)


def test_series_indexing_and_endpoints():
    s = _series()
    assert len(s) == 7
    assert s[0] is s.initial
    assert s[-1] is s.final
    assert list(s)[0] is s.snapshots[0]


def test_save_gexf_snapshots(tmp_path):
    s = _series()
    s.save_gexf(str(tmp_path / "snaps"))
    files = sorted((tmp_path / "snaps").glob("*.gexf"))
    assert len(files) == len(s)
    G0 = nx.read_gexf(str(files[0]))
    assert G0.number_of_edges() == 4


def test_save_convergence_csv(tmp_path):
    s = _series()
    p = tmp_path / "conv.csv"
    s.save_convergence(str(p))
    rows = list(csv.reader(p.open()))
    assert rows[0] == ["iteration", "rc_diff"]
    assert len(rows) == len(s.convergence) + 1


def test_save_edge_curvatures_csv(tmp_path):
    s = _series()
    p = tmp_path / "edges.csv"
    s.save_edge_curvatures(str(p))
    rows = list(csv.DictReader(p.open()))
    assert len(rows) == s.final.number_of_edges()
    assert "ricciCurvature" in rows[0] and "weight" in rows[0]


def test_json_roundtrip(tmp_path):
    s = _series()
    p = tmp_path / "snaps.json"
    s.save_json(str(p))
    data = json.loads(p.read_text())
    assert len(data["snapshots"]) == len(s)
    assert data["convergence"] == list(s.convergence)
    # reconstruct first snapshot
    G0 = nx.node_link_graph(data["snapshots"][0], edges="links")
    assert G0.number_of_edges() == 4


def test_standalone_functions(tmp_path):
    s = _series()
    io.save_gexf_snapshots(s.snapshots, str(tmp_path / "s"))
    io.save_convergence_csv(s.convergence, str(tmp_path / "c.csv"))
    io.save_json(s.snapshots, str(tmp_path / "j.json"), s.convergence)
    assert (tmp_path / "c.csv").exists() and (tmp_path / "j.json").exists()

# ── export alignment and flow/curvature exporters (audit fixes) ─────────────


def test_convergence_csv_iterations_are_committed_state_indices(tmp_path):
    """convergence[i - 1] belongs to snapshots[i]; the CSV used to start at 0."""
    s = _series()
    p = tmp_path / "conv.csv"
    s.save_convergence(str(p))
    rows = list(csv.reader(p.open()))[1:]
    assert [int(r[0]) for r in rows] == list(range(1, len(s.convergence) + 1))
    assert float(rows[-1][1]) == s.convergence[-1]


def test_save_flow_result_records_why_the_run_stopped(tmp_path):
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)], directed=True)
    request = gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed"})
    result = gg.RicciFlow(G, curvature=request).run(iterations=3, step=0.05, early_stop=False)
    io.save_flow_result(result, str(tmp_path / "run"))
    record = json.loads((tmp_path / "run" / "flow.json").read_text())
    assert record["termination_reason"] == "iterations"
    assert record["iterations_completed"] == 3
    assert record["resolved"]["direction_convention"] == "mixed"
    assert [row[0] for row in record["convergence"]] == [1, 2, 3]
    assert len(list((tmp_path / "run" / "snapshots").glob("*.gexf"))) == 4
    series = gg.SnapshotSeries.from_result(result)
    series.save(str(tmp_path / "again"))
    assert (tmp_path / "again" / "flow.json").exists()


def test_save_curvature_csv_writes_values_and_node_means(tmp_path):
    G = nx.cycle_graph(5)
    result = gg.compute_curvature(G, "ollivier")
    io.save_curvature_csv(result, str(tmp_path / "k.csv"), node_means_path=str(tmp_path / "n.csv"))
    rows = list(csv.reader((tmp_path / "k.csv").open()))
    assert rows[0] == ["source", "target", "kappa"] and len(rows) == 6
    assert result.rows()[0] == {"source": 0, "target": 1, "kappa": result.values[(0, 1)]}
    assert len(list(csv.reader((tmp_path / "n.csv").open()))) == 6
