"""Experiment driver tests (dict / JSON / YAML configs, legacy aliases)."""

import json
from pathlib import Path

import pytest

import graph_geometry as gg
from graph_geometry.experiment import load_config, run_experiment

REPO = Path(__file__).resolve().parents[1]  # graph_geometry/
DATA_ROOT = REPO / "data"


def _data_available() -> bool:
    """Data present AND readable (iCloud can evict files to empty placeholders)."""
    sentinel = DATA_ROOT / "mygraph" / "mygraph_3cycle4edge.edgelist"
    try:
        return bool(sentinel.read_bytes().strip())
    except OSError:
        return False


def _edges_config(**over):
    """A config in the canonical schema."""
    cfg = {
        "data": {
            "graph_kind": "directed",
            "edges": [[0, 1], [1, 2], [2, 0], [0, 2]],
            "factors": {"edge_weight": "fixed", "edge_distance": "fixed"},
        },
        "curvature": {"method": "lin_lu_yau", "parameters": {"kernel": "mixed", "beta": 0.8}},
        "flow": {"iterations": 12, "step": 0.05},
    }
    cfg.update(over)
    return cfg


def test_run_from_dict_edges():
    res = run_experiment(_edges_config())
    assert len(res["convergence"]) == 12
    assert res["snapshots"][0].number_of_nodes() == 3
    assert isinstance(res["series"], gg.SnapshotSeries)


def test_run_from_generator():
    cfg = {
        "data": {"graph_kind": "undirected", "generator": "complete", "params": {"n": 4},
                 "factors": {"edge_weight": "fixed"}},
        "curvature": {"method": "lin_lu_yau"},
        "flow": {"iterations": 5, "step": 0.05},
    }
    res = run_experiment(cfg)
    assert res["snapshots"][0].number_of_nodes() == 4


def test_legacy_factor_aliases_and_surgery_param():
    # old-style keys still work, each with a FutureWarning naming the new form
    cfg = {
        "data": {
            "edges": [[0, 1], [1, 2], [2, 0], [0, 2]],
            "directed": True,
            "edge_weight_type": "fixed",
            "edge_distance_type": "fixed",
            "node_weight_type": "fixed",
        },
        "curvature": {"method": "lin_lu_yau", "kernel": "mixed", "beta": 0.8},
        "ricciflow": {"iterations": 6, "step": 0.05, "flow_equation": "normalized"},
        "surgery_param": {"surgery": False},  # disabled
    }
    with pytest.warns(FutureWarning) as record:
        res = run_experiment(cfg)
    assert len(res["convergence"]) == 6
    messages = " ".join(str(w.message) for w in record)
    for legacy in ("data.directed", "flat factor keys", "flat curvature parameters",
                   "ricciflow", "flow_equation", "surgery_param"):
        assert legacy in messages, legacy
    assert len(res["resolved"]["legacy_forms"]) == 6
    canonical = res["canonical_config"]
    assert canonical["data"]["graph_kind"] == "directed"
    assert canonical["curvature"]["parameters"] == {"kernel": "mixed", "beta": 0.8}


def test_forman_method_via_experiment():
    # Forman on this graph converges immediately; disable early-stop to run fully
    cfg = _edges_config(curvature={"method": "forman_directed"})
    cfg["flow"]["early_stop"] = False
    res = run_experiment(cfg)
    assert len(res["convergence"]) == 12
    assert all("ricciCurvature" in res["snapshots"][-1][u][v] for u, v in res["snapshots"][-1].edges())


def test_flow_equation_and_evolve_via_config():
    # the equation may be a preset name or an expression string; evolve selects
    # the evolving quantity — both settable from the config
    cfg = _edges_config()
    cfg["flow"]["equation"] = "-eta*(kappa - kbar)*q"
    cfg["flow"]["evolve"] = "distance"
    cfg["flow"]["early_stop"] = False
    res = run_experiment(cfg)
    Gf = res["snapshots"][-1]
    assert all(Gf[u][v]["weight"] == 1.0 for u, v in Gf.edges())          # weight fixed
    assert any(abs(Gf[u][v]["distance"] - 1.0) > 1e-9 for u, v in Gf.edges())


def test_output_dir_writes_files(tmp_path):
    cfg = _edges_config(output={"dir": str(tmp_path / "run1")})
    run_experiment(cfg)
    assert (tmp_path / "run1" / "convergence.csv").exists()
    assert list((tmp_path / "run1" / "snapshots").glob("*.gexf"))
    record = json.loads((tmp_path / "run1" / "resolved_semantics.json").read_text())
    assert record["curvature"]["method"] == "lin_lu_yau"
    assert record["curvature"]["direction_convention"] == "mixed"


def test_load_config_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_edges_config()))
    cfg = load_config(p)
    assert cfg["data"]["graph_kind"] == "directed"


def test_load_config_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("data:\n  edges: [[0,1],[1,2],[2,0]]\nricciflow:\n  iterations: 3\n")
    cfg = load_config(str(p))
    assert cfg["ricciflow"]["iterations"] == 3


def test_missing_graph_source_raises():
    with pytest.raises(ValueError):
        run_experiment({"data": {"edge_weight": "fixed"}, "ricciflow": {"iterations": 1}})


# ── explicit semantics and resolved provenance ──────────────────────────────


def test_resolved_record_carries_every_convention():
    cfg = _edges_config()
    cfg["curvature"]["method"] = "ollivier"
    cfg["curvature"]["label"] = "ORC mixed"
    cfg["curvature"]["parameters"] = {"kernel": "mixed", "beta": 0.8, "alpha": 0.5}
    cfg["semantics"] = {"weight_attr": "weight", "distance_attr": "distance"}
    cfg["execution"] = {"proc": 1}
    cfg["flow"]["iterations"] = 2
    record = run_experiment(cfg)["resolved"]
    assert json.loads(json.dumps(record)) == record      # JSON-safe
    curvature = record["curvature"]
    assert curvature["requested_method"] == "ollivier"
    assert curvature["method"] == "ollivier"
    assert curvature["label"] == "ORC mixed"
    assert curvature["graph_kind"] == "directed"
    assert curvature["direction_convention"] == "mixed"
    assert curvature["parameters"] == {
        "alpha": 0.5, "kernel": "mixed", "beta_strategy": "constant", "beta": 0.8,
    }
    assert record["semantics"]["weight_attr"] == "weight"
    assert record["execution"] == {"proc": 1}
    assert record["data"]["graph_kind"] == "directed"


def test_requested_alias_is_recorded_with_its_canonical_name():
    cfg = {
        "data": {"graph_kind": "undirected", "generator": "complete",
                 "params": {"n": 4}, "factors": {"edge_weight": "fixed"}},
        "curvature": {"method": "forman_sreejith"},
        "flow": {"iterations": 1, "step": 0.05},
    }
    with pytest.warns(gg.CurvatureDeprecationWarning):
        record = run_experiment(cfg)["resolved"]
    assert record["curvature"]["requested_method"] == "forman_sreejith"
    assert record["curvature"]["method"] == "forman_node_weighted"
    assert any("deprecated" in w for w in record["warnings"])


@pytest.mark.parametrize("parameters", [{}, {"kernel": "auto"}, {"kernel": "auto", "beta": 0.8}])
def test_directed_transport_experiment_cannot_persist_an_implicit_kernel(parameters):
    cfg = _edges_config(curvature={"method": "lin_lu_yau", "parameters": parameters})
    with pytest.raises(gg.CurvatureConfigurationError, match="explicit for a directed"):
        run_experiment(cfg)


def test_graph_kind_is_required_for_edges():
    cfg = _edges_config()
    del cfg["data"]["graph_kind"]
    with pytest.raises(gg.CurvatureConfigurationError, match="graph_kind is required"):
        run_experiment(cfg)


def test_graph_kind_must_match_the_generated_graph():
    cfg = {
        "data": {"graph_kind": "directed", "generator": "complete", "params": {"n": 4}},
        "curvature": {"method": "lin_lu_yau", "parameters": {"kernel": "mixed"}},
        "flow": {"iterations": 1},
    }
    with pytest.raises(gg.CurvatureConfigurationError, match="produced an undirected graph"):
        run_experiment(cfg)


def test_undirected_edges_experiment_records_undirected_convention():
    cfg = _edges_config(curvature={"method": "lin_lu_yau"})
    cfg["data"]["graph_kind"] = "undirected"
    cfg["flow"]["iterations"] = 1
    record = run_experiment(cfg)["resolved"]
    assert record["curvature"]["graph_kind"] == "undirected"
    assert record["curvature"]["direction_convention"] == "undirected"
    assert "beta" not in record["curvature"]["parameters"]


def test_inactive_parameter_is_rejected_in_a_persisted_experiment():
    cfg = _edges_config(curvature={"method": "ollivier",
                                   "parameters": {"kernel": "out", "beta": 0.3}})
    with pytest.raises(gg.CurvatureConfigurationError, match="inactive"):
        run_experiment(cfg)


def test_beta_strategy_is_persisted_by_an_experiment(tmp_path):
    cfg = _edges_config(output={"dir": str(tmp_path / "run")})
    cfg["curvature"]["parameters"] = {"kernel": "mixed", "beta_strategy": "degree_proportional"}
    cfg["flow"]["iterations"] = 2
    run_experiment(cfg)
    record = json.loads((tmp_path / "run" / "resolved_semantics.json").read_text())
    assert record["curvature"]["parameters"]["beta_strategy"] == "degree_proportional"
    assert "beta" not in record["curvature"]["parameters"]
    assert record["semantics"]["beta_attr"] == "beta"


def test_explicit_semantics_rename_the_attributes():
    cfg = _edges_config(semantics={"weight_attr": "strength", "distance_attr": "length"})
    cfg["flow"]["iterations"] = 2
    res = run_experiment(cfg)
    G0 = res["snapshots"][0]
    assert all({"strength", "length"} <= set(G0[u][v]) for u, v in G0.edges())
    assert res["resolved"]["semantics"]["weight_attr"] == "strength"


@pytest.mark.skipif(not _data_available(), reason="research data missing or not materialized")
def test_migrated_yaml_config_runs():
    # the shipped mygraph config, but shortened for speed
    cfg = load_config(str(REPO / "configs" / "mygraph_3cycle4edge.yaml"))
    cfg["data"]["data_root"] = str(DATA_ROOT)
    assert cfg["data"]["graph_kind"] == "directed"
    assert cfg["curvature"]["parameters"]["kernel"] == "mixed"
    cfg["flow"]["iterations"] = 5
    res = run_experiment(cfg)
    assert res["snapshots"][0].number_of_nodes() == 3
