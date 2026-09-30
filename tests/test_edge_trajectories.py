"""Edge trajectories: the data layer (always) and the plot (with the [viz] extra)."""

import csv
import copy
import io
import math

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.flow.trajectories import TRAJECTORY_FIELDS
from graph_geometry.io import save_edge_trajectories_csv


def _barbell_flow(**run):
    G = nx.barbell_graph(4, 0)
    kwargs = dict(iterations=4, step=0.1, early_stop=False)
    kwargs.update(run)
    return gg.RicciFlow(G, curvature="lin_lu_yau").run(**kwargs)


@pytest.fixture(scope="module")
def flow():
    return _barbell_flow()


@pytest.fixture(scope="module")
def surgery_flow():
    return _barbell_flow(surgery={"name": "surgery_n", "interval": 1, "portion": 1})


# ── alignment ───────────────────────────────────────────────────────────────


def test_iteration_zero_is_the_initial_graph(flow):
    data = gg.extract_edge_trajectories(flow)
    G0 = flow.snapshots[0]
    assert data.n_states == len(flow.snapshots) == flow.iterations_completed + 1
    assert set(map(frozenset, data.edges)) == set(map(frozenset, G0.edges()))
    for u, v in data.edges:
        assert data.curvature[(u, v)][0] == G0[u][v]["ricciCurvature"]
        assert data.evolving_quantity[(u, v)][0] == G0[u][v]["weight"] == 1.0


def test_curvature_and_evolving_quantity_read_the_same_state(flow):
    data = gg.extract_edge_trajectories(flow)
    rows = data.rows()
    assert len(rows) == data.n_states * len(data.edges)
    for row in rows:
        S = flow.snapshots[row["iteration"]]
        edge = S[row["source"]][row["target"]]
        assert row["present"] is True
        assert row["curvature"] == edge["ricciCurvature"]
        assert row["evolving_quantity"] == edge["weight"]
    # and the stored curvature is the curvature *of that state*
    last = flow.snapshots[-1]
    fresh = gg.compute_curvature(last, "lin_lu_yau").values
    for (u, v), value in fresh.items():
        assert data.curvature[data.canonical_edge((u, v))][-1] == pytest.approx(value, abs=1e-9)


def test_values_of_present_edges_are_finite(flow):
    data = gg.extract_edge_trajectories(flow)
    for e in data.edges:
        for t, here in enumerate(data.present[e]):
            if here:
                assert math.isfinite(data.curvature[e][t])
                assert math.isfinite(data.evolving_quantity[e][t])


def test_extraction_does_not_mutate_the_result(flow):
    before = [copy.deepcopy(dict(((u, v), dict(d)) for u, v, d in G.edges(data=True)))
              for G in flow.snapshots]
    n_before = len(flow.snapshots)
    data = gg.extract_edge_trajectories(flow, groups={(0, 1): "g"})
    data.rows()
    data.aggregate("curvature")
    assert len(flow.snapshots) == n_before
    after = [dict(((u, v), dict(d)) for u, v, d in G.edges(data=True)) for G in flow.snapshots]
    assert after == before


# ── edge identities ─────────────────────────────────────────────────────────


def test_undirected_edges_are_canonicalised(flow):
    data = gg.extract_edge_trajectories(flow, edges=[(1, 0), (0, 1), (3, 4)])
    assert len(data.edges) == 2
    assert data.canonical_edge((1, 0)) == data.canonical_edge((0, 1))
    assert data.canonical_edge((4, 3)) in data.edges
    assert data.edge_label((1, 0)) == data.edge_label((0, 1))


def test_directed_edges_keep_orientation():
    G = nx.DiGraph()
    G.add_edge("a", "b", weight=1.0, ricciCurvature=0.1)
    G.add_edge("b", "a", weight=2.0, ricciCurvature=-0.2)
    H = G.copy()
    H["a"]["b"]["weight"] = 1.5
    data = gg.extract_edge_trajectories([G, H], evolving_attr="weight", evolve="weight")
    assert set(data.edges) == {("a", "b"), ("b", "a")}
    assert data.evolving_quantity[("a", "b")] == (1.0, 1.5)
    assert data.evolving_quantity[("b", "a")] == (2.0, 2.0)
    assert data.edge_label(("a", "b")) == "'a' → 'b'"
    with pytest.raises(ValueError, match="does not occur"):
        gg.extract_edge_trajectories([G], edges=[("a", "c")])


def test_mixed_node_identifiers():
    G = nx.Graph()
    G.add_edge(0, "x", weight=1.0, ricciCurvature=0.5)
    G.add_edge(("t", 1), 0, weight=2.0, ricciCurvature=0.25)
    data = gg.extract_edge_trajectories([G], evolving_attr="weight")
    assert data.canonical_edge(("x", 0)) in data.edges
    assert data.canonical_edge((0, ("t", 1))) in data.edges


def test_selected_edge_filtering(flow):
    data = gg.extract_edge_trajectories(flow, edges=[(0, 1)])
    assert data.edges == (data.canonical_edge((0, 1)),)
    assert len(data.rows()) == data.n_states
    with pytest.raises(ValueError, match="does not occur"):
        gg.extract_edge_trajectories(flow, edges=[(0, 7)])


# ── groups ──────────────────────────────────────────────────────────────────


def test_explicit_grouping_is_validated_against_the_initial_graph(flow):
    groups = {(0, 1): "clique", (2, 1): "clique", (3, 4): "bridge"}
    data = gg.extract_edge_trajectories(flow, groups=groups)
    members = data.group_members()
    assert list(members) == ["clique", "bridge"]
    assert len(members["clique"]) == 2 and len(members["bridge"]) == 1
    assert {r["group"] for r in data.rows() if r["source"] in (5, 6, 7)} == {None}
    with pytest.raises(ValueError, match="initial state"):
        gg.extract_edge_trajectories(flow, groups={(0, 7): "nope"})
    with pytest.raises(ValueError, match="two groups"):
        gg.extract_edge_trajectories(flow, groups={(0, 1): "a", (1, 0): "b"})


def test_group_aggregation_is_the_mean_of_present_members(flow):
    groups = {(0, 1): "g", (0, 2): "g"}
    data = gg.extract_edge_trajectories(flow, groups=groups)
    mean = data.aggregate("evolving_quantity")["g"]
    a, b = data.canonical_edge((0, 1)), data.canonical_edge((0, 2))
    for t in range(data.n_states):
        expected = (data.evolving_quantity[a][t] + data.evolving_quantity[b][t]) / 2
        assert mean[t] == pytest.approx(expected)
    with pytest.raises(ValueError):
        data.aggregate("curvature", how="mode")


# ── run length and surgery ──────────────────────────────────────────────────


def test_early_termination_gives_short_series():
    result = gg.RicciFlow(nx.complete_graph(4)).run(iterations=20, step=0.1)
    assert result.termination_reason == "converged"
    data = gg.extract_edge_trajectories(result)
    assert data.n_states == len(result.snapshots) < 21
    assert all(len(v) == data.n_states for v in data.curvature.values())


def test_surgery_marks_removed_edges_missing_not_zero(surgery_flow):
    data = gg.extract_edge_trajectories(surgery_flow)
    removed = [e for e in data.edges if not data.present[e][-1]]
    assert removed, "the surgery run should remove at least one edge"
    for e in removed:
        gone = data.present[e].index(False)
        assert all(not p for p in data.present[e][gone:]), "surgery never re-adds edges"
        assert all(v is None for v in data.curvature[e][gone:])
        assert all(v is None for v in data.evolving_quantity[e][gone:])
        assert data.curvature[e][gone - 1] is not None
    # group summaries use the edges still present
    e = removed[0]
    keep = next(x for x in data.edges if data.present[x][-1])
    grouped = gg.extract_edge_trajectories(surgery_flow, groups={e: "g", keep: "g"})
    mean = grouped.aggregate("evolving_quantity")["g"]
    assert mean[-1] == pytest.approx(grouped.evolving_quantity[keep][-1])


def test_rows_round_trip_through_csv(surgery_flow, tmp_path):
    data = gg.extract_edge_trajectories(surgery_flow, groups={(0, 1): "g"})
    path = tmp_path / "traj.csv"
    save_edge_trajectories_csv(data, path)
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == TRAJECTORY_FIELDS
    assert len(rows) == len(data.rows())
    missing = [r for r in rows if r["present"] == "False"]
    assert missing and all(r["curvature"] == "" for r in missing)
    buffer = io.StringIO()
    save_edge_trajectories_csv(data, buffer)
    assert buffer.getvalue().replace("\r\n", "\n") == path.read_text().replace("\r\n", "\n")


# ── roles recorded on FlowResult ────────────────────────────────────────────


@pytest.mark.parametrize("evolve, label", [("weight", "edge weight"),
                                           ("distance", "edge distance")])
def test_evolving_role_and_label(evolve, label):
    result = gg.RicciFlow(nx.cycle_graph(5), evolve=evolve).run(iterations=2, step=0.05)
    assert result.evolve == evolve
    assert result.evolving_attr == evolve
    assert result.to_dict()["evolving_attr"] == evolve
    data = gg.extract_edge_trajectories(result)
    assert data.quantity_label("evolving_quantity") == label
    assert data.quantity_label("curvature") == "edge curvature"
    coupled = gg.RicciFlow(nx.cycle_graph(5), distance="weight").run(iterations=1, step=0.05)
    assert coupled.coupled
    assert gg.extract_edge_trajectories(coupled).quantity_label("evolving") == \
        "edge weight (= distance)"


def test_hand_built_flow_result_stays_compatible():
    G = nx.path_graph(3)
    nx.set_edge_attributes(G, 0.0, "ricciCurvature")
    result = gg.flow.FlowResult(snapshots=[G], convergence=[])
    assert result.evolving_attr is None and not result.coupled
    data = gg.extract_edge_trajectories(result)
    assert data.quantity_label("evolving") == "evolving edge quantity"
    assert all(v == (None,) for v in data.evolving_quantity.values())


# ── plotting ────────────────────────────────────────────────────────────────


@pytest.fixture
def mpl():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    yield plt
    plt.close("all")


def test_plot_into_caller_axes_with_role_labels(mpl, flow):
    from graph_geometry import viz

    fig, (a, b) = mpl.subplots(1, 2)
    assert viz.plot_edge_trajectories(flow, ax=a) is fig
    assert viz.plot_edge_trajectories(flow, quantity="evolving_quantity", ax=b) is fig
    assert a.get_ylabel() == "edge curvature"
    assert b.get_ylabel() == "edge weight"
    assert a.get_xlabel() == "committed iteration"
    distance = gg.RicciFlow(nx.cycle_graph(5), evolve="distance").run(iterations=2, step=0.05)
    fig = viz.plot_edge_trajectories(distance, quantity="evolving_quantity")
    assert fig.axes[0].get_ylabel() == "edge distance"


def test_plot_groups_aggregate_and_highlight(mpl, flow):
    from graph_geometry import viz

    groups = {(0, 1): "left", (0, 2): "left", (3, 4): "bridge"}
    fig = viz.plot_edge_trajectories(
        flow, groups=groups, aggregate=True, highlight_edges=[(2, 1)],
        colors={"bridge": "#D55E00"},
    )
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "left (2 edges)" in labels and "bridge (1 edge)" in labels
    assert "1 — 2" in labels or "2 — 1" in labels
    bridge = next(line for line in ax.get_lines() if line.get_label() == "bridge (1 edge)")
    assert bridge.get_color() == "#D55E00"
    n_edges = flow.snapshots[0].number_of_edges()
    # individual + 2 group summaries + ungrouped summary + 1 highlight
    assert len(ax.get_lines()) == n_edges + 3 + 1


def test_plot_leaves_gaps_for_removed_edges(mpl, surgery_flow):
    from graph_geometry import viz

    data = gg.extract_edge_trajectories(surgery_flow)
    removed = next(e for e in data.edges if not data.present[e][-1])
    fig = viz.plot_edge_trajectories(surgery_flow, edges=[removed])
    y = fig.axes[0].get_lines()[0].get_ydata()
    assert math.isnan(y[-1]) and not math.isnan(y[0])


def test_legacy_snapshot_call_still_works(mpl, flow):
    from graph_geometry import viz

    fig = viz.plot_edge_trajectories(flow.snapshots, "weight", [(0, 1)])
    assert fig.axes[0].get_ylabel() == "evolving edge quantity (weight)"
    fig = viz.plot_edge_trajectories(flow.snapshots)
    assert fig.axes[0].get_ylabel() == "edge curvature"
