"""viz smoke tests — skipped when the [viz] extra (matplotlib) is absent."""

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless

import graph_geometry as gg  # noqa: E402
from graph_geometry import viz  # noqa: E402


@pytest.fixture(scope="module")
def sim():
    return gg.RicciFlowSimulator.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)]).run(
        iterations=10, step=0.05
    )


def test_plot_functions_return_figures(sim):
    Figure = matplotlib.figure.Figure
    assert isinstance(viz.plot_graph(sim.result_graph), Figure)
    assert isinstance(viz.plot_convergence(sim.convergence), Figure)
    assert isinstance(viz.plot_curvature_distribution(sim.snapshots), Figure)
    assert isinstance(viz.plot_weight_distribution(sim.snapshots), Figure)
    fig, pos = viz.plot_flow_comparison(sim.snapshots)
    assert isinstance(fig, Figure)


def test_edge_trajectories_are_aligned_to_committed_snapshots(sim):
    trajectories = viz.edge_trajectories(sim.snapshots, "weight")
    assert set(trajectories) == set(sim.snapshots[0].edges())
    for (u, v), values in trajectories.items():
        assert len(values) == len(sim.snapshots)
        for snapshot, value in zip(sim.snapshots, values):
            assert value == pytest.approx(snapshot[u][v]["weight"])


def test_edge_trajectories_mark_surgery_removal_as_missing():
    G0 = gg.load.from_edges([("a", "b", 1.0), ("b", "c", 2.0)], directed=False)
    G1 = G0.copy()
    G1["a"]["b"]["weight"] = 1.5
    G1.remove_edge("b", "c")

    trajectories = viz.edge_trajectories([G0, G1], "weight")
    assert trajectories[("a", "b")] == [1.0, 1.5]
    assert trajectories[("b", "c")] == [2.0, None]


def test_undirected_reverse_edge_is_not_duplicated():
    G = gg.load.from_edges([(0, 1, 1.0)], directed=False)
    trajectories = viz.edge_trajectories([G], "weight", [(0, 1), (1, 0)])
    assert list(trajectories) == [(0, 1)]


def test_animation_and_save(sim, tmp_path):
    ani = viz.animate_flow(sim.snapshots)
    assert ani is not None
    p = tmp_path / "flow.gif"
    viz.save_animation(sim.snapshots, str(p), fps=4)
    assert p.exists() and p.stat().st_size > 0

def test_curvature_wise_and_flow_wise_plots(sim):
    import networkx as nx

    Figure = matplotlib.figure.Figure
    G = nx.cycle_graph(6)
    result = gg.compute_curvature(G, "ollivier")
    assert isinstance(viz.plot_curvature_histogram(result), Figure)
    assert isinstance(viz.plot_curvature_histogram(dict(result.values)), Figure)
    comparison = gg.compare_curvature(G, ["ollivier", "forman_node_weighted", "eidi_jost"])
    assert isinstance(viz.plot_curvature_comparison(comparison), Figure)
    flows = gg.compare_flow(G, {"n": "normalized", "a": "additive"}, curvature="ollivier",
                            iterations=3)
    assert isinstance(viz.plot_convergence_comparison(flows, log_scale=True), Figure)
    assert isinstance(viz.plot_edge_trajectories(sim.snapshots), Figure)
    matplotlib.pyplot.close("all")
