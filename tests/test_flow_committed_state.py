"""The per-iteration committed-state contract.

An iteration produces exactly one committed state, and the snapshot, the
convergence value, ``result_graph``, the engine's live graph and the iteration
GEXF must all describe *that* state. These tests drive the two early-stop
checkpoints and the surgery paths deterministically, with curvature methods
whose values are chosen by the test rather than by a real flow, so that each
branch of the order in ``flow/engine.py`` is exercised on purpose instead of by
luck.
"""

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.curvature import write_curvature


# ── deterministic doubles ────────────────────────────────────────────────────


def _assign(G):
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    return G


def constant_curvature(G, *, weight="weight", distance="distance", proc=1, **kw):
    """Spread 0 on any topology: every checkpoint sees a converged graph."""
    return write_curvature(G, {e: 1.0 for e in G.edges()})


def staged_curvature(G, *, weight="weight", distance="distance", proc=1, **kw):
    """Spread 0 once the graph is small enough, large while it is not.

    Lets a test make the *pre*-surgery candidate unconverged and the
    *post*-surgery graph converged, which is the only way to reach the second
    checkpoint.
    """
    if G.number_of_edges() <= 3:
        return write_curvature(G, {e: 1.0 for e in G.edges()})
    return write_curvature(G, {e: float(i) for i, e in enumerate(G.edges())})


class CountingSurgery:
    """Surgery on demand, counting how often it actually ran."""

    def __init__(self, cut=1, on=None):
        self.cut = cut
        self.on = on          # None = every iteration
        self.applied = 0

    def should_apply(self, iteration):
        # ``iteration`` is the 1-based count of completed updates
        return True if self.on is None else iteration in self.on

    def apply(self, G, attribute):
        self.applied += 1
        return gg.flow.surgery_n(G, attribute, self.cut)


def _spread(G):
    rc = nx.get_edge_attributes(G, "ricciCurvature")
    return max(rc.values()) - min(rc.values()) if rc else None


# ── the two early-stop checkpoints ───────────────────────────────────────────


def test_pre_surgery_early_stop_prevents_scheduled_surgery():
    """An already converged graph must not be operated on."""
    strategy = CountingSurgery(cut=1)
    engine = gg.RicciFlow(_assign(nx.complete_graph(5)), curvature=constant_curvature)
    result = engine.run(iterations=6, step=0.1, delta=1e-6, surgery=strategy)

    assert result.termination_reason == "converged"
    assert strategy.applied == 0, "surgery ran on a converged graph"
    assert result.iterations_completed == 1
    assert result.snapshots[-1].number_of_edges() == 10


def test_post_surgery_early_stop_ends_the_iteration_that_produced_it():
    """Surgery can itself create convergence; that must be caught at once."""
    strategy = CountingSurgery(cut=2)
    G = _assign(nx.cycle_graph(5))            # 5 edges -> unconverged
    engine = gg.RicciFlow(G, curvature=staged_curvature)
    result = engine.run(iterations=6, step=0.05, delta=1e-6, surgery=strategy)

    assert strategy.applied == 1, "surgery should have run exactly once"
    assert result.termination_reason == "converged"
    assert result.iterations_completed == 1
    assert result.snapshots[-1].number_of_edges() == 3
    assert result.convergence[-1] == pytest.approx(0.0)


def test_early_stop_false_skips_both_checkpoints():
    strategy = CountingSurgery(cut=1, on={3})
    engine = gg.RicciFlow(_assign(nx.complete_graph(5)), curvature=constant_curvature)
    result = engine.run(iterations=4, step=0.1, surgery=strategy, early_stop=False)

    assert result.termination_reason == "iterations"
    assert result.iterations_completed == 4
    assert strategy.applied == 1


# ── surgery lands in the committed state ─────────────────────────────────────


def test_final_iteration_surgery_is_in_the_result(monkeypatch):
    """Surgery on the last iteration used to be dropped from ``result_graph``."""
    strategy = CountingSurgery(cut=1, on={3})          # last requested iteration
    G = _assign(nx.complete_graph(5))
    sim = gg.RicciFlowSimulator(G, curvature=gg.ollivier(alpha=0.5))
    sim.run(iterations=3, step=0.05, surgery=strategy, early_stop=False)

    assert strategy.applied == 1
    before, after = sim.snapshots[-2], sim.snapshots[-1]
    assert after.number_of_edges() == before.number_of_edges() - 1
    assert sim.result_graph.number_of_edges() == after.number_of_edges()

    # curvature on the committed state belongs to the post-surgery topology
    fresh = gg.curvature(after.copy(), method=gg.ollivier(alpha=0.5))
    for (u, v), value in fresh.items():
        assert after[u][v]["ricciCurvature"] == pytest.approx(value, abs=1e-9)


def test_surgery_iteration_records_exactly_one_state():
    """One snapshot and one convergence value per iteration, surgery or not."""
    strategy = CountingSurgery(cut=1, on={2, 4})
    engine = gg.RicciFlow(_assign(nx.complete_graph(6)), curvature="forman_node_weighted")
    result = engine.run(iterations=5, step=0.05, surgery=strategy, early_stop=False)

    assert strategy.applied == 2
    assert result.iterations_completed == 5
    assert len(result.snapshots) == 6
    assert len(result.convergence) == 5


# ── every output describes the same instant ──────────────────────────────────


def test_snapshot_convergence_result_and_gexf_agree(tmp_path):
    """One committed state, four views of it."""
    strategy = CountingSurgery(cut=1, on={2, 3})
    sim = gg.RicciFlowSimulator(
        _assign(nx.complete_graph(6)), curvature=gg.ollivier(alpha=0.5)
    )
    sim.run(iterations=4, step=0.05, surgery=strategy,
            early_stop=False, save_dir=str(tmp_path))

    for i in range(1, len(sim.snapshots)):
        assert _spread(sim.snapshots[i]) == pytest.approx(sim.convergence[i - 1])

    assert sim.result_graph.number_of_edges() == sim.snapshots[-1].number_of_edges()
    assert set(map(str, sim.result_graph.nodes())) == set(map(str, sim.snapshots[-1].nodes()))

    for i, snap in enumerate(sim.snapshots):
        written = nx.read_gexf(tmp_path / f"{i}.gexf")
        assert {frozenset(map(str, e)) for e in written.edges()} == \
               {frozenset(map(str, e)) for e in snap.edges()}
        for u, v in snap.edges():
            assert written[str(u)][str(v)]["ricciCurvature"] == pytest.approx(
                snap[u][v]["ricciCurvature"], abs=1e-9
            )
            assert written[str(u)][str(v)]["weight"] == pytest.approx(
                snap[u][v]["weight"], abs=1e-9
            )


def test_live_graph_equals_the_last_committed_snapshot():
    strategy = CountingSurgery(cut=1, on={2})
    engine = gg.RicciFlow(_assign(nx.complete_graph(5)), curvature="forman_node_weighted")
    result = engine.run(iterations=3, step=0.05, surgery=strategy, early_stop=False)

    assert set(engine.G.edges()) == set(result.snapshots[-1].edges())
    for u, v in engine.G.edges():
        assert engine.G[u][v]["ricciCurvature"] == pytest.approx(
            result.snapshots[-1][u][v]["ricciCurvature"]
        )


# ── failure is atomic ────────────────────────────────────────────────────────


def test_failed_post_surgery_curvature_commits_nothing(tmp_path):
    """An incomplete iteration leaves no output behind."""
    calls = {"n": 0}

    def fails_after_surgery(G, *, weight="weight", distance="distance", proc=1, **kw):
        calls["n"] += 1
        if G.number_of_edges() == 9:          # only the post-surgery topology
            raise RuntimeError("curvature unavailable on this topology")
        return write_curvature(G, {e: float(i) for i, e in enumerate(G.edges())})

    strategy = CountingSurgery(cut=1, on={1})
    engine = gg.RicciFlow(_assign(nx.complete_graph(5)), curvature=fails_after_surgery)
    with pytest.raises(RuntimeError, match="curvature unavailable"):
        engine.run(iterations=3, step=0.05, surgery=strategy,
                   early_stop=False, save_dir=str(tmp_path))

    written = sorted(int(p.stem) for p in tmp_path.glob("*.gexf") if p.stem.isdigit())
    assert written == [0], "the failed iteration must not have written a GEXF"


# ── termination reasons are typed ────────────────────────────────────────────


def test_termination_reasons_are_exactly_the_documented_seven():
    assert gg.flow.TERMINATION_REASONS == (
        "converged", "exhausted", "degenerate", "diverged", "undefined", "numerical",
        "iterations",
    )


def test_every_termination_reason_is_documented_on_the_simulator():
    doc = gg.RicciFlowSimulator.termination_reason.__doc__
    for reason in gg.flow.TERMINATION_REASONS:
        assert f"``{reason}``" in doc


def test_exhausted_is_not_reported_as_converged():
    engine = gg.RicciFlow(_assign(nx.path_graph(4)), curvature="forman_node_weighted")
    result = engine.run(
        iterations=5, step=0.2, early_stop=False,
        surgery={"name": "surgery_n", "interval": 1, "portion": 3},
    )
    assert result.termination_reason == "exhausted"
    assert result.termination_reason != "converged"


def test_result_unpacks_as_the_historical_pair():
    engine = gg.RicciFlow(_assign(nx.cycle_graph(5)), curvature="forman_node_weighted")
    result = engine.run(iterations=2, step=0.05, early_stop=False)
    snapshots, convergence = result
    assert snapshots is result.snapshots
    assert convergence is result.convergence


# ── a solver failure keeps the committed states ─────────────────────────────


def _fails_on_call(n_ok, *, when=None):
    """A curvature that raises CurvatureNumericalError after ``n_ok`` evaluations."""
    calls = {"n": 0}

    def curvature(G, *, weight="weight", distance="distance", proc=1, **kw):
        calls["n"] += 1
        if calls["n"] > n_ok and (when is None or when(G)):
            raise gg.CurvatureNumericalError("solver returned status 'infeasible'")
        return write_curvature(G, {e: float(i) for i, e in enumerate(G.edges())})

    return curvature


def test_solver_failure_on_an_update_stops_as_numerical(tmp_path):
    engine = gg.RicciFlow(_assign(nx.complete_graph(5)), curvature=_fails_on_call(3))
    result = engine.run(iterations=6, step=0.01, early_stop=False, save_dir=str(tmp_path))

    assert result.termination_reason == "numerical"
    # initial state + updates 1 and 2; update 3's candidate is not committed
    assert result.iterations_completed == 2
    assert len(result.snapshots) == 3 and len(result.convergence) == 2
    assert "update 3" in result.diagnosis and "infeasible" in result.diagnosis
    assert set(engine.G.edges()) == set(result.snapshots[-1].edges())
    written = sorted(int(p.stem) for p in tmp_path.glob("*.gexf") if p.stem.isdigit())
    assert written == [0, 1, 2]


def test_solver_failure_after_surgery_stops_as_numerical():
    strategy = CountingSurgery(cut=1, on={2})
    engine = gg.RicciFlow(
        _assign(nx.complete_graph(5)),
        curvature=_fails_on_call(0, when=lambda G: G.number_of_edges() == 9),
    )
    result = engine.run(iterations=4, step=0.01, early_stop=False, surgery=strategy)

    assert result.termination_reason == "numerical"
    assert strategy.applied == 1
    assert result.iterations_completed == 1
    assert "post-surgery graph of update 2" in result.diagnosis


def test_solver_failure_on_the_initial_state_still_raises():
    """Nothing is committed yet, so there is no partial result to return."""
    engine = gg.RicciFlow(_assign(nx.complete_graph(4)), curvature=_fails_on_call(0))
    with pytest.raises(gg.CurvatureNumericalError):
        engine.run(iterations=3)


def test_collapsing_metric_is_not_reported_as_converged():
    """An unnormalized distance flow shrinks the metric geometrically. The old
    absolute cut-off (kappa = 0 once d < 1e-7) made every curvature 0 and
    reported that collapse as ``converged``; the run must now either keep going
    or stop with a reason that says what happened."""
    B = nx.Graph([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)])
    result = gg.RicciFlow(
        _assign(B), curvature="lin_lu_yau", flow_equation="unnormalized",
        evolve="distance",
    ).run(iterations=250, step=0.5)

    assert result.termination_reason in ("numerical", "diverged", "iterations")
    assert len(result.snapshots) == result.iterations_completed + 1
    last = nx.get_edge_attributes(result.snapshots[-1], "ricciCurvature")
    assert any(abs(k) > 1e-3 for k in last.values())
    if result.termination_reason == "numerical":
        assert "ratio" in result.diagnosis
