"""Flow layer: trajectory parity vs ricciflow_sim + engine/equations/surgery."""

import json
import math
from pathlib import Path

import networkx as nx
import pytest

import graph_geometry as gg
from graph_geometry.flow import surgery as surgery_mod
from graph_geometry.flow.equations import resolve_flow
from graph_geometry.flow.surgery import (
    IntervalSurgery,
    NoSurgery,
    resolve_surgery,
    surgery_n,
)

FLOW_GOLDENS = json.loads(
    (Path(__file__).parent / "reference" / "flow_goldens.json").read_text()
)
ATOL = 1e-6
# Flow goldens are LLY-driven; see LLY_GOLDEN_ATOL in test_curvature_ot.py.
LLY_GOLDEN_ATOL = 1e-4


def _assign(G):
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    return G


def _build(case):
    spec = FLOW_GOLDENS[case]
    G = nx.DiGraph() if spec["directed"] else nx.Graph()
    G.add_edges_from(tuple(e) for e in spec["graph"])
    return _assign(G), spec


# ── trajectory parity vs ricciflow_sim ───────────────────────────────────────


@pytest.mark.parametrize("case", sorted(FLOW_GOLDENS))
def test_flow_trajectory_matches_reference(case):
    G, spec = _build(case)
    _, flow_name = case.split("__")
    snaps, conv = gg.RicciFlow(G, curvature="lin_lu_yau", flow_equation=flow_name).run(
        iterations=spec["iterations"], step=spec["step"], delta=1e-6, early_stop=True
    )
    # convergence list
    assert len(conv) == len(spec["convergence"])
    for got, gold in zip(conv, spec["convergence"]):
        assert got == pytest.approx(gold, abs=LLY_GOLDEN_ATOL)
    # final weights + RC
    final = snaps[-1]
    for u, v, gold in spec["final_weight"]:
        assert final[u][v]["weight"] == pytest.approx(gold, abs=LLY_GOLDEN_ATOL)
    for u, v, gold in spec["final_rc"]:
        assert final[u][v]["ricciCurvature"] == pytest.approx(gold, abs=LLY_GOLDEN_ATOL)


# ── engine behaviour ─────────────────────────────────────────────────────────


def test_symmetric_triangle_converges_in_one_iteration():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))
    _, conv = gg.RicciFlow(G).run(iterations=50, step=0.05, delta=1e-5)
    assert len(conv) == 1
    assert conv[0] == pytest.approx(0.0, abs=1e-9)


def test_original_graph_not_mutated():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    gg.RicciFlow(G).run(iterations=5, step=0.05)
    assert "ricciCurvature" not in G[0][1]  # engine copied its input


def test_snapshots_length_is_iters_plus_one():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    snaps, conv = gg.RicciFlow(G).run(iterations=10, step=0.05, delta=1e-9, early_stop=True)
    assert len(snaps) == len(conv) + 1  # initial snapshot + one per iteration


def test_early_stop_false_runs_full_iterations():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0)]))  # would converge in 1
    _, conv = gg.RicciFlow(G).run(iterations=8, step=0.05, early_stop=False)
    assert len(conv) == 8


def test_initial_snapshot_has_original_rc():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    snaps, _ = gg.RicciFlow(G).run(iterations=3, step=0.05)
    assert all("original_RC" in snaps[0][u][v] for u, v in snaps[0].edges())


# ── flow equations ───────────────────────────────────────────────────────────


def test_resolve_flow_names_and_callable():
    assert resolve_flow("normalized")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.1 * 0.5 * 2.0)
    assert resolve_flow("unnormalized")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.1 * 1.0 * 2.0)
    assert resolve_flow("additive")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.1 * 0.5)
    f = resolve_flow(lambda K, w, s, Ka: 42.0)
    assert f(0, 0, 0, 0) == 42.0


def test_resolve_flow_unknown_raises():
    with pytest.raises(ValueError):
        resolve_flow("bogus")  # unknown bare name -> rejected


# ── expression / LaTeX flows ─────────────────────────────────────────────────


def test_expression_flow_matches_normalized_preset():
    from graph_geometry.flow import expression_flow

    ef = expression_flow("-eta*(kappa - kbar)*w")
    ref = resolve_flow("normalized")
    for args in [(1.0, 2.0, 0.1, 0.5), (-0.3, 1.5, 0.05, 0.2)]:
        assert ef(*args) == pytest.approx(ref(*args))


def test_expression_flow_latex_forms():
    from graph_geometry.flow import ExpressionFlow

    # implicit multiplication, \frac, \bar\kappa, \eta all normalize + evaluate
    assert ExpressionFlow(r"-\eta(\kappa-\bar\kappa)w")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.1)
    assert ExpressionFlow(r"-\eta\kappa w")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.2)
    assert ExpressionFlow(r"-\frac{\eta \kappa w}{2}")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.1)


def test_expression_flow_conditional_and_funcs():
    from graph_geometry.flow import ExpressionFlow

    cond = ExpressionFlow("w*(1 - step) if kappa > kbar else w")
    assert cond(1.0, 2.0, 0.1, 0.5) == pytest.approx(1.8)   # kappa>kbar branch
    assert cond(0.0, 2.0, 0.1, 0.5) == pytest.approx(2.0)   # else branch
    assert ExpressionFlow("max(-0.1, -step*kappa*w)")(1.0, 2.0, 0.1, 0.5) == pytest.approx(-0.1)


@pytest.mark.parametrize("expr, expected", [
    ("-1e-3*kappa*q", -0.01),
    ("-2.5E-2*kappa*q", -0.25),
    ("-1e-3 kappa q", -0.01),
    ("-1e-3(kappa-kbar)q", -0.01),
    (r"-\eta log10(q)", -0.01),
    ("-eta*log10(q)", -0.01),
    ("-eta*log10 (q)", -0.01),
    ("-eta*log10(q) if kappa>kbar else 0", -0.01),
])
def test_expression_flow_preserves_number_and_function_tokens(expr, expected):
    assert gg.expression_flow(expr)(1.0, 10.0, 0.01, 0.0) == pytest.approx(expected)


def test_expression_flow_rejects_unsafe():
    from graph_geometry.flow import expression_flow

    for bad in ["__import__('os')", "w.__class__", "(1).__class__", "open('x')", "unknown_var"]:
        with pytest.raises(ValueError):
            expression_flow(bad)


def test_expression_flow_rejects_nonnumeric_and_huge_constants():
    from graph_geometry.flow import expression_flow

    with pytest.raises(ValueError):
        expression_flow("'abc'")            # string constant
    with pytest.raises(ValueError):
        expression_flow("w * 1e12")         # oversized literal


def test_expression_flow_pow_is_float_safe():
    from graph_geometry.flow import expression_flow

    # normal exponentiation works (evaluated as float pow)
    assert expression_flow("-step*w**2")(1.0, 3.0, 0.1, 0.0) == pytest.approx(-0.9)
    # bignum blowup raises immediately instead of hanging the process
    with pytest.raises(OverflowError):
        expression_flow("9**9**9")(1.0, 1.0, 0.1, 0.0)


def test_expression_flow_drives_the_engine():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    # custom expression == normalized preset -> identical trajectory
    s1, c1 = gg.RicciFlow(G.copy(), flow_equation="-eta*(kappa - kbar)*w").run(
        iterations=15, step=0.05, delta=1e-6)
    s2, c2 = gg.RicciFlow(G.copy(), flow_equation="normalized").run(
        iterations=15, step=0.05, delta=1e-6)
    assert c1 == pytest.approx(c2, abs=1e-12)


def test_one_flow_written_three_ways_gives_one_trajectory():
    """Preset name, Python callable and pasted LaTeX are interchangeable.

    The three spellings a user can reach for must be the same object to the
    engine; this is the claim §3.5 of the paper makes, pinned.
    """
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))

    def as_callable(K, w, step, K_avg):
        return -step * (K - K_avg) * w

    spellings = {
        "preset name": "normalized",
        "python callable": as_callable,
        "pasted LaTeX": r"-\eta\,(\kappa-\bar\kappa)\,w",
    }
    trajectories = {}
    for label, spec in spellings.items():
        snaps, conv = gg.RicciFlow(G.copy(), flow_equation=spec).run(
            iterations=20, step=0.05, delta=1e-9, early_stop=False)
        trajectories[label] = (conv, snaps[-1])

    reference, ref_graph = trajectories["preset name"]
    for label, (conv, final) in trajectories.items():
        assert conv == pytest.approx(reference, abs=1e-12), label
        for u, v in ref_graph.edges():
            assert final[u][v]["weight"] == pytest.approx(
                ref_graph[u][v]["weight"], abs=1e-12), (label, u, v)


# ── evolve weight vs distance ────────────────────────────────────────────────


def test_evolve_distance_keeps_weight_fixed():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    snaps, conv = gg.RicciFlow(G, evolve="distance").run(iterations=10, step=0.05, early_stop=False)
    Gf = snaps[-1]
    assert all(Gf[u][v]["weight"] == 1.0 for u, v in Gf.edges())          # weight fixed
    assert any(abs(Gf[u][v]["distance"] - 1.0) > 1e-9 for u, v in Gf.edges())  # distance moved


def test_evolve_weight_is_default_and_unchanged():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    s_def, c_def = gg.RicciFlow(G.copy()).run(iterations=10, step=0.05)
    s_w, c_w = gg.RicciFlow(G.copy(), evolve="weight").run(iterations=10, step=0.05)
    assert c_def == pytest.approx(c_w, abs=1e-12)
    # with evolve=weight, distance stays fixed
    assert all(s_w[-1][u][v]["distance"] == 1.0 for u, v in s_w[-1].edges())


def test_evolve_invalid_raises():
    with pytest.raises(ValueError):
        gg.RicciFlow(_assign(nx.DiGraph([(0, 1), (1, 0)])), evolve="both")


def test_simulator_passes_evolve():
    sim = gg.RicciFlowSimulator.from_edges(
        [(0, 1), (1, 2), (2, 0), (0, 2)], evolve="distance"
    ).run(iterations=8, step=0.05, early_stop=False)
    Gf = sim.result_graph
    assert all(Gf[u][v]["weight"] == 1.0 for u, v in Gf.edges())


def test_coupled_weight_as_distance():
    # point the metric at the weight attr: one quantity drives kernel AND metric
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    coupled, cc = gg.RicciFlow(
        G.copy(), weight="weight", distance="weight", evolve="weight"
    ).run(iterations=12, step=0.05, verbose=False)
    indep, ci = gg.RicciFlow(
        G.copy(), weight="weight", distance="distance", evolve="weight"
    ).run(iterations=12, step=0.05, verbose=False)
    # coupling the metric to the weight changes the trajectory
    assert any(abs(a - b) > 1e-6 for a, b in zip(cc, ci))
    # the single quantity actually evolved
    Gf = coupled[-1]
    assert any(abs(Gf[u][v]["weight"] - 1.0) > 1e-6 for u, v in Gf.edges())


# ── surgery ──────────────────────────────────────────────────────────────────


def test_surgery_n_removes_top_weight_edges():
    G = nx.DiGraph()
    G.add_edge(0, 1, weight=0.1)
    G.add_edge(1, 2, weight=0.9)
    G.add_edge(2, 0, weight=0.5)
    H = surgery_n(G, "weight", 1)
    assert H.number_of_edges() == 2
    assert not H.has_edge(1, 2)  # highest weight removed


def test_surgery_fraction_removes_proportion():
    G = nx.DiGraph()
    for i in range(10):
        G.add_edge(i, (i + 1) % 10, weight=float(i))
    H = surgery_mod.surgery(G, "weight", 0.2)
    assert H.number_of_edges() == 8  # removed 20%


def test_interval_surgery_should_apply():
    s = IntervalSurgery(surgery_n, interval=5, portion=1)
    assert not s.should_apply(0)
    assert s.should_apply(5)
    assert not s.should_apply(7)
    assert s.should_apply(10)


def test_interval_surgery_runs_after_the_interval_th_update():
    """interval=5 operates on the result of update 5: committed state 5 is post-surgery."""
    G = _assign(nx.cycle_graph(8))
    result = gg.RicciFlow(G, curvature="forman_node_weighted").run(
        iterations=11, step=0.01, early_stop=False,
        surgery={"name": "surgery_n", "interval": 5, "portion": 1},
    )
    assert [s.number_of_edges() for s in result.snapshots] == [8] * 5 + [7] * 5 + [6] * 2


def test_interval_one_operates_after_the_first_update():
    G = _assign(nx.cycle_graph(6))
    result = gg.RicciFlow(G, curvature="forman_node_weighted").run(
        iterations=2, step=0.01, early_stop=False,
        surgery={"name": "surgery_n", "interval": 1, "portion": 1},
    )
    assert [s.number_of_edges() for s in result.snapshots] == [6, 5, 4]


@pytest.mark.parametrize("n, p, expected", [
    (5, 0.8, 4),      # int(5 * (1 - 0.8)) == 0 used to cut all five
    (10, 0.9, 9),
    (10, 0.3, 3),
    (10, 0.02, 1),    # ceil: any positive proportion cuts at least one edge
    (100, 0.07, 7),
    (7, 0.0, 0),
    (7, 1.0, 7),
])
def test_surgery_fraction_cuts_ceil_of_the_proportion(n, p, expected):
    G = nx.path_graph(n + 1)
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = float(i)
    H = surgery_mod.surgery(G, "weight", p)
    assert G.number_of_edges() - H.number_of_edges() == expected
    # the heaviest edges are the ones removed
    kept = sorted(d["weight"] for *_, d in H.edges(data=True))
    assert kept == [float(i) for i in range(n - expected)]


def test_resolve_surgery_variants():
    assert isinstance(resolve_surgery(None), NoSurgery)
    assert isinstance(resolve_surgery({"name": "no_surgery"}), NoSurgery)
    s = resolve_surgery({"name": "surgery_n", "interval": 3, "portion": 2})
    assert isinstance(s, IntervalSurgery)
    assert s.interval == 3 and s.portion == 2
    obj = NoSurgery()
    assert resolve_surgery(obj) is obj


def test_resolve_surgery_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_surgery({"name": "nope"})


def test_flow_with_surgery_runs_and_reduces_edges():
    G = _assign(nx.complete_graph(6))
    snaps, _ = gg.RicciFlow(G).run(
        iterations=12, step=0.05, early_stop=False,
        surgery={"name": "surgery_n", "portion": 1, "interval": 4},
    )
    # surgery at iters 4, 8 removes edges -> final has fewer than the initial 15
    assert snaps[-1].number_of_edges() < 15


def test_forman_flow_preserves_all_disconnected_components():
    G = _assign(nx.disjoint_union(nx.path_graph(3), nx.path_graph(4)))
    result = gg.RicciFlow(G, curvature="forman_node_weighted").run(
        iterations=1, step=0.05, early_stop=False
    )
    assert result.snapshots[0].number_of_edges() == G.number_of_edges()
    assert set(result.snapshots[0].nodes()) == set(G.nodes())


def test_eidi_jost_flow_keeps_non_strongly_connected_graph():
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 3), (1, 4)]))
    assert not nx.is_strongly_connected(G)
    result = gg.RicciFlow(G, curvature="eidi_jost").run(
        iterations=1, step=0.01, early_stop=False
    )
    assert set(result.snapshots[0].edges()) == set(G.edges())


def test_surgery_that_breaks_strong_connectivity_stops_as_undefined():
    """A method that needs reachability must
    report the topology surgery produced, not fall back to a component.

    This is exactly what ``configs/mygraph_3cycle4edge.yaml`` hits once its
    scheduled surgery cuts a cycle edge.
    """

    class CutOneEdge:
        def __init__(self, edge):
            self.edge = edge

        def should_apply(self, iteration):
            return iteration == 1

        def apply(self, G, attribute):
            H = G.copy()
            H.remove_edge(*self.edge)
            return H

    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    engine = gg.RicciFlow(G, curvature="lin_lu_yau")
    result = engine.run(iterations=3, step=0.05, early_stop=False,
                        surgery=CutOneEdge((1, 2)))
    # the run stops with a topology diagnosis and keeps its valid states,
    # as for ``diverged``, instead of losing them to the exception
    assert result.termination_reason == "undefined"
    assert "strongly connected" in result.diagnosis
    assert "after update 1" in result.diagnosis
    # the failed iteration is not committed: the live graph is the last good one
    assert result.iterations_completed == 0
    assert len(result.snapshots) == 1
    assert engine.G.number_of_edges() == 4


def test_flow_rejects_input_without_non_self_loop_edges():
    """A genuinely empty graph keeps its own empty-edge-set error."""
    with pytest.raises(gg.CurvatureDomainError, match="at least one non-self-loop edge"):
        gg.RicciFlow(nx.Graph()).run()


# ── self-loops: refused, never silently removed ─────────────────────────────


def _loop_and_edge():
    G = nx.Graph()
    G.add_edge(0, 1, weight=1.0, distance=1.0)
    G.add_edge(1, 2, weight=1.0, distance=1.0)
    G.add_edge(1, 1, weight=1.0, distance=1.0)
    return G


@pytest.mark.parametrize("make", [_loop_and_edge, lambda: nx.Graph([(0, 0)])],
                         ids=["loop-and-edges", "loop-only"])
def test_flow_refuses_self_loops_instead_of_dropping_them(make):
    """The flow used to remove loops from its copy, discarding input silently.

    A graph that is only a self-loop gets the same self-loop error, not the
    empty-edge-set error it would have reached once the loop was stripped.
    """
    G = make()
    engine = gg.RicciFlow(G, curvature="forman_node_weighted")
    with pytest.raises(gg.CurvatureDomainError, match="self-loops"):
        engine.run(iterations=1)
    # refused before the internal copy was touched: the loop is still there and
    # no unit attributes were written onto the copy
    assert nx.number_of_selfloops(engine.G) == nx.number_of_selfloops(G)
    assert all(dict(d) == dict(G.edges[u, v]) for u, v, d in engine.G.edges(data=True))


_SIMPLE_GRAPH_VIOLATIONS = [
    ("self-loops", _loop_and_edge),
    ("self-loops", lambda: nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 2)])),
    ("multigraphs", lambda: nx.MultiGraph([(0, 1), (0, 1), (1, 2)])),
]


@pytest.mark.parametrize("message, make", _SIMPLE_GRAPH_VIOLATIONS,
                         ids=["undirected-loop", "directed-loop", "multigraph"])
def test_standalone_evaluation_and_flow_share_one_simple_graph_policy(message, make):
    method = "forman_directed" if make().is_directed() else "forman_node_weighted"
    with pytest.raises(gg.CurvatureDomainError, match=message) as standalone:
        gg.compute_curvature(make(), method)
    with pytest.raises(gg.CurvatureDomainError, match=message) as flow:
        gg.RicciFlow(make(), curvature=method).run(iterations=1)
    # one validator: the messages differ only in who is asking
    assert _policy_text(standalone) == _policy_text(flow)


def _policy_text(excinfo):
    """The validator's message without its leading ``where``."""
    text = str(excinfo.value)
    for marker in (" requires a graph without self-loops", " does not support multigraphs"):
        if marker in text:
            return text[text.index(marker):]
    raise AssertionError(f"not a simple-graph policy message: {text}")


def test_flow_rejects_a_non_positive_distance_update_before_curvature():
    def constant(G, **_kwargs):
        return {(u, v): float(i) for i, (u, v) in enumerate(G.edges())}

    def negate_quantity(_kappa, quantity, _step, _mean):
        return -2.0 * quantity

    G = _assign(nx.path_graph(3))
    with pytest.raises(ValueError, match="Ricci flow update.*metric"):
        gg.RicciFlow(
            G,
            curvature=constant,
            flow_equation=negate_quantity,
            evolve="distance",
        ).run(iterations=1, early_stop=False)


def test_flow_survives_surgery_emptying_graph():
    # aggressive surgery removes all edges -> engine must stop, not divide by zero
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    snaps, conv = gg.RicciFlow(G, curvature="forman_directed").run(
        iterations=10, step=0.05, early_stop=False,
        surgery={"name": "surgery_n", "portion": 2, "interval": 1},
    )
    # completed without ZeroDivisionError; stopped early once weight ran out
    assert len(conv) < 10
    assert snaps[-1].number_of_edges() < 4


def test_surgery_uses_the_evolving_attribute():
    class RecordingSurgery:
        def __init__(self):
            self.attributes = []

        def should_apply(self, iteration):
            return iteration == 2

        def apply(self, G, attribute):
            self.attributes.append(attribute)
            return G

    strategy = RecordingSurgery()
    G = _assign(nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)]))
    gg.RicciFlow(G, curvature="forman_directed", evolve="distance").run(
        iterations=3, early_stop=False, surgery=strategy
    )
    assert strategy.attributes == ["distance"]


# ── committed-state regressions ──────────────────────────────────────────────
# See RicciFlow-Computation.md, "Committed-state P0: prerequisite landed".


def test_curvature_is_recomputed_after_surgery():
    """Stored curvature used to describe the pre-surgery topology (drift 0.19)."""
    engine = gg.RicciFlow(
        _assign(nx.barbell_graph(5, 0)),
        curvature=gg.ollivier(alpha=0.5),
        flow_equation="normalized",
    )
    engine.run(
        iterations=4,
        step=0.3,
        early_stop=False,
        surgery={"name": "surgery_n", "interval": 1, "portion": 1},
    )
    live = engine.G
    fresh = gg.curvature(live.copy(), method=gg.ollivier(alpha=0.5))
    for edge, value in fresh.items():
        stored = live[edge[0]][edge[1]]["ricciCurvature"]
        assert stored == pytest.approx(value, abs=ATOL)


def test_surgery_that_empties_the_graph_commits_it_as_exhausted():
    """Emptying the graph is not convergence, and contributes no spread value.

    Spread is undefined on an empty edge set, so the
    exhausted iteration commits a snapshot but no convergence value -- the one
    case where ``len(convergence) != len(snapshots) - 1``.
    """
    engine = gg.RicciFlow(_assign(nx.path_graph(4)), curvature=gg.ollivier(alpha=0.5))
    result = engine.run(
        iterations=5,
        step=0.2,
        early_stop=False,
        surgery={"name": "surgery_n", "interval": 1, "portion": 3},
    )
    assert engine.G.number_of_edges() == 0
    assert result.termination_reason == "exhausted"
    assert result.snapshots[-1].number_of_edges() == 0
    assert len(result.convergence) == len(result.snapshots) - 2
    assert engine.G.number_of_edges() == result.snapshots[-1].number_of_edges()


def test_final_gexf_is_named_for_the_iteration_reached(tmp_path):
    """An early-stopped 40-iteration run used to write a misleading 40.gexf."""
    engine = gg.RicciFlow(_assign(nx.complete_graph(5)), curvature=gg.ollivier(alpha=0.5))
    _, convergence = engine.run(
        iterations=40, step=0.05, save_dir=str(tmp_path), early_stop=True
    )
    written = sorted(
        int(p.stem) for p in tmp_path.glob("*.gexf") if p.stem.isdigit()
    )
    assert max(written) == len(convergence)


# ── a diverging flow must explain itself ────────────────────────────────────
# Reported from the Streamlit UI: SBM 2x8 with forman_sreejith at the default
# step failed with only "edge (0, 7) has weight=-0.0143...", which says nothing
# about what to change. The engine knows the iteration, the largest step that
# would have survived, and the spread history; the message now carries them.


def _sbm():
    G = nx.stochastic_block_model([8, 8], [[0.9, 0.1], [0.1, 0.9]], seed=42)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", overwrite=True)
    return G


def test_divergence_terminates_cleanly_and_keeps_every_valid_state():
    """Divergence is a typed termination, like `exhausted` -- not an exception.

    The committed states are what *show* the divergence, so discarding them by
    raising would throw away the result. Nothing invalid is committed: the run
    stops before the offending update.
    """
    engine = gg.RicciFlow(
        _sbm(), curvature="forman_sreejith", flow_equation="normalized"
    )
    result = engine.run(iterations=50, step=0.05)
    assert result.termination_reason == "diverged"
    assert result.iterations_completed == 8
    assert len(result.snapshots) == 9
    # every committed state is valid
    for snapshot in result.snapshots:
        assert all(d["weight"] > 0 for _, _, d in snapshot.edges(data=True))
    # and the spread is rising, which is the finding
    assert result.convergence[-1] > result.convergence[0]


def test_divergence_diagnosis_reports_iteration_step_and_spread_trend():
    result = gg.RicciFlow(
        _sbm(), curvature="forman_sreejith", flow_equation="normalized"
    ).run(iterations=50, step=0.05)
    message = result.diagnosis
    assert "diverged at iteration 8" in message
    assert "largest step that stays positive" in message
    assert "curvature spread over the last" in message
    # the spread grows here, so a smaller step only postpones the failure
    assert "postpone the failure" in message


def test_a_completed_run_carries_no_diagnosis():
    result = gg.RicciFlow(_sbm(), curvature="ollivier").run(
        iterations=5, step=0.05, early_stop=False
    )
    assert result.termination_reason == "iterations"
    assert result.diagnosis == ""


def test_a_step_that_is_merely_too_large_says_to_reduce_it():
    """Not every failure is divergence; the advice must distinguish them."""
    G = nx.barbell_graph(5, 0)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", overwrite=True)
    engine = gg.RicciFlow(G, curvature="ollivier", flow_equation="normalized")
    message = engine.run(iterations=20, step=6.0, early_stop=False).diagnosis
    assert "Re-run with step below" in message
    assert "postpone the failure" not in message


def test_the_reported_safe_step_actually_survives_that_iteration():
    """The suggested bound must be usable, not decorative."""
    G = nx.barbell_graph(5, 0)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", overwrite=True)
    diagnosis = gg.RicciFlow(G, curvature="ollivier", flow_equation="normalized").run(
        iterations=20, step=6.0, early_stop=False
    ).diagnosis
    safe = float(diagnosis.split("step below")[1].split(".\n")[0].strip(" ."))
    H = nx.barbell_graph(5, 0)
    gg.apply_config(H, edge_weight="fixed", edge_distance="fixed", overwrite=True)
    engine = gg.RicciFlow(H, curvature="ollivier", flow_equation="normalized")
    engine.run(iterations=1, step=safe * 0.99, early_stop=False)
    assert all(d["weight"] > 0 for _, _, d in engine.G.edges(data=True))


def test_ollivier_and_lly_still_converge_on_the_same_graph():
    """Control: the engine is fine; the divergence belongs to the pairing."""
    for method in ("ollivier", "lin_lu_yau"):
        engine = gg.RicciFlow(_sbm(), curvature=method, flow_equation="normalized")
        _, convergence = engine.run(iterations=30, step=0.05, early_stop=False)
        assert convergence[-1] < convergence[0]


def test_an_invalid_first_update_still_raises_rather_than_returning_nothing():
    """The two failure modes are distinguished by whether anything was learned.

    A flow equation that inverts the quantity fails at iteration 0, so there is
    no partial result and nothing the states could show: that is an unusable
    configuration and raises. A run that completes iterations and then diverges
    returns them, because they are the finding. See the pair of tests above.
    """
    def constant(G, **_kwargs):
        return {(u, v): float(i) for i, (u, v) in enumerate(G.edges())}

    def negate_quantity(_kappa, quantity, _step, _mean):
        return -2.0 * quantity

    G = _assign(nx.path_graph(3))
    with pytest.raises(ValueError, match="diverged at iteration 0"):
        gg.RicciFlow(
            G, curvature=constant, flow_equation=negate_quantity, evolve="distance"
        ).run(iterations=5, early_stop=False)


@pytest.mark.parametrize("equation, completed, error", [
    ("eta*exp(q)", 3, "OverflowError"),
    (lambda k, q, eta, mean: eta * math.exp(q), 3, "OverflowError"),
    ("1/(2-q)", 1, "ZeroDivisionError"),
    ("sqrt(2-q)+1", 1, "ValueError"),
])
def test_formula_errors_return_only_completed_states(tmp_path, equation, completed, error):
    G = _assign(nx.path_graph(3))
    original = G.copy()
    engine = gg.RicciFlow(G, curvature="ollivier", flow_equation=equation)
    reported = []
    result = engine.run(
        iterations=10, step=1.0, early_stop=False, save_dir=str(tmp_path),
        progress_callback=lambda i, total, spread: reported.append(i + 1),
    )

    assert result.termination_reason == engine.termination_reason == "diverged"
    assert result.iterations_completed == completed
    assert len(result.snapshots) == completed + 1
    assert len(result.convergence) == completed
    assert reported == list(range(1, completed + 1))
    assert error in result.diagnosis
    assert nx.utils.graphs_equal(G, original)
    assert nx.utils.graphs_equal(engine.G, result.snapshots[-1])
    assert not (tmp_path / f"{completed + 1}.gexf").exists()
    for snapshot in result.snapshots:
        assert all(math.isfinite(d["weight"]) and d["weight"] > 0
                   for _, _, d in snapshot.edges(data=True))
        fresh = gg.compute_curvature(snapshot, "ollivier").values
        for edge, value in fresh.items():
            assert snapshot.edges[edge]["ricciCurvature"] == pytest.approx(value)


def test_formula_error_discards_an_entire_partially_updated_candidate():
    G = _assign(nx.path_graph(3))
    G[0][1]["weight"] = 0.5
    engine = gg.RicciFlow(G, curvature="ollivier", flow_equation="1/(2-q)")
    result = engine.run(iterations=5, early_stop=False)
    assert result.iterations_completed == 1
    assert result.termination_reason == "diverged"
    # On update 2, edge (0, 1) updates successfully before edge (1, 2) fails.
    assert engine.G[0][1]["weight"] == pytest.approx(0.5 + 1 / 1.5)
    assert engine.G[1][2]["weight"] == 2.0


def test_formula_error_on_the_first_update_still_raises_a_typed_error():
    from graph_geometry.flow import FlowDivergenceError

    engine = gg.RicciFlow(nx.path_graph(3), curvature="ollivier",
                          flow_equation="exp(1000)*q")
    with pytest.raises(FlowDivergenceError, match="OverflowError") as exc:
        engine.run(iterations=5, early_stop=False)
    assert isinstance(exc.value.__cause__, OverflowError)


def test_formula_safe_step_diagnosis_handles_overflow_during_bisection():
    engine = gg.RicciFlow(_assign(nx.path_graph(3)), curvature="ollivier",
                          flow_equation="exp(eta*q)")
    result = engine.run(iterations=10, step=2.0, early_stop=False)
    assert result.termination_reason == "diverged"
    assert result.iterations_completed == 2
    safe = engine._largest_safe_step(0.0, 2.0, "weight")
    assert 0 < safe < 2.0
    resumed = gg.RicciFlow(result.snapshots[-1], curvature="ollivier",
                           flow_equation="exp(eta*q)")
    assert resumed.run(iterations=1, step=safe * 0.99, early_stop=False).iterations_completed == 1


def test_flow_does_not_swallow_unrelated_plugin_errors():
    def buggy_equation(kappa, q, step, mean):
        raise TypeError("plugin implementation bug")

    with pytest.raises(TypeError, match="plugin implementation bug"):
        gg.RicciFlow(nx.path_graph(3), curvature="ollivier",
                     flow_equation=buggy_equation).run()

# ── surgery defaults and validation (audit fixes) ───────────────────────────


def _weighted_cycle(n=10):
    G = nx.cycle_graph(n)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0 + u
    return G


def test_surgery_n_without_a_portion_cuts_one_edge_not_all():
    """The shared default 0.02 used to make surgery_n slice [-0:] and cut everything."""
    s = resolve_surgery({"name": "surgery_n"})
    assert s.portion == 1
    assert s.apply(_weighted_cycle(), "weight").number_of_edges() == 9


def test_surgery_defaults_are_name_specific_and_shared():
    from graph_geometry.flow.surgery import DEFAULT_INTERVAL, DEFAULT_PORTION

    fraction = resolve_surgery({"name": "surgery"})
    count = resolve_surgery({"name": "surgery_n"})
    assert (fraction.portion, count.portion) == (DEFAULT_PORTION["surgery"], DEFAULT_PORTION["surgery_n"])
    assert fraction.interval == count.interval == DEFAULT_INTERVAL


@pytest.mark.parametrize("spec", [
    {"name": "surgery_n", "portion": 0.02},       # a fraction given to the count form
    {"name": "surgery_n", "portion": 1.5},
    {"name": "surgery", "portion": 3},             # a count given to the fraction form
    {"name": "surgery", "interval": 0},
    {"name": "surgery", "intervall": 3},           # a misspelled key
])
def test_invalid_surgery_specs_are_rejected(spec):
    with pytest.raises(ValueError):
        resolve_surgery(spec)


def test_surgery_n_rejects_a_fractional_count():
    with pytest.raises(ValueError, match="whole number"):
        surgery_n(_weighted_cycle(), "weight", 1.7)


def test_flow_errors_are_typed():
    from graph_geometry.flow import FlowDivergenceError

    with pytest.raises(gg.CurvatureDomainError, match="self-loops"):
        gg.RicciFlow(nx.Graph([(0, 0)]), curvature="forman_node_weighted").run(iterations=1)
    with pytest.raises(gg.CurvatureDomainError, match="non-self-loop edge"):
        gg.RicciFlow(nx.Graph(), curvature="forman_node_weighted").run(iterations=1)
    with pytest.raises(gg.CurvatureConfigurationError, match="evolve"):
        gg.RicciFlow(nx.path_graph(3), evolve="both")
    assert issubclass(FlowDivergenceError, ValueError)
    assert issubclass(FlowDivergenceError, gg.CurvatureNumericalError)


def test_nan_update_is_not_diagnosed_with_a_safe_step():
    """``NaN <= 0`` is False, so the bisection used to certify a NaN update as
    safe and advise 'largest step that stays positive: 0.1'."""
    calls = {"n": 0}

    def goes_nan(K, q, step, K_avg):
        calls["n"] += 1
        return float("nan") if calls["n"] > 5 else 0.0

    result = gg.RicciFlow(_assign(nx.cycle_graph(5)), curvature="ollivier",
                          flow_equation=goes_nan).run(
        iterations=3, step=0.1, early_stop=False
    )
    assert result.termination_reason == "diverged"
    assert "non-finite weight=nan" in result.diagnosis
    assert "largest step" not in result.diagnosis
    assert "smaller step will not help" in result.diagnosis


def test_safe_step_rejects_non_finite_updates():
    engine = gg.RicciFlow(_assign(nx.cycle_graph(4)), curvature="ollivier",
                          flow_equation=lambda K, q, s, Ka: float("inf") * s)
    engine.G = engine.G.copy()
    for u, v in engine.G.edges():
        engine.G[u][v]["ricciCurvature"] = 0.0
    assert engine._largest_safe_step(0.0, 0.1, "weight") == 0.0
