"""The comparison workbench: several definitions, identical input."""

import csv

import networkx as nx
import pytest

import graph_geometry as gg

OT_METHODS = ("lin_lu_yau", "ollivier")


def _undirected():
    G = nx.complete_graph(5)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    return G


def _digraph_with_sinks():
    """3-cycle fanning out to two sinks: OT curvature is undefined, Forman is not."""
    G = gg.load.from_edges(
        [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 4)], directed=True
    )
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    return G


# ── basic behaviour ─────────────────────────────────────────────────────────


def test_defaults_to_every_registered_method():
    cmp = gg.compare_curvature(_undirected())
    assert set(cmp.methods) == set(gg.CURVATURE_REGISTRY)


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown curvature method"):
        gg.compare_curvature(_undirected(), ["nope"])


def test_input_graph_is_not_mutated():
    G = _undirected()
    before = {(u, v): dict(d) for u, v, d in G.edges(data=True)}
    gg.compare_curvature(G, ["forman_node_weighted", "lin_lu_yau"])
    assert "ricciCurvature" not in next(iter(G.edges(data=True)))[2]
    assert {(u, v): dict(d) for u, v, d in G.edges(data=True)} == before


def test_methods_do_not_leak_attributes_into_each_other():
    """Each method gets its own copy, so the contract's writes cannot cross."""
    cmp = gg.compare_curvature(_undirected(), ["forman_node_weighted", "augmented_forman_node_weighted"])
    assert cmp.values["forman_node_weighted"] != cmp.values["augmented_forman_node_weighted"]


# ── failures are data, not exceptions ───────────────────────────────────────


def test_undefined_method_is_recorded_rather_than_raised():
    cmp = gg.compare_curvature(
        _digraph_with_sinks(),
        ["lin_lu_yau", "forman_directed", "eidi_jost"],
    )
    assert "lin_lu_yau" in cmp.errors
    assert "connected" in cmp.errors["lin_lu_yau"]
    assert len(cmp.values["eidi_jost"]) == 6
    assert len(cmp.values["forman_directed"]) == 6


def test_summary_reports_zero_coverage_for_a_failed_method():
    cmp = gg.compare_curvature(_digraph_with_sinks(), ["lin_lu_yau", "eidi_jost"])
    rows = {r["method"]: r for r in cmp.summary()}
    assert rows["lin_lu_yau"]["computable"] == 0
    assert rows["lin_lu_yau"]["error"]
    assert rows["eidi_jost"]["computable"] == rows["eidi_jost"]["of"] == 6
    assert rows["eidi_jost"]["error"] == ""


def test_summary_coverage_is_all_or_none_under_the_exact_edge_contract():
    """A successful run covers every edge; a failed one covers none -- never a part."""
    G = _digraph_with_sinks()
    cmp = gg.compare_curvature(G, ["lin_lu_yau", "ollivier", "forman_directed", "eidi_jost"])
    for row in cmp.summary():
        assert row["of"] == G.number_of_edges()
        if row["failure_kind"]:
            assert row["computable"] == 0
        else:
            assert row["computable"] == row["of"]


def test_empty_graph_summary_reports_zero_coverage_without_crashing():
    cmp = gg.compare_curvature(nx.Graph(), ["forman_node_weighted", "ollivier"])
    assert not cmp.errors
    for row in cmp.summary():
        assert row["computable"] == row["of"] == 0
        assert row["min"] is row["median"] is row["max"] is None
        assert row["mean"] is row["pct_negative"] is None


def test_agreement_skips_failed_methods():
    cmp = gg.compare_curvature(_digraph_with_sinks(), ["lin_lu_yau", "forman_directed", "eidi_jost"])
    pairs = {(r["method_a"], r["method_b"]) for r in cmp.agreement()}
    assert pairs == {("forman_directed", "eidi_jost")}


# ── statistics ──────────────────────────────────────────────────────────────


def test_a_method_agrees_perfectly_with_itself():
    G = nx.barbell_graph(4, 1)          # non-regular, so curvature actually varies
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    a = gg.curvature(G.copy(), method="augmented_forman_node_weighted")
    assert len(set(a.values())) > 1
    cmp = gg.CurvatureComparison.from_values(methods=["x", "y"], values={"x": a, "y": dict(a)},
                                 n_edges=G.number_of_edges())
    row = cmp.agreement()[0]
    assert row["spearman"] == pytest.approx(1.0)
    assert row["pearson"] == pytest.approx(1.0)
    assert row["sign_agreement"] == pytest.approx(1.0)


def test_perfect_anticorrelation_is_detected():
    a = {(0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0}
    b = {e: -v for e, v in a.items()}
    cmp = gg.CurvatureComparison.from_values(methods=["a", "b"], values={"a": a, "b": b}, n_edges=3)
    row = cmp.agreement()[0]
    assert row["spearman"] == pytest.approx(-1.0)
    assert row["sign_agreement"] == pytest.approx(0.0)


def test_constant_method_gives_undefined_correlation_not_a_crash():
    a = {(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0}
    b = {(0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0}
    cmp = gg.CurvatureComparison.from_values(methods=["a", "b"], values={"a": a, "b": b}, n_edges=3)
    row = cmp.agreement()[0]
    assert row["spearman"] != row["spearman"]        # nan
    assert row["n"] == 3


def test_hand_rolled_stats_match_scipy():
    """The module implements rankdata/spearman itself to avoid a scipy import.

    scipy is only present here transitively (via cvxpy), so this pins the
    hand-rolled versions against the reference when it happens to be available.
    """
    stats = pytest.importorskip("scipy.stats")
    import numpy as np

    from graph_geometry.compare import _corr, _rankdata

    rng = np.random.default_rng(0)
    for _ in range(50):
        n = int(rng.integers(3, 12))
        a = rng.integers(0, 4, size=n).astype(float)     # ties on purpose
        b = rng.normal(size=n)
        assert _rankdata(a) == pytest.approx(stats.rankdata(a))
        assert _corr(_rankdata(a), _rankdata(b)) == pytest.approx(
            stats.spearmanr(a, b).statistic
        )


def test_spearman_uses_ranks_not_values():
    """Monotone but non-linear: spearman 1, pearson < 1."""
    a = {(0, 1): 1.0, (1, 2): 2.0, (2, 3): 3.0, (3, 4): 4.0}
    b = {e: v ** 4 for e, v in a.items()}
    row = gg.CurvatureComparison.from_values(methods=["a", "b"], values={"a": a, "b": b},
                                 n_edges=4).agreement()[0]
    assert row["spearman"] == pytest.approx(1.0)
    assert row["pearson"] < 0.99


def test_spearman_ties_survive_roundoff():
    """Exactly proportional values with few levels: roundoff must not break ties.

    On two triangles joined by a bridge, Ollivier(alpha=1/2) is exactly half of
    Lin-Lu-Yau, with three distinct values. Ranking the raw floats let 1e-16
    residue order the tied edges, which reported Spearman 0.92.
    """
    G = gg.load.from_edges([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)],
                           directed=False)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    row = gg.compare_curvature(G, list(OT_METHODS)).agreement()[0]
    assert row["spearman"] == pytest.approx(1.0)


def test_rank_ties_are_anchored_not_chained():
    """Within tol of the group's smallest value ties; a slow ramp is not one tie."""
    import numpy as np

    from graph_geometry.compare import _rankdata

    ramp = np.array([0.0, 0.6, 1.2, 1.8])
    assert _rankdata(ramp, tol=1.0) == pytest.approx([1.5, 1.5, 3.5, 3.5])
    noisy = np.array([0.5, 1e-16, 0.5 + 2e-16, -1e-17])
    assert _rankdata(noisy, tol=1e-4) == pytest.approx([3.5, 1.5, 3.5, 1.5])
    assert _rankdata(noisy) == pytest.approx([3.0, 2.0, 4.0, 1.0])


# ── parameters, export ──────────────────────────────────────────────────────


def test_per_method_params_are_forwarded():
    """alpha=0.0 must reach the method, not be silently replaced by the default."""
    G = _undirected()
    cmp = gg.compare_curvature(G, ["ollivier"], params={"ollivier": {"alpha": 0.0}})

    lazy = gg.curvature(G.copy(), method="ollivier", alpha=0.0)
    default = gg.curvature(G.copy(), method="ollivier")      # alpha=0.5
    assert cmp.values["ollivier"] == pytest.approx(lazy)
    assert cmp.values["ollivier"] != pytest.approx(default)


def test_params_for_an_unlisted_method_are_ignored():
    G = _undirected()
    cmp = gg.compare_curvature(G, ["forman_node_weighted"], params={"ollivier": {"alpha": 0.0}})
    assert set(cmp.values) == {"forman_node_weighted"}
    assert not cmp.errors


def test_tidy_rows_and_wide_table_agree():
    cmp = gg.compare_curvature(_undirected(), ["forman_node_weighted", "augmented_forman_node_weighted"])
    rows = cmp.rows()
    assert len(rows) == 2 * 10
    wide = cmp.wide()
    assert len(wide) == 10
    for edge, per_method in wide.items():
        assert set(per_method) == {"forman_node_weighted", "augmented_forman_node_weighted"}


def test_to_csv_writes_one_row_per_edge(tmp_path):
    cmp = gg.compare_curvature(_digraph_with_sinks(),
                               ["lin_lu_yau", "forman_directed", "eidi_jost"])
    path = cmp.to_csv(str(tmp_path / "cmp.csv"))
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    # the failed method gets no column at all
    assert set(rows[0]) == {"source", "target", "forman_directed", "eidi_jost"}


# ── flow comparison ─────────────────────────────────────────────────────────


def test_compare_flow_runs_presets_and_an_expression():
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)])
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    out = gg.compare_flow(
        G,
        {"normalized": "normalized",
         "additive": "additive",
         "bounded": "-eta*tanh(kappa - kbar)*w"},
        curvature="forman_directed", iterations=5, step=0.05, early_stop=False,
    )
    assert set(out) == {"normalized", "additive", "bounded"}
    for label, result in out.items():
        assert "error" not in result, (label, result.get("error"))
        assert len(result["convergence"]) == 5


def test_compare_flow_records_a_failing_equation():
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0)])
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    out = gg.compare_flow(G, {"bad": "-eta*(kappa - kbar)*w + __import__('os')"},
                          curvature="forman_directed", iterations=2)
    assert "error" in out["bad"]


# ── sign statistics must not be built on solver residue ──────────────────────
# Regression: the historical CVXPY/SCS fallback left a ~1e-5 residual while
# Ollivier returned exact 0.0, so np.sign() of a flat edge disagreed with itself. See
# RicciFlow-Computation.md, "Sign-derived statistics must apply a solver
# tolerance".


def _flat(graph):
    """A graph on which every edge has curvature exactly zero."""
    G = graph
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    return G


def _agreement(G, tol=None):
    comparison = gg.compare_curvature(G, list(OT_METHODS))
    rows = comparison.agreement() if tol is None else comparison.agreement(tol=tol)
    return rows[0]["sign_agreement"]


def _pct_negative(G, method, tol=None):
    comparison = gg.compare_curvature(G, list(OT_METHODS))
    rows = comparison.summary() if tol is None else comparison.summary(tol=tol)
    return next(r["pct_negative"] for r in rows if r["method"] == method)


def test_flat_cycle_methods_agree_on_sign():
    """C8 is flat everywhere; sign_agreement used to be 0.00 instead of 1.00."""
    assert _agreement(_flat(nx.cycle_graph(8))) == pytest.approx(1.0)


def test_flat_grid_methods_agree_on_sign():
    """The 4x4 grid residual is 1.1e-6, which a 1e-6 tolerance would not catch."""
    assert _agreement(_flat(nx.grid_2d_graph(4, 4))) == pytest.approx(1.0)


def test_flat_grid_reports_no_negative_curvature():
    """LLY residue used to make pct_negative 16.7% on a graph with no negatives."""
    assert _pct_negative(_flat(nx.grid_2d_graph(4, 4)), "lin_lu_yau") == 0.0


def test_default_solver_leaves_no_residue_to_tolerate():
    """The default LP solver is exact here, so even a raw sign now agrees."""
    assert _agreement(_flat(nx.cycle_graph(8)), tol=0.0) == pytest.approx(1.0)


@pytest.mark.skipif(
    "SCS" not in __import__("cvxpy").installed_solvers(), reason="SCS not installed"
)
def test_tolerance_still_needed_for_an_inexact_solver():
    """The knob is not vestigial: pick SCS and the raw sign disagrees again.

    This is the defect the tolerance was added for. It is no longer reachable
    with the default solver, but remains reachable for anyone who selects a
    conic solver for the Lin-Lu-Yau LP.
    """
    G = _flat(nx.cycle_graph(8))
    comparison = gg.compare_curvature(
        G, list(OT_METHODS), params={"lin_lu_yau": {"solver": "SCS"}}
    )
    assert comparison.agreement(tol=0.0)[0]["sign_agreement"] < 1.0
    assert comparison.agreement()[0]["sign_agreement"] == pytest.approx(1.0)


def test_tolerance_does_not_hide_real_negative_curvature():
    """A double star has kappa = -0.5 on its bridge; that must still be negative."""
    G = nx.Graph()
    G.add_edge("u", "v")
    for i in range(3):
        G.add_edge("u", f"a{i}")
        G.add_edge("v", f"b{i}")
    G = _flat(G)
    kappa = gg.curvature(G.copy(), method="ollivier")
    assert kappa[("u", "v")] == pytest.approx(-0.5)
    assert _pct_negative(G, "ollivier") > 0.0


def test_sign_tolerance_constant_is_the_documented_solver_budget():
    from graph_geometry.compare import SIGN_TOLERANCE

    assert SIGN_TOLERANCE == pytest.approx(1e-4)

# ── flow comparison as a result object (audit) ──────────────────────────────


def test_flow_comparison_summary_failure_kinds_and_csv():
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], directed=False)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    cmp = gg.compare_flow(
        G,
        {"normalized": "normalized", "bad": "__import__('os')", "explode": "-1000*w"},
        curvature=gg.CurvatureRequest("ollivier"), semantics=gg.GraphSemantics(),
        iterations=3, step=0.05, early_stop=False,
    )
    assert isinstance(cmp, gg.FlowComparison)
    rows = {row["equation"]: row for row in cmp.summary()}
    assert rows["normalized"]["termination_reason"] == "iterations"
    assert rows["normalized"]["spread_ratio"] is not None
    assert rows["bad"]["failure_kind"] == "configuration"
    assert rows["explode"]["failure_kind"] == "numerical_failure"
    assert cmp.csv_text().splitlines()[0] == "iteration,normalized"
    assert cmp["bad"]["error"] and "convergence" in cmp["normalized"]     # mapping view


def test_flow_comparison_records_an_inapplicable_curvature():
    cmp = gg.compare_flow(nx.cycle_graph(5), {"normalized": "normalized"},
                          curvature="eidi_jost", iterations=2)
    assert cmp.failures["normalized"].kind == "inapplicable"
