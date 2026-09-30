"""Numerical parity for OT curvature vs. the ricciflow_sim reference.

Golden values in ``tests/reference/ot_curvature_goldens.json`` were produced by
the reference ``ricciflow_sim`` pipeline. This is the Phase-2 acceptance gate.
"""

import json
import math
from pathlib import Path

import cvxpy
import networkx as nx
import pytest

import graph_geometry as gg

GOLDENS = json.loads(
    (Path(__file__).parent / "reference" / "ot_curvature_goldens.json").read_text()
)

ATOL = 1e-6

# The Lin-Lu-Yau goldens were produced by the reference implementations, which
# all requested CVXPY's ECOS and silently fell back to SCS when it was absent
# (it always was). SCS solves this LP to about 2.6e-5, so the goldens carry that
# error: the K4 golden is 1.33333556 where the hand value is exactly 4/3, and the
# S6 golden is 0.40000659 where it is exactly 2/5. This package now defaults to
# an LP solver (scipy/HiGHS, 5e-16), so it no longer reproduces that error.
# Parity with the reference implementations therefore holds to *their* solver's
# accuracy, not to 1e-6. Measured drift: curvature 1.9e-5, flow 1.6e-5, MRA
# 7.7e-5 -- all inside 1e-4. Do not tighten this back: doing so only passes if
# the package reintroduces the same solver error. Ollivier and Forman goldens are
# unaffected (drift 0.0) and stay tight.
LLY_GOLDEN_ATOL = 1e-4


def _assign(G):
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    return G


def _graph(name):
    if name == "tri3":
        return nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    if name == "c3e4":
        return nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2)])
    if name == "k4":
        return nx.complete_graph(4)
    if name == "c5":
        return nx.cycle_graph(5)
    raise KeyError(name)


# golden config name -> gg.curvature kwargs
METHODS = {
    "lly_auto": dict(method="lin_lu_yau", kernel="auto", beta=0.8),
    "lly_mixed_b08": dict(method="lin_lu_yau", kernel="mixed", beta=0.8),
    "lly_out": dict(method="lin_lu_yau", kernel="out"),
    "ollivier_a05_auto": dict(method="ollivier", alpha=0.5, kernel="auto", beta=0.8),
}


@pytest.mark.parametrize("case", sorted(GOLDENS))
def test_ot_curvature_matches_reference(case):
    gname, cname = case.split("__")
    G = _assign(_graph(gname))
    result = gg.curvature(G, **METHODS[cname])

    # edge parity (both the returned dict and the written attribute)
    golden = GOLDENS[case]
    tol = LLY_GOLDEN_ATOL if "lly" in cname else ATOL
    got_edges = {(str(u), str(v)): val for (u, v), val in result.items()}
    for su, sv, gold in golden["edges"]:
        assert (su, sv) in got_edges, f"missing edge ({su},{sv})"
        assert got_edges[(su, sv)] == pytest.approx(gold, abs=tol)

    # node parity
    got_nodes = {
        str(n): G.nodes[n]["ricciCurvature"]
        for n in G.nodes()
        if "ricciCurvature" in G.nodes[n]
    }
    for sn, gold in golden["nodes"]:
        assert got_nodes[sn] == pytest.approx(gold, abs=tol)


def test_curvature_writes_edge_and_node_attrs():
    G = _assign(nx.complete_graph(4))
    gg.curvature(G, method="lin_lu_yau")
    assert all("ricciCurvature" in G[u][v] for u, v in G.edges())
    assert all("ricciCurvature" in G.nodes[n] for n in G.nodes())


def test_curvature_config_object_dispatch():
    # passing a CurvatureConfig must match dispatching by registered name
    G1 = _assign(nx.complete_graph(4))
    G2 = _assign(nx.complete_graph(4))
    by_config = gg.curvature(G1, method=gg.lin_lu_yau(kernel="auto"))
    by_name = gg.curvature(G2, method="lin_lu_yau", kernel="auto")
    for e in by_name:
        assert by_config[e] == pytest.approx(by_name[e], abs=ATOL)
    # It is the K4 LLY value 4/3. This broad legacy assertion is retained for
    # cross-solver portability; the default solver's 1e-12 contract is pinned below.
    assert all(v == pytest.approx(4 / 3, abs=1e-4) for v in by_config.values())


def test_analytic_anchors_complete_graph_and_cycles():
    """Known Lin-Lu-Yau values, asserted against theory rather than a golden.

    K_n has kappa = n/(n-1); C_5 has kappa = 1/2 and longer cycles are flat.
    These are the analytic anchors quoted in the software paper. This broad
    1e-4 assertion remains portable to an explicitly selected inexact solver;
    separate tests below pin the default SciPy/HiGHS path at 1e-12.
    """
    for n in (3, 4, 5):
        G = _assign(nx.complete_graph(n))
        values = gg.curvature(G, method="lin_lu_yau").values()
        assert all(v == pytest.approx(n / (n - 1), abs=1e-4) for v in values), n

    G = _assign(nx.cycle_graph(5))
    assert all(v == pytest.approx(0.5, abs=1e-4)
               for v in gg.curvature(G, method="lin_lu_yau").values())

    for n in (6, 7):
        G = _assign(nx.cycle_graph(n))
        assert all(v == pytest.approx(0.0, abs=1e-4)
                   for v in gg.curvature(G, method="lin_lu_yau").values()), n


def test_self_loop_is_rejected_before_any_solve():
    # The core policy is simple and loopless. my_ricci_aging silently returned
    # 0.0 on a self-loop; the loop is now an explicit preprocessing decision.
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    G.add_edge(1, 1)
    _assign(G)
    with pytest.raises(gg.CurvatureDomainError, match="self-loops"):
        gg.curvature(G, method="lin_lu_yau")
    assert "ricciCurvature" not in G[0][1]
    G.remove_edges_from(nx.selfloop_edges(G))
    assert set(gg.curvature(G, method="lin_lu_yau")) == set(G.edges())


def test_ot_on_non_strongly_connected_raises_clear_error():
    # 0->1->2 is not strongly connected; OT curvature must raise a helpful error
    G = nx.DiGraph([(0, 1), (1, 2)])
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
        G[u][v]["distance"] = 1.0
    with pytest.raises(ValueError, match="strongly connected"):
        gg.curvature(G, method="lin_lu_yau")
    # a combinatorial method still works on the same graph
    assert gg.curvature(G.copy(), method="forman_directed")


def test_proc2_matches_proc1():
    # parallel path must give identical numbers (picklable config classes)
    G1 = _assign(nx.complete_graph(5))
    G2 = _assign(nx.complete_graph(5))
    r1 = gg.curvature(G1, method="lin_lu_yau", proc=1)
    r2 = gg.curvature(G2, method="lin_lu_yau", proc=2)
    for e in r1:
        assert r1[e] == pytest.approx(r2[e], abs=ATOL)


@pytest.mark.parametrize("proc", [2, 4, 8])
def test_proc_actually_splits_the_work(proc):
    """Regression: a hard-coded batch size of 2000 made every graph under 2000
    edges a SINGLE batch, so ``pool.map`` gave all the work to one worker and
    ``proc > 1`` bought nothing. Batches must now follow ``proc``.
    """
    from graph_geometry.curvature.base import RicciCurvature
    from graph_geometry.curvature.ot_ricci import lin_lu_yau

    G = _assign(nx.complete_graph(12))          # 66 edges, far below 2000
    n_edges = G.number_of_edges()

    serial = RicciCurvature(G, lin_lu_yau(), "weight", "distance", proc=1)
    assert math.ceil(n_edges / serial._batch_size(n_edges)) == 1

    parallel = RicciCurvature(G, lin_lu_yau(), "weight", "distance", proc=proc)
    n_batches = math.ceil(n_edges / parallel._batch_size(n_edges))
    assert n_batches >= proc, (
        f"proc={proc} would produce {n_batches} batch(es) for {n_edges} edges; "
        "workers beyond the first would sit idle"
    )


def test_proc_batching_never_loses_or_duplicates_edges():
    from graph_geometry.curvature.base import RicciCurvature
    from graph_geometry.curvature.ot_ricci import lin_lu_yau

    for n_edges in (1, 2, 7, 66, 1999, 5000):
        for proc in (1, 3, 8):
            engine = RicciCurvature(nx.Graph(), lin_lu_yau(), "weight",
                                    "distance", proc=proc)
            size = engine._batch_size(n_edges)
            assert size >= 1
            edges = list(range(n_edges))
            batches = [edges[i:i + size] for i in range(0, n_edges, size)]
            assert sum(len(b) for b in batches) == n_edges
            assert [e for b in batches for e in b] == edges


# ── Lin-Lu-Yau solver selection ──────────────────────────────────────────────
# Regression: the solve requested ECOS and fell back to SCS on any exception.
# ECOS was never a declared dependency, so every published LLY value came from
# SCS while the documentation attributed it to ECOS.


def test_default_solver_is_installed_and_reported():
    from graph_geometry.curvature.ot_ricci import (
        LLY_SOLVER_PREFERENCE,
        resolve_lly_solver,
    )

    resolved = resolve_lly_solver()
    assert resolved in cvxpy.installed_solvers()
    assert resolved in LLY_SOLVER_PREFERENCE
    assert gg.lin_lu_yau().ot_solver.solver_name == resolved


def test_missing_solver_raises_instead_of_falling_back():
    """The silent ECOS->SCS fallback is what made the old claims untrue."""
    from graph_geometry.curvature.ot_ricci import resolve_lly_solver

    absent = "NO_SUCH_SOLVER"
    assert absent not in cvxpy.installed_solvers()
    with pytest.raises(ValueError, match="not installed"):
        resolve_lly_solver(absent)
    with pytest.raises(ValueError, match="not installed"):
        gg.lin_lu_yau(solver=absent)


def test_default_solver_is_exact_on_a_flat_cycle():
    """C8 is flat; the old default returned +5.266e-08 on every edge."""
    kappa = gg.curvature(_assign(nx.cycle_graph(8)), method="lin_lu_yau")
    assert all(v == pytest.approx(0.0, abs=1e-12) for v in kappa.values())


def test_default_solver_reproduces_closed_forms_exactly():
    """K4 = 4/3 and S6 = 2/5 exactly; the stored goldens are off by ~1e-5."""
    k4 = gg.curvature(_assign(nx.complete_graph(4)), method="lin_lu_yau")
    assert all(v == pytest.approx(4 / 3, abs=1e-12) for v in k4.values())
    s6 = gg.curvature(_assign(nx.star_graph(5)), method="lin_lu_yau")
    assert all(v == pytest.approx(2 / 5, abs=1e-12) for v in s6.values())


def test_solver_is_selectable_through_the_registry():
    kappa = gg.curvature(
        _assign(nx.cycle_graph(6)), method="lin_lu_yau", solver=cvxpy.installed_solvers()[0]
    )
    assert len(kappa) == 6


# ── unit invariance: no absolute thresholds ─────────────────────────────────
# An absolute EPSILON=1e-7 used to (a) define kappa = 0 when d(x, y) < 1e-7 and
# (b) switch the kernel to the uniform measure when a node's weight sum was
# below 1e-7. Both made kappa depend on the unit of measurement.


def _k3(scale):
    G = nx.complete_graph(3)
    for u, v in G.edges():
        G[u][v]["weight"] = scale
        G[u][v]["distance"] = scale
    return G


def _weighted(weight_scale=1.0, distance_scale=1.0):
    G = nx.path_graph(4)
    G.add_edge(1, 3)
    for (u, v), w in {(0, 1): 1, (1, 2): 3, (2, 3): 1, (1, 3): 2}.items():
        G[u][v]["weight"] = w * weight_scale
        G[u][v]["distance"] = (1 + w / 4) * distance_scale
    return G


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1e-6, 1e6, 1e12])
@pytest.mark.parametrize("request_", [
    gg.CurvatureRequest("ollivier", {"alpha": 0.5}),
    gg.CurvatureRequest("ollivier", {"alpha": 0.0}),
    "lin_lu_yau",
])
def test_curvature_is_invariant_under_a_common_scale(scale, request_):
    base = gg.compute_curvature(_k3(1.0), request_).values
    scaled = gg.compute_curvature(_k3(scale), request_).values
    for e in base:
        assert scaled[e] == pytest.approx(base[e], abs=1e-7)

    base = gg.compute_curvature(_weighted(), request_).values
    for kwargs in ({"weight_scale": scale}, {"distance_scale": scale},
                   {"weight_scale": scale, "distance_scale": scale}):
        scaled = gg.compute_curvature(_weighted(**kwargs), request_).values
        for e in base:
            assert scaled[e] == pytest.approx(base[e], abs=1e-7), (kwargs, e)


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_directed_curvature_is_invariant_under_a_common_scale(scale):
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0)])
    for i, (u, v) in enumerate(G.edges()):
        G[u][v]["weight"] = 1.0 + i
        G[u][v]["distance"] = 1.0 + i / 3
    H = G.copy()
    for u, v in H.edges():
        H[u][v]["weight"] *= scale
        H[u][v]["distance"] *= scale
    for request_ in (
        gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed", "beta": 0.8}),
        gg.CurvatureRequest("ollivier", {"kernel": "mixed", "beta_strategy": "weight_proportional"}),
        gg.CurvatureRequest("ollivier", {"kernel": "out"}),
        "eidi_jost",
    ):
        a = gg.compute_curvature(G, request_).values
        b = gg.compute_curvature(H, request_).values
        for e in a:
            assert b[e] == pytest.approx(a[e], abs=1e-7), (request_, e)
