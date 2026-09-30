"""Numerical parity vs. my_ricci_aging's own code (the fork we replace).

Goldens in ``tests/reference/mra_goldens.json`` were produced by running
my_ricci_aging's actual implementations on its 8 metabolite correlation networks:

- ``di_lly`` -- ``ricciflow.RicciCurvature(beta=0.5, alpha=None)`` (directed LLY);
- ``forman`` / ``augmented_forman`` -- ``FormanRicciCurvature``.

The graph structure (weight/distance) is embedded, so the fixture is
self-contained. graph_geometry must reproduce every ``di_lly`` value to ~1e-4
(OT solver tolerance).

The two Forman columns are kept as recorded data but are no longer checked: they
are my_ricci_aging's incident-weight convention (leading term ``W(u)+W(v)``,
no node weights), which matches no published definition and was withdrawn from
the package. The definitions that replaced it, ``forman_directed`` and
``augmented_forman_directed`` (Saucan et al. 2019), are different formulas and
are pinned in ``test_forman_directed.py`` instead.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

import graph_geometry as gg

GOLDENS = json.loads(
    (Path(__file__).parent / "reference" / "mra_goldens.json").read_text()
)

# graph_geometry call matching each my_ricci_aging method.
METHOD_CALL = {
    "di_lly": dict(method="lin_lu_yau", kernel="mixed", beta=0.5),
}


def _build(case):
    G = nx.DiGraph()
    for u, v, w, d in GOLDENS[case]["graph"]:
        G.add_edge(u, v, weight=w, distance=d)
    return G


CASES = sorted(GOLDENS)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("method", list(METHOD_CALL))
def test_matches_my_ricci_aging(case, method):
    G = _build(case)
    got = gg.curvature(G, **METHOD_CALL[method])
    # di_lly was 1e-9, which only held because the goldens and the package
    # shared one inaccurate solver; see LLY_GOLDEN_ATOL in test_curvature_ot.py.
    tol = 1e-4
    for u, v, gold in GOLDENS[case][method]:
        assert got[(u, v)] == pytest.approx(gold, abs=tol), f"{case}/{method} edge ({u},{v})"


def test_covers_all_networks_and_methods():
    assert len(CASES) == 8  # 4 ages x {spearman, lasso_z}
    assert all(set(GOLDENS[c]) >= {"graph", "di_lly", "forman", "augmented_forman"} for c in CASES)
