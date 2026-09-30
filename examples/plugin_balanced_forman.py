"""Plugin example B — a non-OT curvature added through the result contract.

Balanced Forman curvature (Topping, Di Giovanni, Chamberlain, Dong, Bronstein,
*Understanding over-squashing and bottlenecks on graphs via curvature*,
ICLR 2022, arXiv:2111.14522, Definition 1). This is the curvature the
over-squashing / graph-rewiring literature uses; it is a sharp lower bound on
the Lin-Lu-Yau curvature and, unlike the OT family, is purely combinatorial.

For an edge ``i ~ j`` of a simple unweighted graph, with ``S(n)`` the neighbour
set and ``d_n = |S(n)|``:

    #_T(i,j) = S(i) & S(j)                                    triangles at i~j
    #_Q^i    = { k in S(i)\\S(j), k != j :
                   exists w in (S(k) & S(j)) \\ (S(i) | {i}) } 4-cycles at i
    #_Q^j    = symmetric in i, j
    gamma    = max number of such 4-cycles through any single k

    Ric(i,j) = 0                                     if min(d_i, d_j) == 1
    Ric(i,j) = 2/d_i + 2/d_j - 2
               + 2|#_T|/max(d_i,d_j) + |#_T|/min(d_i,d_j)
               + (|#_Q^i| + |#_Q^j|) / (gamma * max(d_i,d_j))    if gamma > 0

Published values (their Table 1), all checked in
``tests/test_example_plugins.py``: ``K_n -> n/(n-1)``, 2-D grid ``-> 0``,
``(r+1)``-regular tree ``-> 4/(r+1) - 2``.

The definition is stated for **simple unweighted undirected** graphs, and the
registration says exactly that instead of reinterpreting other input:

- ``graph_kinds={"undirected"}``: a digraph is refused before the compute
  function runs, rather than silently read through its underlying undirected
  neighbourhood;
- ``consumes={"topology"}``: edge weights and distances are not read, so they
  are not validated, and a flow that evolves them warns that this curvature is
  invariant under it. A flow driven by this curvature cannot converge without
  surgery: its values depend on the topology alone.

The compute function has the modern signature ``(G, *, semantics, proc)`` and
returns ``{(u, v): value}``; validating that mapping, deriving node means and
annotating the graph all belong to the evaluator.

Run this file to print the published anchors::

    python examples/plugin_balanced_forman.py
"""

from __future__ import annotations

import networkx as nx

import graph_geometry as gg


def _balanced_forman_edge(S: dict, i, j) -> float:
    d_i, d_j = len(S[i]), len(S[j])
    if min(d_i, d_j) <= 1:
        return 0.0
    d_max, d_min = max(d_i, d_j), min(d_i, d_j)

    n_tri = len(S[i] & S[j])

    # 4-cycles based at i~j with no diagonal: count, per k, how many exist.
    cycles_i = [
        len((S[k] & S[j]) - S[i] - {i})
        for k in S[i] - S[j] - {j}
    ]
    cycles_j = [
        len((S[k] & S[i]) - S[j] - {j})
        for k in S[j] - S[i] - {i}
    ]
    counts = [c for c in cycles_i + cycles_j if c > 0]

    ric = 2 / d_i + 2 / d_j - 2 + 2 * n_tri / d_max + n_tri / d_min
    if counts:
        ric += len(counts) / (max(counts) * d_max)
    return ric


@gg.register_curvature(
    "balanced_forman",
    capabilities=gg.CurvatureCapabilities(
        family="combinatorial",
        status="experimental",
        graph_kinds=frozenset({"undirected"}),
        scopes=frozenset({"edges"}),
        consumes=frozenset({"topology"}),
    ),
    label="Balanced Forman",
    color="#bcbd22",
    description="Balanced Forman curvature; combinatorial, defined on simple unweighted graphs.",
    citation=(
        "Topping, J., Di Giovanni, F., Chamberlain, B. P., Dong, X. & Bronstein, "
        "M. M. (2022). Understanding over-squashing and bottlenecks on graphs via "
        "curvature. ICLR 2022, arXiv:2111.14522."
    ),
)
def balanced_forman(G, *, semantics, proc):
    """Balanced Forman curvature. See the module docstring for the formula."""
    S = {n: set(G.neighbors(n)) for n in G}
    return {(u, v): _balanced_forman_edge(S, u, v) for u, v in G.edges()}


def main() -> None:
    for n in (3, 4, 5):
        values = gg.compute_curvature(gg.generate.complete(n), "balanced_forman").values
        print(f"K_{n}:   Ric = {next(iter(values.values())):.6f}   "
              f"(expected {n / (n - 1):.6f})")

    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6))
    values = gg.compute_curvature(grid, "balanced_forman").values
    interior = [v for e, v in values.items()
                if grid.degree(e[0]) == 4 and grid.degree(e[1]) == 4]
    print(f"grid:   Ric = {max(interior):.6f}   (expected 0.000000, interior edges)")

    tree = nx.balanced_tree(3, 4)          # branching 3 -> interior degree 4
    values = gg.compute_curvature(tree, "balanced_forman").values
    interior = [v for e, v in values.items()
                if tree.degree(e[0]) == 4 and tree.degree(e[1]) == 4]
    print(f"tree:   Ric = {interior[0]:.6f}   (expected {4 / 4 - 2:.6f})")


if __name__ == "__main__":
    main()
