"""Quickstart: build a graph, compute curvature, run Ricci flow.

Run:  uv run python examples/quickstart.py
"""

import graph_geometry as gg


def main():
    # 1. graph + factors -------------------------------------------------
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)], directed=True)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", node_factor="degree")
    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. curvature (several definitions, every convention explicit) ----
    requests = [
        gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed", "beta": 0.8}),
        gg.CurvatureRequest("ollivier", {"kernel": "out", "alpha": 0.5}),
        gg.CurvatureRequest("eidi_jost"),
        gg.CurvatureRequest("forman_directed"),
        gg.CurvatureRequest("augmented_forman_directed"),
    ]
    for request in requests:
        result = gg.compute_curvature(G, request)
        vals = [round(v, 3) for v in result.values.values()]
        print(f"  {request.method:34s} [{result.resolved.direction_convention}] -> {vals}")

    # 3. Ricci flow ------------------------------------------------------
    sim = gg.RicciFlowSimulator.from_networkx(G, curvature=requests[0])
    sim.run(iterations=30, step=0.05, delta=1e-6)
    print(f"\nflow: {len(sim.convergence)} iterations, "
          f"final RC spread = {sim.convergence[-1]:.6f}")
    sim.summary()

    # 4. export ----------------------------------------------------------
    series = gg.SnapshotSeries(sim.snapshots, sim.convergence)
    print(f"\nsnapshots: {len(series)} (initial + one per iteration)")


if __name__ == "__main__":
    main()
