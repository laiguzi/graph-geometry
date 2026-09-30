"""Balancing factor (beta) sweep for directed Lin-Lu-Yau curvature.

Reproduces the core idea of Bai, Li, Liu, Lai (arXiv:2509.19989): on a directed
graph the transition kernel mixes out- and in-walks,

    P = beta * P_out + (1 - beta) * P_in,

so the "balancing factor" beta interpolates between purely forward (beta=1) and
purely backward (beta=0) curvature. beta = deg_out / deg ("degree_proportional")
gives every incident edge equal attention.

This prints the mean/spread of edge curvature across beta, and (if matplotlib is
installed) saves a figure.

Run:  uv run python examples/balancing_factor.py
      uv run --extra viz python examples/balancing_factor.py   # + figure
"""

import numpy as np

import graph_geometry as gg


def build_graph():
    # an asymmetric directed graph so beta actually matters
    G = gg.load.from_edges(
        [(0, 1), (1, 2), (2, 0), (0, 2), (2, 3), (3, 0), (1, 3)], directed=True
    )
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    return G


def sweep(G, betas):
    rows = []
    for beta in betas:
        kappa = gg.curvature(G.copy(), method="lin_lu_yau", kernel="mixed", beta=beta)
        vals = np.array(list(kappa.values()))
        rows.append((beta, vals.mean(), vals.min(), vals.max()))
    # the degree-proportional strategy (node-wise beta)
    kappa = gg.curvature(G.copy(), method="lin_lu_yau", kernel="mixed",
                         beta="degree_proportional")
    vals = np.array(list(kappa.values()))
    return rows, ("degree_proportional", vals.mean(), vals.min(), vals.max())


def main():
    G = build_graph()
    betas = np.linspace(0.0, 1.0, 11)
    rows, dp = sweep(G, betas)

    print(f"{'beta':>18} | {'mean_kappa':>10} | {'min':>7} | {'max':>7}")
    print("-" * 52)
    for beta, mean, lo, hi in rows:
        print(f"{beta:18.2f} | {mean:10.4f} | {lo:7.3f} | {hi:7.3f}")
    print(f"{dp[0]:>18} | {dp[1]:10.4f} | {dp[2]:7.3f} | {dp[3]:7.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(install the [viz] extra to also save a figure)")
        return

    b = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(b, [r[1] for r in rows], "-o", label="mean κ")
    ax.fill_between(b, [r[2] for r in rows], [r[3] for r in rows], alpha=0.15,
                    label="min–max")
    ax.set_xlabel("balancing factor β")
    ax.set_ylabel("Lin-Lu-Yau curvature")
    ax.set_title("Directed LLY curvature vs. balancing factor")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = "balancing_factor.png"
    fig.savefig(out, dpi=120)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
