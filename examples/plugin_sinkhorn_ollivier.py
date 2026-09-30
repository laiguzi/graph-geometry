"""Plugin example A — an OT curvature added by replacing ONE transport component.

Entropically regularised Ollivier-Ricci curvature: identical to the built-in
``ollivier`` except that the exact EMD solve is replaced by Sinkhorn's
algorithm (Cuturi 2013). A transport definition is a configuration of the four
components ``(K_c, D_c, S_c, F_c)`` run by the shared transport engine, so the
plugin starts from the built-in Ollivier configuration and replaces only the
solver::

    ollivier_ot_config(...)     Sinkhorn variant
    -----------------------     ------------------------------------
    kernel        K_c      ==   same kernel (resolved convention)
    distribution  D_c      ==   OllivierDistribution(alpha)
    solver        S_c     -->   SinkhornOTSolver(reg)    <-- the only change
    curvature     F_c      ==   OllivierCurvature()

The engine still builds the endpoint measures, computes the shortest-path cost
matrix, evaluates every edge or requested pair and batches over ``proc``.

Sinkhorn finds the entropically regularised plan

    P_reg = argmin_P  <P, C> - reg * H(P).

The reported cost is ``<P_reg, C>`` (without the entropy term), which is at
least the exact W_1 for a feasible plan and approaches it as ``reg -> 0``.
Thus its curvature approaches exact Ollivier curvature from below.
``method="sinkhorn_log"`` runs the iteration in log space.

``reg`` is in the original ground-cost units. The solver explicitly supports
engine conditioning through ``for_cost_scale``: dividing C by a scale also
divides reg by that scale, preserving the plan and the user's definition.

The registration declares everything the package needs before any numerical
work: both graph kinds, edges and node pairs, the three channels it reads, and
a validated parameter schema. Its resolver turns ``kernel="auto"`` into the
convention actually used (``undirected``, or ``mixed`` on a digraph, as the
built-in ``ollivier`` does), so no resolved record keeps ``"auto"``.

Run this file to see the convergence table::

    python examples/plugin_sinkhorn_ollivier.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import ot

import graph_geometry as gg
from graph_geometry.curvature import (
    TransportCurvatureEngine,
    TransportResult,
    ollivier_ot_config,
)

KERNELS = ("auto", "undirected", "out", "in", "mixed")


@dataclass
class SinkhornOTSolver:
    """Entropic OT (POT) in the transport-solver protocol. Picklable, so ``proc > 1`` works.

    ``(source_mass, target_mass, cost_matrix, *, return_plan=False) -> TransportResult``.

    Includes a convergence check: with POT's default ``numItermax=1000`` and
    ``reg=0.001`` the log-domain iteration returns ``2.7e-134`` for a transport
    whose exact cost is ``0.5``, without raising. Verifying the returned plan's
    marginals against ``a`` and ``b`` catches that case.
    """

    reg: float = 0.01
    num_iter: int = 100_000
    tol: float = 1e-6

    def for_cost_scale(self, scale: float) -> SinkhornOTSolver | None:
        """Configure a copy for C / scale while retaining the same optimal plan."""
        reg = self.reg / scale
        if not np.isfinite(reg) or reg <= 0.0:
            return None  # preserve original units if reg cannot be represented
        return replace(self, reg=reg)

    def __call__(self, source_mass, target_mass, cost_matrix, *, return_plan=False):
        a = np.asarray(source_mass, dtype=float)
        b = np.asarray(target_mass, dtype=float)
        M = np.asarray(cost_matrix, dtype=float)
        plan = ot.sinkhorn(a, b, M, self.reg,
                           method="sinkhorn_log", numItermax=self.num_iter)
        err = max(np.abs(plan.sum(axis=1) - a).max(),
                  np.abs(plan.sum(axis=0) - b).max())
        if not np.isfinite(err) or err > self.tol:
            raise gg.CurvatureNumericalError(
                f"Sinkhorn did not converge at reg={self.reg}: marginal error "
                f"{err:.3g} > {self.tol}. Increase reg or num_iter."
            )
        return TransportResult(cost=float((plan * M).sum()),
                               plan=plan if return_plan else None)


def sinkhorn_ollivier_config(alpha, reg, kernel, beta=0.8):
    """The built-in Ollivier configuration with its solver replaced."""
    exact = ollivier_ot_config(alpha=alpha, kernel=kernel, beta=beta)
    return replace(exact, name=f"SinkhornOllivier(alpha={alpha}, reg={reg})",
                   solver=SinkhornOTSolver(reg))


def resolve_sinkhorn_ollivier(*, graph_kind, scope, parameters):
    """Name the kernel convention actually used; reject one the graph cannot have."""
    kernel, notes = parameters["kernel"], ()
    if kernel == "auto":
        kernel = "undirected" if graph_kind == "undirected" else "mixed"
        if graph_kind == "directed":
            notes = (f"kernel='auto' on a directed graph resolved to 'mixed' "
                     f"(beta={parameters['beta']!r}); name the kernel explicitly.",)
    elif (kernel == "undirected") != (graph_kind == "undirected"):
        raise gg.CurvatureDomainError(
            f"kernel={kernel!r} does not apply to a {graph_kind} graph; use "
            f"'undirected' on an undirected graph and 'out', 'in' or 'mixed' on a digraph."
        )
    return {**parameters, "kernel": kernel}, kernel, notes


def sinkhorn_ollivier_pairs(G, pairs, *, semantics, proc, alpha, reg, kernel, beta=0.8):
    config = sinkhorn_ollivier_config(alpha, reg, kernel, beta)
    return TransportCurvatureEngine(G, config, semantics, proc).compute_pairs(pairs)


@gg.register_curvature(
    "sinkhorn_ollivier",
    capabilities=gg.CurvatureCapabilities(
        family="transport",
        status="experimental",
        graph_kinds=frozenset({"directed", "undirected"}),
        scopes=frozenset({"edges", "pairs"}),
        consumes=frozenset({"topology", "edge_weight", "edge_distance"}),
    ),
    parameters=(
        gg.ParameterSpec("alpha", "float", 0.5, minimum=0.0, maximum=1.0,
                         description="Idleness: mass kept at the node itself."),
        gg.ParameterSpec("reg", "float", 0.01, minimum=0.0, minimum_inclusive=False,
                         description="Entropic regularisation in ground-cost units; smaller is closer to exact."),
        gg.ParameterSpec("kernel", "choice", "auto", choices=KERNELS,
                         description="Neighbourhood convention; 'auto' is resolved."),
        gg.ParameterSpec("beta", "float", 0.8, minimum=0.0, maximum=1.0,
                         graph_kinds=frozenset({"directed"}),
                         active_when={"kernel": ("mixed",)},
                         description="Mixed-kernel balance: beta*P_out + (1-beta)*P_in."),
    ),
    compute_pairs=sinkhorn_ollivier_pairs,
    resolver=resolve_sinkhorn_ollivier,
    label="Ollivier (Sinkhorn)",
    color="#17becf",
    description="Entropically regularised Ollivier-Ricci curvature (Sinkhorn solver).",
    citation=(
        "Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal "
        "transport. NeurIPS 26, 2292-2300; Ollivier, Y. (2009). Ricci curvature of "
        "Markov chains on metric spaces. J. Funct. Anal. 256(3), 810-864."
    ),
)
def sinkhorn_ollivier(G, *, semantics, proc, alpha, reg, kernel, beta=0.8):
    """Ollivier-Ricci curvature with the EMD solve replaced by Sinkhorn."""
    config = sinkhorn_ollivier_config(alpha, reg, kernel, beta)
    return TransportCurvatureEngine(G, config, semantics, proc).compute_edges()


def main() -> None:
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], directed=True)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")

    request = {"alpha": 0.5, "kernel": "mixed"}
    exact = gg.compute_curvature(G, gg.CurvatureRequest("ollivier", request)).values
    print(f"{'reg':>10} {'max |kappa_reg - kappa|':>26}")
    for reg in (1.0, 0.1, 0.01):
        approx = gg.compute_curvature(
            G, gg.CurvatureRequest("sinkhorn_ollivier", {**request, "reg": reg})
        ).values
        gap = max(abs(approx[e] - exact[e]) for e in exact)
        print(f"{reg:>10} {gap:>26.6f}")


if __name__ == "__main__":
    main()
