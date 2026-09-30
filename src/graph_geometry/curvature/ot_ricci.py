"""OT-based curvature building blocks (Lin-Lu-Yau, Ollivier).

Distributions, solvers and the curvature-from-OT maps are picklable classes so
every configuration works with ``proc > 1``.

- :func:`lin_lu_yau_ot_config` / :func:`ollivier_ot_config` build the
  endpoint-pair :class:`~graph_geometry.curvature.transport.OTCurvatureConfig`
  the registry's built-ins run on the shared engine;
- :func:`lin_lu_yau` / :func:`ollivier` are the low-level builders of the
  legacy :class:`CurvatureConfig`, kept for advanced Python use.

Both forms apply one kernel rule at both endpoints and produce identical values.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cvx
import numpy as np
import ot

from .base import CurvatureConfig
from .errors import CurvatureConfigurationError, CurvatureNumericalError
from .kernels import BetaSpec, check_beta, resolve_kernel
from .transport import EMDTransportSolver, OTCurvatureConfig, TransportResult

__all__ = [
    "LLYDistribution",
    "OllivierDistribution",
    "LLYOTSolver",
    "EMDOTSolver",
    "SignedKRTransportSolver",
    "LLYCurvature",
    "OllivierCurvature",
    "LLY_SOLVER_PREFERENCE",
    "resolve_lly_solver",
    "lin_lu_yau",
    "ollivier",
    "lin_lu_yau_ot_config",
    "ollivier_ot_config",
]


# ── distributions ───────────────────────────────────────────────────────────


@dataclass
class LLYDistribution:
    """Lin-Lu-Yau: no self-loop mass (append 0)."""

    def __call__(self, kernel: np.ndarray) -> list:
        return list(kernel) + [0.0]


@dataclass
class OllivierDistribution:
    """Ollivier: self-loop mass = alpha, neighbors get (1 - alpha) * kernel."""

    alpha: float = 0.5

    def __post_init__(self):
        self.alpha = check_beta(self.alpha, context="alpha")

    def __call__(self, kernel: np.ndarray) -> list:
        return [(1 - self.alpha) * p for p in kernel] + [self.alpha]


# ── OT solvers ───────────────────────────────────────────────────────────────


#: Preference order for the Lin-Lu-Yau solve, best-conditioned first.
#:
#: The signed Kantorovich-Rubinstein problem is a *linear* program, so an LP
#: solver is the right tool and a conic/ADMM solver is not. Measured against an
#: exact scipy-HiGHS reference over 213 edges: SCIPY 5.0e-16, CLARABEL 2.2e-08,
#: SCS 2.6e-05, OSQP 3.3e-05. ECOS is kept ahead of SCS only for continuity with
#: the historical default; it is not a declared dependency.
LLY_SOLVER_PREFERENCE = ("SCIPY", "CLARABEL", "ECOS", "SCS")


def resolve_lly_solver(solver: str | None = None) -> str:
    """Return the CVXPY solver to use, or raise naming what is installed.

    ``None`` picks the first installed entry of :data:`LLY_SOLVER_PREFERENCE`.
    An explicit name that is not installed raises instead of falling back: the
    previous code silently degraded from ECOS to SCS whenever ECOS was absent,
    which it always was, so published residuals were attributed to a solver that
    never ran.
    """
    installed = cvx.installed_solvers()
    if solver is not None:
        if solver not in installed:
            raise CurvatureConfigurationError(
                f"CVXPY solver {solver!r} is not installed. Installed: "
                f"{sorted(installed)}. Install it, or pass one of those, or pass "
                f"solver=None to use the first of {list(LLY_SOLVER_PREFERENCE)}."
            )
        return solver
    for candidate in LLY_SOLVER_PREFERENCE:
        if candidate in installed:
            return candidate
    raise RuntimeError(
        f"No usable CVXPY solver installed for Lin-Lu-Yau. Looked for "
        f"{list(LLY_SOLVER_PREFERENCE)}; CVXPY reports {sorted(installed)}."
    )


@dataclass
class LLYOTSolver:
    """Lin-Lu-Yau signed Kantorovich-Rubinstein distance, solved with CVXPY.

    ``solver`` names the CVXPY solver; ``None`` resolves through
    :func:`resolve_lly_solver`. The resolved name is available as
    :attr:`solver_name` so a run can record which solver produced its numbers.
    """

    solver: str | None = None

    def __post_init__(self):
        self.solver_name = resolve_lly_solver(self.solver)

    def for_cost_scale(self, scale: float) -> LLYOTSolver:
        """The signed KR objective is homogeneous in the cost matrix."""
        return self

    def __call__(self, x, y, d) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = np.asarray(d, dtype=float)

        star = cvx.Variable((len(x), len(y)))
        obj = cvx.Maximize(cvx.sum(cvx.multiply(star, d)))
        cons = [cvx.sum(star) == 0]
        # An endpoint whose measure is the Dirac mass on itself has no neighbour
        # entries, so its marginal and sign blocks are empty and are omitted
        # (CVXPY rejects zero-size expressions). The program then evaluates
        # J = d(x, y) - sum_z nu_x(z) d(z, y), the alpha -> 1 limit of Ollivier
        # curvature with that endpoint fixed at delta_y (and 0 when both are).
        if len(y) > 1:
            cons += [
                cvx.sum(star[:, :-1], axis=0, keepdims=True)
                == np.multiply(-1, y.reshape(1, -1)[:, :-1])
            ]
        if len(x) > 1:
            cons += [
                cvx.sum(star[:-1, :], axis=1, keepdims=True)
                == np.multiply(-1, x.reshape(-1, 1)[:-1])
            ]
        cons += [0 <= star[-1, -1], star[-1, -1] <= 2]
        if len(x) > 1 and len(y) > 1:
            cons += [star[:-1, :-1] <= 0]
        if len(y) > 1:
            cons += [star[-1, :-1] <= 0]
        if len(x) > 1:
            cons += [star[:-1, -1] <= 0]
        prob = cvx.Problem(obj, cons)
        try:
            value = prob.solve(solver=self.solver_name)
        except cvx.error.SolverError as exc:
            raise CurvatureNumericalError(
                f"Lin-Lu-Yau transport solve failed with {self.solver_name}: {exc}"
            ) from exc
        if value is None or not np.isfinite(value):
            raise CurvatureNumericalError(
                f"Lin-Lu-Yau transport solve with {self.solver_name} returned "
                f"{value!r} (status {prob.status!r})."
            )
        return value


@dataclass
class SignedKRTransportSolver:
    """The Lin-Lu-Yau signed Kantorovich-Rubinstein objective as a transport solver.

    Relies on the endpoint-last ordering of
    :class:`~graph_geometry.curvature.transport.KernelPairMeasureBuilder`: the
    last support entry of each measure is the endpoint itself. Returns the
    objective with ``plan=None``; this is not an ordinary non-negative
    Wasserstein distance, so no plan is reported.
    """

    solver: str | None = None

    def __post_init__(self):
        self._lp = LLYOTSolver(self.solver)
        self.solver_name = self._lp.solver_name

    def for_cost_scale(self, scale: float) -> SignedKRTransportSolver:
        """The signed KR objective is homogeneous in the cost matrix."""
        return self

    def __call__(self, source_mass, target_mass, cost_matrix, *, return_plan=False):
        return TransportResult(cost=self._lp(source_mass, target_mass, cost_matrix))


@dataclass
class EMDOTSolver:
    """Standard Earth Mover's Distance via the POT library."""

    def for_cost_scale(self, scale: float) -> EMDOTSolver:
        """Opt the legacy exact EMD solver into cost conditioning."""
        return self

    def __call__(self, x, y, d) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        d = np.asarray(d, dtype=float)
        return ot.emd2(x, y, d)


# ── curvature from OT ────────────────────────────────────────────────────────


@dataclass
class LLYCurvature:
    """Lin-Lu-Yau: kappa = m / d(x, y)."""

    def __call__(self, m: float, d: float) -> float:
        return m / d


@dataclass
class OllivierCurvature:
    """Ollivier: kappa = 1 - m / d(x, y)."""

    def __call__(self, m: float, d: float) -> float:
        return 1 - m / d


# ── presets ──────────────────────────────────────────────────────────────────


def lin_lu_yau(
    kernel: str = "auto", beta: BetaSpec = 0.8, solver: str | None = None
) -> CurvatureConfig:
    """Lin-Lu-Yau Ricci curvature (Lin-Lu-Yau 2011), via signed KR OT (CVXPY).

    ``kernel``: ``"auto"`` | ``"undirected"`` | ``"out"`` | ``"in"`` | ``"mixed"``.
    ``beta``: mixing parameter for ``"mixed"``/``"auto"`` on directed graphs
    (float | dict | ``"degree_proportional"`` | callable).
    ``solver``: CVXPY solver name, or ``None`` for the first installed entry of
    :data:`LLY_SOLVER_PREFERENCE`. An unavailable name raises rather than
    silently falling back.
    """
    ot_solver = LLYOTSolver(solver)
    return CurvatureConfig(
        name="Lin-Lu-Yau",
        transition_kernel=resolve_kernel(kernel, beta),
        distribution=LLYDistribution(),
        ot_solver=ot_solver,
        curvature_from_ot=LLYCurvature(),
    )


def lin_lu_yau_ot_config(
    kernel: str = "auto", beta: BetaSpec = 0.8, solver: str | None = None
) -> OTCurvatureConfig:
    """Endpoint-pair configuration of Lin-Lu-Yau curvature on the shared engine."""
    return OTCurvatureConfig(
        name="Lin-Lu-Yau",
        kernel=resolve_kernel(kernel, beta),
        distribution=LLYDistribution(),
        solver=SignedKRTransportSolver(solver),
        curvature=LLYCurvature(),
    )


def ollivier_ot_config(
    alpha: float = 0.5, kernel: str = "auto", beta: BetaSpec = 0.8
) -> OTCurvatureConfig:
    """Endpoint-pair configuration of Ollivier curvature on the shared engine."""
    return OTCurvatureConfig(
        name=f"Ollivier(alpha={alpha})",
        kernel=resolve_kernel(kernel, beta),
        distribution=OllivierDistribution(alpha),
        solver=EMDTransportSolver(),
        curvature=OllivierCurvature(),
    )


def ollivier(
    alpha: float = 0.5, kernel: str = "auto", beta: BetaSpec = 0.8
) -> CurvatureConfig:
    """Ollivier-Ricci curvature (Ollivier 2009), via EMD (POT).

    ``alpha``: self-loop mass in [0, 1]. ``kernel``/``beta`` as in
    :func:`lin_lu_yau`.
    """
    return CurvatureConfig(
        name=f"Ollivier(alpha={alpha})",
        transition_kernel=resolve_kernel(kernel, beta),
        distribution=OllivierDistribution(alpha),
        ot_solver=EMDOTSolver(),
        curvature_from_ot=OllivierCurvature(),
    )
