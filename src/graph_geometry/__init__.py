"""graph_geometry — discrete graph curvatures and curvature flows.

One canonical, installable package for computing, comparing and evolving
discrete curvatures on graphs, with:

- configurable **node factors** and **edge factors** (weights, distances),
  interpreted through explicit :class:`GraphSemantics`,
- **registered curvature definitions** (Lin-Lu-Yau, Ollivier, Eidi-Jost, the
  four Forman definitions ``forman_node_weighted``,
  ``augmented_forman_node_weighted``, ``forman_directed`` and
  ``augmented_forman_directed``, plugins) with machine-readable capabilities
  and parameter schemas,
- **pluggable flow equations** (normalized / unnormalized / additive / custom).

Example
-------
    import graph_geometry as gg

    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0)], directed=True)
    G = gg.apply_config(G, edge_weight="fixed", node_factor="degree", seed=0)
    request = gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed", "beta": 0.8})
    result = gg.compute_curvature(G, request)
    result.values, result.resolved.to_dict()

    flow = gg.RicciFlow(G, curvature=request, flow_equation="normalized")
    snapshots, convergence = flow.run(iterations=50, step=0.05)
"""

from .curvature import (
    CURVATURE_ALIASES,
    CURVATURE_REGISTRY,
    PAIR_CAPABLE,
    CurvatureCapabilities,
    CurvatureConfig,
    CurvatureConfigurationError,
    CurvatureContractError,
    CurvatureDeprecationWarning,
    CurvatureDomainError,
    CurvatureError,
    CurvatureFailure,
    CurvatureInputError,
    CurvatureNumericalError,
    CurvatureRequest,
    CurvatureResult,
    CurvatureSpec,
    CurvatureWarning,
    ParameterSpec,
    ResolvedCurvature,
    RicciCurvature,
    compute_curvature,
    compute_pair_curvature,
    curvature,
    curvature_pairs,
    evaluate_curvature,
    get_curvature_spec,
    lin_lu_yau,
    list_curvatures,
    ollivier,
    register_curvature,
    register_curvature_alias,
    resolve_curvature,
    signed_double_cover,
    unregister_curvature,
)
from .compare import (
    CurvatureComparison,
    CurvatureRun,
    FlowComparison,
    FlowRun,
    compare_curvature,
    compare_flow,
)
from .experiment import run_experiment
from .flow import (
    EdgeTrajectories,
    RicciFlow,
    additive_flow,
    expression_flow,
    extract_edge_trajectories,
    normalized_flow,
    resolve_surgery,
    unnormalized_flow,
)
from .graph import generators as generate
from .graph import loaders as load
from .graph.config import GraphConfig, apply_config
from .graph.semantics import GraphSemantics
from .io import SnapshotSeries
from .simulator import RicciFlowSimulator

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "GraphConfig",
    "GraphSemantics",
    "apply_config",
    "generate",
    "load",
    # curvature: requests, resolution, evaluation
    "CurvatureRequest",
    "ResolvedCurvature",
    "CurvatureResult",
    "CurvatureFailure",
    "CurvatureCapabilities",
    "ParameterSpec",
    "CurvatureSpec",
    "resolve_curvature",
    "evaluate_curvature",
    "compute_curvature",
    "compute_pair_curvature",
    "list_curvatures",
    "get_curvature_spec",
    "register_curvature",
    "register_curvature_alias",
    "unregister_curvature",
    "CURVATURE_REGISTRY",
    "CURVATURE_ALIASES",
    # typed errors and warnings
    "CurvatureError",
    "CurvatureConfigurationError",
    "CurvatureDomainError",
    "CurvatureInputError",
    "CurvatureNumericalError",
    "CurvatureContractError",
    "CurvatureWarning",
    "CurvatureDeprecationWarning",
    # compatibility wrappers and low-level builders
    "curvature",
    "curvature_pairs",
    "PAIR_CAPABLE",
    "CurvatureConfig",
    "RicciCurvature",
    "signed_double_cover",
    "lin_lu_yau",
    "ollivier",
    "RicciFlow",
    "normalized_flow",
    "unnormalized_flow",
    "additive_flow",
    "expression_flow",
    "resolve_surgery",
    "extract_edge_trajectories",
    "EdgeTrajectories",
    "RicciFlowSimulator",
    "run_experiment",
    "compare_curvature",
    "compare_flow",
    "CurvatureComparison",
    "CurvatureRun",
    "FlowComparison",
    "FlowRun",
    "SnapshotSeries",
]
