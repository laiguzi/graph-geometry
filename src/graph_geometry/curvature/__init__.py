"""Curvature layer: registered definitions, resolution and evaluation.

- ``model``      -- capabilities, parameter schemas, requests, resolved records, results
- ``errors``     -- typed errors (configuration / domain / input / numerical / contract)
- ``registry``   -- canonical specs, aliases and capability queries (no computation)
- ``builtins``   -- adapters + complete metadata for the shipped definitions
- ``evaluator``  -- :func:`resolve_curvature`, :func:`compute_curvature`, the result
  contract, and the compatibility wrappers :func:`curvature` / :func:`curvature_pairs`
- ``transport``  -- endpoint-pair measures, ground costs, solvers, :class:`OTCurvatureConfig`
- ``base``       -- :class:`TransportCurvatureEngine` (shared OT orchestration) plus the
  :class:`CurvatureConfig` / :class:`RicciCurvature` compatibility adapters
- ``combinatorial`` -- the template ``A_c - sum B_c + sum C_c`` as a config + engine
- ``kernels``    -- transition kernels (out/in/mixed/undirected/auto) + beta strategies
- ``ot_ricci``   -- low-level OT builders: :func:`lin_lu_yau`, :func:`ollivier`
- ``directed``, ``forman_variants``, ``forman_directed`` -- formula implementations
"""

from . import builtins as _builtins  # noqa: F401  (side effect: registers the built-ins)
from . import kernels, ot_ricci
from .base import (
    CurvatureConfig,
    RicciCurvature,
    TransportCurvatureEngine,
    write_curvature,
    write_node_means,
)
from .combinatorial import (
    CombinatorialContext,
    CombinatorialCurvatureConfig,
    CombinatorialCurvatureEngine,
)
from .directed import (
    directed_curvature_profile,
    directed_in_out_measure,
    eidi_jost,
    eidi_jost_ot_config,
    signed_double_cover,
)
from .errors import (
    CurvatureConfigurationError,
    CurvatureContractError,
    CurvatureDeprecationWarning,
    CurvatureDomainError,
    CurvatureError,
    CurvatureInputError,
    CurvatureNumericalError,
    CurvatureWarning,
    failure_kind,
)
from .evaluator import (
    compute_curvature,
    compute_pair_curvature,
    curvature,
    curvature_pairs,
    evaluate_curvature,
    incident_node_means,
    resolve_curvature,
)
from .forman_directed import (
    augmented_forman_directed,
    augmented_forman_directed_config,
    feed_forward_faces,
    forman_directed,
    forman_directed_config,
)
from .forman_variants import (
    augmented_forman_node_weighted,
    augmented_forman_node_weighted_config,
    augmented_forman_sreejith,
    forman_node_weighted,
    forman_node_weighted_config,
    forman_sreejith,
    resolve_node_weight,
)
from .kernels import (
    BetaConstant,
    BetaDegreeProportional,
    BetaDict,
    BetaNodeAttr,
    BetaWeightProportional,
    check_beta,
    resolve_beta,
    resolve_kernel,
)
from .model import (
    CurvatureCapabilities,
    CurvatureFailure,
    CurvatureRequest,
    CurvatureResult,
    ParameterSpec,
    ResolvedCurvature,
)
from .ot_ricci import lin_lu_yau, lin_lu_yau_ot_config, ollivier, ollivier_ot_config
from .transport import (
    DirectionalKernel,
    DiscreteMeasure,
    EMDTransportSolver,
    InOutPairMeasureBuilder,
    KernelPairMeasureBuilder,
    NodeKernel,
    NoIdlenessDistribution,
    OTCurvatureConfig,
    ShortestPathGroundCost,
    TransportResult,
)
from .registry import (
    CURVATURE_ALIASES,
    CURVATURE_REGISTRY,
    PAIR_CAPABLE,
    CurvatureAlias,
    CurvatureSpec,
    get_curvature_spec,
    list_curvatures,
    register_curvature,
    register_curvature_alias,
    unregister_curvature,
)

__all__ = [
    # model
    "CurvatureCapabilities",
    "ParameterSpec",
    "CurvatureRequest",
    "ResolvedCurvature",
    "CurvatureResult",
    "CurvatureFailure",
    # errors
    "CurvatureError",
    "CurvatureConfigurationError",
    "CurvatureDomainError",
    "CurvatureInputError",
    "CurvatureNumericalError",
    "CurvatureContractError",
    "CurvatureWarning",
    "CurvatureDeprecationWarning",
    "failure_kind",
    # registry
    "CURVATURE_REGISTRY",
    "CURVATURE_ALIASES",
    "CurvatureSpec",
    "CurvatureAlias",
    "register_curvature",
    "register_curvature_alias",
    "unregister_curvature",
    "get_curvature_spec",
    "list_curvatures",
    "PAIR_CAPABLE",
    # evaluation
    "resolve_curvature",
    "evaluate_curvature",
    "compute_curvature",
    "compute_pair_curvature",
    "incident_node_means",
    "curvature",
    "curvature_pairs",
    # shared transport engine
    "TransportCurvatureEngine",
    "OTCurvatureConfig",
    "DiscreteMeasure",
    "TransportResult",
    "KernelPairMeasureBuilder",
    "NodeKernel",
    "DirectionalKernel",
    "NoIdlenessDistribution",
    "InOutPairMeasureBuilder",
    "ShortestPathGroundCost",
    "EMDTransportSolver",
    "lin_lu_yau_ot_config",
    "ollivier_ot_config",
    "eidi_jost_ot_config",
    # combinatorial template engine
    "CombinatorialCurvatureEngine",
    "CombinatorialCurvatureConfig",
    "CombinatorialContext",
    "forman_directed_config",
    "augmented_forman_directed_config",
    "forman_node_weighted_config",
    "augmented_forman_node_weighted_config",
    # compatibility adapters + low-level builders
    "CurvatureConfig",
    "RicciCurvature",
    "write_curvature",
    "write_node_means",
    "lin_lu_yau",
    "ollivier",
    # raw formulas
    "eidi_jost",
    "forman_directed",
    "augmented_forman_directed",
    "feed_forward_faces",
    "forman_node_weighted",
    "augmented_forman_node_weighted",
    "forman_sreejith",
    "augmented_forman_sreejith",
    "resolve_node_weight",
    "directed_curvature_profile",
    "directed_in_out_measure",
    "signed_double_cover",
    # kernels / beta
    "resolve_kernel",
    "resolve_beta",
    "check_beta",
    "BetaConstant",
    "BetaDegreeProportional",
    "BetaWeightProportional",
    "BetaDict",
    "BetaNodeAttr",
    "kernels",
    "ot_ricci",
]
