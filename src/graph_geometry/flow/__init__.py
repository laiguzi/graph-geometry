"""Flow layer: pluggable Ricci flow equations, surgery, and the engine.

- ``equations`` -- normalized / unnormalized / additive / callable + ``resolve_flow``
- ``surgery``   -- no_surgery / surgery / surgery_n + strategies + ``resolve_surgery``
- ``engine``    -- :class:`RicciFlow` orchestrator + :class:`FlowResult` (committed states)
- ``trajectories`` -- per-edge series across committed states (plot-free)
"""

from . import equations, surgery
from .engine import TERMINATION_REASONS, FlowDivergenceError, FlowResult, RicciFlow
from .equations import (
    AdditiveFlow,
    ExpressionFlow,
    NormalizedFlow,
    UnnormalizedFlow,
    additive_flow,
    expression_flow,
    latex_to_expr,
    normalized_flow,
    resolve_flow,
    unnormalized_flow,
)
from .surgery import (
    IntervalSurgery,
    NoSurgery,
    resolve_surgery,
)
from .surgery import surgery as surgery_fn
from .surgery import surgery_n
from .trajectories import EdgeTrajectories, extract_edge_trajectories

__all__ = [
    "RicciFlow",
    "FlowResult",
    "FlowDivergenceError",
    "TERMINATION_REASONS",
    "EdgeTrajectories",
    "extract_edge_trajectories",
    "normalized_flow",
    "unnormalized_flow",
    "additive_flow",
    "expression_flow",
    "latex_to_expr",
    "resolve_flow",
    "NormalizedFlow",
    "UnnormalizedFlow",
    "AdditiveFlow",
    "ExpressionFlow",
    "resolve_surgery",
    "surgery_fn",
    "surgery_n",
    "NoSurgery",
    "IntervalSurgery",
    "equations",
    "surgery",
]
