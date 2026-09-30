"""Shared callable type aliases for the curvature layer."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

# A node in the graph (int, str, ...).
Node = Any

# Transition kernel: (node, G, neighbors_dict, weight_attr) -> prob array over neighbors[node].
TransitionKernelFn = Callable[[Node, Any, dict, str], np.ndarray]

# Distribution: (kernel_array) -> list of floats (with self-loop mass appended).
DistributionFn = Callable[[np.ndarray], list]

# OT solver: (x, y, cost_matrix) -> transport distance (float).
OTSolverFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]

# Curvature from OT: (ot_distance, edge_distance) -> curvature value.
CurvatureFromOT = Callable[[float, float], float]
