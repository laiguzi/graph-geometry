"""Graph layer: config-driven node/edge factors, generators, and loaders.

- ``config``      -- :class:`GraphConfig` dataclass + :func:`apply_config`
- ``factors``     -- factor strategies: fixed | degree | random | callable
- ``generators``  -- SBM, LFR, cycle, complete, petersen, Gab, ...
- ``loaders``     -- from_edges / from_networkx / from_edgelist / from_gexf
- ``semantics``   -- :class:`GraphSemantics`: how existing attributes are interpreted
"""

from . import factors, generators, loaders
from .config import GraphConfig, apply_config
from .factors import resolve_edge_factor, resolve_node_factor
from .semantics import GraphSemantics

__all__ = [
    "GraphConfig",
    "GraphSemantics",
    "apply_config",
    "resolve_edge_factor",
    "resolve_node_factor",
    "factors",
    "generators",
    "loaders",
]
