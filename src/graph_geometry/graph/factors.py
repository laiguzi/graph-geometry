"""Factor strategies for edge and node attributes.

A *factor* resolves a numeric value for an edge ``(u, v)`` or a node ``n`` on a
graph ``G``. Four strategy kinds are supported, addressable by name (or given
directly):

``"fixed"``
    Constant ``1.0`` on every edge/node. A bare number (``0.5``) is also
    accepted and means "fixed at that constant".
``"degree"``
    Proportional to degree. For a node: ``G.degree(n)``. For an edge:
    ``G.degree(u) + G.degree(v)``. On directed graphs ``degree`` is the total
    (in + out) degree. 
``"random"``
    Independent ``uniform(0, 1)`` draw per edge/node, from a seedable NumPy
    ``Generator`` (replacing the old global-``np.random`` calls, which were not
    reproducible).
``dict``
    Explicit user-given assignments. Edge factor: keyed by ``(u, v)`` (for
    undirected graphs ``(v, u)`` is also tried). Node factor: keyed by node.
    A missing key raises a clear ``ValueError`` — combine with
    ``apply_config(..., overwrite=False)`` and source-provided attributes if
    only some edges/nodes need explicit values.
callable
    A user function. Edge factor: ``f(u, v, G) -> float``. Node factor:
    ``f(n, G) -> float``.

The resolver functions turn a spec into a plain callable with the signature
above, so downstream code (``apply_config``) never branches on strategy kind.
"""

from __future__ import annotations

from typing import Any, Callable, Union

import numpy as np

__all__ = [
    "FactorSpec",
    "EDGE_STRATEGIES",
    "NODE_STRATEGIES",
    "as_rng",
    "resolve_edge_factor",
    "resolve_node_factor",
]

#: What a factor field may hold: a strategy name, a constant, an explicit
#: assignment dict, or a callable.
FactorSpec = Union[str, float, int, dict, Callable]

EDGE_STRATEGIES = ("fixed", "degree", "random")
NODE_STRATEGIES = ("fixed", "degree", "random")


def as_rng(seed_or_rng: Any = None) -> np.random.Generator:
    """Coerce ``None`` / int / ``Generator`` into a NumPy ``Generator``."""
    if isinstance(seed_or_rng, np.random.Generator):
        return seed_or_rng
    return np.random.default_rng(seed_or_rng)


def resolve_edge_factor(
    spec: FactorSpec, rng: np.random.Generator | None = None
) -> Callable[[Any, Any, Any], float]:
    """Return an edge factor ``f(u, v, G) -> float`` for ``spec``.

    ``rng`` is only consulted by the ``"random"`` strategy; when omitted a fresh
    unseeded generator is used.
    """
    if callable(spec):
        return spec
    if isinstance(spec, bool):  # guard: bool is an int subclass
        raise TypeError(f"Invalid edge factor spec: {spec!r}")
    if isinstance(spec, dict):
        assignments = spec

        def edge_lookup(u, v, G):
            if (u, v) in assignments:
                return float(assignments[(u, v)])
            if not G.is_directed() and (v, u) in assignments:
                return float(assignments[(v, u)])
            raise ValueError(
                f"No explicit value for edge ({u!r}, {v!r}) in the assignment "
                f"dict. Provide it, or rely on source attributes with "
                f"overwrite=False."
            )

        return edge_lookup
    if isinstance(spec, (int, float)):
        value = float(spec)
        return lambda u, v, G: value # can be a constant factor for all edges

    name = str(spec).lower()
    if name == "fixed":
        return lambda u, v, G: 1.0
    if name == "degree":
        return lambda u, v, G: float(G.degree(u) + G.degree(v))
    if name == "random":
        gen = as_rng(rng)
        return lambda u, v, G: float(gen.uniform(0.0, 1.0))
    raise ValueError(
        f"Unsupported edge factor {spec!r}. "
        f"Use one of {EDGE_STRATEGIES}, a number, or a callable(u, v, G)."
    )


def resolve_node_factor(
    spec: FactorSpec, rng: np.random.Generator | None = None
) -> Callable[[Any, Any], float]:
    """Return a node factor ``f(n, G) -> float`` for ``spec``.

    ``rng`` is only consulted by the ``"random"`` strategy; when omitted a fresh
    unseeded generator is used.
    """
    if callable(spec):
        return spec
    if isinstance(spec, bool):  # guard: bool is an int subclass
        raise TypeError(f"Invalid node factor spec: {spec!r}")
    if isinstance(spec, dict):
        assignments = spec

        def node_lookup(n, G):
            if n in assignments:
                return float(assignments[n])
            raise ValueError(
                f"No explicit value for node {n!r} in the assignment dict."
            )

        return node_lookup
    if isinstance(spec, (int, float)):
        value = float(spec)
        return lambda n, G: value

    name = str(spec).lower()
    if name == "fixed":
        return lambda n, G: 1.0
    if name == "degree":
        return lambda n, G: float(G.degree(n))
    if name == "random":
        gen = as_rng(rng)
        return lambda n, G: float(gen.uniform(0.0, 1.0))
    raise ValueError(
        f"Unsupported node factor {spec!r}. "
        f"Use one of {NODE_STRATEGIES}, a number, or a callable(n, G)."
    )
