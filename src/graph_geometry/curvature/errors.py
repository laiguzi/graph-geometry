"""Typed curvature errors and warnings.

Every failure the curvature layer *expects* is one of five kinds, so a caller
-- and a comparison recording failures -- can tell a misconfigured request from
a definition that does not apply to this graph, from bad input data, from a
solver that failed, from a plugin that returned something malformed::

    CurvatureError
    |-- CurvatureConfigurationError  # name, parameter or contradictory semantics
    |-- CurvatureDomainError         # graph kind, scope, loops or reachability
    |-- CurvatureInputError          # invalid/missing weight or distance values
    |-- CurvatureNumericalError      # declared solver/convergence failure
    `-- CurvatureContractError       # malformed method/plugin result

The four non-numerical classes also derive from :class:`ValueError`, and the
numerical one from :class:`RuntimeError`, so code written against the earlier
untyped errors keeps catching them. An unexpected programming exception is
never relabelled as one of these: it propagates with its own type and traceback.
"""

from __future__ import annotations

__all__ = [
    "CurvatureError",
    "CurvatureConfigurationError",
    "CurvatureDomainError",
    "CurvatureInputError",
    "CurvatureNumericalError",
    "CurvatureContractError",
    "CurvatureWarning",
    "CurvatureDeprecationWarning",
    "failure_kind",
]


class CurvatureError(Exception):
    """Base class of every expected curvature failure."""


class CurvatureConfigurationError(CurvatureError, ValueError):
    """Unknown method, invalid parameter, or contradictory semantics."""


class CurvatureDomainError(CurvatureError, ValueError):
    """The definition does not apply: graph kind, scope, loops or reachability."""


class CurvatureInputError(CurvatureError, ValueError):
    """An edge or node value the definition consumes is missing or invalid."""


class CurvatureNumericalError(CurvatureError, RuntimeError):
    """A declared solver or convergence failure."""


class CurvatureContractError(CurvatureError, ValueError):
    """A method or plugin returned a result violating the result contract."""


class CurvatureWarning(UserWarning):
    """A run is valid but carries a semantic caveat worth recording."""


class CurvatureDeprecationWarning(FutureWarning):
    """A compatibility name or call form scheduled for removal.

    A :class:`FutureWarning` rather than a :class:`DeprecationWarning` so that it
    is shown by default: the audience is researchers running scripts, not only
    library developers running test suites.
    """


_KINDS = (
    (CurvatureConfigurationError, "configuration"),
    (CurvatureDomainError, "inapplicable"),
    (CurvatureInputError, "invalid_input"),
    (CurvatureNumericalError, "numerical_failure"),
    (CurvatureContractError, "contract_failure"),
)


def failure_kind(exc: BaseException) -> str:
    """Map an exception to one of the six recorded failure kinds.

    Anything that is not a typed curvature error is ``internal_failure``.
    """
    for cls, kind in _KINDS:
        if isinstance(exc, cls):
            return kind
    return "internal_failure"
