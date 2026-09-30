"""Ricci flow update equations.

A flow equation is a callable ``(K_edge, q_edge, step, K_avg) -> delta`` giving
the change in the **evolving** edge quantity for one iteration (``q`` is the
weight or the distance, depending on the engine's ``evolve`` setting). Ported
from ``ricciflow_sim/core/flow_equations.py``, implemented as picklable classes
plus a ``resolve_flow`` that also accepts names, callables, and free-form math
expressions (``ExpressionFlow``).
"""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from typing import Callable, Union

__all__ = [
    "FlowEquationFn",
    "FlowSpec",
    "NormalizedFlow",
    "UnnormalizedFlow",
    "AdditiveFlow",
    "ExpressionFlow",
    "normalized_flow",
    "unnormalized_flow",
    "additive_flow",
    "expression_flow",
    "latex_to_expr",
    "resolve_flow",
]

# (K_edge, q_edge, step, K_avg) -> delta   (q = evolving quantity: weight or distance)
FlowEquationFn = Callable[[float, float, float, float], float]
FlowSpec = Union[str, FlowEquationFn]


@dataclass
class NormalizedFlow:
    """dw = -step * (K - K_avg) * w   (volume-preserving; the default)."""

    def __call__(self, K_edge: float, w_edge: float, step: float, K_avg: float) -> float:
        return -step * (K_edge - K_avg) * w_edge


@dataclass
class UnnormalizedFlow:
    """dw = -step * K * w   (standard Ricci flow, no normalization)."""

    def __call__(self, K_edge: float, w_edge: float, step: float, K_avg: float) -> float:
        return -step * K_edge * w_edge


@dataclass
class AdditiveFlow:
    """dw = -step * (K - K_avg)   (additive, not multiplicative)."""

    def __call__(self, K_edge: float, w_edge: float, step: float, K_avg: float) -> float:
        return -step * (K_edge - K_avg)


def normalized_flow() -> NormalizedFlow:
    return NormalizedFlow()


def unnormalized_flow() -> UnnormalizedFlow:
    return UnnormalizedFlow()


def additive_flow() -> AdditiveFlow:
    return AdditiveFlow()


_FLOWS = {
    "normalized": NormalizedFlow,
    "unnormalized": UnnormalizedFlow,
    "additive": AdditiveFlow,
}


# ── free-form expression flows ───────────────────────────────────────────────

# functions the expression may call
_ALLOWED_FUNCS = {
    "exp": math.exp, "log": math.log, "log10": math.log10, "sqrt": math.sqrt,
    "abs": abs, "min": min, "max": max, "tanh": math.tanh, "sin": math.sin,
    "cos": math.cos, "pow": pow, "floor": math.floor, "ceil": math.ceil,
    "sign": lambda x: (x > 0) - (x < 0),
}

# variable names the expression may reference (aliases for the 4 flow inputs)
_ALLOWED_VARS = {
    "K", "kappa", "k",                       # curvature
    "w", "weight", "d", "distance", "q",     # the evolving quantity
    "step", "eta", "s",                      # step size
    "K_avg", "kbar", "kappa_bar", "Ka",      # mean curvature
}

# AST node types allowed in an expression (no attribute/subscript/lambda/etc.)
_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name, ast.Load,
    ast.Call, ast.IfExp, ast.Compare, ast.BoolOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)


def latex_to_expr(text: str) -> str:
    r"""Best-effort convert a pasted LaTeX formula to a Python expression.

    Handles the common tokens (``\eta``, ``\kappa``, ``\bar\kappa``, ``\cdot``,
    ``\frac{a}{b}``, ``^``, ``\left``/``\right``, ``\,``); anything it doesn't
    recognise is passed through, so plain expressions are returned unchanged.
    """
    s = text.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s.strip("$")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = re.sub(r"\\[,;: ]", " ", s)                       # thin spaces
    s = s.replace(r"\cdot", "*").replace(r"\times", "*")
    # \frac{A}{B} -> ((A)/(B))  (one level of nesting)
    frac = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
    for _ in range(3):
        s, n = frac.subn(r"((\1)/(\2))", s)
        if not n:
            break
    # symbols -> variable names (space-padded so adjacent commands separate)
    s = re.sub(r"\\bar\s*\{?\s*\\kappa\s*\}?", " kbar ", s)
    s = re.sub(r"\\overline\s*\{?\s*\\kappa\s*\}?", " kbar ", s)
    s = s.replace(r"\kappa", " kappa ").replace(r"\eta", " eta ")
    s = re.sub(r"\\[a-zA-Z]+", lambda m: " " + m.group(0)[1:] + " ", s)  # other commands
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\s+", " ", s).strip()
    s = _insert_implicit_mult(s)
    return s


_KEYWORDS = {"if", "else", "and", "or", "not"}


_NOT_KW = r"(?!(?:if|else|and|or|not)\b)"  # don't glue onto a Python keyword


def _insert_implicit_mult(s: str) -> str:
    """Insert ``*`` for LaTeX-style implicit multiplication (``2w``, ``)w``, ``eta w``)."""
    s = re.sub(r"\)\s*\(", ")*(", s)                          # )( -> )*(
    s = re.sub(rf"\)\s*{_NOT_KW}([A-Za-z0-9_])", r")*\1", s)   # )x -> )*x
    s = re.sub(rf"(\d)\s*{_NOT_KW}([A-Za-z_(])", r"\1*\2", s)  # 2x, 2( -> 2*x, 2*(
    # adjacency: name/) then space then name/(, unless a keyword is involved
    s = re.sub(
        rf"([A-Za-z_]\w*|\))\s+{_NOT_KW}(?=[A-Za-z_(])",
        lambda m: m.group(1) + (" " if m.group(1) in _KEYWORDS else "*"),
        s,
    )

    def _name_paren(m):
        name = m.group(1)
        return f"{name}(" if name in _ALLOWED_FUNCS else f"{name}*("
    return re.sub(r"([A-Za-z_]\w*)\s*\(", _name_paren, s)   # eta( -> eta*(, exp( kept


def _compile_expr(expr: str):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid flow expression {expr!r}: {e.msg}") from e
    call_funcs = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Disallowed syntax in flow expression: {type(node).__name__}. "
                "Use +-*/, **, comparisons, if/else, and functions like "
                f"{sorted(_ALLOWED_FUNCS)}."
            )
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS
        ):
            raise ValueError("Only these functions may be called: "
                             f"{sorted(_ALLOWED_FUNCS)}.")
        if isinstance(node, ast.Name) and node.id not in call_funcs:
            if node.id not in _ALLOWED_VARS:
                raise ValueError(
                    f"Unknown name {node.id!r} in flow expression. Allowed variables: "
                    f"{sorted(_ALLOWED_VARS)}."
                )
        if isinstance(node, ast.Constant):
            # numeric literals only, bounded
            if not isinstance(node.value, (int, float)) or isinstance(node.value, complex):
                raise ValueError(
                    f"Only numeric constants are allowed, got {node.value!r}."
                )
            if abs(node.value) > 1e6:
                raise ValueError(
                    f"Constant {node.value!r} too large (|c| must be <= 1e6)."
                )
    # evaluate ** in float arithmetic: int bignum pow (e.g. 9**9**9) would hang,
    # float pow raises OverflowError immediately instead
    tree = _FloatPow().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<flow-expression>", "eval")


class _FloatPow(ast.NodeTransformer):
    """Rewrite ``a ** b`` as ``_pow(a, b)`` (float pow, overflow-safe)."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.Call(
                func=ast.Name(id="_pow", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node


def _float_pow(a: float, b: float) -> float:
    return float(a) ** float(b)


@dataclass
class ExpressionFlow:
    r"""A flow equation defined by a math expression string.

    The expression may use the curvature (``kappa`` / ``K`` / ``k``), the
    evolving quantity (``w`` / ``weight`` / ``d`` / ``distance`` / ``q``), the
    step (``step`` / ``eta`` / ``s``) and the mean curvature (``K_avg`` /
    ``kbar`` / ``kappa_bar``), plus the functions in ``_ALLOWED_FUNCS``. Pasted
    LaTeX is normalised via :func:`latex_to_expr` first, so e.g.::

        ExpressionFlow("-eta*(kappa - kbar)*w")          # normalized flow
        ExpressionFlow(r"-\eta\,(\kappa-\bar\kappa)\,w")  # same, as LaTeX
    """

    expr: str

    def __post_init__(self):
        self._normalized = latex_to_expr(self.expr)
        self._code = _compile_expr(self._normalized)

    def __call__(self, K_edge: float, q_edge: float, step: float, K_avg: float) -> float:
        ns = {
            "K": K_edge, "kappa": K_edge, "k": K_edge,
            "w": q_edge, "weight": q_edge, "d": q_edge, "distance": q_edge, "q": q_edge,
            "step": step, "eta": step, "s": step,
            "K_avg": K_avg, "kbar": K_avg, "kappa_bar": K_avg, "Ka": K_avg,
            "_pow": _float_pow,
            **_ALLOWED_FUNCS,
        }
        return float(eval(self._code, {"__builtins__": {}}, ns))  # noqa: S307 (sandboxed)


def expression_flow(expr: str) -> ExpressionFlow:
    """Build a flow equation from a math (or LaTeX) expression. See :class:`ExpressionFlow`."""
    return ExpressionFlow(expr)


def resolve_flow(spec: FlowSpec) -> FlowEquationFn:
    """Coerce a flow spec into a flow-equation callable.

    Accepts a name (``"normalized"`` | ``"unnormalized"`` | ``"additive"``), any
    callable ``(K, q, step, K_avg) -> delta`` (returned as-is), or — for a string
    that isn't a known name — a free-form math/LaTeX expression
    (see :class:`ExpressionFlow`).
    """
    if callable(spec):
        return spec
    if isinstance(spec, str):
        cls = _FLOWS.get(spec.lower())
        if cls is not None:
            return cls()
        return ExpressionFlow(spec)   # treat any other string as an expression
    raise TypeError(f"flow_equation must be a name, callable, or expression; got {spec!r}")
