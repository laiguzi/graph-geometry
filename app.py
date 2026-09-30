"""
Graph Ricci Flow Simulator — Streamlit app.

Run with:
    uv run --extra app streamlit run app.py
"""

import csv
import io
import os
import random
import tempfile

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np

from graph_geometry import (
    CurvatureRequest,
    GraphSemantics,
    RicciFlowSimulator,
    compare_curvature,
    compare_flow,
    compute_curvature,
    compute_pair_curvature,
    get_curvature_spec,
    list_curvatures,
    unregister_curvature,
    normalized_flow, unnormalized_flow, additive_flow,
    expression_flow, extract_edge_trajectories, load, viz,
)
from graph_geometry.io import save_edge_trajectories_csv
from graph_geometry.flow.equations import latex_to_expr
import importlib.util
import re as _re
import sys


# One operand of a division: a symbol, number, function call, or parenthesized group.
_LATEX_ATOM = (
    r"(?:\\bar\{\\kappa\}|\\mathrm\{[A-Za-z0-9]+\}\([^()]*\)|\\[A-Za-z]+"
    r"|\([^()]*\)|[0-9.]+|[A-Za-z_]\w*)(?:\^\{[^{}]*\})?"
)


def _expr_to_latex(expr: str) -> str:
    """Light, display-only conversion of a flow expression to LaTeX."""
    s = expr
    s = _re.sub(r"\*\*\s*(\([^()]*\)|[A-Za-z_]\w*|[0-9]+\.?[0-9]*)", r"^{\1}", s)
    s = _re.sub(r"\babs\s*\(([^()]*)\)", r"\\left|\1\\right|", s)
    for fn in ("exp", "log10", "log", "sqrt", "tanh", "sin", "cos",
               "min", "max", "sign", "floor", "ceil"):
        s = _re.sub(rf"\b{fn}\b", rf"\\mathrm{{{fn}}}", s)
    s = _re.sub(r"\bkappa_bar\b|\bK_avg\b|\bkbar\b|\bKa\b", r"\\bar{\\kappa}", s)
    s = _re.sub(r"(?<!\\)\b(?:kappa|K|k)\b", r"\\kappa", s)
    s = _re.sub(r"\bstep\b|\beta\b|\bs\b", r"\\eta", s)
    s = _re.sub(rf"({_LATEX_ATOM})\s*/\s*({_LATEX_ATOM})", r"\\frac{\1}{\2}", s)
    s = s.replace("*", r" \cdot ")
    return s


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graph Geometry",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.1rem; }
.stTabs [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)


# ── Built-in graph library ────────────────────────────────────────────────────
def _two_cliques(n1, n2, directed=True):
    if directed:
        G1 = nx.complete_graph(n1).to_directed()
        G2 = nx.relabel_nodes(nx.complete_graph(n2).to_directed(),
                              {i: i + n1 for i in range(n2)})
        G = nx.DiGraph(nx.compose(G1, G2))
    else:
        G1 = nx.complete_graph(n1)
        G2 = nx.relabel_nodes(nx.complete_graph(n2),
                              {i: i + n1 for i in range(n2)})
        G = nx.compose(G1, G2)
    G.add_edge(n1 - 1, n1)
    if directed:
        G.add_edge(n1, n1 - 1)
    return G


BUILTIN_GRAPHS = {
    # ── Directed ──
    "🔀 Bidirectional 3-Cycle":   (lambda: nx.DiGraph([(0,1),(1,2),(2,0),(1,0),(2,1),(0,2)]),
                                   "Directed · 3 nodes, 6 edges. Good for quick tests."),
    "🔀 Bidirectional 4-Cycle":   (lambda: nx.DiGraph([(0,1),(1,2),(2,3),(3,0),(1,0),(2,1),(3,2),(0,3)]),
                                   "Directed · square cycle, fully bidirectional."),
    "🔀 Bidirectional 6-Cycle":   (lambda: nx.cycle_graph(6).to_directed(),
                                   "Directed · hexagonal cycle, fully bidirectional."),
    "🔀 Complete K₄ (directed)":  (lambda: nx.complete_graph(4).to_directed(),
                                   "Directed · every pair connected in both directions."),
    "🔀 Karate Club (directed)":  (lambda: nx.karate_club_graph().to_directed(),
                                   "Directed · Zachary's karate club — 34 nodes, 156 edges."),
    "🔀 Two K₄ + bridge (dir.)":  (lambda: _two_cliques(4, 4, directed=True),
                                   "Directed · two K₄ cliques joined by a bridge. Good for surgery."),
    "🔀 SBM 2×8 (directed)":     (lambda: nx.stochastic_block_model(
                                       [8, 8], [[0.9, 0.1], [0.1, 0.9]], seed=42).to_directed(),
                                   "Directed · two communities of 8 nodes."),
    # ── Undirected ──
    "── 3-Cycle":                 (lambda: nx.cycle_graph(3),
                                   "Undirected · simple triangle."),
    "── 6-Cycle":                 (lambda: nx.cycle_graph(6),
                                   "Undirected · hexagonal cycle."),
    "── Complete K₄":             (lambda: nx.complete_graph(4),
                                   "Undirected · 4 nodes, 6 edges."),
    "── Complete K₅":             (lambda: nx.complete_graph(5),
                                   "Undirected · 5 nodes, 10 edges."),
    "── Petersen":                (lambda: nx.petersen_graph(),
                                   "Undirected · classic Petersen graph, 10 nodes, 15 edges."),
    "── Karate Club":             (lambda: nx.karate_club_graph(),
                                   "Undirected · Zachary's karate club — 34 nodes, 78 edges."),
    "── Two K₄ + bridge":         (lambda: _two_cliques(4, 4, directed=False),
                                   "Undirected · two K₄ cliques joined by a bridge."),
    "── SBM 2×8":                 (lambda: nx.stochastic_block_model(
                                       [8, 8], [[0.9, 0.1], [0.1, 0.9]], seed=42),
                                   "Undirected · two communities of 8 nodes."),
}


# ── Helpers ───────────────────────────────────────────────────────────────────
_EDGE_COLUMNS = ("weight", "distance")


def _detect_delimiter(text: str):
    """``"\t"`` for a tab-separated block, else ``None`` (any whitespace).

    Whitespace splitting cannot represent a node id containing a space. A real
    file hit this: an E. coli regulatory network whose target ``Phantom Gene``
    became two tokens, so the third one was handed to the loader as a weight and
    failed with ``could not convert 'Gene' to float``.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "\t" in line:
            return "\t"
    return None


def _fields(line: str, delimiter=None):
    """Split one row, dropping the empty fields a trailing delimiter leaves."""
    if delimiter is None:
        return line.split()
    return [f for f in line.strip().split(delimiter) if f != ""]


def _edge_data_lines(text: str, delimiter=None):
    """Yield non-comment edge rows, excluding ``N node value`` declarations."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = _fields(stripped, delimiter)
        if parts and parts[0] != "N":
            yield stripped


def _pad_edge_lines(text: str, n_data: int, delimiter=None) -> str:
    """Pad short edge rows with unit values for multi-column parsing."""
    join = delimiter if delimiter else " "
    output = []
    for line in text.splitlines():
        stripped = line.strip()
        parts = _fields(stripped, delimiter) if stripped else []
        if not stripped or stripped.startswith("#") or (parts and parts[0] == "N"):
            output.append(line)
            continue
        parts += ["1.0"] * (n_data - (len(parts) - 2))
        output.append(join.join(parts))
    return "\n".join(output)


def _has_node_lines(text: str, delimiter=None) -> bool:
    """Return whether the edge list includes an ``N node value`` declaration."""
    return any(
        (parts := _fields(stripped, delimiter)) and parts[0] == "N"
        for line in text.splitlines()
        if (stripped := line.strip()) and not stripped.startswith("#")
    )


def _numeric_extra_columns(text: str, delimiter=None) -> bool:
    """Whether every field past the first two parses as a float."""
    for line in _edge_data_lines(text, delimiter):
        for value in _fields(line, delimiter)[2:]:
            try:
                float(value)
            except ValueError:
                return False
    return True


def _drop_extra_columns(text: str, delimiter=None) -> str:
    join = delimiter if delimiter else " "
    output = []
    for line in text.splitlines():
        stripped = line.strip()
        parts = _fields(stripped, delimiter) if stripped else []
        if not stripped or stripped.startswith("#") or (parts and parts[0] == "N"):
            output.append(line)
            continue
        output.append(join.join(parts[:2]))
    return "\n".join(output)


def graph_from_edge_text(text: str, directed: bool, node_attr: str = "weight"):
    """Parse UI edge text through the package loader.

    Supports ``u v``, ``u v weight``, ``u v weight distance`` and
    ``N node value`` rows, tab- or whitespace-separated. Short rows in a
    multi-column block receive unit defaults, matching the simulator's
    structure-only defaults.

    Columns past the first two are edge data only when they are *all* numeric.
    A trailing annotation column (a curated network marking uncertain edges with
    ``?``, say) is dropped rather than mistaken for a weight, and the graph
    carries a note about it in ``G.graph["load_warnings"]``.
    """
    delimiter = _detect_delimiter(text)
    rows = list(_edge_data_lines(text, delimiter))
    n_data = max((len(_fields(line, delimiter)) - 2 for line in rows), default=-1)
    if n_data < 0:
        raise ValueError("No valid edges found.")

    notes = []
    if n_data >= 1 and not _numeric_extra_columns(text, delimiter):
        examples = [
            line for line in rows if len(_fields(line, delimiter)) > 2
        ][:1]
        notes.append(
            f"Ignored {sum(1 for line in rows if len(_fields(line, delimiter)) > 2)} "
            f"non-numeric extra column(s) — treated the file as a plain edge list. "
            f"First such row: {examples[0]!r}" if examples else "Ignored extra columns."
        )
        text = _drop_extra_columns(text, delimiter)
        n_data = 0

    kwargs = {}
    payload = text
    if n_data == 1:
        kwargs["weighted"] = True
    elif n_data >= 2:
        kwargs["columns"] = _EDGE_COLUMNS[:n_data] + tuple(
            f"col{i}" for i in range(2, n_data)
        )
        payload = _pad_edge_lines(text, n_data, delimiter)

    with tempfile.NamedTemporaryFile("w", suffix=".edgelist", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = tmp.name
    try:
        G = load.from_edgelist(
            tmp_path, directed=directed, node_attr=node_attr,
            delimiter=delimiter, **kwargs
        )
    finally:
        os.unlink(tmp_path)
    if notes:
        G.graph["load_warnings"] = notes
    return G


def graph_connectivity_info(G):
    """Return (info_str, warning_str or None).

    Nothing downstream selects a component, so the warning says what each kind
    of definition does with the whole graph instead of promising a restriction.
    """
    n, e = G.number_of_nodes(), G.number_of_edges()
    gt = "directed" if G.is_directed() else "undirected"
    if n == 0:
        return f"**0** nodes · **0** edges · {gt}", "⚠️ The graph is empty."
    if G.is_directed():
        if nx.is_strongly_connected(G):
            return f"**{n}** nodes · **{e}** edges · {gt} · ✅ strongly connected", None
        largest = max(len(c) for c in nx.strongly_connected_components(G))
        return (f"**{n}** nodes · **{e}** edges · {gt}",
                f"⚠️ Not strongly connected (largest strongly connected component: "
                f"{largest} of {n} nodes). The whole graph is used; no component is "
                f"selected for you. Lin-Lu-Yau and Ollivier are undefined on an edge "
                f"whose two neighbourhoods cannot reach each other; Eidi-Jost and the "
                f"Forman curvatures are defined on any digraph.")
    else:
        if nx.is_connected(G):
            return f"**{n}** nodes · **{e}** edges · {gt} · ✅ connected", None
        components = [len(c) for c in nx.connected_components(G)]
        return (f"**{n}** nodes · **{e}** edges · {gt}",
                f"⚠️ Not connected ({len(components)} components; the largest has "
                f"{max(components)} of {n} nodes). The whole graph is used; no "
                f"component is selected for you. An edge's curvature involves only "
                f"its own component, and a node pair split across components has none.")


def prepare_G(G):
    """Ensure weight and distance attributes exist on all edges."""
    for u, v in G.edges():
        G[u][v].setdefault('weight', 1.0)
        G[u][v].setdefault('distance', 1.0)
    return G


def _freeze(value):
    """Convert nested UI/config values into a deterministic comparison key."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted((_freeze(v) for v in value), key=repr))
    return repr(value)


def graph_signature(G):
    """Fingerprint the graph state that a displayed result was computed from."""
    if G is None:
        return None
    nodes = tuple(sorted(
        ((repr(n), _freeze(dict(data))) for n, data in G.nodes(data=True)),
        key=repr,
    ))
    edges = tuple(sorted(
        ((repr(u), repr(v), _freeze(dict(data))) for u, v, data in G.edges(data=True)),
        key=repr,
    ))
    return G.is_directed(), nodes, edges


#: Above this many edges the trajectory view does not draw every line by default.
TRAJECTORY_EDGE_LIMIT = 60


# ── Curvature controls: the UI is generated from registry metadata ────────────
#
# No curvature method is named below. Selectors list ``list_curvatures()``
# grouped by tier, and each definition's controls, defaults and ranges come from
# its ParameterSpec schema -- the same schema the evaluator validates against --
# so a registered plugin appears here with its own controls without any change
# to this file. Compute signatures are never inspected.

_CPU_COUNT = os.cpu_count() or 1

_TIERS = ("core", "specialized", "experimental")
_TIER_TITLE = {"core": "Core", "specialized": "Specialized", "experimental": "Experimental"}


def curvature_specs(scope: str | None = None) -> list:
    """Registered canonical definitions, grouped Core → Specialized → Experimental."""
    specs = list_curvatures(scope=scope)
    return sorted(specs, key=lambda s: _TIERS.index(s.capabilities.status))


def curvature_names(scope: str | None = None) -> list[str]:
    return [spec.name for spec in curvature_specs(scope)]


def spec_label(spec) -> str:
    return spec.label or spec.name


def name_label(name: str) -> str:
    return spec_label(get_curvature_spec(name))


def applicability(spec, graph_kind: str | None, scope: str = "edges") -> str | None:
    """Why ``spec`` cannot run here, or ``None`` when it can."""
    caps = spec.capabilities
    if graph_kind is not None and graph_kind not in caps.graph_kinds:
        kinds = " or ".join(sorted(caps.graph_kinds))
        return (f"{spec_label(spec)} is defined for {kinds} graphs; this graph is "
                f"{graph_kind}. No conversion is applied implicitly.")
    if scope not in caps.scopes:
        return f"{spec_label(spec)} is defined on edges only."
    return None


def describe_spec(spec) -> str:
    caps = spec.capabilities
    parts = [
        f"**{_TIER_TITLE[caps.status]}** · {caps.family}",
        "graphs: " + ", ".join(sorted(caps.graph_kinds)),
        "scope: " + ", ".join(sorted(caps.scopes)),
    ]
    text = " · ".join(parts)
    if spec.description:
        text += f"  \n{spec.description}"
    if spec.citation:
        text += f"  \n*{spec.citation}*"
    return text


def _loaded_plugins() -> dict:
    return st.session_state.setdefault("plugin_names", {})


def load_plugin(path: str):
    """Execute a user .py file so its @register_curvature decorators run.

    Streamlit re-runs this script on every interaction and the registry rejects
    duplicate names, so the definitions a file registered last time are
    unregistered before it is executed again.
    """
    p = os.path.abspath(os.path.expanduser(path))
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    loaded = _loaded_plugins()
    for name in loaded.get(p, []):
        unregister_curvature(name)
    before = set(curvature_names())
    name = f"gg_plugin_{os.path.splitext(os.path.basename(p))[0]}"
    spec = importlib.util.spec_from_file_location(name, p)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module        # dataclasses/pickling need this registered
    try:
        spec.loader.exec_module(module)
    finally:
        loaded[p] = sorted(set(curvature_names()) - before)
    return module


def _parameter_widget(p, key: str):
    """One control for one ParameterSpec, with the schema's default and bounds."""
    label = p.name
    help_text = p.description or None
    if p.kind == "choice":
        options = list(p.choices)
        return st.selectbox(label, options, index=options.index(p.default), key=key,
                            format_func=lambda v: "None (auto-select)" if v is None else str(v),
                            help=help_text)
    if p.kind == "boolean":
        return st.checkbox(label, value=p.default, key=key, help=help_text)
    if p.kind == "string":
        return st.text_input(label, value=p.default, key=key, help=help_text)
    if p.kind == "integer":
        kwargs = {}
        if p.minimum is not None:
            kwargs["min_value"] = int(p.minimum if p.minimum_inclusive else p.minimum + 1)
        if p.maximum is not None:
            kwargs["max_value"] = int(p.maximum if p.maximum_inclusive else p.maximum - 1)
        return int(st.number_input(label, value=int(p.default), step=1, key=key,
                                   help=help_text, **kwargs))
    # float
    lo, hi = p.minimum, p.maximum
    if lo is not None and hi is not None and p.minimum_inclusive and p.maximum_inclusive:
        return float(st.slider(label, float(lo), float(hi), float(p.default),
                               step=(hi - lo) / 1000, format="%.3f", key=key,
                               help=help_text))
    kwargs = {}
    if lo is not None:
        kwargs["min_value"] = float(lo) if p.minimum_inclusive else float(lo) + 1e-9
    if hi is not None:
        kwargs["max_value"] = float(hi) if p.maximum_inclusive else float(hi) - 1e-9
    return float(st.number_input(label, value=float(p.default), key=key,
                                 format="%.6g", help=help_text, **kwargs))


def curvature_param_widgets(spec, graph_kind: str | None, semantics=None,
                            key_prefix: str = "cp") -> dict:
    """Controls for the parameters that are active in the current configuration.

    Parameters are rendered in schema order, so an activating parameter (e.g.
    ``kernel``) is chosen before the ones it activates (``beta``). A parameter
    whose activation cannot be decided yet because the resolver would replace an
    ``auto`` value is shown, since the evaluator decides after resolution.
    """
    params: dict = {}
    kind = graph_kind or "undirected"
    for p in spec.parameters:
        if kind not in p.graph_kinds:
            continue
        if semantics is not None and any(
            getattr(semantics, field) is not None for field in p.inactive_when_semantics
        ):
            continue
        blocked = False
        for other, allowed in p.active_when.items():
            value = params.get(other, spec.parameter(other).default if spec.parameter(other) else None)
            if value == "auto":
                continue
            if value not in allowed:
                blocked = True
        if blocked:
            continue
        params[p.name] = _parameter_widget(p, f"{key_prefix}_{spec.name}_{p.name}")
    return params


# ── Session state ─────────────────────────────────────────────────────────────
_defaults = {
    'sim': None,
    'pos': None,
    'done': False,
    'cmp': None,
    'cur_values': None,
    'cur_graph': None,
    'cur_scope_done': None,
    'cur_resolved': None,
    'cur_signature': None,
    'graph_signature': None,
    'sim_signature': None,
    'cmp_signature': None,
    'cmp_kind': None,
    'fcmp': None,
    'fcmp_signature': None,
}
for key, val in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📐 Ricci Flow")
    st.caption("Pluggable Ricci curvature and Ricci flow on undirected / directed graphs")
    st.divider()

    # ── 1. Graph ──────────────────────────────────────────────────────────────
    st.subheader("1 · Graph")
    source = st.radio("Source", ["Built-in", "Enter edges", "Upload file"],
                      key="graph_source",
                      horizontal=True, label_visibility="collapsed")

    G_input    = None
    input_error = None

    if source == "Built-in":
        name = st.selectbox("Graph", list(BUILTIN_GRAPHS.keys()),
                            label_visibility="collapsed")
        builder, desc = BUILTIN_GRAPHS[name]
        st.caption(desc)
        G_input = prepare_G(builder())

    elif source == "Enter edges":
        st.caption(
            "One row per edge: `u v`, `u v weight`, or `u v weight distance`. "
            "Use `N node value` to attach a node value."
        )
        default_txt = (
            "# Bidirectional 3-cycle\n"
            "0 1\n1 2\n2 0\n"
            "1 0\n2 1\n0 2"
        )
        edge_text = st.text_area("Edges", value=default_txt, height=160,
                                 key="edge_text",
                                 label_visibility="collapsed")
        directed = st.checkbox("Directed", value=True)
        node_attr = "weight"
        if _has_node_lines(edge_text):
            node_attr = st.text_input(
                "Node attribute", value="weight",
                help="Attribute name used by `N node value` rows.",
            )
        try:
            G_input = prepare_G(
                graph_from_edge_text(edge_text, directed, node_attr=node_attr)
            )
            for note in G_input.graph.get("load_warnings", []):
                st.warning(note)
        except Exception as ex:
            input_error = str(ex)

    else:  # Upload
        st.caption("Supported: `.edgelist` · `.gexf`")
        uploaded = st.file_uploader("File", type=["edgelist", "gexf", "txt"],
                                    label_visibility="collapsed")
        upload_directed = st.checkbox("Directed", value=True, key="upload_directed")
        if uploaded:
            suffix = os.path.splitext(uploaded.name)[1].lower()
            try:
                if suffix == ".gexf":
                    tmp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False
                        ) as tmp:
                            tmp.write(uploaded.getvalue())
                            tmp_path = tmp.name
                        G_input = load.from_gexf(tmp_path)
                    finally:
                        if tmp_path and os.path.exists(tmp_path):
                            os.unlink(tmp_path)
                else:
                    text = uploaded.getvalue().decode("utf-8-sig")
                    G_input = graph_from_edge_text(text, upload_directed)
                G_input = prepare_G(G_input)
                for note in G_input.graph.get("load_warnings", []):
                    st.warning(note)
            except Exception as ex:
                input_error = f"Could not load file: {ex}"
        else:
            input_error = "No file uploaded."

    # Graph status badge
    if input_error:
        st.error(input_error)
    elif G_input is not None:
        info, warn = graph_connectivity_info(G_input)
        st.info(info)
        if warn:
            st.warning(warn)

    st.divider()

    # ── 2. Parameters ─────────────────────────────────────────────────────────
    st.subheader("2 · Parameters")

    c1, c2 = st.columns(2)
    with c1:
        iterations = st.number_input("Iterations", 1, 2000, 50, step=5)
    with c2:
        step = st.number_input("Step", 0.001, 1.0, 0.05,
                               step=0.005, format="%.3f")

    delta = st.number_input("Convergence δ", 1e-10, 1.0, 1e-5,
                             format="%.2e",
                             help="Stop early when RC(max)−RC(min) < δ")

    flow_proc = st.number_input(
        "Processes", 1, max(1, _CPU_COUNT), 1, key="flow_proc",
        help=f"Parallel workers for each curvature solve ({_CPU_COUNT} cores here).",
    )

    is_directed_input = G_input is not None and G_input.is_directed()
    input_kind = None if G_input is None else ("directed" if is_directed_input else "undirected")

    # ── Curvature: the selector IS the registry ──────────────────────────────
    with st.expander("➕ Load a curvature plugin", expanded=False):
        st.caption(
            "Path to a `.py` file that calls `@gg.register_curvature`. It is "
            "executed, so only point this at files you trust. "
            "Try `examples/plugin_balanced_forman.py`."
        )
        plugin_path = st.text_input("Plugin file", key="plugin_path",
                                    label_visibility="collapsed")
        if plugin_path:
            try:
                load_plugin(plugin_path)
                st.success(f"Registered: {', '.join(curvature_names())}")
            except Exception as ex:  # noqa: BLE001 - surface any load error
                st.error(f"Could not load plugin: {ex}")

    specs = curvature_specs()
    spec = st.selectbox(
        "Curvature", specs, format_func=spec_label,
        help=("Every registered definition, grouped Core → Specialized → "
              "Experimental, including plugins you load above."),
    )
    st.caption(describe_spec(spec))
    curvature_method = spec.name
    flow_not_applicable = applicability(spec, input_kind)
    if flow_not_applicable:
        st.warning(flow_not_applicable)
    curvature_params = curvature_param_widgets(spec, input_kind)
    kernel_str = curvature_params.get("kernel", "auto")

    st.divider()

    # ── 2b. Flow mode: what evolves, and is the metric coupled to it? ───────
    flow_mode = st.selectbox(
        "Flow mode",
        [
            "Evolve weight, distance fixed",
            "Evolve distance, weight fixed",
            "Coupled: use weight as distance",
        ],
        help=("How the metric relates to the evolving quantity:\n"
              "• Evolve weight, distance fixed — weights evolve, the metric "
              "(distance) stays fixed (Bai-Li-Liu-Lai; the default).\n"
              "• Evolve distance, weight fixed — the metric evolves, weights "
              "fixed (classic metric flow; prefer a multiplicative equation).\n"
              "• Coupled — a single quantity is both the weight and the "
              "distance, so evolving it moves the transition kernel AND the "
              "metric together."),
    )
    # (weight_attr, distance_attr, evolve) for each mode
    _MODES = {
        "Evolve weight, distance fixed":  ("weight", "distance", "weight"),
        "Evolve distance, weight fixed":  ("weight", "distance", "distance"),
        "Coupled: use weight as distance": ("weight", "weight",   "weight"),
    }
    weight_attr, distance_attr, evolve = _MODES[flow_mode]
    ui_semantics = GraphSemantics(weight_attr=weight_attr, distance_attr=distance_attr)
    _q = "d" if evolve == "distance" else "w"

    # ── 2c. Flow equation ──────────────────────────────────────────────────
    flow_label = st.selectbox(
        "Flow equation",
        ["Normalized  (default)", "Unnormalized", "Additive", "Custom (expression)"],
        help=("Normalized: Δ = −s·(κ−κ̄)·q  (volume-preserving)\n"
              "Unnormalized: Δ = −s·κ·q\n"
              "Additive: Δ = −s·(κ−κ̄)\n"
              "Custom: type your own expression (q = the evolving quantity)."),
    )
    _flow_map = {
        "Normalized  (default)": normalized_flow,
        "Unnormalized": unnormalized_flow,
        "Additive": additive_flow,
    }
    if flow_label == "Custom (expression)":
        st.caption(
            f"Δ({_q}) = f(κ, {_q}, η, κ̄).  Variables: `kappa`/`K`, "
            f"`{_q}` (or `w`/`d`/`q`), `step`/`eta`, `kbar`/`K_avg`.  "
            "Functions: exp, log, sqrt, abs, min, max, tanh, sign.  "
            "LaTeX like `-\\eta(\\kappa-\\bar\\kappa)w` is accepted."
        )
        expr = st.text_input(
            "Flow expression",
            value=f"-eta*(kappa - kbar)*{_q}",
            help="Python-style math or pasted LaTeX.",
        )
        try:
            flow_eq = expression_flow(expr)
            flow_eq(0.1, 1.0, 0.05, 0.0)  # validate by test-evaluating
            st.latex(r"\Delta " + f"{_q} = " + _expr_to_latex(latex_to_expr(expr)))
        except Exception as e:  # noqa: BLE001 - surface any parse/eval error to the UI
            st.error(f"Invalid flow expression: {e}")
            st.stop()
    else:
        flow_eq = _flow_map[flow_label]()

    # ── Mathematical definition expander ──
    # Display-only. Known methods get their formula; anything else (including a
    # freshly loaded plugin) falls back to the description it registered with.
    _CURVATURE_LATEX = {
        "lin_lu_yau": r"\kappa(x,y) = \lim_{\alpha\to 1} "
                      r"\frac{\kappa_\alpha(x,y)}{1-\alpha} "
                      r"= \inf_{f \in \mathcal{F}_{xy}} \nabla_{xy} Lf",
        "ollivier": r"\kappa_\alpha(x,y) = 1 - "
                    r"\frac{W_1(\mu_x^\alpha,\, \mu_y^\alpha)}{d(x,y)}",
        "eidi_jost": r"\kappa(x,y) = 1 - "
                     r"\frac{W_1\big(\mu_{x}^{\mathrm{in}},\,"
                     r"\mu_{y}^{\mathrm{out}}\big)}{d(x,y)}",
        "forman_node_weighted": r"\kappa(e) = m_x + m_y - \sum_{z \in \{x,y\}}"
                                r"\sum_{e' \in \delta(z)\setminus e} m_z \sqrt{w_e / w_{e'}}",
        "augmented_forman_node_weighted": r"\kappa(e) = m_x + m_y - \sum_{z \in \{x,y\}}"
                                          r"\sum_{e' \in \delta(z)\setminus(e \cup R_z(e))}"
                                          r" m_z \sqrt{w_e / w_{e'}}"
                                          r" + \sum_{f \ni e} \frac{w_e^2}{\phi}",
        "forman_directed": r"\kappa(x,y) = m_x + m_y - \sum_{e' \in \delta^-(x)} m_x"
                           r"\sqrt{w_e / w_{e'}} - \sum_{e' \in \delta^+(y)} m_y"
                           r"\sqrt{w_e / w_{e'}}",
        "augmented_forman_directed": r"\kappa(x,y) = m_x + m_y"
                                     r" - \sum_{e' \in \delta^-(x)\setminus R(e)} m_x"
                                     r"\sqrt{w_e / w_{e'}}"
                                     r" - \sum_{e' \in \delta^+(y)\setminus R(e)} m_y"
                                     r"\sqrt{w_e / w_{e'}}"
                                     r" + \sum_{t \in \mathcal{F}^{\to}(e)} \frac{w_e^2}{\phi}",
    }
    if G_input is not None and input_error is None:
        with st.expander("Mathematical definition", expanded=False):
            if "kernel" in curvature_params or curvature_method in _CURVATURE_LATEX:
                if is_directed_input:
                    st.markdown("**Directed graph** — curvature is asymmetric: "
                                "κ(x→y) ≠ κ(y→x)")
                    if kernel_str == "out":
                        st.latex(r"P(x,z) = \frac{w_{xz}}{\sum_{u:\, x\to u} w_{xu}}"
                                 r", \quad z \in N^{\mathrm{out}}(x)")
                    elif kernel_str == "in":
                        st.latex(r"P'(x,z) = \frac{w_{zx}}{\sum_{u:\, u\to x} w_{ux}}"
                                 r", \quad z \in N^{\mathrm{in}}(x)")
                    elif "kernel" in curvature_params:
                        st.latex(r"\mathcal{P}(x,z) = \beta(x)\, P(x,z) "
                                 r"+ (1-\beta(x))\, P'(x,z)")
                else:
                    st.markdown("**Undirected graph** — curvature is symmetric: "
                                "κ(x,y) = κ(y,x)")
                    st.latex(r"P(x,z) = \frac{w_{xz}}{\sum_{u \sim x} w_{xu}}")

            if curvature_method in _CURVATURE_LATEX:
                st.latex(_CURVATURE_LATEX[curvature_method])
            else:
                st.markdown(f"**{spec.label}** — {spec.description or 'no description'}")

            if flow_mode == "Evolve weight, distance fixed":
                st.markdown("**Weight** $w$ evolves; **distance** $d$ is fixed.")
            elif flow_mode == "Evolve distance, weight fixed":
                st.markdown("**Distance** $d$ evolves; **weight** $w$ is fixed.")
            else:
                st.markdown("One coupled attribute acts as both $w$ and $d$.")

            # Written in the evolving quantity: w, or d when distance evolves.
            q_next, q_now = f"{_q}_e^{{t+1}}", f"{_q}_e^t"
            if flow_label.startswith("Normalized"):
                st.latex(rf"{q_next} = {q_now} - s \cdot {q_now} "
                         r"\cdot (\kappa_e^t - \bar{\kappa}^t)")
            elif flow_label == "Unnormalized":
                st.latex(rf"{q_next} = {q_now} - s \cdot \kappa_e^t \cdot {q_now}")
            elif flow_label == "Additive":
                st.latex(rf"{q_next} = {q_now} - s "
                         r"\cdot (\kappa_e^t - \bar{\kappa}^t)")
            else:
                st.latex(rf"{q_next} = {q_now} + \Delta {_q}, \quad \Delta {_q} = "
                         + _expr_to_latex(latex_to_expr(expr)))

    st.divider()

    # ── 3. Surgery ────────────────────────────────────────────────────────────
    st.subheader("3 · Surgery  *(optional)*")
    surgery_name = st.selectbox(
        "Type", ["no_surgery", "surgery", "surgery_n"],
        help=("surgery: remove top *fraction* of edges by weight.\n"
              "surgery_n: remove top *N* edges by weight."),
        label_visibility="collapsed",
    )

    surgery_portion  = 0.02
    surgery_interval = 10
    if surgery_name == "surgery":
        surgery_portion  = st.slider("Cut fraction", 0.001, 0.5, 0.02, format="%.3f")
        surgery_interval = st.slider("Every N iterations", 1, 50, 10)
    elif surgery_name == "surgery_n":
        max_cut = G_input.number_of_edges() if G_input is not None else 200
        max_cut = max(1, min(max_cut, 200))
        surgery_portion  = int(st.number_input(
            "Cut N edges", 1, max_cut, 1, step=1,
            help=f"Capped at the graph's edge count ({max_cut})."))
        surgery_interval = st.slider("Every N iterations", 1, 50, 10)

    st.divider()

    can_run = (G_input is not None) and (input_error is None) and not flow_not_applicable
    run_btn = st.button("▶  Run Simulation", type="primary", key="run_flow",
                        width="stretch", disabled=not can_run)


# Bind cached/displayed results to the graph and full simulation configuration.
# Streamlit reruns after every widget change; a signature mismatch means the old
# object no longer describes the controls currently visible on screen.
current_graph_signature = graph_signature(G_input) if input_error is None else None
if st.session_state.graph_signature != current_graph_signature:
    st.session_state.graph_signature = current_graph_signature
    st.session_state.sim = None
    st.session_state.pos = None
    st.session_state.done = False
    st.session_state.sim_signature = None
    st.session_state.cmp = None
    st.session_state.cmp_signature = None
    st.session_state.fcmp = None
    st.session_state.fcmp_signature = None
    # a computed curvature describes the graph it was computed on
    st.session_state.cur_values = None
    st.session_state.cur_graph = None
    st.session_state.cur_scope_done = None
    st.session_state.cur_resolved = None
    st.session_state.cur_signature = None

# The comparison preselects the core definitions applicable to the graph kind.
# A selection made for a digraph means nothing on an undirected graph, so a
# change of kind drops it and the defaults are recomputed for the new kind.
if st.session_state.cmp_kind != input_kind:
    st.session_state.cmp_kind = input_kind
    st.session_state.pop("cmp_methods", None)

flow_expression =expr if flow_label == "Custom (expression)" else flow_label
current_sim_signature = (
    current_graph_signature,
    curvature_method,
    _freeze(curvature_params),
    int(flow_proc),
    flow_mode,
    flow_expression,
    int(iterations),
    float(step),
    float(delta),
    surgery_name,
    _freeze(surgery_portion),
    int(surgery_interval),
)
if (
    st.session_state.sim is not None
    and st.session_state.sim_signature != current_sim_signature
):
    st.session_state.sim = None
    st.session_state.pos = None
    st.session_state.done = False
    st.session_state.sim_signature = None


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("Graph Ricci Flow Simulator")

# ── Compute curvature (independent of the flow) ───────────────────────────────
if G_input is not None and input_error is None:
    with st.expander("📐  Compute curvature on this graph", expanded=False):
        st.caption(
            "Curvature on its own, without running a flow. Optimal-transport "
            "definitions need only two measures and the metric between their "
            "supports, so they are defined on **any pair of nodes**, not just on "
            "edges; the combinatorial definitions are built from an edge's own "
            "incidence structure and exist only on edges."
        )
        c_a, c_b = st.columns([2, 1])
        with c_a:
            cur_name = st.selectbox(
                "Definition",
                curvature_names(),
                format_func=name_label,
                key="cur_name",
            )
        cur_spec = get_curvature_spec(cur_name)
        pair_capable = "pairs" in cur_spec.capabilities.scopes
        with c_b:
            scope = st.radio(
                "Compute on",
                ["Edges", "All node pairs"],
                key="cur_scope",
                help=(
                    "This definition supplies a validated pair construction."
                    if pair_capable else
                    f"{spec_label(cur_spec)} is defined on edges only."
                ),
                disabled=not pair_capable,
            )
        if not pair_capable:
            scope = "Edges"
        st.caption(describe_spec(cur_spec))
        cur_not_applicable = applicability(cur_spec, input_kind)
        if cur_not_applicable:
            st.warning(cur_not_applicable)
        cur_params = curvature_param_widgets(
            cur_spec, input_kind, key_prefix="curv"
        )
        cur_proc = st.number_input(
            "Processes", 1, max(1, _CPU_COUNT), 1, key="cur_proc",
            help=(
                f"Parallel worker processes for the transport solves "
                f"(this machine reports {_CPU_COUNT} cores). Speed-up is "
                "sub-linear: each worker is sent the graph and the all-pairs "
                "distances, and small jobs are dominated by that transfer."
            ),
        )

        n_nodes = G_input.number_of_nodes()
        n_pairs = n_nodes * (n_nodes - 1) // (1 if G_input.is_directed() else 2)
        if scope == "All node pairs":
            st.caption(
                f"{n_pairs:,} pairs versus {G_input.number_of_edges():,} edges — "
                "each pair is a separate transport problem, so this is the "
                "expensive option."
            )
        current_cur_signature = (
            current_graph_signature,
            cur_name,
            scope,
            _freeze(cur_params),
            int(cur_proc),
            weight_attr,
            distance_attr,
        )
        if (
            st.session_state.cur_values is not None
            and st.session_state.cur_signature != current_cur_signature
        ):
            # the displayed values describe other controls than those on screen
            st.session_state.cur_values = None
            st.session_state.cur_graph = None
            st.session_state.cur_scope_done = None
            st.session_state.cur_resolved = None
            st.session_state.cur_signature = None
        if st.button("Compute curvature", key="cur_run",
                     disabled=bool(cur_not_applicable)):
            request = CurvatureRequest(cur_name, cur_params)
            try:
                with st.spinner(f"Computing {scope.lower()}…"):
                    if scope == "Edges":
                        H = G_input.copy()
                        result = compute_curvature(
                            H, request, semantics=ui_semantics,
                            proc=int(cur_proc), annotate=True,
                        )
                        st.session_state.cur_graph = H
                    else:
                        n_all = list(G_input.nodes())
                        pairs = (
                            [(x, y) for x in n_all for y in n_all if x != y]
                            if G_input.is_directed() else
                            [(n_all[i], n_all[j]) for i in range(len(n_all))
                             for j in range(i + 1, len(n_all))]
                        )
                        result = compute_pair_curvature(
                            G_input, request, pairs, semantics=ui_semantics,
                            proc=int(cur_proc),
                        )
                        st.session_state.cur_graph = None
                st.session_state.cur_values = dict(result.values)
                st.session_state.cur_resolved = result.resolved.to_dict()
                st.session_state.cur_scope_done = scope
                st.session_state.cur_signature = current_cur_signature
            except Exception as ex:  # noqa: BLE001 - surface it in the UI
                st.session_state.cur_values = None
                st.session_state.cur_resolved = None
                st.session_state.cur_signature = None
                st.error(f"{type(ex).__name__}: {ex}")

        values = st.session_state.get("cur_values")
        if values:
            done_scope = st.session_state.get("cur_scope_done", "Edges")
            vals = list(values.values())
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("count", f"{len(vals):,}")
            m2.metric("min", f"{min(vals):.4f}")
            m3.metric("max", f"{max(vals):.4f}")
            m4.metric("mean", f"{sum(vals) / len(vals):.4f}")

            left, right = st.columns(2)
            with left:
                cur_graph = st.session_state.get("cur_graph")
                if done_scope == "Edges" and cur_graph is not None:
                    fig, ax = plt.subplots(figsize=(5.2, 4.4))
                    viz.plot_graph(cur_graph, pos=nx.spring_layout(cur_graph, seed=42),
                                   ax=ax, weight_attr=weight_attr, colorbar=True)
                    fig.tight_layout()
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
                else:
                    st.caption(
                        "No graph drawing for all-pairs: most pairs are not "
                        "edges, so there is nothing to colour."
                    )
            with right:
                fig, ax = plt.subplots(figsize=(5.2, 4.4))
                viz.plot_curvature_histogram(vals, ax=ax, color=cur_spec.color or None)
                ax.set_ylabel("edges" if done_scope == "Edges" else "pairs")
                fig.tight_layout()
                st.pyplot(fig, width="stretch")
                plt.close(fig)

            resolved_record = st.session_state.get("cur_resolved")
            if resolved_record:
                st.caption("Resolved configuration (what actually ran)")
                st.json(resolved_record, expanded=False)
            rows = [{"source": u, "target": v, "kappa": k}
                    for (u, v), k in values.items()]
            rows.sort(key=lambda r: r["kappa"])
            st.dataframe(rows, width="stretch", hide_index=True, height=240)

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["source", "target", "kappa"])
            writer.writeheader()
            writer.writerows(rows)
            st.download_button(
                "⬇  Download curvature CSV", data=buf.getvalue(),
                file_name=f"curvature_{cur_name}_{done_scope.replace(' ', '_').lower()}.csv",
                mime="text/csv", key="cur_dl",
            )


# ── Compare curvature definitions (independent of the flow) ───────────────────
if G_input is not None and input_error is None:
    with st.expander("⚖️  Compare curvature definitions on this graph",
                     expanded=False):
        st.caption(
            "Each definition runs on its own copy of the same graph, so the "
            "definition is the only thing that varies. A method that is "
            "undefined here (for example an OT curvature on a digraph that is "
            "not strongly connected) is reported as 0 computable with its "
            "error message; the other methods still run."
        )
        all_names = curvature_names()
        chosen = st.multiselect(
            "Methods",
            all_names,
            default=[spec.name for spec in curvature_specs()
                     if spec.capabilities.status == "core"
                     and applicability(spec, input_kind) is None],
            format_func=name_label,
            key="cmp_methods",
            help=("Core definitions applicable to this graph are preselected. "
                  "Each runs with its schema defaults; an inapplicable choice is "
                  "recorded with its failure kind rather than aborting the rest."),
        )
        cmp_proc = st.number_input(
            "Processes", 1, max(1, _CPU_COUNT), 1, key="cmp_proc",
            help="Parallel workers for each definition's transport solves.",
        )
        current_cmp_signature = (
            current_graph_signature,
            tuple(chosen),
            int(cmp_proc),
            weight_attr,
            distance_attr,
        )
        if (
            st.session_state.cmp is not None
            and st.session_state.cmp_signature != current_cmp_signature
        ):
            st.session_state.cmp = None
            st.session_state.cmp_signature = None
        if st.button("Compare", key="cmp_run", disabled=not chosen):
            with st.spinner(f"Computing {len(chosen)} definitions…"):
                st.session_state.cmp = compare_curvature(
                    G_input,
                    requests=[CurvatureRequest(name, label=name) for name in chosen],
                    semantics=ui_semantics,
                    proc=int(cmp_proc),
                )
                st.session_state.cmp_signature = current_cmp_signature

        cmp = st.session_state.get("cmp")
        if cmp is not None:
            st.markdown("**Applicability and distribution**")
            st.dataframe(cmp.summary(), width="stretch", hide_index=True)

            agreement = cmp.agreement()
            if agreement:
                st.markdown("**Pairwise agreement** (on edges both could compute)")
                st.dataframe(agreement, width="stretch", hide_index=True)

            if any(run.ok for run in cmp.runs):
                fig = viz.plot_curvature_comparison(cmp)
                st.pyplot(fig, width="stretch")
                plt.close(fig)

                st.download_button(
                    "⬇  Download comparison CSV",
                    data=cmp.csv_text(),
                    file_name="curvature_comparison.csv",
                    mime="text/csv",
                    key="cmp_dl",
                )

# ── Compare flow equations (independent of the main run) ──────────────────────
FLOW_PRESETS = ("normalized", "unnormalized", "additive")

# A run that stopped abnormally still returns valid committed states: they are
# displayed with this explanation and the engine's diagnosis above them.
ABNORMAL_STOPS = {
    "diverged": (
        "Diverged",
        "**The flow diverged — the results below are still valid.**\n\n"
        "Every state shown was committed before an update failed to produce "
        "finite positive edge quantities. The diagnosis below identifies "
        "the invalid value or flow-expression error.",
    ),
    "undefined": (
        "Stopped",
        "**Surgery produced a graph this curvature is not defined on — "
        "the results below are still valid.**\n\n"
        "Every state shown was committed before that surgery; the "
        "post-surgery graph was not recorded.",
    ),
    "numerical": (
        "Stopped",
        "**The curvature solver failed on the next state — the results "
        "below are still valid.**\n\n"
        "Every state shown was committed before that failure. It usually "
        "means the flow has collapsed or blown up the evolving quantity "
        "beyond what double precision can resolve.",
    ),
}

if G_input is not None and input_error is None:
    with st.expander("🌊  Compare flow equations on this graph", expanded=False):
        st.caption(
            "The curvature, flow mode, surgery, iterations and step from the "
            "sidebar are shared; only the flow equation varies. An equation that "
            "cannot run is reported with its failure kind."
        )
        chosen_eqs = st.multiselect(
            "Flow equations", list(FLOW_PRESETS), default=list(FLOW_PRESETS),
            key="fcmp_equations",
        )
        extra_expr = st.text_input(
            "Additional expression (optional)", value="", key="fcmp_expression",
            help="e.g. -eta*tanh(kappa - kbar)*w",
        )
        fcmp_proc = st.number_input(
            "Processes", 1, max(1, _CPU_COUNT), 1, key="fcmp_proc",
            help="Parallel workers for each curvature solve.",
        )
        equations = {name: name for name in chosen_eqs}
        if extra_expr.strip():
            equations[extra_expr.strip()] = extra_expr.strip()
        ui_surgery = None if surgery_name == "no_surgery" else {
            "name": surgery_name, "portion": surgery_portion, "interval": surgery_interval,
        }
        current_fcmp_signature = (
            current_graph_signature, curvature_method, _freeze(curvature_params),
            tuple(equations), flow_mode, int(iterations), float(step), float(delta),
            _freeze(ui_surgery), int(fcmp_proc),
        )
        if (
            st.session_state.fcmp is not None
            and st.session_state.fcmp_signature != current_fcmp_signature
        ):
            st.session_state.fcmp = None
            st.session_state.fcmp_signature = None
        if st.button("Compare equations", key="fcmp_run",
                     disabled=not equations or bool(flow_not_applicable)):
            with st.spinner(f"Running {len(equations)} flows…"):
                st.session_state.fcmp = compare_flow(
                    G_input, equations,
                    curvature=CurvatureRequest(curvature_method, curvature_params),
                    semantics=ui_semantics, evolve=evolve, proc=int(fcmp_proc),
                    iterations=int(iterations), step=float(step), delta=float(delta),
                    surgery=ui_surgery,
                )
                st.session_state.fcmp_signature = current_fcmp_signature

        fcmp = st.session_state.get("fcmp")
        if fcmp is not None:
            st.dataframe(fcmp.summary(), width="stretch", hide_index=True)
            if any(run.ok for run in fcmp.runs):
                fig = viz.plot_convergence_comparison(fcmp)
                st.pyplot(fig, width="stretch")
                plt.close(fig)
                st.download_button(
                    "⬇  Download convergence comparison CSV", data=fcmp.csv_text(),
                    file_name="flow_comparison.csv", mime="text/csv", key="fcmp_dl",
                )

if run_btn and G_input is not None:
    # ── Run ───────────────────────────────────────────────────────────────────
    st.session_state.sim  = None
    st.session_state.done = False

    surgery_cfg = {
        'name': surgery_name,
        'portion': surgery_portion,
        'interval': surgery_interval,
    }

    sim = RicciFlowSimulator(G_input,
                             curvature=CurvatureRequest(curvature_method, curvature_params),
                             flow_equation=flow_eq, evolve=evolve,
                             proc=int(flow_proc), semantics=ui_semantics)

    status_box   = st.status("Running Ricci flow…", expanded=True)
    progress_bar = st.progress(0, text="Computing initial curvature…")
    log_lines    = []
    log_ph       = st.empty()

    def on_progress(i, total, diff):
        pct = (i + 1) / total
        progress_bar.progress(pct,
                              text=f"Iteration {i+1}/{total} · RC diff {diff:.6f}")
        log_lines.append(f"  {i+1:>4d}  RC diff = {diff:.8f}")
        if len(log_lines) > 10:
            log_lines.pop(0)
        log_ph.code('\n'.join(log_lines), language=None)

    try:
        sim.run(
            iterations=iterations,
            step=step,
            delta=delta,
            surgery=surgery_cfg,
            progress_callback=on_progress,
            verbose=False,
        )
        n_done = sim.result.iterations_completed
        reason = sim.termination_reason
        stopped = ABNORMAL_STOPS.get(reason)
        icon = "⚠️" if stopped else "✅"
        verb = stopped[0] if stopped else "Finished"
        status_box.update(
            label=(f"{icon} {verb} — {n_done} iteration"
                   f"{'s' if n_done != 1 else ''} · {reason}"),
            state="error" if stopped else "complete", expanded=False,
        )
        progress_bar.empty()
        log_ph.empty()
        # The explanation is shown with the results: anything drawn here is
        # cleared by the st.rerun() below.
    except Exception as ex:
        status_box.update(label="❌ Error during simulation", state="error")
        # A diverging flow or an out-of-range step is an ordinary, actionable
        # outcome, not a crash: the engine's message already names the offending
        # iteration, the largest step that would have worked, and whether a
        # smaller step would help at all. Show that, and keep the traceback for
        # the genuinely unexpected cases behind a fold.
        st.error(str(ex))
        with st.expander("Technical details"):
            st.exception(ex)
        st.stop()

    if sim.initial_graph is None or not sim.snapshots:
        st.error("Flow terminated before producing any snapshots — "
                 "check that the input graph has non-zero edge weights.")
        st.stop()

    pos = nx.spring_layout(sim.initial_graph, seed=42)
    st.session_state.sim  = sim
    st.session_state.pos  = pos
    st.session_state.done = True
    st.session_state.sim_signature = current_sim_signature
    st.rerun()


# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.done and st.session_state.sim is not None:
    sim       = st.session_state.sim
    pos       = st.session_state.pos
    snapshots = sim.snapshots
    n_snaps   = len(snapshots)
    # The quantity the flow evolved: the weight, or the distance when the
    # distance evolves. Every "w" display below reads this attribute.
    q_attr = sim.result.evolving_attr or sim.weight
    q_role = sim.result.evolve or "weight"
    q_sym  = "d" if q_role == "distance" else "w"

    stopped = ABNORMAL_STOPS.get(sim.termination_reason)
    if stopped:
        st.warning(stopped[1])
        if sim.result.diagnosis:
            st.code(sim.result.diagnosis, language=None)

    # Summary metrics
    rc_init  = list(nx.get_edge_attributes(sim.initial_graph, 'ricciCurvature').values())
    rc_final = list(nx.get_edge_attributes(sim.result_graph,  'ricciCurvature').values())
    q_final  = list(nx.get_edge_attributes(sim.result_graph,  q_attr).values())

    def _range(vals: list[float]) -> str:
        return f"[{min(vals):.3f}, {max(vals):.3f}]" if vals else "—"

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Iterations",      sim.result.iterations_completed)
    m2.metric("Final RC diff",
              f"{max(rc_final) - min(rc_final):.2e}" if rc_final else "undefined")
    m3.metric("Stopped because", sim.termination_reason or "—")
    m4.metric("Initial RC range", _range(rc_init))
    m5.metric("Final RC range",   _range(rc_final))
    m6.metric(f"Final {q_role} range", _range(q_final))
    st.divider()

    # Tabs
    tab_snap, tab_conv, tab_edges, tab_dist, tab_export = st.tabs([
        "🔍 Snapshot Viewer",
        "📈 Convergence",
        "📉 Edge trajectories",
        "📊 Distributions",
        "💾 Export",
    ])

    # ── Snapshot viewer ───────────────────────────────────────────────────────
    with tab_snap:
        col_ctrl, col_graph = st.columns([1, 2.8])

        with col_ctrl:
            # A run that stopped on its first update has only the input state,
            # and a slider needs two distinct ends.
            if n_snaps > 1:
                snap_idx = st.slider(
                    "Iteration", 0, n_snaps - 1, 0,
                    key="snap_slider",
                    help="Drag to scrub through the Ricci flow.",
                )
            else:
                snap_idx = 0
                st.caption("Only the input state was committed.")
            G_snap   = snapshots[snap_idx]
            rc_snap  = list(nx.get_edge_attributes(G_snap, 'ricciCurvature').values())
            q_snap   = list(nx.get_edge_attributes(G_snap, q_attr).values())

            st.markdown(f"**Iteration {snap_idx}**")
            if rc_snap:
                st.metric("RC min",  f"{min(rc_snap):.5f}")
                st.metric("RC max",  f"{max(rc_snap):.5f}")
                st.metric("RC mean", f"{np.mean(rc_snap):.5f}")
                if snap_idx > 0 and snap_idx - 1 < len(sim.convergence):
                    st.metric("RC diff", f"{sim.convergence[snap_idx-1]:.2e}")
            elif snap_idx == n_snaps - 1 and sim.termination_reason == "exhausted":
                st.metric("RC diff", "undefined")
                st.caption("Surgery removed every edge; this is exhaustion, not convergence.")
            if q_snap:
                st.markdown("---")
                st.metric(f"{q_sym} min",  f"{min(q_snap):.5f}")
                st.metric(f"{q_sym} max",  f"{max(q_snap):.5f}")
                st.metric(f"{q_sym} mean", f"{np.mean(q_snap):.5f}")
            st.markdown("---")
            st.caption(f"Nodes: {G_snap.number_of_nodes()} · "
                       f"Edges: {G_snap.number_of_edges()}")
            show_labels = st.checkbox("Edge labels", value=True)

        with col_graph:
            # Compute a global RC range for a consistent colorbar
            all_rc = [v for G in snapshots
                      for v in nx.get_edge_attributes(G, 'ricciCurvature').values()]
            abs_max = max(abs(min(all_rc, default=1)), abs(max(all_rc, default=1)), 1e-6)
            norm    = mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            cmap    = cm.RdBu_r

            # Edges are coloured on the same scale as the colorbar, so a
            # colour means the same curvature in every iteration.
            fig, ax = plt.subplots(figsize=(7, 5.5))
            viz.plot_graph(G_snap, pos=pos, ax=ax,
                           title=f"Iteration {snap_idx}", weight_attr=q_attr,
                           show_labels=show_labels, colorbar=False, norm=norm)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, label="Ricci Curvature")
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

    # ── Convergence ───────────────────────────────────────────────────────────
    with tab_conv:
        log_y = st.checkbox("Log scale", value=False)
        fig   = viz.plot_convergence(sim.convergence, log_scale=log_y)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # ── Distributions ─────────────────────────────────────────────────────────
    with tab_edges:
        # Every value shown here comes from the package's trajectory API, which
        # reads kappa and the evolving quantity from the same committed state.
        all_traj = extract_edge_trajectories(sim.result)
        directed = all_traj.directed
        edge_options = list(all_traj.edges)
        n_edges = len(edge_options)
        role_label = all_traj.quantity_label("evolving_quantity")

        def trajectory_score(edge):
            present = [v for v in all_traj.evolving_quantity[edge] if v is not None]
            span = max(present) - min(present) if present else 0.0
            return not all_traj.present[edge][-1], span

        default_edges = (
            edge_options if n_edges <= 12
            else sorted(edge_options, key=trajectory_score, reverse=True)[:8]
        )

        st.caption(
            "Each point is one committed flow state (0 is the input). A line stops "
            "when surgery removes that edge; missing edges are never plotted as zero."
        )
        c_scope, c_quantity = st.columns([1.3, 1])
        with c_scope:
            scope = st.radio(
                "Edges to draw",
                ["All edges", "Selected edges", "Random sample"],
                index=0 if n_edges <= TRAJECTORY_EDGE_LIMIT else 1,
                horizontal=True,
                key="trajectory_scope",
                help=(
                    f"Graphs with more than {TRAJECTORY_EDGE_LIMIT} edges start on "
                    "'Selected edges' so hundreds of traces are not drawn by default."
                ),
            )
        with c_quantity:
            quantity = st.segmented_control(
                "Quantity",
                ["Curvature", role_label.capitalize(), "Both"],
                default="Both",
                key="trajectory_quantity",
            )
        selected_edges = st.multiselect(
            "Edges",
            edge_options,
            default=default_edges,
            format_func=all_traj.edge_label,
            key="trajectory_edges",
            disabled=scope != "Selected edges",
            help=(
                "Used when 'Selected edges' is chosen. On larger graphs the default "
                "is the eight edges with the largest change, prioritising edges "
                "removed by surgery."
            ),
        )
        if scope == "Random sample":
            sample_size = st.number_input(
                "Sample size", min_value=1, max_value=n_edges,
                value=min(TRAJECTORY_EDGE_LIMIT, n_edges), step=1,
                key="trajectory_sample_size",
                help="A fixed-seed sample, so it does not change between reruns.",
            )
            shown_edges = random.Random(0).sample(edge_options, int(sample_size))
        elif scope == "Selected edges":
            shown_edges = list(selected_edges)
        else:
            shown_edges = edge_options

        c_hl, c_group = st.columns(2)
        with c_hl:
            highlight_choice = st.multiselect(
                "Highlight", edge_options, default=None,
                format_func=all_traj.edge_label, key="trajectory_highlight",
                help="Drawn bold on top and named in the legend.",
            )
        G0 = sim.initial_graph
        flow_attrs = {"ricciCurvature", "original_RC", sim.weight, sim.distance}
        group_attrs = sorted({
            key for _, _, data in G0.edges(data=True) for key in data
            if key not in flow_attrs
            and len({_freeze(d.get(key)) for _, _, d in G0.edges(data=True)}) <= 12
        })
        with c_group:
            group_attr = st.selectbox(
                "Group by edge attribute", ["(none)", *group_attrs],
                key="trajectory_group_attr",
                help="Categorical edge attributes of the input graph (at most 12 values).",
            )
        c_ind, c_agg = st.columns(2)
        show_individual = c_ind.checkbox(
            "Show individual trajectories", value=True, key="trajectory_individual"
        )
        aggregate = c_agg.checkbox(
            "Mean by group", value=False, key="trajectory_aggregate",
            help="One mean line per group (all drawn edges if no grouping), over "
                 "the edges still present in each state.",
        )
        groups = None
        if group_attr != "(none)":
            groups = {
                (u, v): str(data[group_attr])
                for u, v, data in G0.edges(data=True) if group_attr in data
            }

        shown_set = set(shown_edges)
        highlight = [e for e in highlight_choice if e in shown_set]
        if len(highlight) < len(highlight_choice):
            st.info(
                f"{len(highlight_choice) - len(highlight)} highlighted edge(s) are "
                "not among the edges being drawn and are not shown."
            )

        too_many = show_individual and len(shown_edges) > TRAJECTORY_EDGE_LIMIT
        if too_many:
            st.info(
                f"{len(shown_edges)} edges are selected, more than the "
                f"{TRAJECTORY_EDGE_LIMIT} individual trajectories drawn at once. "
                "Individual lines are hidden: select or sample fewer edges, "
                "highlight the edges of interest, or turn on 'Mean by group'."
            )
        draw_individual = show_individual and not too_many

        if not shown_edges:
            st.info("Select at least one edge to draw its trajectory.")
        elif not (draw_individual or aggregate or highlight):
            st.info("Nothing to draw: show individual trajectories, highlight "
                    "edges, or turn on 'Mean by group'.")
        else:
            wanted = {
                "Curvature": ["curvature"],
                "Both": ["curvature", "evolving_quantity"],
            }.get(quantity, ["evolving_quantity"])
            fig, axes = plt.subplots(1, len(wanted), figsize=(5.2 * len(wanted), 3.4),
                                     squeeze=False)
            for ax, q in zip(axes[0], wanted):
                viz.plot_edge_trajectories(
                    sim.result, quantity=q, edges=shown_edges, groups=groups,
                    aggregate=aggregate, show_individual=draw_individual,
                    highlight_edges=highlight, ax=ax,
                )
            fig.tight_layout()
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        if shown_edges:
            export = extract_edge_trajectories(sim.result, edges=shown_edges, groups=groups)
            buffer = io.StringIO()
            save_edge_trajectories_csv(export, buffer)
            st.download_button(
                "Download displayed trajectories",
                data=buffer.getvalue(),
                file_name="edge_trajectories.csv",
                mime="text/csv",
                icon=":material/download:",
                key="trajectory_download",
            )

    # ── Distributions ───────────────────────────────────────────────────────────────────────────
    with tab_dist:
        dist_steps = sorted(set([
            0, n_snaps // 4, n_snaps // 2, 3 * n_snaps // 4, n_snaps - 1
        ]))

        st.markdown("#### Ricci Curvature")
        fig = viz.plot_curvature_distribution(snapshots, steps=dist_steps)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

        st.markdown(f"#### Edge {q_role.capitalize()}")
        fig = viz.plot_weight_distribution(snapshots, steps=dist_steps, weight_attr=q_attr)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        st.markdown("### Download")

        col_a, col_b = st.columns(2)

        with col_a:
            # Final graph as GEXF
            with tempfile.NamedTemporaryFile(suffix='.gexf', delete=False) as tmp:
                nx.write_gexf(sim.result_graph, tmp.name)
                gexf_path = tmp.name
            with open(gexf_path, 'rb') as f:
                gexf_bytes = f.read()
            os.unlink(gexf_path)
            st.download_button(
                "⬇ Final graph (.gexf)",
                data=gexf_bytes,
                file_name="ricci_flow_result.gexf",
                mime="application/octet-stream",
                width="stretch",
            )

        with col_b:
            # Convergence CSV
            conv_csv = "iteration,rc_diff\n" + "\n".join(
                f"{i+1},{v}" for i, v in enumerate(sim.convergence)
            )
            st.download_button(
                "⬇ Convergence data (.csv)",
                data=conv_csv,
                file_name="convergence.csv",
                mime="text/csv",
                width="stretch",
            )

        st.divider()
        st.markdown("### Generate animation")
        c_fps, c_labels = st.columns(2)
        with c_fps:
            fps = st.slider("FPS", 1, 15, 5)
        with c_labels:
            anim_labels = st.checkbox("Show edge labels", value=False)

        if st.button("🎬 Generate GIF", width="stretch"):
            with st.spinner("Rendering frames…"):
                with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
                    gif_path = tmp.name
                viz.save_animation(snapshots, gif_path, pos=pos, weight_attr=q_attr,
                                   fps=fps, show_labels=anim_labels)
                with open(gif_path, 'rb') as f:
                    gif_bytes = f.read()
                os.unlink(gif_path)
            st.download_button(
                "⬇ Download animation (.gif)",
                data=gif_bytes,
                file_name="ricci_flow.gif",
                mime="image/gif",
                width="stretch",
            )


# ── Empty state ───────────────────────────────────────────────────────────────
elif not st.session_state.done:
    if G_input is not None and input_error is None:
        st.markdown("### Graph Preview")
        pos_preview = nx.spring_layout(G_input, seed=42)
        fig, ax = plt.subplots(figsize=(7, 5))
        small = G_input.number_of_nodes() <= 30
        viz.plot_graph(G_input, pos=pos_preview, ax=ax, colorbar=False,
                       show_labels=small,
                       title=(f"Input graph — "
                              f"{G_input.number_of_nodes()} nodes, "
                              f"{G_input.number_of_edges()} edges"))
        fig.tight_layout()
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.info("👈 Set parameters in the sidebar, then click **▶ Run Simulation**.")
    else:
        st.info("👈 Choose or define a graph in the sidebar to get started.")
        st.markdown("""
**What this app does:**

1. Build any graph (built-in, custom edges, or file upload)
2. Configure Ricci flow parameters (step size, iterations, optional surgery)
3. Run the simulation — watch live progress
4. Explore results interactively: scrub through iterations with a slider, inspect convergence curves, RC & weight distributions
5. Download the final graph or an animated GIF
""")
