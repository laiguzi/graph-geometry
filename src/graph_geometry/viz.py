"""Visualization utilities (optional — requires the ``[viz]`` extra: matplotlib).

Curvature-wise:
    :func:`plot_graph` (graph coloured by curvature), :func:`plot_curvature_histogram`
    (one result), :func:`plot_curvature_comparison` (a :class:`CurvatureComparison`).
    Value tables come from ``CurvatureResult.rows()`` / ``CurvatureComparison.rows()``.
Flow-wise:
    :func:`plot_convergence`, :func:`plot_convergence_comparison` (a
    :class:`FlowComparison`), :func:`edge_trajectories` / :func:`plot_edge_trajectories` (per-edge
    series from :func:`graph_geometry.flow.extract_edge_trajectories`),
    :func:`plot_curvature_distribution` / :func:`plot_weight_distribution`,
    :func:`plot_flow_comparison` (states side by side), :func:`animate_flow`.

All plotting functions return a ``matplotlib.figure.Figure`` (or an animation),
so the caller decides what to do (``plt.show()`` / ``st.pyplot(fig)`` /
``plt.close(fig)``). Not imported by the package root, so the core stays
matplotlib-free; use ``from graph_geometry import viz``.
"""

from __future__ import annotations

import networkx as nx

try:
    import matplotlib.animation as animation
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "graph_geometry.viz requires matplotlib. Install it with: "
        "pip install 'graph-geometry[viz]'  (or: uv sync --extra viz)"
    ) from e

__all__ = [
    "edge_trajectories",
    "plot_graph",
    "plot_curvature_histogram",
    "plot_curvature_comparison",
    "plot_edge_trajectories",
    "plot_convergence_comparison",
    "plot_flow_comparison",
    "plot_convergence",
    "plot_curvature_distribution",
    "plot_weight_distribution",
    "animate_flow",
    "save_animation",
]


# ── internal helpers ─────────────────────────────────────────────────────────


def _consistent_layout(G, seed=42):
    return nx.spring_layout(G, seed=seed)


def _rc_norm(values):
    """TwoSlopeNorm centred at 0 for diverging RC colours."""
    if not values:
        return mcolors.Normalize(vmin=-1, vmax=1)
    abs_max = max(abs(min(values)), abs(max(values)), 1e-6)
    return mcolors.TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)


def _axes(ax, figsize):
    """``(fig, ax, standalone)`` for an optional caller-supplied axis."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax, True
    return ax.get_figure(), ax, False


def _values(result):
    """Curvature values of a result, a mapping, or an iterable of numbers."""
    from collections.abc import Mapping

    if isinstance(result, Mapping):
        return [float(v) for v in result.values()]
    values = getattr(result, "values", None)
    if isinstance(values, Mapping):
        return [float(v) for v in values.values()]
    return [float(v) for v in result]


def _definition_color(resolved):
    if resolved is None:
        return None
    from .curvature.registry import get_curvature_spec

    try:
        return get_curvature_spec(resolved.method).color or None
    except Exception:  # noqa: BLE001 - an unregistered plugin simply has no colour
        return None


def _default_steps(n):
    return sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})


def _draw_graph(G, pos, ax, weight_attr="weight", show_labels=True,
                node_size=300, font_size=8, title=None, norm=None):
    """Draw ``G`` on ``ax`` in place; return ``(cmap, norm)`` for a colorbar.

    ``norm`` defaults to one centred on 0 and scaled to this graph's values.
    """
    edges = list(G.edges())
    rc_vals = [G[u][v].get("ricciCurvature", 0.0) for u, v in edges]
    w_vals = [G[u][v].get(weight_attr, 1.0) for u, v in edges]

    cmap = cm.RdBu_r
    if norm is None:
        norm = _rc_norm(rc_vals)
    edge_colors = [cmap(norm(v)) for v in rc_vals] if rc_vals else ["#888888"] * len(edges)

    if w_vals and max(w_vals) > min(w_vals):
        w_min, w_max = min(w_vals), max(w_vals)
        edge_widths = [0.5 + 3.5 * (w - w_min) / (w_max - w_min) for w in w_vals]
    else:
        edge_widths = [1.5] * len(edges)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#4A90D9", node_size=node_size, alpha=0.9)
    edge_kw = dict(edge_color=edge_colors, width=edge_widths, alpha=0.85)
    if G.is_directed():
        edge_kw.update(arrows=True, arrowsize=12, connectionstyle="arc3,rad=0.15")
    nx.draw_networkx_edges(G, pos, ax=ax, **edge_kw)

    if show_labels:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                                font_color="white", font_weight="bold")
        w_labels = {(u, v): f"{G[u][v].get(weight_attr, 1.0):.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=w_labels, ax=ax,
                                     font_size=max(5, font_size - 1),
                                     font_color="#222222", label_pos=0.65)
        if any("ricciCurvature" in G[u][v] for u, v in edges):
            rc_labels = {(u, v): f"{G[u][v]['ricciCurvature']:.2f}" for u, v in edges}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=rc_labels, ax=ax,
                                         font_size=max(5, font_size - 1),
                                         font_color="#CC0000", label_pos=0.35)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=10, pad=4)
    return cmap, norm


# ── public API ───────────────────────────────────────────────────────────────


def edge_trajectories(snapshots, attr="ricciCurvature", edges=None):
    """Return one snapshot-aligned value series per edge.

    The result is ``dict[(u, v)] -> list[float | None]``. List index ``i``
    refers to committed state ``snapshots[i]``. ``None`` means that the edge is
    absent (for example after surgery) or that the selected attribute is not
    present in that state; it is deliberately not replaced by zero.

    ``edges`` defaults to the union of edges seen across all snapshots. For an
    undirected series, ``(u, v)`` and ``(v, u)`` identify the same edge while
    preserving the orientation of its first appearance for display.
    """
    snapshots = list(snapshots)
    if not snapshots:
        return {}

    directed = snapshots[0].is_directed()
    if any(G.is_directed() != directed for G in snapshots):
        raise ValueError("Every flow snapshot must have the same graph direction.")

    def identity(edge):
        return tuple(edge) if directed else frozenset(edge)

    selected = []
    seen = set()
    candidates = (
        (edge for G in snapshots for edge in G.edges())
        if edges is None else edges
    )
    for edge in candidates:
        edge = tuple(edge)
        if len(edge) != 2:
            raise ValueError(f"An edge must contain two nodes, got {edge!r}.")
        key = identity(edge)
        if key not in seen:
            seen.add(key)
            selected.append(edge)

    trajectories = {}
    for u, v in selected:
        values = []
        for G in snapshots:
            if not G.has_edge(u, v) or attr not in G[u][v]:
                values.append(None)
            else:
                values.append(float(G[u][v][attr]))
        trajectories[(u, v)] = values
    return trajectories


def plot_graph(G, pos=None, ax=None, title=None, weight_attr="weight",
               show_labels=True, node_size=300, font_size=8,
               figsize=(7, 5), colorbar=True, norm=None):
    """Plot one snapshot with RC-coloured edges (blue<0<red), width ~ weight.

    ``norm`` maps curvature to colour. It defaults to a scale centred on 0 and
    fitted to this graph; pass one shared norm to draw several snapshots on the
    same scale, e.g. against a colorbar drawn once for a whole flow.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    if pos is None:
        pos = _consistent_layout(G)
    cmap, norm = _draw_graph(G, pos, ax, weight_attr=weight_attr, show_labels=show_labels,
                             node_size=node_size, font_size=font_size, title=title,
                             norm=norm)
    if colorbar and standalone:
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, label="Ricci Curvature")
    if standalone:
        fig.tight_layout()
    return fig


def plot_curvature_histogram(result, ax=None, bins=25, color=None, title=None,
                             figsize=(5.2, 4.0)):
    """Histogram of one curvature result (a ``CurvatureResult`` or ``{edge: kappa}``).

    The colour defaults to the registered definition's colour when the result
    carries a resolved record.
    """
    fig, ax, standalone = _axes(ax, figsize)
    values = _values(result)
    color = color or _definition_color(getattr(result, "resolved", None))
    ax.hist(values, bins=bins, color=color, edgecolor="white", alpha=0.9)
    ax.axvline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("curvature")
    ax.set_ylabel("count")
    if title:
        ax.set_title(title, fontsize=10)
    if standalone:
        fig.tight_layout()
    return fig


def plot_curvature_comparison(comparison, ax=None, bins=25, figsize=(7, 3.2)):
    """Overlaid histograms of every successful run of a ``CurvatureComparison``."""
    fig, ax, standalone = _axes(ax, figsize)
    for run in comparison.runs:
        if not run.ok:
            continue
        ax.hist(_values(run.outcome), bins=bins, alpha=0.5, label=run.label,
                color=_definition_color(run.outcome.resolved))
    ax.axvline(0.0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("curvature")
    ax.set_ylabel("edges")
    if any(run.ok for run in comparison.runs):
        ax.legend(fontsize=8)
    if standalone:
        fig.tight_layout()
    return fig


def plot_edge_trajectories(flow, attr=None, edges=None, ax=None,
                           figsize=(8, 3.5), legend=True, *,
                           quantity=None, groups=None, aggregate=False,
                           show_individual=True, highlight_edges=None,
                           colors=None, evolving_attr=None, evolve=None,
                           individual_alpha=None, max_legend_entries=12,
                           title=None):
    """Per-edge trajectories of one quantity over the committed states of a flow.

    Draws one axes for one quantity; call it twice (or pass two axes) for
    curvature and the evolving quantity side by side. Values come from
    :func:`graph_geometry.flow.extract_edge_trajectories`, so ``kappa_e^t`` and
    ``q_e^t`` both belong to state ``t``; nothing is recomputed.

    Parameters
    ----------
    flow :
        A ``FlowResult`` (or a run simulator), or a sequence of graph states.
    attr :
        Legacy selector: ``"ricciCurvature"`` means ``quantity="curvature"``;
        any other attribute is plotted as the evolving quantity. Prefer
        ``quantity``.
    edges :
        Edges to draw (either orientation if undirected). Default: all.
    quantity :
        ``"curvature"`` (default) or ``"evolving_quantity"``. The y label is
        ``edge curvature``, ``edge weight`` or ``edge distance`` according to
        the role recorded on the ``FlowResult`` (or given by ``evolve``).
    groups :
        ``{edge: group name}``; every key must be an initial-graph edge.
        Individual lines take their group's colour.
    aggregate :
        ``False``, ``True``/``"mean"``, ``"median"``, ``"min"`` or ``"max"``:
        overlay one summary line per group (all drawn edges form one group
        when ``groups`` is omitted). A group summary at state ``t`` uses the
        group's edges still present at ``t``; if surgery removed them all the
        line stops.
    show_individual :
        Draw every selected edge. Individual lines are faint whenever groups,
        summaries or highlights are drawn, or when there are more edges than
        ``max_legend_entries``; they are then left out of the legend.
    highlight_edges :
        Edges drawn bold on top and named in the legend.
    colors :
        ``{group or edge: colour}``; unlisted groups/edges get deterministic
        colours from a colour-blind-safe cycle, in first-appearance order.
    evolving_attr, evolve :
        Override the evolving attribute and its role (``"weight"`` or
        ``"distance"``); needed for a bare sequence of states.

    A removed edge leaves a gap: its line ends at the last state containing it.
    Returns the ``matplotlib.figure.Figure``.
    """
    import math

    from matplotlib.ticker import MaxNLocator

    from .flow.trajectories import extract_edge_trajectories

    if attr is not None and quantity is None:
        if attr == "ricciCurvature":
            quantity = "curvature"
        else:
            quantity, evolving_attr = "evolving_quantity", evolving_attr or attr
    quantity = quantity or "curvature"
    if aggregate is True:
        aggregate = "mean"

    fig, ax, standalone = _axes(ax, figsize)
    data = extract_edge_trajectories(flow, edges=edges, groups=groups,
                                     evolving_attr=evolving_attr, evolve=evolve)
    series = data.series(quantity)
    highlighted = [data.canonical_edge(e) for e in (highlight_edges or ())]
    colors = dict(colors or {})
    x = list(range(data.n_states))

    def ys(values):
        return [math.nan if v is None else v for v in values]

    palette = iter(_TRAJECTORY_PALETTE * 8)
    group_order = list(dict.fromkeys((groups or {}).values()))
    group_color = {}
    for g in group_order:
        group_color[g] = colors[g] if g in colors else next(palette)

    n = len(data.edges)
    emphasis = bool(data.groups or aggregate or highlighted)
    plain_distinct = not emphasis and n <= min(10, max_legend_entries)
    alpha = individual_alpha if individual_alpha is not None else (
        0.95 if plain_distinct else 0.22 if emphasis else 0.35
    )
    handles = []
    edge_color = {}
    for e in data.edges:
        if e in colors:
            edge_color[e] = colors[e]
        elif e in highlighted:
            edge_color[e] = next(palette)
        elif e in data.groups:
            edge_color[e] = group_color[data.groups[e]]
        elif plain_distinct:
            edge_color[e] = next(palette)
        else:
            edge_color[e] = _NEUTRAL

    if show_individual:
        for e in data.edges:
            label = data.edge_label(e) if plain_distinct else None
            ax.plot(x, ys(series[e]), color=edge_color[e], alpha=alpha,
                    lw=1.2 if plain_distinct else 0.8, label=label, zorder=1)
        if not plain_distinct and not aggregate and data.groups:
            members = data.group_members()
            for g in group_order:
                if g in members:
                    handles.append(plt.Line2D([], [], color=group_color[g], lw=1.6,
                                              label=_count_label(g, len(members[g]))))
        elif not plain_distinct and not data.groups:
            handles.append(plt.Line2D([], [], color=_NEUTRAL, alpha=max(alpha, 0.6),
                                      lw=1.0, label=_count_label("edges", n)))

    if aggregate:
        members = data.group_members() if data.groups else {"all edges": data.edges}
        ungrouped = tuple(e for e in data.edges if e not in data.groups)
        if data.groups and ungrouped:
            members["ungrouped"] = ungrouped
            group_color.setdefault("ungrouped", _NEUTRAL)
        summary = data.aggregate(quantity, aggregate, members=members)
        for g, values in summary.items():
            color = group_color.get(g, colors.get(g, "#222222"))
            line, = ax.plot(x, ys(values), color=color, lw=1.9, zorder=3,
                            label=_count_label(g, len(members[g])))
            handles.append(line)

    for e in highlighted:
        line, = ax.plot(x, ys(series[e]), color=edge_color[e], lw=2.2, zorder=4,
                        label=data.edge_label(e))
        handles.append(line)

    if plain_distinct and show_individual:
        handles = ax.get_legend_handles_labels()[0] + handles
    ax.set_xlabel("committed iteration")
    ax.set_ylabel(data.quantity_label(quantity))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if data.n_states > 1:
        ax.set_xlim(0, data.n_states - 1)
    ax.grid(True, alpha=0.25, lw=0.4)
    if title:
        ax.set_title(title, fontsize=10)
    if legend and handles:
        if len(handles) > max_legend_entries:
            handles = handles[:max_legend_entries]
            handles.append(plt.Line2D([], [], lw=0, label="(legend truncated)"))
        ax.legend(handles=handles, fontsize=7, frameon=False)
    if standalone:
        fig.tight_layout()
    return fig


def _count_label(name, count):
    return f"{name} ({count} edge{'' if count == 1 else 's'})"


#: Okabe-Ito hues without yellow (too faint on white), then the tab10 order.
_TRAJECTORY_PALETTE = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000",
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
)
_NEUTRAL = "#7A7A7A"


def plot_convergence_comparison(flow_comparison, ax=None, log_scale=False, figsize=(8, 3.5)):
    """Curvature spread per committed state for every successful run of a ``FlowComparison``."""
    fig, ax, standalone = _axes(ax, figsize)
    for run in flow_comparison.runs:
        if not run.ok:
            continue
        convergence = run.outcome.convergence
        ax.plot(range(1, len(convergence) + 1), convergence, linewidth=1.6,
                label=f"{run.label} ({run.outcome.termination_reason})")
    ax.set_xlabel("iteration")
    ax.set_ylabel("curvature spread (max − min)")
    ax.grid(True, alpha=0.3)
    if log_scale and all(
        min(run.outcome.convergence, default=1.0) > 0 for run in flow_comparison.runs if run.ok
    ):
        ax.set_yscale("log")
    if any(run.ok for run in flow_comparison.runs):
        ax.legend(fontsize=8)
    if standalone:
        fig.tight_layout()
    return fig


def plot_flow_comparison(snapshots, steps=None, pos=None, weight_attr="weight",
                         figsize=None, show_labels=True):
    """Plot several snapshots side by side; returns ``(fig, pos)``."""
    n = len(snapshots)
    steps = steps if steps is not None else _default_steps(n)
    if pos is None:
        pos = _consistent_layout(snapshots[0])
    ncols = len(steps)
    fig, axes = plt.subplots(1, ncols, figsize=figsize or (4 * ncols, 4))
    if ncols == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        _draw_graph(snapshots[min(step, n - 1)], pos, ax, weight_attr=weight_attr,
                    show_labels=show_labels, title=f"Iteration {step}")
    fig.suptitle("Ricci Flow Snapshots", fontsize=12)
    fig.tight_layout()
    return fig, pos


def plot_convergence(convergence, ax=None, figsize=(9, 3.5), log_scale=False):
    """Plot the RC difference (max − min) over iterations."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    iters = range(1, len(convergence) + 1)
    ax.plot(iters, convergence, color="#2E86AB", linewidth=1.8)
    ax.fill_between(iters, convergence, alpha=0.15, color="#2E86AB")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("RC Difference (max − min)", fontsize=11)
    ax.set_title("Ricci Curvature Convergence", fontsize=12)
    ax.grid(True, alpha=0.3)
    if log_scale and convergence and min(convergence) > 0:
        ax.set_yscale("log")
    if standalone:
        fig.tight_layout()
    return fig


def _distribution(snapshots, attr, steps, figsize, bins, color, title, xlabel):
    n = len(snapshots)
    steps = steps if steps is not None else _default_steps(n)
    ncols = len(steps)
    fig, axes = plt.subplots(1, ncols, figsize=figsize or (3.5 * ncols, 3.5), sharey=True)
    if ncols == 1:
        axes = [axes]
    allv = []
    for s in steps:
        allv.extend(nx.get_edge_attributes(snapshots[min(s, n - 1)], attr).values())
    lo, hi = (min(allv), max(allv)) if allv else (-1, 1)
    if hi - lo < 1e-6:
        c = (lo + hi) / 2
        lo, hi = c - 0.5, c + 0.5
    for ax, step in zip(axes, steps):
        vals = list(nx.get_edge_attributes(snapshots[min(step, n - 1)], attr).values())
        ax.hist(vals, bins=bins, range=(lo, hi), color=color, edgecolor="white", alpha=0.85)
        if attr == "ricciCurvature":
            ax.axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.6)
        ax.set_title(f"Iter {step}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Count", fontsize=10)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_curvature_distribution(snapshots, steps=None, weight_attr="weight",
                                figsize=None, bins=20):
    """Histograms of edge Ricci curvature at selected iterations."""
    return _distribution(snapshots, "ricciCurvature", steps, figsize, bins,
                         "#5C85D6", "Ricci Curvature Distribution", "RC")


def plot_weight_distribution(snapshots, steps=None, weight_attr="weight",
                             figsize=None, bins=20):
    """Histograms of an edge attribute (``weight`` by default) at selected iterations.

    Titles and axis labels name ``weight_attr``, so a distance flow plotted with
    ``weight_attr="distance"`` is labelled as distances.
    """
    label = weight_attr.capitalize()
    return _distribution(snapshots, weight_attr, steps, figsize, bins,
                         "#E8834B", f"Edge {label} Distribution", label)


def animate_flow(snapshots, pos=None, weight_attr="weight", figsize=(7, 5.5),
                 interval=200, show_labels=True):
    """A ``FuncAnimation`` of the flow (display via ``HTML(ani.to_jshtml())``)."""
    if pos is None:
        pos = _consistent_layout(snapshots[0])
    fig, ax = plt.subplots(figsize=figsize)

    def draw_frame(i):
        ax.clear()
        ax.set_axis_off()
        _draw_graph(snapshots[i], pos, ax, weight_attr=weight_attr,
                    show_labels=show_labels, title=f"Ricci Flow — Iteration {i}")

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(snapshots),
                                  interval=interval, repeat=True)
    plt.close(fig)
    return ani


def save_animation(snapshots, path, pos=None, weight_attr="weight", fps=5,
                   figsize=(7, 5.5), dpi=100, show_labels=True):
    """Save an animation to GIF (pillow) or MP4 (ffmpeg)."""
    if pos is None:
        pos = _consistent_layout(snapshots[0])
    fig, ax = plt.subplots(figsize=figsize)

    def draw_frame(i):
        ax.clear()
        ax.set_axis_off()
        _draw_graph(snapshots[i], pos, ax, weight_attr=weight_attr,
                    show_labels=show_labels, title=f"Ricci Flow — Iteration {i}")

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(snapshots), interval=1000 // fps)
    writer = "pillow" if path.endswith(".gif") else "ffmpeg"
    ani.save(path, writer=writer, fps=fps, dpi=dpi, savefig_kwargs={"bbox_inches": "tight"})
    plt.close(fig)
