"""The two example plugins: full metadata, published values, and every consumer.

These are *user* plugins: they live in ``examples/``, import nothing private,
and are registered by importing the module. Between them they exercise both
extension paths described in the paper's software section:

- ``plugin_sinkhorn_ollivier`` is a transport configuration: the built-in
  Ollivier ``OTCurvatureConfig`` with only its solver ``S_c`` replaced, run by
  ``TransportCurvatureEngine`` on edges and on node pairs;
- ``plugin_balanced_forman`` is a combinatorial definition written directly as
  a compute function against the result contract.

Both register with explicit capabilities, a parameter schema, and citation, so
neither goes through the deprecated registration adapter.
"""

import dataclasses
import importlib.util
import math
import pickle
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import graph_geometry as gg
from graph_geometry.curvature import TransportResult, ollivier_ot_config

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
_REGISTERED = ("sinkhorn_ollivier", "balanced_forman")
_MODULES = ("plugin_sinkhorn_ollivier", "plugin_balanced_forman")


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    # A dataclass defined under ``from __future__ import annotations`` resolves
    # its annotations through ``sys.modules[cls.__module__]``, so the module has
    # to be registered before it is executed -- and pickling needs it too.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def plugins():
    """Import both plugins (registering them), recording import-time warnings."""
    mods, caught = {}, {}
    for key, module in (("sinkhorn", "plugin_sinkhorn_ollivier"),
                        ("balanced", "plugin_balanced_forman")):
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            mods[key] = _load(module)
        caught[module] = [w.category for w in record]
    mods["import_warnings"] = caught
    yield mods
    for name in _REGISTERED:
        gg.unregister_curvature(name)
    for name in _MODULES:
        sys.modules.pop(name, None)


def _undirected(n=5):
    G = nx.complete_graph(n)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    return G


def _cycle_with_chord(directed=False):
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)], directed=directed)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    return G


# ── registration and metadata ───────────────────────────────────────────────


@pytest.mark.parametrize("module", _MODULES)
def test_importing_an_example_uses_no_deprecated_registration(plugins, module):
    categories = plugins["import_warnings"][module]
    assert gg.CurvatureDeprecationWarning not in categories
    assert not any(issubclass(c, gg.CurvatureWarning) for c in categories)


@pytest.mark.parametrize("name", _REGISTERED)
def test_both_plugins_have_complete_metadata(name):
    spec = gg.get_curvature_spec(name)
    assert spec.metadata_complete
    assert spec.capabilities.status == "experimental"
    assert spec.label and spec.color and spec.description and spec.citation
    colours = [s.color for s in gg.CURVATURE_REGISTRY.values() if s.name != name]
    assert spec.color not in colours


def test_sinkhorn_declares_a_transport_definition_on_edges_and_pairs():
    spec = gg.get_curvature_spec("sinkhorn_ollivier")
    caps = spec.capabilities
    assert caps.family == "transport"
    assert caps.graph_kinds == frozenset({"directed", "undirected"})
    assert caps.scopes == frozenset({"edges", "pairs"})
    assert caps.consumes == frozenset({"topology", "edge_weight", "edge_distance"})
    assert spec.compute_pairs is not None and spec.resolver is not None


def test_balanced_forman_declares_an_undirected_topology_only_definition():
    spec = gg.get_curvature_spec("balanced_forman")
    caps = spec.capabilities
    assert caps.family == "combinatorial"
    assert caps.graph_kinds == frozenset({"undirected"})
    assert caps.scopes == frozenset({"edges"})
    assert caps.consumes == frozenset({"topology"})
    assert spec.parameters == () and spec.compute_pairs is None


def test_sinkhorn_parameter_schema_defaults_and_bounds():
    params = {p.name: p for p in gg.get_curvature_spec("sinkhorn_ollivier").parameters}
    assert list(params) == ["alpha", "reg", "kernel", "beta"]

    alpha = params["alpha"]
    assert (alpha.kind, alpha.default, alpha.minimum, alpha.maximum) == ("float", 0.5, 0.0, 1.0)
    assert alpha.minimum_inclusive and alpha.maximum_inclusive

    reg = params["reg"]
    assert (reg.kind, reg.default, reg.minimum, reg.maximum) == ("float", 0.01, 0.0, None)
    assert not reg.minimum_inclusive

    kernel = params["kernel"]
    assert (kernel.kind, kernel.default) == ("choice", "auto")
    assert kernel.choices == ("auto", "undirected", "out", "in", "mixed")

    beta = params["beta"]
    assert (beta.kind, beta.default, beta.minimum, beta.maximum) == ("float", 0.8, 0.0, 1.0)
    assert beta.graph_kinds == frozenset({"directed"})
    assert dict(beta.active_when) == {"kernel": ("mixed",)}


@pytest.mark.parametrize(
    "parameters, match",
    [
        ({"alpha": 1.5}, "alpha"),
        ({"reg": 0.0}, "reg"),
        ({"reg": -0.1}, "reg"),
        ({"kernel": "bogus"}, "kernel"),
        ({"beta": 0.5}, "inactive"),            # beta has no meaning on an undirected graph
        ({"tolerance": 1e-3}, "no parameter"),
    ],
)
def test_sinkhorn_rejects_invalid_parameters_before_computing(parameters, match):
    request = gg.CurvatureRequest("sinkhorn_ollivier", parameters)
    with pytest.raises(gg.CurvatureConfigurationError, match=match):
        gg.resolve_curvature(_undirected(), request)


def test_sinkhorn_resolver_never_leaves_auto():
    undirected = gg.resolve_curvature(_undirected(), "sinkhorn_ollivier")
    assert undirected.parameters["kernel"] == "undirected"
    assert undirected.direction_convention == "undirected"
    assert "beta" not in undirected.parameters

    with pytest.warns(gg.CurvatureWarning, match="resolved to 'mixed'"):
        directed = gg.resolve_curvature(_cycle_with_chord(directed=True), "sinkhorn_ollivier")
    assert directed.parameters["kernel"] == "mixed"
    assert directed.direction_convention == "mixed"
    assert directed.parameters["beta"] == 0.8
    assert "auto" not in _leaf_values(directed.to_dict())


def _leaf_values(record):
    out = []
    for value in record.values():
        out.extend(_leaf_values(value) if isinstance(value, dict) else [value])
    return out


@pytest.mark.parametrize(
    "graph, kernel",
    [(_undirected, "out"), (_undirected, "mixed"),
     (lambda: _cycle_with_chord(directed=True), "undirected")],
)
def test_sinkhorn_resolver_refuses_a_kernel_the_graph_cannot_have(graph, kernel):
    with pytest.raises(gg.CurvatureDomainError, match="does not apply"):
        gg.resolve_curvature(graph(), gg.CurvatureRequest("sinkhorn_ollivier", {"kernel": kernel}))


def test_balanced_forman_refuses_a_directed_graph():
    """No underlying-undirected reinterpretation: the graph kind is declared."""
    with pytest.raises(gg.CurvatureDomainError, match="undirected graphs only"):
        gg.compute_curvature(_cycle_with_chord(directed=True), "balanced_forman")


@pytest.mark.parametrize("module", _MODULES)
def test_plugins_use_only_public_api(module):
    """Pins the claim that a plugin needs no privileged access to the package.

    Every name a plugin imports from ``graph_geometry`` must be public (no
    leading underscore) and exported by the module it comes from, so the two
    example plugins are reachable by any user copying them.
    """
    import ast
    import importlib

    source = (EXAMPLES / f"{module}.py").read_text()
    imported: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "graph_geometry"
        ):
            imported += [(node.module, alias.name) for alias in node.names]

    for module_name, name in imported:
        assert not name.startswith("_"), f"{module_name}.{name} is private"
        origin = importlib.import_module(module_name)
        exported = getattr(origin, "__all__", None)
        assert exported is None or name in exported, (
            f"{module_name}.{name} is not in that module's __all__"
        )


# ── the transport path: one component replaced ──────────────────────────────


def test_sinkhorn_config_is_ollivier_with_only_name_and_solver_replaced(plugins):
    exact = ollivier_ot_config(alpha=0.3, kernel="mixed", beta=0.6)
    variant = plugins["sinkhorn"].sinkhorn_ollivier_config(0.3, 0.05, "mixed", 0.6)
    changed = {f.name for f in dataclasses.fields(exact)
               if getattr(exact, f.name) != getattr(variant, f.name)}
    assert changed == {"name", "solver"}
    assert isinstance(variant.solver, plugins["sinkhorn"].SinkhornOTSolver)
    assert variant.solver.reg == 0.05


def test_sinkhorn_solver_follows_the_transport_solver_protocol(plugins):
    solver = plugins["sinkhorn"].SinkhornOTSolver(reg=0.05)
    a = np.array([0.25, 0.25, 0.5])
    b = np.array([0.5, 0.25, 0.25])
    M = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    plain = solver(a, b, M)
    assert isinstance(plain, TransportResult) and plain.plan is None
    with_plan = solver(a, b, M, return_plan=True)
    assert with_plan.cost == plain.cost
    assert np.allclose(with_plan.plan.sum(axis=1), a, atol=1e-6)
    assert np.allclose(with_plan.plan.sum(axis=0), b, atol=1e-6)


@pytest.mark.filterwarnings("ignore:Sinkhorn did not converge")
def test_sinkhorn_raises_instead_of_returning_a_non_converged_value(plugins):
    """POT returns 2.7e-134 for a cost of 0.5 at reg=1e-3, numItermax=1000."""
    solver = plugins["sinkhorn"].SinkhornOTSolver(reg=0.001, num_iter=1000)
    a = np.array([0.25, 0.25, 0.5])
    b = np.array([0.5, 0.25, 0.25])
    M = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    with pytest.raises(gg.CurvatureNumericalError, match="did not converge"):
        solver(a, b, M)

    generous = plugins["sinkhorn"].SinkhornOTSolver(reg=0.001, num_iter=200_000)
    assert generous(a, b, M).cost == pytest.approx(0.5, abs=1e-6)


def test_sinkhorn_solver_is_picklable(plugins):
    """Picklable => usable with proc > 1, like every built-in component."""
    solver = plugins["sinkhorn"].SinkhornOTSolver(reg=0.05)
    assert pickle.loads(pickle.dumps(solver)).reg == 0.05


def test_sinkhorn_converges_to_exact_ollivier_through_compute_curvature():
    G = _cycle_with_chord()
    exact = gg.compute_curvature(G, gg.CurvatureRequest("ollivier", {"alpha": 0.5})).values

    gaps = []
    for reg in (1.0, 0.1, 0.01):
        result = gg.compute_curvature(
            G, gg.CurvatureRequest("sinkhorn_ollivier", {"alpha": 0.5, "reg": reg})
        )
        assert result.resolved.method == "sinkhorn_ollivier"
        gaps.append(max(abs(result.values[e] - exact[e]) for e in exact))

    assert gaps[0] > gaps[-1]              # smaller reg is closer
    assert gaps[-1] < 1e-4                 # and close enough to be useful


def test_sinkhorn_on_node_pairs_through_compute_pair_curvature():
    G = _cycle_with_chord()
    request = gg.CurvatureRequest("sinkhorn_ollivier", {"alpha": 0.5, "reg": 0.1})
    pairs = [(1, 3), (3, 1), (0, 1)]                   # a non-edge both ways, and an edge
    result = gg.compute_pair_curvature(G, request, pairs)
    assert result.resolved.scope == "pairs"
    assert list(result.values) == pairs
    assert all(math.isfinite(v) for v in result.values.values())

    edge_value = gg.compute_curvature(G, request).values[(0, 1)]
    assert result.values[(0, 1)] == edge_value
    exact = gg.compute_pair_curvature(
        G, gg.CurvatureRequest("ollivier", {"alpha": 0.5}), pairs
    ).values
    for pair in pairs:
        assert result.values[pair] == pytest.approx(exact[pair], abs=1e-3)
    assert all("ricciCurvature" not in d for _, _, d in G.edges(data=True))


@pytest.mark.parametrize("name", _REGISTERED)
def test_plugin_compute_returns_a_raw_mapping_and_never_annotates(name):
    G = _undirected(4)
    spec = gg.get_curvature_spec(name)
    params = {"alpha": 0.5, "reg": 0.1, "kernel": "undirected"} if spec.parameters else {}
    raw = spec.compute_edges(G, semantics=gg.GraphSemantics(), proc=1, **params)
    assert type(raw) is dict and set(raw) == set(G.edges())
    assert all("ricciCurvature" not in d for _, _, d in G.edges(data=True))
    assert all("ricciCurvature" not in d for _, d in G.nodes(data=True))


# ── balanced Forman: Topping et al. (2022) Table 1 ───────────────────────────


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_balanced_forman_complete_graph(n):
    """K_n has Ric = n / (n - 1) on every edge."""
    kappa = gg.curvature(nx.complete_graph(n), method="balanced_forman")
    assert kappa
    for value in kappa.values():
        assert value == pytest.approx(n / (n - 1))


def test_balanced_forman_grid_interior_is_flat():
    """The 2-D grid is Ricci-flat away from the boundary."""
    G = nx.convert_node_labels_to_integers(nx.grid_2d_graph(7, 7))
    kappa = gg.curvature(G, method="balanced_forman")
    interior = [v for (u, w), v in kappa.items()
                if G.degree(u) == 4 and G.degree(w) == 4]
    assert len(interior) > 10
    for value in interior:
        assert value == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("branching", [2, 3, 4])
def test_balanced_forman_regular_tree(branching):
    """A tree whose interior degree is r+1 has Ric = 4/(r+1) - 2."""
    degree = branching + 1
    G = nx.balanced_tree(branching, 4)
    kappa = gg.curvature(G, method="balanced_forman")
    interior = [v for (u, w), v in kappa.items()
                if G.degree(u) == degree and G.degree(w) == degree]
    assert interior
    for value in interior:
        assert value == pytest.approx(4 / degree - 2)


def test_balanced_forman_leaf_edge_is_zero():
    """min(d_i, d_j) == 1 is defined to be 0, not 2/d_j."""
    kappa = gg.curvature(nx.path_graph(3), method="balanced_forman")
    assert all(v == 0.0 for v in kappa.values())


def _les_mis_core():
    G = nx.les_miserables_graph()
    G = nx.Graph(G.subgraph(sorted(G.nodes)[:25]))
    G.remove_edges_from(nx.selfloop_edges(G))
    G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", overwrite=True)
    return G


def test_balanced_forman_lower_bounds_lin_lu_yau():
    """Their Theorem 2, ``kappa(i,j) >= Ric(i,j)``.

    The ``kappa`` of that theorem is the **Lin-Lu-Yau** limit, not this
    package's ``ollivier``: their Table 1 gives ``Ric(K_n) = n/(n-1)``, which is
    exactly the LLY value of ``K_n``, so the bound is tight there. Naming is the
    trap -- the over-squashing literature writes "Ollivier curvature" for the
    limit curvature, while ``method="ollivier"`` here is the alpha-lazy one. The
    companion test below pins how far apart the two readings are.
    """
    G = _les_mis_core()
    bf = gg.curvature(G.copy(), method="balanced_forman")
    lly = gg.curvature(G.copy(), method="lin_lu_yau")
    for edge, value in bf.items():
        assert value <= lly[edge] + 1e-4, f"bound violated on {edge}"


def test_the_bound_is_against_lly_not_the_alpha_lazy_ollivier():
    """Reading ``kappa`` as ``method='ollivier'`` breaks the bound almost everywhere."""
    G = _les_mis_core()
    bf = gg.curvature(G.copy(), method="balanced_forman")
    lazy = gg.curvature(G.copy(), method="ollivier", alpha=0.0)
    violations = [e for e in bf if bf[e] > lazy[e] + 1e-4]
    assert len(violations) > len(bf) // 2


@pytest.mark.parametrize("n", [3, 4, 5])
def test_balanced_forman_is_tight_on_complete_graphs(n):
    """Ric(K_n) == kappa_LLY(K_n) == n/(n-1): the bound is an equality."""
    G = _undirected(n)
    bf = gg.curvature(G.copy(), method="balanced_forman")
    lly = gg.curvature(G.copy(), method="lin_lu_yau")
    for edge in bf:
        assert bf[edge] == pytest.approx(n / (n - 1))
        assert lly[edge] == pytest.approx(n / (n - 1), abs=1e-4)


# ── both are first-class: usable everywhere a built-in name is ───────────────


@pytest.mark.parametrize("method", _REGISTERED)
def test_plugin_honours_the_result_contract(method):
    G = _undirected(5)
    kappa = gg.curvature(G, method=method)

    assert set(kappa) == set(G.edges())
    for u, v in G.edges():
        assert "ricciCurvature" in G[u][v]
    for n in G.nodes():
        assert "ricciCurvature" in G.nodes[n]


@pytest.mark.parametrize(
    "name, output, match",
    [
        ("balanced_forman", "drops_an_edge", "exactly the graph's"),
        ("balanced_forman", "non_finite", "non-finite"),
        ("sinkhorn_ollivier", "drops_an_edge", "exactly the graph's"),
    ],
)
def test_malformed_plugin_output_is_still_rejected(monkeypatch, name, output, match):
    spec = gg.get_curvature_spec(name)
    honest = spec.compute_edges

    def malformed(G, **kwargs):
        values = dict(honest(G, **kwargs))
        edge = next(iter(values))
        if output == "drops_an_edge":
            del values[edge]
        else:
            values[edge] = float("nan")
        return values

    monkeypatch.setitem(gg.CURVATURE_REGISTRY, name,
                        dataclasses.replace(spec, compute_edges=malformed))
    with pytest.raises(gg.CurvatureContractError, match=match):
        gg.compute_curvature(_undirected(4), name)


def test_both_plugins_run_in_one_comparison():
    requests = [
        gg.CurvatureRequest("ollivier", {"alpha": 0.5}),
        gg.CurvatureRequest("sinkhorn_ollivier", {"alpha": 0.5, "reg": 0.1}),
        gg.CurvatureRequest("balanced_forman"),
    ]
    G = _undirected(6)
    comparison = gg.compare_curvature(G, requests=requests)
    rows = {row["method"]: row for row in comparison.summary()}
    for label in ("sinkhorn_ollivier", "balanced_forman"):
        assert rows[label]["failure_kind"] in ("", None)
        assert rows[label]["computable"] == rows[label]["of"] == G.number_of_edges()
    pairs = {frozenset((r["method_a"], r["method_b"])) for r in comparison.agreement()}
    assert frozenset({"sinkhorn_ollivier", "balanced_forman"}) in pairs


def test_a_comparison_records_balanced_forman_as_inapplicable_on_a_digraph():
    comparison = gg.compare_curvature(
        _cycle_with_chord(directed=True),
        requests=[gg.CurvatureRequest("sinkhorn_ollivier", {"kernel": "mixed", "reg": 0.1}),
                  gg.CurvatureRequest("balanced_forman")],
    )
    rows = {row["method"]: row for row in comparison.summary()}
    assert rows["balanced_forman"]["failure_kind"] == "inapplicable"
    assert rows["sinkhorn_ollivier"]["computable"] == rows["sinkhorn_ollivier"]["of"]


def test_balanced_forman_drives_flow_on_an_explicitly_undirected_graph():
    """A user plugin is usable as the curvature of the flow engine.

    It reads only the topology, so the flow -- which evolves weights -- warns
    that the curvature is invariant under it, and the spread never moves.
    """
    G = gg.load.from_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2)], directed=False)
    gg.apply_config(G, edge_weight="fixed", edge_distance="fixed")
    assert not G.is_directed()
    with pytest.warns(gg.CurvatureWarning, match="invariant under the flow"):
        result = gg.RicciFlow(G, curvature="balanced_forman").run(
            iterations=5, step=0.05, early_stop=False
        )
    assert result.resolved.method == "balanced_forman"
    assert result.iterations_completed == 5
    assert len(result.convergence) == 5
    assert len(set(result.convergence)) == 1


# ── experiments address a loaded plugin by name ─────────────────────────────


def _experiment(graph_kind, curvature):
    return {
        "data": {
            "graph_kind": graph_kind,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [2, 0]]
            if graph_kind == "directed" else [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            "factors": {"edge_weight": "fixed", "edge_distance": "fixed"},
        },
        "curvature": curvature,
        "flow": {"iterations": 2, "step": 0.05, "early_stop": False},
    }


@pytest.mark.filterwarnings("ignore::graph_geometry.CurvatureWarning")
@pytest.mark.parametrize(
    "graph_kind, curvature, parameters, convention",
    [
        ("undirected", {"method": "balanced_forman"}, {}, "undirected"),
        ("undirected", {"method": "sinkhorn_ollivier", "parameters": {"reg": 1.0}},
         {"alpha": 0.5, "reg": 1.0, "kernel": "undirected"}, "undirected"),
        ("directed",
         {"method": "sinkhorn_ollivier", "parameters": {"reg": 1.0, "kernel": "mixed", "beta": 0.6}},
         {"alpha": 0.5, "reg": 1.0, "kernel": "mixed", "beta": 0.6}, "mixed"),
    ],
    ids=["balanced-forman", "sinkhorn-undirected", "sinkhorn-directed"],
)
def test_plugin_runs_from_a_canonical_experiment_with_full_provenance(
    graph_kind, curvature, parameters, convention
):
    """Addressed only by name, once the plugin module is loaded in this process."""
    result = gg.run_experiment(_experiment(graph_kind, curvature))
    assert len(result["snapshots"]) == 3

    record = result["resolved"]
    assert record["data"]["graph_kind"] == graph_kind
    assert record["curvature"]["requested_method"] == curvature["method"]
    assert record["curvature"]["method"] == curvature["method"]
    assert record["curvature"]["graph_kind"] == graph_kind
    assert record["curvature"]["direction_convention"] == convention
    assert record["curvature"]["parameters"] == parameters
