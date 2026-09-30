"""The Streamlit UI is generated from the registry, not from a hard-coded list.

These run the real ``app.py`` headlessly through Streamlit's ``AppTest``. The
claim under test is the one the plugin architecture exists to support: a
curvature registered by a user appears in the interface, with its own
parameters, without any edit to ``app.py``.

Skipped unless the ``[app]`` extra is installed.
"""

import pytest

pytest.importorskip("streamlit", reason="needs the [app] extra")

from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

import graph_geometry as gg  # noqa: E402
from graph_geometry.curvature import write_curvature  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app.py"


def _run():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def _curvature_select(at):
    """The curvature selectbox is the one whose options are CurvatureSpecs."""
    for sb in at.selectbox:
        if sb.label == "Curvature":
            return sb
    raise AssertionError("no curvature selectbox found")


def _labels(at):
    """``AppTest`` reports selectbox options already run through ``format_func``."""
    return list(_curvature_select(at).options)


def _by_label(elements, label):
    """Return the only AppTest element carrying ``label``."""
    matches = [element for element in elements if element.label == label]
    assert len(matches) == 1, (label, [element.label for element in elements])
    return matches[0]


def _label_of(name):
    spec = gg.get_curvature_spec(name)
    return spec.label or spec.name


# ── the registry drives the selector ────────────────────────────────────────


def test_every_registered_curvature_is_offered():
    offered = _labels(_run())
    assert len(offered) == len(gg.CURVATURE_REGISTRY)
    for spec in gg.CURVATURE_REGISTRY.values():
        assert (spec.label or spec.name) in offered


def test_combinatorial_and_directed_methods_are_reachable():
    """The combinatorial and directed definitions were unreachable before."""
    offered = set(_labels(_run()))
    for name in ("lin_lu_yau", "ollivier", "eidi_jost",
                 "forman_node_weighted", "augmented_forman_node_weighted",
                 "forman_directed", "augmented_forman_directed"):
        assert _label_of(name) in offered


def test_a_user_plugin_appears_without_touching_app_py():
    """Register a curvature, re-run the app, find it in the dropdown."""
    with pytest.warns(gg.CurvatureDeprecationWarning, match="without capabilities"):
        @gg.register_curvature("paper_demo_kappa", label="Paper demo κ",
                               color="#123456", description="Registered by a test.")
        def _demo(G, *, weight="weight", distance="distance", proc=1, scale=2.0, **kw):
            return write_curvature(G, {(u, v): scale for u, v in G.edges()})

    try:
        at = _run()
        assert "Paper demo κ" in _labels(at)
    finally:
        gg.unregister_curvature("paper_demo_kappa")


# ── parameter widgets come from the compute signature ───────────────────────


def test_selecting_ollivier_exposes_its_alpha_parameter():
    at = _run()
    _curvature_select(at).set_value(gg.CURVATURE_REGISTRY["ollivier"]).run()
    assert not at.exception, at.exception
    assert any(s.label == "alpha" for s in at.slider), \
        [s.label for s in at.slider]


def test_a_plugins_own_parameter_becomes_a_widget():
    """``scale=3.5`` is declared only by the plugin; the UI has never seen it."""
    with pytest.warns(gg.CurvatureDeprecationWarning):
        @gg.register_curvature("paper_demo_kappa2", label="Paper demo κ2",
                               description="Registered by a test.")
        def _demo(G, *, weight="weight", distance="distance", proc=1, scale=3.5, **kw):
            return write_curvature(G, {(u, v): scale for u, v in G.edges()})

    try:
        at = _run()
        _curvature_select(at).set_value(
            gg.CURVATURE_REGISTRY["paper_demo_kappa2"]).run()
        assert not at.exception, at.exception
        widgets = [w.label for w in at.number_input] + [w.label for w in at.slider]
        assert "scale" in widgets, widgets
    finally:
        gg.unregister_curvature("paper_demo_kappa2")


def test_a_declared_parameter_schema_drives_the_widget_and_its_bounds():
    """A metadata plugin's ParameterSpec -- not its signature -- makes the control."""
    @gg.register_curvature(
        "paper_demo_kappa3",
        capabilities=gg.CurvatureCapabilities(
            family="combinatorial", status="experimental",
            graph_kinds=frozenset({"directed", "undirected"}),
            scopes=frozenset({"edges"}),
            consumes=frozenset({"topology"}),
        ),
        parameters=(gg.ParameterSpec("gain", "float", 0.25, minimum=0.0, maximum=1.0),),
        label="Paper demo κ3",
    )
    def _demo(G, *, semantics, proc, gain):
        return {(u, v): gain for u, v in G.edges()}

    try:
        at = _run()
        _curvature_select(at).set_value(
            gg.CURVATURE_REGISTRY["paper_demo_kappa3"]).run()
        assert not at.exception, at.exception
        gain = [s for s in at.slider if s.label == "gain"]
        assert gain and gain[0].value == pytest.approx(0.25)
    finally:
        gg.unregister_curvature("paper_demo_kappa3")


def test_forman_exposes_no_ot_parameters():
    """A combinatorial method declares no alpha/kernel, so none are rendered."""
    at = _run()
    _curvature_select(at).set_value(gg.CURVATURE_REGISTRY["forman_directed"]).run()
    assert not at.exception, at.exception
    assert not any(s.label == "alpha" for s in at.slider)


# ── the plugin loader ───────────────────────────────────────────────────────


def test_plugin_file_loader_registers_and_offers_the_method():
    at = _run()
    path = str(APP.parent / "examples" / "plugin_balanced_forman.py")
    try:
        at.text_input(key="plugin_path").set_value(path).run()
        assert not at.exception, at.exception
        assert "balanced_forman" in gg.CURVATURE_REGISTRY
        assert _label_of("balanced_forman") in _labels(at)
        # Streamlit re-runs the script on every interaction; re-loading the same
        # file must not trip the registry's duplicate-name check.
        at.run()
        assert not at.exception, at.exception
        assert not [e for e in at.error if "Could not load plugin" in e.value]
    finally:
        gg.unregister_curvature("balanced_forman")


def test_plugin_loader_reports_a_bad_path_instead_of_crashing():
    at = _run()
    at.text_input(key="plugin_path").set_value("/definitely/not/here.py").run()
    assert not at.exception, at.exception
    assert any("Could not load plugin" in e.value for e in at.error)


# ── comparison view ─────────────────────────────────────────────────────────


def test_comparison_offers_every_registered_method():
    at = _run()
    options = set(at.multiselect(key="cmp_methods").options)
    assert options == {_label_of(n) for n in gg.CURVATURE_REGISTRY}


def test_comparison_runs_and_produces_a_downloadable_table():
    at = _run()
    names = ["forman_directed", "augmented_forman_directed"]
    at.multiselect(key="cmp_methods").set_value(names).run()
    at.button(key="cmp_run").click().run()
    assert not at.exception, at.exception

    cmp = at.session_state["cmp"]
    assert {r["method"] for r in cmp.summary()} == set(names)
    assert cmp.csv_text().splitlines()[0] == "source,target," + ",".join(names)
    headings = [m.value for m in at.markdown]
    assert "**Applicability and distribution**" in headings
    assert "**Coverage and distribution**" not in headings


def test_comparison_survives_a_method_that_cannot_run_here():
    """The undirected default graph has no OT problem; force one via edge input."""
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    at.radio(key="graph_source").set_value("Enter edges").run()
    at.text_area(key="edge_text").set_value("0 1\n1 2\n2 0\n0 3\n1 3\n2 4").run()
    at.multiselect(key="cmp_methods").set_value(["lin_lu_yau", "eidi_jost"]).run()
    at.button(key="cmp_run").click().run()
    assert not at.exception, at.exception

    cmp = at.session_state["cmp"]
    rows = {r["method"]: r for r in cmp.summary()}
    assert rows["lin_lu_yau"]["computable"] == 0
    assert rows["lin_lu_yau"]["error"]
    assert rows["lin_lu_yau"]["failure_kind"] == "inapplicable"
    assert rows["eidi_jost"]["computable"] == 6


def test_edge_input_accepts_weight_distance_and_node_values():
    at = _run()
    at.radio(key="graph_source").set_value("Enter edges").run()
    at.text_area(key="edge_text").set_value(
        "N 0 2.5\n0 1 0.5 2.0\n1 0 0.8 3.0"
    ).run()
    assert not at.exception, at.exception
    assert not at.error


def test_upload_uses_the_package_multicolumn_parser():
    """A numeric weight/distance upload must not go through nx.read_edgelist."""
    at = _run()
    at.radio(key="graph_source").set_value("Upload file").run()
    uploader = at.get("file_uploader")[0]
    uploader.upload(
        "weighted.edgelist",
        b"0 1 0.5 2.0\n1 0 0.8 3.0\n",
        "text/plain",
    ).run()
    assert not at.exception, at.exception
    assert not at.error

    _by_label(at.number_input, "Iterations").set_value(1).run()
    at.button(key="run_flow").click().run()
    assert not at.exception, at.exception
    G = at.session_state["sim"]._G0
    assert G["0"]["1"]["weight"] == 0.5
    assert G["0"]["1"]["distance"] == 2.0
    assert G["1"]["0"]["weight"] == 0.8
    assert G["1"]["0"]["distance"] == 3.0


def test_changing_the_graph_invalidates_simulation_and_comparison_results():
    at = _run()
    _by_label(at.number_input, "Iterations").set_value(1).run()
    at.button(key="run_flow").click().run()
    assert at.session_state["done"] is True
    assert at.session_state["sim"] is not None

    at.multiselect(key="cmp_methods").set_value(
        ["forman_directed", "augmented_forman_directed"]).run()
    at.button(key="cmp_run").click().run()
    assert at.session_state["cmp"] is not None

    _by_label(at.selectbox, "Graph").set_value("── Complete K₅").run()
    assert not at.exception, at.exception
    assert at.session_state["done"] is False
    assert at.session_state["sim"] is None
    assert at.session_state["cmp"] is None


def test_changing_a_flow_parameter_invalidates_only_the_simulation():
    at = _run()
    _by_label(at.number_input, "Iterations").set_value(1).run()
    at.button(key="run_flow").click().run()
    assert at.session_state["done"] is True

    _by_label(at.number_input, "Step").set_value(0.1).run()
    assert not at.exception, at.exception
    assert at.session_state["done"] is False
    assert at.session_state["sim"] is None


@pytest.mark.parametrize("interval", [1, 2])
def test_exhausted_final_snapshot_renders_without_a_convergence_value(interval):
    at = _run()
    _by_label(at.selectbox, "Graph").set_value("🔀 Two K₄ + bridge (dir.)").run()
    # With interval 2 the history contains a nonempty state's spread, which
    # must not be displayed as the final empty graph's spread.
    _by_label(at.number_input, "Iterations").set_value(2).run()
    _by_label(at.selectbox, "Type").set_value("surgery_n").run()
    _by_label(at.number_input, "Cut N edges").set_value(26).run()
    _by_label(at.slider, "Every N iterations").set_value(interval).run()

    at.button(key="run_flow").click().run()
    assert not at.exception, at.exception
    sim = at.session_state["sim"]
    assert sim.termination_reason == "exhausted"
    assert sim.result_graph.number_of_edges() == 0
    assert len(sim.convergence) == len(sim.snapshots) - 2
    assert _by_label(at.metric, "Final RC diff").value == "undefined"

    _by_label(at.slider, "Iteration").set_value(len(sim.snapshots) - 1).run()
    assert not at.exception, at.exception
    assert any(metric.label == "RC diff" and metric.value == "undefined"
               for metric in at.metric)


def test_formula_error_keeps_the_trajectory_visible_in_the_interface():
    at = _run()
    _by_label(at.selectbox, "Graph").set_value("── Two K₄ + bridge").run()
    _curvature_select(at).set_value(gg.CURVATURE_REGISTRY["ollivier"]).run()
    _by_label(at.selectbox, "Flow equation").set_value("Custom (expression)").run()
    _by_label(at.text_input, "Flow expression").set_value("eta*exp(q)").run()
    _by_label(at.number_input, "Step").set_value(1.0).run()
    at.button(key="run_flow").click().run()

    assert not at.exception, at.exception
    sim = at.session_state["sim"]
    assert at.session_state["done"] is True
    assert sim.termination_reason == "diverged"
    assert sim.result.iterations_completed == 3
    assert any(sim.result.diagnosis == block.value for block in at.code)
    assert "OverflowError" in sim.result.diagnosis
    values = [d["ricciCurvature"] for _, _, d in sim.result_graph.edges(data=True)]
    assert _by_label(at.metric, "Final RC diff").value == f"{max(values) - min(values):.2e}"


def test_scientific_notation_and_log10_run_from_the_interface():
    at = _run()
    _by_label(at.selectbox, "Flow equation").set_value("Custom (expression)").run()
    _by_label(at.text_input, "Flow expression").set_value("-1e-3*log10(q+1)").run()
    at.button(key="run_flow").click().run()
    assert not at.exception, at.exception
    assert at.session_state["done"] is True
    assert at.session_state["sim"].result.iterations_completed >= 1


# ── computing curvature in the UI, without running a flow ───────────────────
# The UI used to expose curvature only as a by-product of a flow run or inside
# the comparison expander. These pin the standalone workflow, including the
# edges/all-pairs distinction that separates the transport family from the
# combinatorial one.


def _curvature_section(at):
    return at.expander[0]


def test_the_curvature_section_offers_every_registered_definition():
    at = _run()
    assert not at.exception
    names = [spec.label or name for name, spec in gg.CURVATURE_REGISTRY.items()]
    assert at.selectbox(key="cur_name").options == names


def test_computing_on_edges_produces_values_a_table_and_a_download():
    at = _run()
    at = at.button(key="cur_run").click().run()
    assert not at.exception
    counts = {m.label: m.value for m in at.metric}
    assert int(counts["count"].replace(",", "")) > 0
    assert at.dataframe, "per-value table missing"
    assert any("curvature CSV" in b.label for b in at.download_button)


def test_all_pairs_is_offered_for_a_transport_definition():
    at = _run()
    at.selectbox(key="cur_name").set_value("Ollivier").run()
    scope = at.radio(key="cur_scope")
    assert not scope.disabled
    at.radio(key="cur_scope").set_value("All node pairs").run()
    at = at.button(key="cur_run").click().run()
    assert not at.exception
    rows = at.dataframe[0].value
    assert len(rows) > 0


def test_all_pairs_is_disabled_for_a_combinatorial_definition():
    """Forman has no value off an edge, so the UI must not offer to compute one."""
    at = _run()
    at.selectbox(key="cur_name").set_value("Forman-Ricci (directed)").run()
    assert at.radio(key="cur_scope").disabled
    # and computing on edges still works
    at = at.button(key="cur_run").click().run()
    assert not at.exception
    assert {m.label for m in at.metric} >= {"count", "min", "max", "mean"}


def test_curvature_widgets_do_not_collide_with_the_flow_widgets():
    """Both sections build parameter widgets from the same registry spec.

    They shared a key prefix at first, which made Streamlit raise
    StreamlitDuplicateElementKey and blanked the whole page.
    """
    at = _run()
    assert not at.exception
    at.selectbox(key="cur_name").set_value("Lin-Lu-Yau").run()
    assert not at.exception


def test_both_sections_expose_a_process_count():
    """`proc` was accepted by the engine and the simulator but had no control."""
    at = _run()
    keys = {n.key for n in at.number_input}
    assert "cur_proc" in keys, "curvature section has no Processes control"
    assert "flow_proc" in keys, "flow section has no Processes control"


def test_flow_results_expose_snapshot_aligned_edge_trajectories():
    at = _run()
    _by_label(at.number_input, "Iterations").set_value(2).run()
    at = at.button(key="run_flow").click().run()
    assert not at.exception, at.exception

    selector = at.multiselect(key="trajectory_edges")
    sim = at.session_state["sim"]
    assert selector.options
    assert len(selector.options) == sim.initial_graph.number_of_edges()
    assert at.segmented_control(key="trajectory_quantity").value == "Both"
    assert at.download_button(key="trajectory_download")


def _run_flow_on(at, edge_text, iterations=2, directed=True):
    at.radio(key="graph_source").set_value("Enter edges").run()
    at.text_area(key="edge_text").set_value(edge_text).run()
    if not directed:
        _by_label(at.checkbox, "Directed").uncheck().run()
    _by_label(at.number_input, "Iterations").set_value(iterations).run()
    at = at.button(key="run_flow").click().run()
    assert not at.exception, at.exception
    return at


def test_trajectory_view_labels_the_evolving_role_and_uses_the_package_export():
    at = _run()
    _by_label(at.selectbox, "Flow mode").set_value("Evolve distance, weight fixed").run()
    at = _run_flow_on(at, "0 1\n1 2\n2 0\n1 0\n2 1\n0 2")
    assert "Edge distance" in at.segmented_control(key="trajectory_quantity").options
    first_edge = next(iter(at.session_state["sim"].initial_graph.edges()))
    at.multiselect(key="trajectory_highlight").set_value([first_edge]).run()
    at.checkbox(key="trajectory_aggregate").check().run()
    assert not at.exception, at.exception
    assert at.download_button(key="trajectory_download")


def test_trajectory_view_does_not_draw_hundreds_of_lines_by_default():
    import app  # noqa: F401  (only for the threshold constant)

    edges = "\n".join(f"{i} {j}" for i in range(12) for j in range(12) if i < j)
    at = _run_flow_on(_run(), edges, iterations=1, directed=False)
    assert at.session_state["sim"].initial_graph.number_of_edges() > app.TRAJECTORY_EDGE_LIMIT
    assert at.radio(key="trajectory_scope").value == "Selected edges"

    at.radio(key="trajectory_scope").set_value("All edges").run()
    assert not at.exception, at.exception
    assert any("Individual lines are hidden" in info.value for info in at.info)

    at.checkbox(key="trajectory_aggregate").check().run()
    assert not at.exception, at.exception

    at.radio(key="trajectory_scope").set_value("Random sample").run()
    assert not at.exception, at.exception
    assert not any("Individual lines are hidden" in info.value for info in at.info)


def test_curvature_computes_with_more_than_one_process():
    at = _run()
    at.number_input(key="cur_proc").set_value(2).run()
    at = at.button(key="cur_run").click().run()
    assert not at.exception
    assert int({m.label: m.value for m in at.metric}["count"].replace(",", "")) > 0


# ── uploading a real tab-separated network ──────────────────────────────────


def _ecoli_like():
    """The shape that failed: tabs, a node id with a space, a `?` annotation."""
    return (
        "AcrR\tacrA\t\n"
        "ArcA\tPhantom Gene\t\n"
        "CRP\tlyx\t?\n"
        "CRP\tacrA\t\n"
    )


def test_tab_separated_upload_with_spaced_ids_loads():
    at = _run()
    at.radio(key="graph_source").set_value("Enter edges").run()
    at = at.text_area(key="edge_text").set_value(_ecoli_like()).run()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]


def test_a_non_numeric_annotation_column_is_reported_not_fatal():
    at = _run()
    at.radio(key="graph_source").set_value("Enter edges").run()
    at = at.text_area(key="edge_text").set_value(_ecoli_like()).run()
    assert not at.error
    assert any("non-numeric" in w.value for w in at.warning), \
        "the dropped annotation column must be reported"


# ── metadata-driven applicability, defaults and cache signatures ────────────


def _directed_edges(at, text="0 1\n1 2\n2 0\n1 0\n2 1\n0 2"):
    at.radio(key="graph_source").set_value("Enter edges").run()
    at.text_area(key="edge_text").set_value(text).run()
    assert not at.exception, at.exception
    return at


def test_beta_default_comes_from_the_schema():
    """One beta default, 0.8, shared by the evaluator, the kernel and the UI."""
    at = _directed_edges(_run())
    at.selectbox(key="cur_name").set_value("Ollivier").run()
    at.selectbox(key="curv_ollivier_kernel").set_value("mixed").run()
    assert not at.exception, at.exception
    beta = at.slider(key="curv_ollivier_beta")
    assert beta.value == pytest.approx(0.8)
    assert beta.value == gg.get_curvature_spec("ollivier").parameter("beta").default


def test_beta_strategies_are_offered_from_the_schema_and_hide_beta():
    """β strategies are a declarative choice again, not a bespoke widget."""
    at = _directed_edges(_run())
    at.selectbox(key="cur_name").set_value("Ollivier").run()
    at.selectbox(key="curv_ollivier_kernel").set_value("mixed").run()
    strategy = at.selectbox(key="curv_ollivier_beta_strategy")
    assert strategy.options == ["constant", "degree_proportional",
                                "weight_proportional", "node_attr"]
    assert strategy.value == "constant"
    assert "curv_ollivier_beta" in {s.key for s in at.slider}

    strategy.set_value("degree_proportional").run()
    assert not at.exception, at.exception
    assert "curv_ollivier_beta" not in {s.key for s in at.slider}
    at.button(key="cur_run").click().run()
    assert not at.exception, at.exception
    assert at.session_state["cur_resolved"]["parameters"]["beta_strategy"] == "degree_proportional"


def _undirected(at):
    _by_label(at.selectbox, "Graph").set_value("── 6-Cycle").run()
    assert not at.exception, at.exception
    return at


def test_beta_is_not_offered_where_it_is_inactive():
    at = _undirected(_run())
    at.selectbox(key="cur_name").set_value("Ollivier").run()
    keys = {s.key for s in at.slider}
    assert "curv_ollivier_beta" not in keys


def test_an_inapplicable_definition_is_disabled_with_its_reason():
    """Eidi-Jost on an undirected graph: no silent symmetrisation."""
    at = _undirected(_run())
    at.selectbox(key="cur_name").set_value("Eidi-Jost (in-out)").run()
    assert not at.exception, at.exception
    assert at.button(key="cur_run").disabled
    assert any("directed graphs" in w.value for w in at.warning)


def test_changing_a_curvature_parameter_invalidates_the_standalone_result():
    at = _run()
    at.selectbox(key="cur_name").set_value("Ollivier").run()
    at.button(key="cur_run").click().run()
    assert at.session_state["cur_values"] is not None
    assert at.session_state["cur_resolved"]["method"] == "ollivier"

    at.slider(key="curv_ollivier_alpha").set_value(0.25).run()
    assert not at.exception, at.exception
    assert at.session_state["cur_values"] is None


def test_changing_the_process_count_invalidates_standalone_and_flow_results():
    at = _run()
    at.button(key="cur_run").click().run()
    assert at.session_state["cur_values"] is not None
    at.number_input(key="cur_proc").set_value(2).run()
    assert at.session_state["cur_values"] is None

    _by_label(at.number_input, "Iterations").set_value(1).run()
    at.button(key="run_flow").click().run()
    assert at.session_state["sim"] is not None
    at.number_input(key="flow_proc").set_value(2).run()
    assert at.session_state["sim"] is None


def test_flow_run_records_the_resolved_configuration():
    at = _run()
    _by_label(at.number_input, "Iterations").set_value(1).run()
    at.button(key="run_flow").click().run()
    assert not at.exception, at.exception
    resolved = at.session_state["sim"].resolved
    assert resolved is not None and resolved.method == "lin_lu_yau"
    assert resolved.direction_convention in ("undirected", "out", "in", "mixed")


# ── flow-wise comparison in the interface ───────────────────────────────────


def test_flow_equations_can_be_compared_in_the_interface():
    at = _run()
    _by_label(at.number_input, "Iterations").set_value(3).run()
    at.multiselect(key="fcmp_equations").set_value(["normalized", "additive"]).run()
    at.text_input(key="fcmp_expression").set_value("-eta*tanh(kappa - kbar)*w").run()
    at.button(key="fcmp_run").click().run()
    assert not at.exception, at.exception
    fcmp = at.session_state["fcmp"]
    assert fcmp.labels == ["normalized", "additive", "-eta*tanh(kappa - kbar)*w"]
    assert all(row["failure_kind"] == "" for row in fcmp.summary())
    assert at.download_button(key="fcmp_dl")

    _by_label(at.number_input, "Step").set_value(0.02).run()   # a shared setting changed
    assert at.session_state["fcmp"] is None
