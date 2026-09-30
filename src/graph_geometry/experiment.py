"""Config-driven experiment driver: ``run_experiment(config)``.

Reads a JSON or YAML config (or a dict) describing the graph, how its attributes
are interpreted, the curvature request, execution and flow parameters and
surgery; builds the graph, runs the flow, and (optionally) saves outputs
together with a fully resolved provenance record.

Canonical schema
----------------
    data:
      graph_kind: directed          # required: directed | undirected
      # graph source -- exactly one of:
      edges: [[0, 1], [1, 2], [2, 0]]
      generator: sbm                # + params: {...}
      dataset: mygraph              # + data_root (+ variant for mygraph)
      factors:
        edge_weight: fixed
        edge_distance: fixed
        node_factor: fixed
        seed: 0

    semantics:
      weight_attr: weight
      distance_attr: distance
      node_weight_attr: null
      beta_attr: beta               # read by beta_strategy: node_attr
      missing_weight: unit
      missing_distance: unit

    curvature:
      method: ollivier
      label: ORC mixed
      parameters:
        kernel: mixed
        beta_strategy: constant     # constant | degree_proportional | weight_proportional | node_attr
        beta: 0.8                   # only with beta_strategy: constant
        alpha: 0.5

    execution:
      proc: 1

    flow:
      equation: normalized          # name, or a math/LaTeX expression string
      evolve: weight                # weight | distance
      iterations: 50
      step: 0.01
      delta: 1.0e-6
      early_stop: true

    surgery:                        # optional
      name: surgery_n               # or no_surgery
      portion: 1
      interval: 30

    output:
      dir: results/run1             # optional

Persisted experiments do not default the graph direction or a directed
transport kernel: ``data.graph_kind`` is required and must agree with the graph
the source actually produces, and a directed Ollivier / Lin-Lu-Yau run must name
``out``, ``in`` or ``mixed``.

Legacy keys
-----------
The historical schema is read through a migration adapter, with a
:class:`FutureWarning` per legacy form: ``data.directed``; flat factor keys on
``data`` (including ``edge_weight_type``, ``edge_distance_type``,
``node_weight_type``); flat curvature parameters beside ``curvature.method``;
the ``ricciflow`` section and its ``flow_equation`` key; and ``surgery_param``.
A generator config without ``graph_kind`` takes the kind of the generated graph,
with a warning; an edge list or dataset without it is rejected.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .curvature.errors import CurvatureConfigurationError
from .curvature.evaluator import resolve_curvature
from .curvature.model import CurvatureRequest, graph_kind_of
from .curvature.registry import get_curvature_spec
from .graph import generators as generate
from .graph import loaders as load
from .graph.config import apply_config
from .flow.surgery import resolve_surgery
from .graph.semantics import GraphSemantics
from .io import SnapshotSeries
from .simulator import RicciFlowSimulator

__all__ = [
    "run_experiment",
    "load_config",
    "normalize_config",
    "RESOLVED_SEMANTICS_FILENAME",
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = 1
RESOLVED_SEMANTICS_FILENAME = "resolved_semantics.json"

_FACTOR_KEYS = ("edge_weight", "edge_distance", "node_factor", "seed")
# legacy factor-key aliases -> GraphConfig field names
_FACTOR_ALIASES = {
    "edge_weight_type": "edge_weight",
    "edge_distance_type": "edge_distance",
    "node_weight_type": "node_factor",
}
_FLOW_KEYS = ("equation", "evolve", "iterations", "step", "delta", "early_stop", "verbose")
_SOURCE_KEYS = ("edges", "generator", "dataset")


def load_config(config: Union[str, Path, dict]) -> dict:
    """Load a config from a path (``.json`` / ``.yaml`` / ``.yml``) or pass a dict through."""
    if isinstance(config, dict):
        return config
    path = Path(config)
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def _legacy(notes: list, message: str) -> None:
    notes.append(message)
    warnings.warn(f"experiment config: {message}", FutureWarning, stacklevel=4)


def normalize_config(config: Union[str, Path, dict]) -> tuple[dict, list[str]]:
    """Migrate a config to the canonical schema; return ``(canonical, notes)``.

    ``notes`` lists every legacy form that was read. Only structure is migrated
    here; graph kind agreement and curvature resolution need the built graph.
    """
    cfg = load_config(config) or {}
    notes: list[str] = []
    data = dict(cfg.get("data") or {})

    sources = [key for key in _SOURCE_KEYS if key in data]
    if len(sources) != 1:
        raise CurvatureConfigurationError(
            f"data section must have exactly one of {list(_SOURCE_KEYS)}; got {sources}."
        )

    graph_kind = data.get("graph_kind")
    if graph_kind is None and "directed" in data:
        directed = data["directed"]
        if not isinstance(directed, bool):
            raise CurvatureConfigurationError(f"data.directed must be a bool; got {directed!r}.")
        graph_kind = "directed" if directed else "undirected"
        _legacy(notes, f"data.directed is legacy; use data.graph_kind: {graph_kind}.")
    elif graph_kind is not None and "directed" in data:
        if data["directed"] != (graph_kind == "directed"):
            raise CurvatureConfigurationError(
                f"data.graph_kind={graph_kind!r} contradicts data.directed={data['directed']!r}."
            )
        _legacy(notes, "data.directed is legacy and redundant with data.graph_kind.")
    if graph_kind is not None and graph_kind not in ("directed", "undirected"):
        raise CurvatureConfigurationError(
            f"data.graph_kind must be 'directed' or 'undirected'; got {graph_kind!r}."
        )
    if graph_kind is None and sources[0] != "generator":
        raise CurvatureConfigurationError(
            f"data.graph_kind is required for a {sources[0]!r} source: a persisted "
            f"experiment must not default the graph direction."
        )

    factors = dict(data.get("factors") or {})
    unknown_factors = sorted(set(factors) - set(_FACTOR_KEYS))
    if unknown_factors:
        raise CurvatureConfigurationError(
            f"unknown data.factors key(s) {unknown_factors}; expected {list(_FACTOR_KEYS)}."
        )
    flat = {k: data[k] for k in _FACTOR_KEYS if k in data}
    flat.update({new: data[old] for old, new in _FACTOR_ALIASES.items()
                 if old in data and new not in data})
    if flat:
        _legacy(notes, f"flat factor keys {sorted(flat)} on data are legacy; use data.factors.")
        for key, value in flat.items():
            if key in factors and factors[key] != value:
                raise CurvatureConfigurationError(
                    f"data.{key} contradicts data.factors.{key}."
                )
            factors.setdefault(key, value)

    source = {key: data[key] for key in ("edges", "generator", "params", "dataset",
                                         "data_root", "variant") if key in data}
    canonical_data = {"graph_kind": graph_kind, **source, "factors": factors}

    semantics = GraphSemantics.from_dict(cfg.get("semantics"))

    curv = dict(cfg.get("curvature") or {})
    if "method" not in curv:
        raise CurvatureConfigurationError(
            "curvature.method is required: a persisted experiment must name its "
            "curvature definition."
        )
    parameters = dict(curv.get("parameters") or {})
    flat_params = {k: v for k, v in curv.items() if k not in ("method", "label", "parameters")}
    if flat_params:
        _legacy(notes, f"flat curvature parameters {sorted(flat_params)} are legacy; "
                       f"use curvature.parameters.")
        for key, value in flat_params.items():
            if key in parameters and parameters[key] != value:
                raise CurvatureConfigurationError(
                    f"curvature.{key} contradicts curvature.parameters.{key}."
                )
            parameters.setdefault(key, value)
    curvature = {"method": curv["method"], "label": curv.get("label"), "parameters": parameters}

    execution = dict(cfg.get("execution") or {})
    unknown_exec = sorted(set(execution) - {"proc"})
    if unknown_exec:
        raise CurvatureConfigurationError(f"unknown execution key(s) {unknown_exec}.")
    execution.setdefault("proc", 1)

    if "flow" in cfg and "ricciflow" in cfg:
        raise CurvatureConfigurationError("give either flow or the legacy ricciflow section.")
    flow_in = dict(cfg.get("flow") or cfg.get("ricciflow") or {})
    if "ricciflow" in cfg:
        _legacy(notes, "the ricciflow section is legacy; use flow.")
    if "flow_equation" in flow_in:
        if "equation" in flow_in:
            raise CurvatureConfigurationError("give either flow.equation or flow_equation.")
        flow_in["equation"] = flow_in.pop("flow_equation")
        _legacy(notes, "flow_equation is legacy; use flow.equation.")
    unknown_flow = sorted(set(flow_in) - set(_FLOW_KEYS))
    if unknown_flow:
        raise CurvatureConfigurationError(f"unknown flow key(s) {unknown_flow}.")
    flow = {
        "equation": flow_in.get("equation", "normalized"),
        "evolve": flow_in.get("evolve", "weight"),
        "iterations": flow_in.get("iterations", 50),
        "step": flow_in.get("step", 0.01),
        "delta": flow_in.get("delta", 1e-6),
        "early_stop": flow_in.get("early_stop", True),
        "verbose": flow_in.get("verbose", False),
    }

    if "surgery_param" in cfg:
        _legacy(notes, "surgery_param is legacy; use surgery.")
    surgery = _surgery_spec(cfg)

    canonical = {
        "schema_version": SCHEMA_VERSION,
        "data": canonical_data,
        "semantics": semantics.to_dict(),
        "curvature": curvature,
        "execution": execution,
        "flow": flow,
        "surgery": surgery,
        "output": dict(cfg.get("output") or {}),
    }
    return canonical, notes


def _build_graph(data: dict):
    """Construct the graph from the canonical ``data`` section."""
    kind = data["graph_kind"]
    if "edges" in data:
        return load.from_edges(
            [tuple(e) for e in data["edges"]], directed=kind == "directed"
        )
    if "generator" in data:
        fn = generate.GENERATORS[data["generator"]]
        return fn(**data.get("params", {}))
    root = data.get("data_root")
    if root is None:
        raise CurvatureConfigurationError("data.dataset requires data.data_root")
    return load.from_dataset(
        data["dataset"].lower(),
        root,
        directed=kind == "directed",
        variant=data.get("variant", "3cycle4edge"),
    )


def _surgery_spec(config: dict) -> Optional[dict]:
    """Resolve the surgery section (new ``surgery`` or legacy ``surgery_param``)."""
    spec = config.get("surgery", config.get("surgery_param"))
    if spec is None:
        return None
    # legacy on/off switch
    if spec.get("surgery") is False or spec.get("name", "no_surgery") == "no_surgery":
        return None
    # defaults are name-specific (a fraction for 'surgery', a count for
    # 'surgery_n') and live in one place, resolve_surgery
    resolved = {"name": spec["name"]}
    for key in ("portion", "interval"):
        if key in spec:
            resolved[key] = spec[key]
    strategy = resolve_surgery(resolved)      # validates now, not mid-run
    return {"name": spec["name"], "portion": strategy.portion, "interval": strategy.interval}


def _reject_implicit_conventions(resolved, request: CurvatureRequest) -> None:
    """A persisted directed run must name every convention an ``auto`` would pick."""
    if resolved.graph_kind != "directed":
        return
    spec = get_curvature_spec(resolved.method)
    for p in spec.parameters:
        if "auto" not in p.choices:
            continue
        if request.parameters.get(p.name, p.default) == "auto":
            explicit = [c for c in p.choices if c not in ("auto", "undirected")]
            raise CurvatureConfigurationError(
                f"curvature.parameters.{p.name} must be explicit for a directed "
                f"graph in a persisted experiment (one of {explicit}); "
                f"{p.name!r}='auto' would resolve silently to "
                f"{resolved.parameters.get(p.name)!r}."
            )


@dataclass
class _Run:
    """Everything one experiment produced, before the provenance record is built."""

    canonical: dict
    notes: list
    graph: Any
    semantics: GraphSemantics
    resolved: Any
    simulator: RicciFlowSimulator


def run_experiment(config: Union[str, Path, dict]) -> dict:
    """Run a full experiment from a config; return a result dict.

    Returns ``{"simulator", "series", "snapshots", "convergence", "result",
    "config", "canonical_config", "resolved"}``. ``config`` is the original
    input; ``resolved`` is the JSON-safe provenance record (canonical method,
    requested alias, graph kind, direction convention, effective parameters,
    attribute roles, execution, flow and every warning). With ``output.dir``,
    GEXF snapshots, ``convergence.csv``, ``flow.json`` and
    ``resolved_semantics.json`` are written there.
    """
    original = load_config(config)
    caught: list = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = _run(original)
    finally:
        # Recorded so they can go into the provenance record, then re-emitted so
        # a caller still sees them -- also when the run itself failed.
        for warning in caught:
            warnings.warn_explicit(
                warning.message, warning.category, warning.filename, warning.lineno
            )
    run_warnings = list(dict.fromkeys(str(w.message) for w in caught))
    record = _provenance(run, run_warnings)
    sim = run.simulator
    series = SnapshotSeries.from_result(sim.result, weight=sim.weight)

    out = run.canonical["output"]
    if out.get("dir"):
        out_dir = Path(out["dir"])
        series.save(str(out_dir))
        (out_dir / RESOLVED_SEMANTICS_FILENAME).write_text(json.dumps(record, indent=2) + "\n")

    return {
        "simulator": sim,
        "series": series,
        "snapshots": sim.snapshots,
        "convergence": sim.convergence,
        "result": sim.result,
        "config": original,
        "canonical_config": run.canonical,
        "resolved": record,
    }


def _run(original: dict) -> _Run:
    """Migrate, build, resolve and run: every step that may warn."""
    canonical, notes = normalize_config(original)
    data = canonical["data"]

    # 1. graph + factors; the source must produce the declared kind
    G = _build_graph(data)
    actual = graph_kind_of(G)
    if data["graph_kind"] is None:
        data["graph_kind"] = actual
        message = (f"data.graph_kind is missing; the generator produced a "
                   f"{actual} graph. Record data.graph_kind: {actual}.")
        notes.append(message)
        warnings.warn(f"experiment config: {message}", FutureWarning, stacklevel=2)
    elif data["graph_kind"] != actual:
        source = "generator" if "generator" in data else "source"
        article = "an" if actual == "undirected" else "a"
        raise CurvatureConfigurationError(
            f"data.graph_kind={data['graph_kind']!r} but the {source} "
            f"produced {article} {actual} graph."
        )
    semantics = GraphSemantics.from_dict(canonical["semantics"])
    apply_config(
        G, **data["factors"],
        weight_attr=semantics.weight_attr, distance_attr=semantics.distance_attr,
    )

    # 2. resolve the curvature once, before any numerical work
    curv = canonical["curvature"]
    request = CurvatureRequest(curv["method"], curv["parameters"], label=curv["label"])
    resolved = resolve_curvature(G, request, semantics=semantics)
    _reject_implicit_conventions(resolved, request)

    # 3. run
    flow = canonical["flow"]
    sim = RicciFlowSimulator(
        G,
        curvature=request,
        flow_equation=flow["equation"],
        evolve=flow["evolve"],
        proc=canonical["execution"]["proc"],
        semantics=semantics,
    )
    sim.run(
        iterations=flow["iterations"],
        step=flow["step"],
        delta=flow["delta"],
        early_stop=flow["early_stop"],
        surgery=canonical["surgery"],
        verbose=flow["verbose"],
    )
    return _Run(canonical, notes, G, semantics, resolved, sim)


def _provenance(run: _Run, run_warnings: list) -> dict:
    """The JSON-safe record of what ran."""
    canonical, result = run.canonical, run.simulator.result
    data, flow = canonical["data"], canonical["flow"]
    return {
        "schema_version": SCHEMA_VERSION,
        "data": {
            "graph_kind": data["graph_kind"],
            "source": {k: v for k, v in data.items() if k not in ("graph_kind", "factors")},
            "factors": data["factors"],
            "nodes": run.graph.number_of_nodes(),
            "edges": run.graph.number_of_edges(),
        },
        "semantics": run.semantics.to_dict(),
        "curvature": (result.resolved or run.resolved).to_dict(),
        "execution": dict(canonical["execution"]),
        "flow": {
            **{k: v for k, v in flow.items() if k != "verbose"},
            "equation": flow["equation"] if isinstance(flow["equation"], str)
            else repr(flow["equation"]),
            "termination_reason": result.termination_reason,
            "iterations_completed": result.iterations_completed,
        },
        "surgery": canonical["surgery"],
        "legacy_forms": run.notes,
        "warnings": run_warnings,
    }
