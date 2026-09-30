# graph_geometry

Discrete Ricci curvature and Ricci flow on graphs, with curvature definitions
and flow equations as plugins.

Every curvature is registered under a name with its capabilities (graph kinds,
scopes, the attributes it reads) and a parameter schema. A request is resolved
and recorded before anything is computed, so the same definition behaves
identically from the Python API, a YAML experiment file and the Streamlit
interface, and every result says exactly which convention produced it.

- **Curvature** — Lin–Lu–Yau, Ollivier, Eidi–Jost, and four Forman variants,
  on undirected and directed weighted graphs; edges or arbitrary node pairs;
  parallel optimal transport (`proc > 1`).
- **Ricci flow** — normalized, unnormalized, additive or user-defined equations
  (callable or typed formula), surgery strategies, per-iteration snapshots and
  per-edge trajectories.
- **Comparison** — run several definitions or flow equations on one graph and
  get coverage, typed failures and pairwise agreement.
- **Reproducibility** — declarative YAML/JSON experiments that write the
  resolved conventions next to the results.
- **Interface** — a Streamlit app generated from the same registry metadata.

## Installation

Requires Python ≥ 3.10. From a clone, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/laiguzi/graph-geometry.git
cd graph-geometry
uv sync                   # core: networkx, numpy, POT, CVXPY, PyYAML
uv sync --extra viz       # + matplotlib, for graph_geometry.viz
uv sync --extra app       # + Streamlit, for the interface
```

or with pip:

```bash
pip install "graph-geometry[viz] @ git+https://github.com/laiguzi/graph-geometry.git"
```

The installed wheel contains the library only. `app.py`, `examples/`,
`configs/` and `data/` are in the source repository.

## Quick start

```python
import graph_geometry as gg

G = gg.load.from_edges([(0, 1), (1, 2), (2, 0), (0, 2)], directed=True)
gg.apply_config(G, edge_weight="fixed", edge_distance="fixed", node_factor="degree")

# Curvature: a request is resolved, validated and evaluated.
# G is left untouched unless annotate=True.
request = gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed", "beta": 0.8})
result = gg.compute_curvature(G, request)
result.values               # {(0, 1): 1.2, (0, 2): 1.6, (1, 2): 0.4, (2, 0): 1.3}
result.resolved.to_dict()   # method, graph kind, direction convention, parameters

gg.compute_curvature(G, "forman_directed", annotate=True)   # writes ricciCurvature
gg.list_curvatures(graph_kind="directed", scope="pairs")    # capability query

# Ricci flow
sim = gg.RicciFlowSimulator.from_networkx(G, curvature=request)
sim.run(iterations=50, step=0.05)
sim.snapshots       # one graph per committed state
sim.convergence     # curvature spread (max − min) per state
sim.summary()
```

A runnable version is [`examples/quickstart.py`](examples/quickstart.py).

## Built-in curvatures

| name | graph kinds | scope | direction convention |
|---|---|---|---|
| `lin_lu_yau` | directed, undirected | edges, pairs | `undirected` or `out` / `in` / `mixed` |
| `ollivier` | directed, undirected | edges, pairs | `undirected` or `out` / `in` / `mixed` |
| `eidi_jost` | directed | edges | `in_out` |
| `forman_node_weighted` | undirected | edges | `undirected_incidence` |
| `augmented_forman_node_weighted` | undirected | edges | `undirected_incidence_triangles` |
| `forman_directed` | directed | edges | `in_tail_out_head` |
| `augmented_forman_directed` | directed | edges | `in_tail_out_head_feed_forward_faces` |

No layer symmetrises a digraph, reverses edges or reads a weight as a distance
implicitly. A definition applied outside its domain raises
`CurvatureDomainError`. The one convenience, `kernel="auto"` on a digraph,
resolves to `mixed` with β = 0.8, warns, and is recorded as such.

**Directed graphs.** The `mixed` kernel is `β·P_out + (1−β)·P_in`
([Bai, Li, Liu & Lai, arXiv:2509.19989](https://arxiv.org/abs/2509.19989)).
β is chosen by `beta_strategy`: `constant`, `degree_proportional`,
`weight_proportional`, or `node_attr` (read from each node). β lies in [0, 1]
and is clamped to 0 at sinks and 1 at sources, so every measure has mass 1.
`lin_lu_yau` and `ollivier` require finite directed distances for the evaluated
endpoint pairs and the support pairs used by the selected kernels. Strong
connectivity is sufficient, but some requests are also defined on digraphs
that are not strongly connected; unreachable required pairs raise
`CurvatureDomainError`. `eidi_jost`
([Eidi & Jost 2020](https://doi.org/10.1038/s41598-020-68619-6)) reads measures
from in-neighbours of the tail and out-neighbours of the head, so its required
routes exist without a strong-connectivity assumption. Its bound
κ ∈ [−2, 1] holds when all edge distances are equal (including the default
unit distances); with non-uniform edge distances, curvature can be below −2.

**Signed graphs.** No transport curvature accepts negative weights, since they
would give signed measures. Compute on `|w|` and keep the sign as an attribute,
or move the sign into the topology with
`graph_geometry.curvature.signed_double_cover` and then use any definition.

## Ricci flow

`RicciFlow` evolves edge weights or distances by an equation
`(K, q, step, K_avg) -> delta`. Built in are `normalized`, `unnormalized` and
`additive`. A custom equation is a callable or a formula string such as
`"-eta*tanh(kappa-kbar)*w"`. Surgery is passed as
`surgery={"name": "surgery_n", "portion": 1, "interval": 30}`. Curvature is
recomputed on every committed state, including after surgery. Every run reports
a `termination_reason`, for example `converged`, `iterations`, `undefined`
(surgery disconnected a digraph) or `numerical` (the solver failed). A run that
stops early keeps every state before the stop.

```python
import networkx as nx
from graph_geometry import viz

G = nx.barbell_graph(5, 0)
result = gg.RicciFlow(G, curvature="lin_lu_yau").run(iterations=30, step=0.1)

traj = gg.extract_edge_trajectories(result)     # per-edge curvature and weight
viz.plot_edge_trajectories(result, highlight_edges=[(4, 5)])
```

## Comparing definitions

```python
cmp = gg.compare_curvature(G, requests=[
    gg.CurvatureRequest("ollivier", {"alpha": 0.0}, label="ORC α=0"),
    gg.CurvatureRequest("ollivier", {"alpha": 0.5}, label="ORC α=0.5"),
    gg.CurvatureRequest("forman_node_weighted"),
])
cmp.summary()      # coverage, min / median / max, % negative, failure kind
cmp.agreement()    # pairwise Spearman, Pearson and sign agreement
cmp.to_csv("comparison.csv")
```

A request that cannot run is recorded as a `CurvatureFailure` (configuration,
inapplicable, invalid input, numerical, contract or internal) and the others
still run. `compare_flow` does the same for flow equations.

## Experiments

An experiment file names every convention explicitly:

```yaml
data:
  graph_kind: directed
  dataset: cora
  data_root: data
curvature:
  method: lin_lu_yau
  parameters: {kernel: mixed, beta_strategy: constant, beta: 0.8}
execution: {proc: 4}
flow: {equation: normalized, evolve: weight, iterations: 2000, step: 0.01}
surgery: {name: surgery_n, portion: 1, interval: 30}
```

```python
result = gg.run_experiment("configs/cora.yaml")
```

With `output.dir` set, the run writes `resolved_semantics.json` (canonical
method, graph kind, direction convention, effective parameters, warnings)
alongside the results. See [`configs/`](configs/) for a complete file.

## Interface

```bash
uv run --extra app streamlit run app.py
```

Build or load a graph, choose a curvature and flow equation, run the flow, and
inspect the graph, convergence, edge trajectories and distributions. Controls,
defaults and applicability messages come from each definition's registry
entry, so a plugin appears with its own controls without editing `app.py`.
Plugins can be loaded from a `.py` file in the interface. The file is executed,
so load only files you trust.

## Writing a plugin

A combinatorial curvature is one function returning `{(u, v): value}`. A
transport curvature reuses the built-in engine and replaces any of its four
components (measure, cost, solver, formula). Two examples ship with the
repository, each written without touching `src/`:

- [`examples/plugin_balanced_forman.py`](examples/plugin_balanced_forman.py) —
  Balanced Forman curvature (Topping et al. 2022).
- [`examples/plugin_sinkhorn_ollivier.py`](examples/plugin_sinkhorn_ollivier.py) —
  Ollivier curvature with the exact solver replaced by Sinkhorn.

## Repository layout

```
src/graph_geometry/
├── graph/          factors, configuration, semantics, generators, loaders
├── curvature/      registry, resolution, OT engine, kernels, built-in definitions
├── flow/           equations, surgery, RicciFlow engine
├── compare.py      compare_curvature, compare_flow
├── simulator.py    RicciFlowSimulator facade
├── experiment.py   run_experiment
├── io.py           CSV / JSON / GEXF export, SnapshotSeries
└── viz.py          matplotlib plots (viz extra)
app.py              Streamlit interface (app extra)
configs/            example experiment files
data/               Cora (MIT, from Planetoid) and small test graphs
examples/           quickstart and plugin examples
tests/              pytest suite, with reference values in tests/reference/
```

## Development

```bash
uv sync --all-extras
uv run pytest -q
uv run ruff check src tests examples app.py
```

The suite checks closed-form values, invariants (relabelling, scale,
orientation), the result contract, flow semantics, and numerical parity with
the research code this package replaced.

## Citation

If you use this software, please cite it using [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).

Datasets keep their own terms. `data/cora/ind.cora.graph` is from
[Planetoid](https://github.com/kimiyoung/planetoid) (MIT, see
[`data/cora/LICENSE`](data/cora/LICENSE)); cite Yang, Cohen & Salakhutdinov
(ICML 2016). email-Eu-core is not redistributed: `from_dataset("email_eu_core",
...)` downloads it from [SNAP](https://snap.stanford.edu/data/email-Eu-core.html)
on first use; cite Yin et al. (KDD 2017) and Leskovec et al. (TKDD 2007).
