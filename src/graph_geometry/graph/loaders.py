"""Graph loaders / constructors.

Ported from ``ricciflow_sim/simulator.py`` factory methods (plus the old
``GraphDataLoader`` for ``from_dataset``), but returning plain NetworkX graphs
(not a simulator) and *without* inventing weight/distance attributes — factor
assignment is the job of :func:`graph_geometry.graph.config.apply_config`.
Weights present in the *source* (3-tuple edges, weighted edge lists, GEXF files)
are preserved.
"""

from __future__ import annotations

import gzip
import pickle
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx

__all__ = [
    "from_edges",
    "from_networkx",
    "from_edgelist",
    "from_gexf",
    "from_dataset",
    "DATASETS",
]

#: Built-in research datasets recognised by :func:`from_dataset`.
DATASETS = ("cora", "email_eu_core", "mygraph")

#: email-Eu-core is not redistributed with the package (SNAP publishes it
#: without a licence); :func:`from_dataset` fetches it from SNAP on first use.
EMAIL_EU_CORE_URLS = {
    "email-Eu-core.txt": "https://snap.stanford.edu/data/email-Eu-core.txt.gz",
    "email-Eu-core-department-labels.txt":
        "https://snap.stanford.edu/data/email-Eu-core-department-labels.txt.gz",
}


def _fetch_email_eu_core(folder: Path) -> None:
    """Download and unpack any missing email-Eu-core file into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    for filename, url in EMAIL_EU_CORE_URLS.items():
        target = folder / filename
        if target.exists():
            continue
        partial = target.with_name(target.name + ".part")
        try:
            with urllib.request.urlopen(url, timeout=60) as response, \
                    gzip.GzipFile(fileobj=response) as source, \
                    open(partial, "wb") as out:
                shutil.copyfileobj(source, out)
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise OSError(
                f"could not download email-Eu-core from {url} ({exc}). "
                f"Download and gunzip it by hand into {folder}."
            ) from exc
        partial.replace(target)


class _Unset:
    """Marker for a ``directed`` argument the caller did not pass."""

    def __repr__(self) -> str:
        return "<unset: directed=True>"


#: Default of ``directed`` in the loaders. It still means ``True`` for one
#: deprecation cycle, but an implicit graph direction is not reproducible, so
#: omitting the argument warns.
DIRECTED_UNSET: Any = _Unset()


def _directed(value: Any, where: str) -> bool:
    if value is DIRECTED_UNSET:
        warnings.warn(
            f"{where}: 'directed' was not passed and defaults to True. Pass "
            f"directed=True or directed=False explicitly; reproducible "
            f"experiments must record the graph kind (data.graph_kind).",
            FutureWarning,
            stacklevel=3,
        )
        return True
    if not isinstance(value, bool):
        raise TypeError(f"{where}: directed must be a bool; got {value!r}.")
    return value


def _split_node_lines(
    path: str | Path, delimiter: str | None = None
) -> tuple[list[str], list[tuple[str, float]]]:
    """Split an edge-list file into edge lines and ``N``-prefixed node lines.

    A line whose first whitespace-separated token is exactly ``N`` declares a
    node-weight assignment: ``N <node> <value>``. (``N`` is therefore reserved
    as a line marker and cannot be a node id in files using this convention.)
    Comment lines (``#``) and blank lines are dropped. Returns
    ``(edge_lines, [(node, value), ...])``.
    """
    edge_lines: list[str] = []
    node_assignments: list[tuple[str, float]] = []
    with open(path) as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(delimiter) if delimiter else line.split()
            parts = [p for p in parts if p != ""] if delimiter else parts
            if parts and parts[0] == "N":
                if len(parts) != 3:
                    raise ValueError(
                        f"{path}:{lineno}: node line must be 'N <node> <value>', "
                        f"got {line!r}"
                    )
                node_assignments.append((parts[1], float(parts[2])))
            else:
                edge_lines.append(line)
    return edge_lines, node_assignments


def from_edges(
    edges: Iterable[Sequence],
    directed: bool = DIRECTED_UNSET,
    weight_attr: str = "weight",
    distance_attr: str = "distance",
) -> nx.Graph:
    """Build a graph from edge tuples with optional user-given attributes.

    Accepted forms (mixable):

    - ``(u, v)``                    -- bare edge
    - ``(u, v, w)``                 -- weight under ``weight_attr``
    - ``(u, v, w, d)``              -- weight + distance
    - ``(u, v, {attr: value, ...})``-- arbitrary edge attributes

    Pass ``directed`` explicitly; omitting it still builds a digraph for one
    deprecation cycle but warns. Attributes given here are *source* attributes,
    so a later ``apply_config(..., overwrite=False)`` will not clobber them.
    """
    directed = _directed(directed, "from_edges")
    G: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    for e in edges:
        e = tuple(e)
        if len(e) == 2:
            G.add_edge(e[0], e[1])
        elif len(e) == 3 and isinstance(e[2], dict):
            G.add_edge(e[0], e[1], **e[2])
        elif len(e) == 3:
            G.add_edge(e[0], e[1], **{weight_attr: float(e[2])})
        elif len(e) == 4:
            G.add_edge(
                e[0], e[1],
                **{weight_attr: float(e[2]), distance_attr: float(e[3])},
            )
        else:
            raise ValueError(
                f"Edge tuple must be (u,v), (u,v,w), (u,v,w,d) or "
                f"(u,v,attrs_dict); got {e!r}"
            )
    return G


def from_networkx(G: nx.Graph, *, copy: bool = True) -> nx.Graph:
    """Adopt an existing NetworkX graph (directed/undirected preserved).

    Returns a copy by default so later annotation does not mutate the caller's
    graph; pass ``copy=False`` to adopt in place.
    """
    return G.copy() if copy else G


def from_edgelist(
    path: str,
    directed: bool = DIRECTED_UNSET,
    weighted: bool = False,
    weight_attr: str = "weight",
    columns: Sequence[str] | None = None,
    node_attr: str = "weight",
    delimiter: str | None = None,
) -> nx.Graph:
    """Read an edge list file, with optional ``N``-prefixed node-weight lines.

    Line formats supported (mixable in one file):

    - ``u v``                        -- default (structure only)
    - ``u v w``                      -- ``weighted=True`` (weight under ``weight_attr``)
    - ``u v w d ...``                -- ``columns=("weight", "distance", ...)``:
      one float column per named attribute, in order
    - ``N n w``                      -- node line: ``G.nodes[n][node_attr] = w``;
      also declares ``n``, so isolated nodes can be listed
    - ``u v {'weight': .., 'distance': ..}`` -- dict-format lines: use
      ``nx.read_edgelist(path)`` directly (NetworkX's default mode) and wrap
      with :func:`from_networkx`

    ``columns`` takes precedence over ``weighted``. Node ids are read as
    strings (NetworkX edge-list convention); ``N`` is reserved as the node-line
    marker.

    ``delimiter`` selects the field separator; ``None`` splits on any run of
    whitespace, which is NetworkX's default and cannot represent a node id that
    contains a space. Pass ``"\t"`` for a tab-separated file whose ids may
    contain spaces.

    Pass ``directed`` explicitly; omitting it still reads a digraph for one
    deprecation cycle but warns.
    """
    directed = _directed(directed, "from_edgelist")
    create_using = nx.DiGraph if directed else nx.Graph
    if columns is not None:
        data = [(c, float) for c in columns]
    elif weighted:
        data = [(weight_attr, float)]
    else:
        data = False

    edge_lines, node_assignments = _split_node_lines(path, delimiter)
    G = nx.parse_edgelist(
        edge_lines, create_using=create_using, data=data, delimiter=delimiter
    )  # type: ignore[arg-type]
    for n, value in node_assignments:
        G.add_node(n)
        G.nodes[n][node_attr] = value
    return G


def from_gexf(path: str) -> nx.Graph:
    """Read a GEXF file (directed/undirected as stored in the file)."""
    return nx.read_gexf(path)


def from_dataset(
    name: str,
    data_root: str | Path,
    *,
    directed: bool = DIRECTED_UNSET,
    variant: str = "3cycle4edge",
    columns: Sequence[str] | None = None,
) -> nx.Graph:
    """Load a built-in research dataset from ``data_root``.

    Ported from ``directed_ricciflow``'s ``GraphDataLoader``. Returns the graph
    only (features/labels are out of scope). Recognised ``name`` values are in
    :data:`DATASETS`:

    - ``"cora"``          -- citation graph from ``cora/ind.cora.graph`` (pickle).
    - ``"email_eu_core"`` -- ``email_EU_core/email-Eu-core.txt`` edge list,
      downloaded from SNAP into ``data_root`` on first use (see
      :data:`EMAIL_EU_CORE_URLS`). Cite Yin et al. (KDD 2017) and Leskovec
      et al. (TKDD 2007).
    - ``"mygraph"``       -- a small hand-built ``mygraph/mygraph_<variant>.edgelist``.
      Lines may be bare ``u v``, dict-format ``u v {'weight': .., 'distance': ..}``
      (parsed by default), numeric columns ``u v w d`` with
      ``columns=("weight", "distance")``, and ``N n w`` node-weight lines
      (written to ``G.nodes[n]["weight"]``; ``N`` is reserved as the marker).

    ``data_root`` is the directory holding the dataset subfolders (e.g. the
    in-repo ``data`` directory). Node labels follow the source files (Cora uses
    ints from the adjacency dict; the edge-list datasets use strings).
    """
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {DATASETS}.")

    root = Path(data_root)
    directed = _directed(directed, "from_dataset")
    create_using = nx.DiGraph if directed else nx.Graph

    if key == "cora":
        with open(root / "cora" / "ind.cora.graph", "rb") as f:
            adjacency = pickle.load(f, encoding="latin1")
        return nx.from_dict_of_lists(adjacency, create_using=create_using())

    if key == "email_eu_core":
        path = root / "email_EU_core" / "email-Eu-core.txt"
        if not path.exists():
            _fetch_email_eu_core(path.parent)
        return nx.read_edgelist(path, create_using=create_using)

    # mygraph
    path = root / "mygraph" / f"mygraph_{variant}.edgelist"
    data = [(c, float) for c in columns] if columns is not None else True
    edge_lines, node_assignments = _split_node_lines(path)
    G = nx.parse_edgelist(edge_lines, create_using=create_using, data=data)  # type: ignore[arg-type]
    for n, value in node_assignments:
        G.add_node(n)
        G.nodes[n]["weight"] = value
    return G
