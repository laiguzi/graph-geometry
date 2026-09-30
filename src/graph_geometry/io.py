"""Export of curvature results and flow runs: GEXF / CSV / JSON.

- curvature-wise: :func:`save_curvature_csv` writes a
  :class:`~graph_geometry.curvature.model.CurvatureResult` (edges or pairs);
- flow-wise: :func:`save_flow_result` writes a
  :class:`~graph_geometry.flow.FlowResult` -- one GEXF per committed state, the
  convergence series, and ``flow.json`` with the stopping reason, diagnosis,
  resolved curvature record and warnings; :func:`save_edge_trajectories_csv`
  writes the tidy per-edge records of
  :func:`~graph_geometry.flow.extract_edge_trajectories`;
- :class:`SnapshotSeries` bundles a run's snapshots and convergence with the same
  exporters.

Iteration numbers are committed-state indices: ``snapshots[i]`` is state ``i``
(``0`` is the input), and the spread ``convergence[i - 1]`` belongs to state
``i``.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx

__all__ = [
    "save_gexf_snapshots",
    "save_convergence_csv",
    "save_edge_curvatures_csv",
    "save_curvature_csv",
    "save_flow_result",
    "save_edge_trajectories_csv",
    "snapshots_to_json",
    "save_json",
    "SnapshotSeries",
]


def save_gexf_snapshots(snapshots, directory: str) -> None:
    """Write each snapshot as ``<i>.gexf`` into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    for i, G in enumerate(snapshots):
        nx.write_gexf(G, os.path.join(directory, f"{i}.gexf"))


def save_convergence_csv(convergence, path: str) -> None:
    """Write the convergence series as ``iteration, rc_diff``.

    ``iteration`` is the committed state the spread belongs to, starting at 1
    (state 0, the input, has no spread) -- the numbering of the GEXF files and
    of the interface's download. It was 0-based before, one off from both.
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "rc_diff"])
        for i, d in enumerate(convergence, start=1):
            w.writerow([i, d])


def save_edge_curvatures_csv(
    G: nx.Graph, path: str, weight: str = "weight", attr: str = "ricciCurvature"
) -> None:
    """Write per-edge ``source, target, weight, <attr>`` for one graph."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", weight, attr])
        for u, v, data in G.edges(data=True):
            w.writerow([u, v, data.get(weight, ""), data.get(attr, "")])


def save_curvature_csv(result, path: str, *, node_means_path: Optional[str] = None) -> None:
    """Write a curvature result as ``source, target, kappa`` rows.

    ``node_means_path`` additionally writes ``node, kappa_mean`` (the incident
    node means; empty for a pair result).
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "target", "kappa"])
        for row in result.rows():
            w.writerow([row["source"], row["target"], row["kappa"]])
    if node_means_path is not None:
        with open(node_means_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["node", "kappa_mean"])
            for node, value in result.incident_node_means.items():
                w.writerow([node, value])


def snapshots_to_json(snapshots, convergence=None) -> dict:
    """Serialise snapshots (+ optional convergence) to a JSON-able dict."""
    return {
        "snapshots": [nx.node_link_data(G, edges="links") for G in snapshots],
        "convergence": list(convergence) if convergence is not None else None,
    }


def save_json(snapshots, path: str, convergence=None) -> None:
    """Write snapshots (+ convergence) as JSON."""
    with open(path, "w") as f:
        json.dump(snapshots_to_json(snapshots, convergence), f)


def save_flow_result(result, directory: str) -> None:
    """Write a flow run: ``snapshots/<i>.gexf``, ``convergence.csv``, ``flow.json``."""
    os.makedirs(directory, exist_ok=True)
    save_gexf_snapshots(result.snapshots, os.path.join(directory, "snapshots"))
    save_convergence_csv(result.convergence, os.path.join(directory, "convergence.csv"))
    with open(os.path.join(directory, "flow.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)
        f.write("\n")


def save_edge_trajectories_csv(trajectories, path_or_buffer) -> None:
    """Write :meth:`EdgeTrajectories.rows` as CSV, one row per ``(iteration, edge)``.

    ``path_or_buffer`` is a path or an open text buffer (e.g. ``io.StringIO``
    for a download). A value that is absent -- an edge removed by surgery -- is
    an empty cell, never ``0``.
    """
    from .flow.trajectories import TRAJECTORY_FIELDS

    def write(f) -> None:
        writer = csv.DictWriter(f, fieldnames=list(TRAJECTORY_FIELDS))
        writer.writeheader()
        writer.writerows(trajectories.rows())

    if hasattr(path_or_buffer, "write"):
        write(path_or_buffer)
        return
    with open(path_or_buffer, "w", newline="") as f:
        write(f)


@dataclass
class SnapshotSeries:
    """A flow result: the snapshot list + convergence series, with exporters.

    ``result`` keeps the originating :class:`~graph_geometry.flow.FlowResult`
    when the series was built with :meth:`from_result`, so the stopping reason
    and resolved record are exported with the states.
    """

    snapshots: list = field(default_factory=list)
    convergence: list = field(default_factory=list)
    weight: str = "weight"
    result: Any = None

    @classmethod
    def from_result(cls, result, weight: str = "weight") -> "SnapshotSeries":
        return cls(list(result.snapshots), list(result.convergence), weight=weight, result=result)

    def __len__(self) -> int:
        return len(self.snapshots)

    def __getitem__(self, i):
        return self.snapshots[i]

    def __iter__(self):
        return iter(self.snapshots)

    @property
    def initial(self):
        return self.snapshots[0] if self.snapshots else None

    @property
    def final(self):
        return self.snapshots[-1] if self.snapshots else None

    def save_gexf(self, directory: str) -> None:
        save_gexf_snapshots(self.snapshots, directory)

    def save_convergence(self, path: str) -> None:
        save_convergence_csv(self.convergence, path)

    def save_edge_curvatures(self, path: str, attr: str = "ricciCurvature") -> None:
        if self.final is not None:
            save_edge_curvatures_csv(self.final, path, weight=self.weight, attr=attr)

    def to_json(self) -> dict:
        return snapshots_to_json(self.snapshots, self.convergence)

    def save_json(self, path: str) -> None:
        save_json(self.snapshots, path, self.convergence)

    def save(self, directory: str) -> None:
        """Everything: GEXF states, convergence CSV and, with a result, ``flow.json``."""
        if self.result is not None:
            save_flow_result(self.result, directory)
            return
        os.makedirs(directory, exist_ok=True)
        self.save_gexf(os.path.join(directory, "snapshots"))
        self.save_convergence(os.path.join(directory, "convergence.csv"))
