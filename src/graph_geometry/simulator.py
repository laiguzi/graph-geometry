"""RicciFlowSimulator — high-level facade over the flow engine.

A thin, ergonomic wrapper around :class:`graph_geometry.flow.RicciFlow` with
factory constructors that delegate to :mod:`graph_geometry.graph.loaders`, plus
convenience accessors (``snapshots``, ``convergence``, ``initial_graph``,
``result_graph``) and a ``summary()``.

Ported from ``ricciflow_sim/simulator.py``.

Quick start
-----------
    import graph_geometry as gg

    sim = gg.RicciFlowSimulator.from_edges(
        [(0, 1), (1, 2), (2, 0), (0, 2)], directed=True,
        curvature=gg.CurvatureRequest("lin_lu_yau", {"kernel": "mixed", "beta": 0.8}),
    )
    sim.run(iterations=50, step=0.05)
    sim.snapshots        # list of graph copies
    sim.convergence      # RC spread per iteration
    sim.summary()
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Optional

import networkx as nx
import numpy as np

from .flow import RicciFlow
from .flow.surgery import SurgerySpec
from .graph import loaders as load
from .graph.loaders import DIRECTED_UNSET
from .graph.semantics import GraphSemantics

__all__ = ["RicciFlowSimulator"]


class RicciFlowSimulator:
    """High-level interface for Ricci flow on graphs."""

    def __init__(
        self,
        G: nx.Graph,
        curvature: Any = "lin_lu_yau",
        flow_equation: Any = "normalized",
        weight: str = "weight",
        distance: str = "distance",
        evolve: str = "weight",
        proc: int = 1,
        semantics: Optional[GraphSemantics] = None,
        **curvature_params: Any,
    ):
        self._G0 = G.copy()
        self.semantics = semantics
        self.curvature = curvature
        self.flow_equation = flow_equation
        self.weight = weight
        self.distance = distance
        self.evolve = evolve
        self.proc = proc
        self.curvature_params = curvature_params

        self.snapshots: list = []
        self.convergence: list[float] = []
        self.result: Any = None
        self._ran = False

    # ── factory constructors (delegate to graph.loaders) ────────────────

    @classmethod
    def from_edges(cls, edges, directed: bool = DIRECTED_UNSET, **kwargs) -> "RicciFlowSimulator":
        """From ``(u, v)`` or ``(u, v, weight)`` tuples."""
        return cls(load.from_edges(edges, directed=directed), **kwargs)

    @classmethod
    def from_networkx(cls, G: nx.Graph, **kwargs) -> "RicciFlowSimulator":
        """From an existing NetworkX graph (copied)."""
        return cls(load.from_networkx(G), **kwargs)

    @classmethod
    def from_edgelist(
        cls,
        path: str,
        directed: bool = DIRECTED_UNSET,
        weighted: bool = False,
        *,
        weight_attr: str = "weight",
        columns: Sequence[str] | None = None,
        node_attr: str = "weight",
        **kwargs,
    ):
        """From an edge-list file, preserving explicitly named data columns."""
        G = load.from_edgelist(
            path,
            directed=directed,
            weighted=weighted,
            weight_attr=weight_attr,
            columns=columns,
            node_attr=node_attr,
        )
        return cls(G, **kwargs)

    @classmethod
    def from_gexf(cls, path: str, **kwargs) -> "RicciFlowSimulator":
        """From a GEXF file."""
        return cls(load.from_gexf(path), **kwargs)

    @classmethod
    def from_dataset(
        cls, name: str, data_root, directed: bool = DIRECTED_UNSET, **kwargs
    ) -> "RicciFlowSimulator":
        """From a built-in research dataset (see :func:`graph_geometry.load.from_dataset`)."""
        return cls(load.from_dataset(name, data_root, directed=directed), **kwargs)

    # ── run ─────────────────────────────────────────────────────────────

    def run(
        self,
        iterations: int = 50,
        step: float = 0.01,
        delta: float = 1e-6,
        surgery: SurgerySpec = None,
        early_stop: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> "RicciFlowSimulator":
        """Run the flow; results land in ``self.snapshots`` / ``self.convergence``."""
        engine = RicciFlow(
            self._G0,
            curvature=self.curvature,
            flow_equation=self.flow_equation,
            weight=self.weight,
            distance=self.distance,
            evolve=self.evolve,
            proc=self.proc,
            semantics=self.semantics,
            **self.curvature_params,
        )
        self.result = engine.run(
            iterations=iterations,
            step=step,
            delta=delta,
            surgery=surgery,
            early_stop=early_stop,
            save_dir=save_dir,
            verbose=verbose,
            progress_callback=progress_callback,
        )
        self.snapshots = self.result.snapshots
        self.convergence = self.result.convergence
        self._ran = True
        return self

    # ── accessors ───────────────────────────────────────────────────────

    @property
    def termination_reason(self) -> Optional[str]:
        """Why the run stopped: ``converged`` | ``exhausted`` | ``degenerate`` |
        ``diverged`` | ``undefined`` | ``numerical`` | ``iterations``
        (:data:`~graph_geometry.flow.TERMINATION_REASONS`), or ``None`` before a
        run. ``exhausted`` means surgery removed every edge; it is deliberately
        not reported as convergence. ``diverged`` means an update would have
        made the evolving quantity invalid, ``undefined`` that surgery produced
        a topology the curvature is not defined on, and ``numerical`` that the
        curvature solver failed on a candidate state; in all three cases the
        run kept its committed states and stopped."""
        return self.result.termination_reason if self.result else None

    @property
    def resolved(self):
        """The resolved curvature record of the last run, or ``None``."""
        return self.result.resolved if self.result else None

    @property
    def initial_graph(self) -> Optional[nx.Graph]:
        """Graph at iteration 0 (with initial curvatures), or ``None``."""
        return self.snapshots[0] if self.snapshots else None

    @property
    def result_graph(self) -> Optional[nx.Graph]:
        """Final graph after the flow, or ``None``."""
        return self.snapshots[-1] if self.snapshots else None

    def snapshot_at(self, i: int) -> nx.Graph:
        """Snapshot at iteration ``i`` (0 = initial)."""
        if not self.snapshots:
            raise RuntimeError("Run .run() first.")
        return self.snapshots[i]

    def summary(self) -> None:
        """Print final-state curvature and the recorded evolving edge quantity."""
        if not self._ran:
            print("Simulation not run yet. Call .run() first.")
            return
        G0, Gf = self.initial_graph, self.result_graph
        rc0 = list(nx.get_edge_attributes(G0, "ricciCurvature").values())
        rcf = list(nx.get_edge_attributes(Gf, "ricciCurvature").values())
        q_role = self.result.evolve or self.evolve
        q_attr = self.result.evolving_attr or (
            self.distance if q_role == "distance" else self.weight
        )
        q_symbol = "d" if q_role == "distance" else "w"
        q0 = list(nx.get_edge_attributes(G0, q_attr).values())
        qf = list(nx.get_edge_attributes(Gf, q_attr).values())
        kind = "directed" if G0.is_directed() else "undirected"
        print("=" * 55)
        print(f"  Ricci Flow Simulation Summary ({kind})")
        print("=" * 55)
        print(f"  Iterations completed : {self.result.iterations_completed}")
        print(f"  Terminated because   : {self.termination_reason}")
        final_spread = f"{max(rcf) - min(rcf):.6f}" if rcf else "undefined"
        print(f"  Final RC difference  : {final_spread}")
        print(f"  Evolving quantity    : {q_role} ({q_attr})")
        print(f"  Initial : {G0.number_of_nodes()} nodes, {G0.number_of_edges()} edges")
        print(f"  Final   : {Gf.number_of_nodes()} nodes, {Gf.number_of_edges()} edges")
        if rc0:
            print(f"  RC  initial : [{min(rc0):.4f}, {max(rc0):.4f}]  mean={np.mean(rc0):.4f}")
        if rcf:
            print(f"  RC  final   : [{min(rcf):.4f}, {max(rcf):.4f}]  mean={np.mean(rcf):.4f}")
        else:
            print("  RC  final   : undefined")
        if q0:
            print(f"  {q_symbol}   initial : [{min(q0):.4f}, {max(q0):.4f}]  mean={np.mean(q0):.4f}")
        if qf:
            print(f"  {q_symbol}   final   : [{min(qf):.4f}, {max(qf):.4f}]  mean={np.mean(qf):.4f}")
        else:
            print(f"  {q_symbol}   final   : undefined")
        print("=" * 55)
