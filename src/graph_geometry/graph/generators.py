"""Graph generators.

Ported from ``directed_ricciflow/RicciFlow/graphs/graph_generate.py`` with clean
keyword signatures (the originals took an untyped ``dict``). All return NetworkX
graphs; the ``GENERATORS`` registry maps names to functions for config-driven use.

Graphs are returned *without* weight/distance/node attributes — assign those with
:func:`graph_geometry.graph.config.apply_config`.
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np

__all__ = [
    "gab",
    "sbm",
    "lfr",
    "cycle",
    "petersen",
    "complete",
    "petersen_complete",
    "complete_complete",
    "GENERATORS",
]


def gab(a: int, b: int) -> nx.Graph:
    """G(a, b): ``b + 1`` cliques of size ``a + 1`` joined by a clique on their
    first nodes. (Original ``Gab``.)"""
    G = nx.Graph()
    a = a + 1
    b = b + 1
    for i in range(b):
        G.add_edges_from(itertools.combinations(range(i * a, (i + 1) * a), 2))
    G.add_edges_from(itertools.combinations(range(0, a * b, a), 2))
    return G


def sbm(sizes: list[int], probs: list[list[float]], seed: int | None = None) -> nx.Graph:
    """Stochastic block model. (Original ``SBM``.)"""
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def lfr(
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    average_degree: float,
    seed: int | None = 0,
    **kwargs: object,
) -> nx.Graph:
    """LFR benchmark graph, flattening the per-node ``community`` frozensets into
    an integer ``community_idx`` node attribute. (Original ``LFR``.)"""
    G = nx.generators.community.LFR_benchmark_graph(  # type: ignore[attr-defined]
        n, tau1, tau2, mu, average_degree, seed=seed, **kwargs
    )
    nx.set_node_attributes(G, {node: None for node in G.nodes()}, "community_idx")
    index = 0
    for node in G.nodes():
        if G.nodes[node]["community_idx"] is None:
            for c_node in G.nodes[node]["community"]:
                G.nodes[c_node]["community_idx"] = index
                del G.nodes[c_node]["community"]
            index += 1
    return G


def cycle(n: int) -> nx.Graph:
    """Cycle graph on ``n`` nodes. (Original ``cycle_graph``.)"""
    return nx.cycle_graph(n)


def petersen() -> nx.Graph:
    """The Petersen graph. (Original ``petersen_graph``.)"""
    return nx.petersen_graph()


def complete(n: int) -> nx.Graph:
    """Complete graph on ``n`` nodes. (Original ``complete_graph``.)"""
    return nx.complete_graph(n)


def petersen_complete() -> nx.Graph:
    """Petersen graph (nodes 0-9) bridged by a single edge (9-10) to a complete
    graph on 6 nodes (10-15). (Original ``petersen_complete``.)"""
    petersen_edges = np.array(nx.petersen_graph().edges())
    complete_edges = np.array(nx.complete_graph(6).edges()) + 10
    edges = np.concatenate([petersen_edges, complete_edges, np.array([[9, 10]])], axis=0)
    G = nx.Graph()
    G.add_edges_from(edges.tolist())
    return G


def complete_complete(n1: int, n2: int) -> nx.Graph:
    """Two complete graphs (sizes ``n1``, ``n2``) bridged by a single edge
    ``(n1 - 1, n1)``. (Original ``complete_complete``.)"""
    g1_edges = np.array(nx.complete_graph(n1).edges())
    g2_edges = np.array(nx.complete_graph(n2).edges()) + n1
    edges = np.concatenate([g1_edges, g2_edges, np.array([[n1 - 1, n1]])], axis=0)
    G = nx.Graph()
    G.add_edges_from(edges.tolist())
    return G


#: Name -> generator, for config-/experiment-driven construction.
GENERATORS = {
    "gab": gab,
    "sbm": sbm,
    "lfr": lfr,
    "cycle": cycle,
    "petersen": petersen,
    "complete": complete,
    "petersen_complete": petersen_complete,
    "complete_complete": complete_complete,
}
