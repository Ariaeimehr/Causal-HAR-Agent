"""
Data structures and graph algorithms for Structural Causal Models (SCMs) in HAR.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
import json
import networkx as nx
import numpy as np
from pydantic import BaseModel, Field


class CausalNode(BaseModel):
    """Represents an observed sensor node in the Structural Causal Model."""
    id: str
    anatomical_location: str = "body"
    sensor_type: str = "inertial"
    axis: str = "unknown"
    description: str = ""


class CausalEdge(BaseModel):
    """Represents a directed cause-and-effect link between physical sensor streams."""
    source: str
    target: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    mechanism: str = ""
    time_delay_ms: float = 0.0


class CausalGraph:
    """
    Structural Causal Model wrapper backed by NetworkX DiGraph.
    Provides graph theory operations: acyclicity validation, topological sorting,
    d-separation queries, and generation of causal attention prior matrices.
    """

    def __init__(
        self,
        nodes: Optional[List[CausalNode]] = None,
        edges: Optional[List[CausalEdge]] = None,
        activity: str = "General",
        reasoning_summary: str = "",
    ) -> None:
        """
        Initialize a CausalGraph.

        Args:
            nodes: List of sensor nodes.
            edges: List of directed causal edges.
            activity: Name of the activity context.
            reasoning_summary: Biomechanical description.
        """
        self.activity = activity
        self.reasoning_summary = reasoning_summary
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self.nx_graph = nx.DiGraph()

        if nodes:
            for node in nodes:
                self.add_node(node)

        if edges:
            for edge in edges:
                self.add_edge(edge)

        self._ensure_dag()

    def add_node(self, node: CausalNode) -> None:
        """Adds a sensor node to the graph."""
        self.nodes[node.id] = node
        self.nx_graph.add_node(
            node.id,
            location=node.anatomical_location,
            sensor_type=node.sensor_type,
            axis=node.axis,
            description=node.description,
        )

    def add_edge(self, edge: CausalEdge) -> None:
        """Adds a directed causal edge between sensor nodes."""
        if edge.source not in self.nodes:
            self.add_node(CausalNode(id=edge.source))
        if edge.target not in self.nodes:
            self.add_node(CausalNode(id=edge.target))

        self.edges.append(edge)
        self.nx_graph.add_edge(
            edge.source,
            edge.target,
            weight=edge.confidence,
            confidence=edge.confidence,
            mechanism=edge.mechanism,
            time_delay_ms=edge.time_delay_ms,
        )

    def _ensure_dag(self) -> None:
        """
        Verifies acyclicity. If cycles exist due to LLM reasoning errors,
        prunes minimum-confidence edges until a strict DAG is achieved.
        """
        while not nx.is_directed_acyclic_graph(self.nx_graph):
            try:
                cycle = nx.find_cycle(self.nx_graph, orientation="original")
                # Find edge with minimum confidence in the cycle
                min_conf = float("inf")
                edge_to_remove = None
                for u, v, _ in cycle:
                    conf = self.nx_graph[u][v].get("confidence", 1.0)
                    if conf < min_conf:
                        min_conf = conf
                        edge_to_remove = (u, v)

                if edge_to_remove:
                    u, v = edge_to_remove
                    self.nx_graph.remove_edge(u, v)
                    self.edges = [
                        e for e in self.edges if not (e.source == u and e.target == v)
                    ]
            except nx.NetworkXNoCycle:
                break

    def get_parents(self, node_id: str) -> List[str]:
        """Returns direct causal parents of a node."""
        if node_id in self.nx_graph:
            return list(self.nx_graph.predecessors(node_id))
        return []

    def get_children(self, node_id: str) -> List[str]:
        """Returns direct causal children of a node."""
        if node_id in self.nx_graph:
            return list(self.nx_graph.successors(node_id))
        return []

    def get_ancestors(self, node_id: str) -> Set[str]:
        """Returns all ancestral causal nodes (transitive closure)."""
        if node_id in self.nx_graph:
            return nx.ancestors(self.nx_graph, node_id)
        return set()

    def get_causal_adjacency_matrix(
        self,
        ordered_channels: List[str],
        include_self: bool = True,
        soft_weighting: bool = True,
    ) -> np.ndarray:
        """
        Generates the Causal Prior Adjacency Matrix M_causal for regularizing attention.

        M[i, j] > 0 if channel j causally influences channel i (or j == i if include_self).

        Args:
            ordered_channels: List of channel names matching tensor dimension order.
            include_self: Whether diagonal self-loops receive prior weight 1.0.
            soft_weighting: Whether to use edge confidence score as weight.

        Returns:
            np.ndarray: Matrix of shape (K, K) where K = len(ordered_channels).
        """
        K = len(ordered_channels)
        M = np.zeros((K, K), dtype=np.float32)

        ch_to_idx = {name: idx for idx, name in enumerate(ordered_channels)}

        for i, target_ch in enumerate(ordered_channels):
            if target_ch not in self.nx_graph:
                continue

            # Ancestors of target_ch causally influence target_ch
            ancestors = self.get_ancestors(target_ch)

            for parent_ch in self.get_parents(target_ch):
                if parent_ch in ch_to_idx:
                    j = ch_to_idx[parent_ch]
                    conf = (
                        self.nx_graph[parent_ch][target_ch].get("confidence", 1.0)
                        if soft_weighting
                        else 1.0
                    )
                    M[i, j] = max(M[i, j], conf)

            for anc_ch in ancestors:
                if anc_ch in ch_to_idx and anc_ch not in self.get_parents(target_ch):
                    j = ch_to_idx[anc_ch]
                    # Transitive ancestor gets slightly attenuated weight
                    M[i, j] = max(M[i, j], 0.5)

        if include_self:
            np.fill_diagonal(M, 1.0)

        return M

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the causal graph to a standard dictionary format."""
        return {
            "activity": self.activity,
            "reasoning_summary": self.reasoning_summary,
            "is_dag": nx.is_directed_acyclic_graph(self.nx_graph),
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "edges": [e.model_dump() for e in self.edges],
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes the causal graph to a JSON formatted string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalGraph":
        """Reconstructs a CausalGraph from a dictionary representation."""
        nodes = [CausalNode(**n) for n in data.get("nodes", [])]
        edges = [CausalEdge(**e) for e in data.get("edges", [])]
        return cls(
            nodes=nodes,
            edges=edges,
            activity=data.get("activity", "General"),
            reasoning_summary=data.get("reasoning_summary", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "CausalGraph":
        """Reconstructs a CausalGraph from a JSON string."""
        return cls.from_dict(json.loads(json_str))
