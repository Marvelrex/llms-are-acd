import ast
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
from typing_extensions import Literal

ACTION_GRAPH_FORMAT_VERSION = 2


def edge_combined_score(
    mean_contribution: float,
    visit_count: int,
    m2: float,
    quality_scale: float,
    *,
    pessimistic_default: float = 0.0,
) -> float:
    """Unified combined score: quality + uncertainty.

    Quality term: mean_contribution (running mean of credit-weighted advantage).
    For unvisited edges (visit_count == 0), ``pessimistic_default`` is used instead
    of mean_contribution to avoid the zero-initialization bias (where mc=0 appears
    artificially "best" when all real rewards are negative).
    Uncertainty term: ``EXPLORE_CAP * quality_scale / sqrt(visit_count + 1)`` plus an
    optional variance contribution for edges with >= 2 visits.  The bonus decreases
    smoothly with every additional visit (no plateau at low visit counts).
    """
    EXPLORE_CAP = 0.5     # exploration coefficient

    quality = pessimistic_default if visit_count == 0 else mean_contribution

    # Exploration bonus: decreases smoothly with visits, no plateau
    explore_bonus = EXPLORE_CAP * quality_scale / math.sqrt(visit_count + 1)

    # Variance contribution (additional uncertainty from high-variance edges)
    var_bonus = 0.0
    if visit_count >= 2:
        variance = m2 / (visit_count - 1)
        if variance > 0:
            var_bonus = math.sqrt(variance) / math.sqrt(visit_count + 1)

    uncertainty = explore_bonus + var_bonus
    return quality + uncertainty


@dataclass(frozen=True)
class ActionNode:
    """Represents a defender or attacker action in the turn-based graph."""

    action_id: str
    label: str
    agent_type: Literal["defender", "attacker"]
    preconditions: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation of this node."""
        return asdict(self)


class ActionGraph:
    """Directed graph enforcing alternating defender/attacker turns."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        # State-conditioned edge stats: state_signature -> from_id -> to_id -> stats dict.
        # Each stats dict stores: visit_count, mean_contribution, m2.
        #
        # Kept separate from NetworkX edge attributes for backwards compatibility:
        # older persisted graphs only contain global edge stats.
        self.state_edge_stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        # Coarse bucket-conditioned edge stats: bucket_id -> from_id -> to_id -> stats dict.
        # Buckets are ~36 coarse states (comp × alerts × step), accumulating 25x faster than
        # full state signatures. Used for mixed scoring via get_mixed_edge_stats().
        self.bucket_edge_stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
        # Reward magnitude anchor: updated by SelfEvolveDefender after each observe_round.
        # Used by _compute_quality_scale as an environment-scale anchor for S_r.
        self.reward_magnitude_anchor: float = 0.0
        # Pessimistic default for unvisited edges: set to the current baseline so
        # that mc=0 doesn't appear artificially "best" when all real rewards are negative.
        self.baseline_default: float = 0.0

    def add_action_node(self, action: ActionNode) -> None:
        """Add a new action node, ensuring unique identifiers."""
        if action.action_id in self.graph:
            raise ValueError(f"Action '{action.action_id}' already exists.")
        self.graph.add_node(
            action.action_id,
            action=action,
            agent_type=action.agent_type,
        )

    def _assert_node_exists(self, action_id: str) -> None:
        if action_id not in self.graph:
            raise ValueError(f"Unknown action id '{action_id}'.")

    def _get_agent_type(self, action_id: str) -> str:
        return self.graph.nodes[action_id]["agent_type"]

    def is_defender_node(self, action_id: str) -> bool:
        """Return True if the node belongs to the defender."""
        return self._get_agent_type(action_id) == "defender"

    def is_attacker_node(self, action_id: str) -> bool:
        """Return True if the node belongs to the attacker."""
        return self._get_agent_type(action_id) == "attacker"

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        description: Optional[str] = None,
    ) -> None:
        """Create a legal transition."""
        self._assert_node_exists(from_id)
        self._assert_node_exists(to_id)

        if self._get_agent_type(from_id) == self._get_agent_type(to_id):
            raise ValueError(
                f"Illegal transition from '{from_id}' to '{to_id}': "
                "turns must alternate between defender and attacker."
            )

        self.graph.add_edge(
            from_id,
            to_id,
            description=description,
            visit_count=0,
            mean_contribution=0.0,
            m2=0.0,
            # [avg_ep_reward disabled]
            # avg_ep_reward=0.0,
            # ep_count=0,
        )

    def ensure_edge(
        self,
        from_id: str,
        to_id: str,
        *,
        description: Optional[str] = None,
    ) -> bool:
        """Create a legal transition if missing (used for cold-start edge discovery).

        Returns True if a new edge was created, False if it already existed.
        """
        self._assert_node_exists(from_id)
        self._assert_node_exists(to_id)
        if self.graph.has_edge(from_id, to_id):
            return False
        self.add_edge(from_id, to_id, description=description)
        return True

    def add_llm_score(self, node_id: str, score: float) -> None:
        """Attach an LLM foresight score to a node."""
        self._assert_node_exists(node_id)
        self.graph.nodes[node_id]["llm_score"] = float(score)

    def get_all_valid_next_actions(self, node_id: str) -> List[str]:
        """List the immediate successors for the given action id."""
        self._assert_node_exists(node_id)
        return sorted(self.graph.successors(node_id))

    def render_to_png(self, path: Path) -> None:
        fig, ax = self.visualize_graph()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: Path) -> "ActionGraph":
        data = json.loads(Path(path).read_text())
        # If the file is just a list of edge scores (legacy), load a fresh graph then apply scores.
        if isinstance(data, list):
            # Legacy files contain no nodes; use a fully-connected scaffold so every edge id is legal.
            graph = build_cage4_turn_graph(init_mode="full")
            for edge in data:
                u, v = edge.get("from"), edge.get("to")
                if u in graph.graph and v in graph.graph and graph.graph.has_edge(u, v):
                    graph.graph.edges[u, v]["visit_count"] = edge.get("visit_count", 0)
                    graph.graph.edges[u, v]["mean_contribution"] = edge.get("mean_contribution", edge.get("average_reward", 0.0))
                    graph.graph.edges[u, v]["m2"] = edge.get("m2", 0.0)
                    # [avg_ep_reward disabled]
                    # graph.graph.edges[u, v]["avg_ep_reward"] = edge.get("avg_ep_reward", edge.get("average_reward", 0.0))
                    # graph.graph.edges[u, v]["ep_count"] = edge.get("ep_count", 0)
            return graph

        if not isinstance(data, dict):
            raise ValueError("Unrecognised ActionGraph JSON format.")

        try:
            version = int(data.get("format_version") or data.get("version") or 1)
        except Exception:
            version = 1

        graph = ActionGraph()
        for node in data.get("nodes", []) or []:
            action = ActionNode(
                action_id=node["action_id"],
                label=node["label"],
                agent_type=node["agent_type"],
                preconditions=node.get("preconditions"),
            )
            graph.add_action_node(action)
            if "llm_score" in node:
                graph.graph.nodes[action.action_id]["llm_score"] = node.get("llm_score")
        for edge in data.get("edges", []) or []:
            graph.graph.add_edge(
                edge["from"],
                edge["to"],
                description=edge.get("description"),
                visit_count=edge.get("visit_count", 0),
                mean_contribution=edge.get("mean_contribution", edge.get("average_reward", 0.0)),
                m2=edge.get("m2", 0.0),
                # [avg_ep_reward disabled]
                # avg_ep_reward=edge.get("avg_ep_reward", edge.get("average_reward", 0.0)),
                # ep_count=edge.get("ep_count", 0),
            )

        graph.reward_magnitude_anchor = float(data.get("reward_magnitude_anchor", 0.0))
        graph.baseline_default = float(data.get("baseline_default", 0.0))

        # v2+: optional state-conditioned stats stored separately from the NetworkX edge attrs.
        if version >= 2:
            for entry in data.get("state_edges", []) or []:
                if not isinstance(entry, dict):
                    continue
                state_sig = entry.get("state_signature")
                frm = entry.get("from")
                to = entry.get("to")
                if not isinstance(state_sig, str) or not isinstance(frm, str) or not isinstance(to, str):
                    continue
                if frm not in graph.graph or to not in graph.graph:
                    continue
                graph.set_state_edge_stats(
                    state_sig,
                    frm,
                    to,
                    visit_count=int(entry.get("visit_count", 0) or 0),
                    mean_contribution=float(entry.get("mean_contribution", entry.get("average_reward", 0.0)) or 0.0),
                    m2=float(entry.get("m2", 0.0) or 0.0),
                )
        return graph

    def reset(self) -> None:
        """Zero out stats."""
        for _, _, data in self.graph.edges(data=True):
            data["visit_count"] = 0
            data["mean_contribution"] = 0.0
            data["m2"] = 0.0
            # [avg_ep_reward disabled]
            # data["avg_ep_reward"] = 0.0
            # data["ep_count"] = 0
        self.state_edge_stats = {}

    def log_top_transitions(self, n: int = 10) -> List[Tuple[str, str, float, int]]:
        ranked = sorted(
            [
                (
                    u,
                    v,
                    data.get("mean_contribution", 0.0),
                    data.get("visit_count", 0),
                )
                for u, v, data in self.graph.edges(data=True)
            ],
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:n]

    def get_outgoing_edges(
        self,
        action_id: str,
        *,
        state_signature: Optional[str] = None,
        backoff: bool = True,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """Return (to_id, label, stats) for outgoing edges.

        If state_signature is provided, returns state-conditioned stats when available and
        falls back to global edge stats when `backoff=True`.
        """
        self._assert_node_exists(action_id)
        outgoing = []
        for _, to_id, _data in self.graph.edges(action_id, data=True):
            label = (
                self.graph.nodes[to_id]["action"].label
                if "action" in self.graph.nodes[to_id]
                else to_id
            )
            stats = self.get_edge_stats(
                action_id,
                to_id,
                state_signature=state_signature,
                backoff=backoff,
            )
            outgoing.append(
                (
                    to_id,
                    label,
                    stats,
                )
            )
        return outgoing

    def get_edge_stats(
        self,
        from_id: str,
        to_id: str,
        *,
        state_signature: Optional[str] = None,
        backoff: bool = True,
    ) -> Dict[str, Any]:
        """Return edge stats, optionally state-conditioned with global backoff."""
        self._assert_node_exists(from_id)
        self._assert_node_exists(to_id)

        if state_signature:
            state_stats = (
                self.state_edge_stats.get(state_signature, {})
                .get(from_id, {})
                .get(to_id)
            )
            if isinstance(state_stats, dict):
                return {
                    "visit_count": int(state_stats.get("visit_count", 0) or 0),
                    "mean_contribution": float(state_stats.get("mean_contribution", 0.0) or 0.0),
                    "m2": float(state_stats.get("m2", 0.0) or 0.0),
                    # [avg_ep_reward disabled]
                    # "avg_ep_reward": float(global_data.get("avg_ep_reward", 0.0) or 0.0),
                    # "ep_count": int(global_data.get("ep_count", 0) or 0),
                }

        if backoff and self.graph.has_edge(from_id, to_id):
            data = self.graph.edges[from_id, to_id]
            return {
                "visit_count": int(data.get("visit_count", 0) or 0),
                "mean_contribution": float(data.get("mean_contribution", 0.0) or 0.0),
                "m2": float(data.get("m2", 0.0) or 0.0),
                # [avg_ep_reward disabled]
                # "avg_ep_reward": float(data.get("avg_ep_reward", 0.0) or 0.0),
                # "ep_count": int(data.get("ep_count", 0) or 0),
            }

        return {"visit_count": 0, "mean_contribution": 0.0, "m2": 0.0}

    def set_state_edge_stats(
        self,
        state_signature: str,
        from_id: str,
        to_id: str,
        *,
        visit_count: int,
        mean_contribution: float,
        m2: float = 0.0,
    ) -> None:
        """Write state-conditioned stats for a particular (state_signature, edge)."""
        self._assert_node_exists(from_id)
        self._assert_node_exists(to_id)
        bucket = self.state_edge_stats.setdefault(str(state_signature), {}).setdefault(from_id, {})
        bucket[to_id] = {
            "visit_count": int(visit_count),
            "mean_contribution": float(mean_contribution),
            "m2": float(m2),
        }

    def set_bucket_edge_stats(
        self,
        bucket_id: str,
        from_id: str,
        to_id: str,
        *,
        visit_count: int,
        mean_contribution: float,
        m2: float = 0.0,
    ) -> None:
        """Write coarse-bucket-conditioned stats for a particular (bucket_id, edge)."""
        bucket = self.bucket_edge_stats.setdefault(str(bucket_id), {}).setdefault(from_id, {})
        bucket[to_id] = {
            "visit_count": int(visit_count),
            "mean_contribution": float(mean_contribution),
            "m2": float(m2),
        }

    def get_bucket_edge_stats(
        self,
        bucket_id: Optional[str],
        from_id: str,
        to_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return coarse-bucket-conditioned stats for an edge, or None if not found."""
        if not bucket_id:
            return None
        return self.bucket_edge_stats.get(str(bucket_id), {}).get(from_id, {}).get(to_id)

    def get_mixed_edge_stats(
        self,
        from_id: str,
        to_id: str,
        bucket_id: Optional[str],
        lambda_state: float,
        min_bucket_visits: int,
    ) -> Dict[str, Any]:
        """Return stats mixed between global and bucket-conditioned.

        Mirrors get_edge_stats return format. When bucket has enough visits, mixes
        global mean_contribution with bucket mean_contribution using lambda_state weight.
        """
        # Global stats (backoff=True gives us the NetworkX edge data)
        global_stats = self.get_edge_stats(from_id, to_id, state_signature=None, backoff=True)
        if not bucket_id:
            return global_stats
        bucket_raw = self.get_bucket_edge_stats(bucket_id, from_id, to_id)
        if bucket_raw is None:
            return global_stats
        vc_b = int(bucket_raw.get("visit_count", 0) or 0)
        if vc_b < int(min_bucket_visits):
            return global_stats  # lambda_eff = 0
        # lambda_eff = lambda_state
        mc_g = float(global_stats.get("mean_contribution", 0.0) or 0.0)
        mc_b = float(bucket_raw.get("mean_contribution", 0.0) or 0.0)
        vc_g = int(global_stats.get("visit_count", 0) or 0)
        m2_g = float(global_stats.get("m2", 0.0) or 0.0)
        la = float(lambda_state)
        return {
            "visit_count": vc_g,  # use global visit count for exploration bonus
            "mean_contribution": (1.0 - la) * mc_g + la * mc_b,
            "m2": m2_g,
            "_bucket_vc": vc_b,  # informational only
            # [avg_ep_reward disabled]
            # "avg_ep_reward": float(global_stats.get("avg_ep_reward", 0.0) or 0.0),
            # "ep_count": int(global_stats.get("ep_count", 0) or 0),
        }

    def get_nodes_by_agent(self, agent_type: str) -> List[str]:
        return [
            node for node, data in self.graph.nodes(data=True) if data.get("agent_type") == agent_type
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert the graph to a dictionary."""
        nodes = []
        for _, data in self.graph.nodes(data=True):
            node_dict = data["action"].to_dict()
            if "llm_score" in data:
                node_dict["llm_score"] = data["llm_score"]
            nodes.append(node_dict)
        edges = [
            {
                "from": u,
                "to": v,
                "description": data.get("description"),
                "visit_count": data.get("visit_count", 0),
                "mean_contribution": data.get("mean_contribution", 0.0),
                "m2": data.get("m2", 0.0),
                # [avg_ep_reward disabled]
                # "avg_ep_reward": data.get("avg_ep_reward", 0.0),
                # "ep_count": data.get("ep_count", 0),
            }
            for u, v, data in self.graph.edges(data=True)
        ]
        state_edges: List[Dict[str, Any]] = []
        for state_sig, from_map in self.state_edge_stats.items():
            for frm, to_map in from_map.items():
                for to, stats in to_map.items():
                    if not isinstance(stats, dict):
                        continue
                    state_edges.append(
                        {
                            "state_signature": state_sig,
                            "from": frm,
                            "to": to,
                            "visit_count": int(stats.get("visit_count", 0) or 0),
                            "mean_contribution": float(stats.get("mean_contribution", 0.0) or 0.0),
                            "m2": float(stats.get("m2", 0.0) or 0.0),
                        }
                    )

        result = {
            "format_version": ACTION_GRAPH_FORMAT_VERSION,
            "nodes": nodes,
            "edges": edges,
            "state_edges": state_edges,
        }
        result["reward_magnitude_anchor"] = self.reward_magnitude_anchor
        result["baseline_default"] = self.baseline_default
        return result

    def export_to_json(self, *, indent: int = 2) -> str:
        """Serialise the graph to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def visualize_graph(
        self,
        with_labels: bool = True,
        spring_k: Optional[float] = 1.2,
        iterations: int = 100,
        figsize: Tuple[int, int] = (14, 10),
        min_visits: int = 1,
    ):
        """Render the graph using matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for visualization; install or add it to requirements."
            ) from exc

        # Filter edges by visit_count
        filtered_graph = nx.DiGraph()
        for node, data in self.graph.nodes(data=True):
            filtered_graph.add_node(node, **data)
        for u, v, data in self.graph.edges(data=True):
            if data.get("visit_count", 0) >= min_visits:
                filtered_graph.add_edge(u, v, **data)

        if filtered_graph.number_of_edges() == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No edges with visits >= min_visits", ha="center", va="center")
            return fig, ax

        pos = nx.spring_layout(filtered_graph, seed=42, k=spring_k, iterations=iterations)
        node_colors = [
            "#1f77b4" if self._get_agent_type(node) == "defender" else "#d62728"
            for node in filtered_graph.nodes
        ]

        # Edge widths scaled by episode visit count (log scale, min 0.5)
        edge_widths = [
            max(0.5, math.log(data.get("visit_count", 0) + 1) * 1.5)
            for _, _, data in filtered_graph.edges(data=True)
        ]

        fig, ax = plt.subplots(figsize=figsize)
        nx.draw(
            filtered_graph,
            pos,
            ax=ax,
            with_labels=with_labels,
            node_color=node_colors,
            edge_color="#7a7a7a",
            width=edge_widths,
            linewidths=1.5,
            arrows=True,
        )
        # Labels: mean_contribution (2dp), visit_count (episodes), total_traversals (steps)
        edge_labels = {
            (u, v): (
                f"mc={data.get('mean_contribution', 0.0):.2f}\n"
                f"n={data.get('visit_count', 0)} tt={data.get('total_traversals', 0)}"
            )
            for u, v, data in filtered_graph.edges(data=True)
        }
        nx.draw_networkx_edge_labels(filtered_graph, pos, edge_labels=edge_labels, ax=ax, font_size=7)
        return fig, ax


def _compute_quality_scale(nx_graph, graph=None, state_signature=None, reward_magnitude_anchor: float = 0.0) -> float:
    """Compute robust spread of mean_contribution across all visited edges.

    Uses a magnitude-aware floor so the scale never collapses to a tiny value
    when the spread is near zero but absolute mean_contributions are large
    (cold-start scenario).

    S = max(S_mu, kappa * S_r, epsilon)
      S_mu  = abs(p90 - p10 spread of mean_contributions)
      S_r   = reward_magnitude_anchor if available, else median of absolute
              mean_contributions across visited edges (fallback)
      kappa = 0.3
      epsilon = 1e-6
    """
    import statistics as _stats

    values = []
    for frm, to, data in nx_graph.edges(data=True):
        if state_signature and hasattr(graph, "get_edge_stats"):
            stats = graph.get_edge_stats(frm, to, state_signature=state_signature, backoff=True)
            vc = int(stats.get("visit_count", 0) or 0)
            mc = float(stats.get("mean_contribution", 0.0) or 0.0)
        else:
            vc = int(data.get("visit_count", 0) or 0)
            mc = float(data.get("mean_contribution", 0.0) or 0.0)
        if vc >= 1:
            values.append(mc)

    kappa = 0.3
    epsilon = 1e-6

    if len(values) < 2:
        # Even with <2 visited edges, anchor to magnitude if available.
        median_abs = abs(values[0]) if values else 0.0
        s_r = reward_magnitude_anchor if reward_magnitude_anchor > 0 else median_abs
        return max(kappa * s_r, epsilon)

    values.sort()
    lo_idx = max(0, int(len(values) * 0.1))
    hi_idx = min(len(values) - 1, int(len(values) * 0.9))
    s_mu = abs(values[hi_idx] - values[lo_idx])

    median_abs = _stats.median(abs(v) for v in values)
    s_r = reward_magnitude_anchor if reward_magnitude_anchor > 0 else median_abs

    return max(s_mu, kappa * s_r, epsilon)


def select_topk_actions_by_edge_score(
    graph,
    k: int = 5,
    score_weight: float = 1.0,
    visit_weight: float = 0.3,
    state_signature: Optional[str] = None,
    bucket_id: Optional[str] = None,
    lambda_state: float = 0.3,
    min_bucket_visits: int = 5,
) -> List[str]:
    """
    Rank defender actions by their outgoing defender->attacker edges using combined quality+uncertainty score.

    Args:
        graph: ActionGraph or raw NetworkX DiGraph containing defender/attacker nodes.
        k: number of unique defender actions to return.
        score_weight: unused (kept for API compat).
        visit_weight: unused (kept for API compat).
        state_signature: optional state signature for state-conditioned stats.
        bucket_id: optional coarse bucket id for mixed scoring (takes precedence over state_signature).
        lambda_state: mixing weight for bucket stats when bucket visits sufficient.
        min_bucket_visits: minimum bucket visits before lambda_state takes effect.
    """
    if k <= 0:
        return []

    nx_graph = graph.graph if hasattr(graph, "graph") else graph

    def _is_defender(node_id: str) -> bool:
        if hasattr(graph, "is_defender_node"):
            return graph.is_defender_node(node_id)
        return nx_graph.nodes[node_id].get("agent_type") == "defender"

    def _is_attacker(node_id: str) -> bool:
        if hasattr(graph, "is_attacker_node"):
            return graph.is_attacker_node(node_id)
        return nx_graph.nodes[node_id].get("agent_type") == "attacker"

    defenders_all = [
        node for node, data in nx_graph.nodes(data=True) if data.get("agent_type") == "defender"
    ]

    rma = graph.reward_magnitude_anchor if hasattr(graph, 'reward_magnitude_anchor') else 0.0
    quality_scale = _compute_quality_scale(nx_graph, graph=graph, state_signature=state_signature, reward_magnitude_anchor=rma)
    pess_default = graph.baseline_default if hasattr(graph, 'baseline_default') else 0.0

    ranked_edges: List[Tuple[float, str, str]] = []
    for frm, to, data in nx_graph.edges(data=True):
        if not (_is_defender(frm) and _is_attacker(to)):
            continue
        if bucket_id and hasattr(graph, "get_mixed_edge_stats"):
            stats = graph.get_mixed_edge_stats(frm, to, bucket_id, lambda_state, min_bucket_visits)
            mc = float(stats.get("mean_contribution", 0.0) or 0.0)
            vc = int(stats.get("visit_count", 0) or 0)
            m2_val = float(stats.get("m2", 0.0) or 0.0)
        elif state_signature and hasattr(graph, "get_edge_stats"):
            stats = graph.get_edge_stats(frm, to, state_signature=state_signature, backoff=True)
            mc = float(stats.get("mean_contribution", 0.0) or 0.0)
            vc = int(stats.get("visit_count", 0) or 0)
            m2_val = float(stats.get("m2", 0.0) or 0.0)
        else:
            mc = float(data.get("mean_contribution", 0.0) or 0.0)
            vc = int(data.get("visit_count", 0) or 0)
            m2_val = float(data.get("m2", 0.0) or 0.0)
        rank_score = edge_combined_score(mc, vc, m2_val, quality_scale, pessimistic_default=pess_default)
        ranked_edges.append((rank_score, frm, to))

    if not ranked_edges:
        shuffled = list(defenders_all)[:max(k, 0)] if k > 0 else []
        random.shuffle(shuffled)
        return shuffled

    # Shuffle first so equal-score ties are randomly ordered (avoids alphabetical bias).
    random.shuffle(ranked_edges)
    ranked_edges.sort(key=lambda item: -item[0])  # stable sort preserves shuffle for ties

    selected: List[str] = []
    seen: set[str] = set()
    for _, frm, _ in ranked_edges:
        if len(selected) >= k:
            break
        if frm in seen:
            continue
        seen.add(frm)
        selected.append(frm)

    # Fill remaining slots with defenders not yet selected.
    if len(selected) < k:
        remaining = [d for d in defenders_all if d not in seen]

        filler: List[Tuple[float, str]] = []
        for node in remaining:
            if hasattr(graph, "get_outgoing_edges"):
                outgoing = [
                    (node, to_id, stats)
                    for to_id, _label, stats in graph.get_outgoing_edges(
                        node, state_signature=state_signature, backoff=True
                    )
                ]
            else:
                outgoing = list(nx_graph.edges(node, data=True))
            best_score = edge_combined_score(0.0, 0, 0.0, quality_scale, pessimistic_default=pess_default)  # default for nodes with no edges
            for ed in outgoing:
                mc = float(ed[2].get("mean_contribution", 0.0) or 0.0)
                vc = int(ed[2].get("visit_count", 0) or 0)
                m2_val = float(ed[2].get("m2", 0.0) or 0.0)
                s = edge_combined_score(mc, vc, m2_val, quality_scale, pessimistic_default=pess_default)
                if s > best_score:
                    best_score = s
            filler.append((-best_score, node))

        filler.sort(key=lambda item: (item[0], item[1]))
        for _, node in filler:
            if len(selected) >= k:
                break
            seen.add(node)
            selected.append(node)

    return selected


def build_cage4_turn_graph(
    include_terminal: bool = True,
    *,
    init_mode: Literal["full", "sparse"] = "sparse",
    sparse_m: int = 3,
    seed: int = 1337,
) -> ActionGraph:
    """Create an alternating-turn graph for all CAGE4 blue/red actions.

    Cold-start defaults to a *sparse* graph to reduce reliance on dense, fully-connected priors.
    Missing transitions can be discovered on-the-fly during rollouts (edge discovery).
    """
    defender_names, attacker_names = _get_cage4_action_names()
    defenders = tuple(
        ActionNode(
            action_id=f"defender_{_camel_to_snake(name)}",
            label=name,
            agent_type="defender",
        )
        for name in defender_names
    )
    attackers = tuple(
        ActionNode(
            action_id=f"attacker_{_camel_to_snake(name)}",
            label=name,
            agent_type="attacker",
        )
        for name in attacker_names
    )

    terminal_node = ActionNode(
        action_id="system_compromise",
        label="System compromise",
        agent_type="attacker",
        preconditions=None,
    )

    graph = ActionGraph()
    for action in list(defenders) + list(attackers):
        graph.add_action_node(action)
    if include_terminal:
        graph.add_action_node(terminal_node)

    init_mode = str(init_mode).lower().strip()
    if init_mode not in {"full", "sparse"}:
        raise ValueError("init_mode must be 'full' or 'sparse'")

    if init_mode == "full":
        for defender in defenders:
            for attacker in attackers:
                graph.add_edge(defender.action_id, attacker.action_id)
            if include_terminal:
                graph.add_edge(defender.action_id, terminal_node.action_id)

        for attacker in attackers:
            for defender in defenders:
                graph.add_edge(attacker.action_id, defender.action_id)
        return graph

    # Sparse init: connect each node to a small random subset on the opposite side.
    rng = random.Random(int(seed))
    defenders_ids = [d.action_id for d in defenders]
    attackers_ids = [a.action_id for a in attackers]
    m = max(1, int(sparse_m))

    for defender in defenders:
        subset = rng.sample(attackers_ids, k=min(m, len(attackers_ids))) if attackers_ids else []
        for attacker_id in subset:
            graph.add_edge(defender.action_id, attacker_id)
        if include_terminal:
            graph.add_edge(defender.action_id, terminal_node.action_id)

    for attacker in attackers:
        subset = rng.sample(defenders_ids, k=min(m, len(defenders_ids))) if defenders_ids else []
        for defender_id in subset:
            graph.add_edge(attacker.action_id, defender_id)

    return graph


def _get_cage4_action_names() -> Tuple[Sequence[str], Sequence[str]]:
    """Scrape the blue/red action lists from EnterpriseScenarioGenerator."""
    scenario_path = Path(__file__).resolve().parents[3] / "Simulator" / "Scenarios" / "EnterpriseScenarioGenerator.py"
    default_defender = (
        "AllowTrafficZone",
        "BlockTrafficZone",
        "Monitor",
        "Analyse",
        "Restore",
        "Remove",
        "DeployDecoy",
        "Sleep",
    )
    default_attacker = (
        "DiscoverRemoteSystems",
        "AggressiveServiceDiscovery",
        "StealthServiceDiscovery",
        "ExploitRemoteService",
        "PrivilegeEscalate",
        "DegradeServices",
        "DiscoverDeception",
        "Impact",
        "Withdraw",
        "Sleep",
    )

    if not scenario_path.exists():
        return default_defender, default_attacker

    try:
        tree = ast.parse(scenario_path.read_text())
        blue = _extract_action_list(tree, "blue_actions")
        red = _extract_action_list(tree, "red_actions")
        defender = tuple(blue or default_defender)
        attacker = tuple(red or default_attacker)
        return defender, attacker
    except Exception:
        return default_defender, default_attacker


def _extract_action_list(tree: ast.AST, target_name: str) -> List[str]:
    """Find a list assignment to target_name and return contained names."""
    found: List[str] = []

    class ListVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == target_name:
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Name):
                                found.append(elt.id)
                    return
            self.generic_visit(node)

    ListVisitor().visit(tree)
    return found


def _camel_to_snake(name: str) -> str:
    first_pass = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", first_pass).lower()
