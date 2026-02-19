import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from typing_extensions import Literal

from CybORG.Agents.LLMAgents.llm_adapter.action_graph import ActionGraph


@dataclass
class EdgeStats:
    # Smoothed "edge score" used by priors/ranking.
    score: float = 0.0
    visit_count: int = 0
    total_reward: float = 0.0
    average_reward: float = 0.0

    def update(self, score_delta: float, raw_reward: float, *, decay: float, alpha: float) -> None:
        """Update stats separating advantage-based score from raw-reward logging."""
        self.visit_count += 1
        # Score tracks (possibly advantage-weighted) deltas.
        self.score = float(self.score) + float(alpha) * (float(score_delta) - float(self.score))
        # Raw reward is kept for diagnostics; no advantage applied here.
        self.total_reward = self.total_reward * decay + float(raw_reward)
        self.average_reward = self.total_reward / self.visit_count


class SelfEvolveDefender:
    """Adaptive scorer/pruner operating on an ActionGraph."""

    def __init__(
        self,
        action_graph: ActionGraph,
        reward_decay: float = 0.9,
        default_success_reward: float = 1.0,
        default_failure_penalty: float = 0.1,
        min_visits_to_prune: int = 3,
        success_threshold: float = -0.5,
        max_failure_penalty: Optional[float] = None,
        alpha: float = 0.3,
        *,
        use_discounted_credit: bool = True,
        credit_gamma: float = 0.97,
        use_baseline: bool = True,
        baseline_beta: float = 0.05,
        baseline_scope: Literal["global", "role"] = "global",
        use_state_conditioning: bool = True,
    ) -> None:
        self.graph = action_graph
        self.reward_decay = reward_decay
        self.default_success_reward = default_success_reward
        self.default_failure_penalty = default_failure_penalty
        self.min_visits_to_prune = min_visits_to_prune
        self.success_threshold = success_threshold
        self.max_failure_penalty = max_failure_penalty
        self.alpha = alpha
        self.use_discounted_credit = bool(use_discounted_credit)
        self.credit_gamma = float(credit_gamma)
        self.use_baseline = bool(use_baseline)
        self.baseline_beta = float(baseline_beta)
        self.baseline_scope = baseline_scope
        self.use_state_conditioning = bool(use_state_conditioning)

        self.edge_stats: Dict[str, Dict[str, EdgeStats]] = defaultdict(dict)
        self.state_edge_stats: Dict[str, Dict[str, Dict[str, EdgeStats]]] = defaultdict(
            lambda: defaultdict(dict)
        )

        # Advantage baseline (EMA over episode rewards).
        self._baseline_by_key: Dict[str, float] = defaultdict(float)

        self._bootstrap_edge_stats_from_graph()
        self._bootstrap_state_edge_stats_from_graph()

    def _bootstrap_edge_stats_from_graph(self) -> None:
        """Seed edge_stats from existing graph metadata so persisted counts aren't reset each run."""
        for frm, to, data in self.graph.graph.edges(data=True):
            score = float(data.get("score", 0.0) or 0.0)
            visit_count = int(data.get("visit_count", 0) or 0)
            avg = float(data.get("average_reward", 0.0) or 0.0)
            if score == 0.0 and visit_count <= 0 and avg == 0.0:
                continue
            total_reward = avg * visit_count if visit_count else 0.0
            stats = EdgeStats(
                score=score,
                visit_count=visit_count,
                total_reward=total_reward,
                average_reward=avg if visit_count else 0.0,
            )
            self.edge_stats[frm][to] = stats

    def _bootstrap_state_edge_stats_from_graph(self) -> None:
        """Seed state_edge_stats from persisted ActionGraph.state_edge_stats (if present)."""
        state_blob = getattr(self.graph, "state_edge_stats", None)
        if not isinstance(state_blob, dict):
            return

        for state_sig, from_map in state_blob.items():
            if not isinstance(state_sig, str) or not isinstance(from_map, dict):
                continue
            for frm, to_map in from_map.items():
                if not isinstance(frm, str) or not isinstance(to_map, dict):
                    continue
                for to, stats in to_map.items():
                    if not isinstance(to, str) or not isinstance(stats, dict):
                        continue
                    score = float(stats.get("score", 0.0) or 0.0)
                    visit_count = int(stats.get("visit_count", 0) or 0)
                    avg = float(stats.get("average_reward", 0.0) or 0.0)
                    if score == 0.0 and visit_count <= 0 and avg == 0.0:
                        continue
                    total_reward = avg * visit_count if visit_count else 0.0
                    self.state_edge_stats[state_sig][frm][to] = EdgeStats(
                        score=score,
                        visit_count=visit_count,
                        total_reward=total_reward,
                        average_reward=avg if visit_count else 0.0,
                    )

    # Bug-7 fix: minimum weight floor so early-episode actions still receive
    # a meaningful learning signal.  Without a floor, w_0 = gamma^(T-1) can be
    # as low as 0.05 for T=100, gamma=0.97, starving early edges of credit.
    _WEIGHT_FLOOR: float = 0.2

    @staticmethod
    def discounted_weights(num_edges: int, gamma: float, *, floor: float = 0.2) -> List[float]:
        """Return normalized time-discounted weights with a minimum weight floor.

        For t in [0, T-1], w_t = max(gamma^(T-1-t), floor).
        Weights are normalized to sum to 1.
        """
        T = int(num_edges)
        if T <= 0:
            return []
        g = float(gamma)
        fl = float(floor)
        raw = [max(g ** (T - 1 - t), fl) for t in range(T)]
        s = float(sum(raw))
        if s <= 0:
            return [1.0 / T for _ in range(T)]
        return [float(w) / s for w in raw]

    @staticmethod
    def ema_update(old: float, new: float, beta: float) -> float:
        """Exponential moving average update."""
        b = float(beta)
        return (1.0 - b) * float(old) + b * float(new)

    def _role_from_action(self, action_id: str) -> str:
        if self.baseline_scope == "global":
            return "global"
        if action_id.startswith("defender_"):
            return "defender"
        if action_id.startswith("attacker_"):
            return "attacker"
        try:
            if self.graph.is_defender_node(action_id):
                return "defender"
            if self.graph.is_attacker_node(action_id):
                return "attacker"
        except Exception:
            pass
        return "global"

    def _sync_global_stats_to_graph(self, frm: str, to: str, stats: EdgeStats) -> None:
        edge = self.graph.graph.edges[frm, to]
        edge["score"] = float(stats.score)
        edge["visit_count"] = int(stats.visit_count)
        edge["average_reward"] = float(stats.average_reward)

    def _sync_state_stats_to_graph(self, state_sig: str, frm: str, to: str, stats: EdgeStats) -> None:
        # ActionGraph persists state-conditioned stats separately.
        self.graph.set_state_edge_stats(
            state_sig,
            frm,
            to,
            score=float(stats.score),
            visit_count=int(stats.visit_count),
            average_reward=float(stats.average_reward),
        )

    def observe_round(
        self,
        trace: List[str],
        result: Dict[str, Any],
        *,
        state_signatures: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """Record outcomes for each transition in the trace.

        When use_discounted_credit is enabled, assign episode-level advantage across *edge occurrences*
        using normalized time-discounted weights. Optionally condition updates on a state_signature
        aligned with the "from" node of each edge occurrence.
        """
        if len(trace) < 2:
            return {
                "total_reward": float(result.get("reward", 0.0) or 0.0),
                "baseline": float(self._baseline_by_key.get(self._baseline_key(), 0.0)),
                "advantage": 0.0,
                "edges_updated": 0,
                "state_edges_updated": 0,
            }

        reward_val = float(result.get("reward", 0.0) or 0.0)

        # Compute per-role baselines / advantages (single update per observe_round).
        roles_in_trace = {
            self._role_from_action(trace[i])
            for i in range(len(trace) - 1)
        } if len(trace) >= 2 else set()
        if not roles_in_trace:
            roles_in_trace = {"global"}

        baseline_before_map: Dict[str, float] = {}
        advantage_map: Dict[str, float] = {}

        for role in roles_in_trace:
            b_prev = float(self._baseline_by_key.get(role, 0.0))
            baseline_before_map[role] = b_prev
            adv = reward_val - b_prev if self.use_baseline else reward_val
            advantage_map[role] = adv
            if self.use_baseline:
                self._baseline_by_key[role] = self.ema_update(b_prev, reward_val, self.baseline_beta)

        # For reporting, pick the first role (deterministic order) or global.
        report_role = sorted(roles_in_trace)[0]
        baseline_before = baseline_before_map.get(report_role, 0.0)
        advantage = advantage_map.get(report_role, reward_val)

        edges_updated = 0
        state_edges_updated = 0

        if self.use_discounted_credit:
            T = len(trace) - 1
            weights = self.discounted_weights(T, self.credit_gamma)
            for t in range(T):
                frm, to = trace[t], trace[t + 1]
                role = self._role_from_action(frm)
                adv_val = advantage_map.get(role, reward_val)
                state_sig = None
                if state_signatures is not None and t < len(state_signatures):
                    state_sig = state_signatures[t]
                delta = float(adv_val) * float(weights[t])
                if not self.graph.graph.has_edge(frm, to):
                    continue

                stats = self.edge_stats[frm].get(to)
                if stats is None:
                    data = self.graph.graph.edges[frm, to]
                    visit_count = int(data.get("visit_count", 0) or 0)
                    avg = float(data.get("average_reward", 0.0) or 0.0)
                    stats = EdgeStats(
                        score=float(data.get("score", 0.0) or 0.0),
                        visit_count=visit_count,
                        total_reward=avg * visit_count if visit_count else 0.0,
                        average_reward=avg if visit_count else 0.0,
                    )

                stats.update(delta, reward_val, decay=self.reward_decay, alpha=self.alpha)
                self.edge_stats[frm][to] = stats
                self._sync_global_stats_to_graph(frm, to, stats)
                edges_updated += 1

                if self.use_state_conditioning and isinstance(state_sig, str) and state_sig:
                    s_stats = self.state_edge_stats[state_sig][frm].get(to, EdgeStats())
                    s_stats.update(delta, reward_val, decay=self.reward_decay, alpha=self.alpha)
                    self.state_edge_stats[state_sig][frm][to] = s_stats
                    self._sync_state_stats_to_graph(state_sig, frm, to, s_stats)
                    state_edges_updated += 1
        else:
            # Legacy fallback: update only unique edges to avoid repeated debits/credits on long traces.
            penalty_val = float(result.get("penalty", 0.0) or self.default_failure_penalty)
            unique_edges = list({(trace[i], trace[i + 1]) for i in range(len(trace) - 1)})
            num_edges = max(len(unique_edges), 1)
            edge_to_state_sig: Dict[Tuple[str, str], str] = {}
            if self.use_state_conditioning and state_signatures is not None:
                for i in range(len(trace) - 1):
                    if i >= len(state_signatures):
                        break
                    sig = state_signatures[i]
                    if isinstance(sig, str) and sig:
                        edge_to_state_sig[(trace[i], trace[i + 1])] = sig

            for frm, to in unique_edges:
                if not self.graph.graph.has_edge(frm, to):
                    continue
                role = self._role_from_action(frm)
                adv_val = advantage_map.get(role, reward_val)
                score_delta = adv_val / num_edges
                if not self.use_baseline and reward_val <= 0 and self.max_failure_penalty is not None:
                    score_delta = max(score_delta, -abs(self.max_failure_penalty))
                stats = self.edge_stats[frm].get(to)
                if stats is None:
                    data = self.graph.graph.edges[frm, to]
                    visit_count = int(data.get("visit_count", 0) or 0)
                    avg = float(data.get("average_reward", 0.0) or 0.0)
                    stats = EdgeStats(
                        score=float(data.get("score", 0.0) or 0.0),
                        visit_count=visit_count,
                        total_reward=avg * visit_count if visit_count else 0.0,
                        average_reward=avg if visit_count else 0.0,
                    )
                stats.update(score_delta, reward_val, decay=self.reward_decay, alpha=self.alpha)
                self.edge_stats[frm][to] = stats
                self._sync_global_stats_to_graph(frm, to, stats)
                edges_updated += 1
                state_sig = edge_to_state_sig.get((frm, to))
                if isinstance(state_sig, str) and state_sig:
                    s_stats = self.state_edge_stats[state_sig][frm].get(to, EdgeStats())
                    s_stats.update(score_delta, reward_val, decay=self.reward_decay, alpha=self.alpha)
                    self.state_edge_stats[state_sig][frm][to] = s_stats
                    self._sync_state_stats_to_graph(state_sig, frm, to, s_stats)
                    state_edges_updated += 1

        return {
            "total_reward": reward_val,
            "baseline": baseline_before,
            "advantage": float(advantage),
            "edges_updated": int(edges_updated),
            "state_edges_updated": int(state_edges_updated),
        }

    def prune_graph(self, threshold: float) -> List[Tuple[str, str]]:
        """Prune edges with low average reward once they have enough observations."""
        removed: List[Tuple[str, str]] = []
        for frm, to_stats in list(self.edge_stats.items()):
            for to, stats in list(to_stats.items()):
                if stats.visit_count < self.min_visits_to_prune:
                    continue
                if stats.average_reward >= threshold:
                    continue
                if self.graph.graph.has_edge(frm, to):
                    self.graph.graph.remove_edge(frm, to)
                removed.append((frm, to))
                del self.edge_stats[frm][to]
            if not self.edge_stats[frm]:
                del self.edge_stats[frm]
        return removed

    def update_edge_score(self, from_id: str, to_id: str, delta_score: float) -> None:
        """Manual hook to bump an edge score."""
        self.graph.update_edge_score(from_id, to_id, delta_score)

    def get_edge_score(self, from_id: str, to_id: str) -> float:
        return self.graph.get_edge_score(from_id, to_id)

    def print_top_transitions(self, n: int = 10) -> None:
        """Print the best transitions by average reward."""
        ranked = sorted(
            self._iter_stats(),
            key=lambda x: x[2].average_reward,
            reverse=True,
        )[:n]
        for frm, to, stats in ranked:
            print(
                f"{frm} -> {to}: avg={stats.average_reward:.2f}, "
                f"visits={stats.visit_count}"
            )

    def export_edge_scores_to_json(self) -> str:
        """Return a JSON string of edge stats and scores."""
        payload = []
        for frm, to, stats in self._iter_stats():
            payload.append(
                {
                    "from": frm,
                    "to": to,
                    "score": self.get_edge_score(frm, to) if self.graph.graph.has_edge(frm, to) else None,
                    "visit_count": stats.visit_count,
                    "average_reward": stats.average_reward,
                }
            )
        return json.dumps(payload, indent=2)

    def top_state_edges(
        self,
        state_signatures: Iterable[str],
        *,
        n: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return a small summary of top edges per provided state signature."""
        out: Dict[str, List[Dict[str, Any]]] = {}
        for state_sig in state_signatures:
            edges: List[Dict[str, Any]] = []
            for frm, to_map in self.state_edge_stats.get(state_sig, {}).items():
                for to, stats in to_map.items():
                    edges.append(
                        {
                            "from": frm,
                            "to": to,
                            "score": float(stats.score),
                            "visit_count": int(stats.visit_count),
                            "average_reward": float(stats.average_reward),
                        }
                    )
            edges.sort(key=lambda e: (e["score"], e["visit_count"]), reverse=True)
            out[state_sig] = edges[: max(0, int(n))]
        return out

    def plot_score_distribution(self):
        """Plot a histogram of average rewards (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting score distribution.") from exc

        averages = [stats.average_reward for _, _, stats in self._iter_stats()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(averages, bins=15, color="#1f77b4", alpha=0.8)
        ax.set_title("Edge Average Reward Distribution")
        ax.set_xlabel("Average Reward")
        ax.set_ylabel("Count")
        return fig, ax

    def _iter_stats(self) -> Iterable[Tuple[str, str, EdgeStats]]:
        for frm, to_stats in self.edge_stats.items():
            for to, stats in to_stats.items():
                yield frm, to, stats

    @staticmethod
    def _pairwise(items: List[str]) -> Iterable[Tuple[str, str]]:
        for i in range(len(items) - 1):
            yield items[i], items[i + 1]
