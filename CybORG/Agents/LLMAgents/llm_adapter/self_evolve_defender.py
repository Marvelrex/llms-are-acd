import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from typing_extensions import Literal

from CybORG.Agents.LLMAgents.llm_adapter.action_graph import ActionGraph
from CybORG.Agents.LLMAgents.llm_adapter.state_signature import sig_to_bucket_id


@dataclass
class EdgeStats:
    visit_count: int = 0            # number of episode-level Welford updates (episode count)
    mean_contribution: float = 0.0  # running mean of frequency-boosted episode credit
    m2: float = 0.0                 # Welford's running sum of squared deviations (for variance)
    total_traversals: int = 0       # cumulative step-level traversal count (logging only, not for UCB)

    @property
    def variance(self) -> float:
        if self.visit_count < 2:
            return 0.0
        return self.m2 / (self.visit_count - 1)

    def update(self, value: float) -> None:
        """Update running mean and variance using Welford's online algorithm.

        After Fix 1, each call to update() represents ONE episode's aggregated credit,
        so visit_count tracks episode count, not step count.
        """
        self.visit_count += 1
        delta = value - self.mean_contribution
        self.mean_contribution += delta / self.visit_count
        delta2 = value - self.mean_contribution
        self.m2 += delta * delta2


class RewardBaseline:
    """Per-bucket episode reward baseline using EMA + Welford variance.

    Uses hierarchical fallback: per-bucket when sufficient data exists,
    global otherwise. This prevents fragile early estimates in sparse buckets.
    """

    def __init__(self, ema_beta: float = 0.03, min_bucket_count: int = 5):
        self.ema_beta = float(ema_beta)
        self.min_bucket_count = int(min_bucket_count)
        # Each entry: {"mean": float, "var_m2": float, "n": int}
        self.buckets: Dict[str, Dict] = {}
        self.global_stats: Dict = {"mean": 0.0, "var_m2": 0.0, "n": 0}

    def get(self, bucket_id: Optional[str]) -> Tuple[float, float]:
        """Returns (baseline_mean, baseline_std) for a bucket.

        Falls back to global baseline if bucket has fewer than
        min_bucket_count observations.
        """
        bucket = self.buckets.get(str(bucket_id)) if bucket_id else None
        if bucket and bucket["n"] >= self.min_bucket_count:
            stats = bucket
        else:
            stats = self.global_stats

        if stats["n"] < 2:
            return (stats["mean"], 1.0)  # default std=1.0 to avoid divide-by-zero

        variance = stats["var_m2"] / (stats["n"] - 1)
        return (stats["mean"], max(variance ** 0.5, 1e-8))

    def update(self, bucket_id: Optional[str], episode_reward: float) -> None:
        """Update both the bucket-specific and global baselines.

        Call this AFTER using the baseline for credit assignment,
        so the current episode doesn't influence its own baseline.
        """
        targets = [self.global_stats]
        if bucket_id:
            targets.append(self._ensure_bucket(str(bucket_id)))
        for stats in targets:
            stats["n"] += 1
            # EMA mean update
            stats["mean"] = (1 - self.ema_beta) * stats["mean"] + self.ema_beta * episode_reward
            # Welford variance update
            delta = episode_reward - stats["mean"]
            stats["var_m2"] += delta * (episode_reward - stats["mean"])

    def _ensure_bucket(self, bucket_id: str) -> Dict:
        if bucket_id not in self.buckets:
            self.buckets[bucket_id] = {"mean": 0.0, "var_m2": 0.0, "n": 0}
        return self.buckets[bucket_id]


class SelfEvolveDefender:
    """Adaptive scorer/pruner operating on an ActionGraph."""

    def __init__(
        self,
        action_graph: ActionGraph,
        reward_decay: float = 0.8,
        default_success_reward: float = 1.0,
        default_failure_penalty: float = 0.1,
        min_visits_to_prune: int = 3,
        success_threshold: float = -0.5,
        max_failure_penalty: Optional[float] = None,
        alpha: float = 0.5,
        *,
        use_discounted_credit: bool = True,
        credit_gamma: float = 0.92,
        use_baseline: bool = True,
        baseline_beta: float = 0.2,
        baseline_scope: Literal["global", "role"] = "global",
        use_state_conditioning: bool = True,
        ema_beta: float = 0.03,
        reward_clip: float = 3.0,
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
        self._reward_baseline = RewardBaseline(ema_beta=ema_beta)
        self._reward_clip = float(reward_clip)

        self.edge_stats: Dict[str, Dict[str, EdgeStats]] = defaultdict(dict)
        self.state_edge_stats: Dict[str, Dict[str, Dict[str, EdgeStats]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # Coarse bucket-conditioned edge stats: bucket_id -> from_id -> to_id -> EdgeStats.
        # Aggregates credits from all full state signatures sharing the same coarse bucket,
        # achieving ~25x faster accumulation than full-signature conditioning.
        self._bucket_stats: Dict[str, Dict[str, Dict[str, EdgeStats]]] = {}

        # Advantage baseline (EMA over episode rewards).
        self._baseline_by_key: Dict[str, float] = defaultdict(float)

        self._bootstrap_edge_stats_from_graph()
        self._bootstrap_state_edge_stats_from_graph()

    def _bootstrap_edge_stats_from_graph(self) -> None:
        """Seed edge_stats from existing graph metadata so persisted counts aren't reset each run."""
        for frm, to, data in self.graph.graph.edges(data=True):
            visit_count = int(data.get("visit_count", 0) or 0)
            mc = float(data.get("mean_contribution", 0.0) or 0.0)
            m2_val = float(data.get("m2", 0.0) or 0.0)
            # [avg_ep_reward disabled]
            # ep_count = int(data.get("ep_count", 0) or 0)
            # ep_reward = float(data.get("avg_ep_reward", 0.0) or 0.0)
            if visit_count <= 0 and mc == 0.0:
                continue
            stats = EdgeStats(
                visit_count=visit_count,
                mean_contribution=mc if visit_count else 0.0,
                m2=m2_val if visit_count else 0.0,
                # episode_count=ep_count,
                # episode_reward_mean=ep_reward if ep_count else 0.0,
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
                    visit_count = int(stats.get("visit_count", 0) or 0)
                    mc = float(stats.get("mean_contribution", 0.0) or 0.0)
                    m2_val = float(stats.get("m2", 0.0) or 0.0)
                    if visit_count <= 0 and mc == 0.0:
                        continue
                    self.state_edge_stats[state_sig][frm][to] = EdgeStats(
                        visit_count=visit_count,
                        mean_contribution=mc if visit_count else 0.0,
                        m2=m2_val if visit_count else 0.0,
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

    def _sync_bucket_stats_to_graph(self) -> None:
        """Bulk-sync in-memory bucket stats to ActionGraph.bucket_edge_stats."""
        if not hasattr(self.graph, "set_bucket_edge_stats"):
            return
        for bucket_id, from_dict in self._bucket_stats.items():
            for frm, to_dict in from_dict.items():
                for to, stats in to_dict.items():
                    if self.graph.graph.has_edge(frm, to):
                        self.graph.set_bucket_edge_stats(
                            bucket_id,
                            frm,
                            to,
                            visit_count=stats.visit_count,
                            mean_contribution=stats.mean_contribution,
                            m2=stats.m2,
                        )

    def _sync_global_stats_to_graph(self, frm: str, to: str, stats: EdgeStats) -> None:
        edge = self.graph.graph.edges[frm, to]
        edge["visit_count"] = int(stats.visit_count)   # episode count (UCB n)
        edge["mean_contribution"] = float(stats.mean_contribution)
        edge["m2"] = float(stats.m2)
        edge["total_traversals"] = int(stats.total_traversals)  # step count (logging only)

    def _sync_state_stats_to_graph(self, state_sig: str, frm: str, to: str, stats: EdgeStats) -> None:
        self.graph.set_state_edge_stats(
            state_sig,
            frm,
            to,
            visit_count=int(stats.visit_count),
            mean_contribution=float(stats.mean_contribution),
            m2=float(stats.m2),
        )

    def observe_round(
        self,
        trace: List[str],
        result: Dict[str, Any],
        *,
        state_signatures: Optional[Sequence[Optional[str]]] = None,
    ) -> Dict[str, Any]:
        """Record outcomes for each transition in the trace.

        Fix 1: Episode-level frequency-boosted credit. Each unique edge is updated ONCE per
        episode with credit = clipped_advantage * frequency_boost * time_discount.
        - frequency_boost = traversal_count / total_steps recovers the old average_reward system's
          natural frequency-weighting (edges traversed often in good episodes accumulate more).
        - Updates are per-episode (visit_count tracks episodes, not steps), keeping UCB n calibrated.
        - State/bucket conditioning updates each (sig, frm, to) triple once per episode.
        """
        if len(trace) < 2:
            return {
                "total_reward": float(result.get("reward", 0.0) or 0.0),
                "baseline": self._reward_baseline.get(None)[0],
                "advantage": 0.0,
                "edges_updated": 0,
                "state_edges_updated": 0,
            }

        reward_val = float(result.get("reward", 0.0) or 0.0)
        T = len(trace) - 1  # number of steps (edges) in episode

        # --- 1. Determine bucket_id for baseline lookup (use most-common bucket in episode) ---
        bucket_id_for_baseline: Optional[str] = None
        if state_signatures:
            bucket_counts: Dict[str, int] = {}
            for sig in state_signatures:
                if sig:
                    bid = sig_to_bucket_id(sig)
                    if bid:
                        bucket_counts[bid] = bucket_counts.get(bid, 0) + 1
            if bucket_counts:
                bucket_id_for_baseline = max(bucket_counts, key=bucket_counts.__getitem__)

        # --- 2. Compute normalized, clipped advantage ---
        baseline_mean, baseline_std = self._reward_baseline.get(bucket_id_for_baseline)
        raw_advantage = reward_val - baseline_mean
        normalized_advantage = raw_advantage / (baseline_std + 1e-8)
        clipped_advantage = max(min(normalized_advantage, self._reward_clip), -self._reward_clip)

        # --- 3. Count traversals per (frm, to) and collect step indices ---
        # global: (frm, to) -> [step_indices]
        global_step_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # state-conditioned: (state_sig, frm, to) -> [step_indices]
        state_step_indices: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        # bucket-conditioned: (bucket_id, frm, to) -> [step_indices]
        bucket_step_indices: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

        for t in range(T):
            frm, to = trace[t], trace[t + 1]
            if not self.graph.graph.has_edge(frm, to):
                continue
            global_step_indices[(frm, to)].append(t)
            if state_signatures is not None and t < len(state_signatures):
                sig = state_signatures[t]
                if isinstance(sig, str) and sig:
                    state_step_indices[(sig, frm, to)].append(t)
                    bid = sig_to_bucket_id(sig)
                    if bid:
                        bucket_step_indices[(bid, frm, to)].append(t)

        edges_updated = 0
        state_edges_updated = 0

        if self.use_discounted_credit:
            # --- 4. Update global edge stats (ONCE per edge per episode) ---
            for (frm, to), step_idx in global_step_indices.items():
                count = len(step_idx)
                frequency_boost = count / T
                avg_t = sum(step_idx) / count
                time_discount = self.credit_gamma ** (T - avg_t)
                credit = clipped_advantage * frequency_boost * time_discount

                stats = self.edge_stats[frm].get(to)
                if stats is None:
                    data = self.graph.graph.edges[frm, to]
                    vc = int(data.get("visit_count", 0) or 0)
                    mc = float(data.get("mean_contribution", 0.0) or 0.0)
                    m2v = float(data.get("m2", 0.0) or 0.0)
                    tt = int(data.get("total_traversals", 0) or 0)
                    stats = EdgeStats(
                        visit_count=vc,
                        mean_contribution=mc if vc else 0.0,
                        m2=m2v if vc else 0.0,
                        total_traversals=tt,
                    )
                stats.update(credit)           # episode-level Welford: visit_count tracks episodes
                stats.total_traversals += count  # cumulative step count (logging only)
                self.edge_stats[frm][to] = stats
                self._sync_global_stats_to_graph(frm, to, stats)
                edges_updated += 1

            # --- 5. Update state-conditioned edge stats (ONCE per (sig, frm, to) per episode) ---
            if self.use_state_conditioning:
                for (sig, frm, to), step_idx in state_step_indices.items():
                    count = len(step_idx)
                    frequency_boost = count / T
                    avg_t = sum(step_idx) / count
                    time_discount = self.credit_gamma ** (T - avg_t)
                    credit = clipped_advantage * frequency_boost * time_discount

                    s_stats = self.state_edge_stats[sig][frm].get(to, EdgeStats())
                    s_stats.update(credit)
                    self.state_edge_stats[sig][frm][to] = s_stats
                    self._sync_state_stats_to_graph(sig, frm, to, s_stats)
                    state_edges_updated += 1

            # --- 6. Update bucket-conditioned edge stats (ONCE per (bucket, frm, to) per episode) ---
            for (bid, frm, to), step_idx in bucket_step_indices.items():
                count = len(step_idx)
                frequency_boost = count / T
                avg_t = sum(step_idx) / count
                time_discount = self.credit_gamma ** (T - avg_t)
                credit = clipped_advantage * frequency_boost * time_discount

                bucket_es = self._bucket_stats.setdefault(bid, {}).setdefault(frm, {})
                if to not in bucket_es:
                    bucket_es[to] = EdgeStats()
                bucket_es[to].update(credit)

        else:
            # Legacy fallback: update only unique edges to avoid repeated debits/credits on long traces.
            # Uses old per-role baseline (not RewardBaseline) for backward compat.
            roles_in_trace = {
                self._role_from_action(trace[i])
                for i in range(T)
            } or {"global"}
            advantage_map: Dict[str, float] = {}
            for role in roles_in_trace:
                b_prev = float(self._baseline_by_key.get(role, 0.0))
                adv = reward_val - b_prev if self.use_baseline else reward_val
                advantage_map[role] = adv
                if self.use_baseline:
                    self._baseline_by_key[role] = self.ema_update(b_prev, reward_val, self.baseline_beta)

            unique_edges = list({(trace[i], trace[i + 1]) for i in range(T)})
            edge_to_state_sig: Dict[Tuple[str, str], str] = {}
            if self.use_state_conditioning and state_signatures is not None:
                for i in range(T):
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

                stats = self.edge_stats[frm].get(to)
                if stats is None:
                    data = self.graph.graph.edges[frm, to]
                    vc = int(data.get("visit_count", 0) or 0)
                    mc = float(data.get("mean_contribution", 0.0) or 0.0)
                    m2v = float(data.get("m2", 0.0) or 0.0)
                    stats = EdgeStats(
                        visit_count=vc,
                        mean_contribution=mc if vc else 0.0,
                        m2=m2v if vc else 0.0,
                    )
                stats.update(adv_val)
                self.edge_stats[frm][to] = stats
                self._sync_global_stats_to_graph(frm, to, stats)
                edges_updated += 1
                state_sig = edge_to_state_sig.get((frm, to))
                if isinstance(state_sig, str) and state_sig:
                    s_stats = self.state_edge_stats[state_sig][frm].get(to, EdgeStats())
                    s_stats.update(adv_val)
                    self.state_edge_stats[state_sig][frm][to] = s_stats
                    self._sync_state_stats_to_graph(state_sig, frm, to, s_stats)
                    state_edges_updated += 1

                    bucket_id = sig_to_bucket_id(state_sig)
                    if bucket_id:
                        bucket_es = self._bucket_stats.setdefault(bucket_id, {}).setdefault(frm, {})
                        if to not in bucket_es:
                            bucket_es[to] = EdgeStats()
                        bucket_es[to].update(adv_val)

        # Store reward magnitude anchor for exploration scaling (use absolute reward value).
        if hasattr(self.graph, 'reward_magnitude_anchor'):
            self.graph.reward_magnitude_anchor = max(abs(reward_val), getattr(self.graph, 'reward_magnitude_anchor', 0.0))

        # Store baseline mean as pessimistic default for unvisited edges.
        if hasattr(self.graph, 'baseline_default'):
            self.graph.baseline_default = baseline_mean

        # Update baseline AFTER using it for this episode's credit assignment.
        self._reward_baseline.update(bucket_id_for_baseline, reward_val)

        # Sync coarse bucket stats to the graph for use in get_mixed_edge_stats.
        self._sync_bucket_stats_to_graph()

        return {
            "total_reward": reward_val,
            "baseline": baseline_mean,
            "advantage": float(raw_advantage),
            "edges_updated": int(edges_updated),
            "state_edges_updated": int(state_edges_updated),
        }

    def prune_graph(self, threshold: float) -> List[Tuple[str, str]]:
        """Prune edges with low mean_contribution once they have enough observations."""
        removed: List[Tuple[str, str]] = []
        for frm, to_stats in list(self.edge_stats.items()):
            for to, stats in list(to_stats.items()):
                if stats.visit_count < self.min_visits_to_prune:
                    continue
                if stats.mean_contribution >= threshold:
                    continue
                if self.graph.graph.has_edge(frm, to):
                    self.graph.graph.remove_edge(frm, to)
                removed.append((frm, to))
                del self.edge_stats[frm][to]
            if not self.edge_stats[frm]:
                del self.edge_stats[frm]
        return removed

    def print_top_transitions(self, n: int = 10) -> None:
        """Print the best transitions by mean_contribution."""
        ranked = sorted(
            self._iter_stats(),
            key=lambda x: x[2].mean_contribution,
            reverse=True,
        )[:n]
        for frm, to, stats in ranked:
            print(
                f"{frm} -> {to}: mean_contrib={stats.mean_contribution:.2f}, "
                f"visits={stats.visit_count}"
            )

    def export_edge_scores_to_json(self) -> str:
        """Return a JSON string of edge stats."""
        payload = []
        for frm, to, stats in self._iter_stats():
            payload.append(
                {
                    "from": frm,
                    "to": to,
                    "visit_count": stats.visit_count,
                    "mean_contribution": stats.mean_contribution,
                    "m2": stats.m2,
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
                            "visit_count": int(stats.visit_count),
                            "mean_contribution": float(stats.mean_contribution),
                        }
                    )
            edges.sort(key=lambda e: (e["mean_contribution"], e["visit_count"]), reverse=True)
            out[state_sig] = edges[: max(0, int(n))]
        return out

    def plot_score_distribution(self):
        """Plot a histogram of average rewards (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError("matplotlib is required for plotting score distribution.") from exc

        averages = [stats.mean_contribution for _, _, stats in self._iter_stats()]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(averages, bins=15, color="#1f77b4", alpha=0.8)
        ax.set_title("Edge Mean Contribution Distribution")
        ax.set_xlabel("Mean Contribution")
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
