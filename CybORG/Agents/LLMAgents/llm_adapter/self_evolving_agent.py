from __future__ import annotations

import json
import re
import random
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math
import numpy as np

from CybORG.Agents import BaseAgent
from CybORG.Agents.LLMAgents.llm_adapter.action_graph import (
    ActionGraph,
    build_cage4_turn_graph,
    _camel_to_snake,
    select_topk_actions_by_edge_score,
    edge_combined_score,
    _compute_quality_scale,
)
from CybORG.Agents.LLMAgents.llm_adapter.graph_agent_config import GraphAgentConfig
from CybORG.Agents.LLMAgents.llm_adapter.state_signature import compute_state_signature, sig_to_bucket_id
from CybORG.Agents.LLMAgents.llm_adapter.self_evolve_defender import SelfEvolveDefender
from CybORG.Agents.LLMAgents.llm_policy import LLMDefenderPolicy
from CybORG.Simulator.Actions import Action, Sleep
from CybORG.Agents.LLMAgents.llm_adapter import obs_formatter
from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger
from CybORG.Shared.Enums import TernaryEnum


def _ucb_score(mean: float, n: int, N: int, c: float) -> float:
    """UCB1 score for an edge.

    Args:
        mean: Welford running mean of frequency-boosted episode credit.
        n: number of episodes this edge was updated (visit_count).
        N: total episode updates across all edges from this action (sum of visit_counts).
        c: exploration coefficient (config.ucb_c, default 1.0).

    Returns inf for n<2 to guarantee under-visited edges are explored first.
    """
    if n < 2:
        return float('inf')
    return mean + c * math.sqrt(math.log(N + 1) / (n + 1))


def _map_to_prior_scale(ucb_scores: List[float], tau: float = 1.0) -> List[float]:
    """Map raw UCB scores to [1, 10] using robust IQR-sigmoid normalization.

    Strategy:
    - Use median + IQR for robustness to outliers.
    - Sigmoid squashes z-scores to (0, 1), then scales to [1, 10].
    - Falls back to rank-based mapping when IQR≈0 or <3 candidates.
      The rank-based fallback guarantees spread even during cold-start.

    Args:
        ucb_scores: raw UCB scores for all candidates (may contain inf for unvisited).
        tau: sigmoid sharpness. Lower = sharper separation.

    Returns:
        List of prior scores in [1, 10], same order as input.
    """
    scores = np.array(ucb_scores, dtype=float)

    # Replace inf values with max_finite + 1 (unvisited edges explored first).
    finite_mask = np.isfinite(scores)
    if not finite_mask.all():
        max_finite = float(scores[finite_mask].max()) if finite_mask.any() else 0.0
        scores[~finite_mask] = max_finite + 1.0

    if len(scores) < 3:
        # Too few candidates — rank-based fallback guarantees spread.
        ranks = np.argsort(np.argsort(scores)).astype(float)
        n_candidates = max(len(scores) - 1, 1)
        return (1.0 + 9.0 * ranks / n_candidates).tolist()

    median = float(np.median(scores))
    q75, q25 = float(np.percentile(scores, 75)), float(np.percentile(scores, 25))
    iqr = q75 - q25

    if iqr < 1e-8:
        # All scores nearly identical — rank-based fallback still guarantees spread.
        ranks = np.argsort(np.argsort(scores)).astype(float)
        n_candidates = max(len(scores) - 1, 1)
        return (1.0 + 9.0 * ranks / n_candidates).tolist()

    z_scores = (scores - median) / (iqr + 1e-8)
    sigmoid_scores = 1.0 / (1.0 + np.exp(-z_scores / tau))
    priors = 1.0 + 9.0 * sigmoid_scores
    return priors.tolist()


def _compute_alpha(
    prior_gap: float,
    alpha_min: float = 0.4,
    alpha_max: float = 0.8,
    alpha_midpoint: float = 1.0,
    alpha_steepness: float = 2.0,
) -> float:
    """Compute graph weight in blend based on prior confidence.

    The prior_gap is max_prior - second_best_prior on [1, 10] scale.
    Larger gap → graph more confident → graph gets more weight.

    Returns alpha in [alpha_min, alpha_max]:
    - alpha_min: flat priors, LLM leads with light graph input.
    - alpha_max: sharp priors, graph strongly guides.
    The LLM ALWAYS retains at least (1 - alpha_max) = 20% weight.
    """
    raw = alpha_min + (alpha_max - alpha_min) * (
        1.0 / (1.0 + math.exp(-alpha_steepness * (prior_gap - alpha_midpoint)))
    )
    return min(max(raw, alpha_min), alpha_max)


def _blend_scores(
    priors: Dict[str, float],
    llm_ranking: List[str],
    alpha: float,
) -> Dict[str, float]:
    """Weighted blend of graph prior and LLM rank-based score.

    Converts LLM ranking to [1, 10] (best=10, worst=1), then blends
    with graph priors: blended = alpha * prior + (1-alpha) * llm_score.

    Args:
        priors: graph-computed prior scores [1, 10], keyed by action name.
        llm_ranking: LLM's preferred ordering, best first.
        alpha: graph weight in [alpha_min, alpha_max].

    Returns:
        Blended scores keyed by action name; higher = better.
    """
    n = len(llm_ranking)
    llm_scores: Dict[str, float] = {}
    for rank, action in enumerate(llm_ranking):
        llm_scores[action] = 10.0 - (9.0 * rank / max(n - 1, 1)) if n > 1 else 10.0

    blended: Dict[str, float] = {}
    for action, prior in priors.items():
        llm_score = llm_scores.get(action, 5.0)  # midpoint for unranked actions
        blended[action] = alpha * prior + (1.0 - alpha) * llm_score
    return blended


def action_to_node_id(agent_name: str, action: Action) -> str:
    prefix = "defender" if "blue" in agent_name else "attacker"
    return f"{prefix}_{_camel_to_snake(action.__class__.__name__)}"


class SelfEvolvingGraphAgent(BaseAgent):
    """Blue agent that uses an action graph with adaptive scoring/pruning and an LLM for action selection."""

    def __init__(
        self,
        name: str,
        graph: ActionGraph,
        reward_decay: float = 0.8,
        prune_threshold: float = -3.0,
        log_dir: Optional[Path] = None,
        persist_path: Optional[Path] = None,
        snapshot_every: int = 0,
        candidate_top_k: Optional[int] = None,
        candidate_score_weight: float = 1.0,
        candidate_visit_weight: float = 0.3,
        config: Optional[GraphAgentConfig] = None,
    ):
        super().__init__(name)
        self.config = config or GraphAgentConfig()
        self.graph: ActionGraph = graph
        self.learner = SelfEvolveDefender(
            self.graph,
            reward_decay=reward_decay,
            min_visits_to_prune=3,
            use_discounted_credit=self.config.use_discounted_credit,
            credit_gamma=self.config.credit_gamma,
            use_baseline=self.config.use_baseline,
            baseline_beta=self.config.baseline_beta,
            baseline_scope=self.config.baseline_scope,
            use_state_conditioning=self.config.use_state_conditioning,
            ema_beta=float(getattr(self.config, 'ema_beta', 0.03)),
            reward_clip=float(getattr(self.config, 'reward_clip', 3.0)),
        )
        self.prune_threshold = prune_threshold
        self.trace: List[str] = []
        # Optional per-node state signature aligned with self.trace indices (signature refers to the "from" node).
        self.state_trace: List[Optional[str]] = []
        self._episode_state_signatures: set[str] = set()
        self._new_edges_discovered: int = 0
        self.last_action_node: Optional[str] = None
        self.llm_policy = LLMDefenderPolicy([], None, {"agent_name": name})
        base_log_dir = log_dir or Path(__file__).resolve().parent / "logs"
        timestamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.log_dir = base_log_dir / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.trace_log_path = self.log_dir / "self_evolve_trace.jsonl"
        self.persist_path = persist_path or (base_log_dir / "graph_scores.json")
        self.snapshot_every = snapshot_every
        self.episode_counter = 0
        self.last_action: Optional[Action] = None
        self._actions_cache: List[Action] = []
        self._action_labels_cache: List[str] = []
        self._last_llm_scores: Optional[Dict[str, float]] = None
        self._last_llm_conf: Optional[Dict[str, float]] = None
        self.candidate_top_k = candidate_top_k or int(self.config.candidate_top_k_base)
        self.candidate_score_weight = candidate_score_weight
        self.candidate_visit_weight = candidate_visit_weight
        self._episode_finalized: bool = False
        # Deterministic round-robin counters for parameterized actions (hostname/subnet choice).
        self._rr_index_by_action_id: Dict[str, int] = {}
        # When an action takes multiple ticks, the simulator ignores new actions until completion.
        # Avoid spending tokens during that window and avoid clobbering last_action_node on forced Sleep ticks.
        self._waiting_in_progress: bool = False
        # Incident-response mode: once compromise is detected within an episode, bias towards remediation.
        self._post_compromise_mode: bool = False
        self._compromised_hosts: List[str] = []
        # Defender-step index of the most recent compromise signal (used to decay post-compromise mode).
        self._last_compromise_step: Optional[int] = None
        # Track which compromise alert strings we've already used as evidence (to avoid "sticky" signals
        # when wrappers repeat historical messages each step).
        self._seen_compromise_messages: set[str] = set()
        self._post_compromise_steps_since_restore: int = 0
        # Restore anti-spam tracking.
        self._last_restore_step: Optional[int] = None
        self._last_restore_status: Optional[str] = None
        # Defender-step index (exclusive) until which Restore should be de-biased (cooldown).
        self._restore_cooldown_until_step: int = 0
        # One-time warning guard for prompt sanitizer regressions.
        self._warned_banned_prompt_tokens: bool = False
        # Bug-5 fix: track recently targeted hosts to avoid deterministic lock-in.
        # Each entry is (hostname, defender_step_idx).
        self._recently_targeted_hosts: deque[Tuple[str, int]] = deque(maxlen=20)
        # Diversity + traffic safety trackers.
        self._candidate_history: deque[list[str]] = deque(maxlen=int(self.config.candidate_diversity_window))
        self._recent_def_actions: deque[str] = deque(maxlen=32)
        self._traffic_history: deque[Tuple[str, str, str, int]] = deque(maxlen=20)  # (action_id, from, to, step)
        self._traffic_last_pair: Optional[Tuple[str, str]] = None
        self._defender_step_counter: int = 0

    @staticmethod
    def _coerce_ternary(value: Any) -> Optional[TernaryEnum]:
        """Best-effort normalize common success encodings into TernaryEnum."""
        if isinstance(value, TernaryEnum):
            return value
        if isinstance(value, bool):
            return TernaryEnum.TRUE if value else TernaryEnum.FALSE
        if isinstance(value, str):
            s = value.strip().upper()
            if s in {"TRUE", "FALSE", "IN_PROGRESS"}:
                try:
                    return TernaryEnum[s]
                except Exception:
                    return None
        return None

    def _update_last_action_outcome(self, observation: Dict[str, Any]) -> None:
        """No-op: streak tracking removed (most CAGE-4 actions return FALSE normally)."""
        pass

    def _apply_false_repeat_penalty(
        self, observation: Dict[str, Any], scores: Dict[str, float]
    ) -> Dict[str, float]:
        """No-op: streak penalties removed. Returns empty dict for logging compatibility."""
        return {}

    _REMEDIATION_ACTIONS = {"defender_remove", "defender_restore"}
    _ANALYSE_ACTION = "defender_analyse"
    _MAX_CONSECUTIVE_ANALYSE = 3

    def _apply_analyse_cap(
        self,
        per_action_scores: Dict[str, Dict[str, float]],
        candidates: List[str],
    ) -> bool:
        """If Analyse has been chosen >= _MAX_CONSECUTIVE_ANALYSE times in a row,
        boost Remove/Restore scores and penalize Analyse to force remediation.
        Returns True if the cap was applied."""
        recent = list(self._recent_def_actions)
        if len(recent) < self._MAX_CONSECUTIVE_ANALYSE:
            return False

        tail = recent[-self._MAX_CONSECUTIVE_ANALYSE:]
        if not all(a == self._ANALYSE_ACTION for a in tail):
            return False

        # Cap triggered: boost remediation, penalize Analyse
        remediation_in_candidates = [c for c in candidates if c in self._REMEDIATION_ACTIONS]
        if not remediation_in_candidates:
            return False

        for cid, entry in per_action_scores.items():
            if cid == self._ANALYSE_ACTION:
                # Penalize Analyse heavily
                entry["score"] = max(1.0, entry["score"] - 3.0)
                entry["confidence"] = max(0.1, entry["confidence"] * 0.5)
            elif cid in self._REMEDIATION_ACTIONS:
                # Boost Remove/Restore
                entry["score"] = min(10.0, entry["score"] + 2.0)
                entry["confidence"] = min(1.0, entry["confidence"] + 0.2)
        return True

    def _detect_compromise(self, observation: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Best-effort compromise detection driven by observation (cheap + deterministic).

        Returns:
            compromise_detected: True if we see a compromise signal.
            compromised_hosts: List of hostnames extracted from messages when available.
        """
        if not isinstance(observation, dict):
            return False, []

        msg = observation.get("message", None)
        compromise_detected = False
        compromised_hosts: List[str] = []

        # Prefer explicit commvector compromise bits [5-6] when available for *this agent*.
        def _to_bits(v: Any) -> List[int]:
            try:
                return [1 if bool(x) else 0 for x in v]
            except Exception:
                return []

        # Known hostnames to de-noise regex extraction.
        reserved = {"success", "action", "phase", "message"}
        known_hosts: set[str] = set()
        for key, value in observation.items():
            if key in reserved or not isinstance(value, dict):
                continue
            known_hosts.add(str(key))
            sysinfo = value.get("System info")
            if isinstance(sysinfo, dict):
                hn = sysinfo.get("Hostname")
                if hn:
                    known_hosts.add(str(hn))
        for act in self._actions_cache:
            hn = getattr(act, "hostname", None)
            if isinstance(hn, str) and hn:
                known_hosts.add(hn)

        def _dedup_keep_order(items: List[str]) -> List[str]:
            seen: set[str] = set()
            out: List[str] = []
            for it in items:
                if it in seen:
                    continue
                seen.add(it)
                out.append(it)
            return out

        def _extract_hosts(text: str) -> List[str]:
            if not isinstance(text, str) or not text:
                return []

            hits: List[str] = []

            # Common "Hostname: ..." patterns used in formatted observations/logs.
            for m in re.findall(r"\bHostname\s*[:=]\s*([\w.-]+)\b", text):
                hits.append(str(m))

            # CC4-style hostnames include "_host_<n>".
            for m in re.findall(r"\b[\w.-]*_host_\d+\b", text):
                hits.append(str(m))

            # Routers are also explicit hosts in CC4.
            for m in re.findall(r"\b[\w.-]+_router\b", text):
                hits.append(str(m))

            if known_hosts:
                hits = [h for h in hits if h in known_hosts]

            return _dedup_keep_order(hits)

        def _message_strings(v: Any) -> List[str]:
            if isinstance(v, str):
                return [v]
            if isinstance(v, (list, tuple)):
                return [x for x in v if isinstance(x, str)]
            return []

        def _looks_like_bitvector(v: Any) -> bool:
            if v is None or isinstance(v, (str, bytes, dict)):
                return False
            if isinstance(v, (list, tuple)) and any(isinstance(x, (str, bytes, dict)) for x in v):
                return False
            return hasattr(v, "__len__")

        def _pick_self_commvector(v: Any) -> Optional[Any]:
            # Some wrappers provide a *single* commvector as the message.
            if _looks_like_bitvector(v) and not isinstance(v, (list, tuple)):
                return v

            # If message is a list of commvectors and includes self, it is typically length 5.
            if not isinstance(v, (list, tuple)):
                return None
            if len(v) == 1 and _looks_like_bitvector(v[0]):
                return v[0]
            try:
                self_idx = int(str(self.name)[-1])
            except Exception:
                self_idx = None
            if self_idx is not None and len(v) == 5 and 0 <= self_idx < len(v) and _looks_like_bitvector(v[self_idx]):
                return v[self_idx]
            return None

        self_cv = _pick_self_commvector(msg)
        if self_cv is not None:
            bits = _to_bits(self_cv)
            if len(bits) >= 7 and (bits[5] or bits[6]):
                compromise_detected = True

        # Fallback: parse message strings for compromise phrases + hostnames.
        for s in _message_strings(msg):
            sl = s.lower()
            if ("user-level compromise" in sl) or ("admin-level compromise" in sl):
                # Some wrappers repeat historical alert strings every step; treat each distinct
                # alert string as a *single* compromise signal to avoid sticky incident mode.
                if s not in self._seen_compromise_messages:
                    compromise_detected = True
                    self._seen_compromise_messages.add(s)
            compromised_hosts.extend(_extract_hosts(s))

        # Final fallback: derive compromise from the observation itself (no commvector required).
        # Keep this conservative and deterministic.
        ioc_user = {"cmd.sh", "cmd.exe"}
        ioc_admin = {"escalate.sh", "escalate.exe"}
        proc_compromise_hosts: List[str] = []
        for key, value in observation.items():
            if key in reserved or not isinstance(value, dict):
                continue

            hostname = str(key)
            sysinfo = value.get("System info")
            if isinstance(sysinfo, dict) and sysinfo.get("Hostname"):
                hostname = str(sysinfo.get("Hostname"))

            host_compromised = False

            files = value.get("Files")
            if isinstance(files, list):
                for f in files:
                    if not isinstance(f, dict):
                        continue
                    fname = f.get("File Name")
                    if fname in ioc_admin or fname in ioc_user:
                        host_compromised = True
                        break

            sessions = value.get("Sessions")
            if not host_compromised and isinstance(sessions, list) and len(sessions) > 1:
                host_compromised = True

            procs = value.get("Processes")
            if not host_compromised and isinstance(procs, list):
                remotes: set[str] = set()
                for proc in procs:
                    if not isinstance(proc, dict):
                        continue
                    if "Connections" not in proc and len(proc) < 2:
                        host_compromised = True
                        break
                    conns = proc.get("Connections")
                    if isinstance(conns, list) and conns:
                        ra = conns[0].get("remote_address")
                        if ra:
                            ra_s = str(ra)
                            if ra_s in remotes:
                                host_compromised = True
                                break
                            remotes.add(ra_s)

            if host_compromised:
                compromise_detected = True
                proc_compromise_hosts.append(hostname)

        compromised_hosts.extend(proc_compromise_hosts)

        return bool(compromise_detected), _dedup_keep_order(compromised_hosts)

    def get_action(self, observation: Dict[str, Any], action_space) -> Action:
        """Select an action via LLM, constrained by the graph if an edge is pruned."""
        # New episode begins once we request an action; allow finalize_episode to run once later.
        self._episode_finalized = False
        self._update_last_action_outcome(observation)

        last_status = self._coerce_ternary(observation.get("success", None))

        # If the previous action is still executing, the simulator will ignore new actions anyway.
        # Returning Sleep avoids wasting tokens and keeps last_action pointing at the in-flight action.
        if last_status == TernaryEnum.IN_PROGRESS:
            self._waiting_in_progress = True
            return Sleep()
        self._waiting_in_progress = False

        # Defender-step index for this *decision* (counts prior defender actions in trace).
        defender_step_idx = sum(1 for nid in self.trace if str(nid).startswith("defender_"))
        self._defender_step_counter = defender_step_idx

        # Update restore outcome tracking (status corresponds to the *previous* action).
        restore_id = "defender_restore"
        if self.last_action_node == restore_id and last_status in (TernaryEnum.TRUE, TernaryEnum.FALSE):
            self._last_restore_status = str(last_status)
            # If Restore succeeded on a tracked compromised host, drop it to avoid repeated restores.
            if last_status == TernaryEnum.TRUE:
                try:
                    restored_host = getattr(self.last_action, "hostname", None)
                except Exception:
                    restored_host = None
                if isinstance(restored_host, str) and restored_host:
                    self._compromised_hosts = [h for h in self._compromised_hosts if h != restored_host]

        state_signature: Optional[str] = None
        if self.config.use_state_conditioning:
            state_signature = compute_state_signature(
                observation,
                last_action_status=observation.get("success", None),
                step_idx=defender_step_idx,
                action_space=action_space,
                config=self.config.state_signature_config,
            )
            if state_signature:
                self._episode_state_signatures.add(state_signature)

        # Coarse bucket_id (~36 possible values) for faster-accumulating mixed priors.
        bucket_id: Optional[str] = sig_to_bucket_id(state_signature) if state_signature else None

        prev_node = self.trace[-1] if self.trace else None
        # Adaptive candidate K based on recent diversity.
        recent_unique = len(set(aid for bucket in self._candidate_history for aid in bucket))
        effective_k = self.candidate_top_k
        if recent_unique and recent_unique < int(self.config.candidate_diversity_floor):
            effective_k = self.candidate_top_k + int(self.config.candidate_top_k_boost)
        candidates = select_topk_actions_by_edge_score(
            self.graph,
            k=effective_k,
            score_weight=self.candidate_score_weight,
            visit_weight=self.candidate_visit_weight,
            state_signature=state_signature if self.config.use_state_conditioning else None,
            bucket_id=bucket_id,
            lambda_state=float(getattr(self.config, "lambda_state", 0.3)),
            min_bucket_visits=int(getattr(self.config, "min_bucket_visits", 5)),
        )
        if not candidates:
            candidates = self.graph.get_nodes_by_agent("defender")
        num_candidates_before = len(candidates)

        # If edge discovery is disabled, ensure we only propose defender actions that are reachable
        # from the current trace node (avoids post-hoc Sleep fallbacks on sparse graphs).
        if prev_node and not self.config.use_edge_discovery and prev_node in self.graph.graph:
            reachable = [cid for cid in candidates if self.graph.graph.has_edge(prev_node, cid)]
            if reachable:
                candidates = reachable
            else:
                # Fallback to any reachable defender successors (may expand candidate set).
                succ = [
                    nid
                    for nid in self.graph.get_all_valid_next_actions(prev_node)
                    if self.graph.is_defender_node(nid)
                ]
                if succ:
                    candidates = succ

        # Filter to actions that are actually available in the current action space when possible.
        available_ids = self._available_action_ids(action_space)
        if available_ids:
            candidates = [cid for cid in candidates if cid in available_ids]
            if not candidates:
                # Top-K may be empty after filtering; fall back to all available defender actions.
                in_graph = sorted(
                    [
                        cid
                        for cid in available_ids
                        if str(cid).startswith("defender_") and cid in self.graph.graph
                    ]
                )
                candidates = in_graph or sorted(
                    [cid for cid in available_ids if str(cid).startswith("defender_")]
                )
                if not candidates:
                    candidates = self.graph.get_nodes_by_agent("defender")

        # Epsilon candidate exploration (adds options; LLM still dominates).
        eps_cand_used = False
        num_candidates_after_filter = len(candidates)
        if available_ids:
            eps = float(getattr(self.config, "candidate_eps", 0.0) or 0.0)
            add_n = max(0, int(getattr(self.config, "candidate_eps_n", 0) or 0))
            if eps > 0 and add_n > 0:
                try:
                    if random.random() < eps:
                        pool = [cid for cid in available_ids if cid not in candidates and str(cid).startswith("defender_")]
                        random.shuffle(pool)
                        extra = pool[:add_n]
                        if extra:
                            candidates = list(dict.fromkeys(candidates + extra))
                            eps_cand_used = True
                except Exception:
                    pass

        # Bug-6 fix: consolidated traffic gate replaces four separate filters.
        disruptive = self._DISRUPTIVE_ACTIONS
        has_traffic_evidence, traffic_evidence = self._has_strong_traffic_evidence(observation)
        cooldown_active = False
        if self.config.traffic_gate_enabled:
            filtered = [
                cid for cid in candidates
                if self._should_allow_traffic_action(
                    cid, has_traffic_evidence=has_traffic_evidence
                )
            ]
            if filtered and len(filtered) < len(candidates):
                cooldown_active = True
                candidates = filtered

        if not candidates and available_ids:
            candidates = [cid for cid in available_ids if str(cid).startswith("defender_")]
        if not candidates:
            candidates = self.graph.get_nodes_by_agent("defender")

        num_candidates_after_diversify = len(candidates)
        traffic_candidates_included = any(cid in disruptive for cid in candidates)

        priors = self._graph_priors_batch(
            candidates,
            state_signature=state_signature if self.config.use_state_conditioning else None,
            bucket_id=bucket_id,
        )
        prompt_raw = self.build_llm_prompt(
            observation,
            candidates,
            priors,
            state_signature=state_signature if self.config.use_state_conditioning else None,
            bucket_id=bucket_id,
        )
        prompt_contains_banned_pre = self._prompt_contains_banned_tokens(prompt_raw)
        if prompt_contains_banned_pre and not self._warned_banned_prompt_tokens:
            Logger.warning(
                f"Banned incident tokens detected in prompt; sanitizing. tokens={list(getattr(self.config, 'banned_prompt_tokens', ()))!r}"
            )
            self._warned_banned_prompt_tokens = True

        prompt = self._sanitize_prompt(prompt_raw)
        prompt_contains_banned_post = self._prompt_contains_banned_tokens(prompt)

        response = self.llm_policy.model_manager.generate_response(prompt)
        (
            ranked_actions,
            llm_confidence,
            justification,
            disruption_risk,
            llm_valid,
            parse_debug,
            per_action_scores,
        ) = self.parse_llm_response(
            response, candidates
        )

        llm_confidence = float(min(1.0, max(0.0, float(llm_confidence))))
        conf_min = float(getattr(self.config, "llm_conf_min", 0.55) or 0.55)

        # Graph-only fallback ranking: priors boosted by exploration bonus for
        # low-evidence actions.  Bug-9 fix: visits_evidence now feeds into
        # the fallback scores instead of being dead diagnostic code.
        visits_evidence: Dict[str, float] = {
            cid: float(
                self._visits_evidence(
                    cid,
                    state_signature=state_signature if self.config.use_state_conditioning else None,
                )
            )
            for cid in candidates
        }
        final_scores: Dict[str, float] = {}
        for cid in candidates:
            prior = float(priors.get(cid, 5.5))
            ev = float(visits_evidence.get(cid, 0.0))
            # Exploration bonus: boost under-visited actions (low evidence → higher bonus).
            exploration_bonus = 1.0 - ev  # ev is in [0, 1]
            final_scores[cid] = prior + exploration_bonus
        false_repeat_penalties = self._apply_false_repeat_penalty(observation, final_scores)

        # Fix 3: Compute adaptive blend weight from prior gap.
        sorted_prior_vals = sorted(priors.values(), reverse=True)
        prior_gap = (sorted_prior_vals[0] - sorted_prior_vals[1]) if len(sorted_prior_vals) >= 2 else 0.0
        alpha = _compute_alpha(
            prior_gap,
            alpha_min=float(getattr(self.config, 'alpha_min', 0.4)),
            alpha_max=float(getattr(self.config, 'alpha_max', 0.8)),
            alpha_midpoint=float(getattr(self.config, 'alpha_midpoint', 1.0)),
            alpha_steepness=float(getattr(self.config, 'alpha_steepness', 2.0)),
        )

        gate_g: Optional[float] = None
        gate_info: Dict[str, Any] = {}
        chosen_from = "GRAPH_FALLBACK"
        traffic_gate_applied = False
        traffic_gate_reason = ""
        analyse_cap_applied = False

        invalid_reasons: List[str] = []
        if not llm_valid:
            invalid_reasons.append("invalid_llm")

        chosen_id: str
        traffic_score_caps: Dict[str, Dict[str, float]] = {}
        best_from_llm: Optional[str] = None
        best_after_postprocess: Optional[str] = None

        # Fix 3: Always take the LLM path if valid (no confidence gate).
        # Blend graph priors and LLM ranking with adaptive alpha.
        if llm_valid and ranked_actions:
            chosen_from = "graph_guided_llm"
            if self.config.disable_graph:
                gate_g = 0.0  # pure LLM when graph disabled
                gate_info = {"gate_g": 0.0, "graph_disabled": True, "alpha": 0.0}
                for cid in candidates:
                    final_scores[cid] = float(priors.get(cid, 5.5))
            else:
                # Compute gate_g for legacy diagnostics (not used for gating decisions).
                gate_g, gate_info = self._compute_circuit_breaker_gate(llm_confidence, graph_priors=priors)

            # Adaptive blend: final_scores = alpha * prior + (1-alpha) * llm_rank_score
            if ranked_actions and not self.config.disable_graph:
                blended = _blend_scores(priors, ranked_actions, alpha)
                for aid in blended:
                    final_scores[aid] = blended[aid]
                    # Keep per_action_scores aligned.
                    if per_action_scores and aid in per_action_scores:
                        per_action_scores[aid]["score"] = final_scores[aid]

            # Default traffic risk to high unless we have explicit traffic evidence this step.
            if not has_traffic_evidence:
                for aid in disruptive:
                    if str(disruption_risk.get(aid, "")).lower() not in {"low", "med", "high"}:
                        disruption_risk[aid] = "high"

            if isinstance(parse_debug, dict) and isinstance(parse_debug.get("best", None), str):
                best_from_llm = str(parse_debug.get("best"))

            # Keep per_action_scores consistent with the current ranking if the response used a legacy schema.
            if not per_action_scores and ranked_actions:
                per_action_scores = {
                    str(aid): {"score": float(max(1.0, 10.0 - float(idx))), "confidence": float(llm_confidence)}
                    for idx, aid in enumerate(ranked_actions)
                    if isinstance(aid, str)
                }

            # Sanity-filter disruptive actions when risk is high and no trigger, and keep scores aligned.
            if disruption_risk:
                ranked_actions = self._filter_disruptive_by_risk(ranked_actions, disruption_risk, justification)
                if per_action_scores:
                    per_action_scores = {aid: per_action_scores[aid] for aid in ranked_actions if aid in per_action_scores}

            # Bug-6 fix: post-LLM traffic filtering uses the consolidated gate.
            # Only filter traffic actions the LLM ranked if the consolidated gate rejects them.
            if self.config.traffic_gate_enabled:
                filtered_rank = [
                    aid for aid in ranked_actions
                    if self._should_allow_traffic_action(
                        aid,
                        has_traffic_evidence=has_traffic_evidence,
                        llm_confidence=float(
                            (per_action_scores or {}).get(aid, {}).get("confidence", llm_confidence)
                        ),
                    )
                ]
                if filtered_rank and len(filtered_rank) < len(ranked_actions):
                    ranked_actions = filtered_rank
                    traffic_gate_applied = True
                    traffic_gate_reason = "consolidated_traffic_gate"
                    if per_action_scores:
                        per_action_scores = {aid: per_action_scores[aid] for aid in ranked_actions if aid in per_action_scores}

            # Analyse cap: if Analyse repeated too many times, force remediation.
            analyse_cap_applied = self._apply_analyse_cap(per_action_scores, candidates) if per_action_scores else False

            # Ensure ranking is derived from numeric scores (score desc, confidence desc, id stable).
            if per_action_scores:
                ranked_actions = sorted(
                    list(per_action_scores.keys()),
                    key=lambda a: (
                        -float(per_action_scores[a]["score"]),
                        -float(per_action_scores[a]["confidence"]),
                        str(a),
                    ),
                )
                if ranked_actions:
                    best_after_postprocess = str(ranked_actions[0])

            # Choose the best action after post-processing when available; else fall back to the LLM "best".
            available_ranked = (
                [aid for aid in ranked_actions if aid in available_ids] if available_ids else list(ranked_actions)
            )
            preferred = (
                best_after_postprocess
                if isinstance(best_after_postprocess, str) and best_after_postprocess in available_ranked
                else (best_from_llm if isinstance(best_from_llm, str) and best_from_llm in available_ranked else None)
            )
            if preferred is None and available_ranked:
                preferred = str(available_ranked[0])
            if preferred and preferred in available_ranked:
                available_ranked = [preferred] + [aid for aid in available_ranked if aid != preferred]

            picked = self._pick_from_ranking(available_ranked, last_status=last_status)
            if picked is None:
                chosen_from = "GRAPH_FALLBACK"
                picked = self.select_top_action(final_scores, candidates)
            chosen_id = str(picked)
        else:
            # Invalid LLM (parse failure) → graph priors only, no LLM influence.
            chosen_from = "GRAPH_FALLBACK"
            for cid in candidates:
                final_scores[cid] = float(priors.get(cid, 5.5))
            # Apply analyse cap to graph fallback scores.
            recent = list(self._recent_def_actions)
            if (len(recent) >= self._MAX_CONSECUTIVE_ANALYSE
                    and all(a == self._ANALYSE_ACTION for a in recent[-self._MAX_CONSECUTIVE_ANALYSE:])):
                for cid in final_scores:
                    if cid == self._ANALYSE_ACTION:
                        final_scores[cid] -= 3.0
                    elif cid in self._REMEDIATION_ACTIONS:
                        final_scores[cid] += 2.0
            chosen_id = self.select_top_action(final_scores, candidates)

        # LLM scores for logging/visualization. Prefer numeric per-action scores when present.
        llm_scores: Dict[str, float] = {
            str(aid): float(v.get("score", 0.0) or 0.0) for aid, v in (per_action_scores or {}).items()
        }
        if not llm_scores and ranked_actions:
            # Legacy fallback: rank-based pseudo-scores.
            for idx, aid in enumerate(ranked_actions or []):
                llm_scores[str(aid)] = float(max(1.0, 10.0 - float(idx)))

        # Attach LLM scores to graph nodes (purely diagnostic).
        for cid, sc in llm_scores.items():
            if cid in self.graph.graph:
                try:
                    self.graph.add_llm_score(cid, float(sc))
                    self.graph.graph.nodes[cid]["llm_confidence"] = float(llm_confidence)
                except Exception:
                    continue

        llm_debug: Dict[str, Any] = {
            "llm_valid": bool(llm_valid),
            "llm_confidence": float(llm_confidence),
            "llm_brief_reason": str(justification or ""),
            "ranked_actions": list(ranked_actions or []),
            "invalid_llm_reasons": invalid_reasons,
            "chosen_from": str(chosen_from),
            "gate_g": gate_g,
            "prior_gap": round(float(prior_gap), 4),
            "alpha": round(float(alpha), 4),
            "prompt_contains_banned_incident_words_pre": bool(prompt_contains_banned_pre),
            "prompt_contains_banned_incident_words_post": bool(prompt_contains_banned_post),
            # Backwards-compatible aliases for older analysis scripts.
            "prompt_contains_banned_pre_sanitize": bool(prompt_contains_banned_pre),
            "prompt_contains_banned_post_sanitize": bool(prompt_contains_banned_post),
            "candidate_top_k_effective": int(effective_k),
            "eps_cand_used": bool(eps_cand_used),
            "num_candidates_before": int(num_candidates_before),
            "num_candidates_after_filter": int(num_candidates_after_filter),
            "num_candidates_after_diversify": int(num_candidates_after_diversify),
            "unique_candidates_last_50": int(
                len(set(aid for bucket in list(self._candidate_history) + [candidates] for aid in bucket))
            ),
            "traffic_candidates_included": bool(traffic_candidates_included),
            "traffic_gate_applied": bool(traffic_gate_applied),
            "traffic_gate_reason": traffic_gate_reason,
            "traffic_evidence": traffic_evidence,
            "analyse_cap_applied": bool(analyse_cap_applied),
        }
        if gate_info:
            llm_debug["adaptive_gate"] = gate_info
        if isinstance(parse_debug, dict):
            llm_debug.update(parse_debug)
        if disruption_risk:
            llm_debug["disruption_risk"] = disruption_risk
        if best_after_postprocess:
            llm_debug["best_after_postprocess"] = str(best_after_postprocess)
        if traffic_score_caps:
            llm_debug["traffic_score_caps"] = traffic_score_caps

        # Compact per-action score stats to make regressions visible without logging the full map.
        if per_action_scores:
            try:
                score_vals = sorted(float(v.get("score", 0.0) or 0.0) for v in per_action_scores.values())
            except Exception:
                score_vals = []
            if score_vals:
                n = len(score_vals)
                if n % 2 == 1:
                    median = float(score_vals[n // 2])
                else:
                    median = float(0.5 * (score_vals[n // 2 - 1] + score_vals[n // 2]))

                llm_debug["score_spread"] = float(score_vals[-1] - score_vals[0])
                llm_debug["num_distinct_scores"] = int(len(set(round(v, 3) for v in score_vals)))

                top3 = sorted(
                    list(per_action_scores.items()),
                    key=lambda kv: (
                        -float(kv[1].get("score", 0.0) or 0.0),
                        -float(kv[1].get("confidence", 0.0) or 0.0),
                        str(kv[0]),
                    ),
                )[:3]
                llm_debug["llm_action_scores"] = {
                    "min": float(score_vals[0]),
                    "median": float(median),
                    "max": float(score_vals[-1]),
                    "top3": [
                        {"action_id": str(aid), "score": float(v.get("score", 0.0) or 0.0)} for aid, v in top3
                    ],
                }

        action = self._map_action_id_to_action(
            chosen_id,
            action_space,
            observation,
        )
        node_id = action_to_node_id(self.name, action)

        # Bug-5 fix: record which host was targeted so _rank_hosts_by_suspicion
        # can penalize recently-targeted hosts and break lock-in loops.
        targeted_host = getattr(action, "hostname", None)
        if isinstance(targeted_host, str) and targeted_host:
            self._recently_targeted_hosts.append((targeted_host, self._defender_step_counter))

        if prev_node and prev_node in self.graph.graph and node_id in self.graph.graph:
            prev_type = self.graph._get_agent_type(prev_node)
            curr_type = self.graph._get_agent_type(node_id)
            if prev_type != curr_type and not self.graph.graph.has_edge(prev_node, node_id):
                if self.config.use_edge_discovery:
                    created = self.graph.ensure_edge(prev_node, node_id, description="discovered")
                    if created:
                        self._new_edges_discovered += 1
                        self._log_step({"event": "edge_discovered", "from": prev_node, "to": node_id})
                else:
                    self._log_step(
                        {
                            "event": "pruned_edge_fallback",
                            "from": prev_node,
                            "to": node_id,
                            "reason": "edge_pruned_or_missing",
                        }
                    )
                    action = Sleep()
                    node_id = action_to_node_id(self.name, action)

        self.trace.append(node_id)
        self.state_trace.append(state_signature if self.config.use_state_conditioning else None)
        self.last_action_node = node_id
        self.last_action = action
        # Track candidate diversity history and recent defender choices.
        if candidates:
            self._candidate_history.append(list(candidates))
        if str(node_id).startswith("defender_"):
            self._recent_def_actions.append(str(node_id))
        traffic_action_cooldown_active = cooldown_active if 'cooldown_active' in locals() else False
        traffic_flipflop_detected = False
        chosen_traffic_zone: Optional[Tuple[str, str]] = None
        if str(chosen_id) in {"defender_block_traffic_zone", "defender_allow_traffic_zone"}:
            fz = getattr(action, "from_subnet", None)
            tz = getattr(action, "to_subnet", None)
            if isinstance(fz, str) and isinstance(tz, str):
                chosen_traffic_zone = (fz, tz)
                self._traffic_history.append((str(chosen_id), fz, tz, self._defender_step_counter))
                # Detect flip-flop (block vs allow same pair) within horizon.
                horizon = int(getattr(self.config, "traffic_flipflop_horizon", 0) or 0)
                if horizon > 0:
                    opposite = "defender_allow_traffic_zone" if str(chosen_id) == "defender_block_traffic_zone" else "defender_block_traffic_zone"
                    for aid, pf, pt, step_idx in reversed(self._traffic_history):
                        if self._defender_step_counter - step_idx > horizon:
                            break
                        if aid == opposite and pf == fz and pt == tz:
                            traffic_flipflop_detected = True
                            break
        self._log_llm_eval(
            prompt=prompt,
            response=response,
            candidates=list(candidates),
            prompt_contains_banned_incident_words=bool(prompt_contains_banned_post),
            prompt_contains_banned_incident_words_pre=bool(prompt_contains_banned_pre),
            prompt_contains_banned_incident_words_post=bool(prompt_contains_banned_post),
            llm_valid=bool(llm_valid),
            llm_confidence=float(llm_confidence),
            llm_ranked_actions=list(ranked_actions or []),
            llm_brief_reason=str(justification or ""),
            llm_scores=llm_scores,
            graph_prior=priors,
            chosen_action=str(chosen_id),
            chosen_from=str(chosen_from),
            gate_g=gate_g,
            visits_evidence=visits_evidence,
            state_signature=state_signature if self.config.use_state_conditioning else None,
            llm_debug=llm_debug,
            final_scores=final_scores,
            false_repeat_penalties=false_repeat_penalties,
            last_action_status=str(observation.get("success", "")),
            traffic_action_cooldown_active=bool(traffic_action_cooldown_active),
            chosen_traffic_zone=chosen_traffic_zone,
            traffic_flipflop_detected=bool(traffic_flipflop_detected),
            traffic_gate_applied=bool(traffic_gate_applied),
            traffic_gate_reason=traffic_gate_reason,
            traffic_evidence=traffic_evidence,
        )
        self._log_step(
            {
                "event": "action_select",
                "state_signature": state_signature if self.config.use_state_conditioning else None,
                "chosen_action": str(chosen_id),
                "chosen_from": str(chosen_from),
                "llm_valid": bool(llm_valid),
                "llm_confidence": float(llm_confidence),
                "gate_g": gate_g,
                "prompt_contains_banned_incident_words": bool(prompt_contains_banned_post),
                "prompt_contains_banned_incident_words_pre": bool(prompt_contains_banned_pre),
                "prompt_contains_banned_incident_words_post": bool(prompt_contains_banned_post),
                "chosen_prior": float(priors.get(chosen_id, 0.0) or 0.0),
                "chosen_rank_score": float(
                    (per_action_scores or {}).get(chosen_id, {}).get("score", llm_scores.get(chosen_id, 0.0) or 0.0)
                ),
                "chosen_visits_evidence": float(visits_evidence.get(chosen_id, 0.0) or 0.0),
                "repeat_failures": 0,
                "candidate_top_k_effective": int(effective_k),
                "eps_cand_used": bool(eps_cand_used),
                "num_candidates_before": int(num_candidates_before),
                "num_candidates_after_filter": int(num_candidates_after_filter),
                "num_candidates_after_diversify": int(num_candidates_after_diversify),
                "unique_candidates_last_50": int(
                    len(set(aid for bucket in list(self._candidate_history) for aid in bucket))
                ),
                "traffic_action_cooldown_active": bool(traffic_action_cooldown_active),
                "chosen_traffic_zone": chosen_traffic_zone,
                "traffic_flipflop_detected": bool(traffic_flipflop_detected),
                "traffic_gate_applied": bool(traffic_gate_applied),
                "traffic_gate_reason": traffic_gate_reason,
                "traffic_evidence": traffic_evidence,
            }
        )
        return action

    def note_transition(self, agent_name: str, action: Action) -> None:
        """Record transitions for other agents (red) to keep the trace alternating."""
        node_id = action_to_node_id(agent_name, action)
        if self.trace and self.trace[-1] == node_id:
            return

        prev_node = self.trace[-1] if self.trace else None
        if prev_node and self.config.use_edge_discovery:
            self._maybe_discover_edge(prev_node, node_id)

        self.trace.append(node_id)
        self.state_trace.append(None)
        if "blue" in agent_name:
            # While a multi-tick action is IN_PROGRESS, the controller executes Sleep ticks.
            # Don't clobber last_action_node with that placeholder Sleep.
            if isinstance(action, Sleep) and self._waiting_in_progress:
                return
            self.last_action_node = node_id

    def finalize_episode(self, total_reward: float) -> None:
        """Update scores, prune, visualise, and reset trace."""
        if self._episode_finalized:
            return
        self._episode_finalized = True
        success = total_reward > 0
        if self.config.disable_graph:
            update_info = {"baseline": 0.0, "advantage": 0.0, "edges_updated": 0, "state_edges_updated": 0}
        else:
            update_info = self.learner.observe_round(
                self.trace,
                {"success": success, "reward": total_reward},
                state_signatures=self.state_trace,
            )
        pruned = []  # Pruning disabled; selection already constrains via top-K
        self._save_scores()
        self._visualize_graph()
        top_state_edges = self.learner.top_state_edges(self._episode_state_signatures, n=3)
        self._log_step(
            {
                "event": "episode_end",
                "reward": total_reward,
                "success": success,
                "baseline": float(update_info.get("baseline", 0.0) or 0.0),
                "advantage": float(update_info.get("advantage", 0.0) or 0.0),
                "edges_updated": int(update_info.get("edges_updated", 0) or 0),
                "state_edges_updated": int(update_info.get("state_edges_updated", 0) or 0),
                "new_edges_discovered": int(self._new_edges_discovered),
                "top_state_edges": top_state_edges,
                "pruned_edges": pruned,
            }
        )
        self.trace = []
        self.state_trace = []
        self._episode_state_signatures = set()
        self._new_edges_discovered = 0
        self.last_action_node = None
        self.last_action = None
        self._post_compromise_mode = False
        self._compromised_hosts = []
        self._last_compromise_step = None
        self._seen_compromise_messages = set()
        self._post_compromise_steps_since_restore = 0
        self._last_restore_step = None
        self._last_restore_status = None
        self._restore_cooldown_until_step = 0
        self._recently_targeted_hosts.clear()
        self._traffic_last_pair = None
        self._traffic_history.clear()
        self._recent_def_actions.clear()
        self._candidate_history.clear()
        self.episode_counter += 1
        self.llm_policy.end_episode()
        # No BaseAgent end_episode implementation; just reset local state.

    def _save_scores(self) -> None:
        payload = self.learner.export_edge_scores_to_json()
        # Run-local learner stats
        path = self.log_dir / "action_graph_scores_live.json"
        path.write_text(payload)
        # Save full graph (including llm_score) persistently and per-run
        Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
        self.graph.save(self.persist_path)
        graph_json = self.log_dir / "action_graph_full.json"
        self.graph.save(graph_json)
        # Snapshot
        if self.snapshot_every and self.episode_counter % self.snapshot_every == 0:
            snap_path = self.log_dir / f"action_graph_scores_episode_{self.episode_counter}.json"
            snap_path.write_text(payload)

    def _visualize_graph(self) -> None:
        out_path = self.log_dir / "action_graph_live.png"
        self.graph.render_to_png(out_path)
        if self.snapshot_every and self.episode_counter % self.snapshot_every == 0:
            snap_path = self.log_dir / f"action_graph_episode_{self.episode_counter}.png"
            self.graph.render_to_png(snap_path)

    # LLM evaluation helpers
    def build_llm_prompt(
        self,
        observation: Dict[str, Any],
        candidates: List[str],
        graph_priors: Dict[str, float],
        top_k: int = 3,
        state_signature: Optional[str] = None,
        bucket_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        prompts: List[Dict[str, str]] = []
        disable_graph = bool(getattr(self.config, 'disable_graph', False))

        if disable_graph:
            system_msg = (
                "You are a CAGE-4 Blue defender. Rank the provided candidate actions best-to-worst.\n"
                "\n"
                "Return ONLY a raw JSON object (no Markdown, no ``` fences) with this exact schema:\n"
                "{\n"
                "  \"ranked_actions\": [\"<action_id>\", \"...\"],\n"
                "  \"confidence\": 0.0,\n"
                "  \"brief_reason\": \"1-2 sentences citing 1-2 observation signals\"\n"
                "}\n"
                "\n"
                "Hard rules:\n"
                "- ranked_actions MUST include EVERY candidate action_id exactly once "
                "(no missing, no duplicates, no extras).\n"
                "- best action is ranked_actions[0].\n"
                "- confidence is a float in [0.0, 1.0] reflecting certainty of your ranking.\n"
                "- brief_reason: cite observation signals only.\n"
                "- Do NOT output per-action scores. Do NOT output an \"actions\" dict. "
                "Do NOT output extra keys.\n"
                "- If uncertain, still output a full ranking and express uncertainty via "
                "low confidence (< 0.4).\n"
            )
        else:
            system_msg = (
                "You are a CAGE-4 Blue defender. Rank the provided candidate actions best-to-worst.\n"
                "\n"
                "Return ONLY a raw JSON object (no Markdown, no ``` fences) with this exact schema:\n"
                "{\n"
                "  \"ranked_actions\": [\"<action_id>\", \"...\"],\n"
                "  \"confidence\": 0.0,\n"
                "  \"brief_reason\": \"1-2 sentences citing 1-2 observation signals\"\n"
                "}\n"
                "\n"
                "Hard rules:\n"
                "- ranked_actions MUST include EVERY candidate action_id exactly once "
                "(no missing, no duplicates, no extras).\n"
                "- best action is ranked_actions[0].\n"
                "- confidence is a float in [0.0, 1.0] reflecting certainty of your ranking.\n"
                "- brief_reason: cite observation signals; mention whether graph priors were used.\n"
                "- Do NOT output per-action scores. Do NOT output an \"actions\" dict. "
                "Do NOT output extra keys.\n"
                "- If uncertain, still output a full ranking and express uncertainty via "
                "low confidence (< 0.4).\n"
                "\n"
                "Signal legend (shown per candidate):\n"
                "  prior: graph-based score mapped to [1,10]; 10=historically best, "
                "1=historically worst, 5.5=no data yet.\n"
                "  visits: total times this action appeared in the graph trajectory.\n"
                "\n"
                "Candidates are listed in descending prior order. "
                "Use prior as a starting hint; override when observation clearly indicates otherwise.\n"
            )
        prompts.append({"role": "system", "content": system_msg})

        # Observation summary
        obs_msg = obs_formatter.format_observation(observation, self.last_action, self.name)
        prompts.append({"role": "user", "content": obs_msg})
        commvector_legend = (
            "Commvector legend: bits [0-4] = alert flags for Blue agents 0-4; "
            "bits [5-6] = severity level (00 none, 01 scan/remote activity, 10 user-level access, 11 admin-level access); "
            "bit [7] = action pending (1) or not (0)."
        )
        prompts.append({"role": "user", "content": commvector_legend})

        # Recent action history so the LLM avoids repetition.
        recent_actions = list(self._recent_def_actions)[-5:]
        if recent_actions:
            history_lines = []
            for i, aid in enumerate(recent_actions):
                label = self.graph.graph.nodes[aid]["action"].label if aid in self.graph.graph else aid
                history_lines.append(f"  {i+1}. {label}")
            history_block = (
                "Your recent actions (oldest to newest):\n"
                + "\n".join(history_lines)
                + "\n"
                + "IMPORTANT: Avoid repeating the same action type more than 2-3 times in a row. "
                + "If you have been Analysing repeatedly, switch to Remove or Restore to act on findings.\n"
            )
            prompts.append({"role": "user", "content": history_block})

        if disable_graph:
            # No graph context: show candidates as plain labeled list in arbitrary order.
            env_summary = f"Mission phase: {observation.get('phase', '?')}. Hosts seen: {len(observation.keys())}"
            lines = []
            for cid in candidates:
                label = self.graph.graph.nodes[cid]["action"].label if cid in self.graph.graph else cid
                lines.append(f"{cid} ({label})")
            candidate_prompt = (
                f"{env_summary}\n"
                + "\n".join(lines)
                + "\n\n"
                + f"Candidates (action_id list): {json.dumps(candidates)}\n"
                + "Return JSON only using the schema from the system message."
            )
            prompts.append({"role": "user", "content": candidate_prompt})
        else:
            # Sort candidates by descending prior before building the candidates block.
            sorted_candidates = sorted(candidates, key=lambda c: -float(graph_priors.get(c, 5.5)))

            # Graph-aware augmentation (safe by default: no attacker action IDs).
            env_summary = f"Mission phase: {observation.get('phase', '?')}. Hosts seen: {len(observation.keys())}"
            blocks: List[str] = []
            for cid in sorted_candidates:
                label = self.graph.graph.nodes[cid]["action"].label if cid in self.graph.graph else cid
                prior = float(graph_priors.get(cid, 5.5))
                visits = 0
                if cid in self.graph.graph:
                    edges_out = self.graph.get_outgoing_edges(cid, state_signature=None, backoff=True)
                    visits = int(sum(int(s.get("visit_count", 0) or 0) for _, _, s in edges_out))
                blocks.append(f"{cid} ({label}) prior={prior:.2f} visits={visits}")
                # [avg_ep_reward disabled] ep_score removed — all actions share the same global
                # episode mean (~-278 for all edges), zero differentiation between actions.
                # Needs redesign (per-step credit or counterfactual) before re-enabling.

            graph_prompt = (
                "ActionGraph priors (candidates listed in descending prior order):\n"
                f"{env_summary}\n"
                + (f"State signature: {state_signature}\n" if state_signature else "")
                + "\n".join(blocks)
                + "\n\n"
                + f"Candidates (action_id list): {json.dumps(sorted_candidates)}\n"
                + "Return JSON only using the schema from the system message."
            )
            prompts.append({"role": "user", "content": graph_prompt})
        return prompts

    def _prompt_contains_banned_tokens(self, prompt: List[Dict[str, str]]) -> bool:
        tokens = getattr(self.config, "banned_prompt_tokens", None)
        if not tokens:
            return False
        blob = "\n".join(
            str(m.get("content", ""))
            for m in (prompt or [])
            if isinstance(m, dict)
        ).lower()
        token_set: set[str] = set()
        for tok in tokens:
            if not isinstance(tok, str):
                continue
            t = tok.strip().lower()
            if not t:
                continue
            token_set.add(t)
            if t in blob:
                return True

        # Special-case: post_compromise variants ("post-compromise", "post compromise", etc.).
        # This is intentionally redundant with substring checks for "compromise"/"compromised", but
        # ensures correct detection even if configs are customized.
        if "post_compromise" in token_set:
            if re.search(
                r"(?<![A-Za-z0-9])post(?:[_\-\s]+)compromise(?:d)?(?![A-Za-z0-9])",
                blob,
                flags=re.IGNORECASE,
            ):
                return True
        return False

    def _sanitize_prompt(self, prompt: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Last-mile sanitizer to prevent persistent narrative priming from leaking into prompts."""
        tokens = getattr(self.config, "banned_prompt_tokens", None)
        if not tokens:
            return list(prompt or [])

        # Prefer replacing longer tokens first (e.g., "post_compromise" before "compromise").
        ordered = sorted(
            (str(t).strip().lower() for t in tokens if isinstance(t, str) and str(t).strip()),
            key=len,
            reverse=True,
        )
        if not ordered:
            return list(prompt or [])

        replacements = {
            "post_compromise": "post_event",
            "compromised": "at_risk",
            "compromise": "risk",
            "incident": "event",
            "breach": "exposure",
            "intrusion": "anomaly",
        }
        token_patterns: Dict[str, str] = {}
        for tok in ordered:
            # Special-case: catch common variants/separators for "post_compromise".
            if tok == "post_compromise":
                token_patterns[tok] = r"(?<![A-Za-z0-9])post(?:[_\-\s]+)compromise(?:d)?(?![A-Za-z0-9])"
            else:
                token_patterns[tok] = re.escape(tok)

        out: List[Dict[str, str]] = []
        for m in prompt or []:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            sanitized = content
            for tok in ordered:
                rep = replacements.get(tok, "risk")
                pat = token_patterns.get(tok) or re.escape(tok)
                sanitized = re.sub(pat, rep, sanitized, flags=re.IGNORECASE)
            out.append({"role": role, "content": sanitized})
        return out

    def _compute_circuit_breaker_gate(
        self, llm_confidence: float, graph_priors: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute adaptive blending weight g in [0.5, 0.95].

        g controls: blended = g * llm_score + (1 - g) * graph_prior

        Early episodes (priors all ~5.5, variance ~0): gate_g high → trust LLM.
        Later episodes (priors diverge, variance high): gate_g lower → trust graph more.

        Returns:
            gate_g: the blending weight.
            gate_info: diagnostic dict for logging.
        """
        import statistics

        base = float(getattr(self.config, "gate_base", 0.85) or 0.85)
        k = float(getattr(self.config, "gate_k", 0.10) or 0.10)
        c = self._clamp01(float(llm_confidence))
        base_gate = float(base) + float(k) * float(c)  # [0.85, 0.95]
        raw_base_gate = base_gate

        prior_variance = 0.0
        graph_trust_bonus = 0.0
        prior_mean = 5.5
        prior_min = 5.5
        prior_max = 5.5

        # Adapt based on how much the graph has learned (prior variance).
        if graph_priors and len(graph_priors) >= 2:
            prior_values = list(graph_priors.values())
            prior_variance = statistics.variance(prior_values)
            prior_mean = statistics.mean(prior_values)
            prior_min = min(prior_values)
            prior_max = max(prior_values)
            # Higher variance → graph has learned → trust graph more → lower gate.
            # variance=0 → no adjustment; variance>=2 → reduce gate by up to 0.35.
            graph_trust_bonus = min(prior_variance / 2.0, 0.35)
            base_gate -= graph_trust_bonus

        # Clamp: LLM always has >=50% influence; graph always has >=5%.
        gate_g = float(min(max(base_gate, 0.5), 0.95))

        gate_info: Dict[str, Any] = {
            "gate_g": gate_g,
            "gate_base_before_adapt": round(raw_base_gate, 4),
            "prior_variance": round(prior_variance, 4),
            "prior_mean": round(prior_mean, 4),
            "prior_min": round(prior_min, 4),
            "prior_max": round(prior_max, 4),
            "graph_trust_bonus": round(graph_trust_bonus, 4),
            "graph_influence_pct": round(100.0 * (1.0 - gate_g), 1),
        }

        return gate_g, gate_info

    def _pick_from_ranking(self, ranked_actions: List[str], *, last_status: Optional[TernaryEnum]) -> Optional[str]:
        """Pick the top-ranked action from the LLM ranking.

        No streak filtering — in CAGE-4 most actions normally return FALSE,
        so streak-based lockout is counterproductive.
        """
        filtered: List[str] = []
        seen: set[str] = set()
        for aid in ranked_actions or []:
            if not isinstance(aid, str):
                continue
            if aid in seen:
                continue
            seen.add(aid)
            filtered.append(aid)

        if not filtered:
            return None
        return filtered[0]

    @staticmethod
    def _extract_json_str(text: str) -> Optional[str]:
        """
        Extract a JSON object substring from model output.

        The OpenRouter/OpenAI ecosystem frequently wraps JSON in Markdown code fences or
        surrounds it with prose. We prefer the *first* parseable JSON object.
        """

        def _iter_candidate_objects(s: str):
            s = str(s or "")
            i = 0
            while True:
                start = s.find("{", i)
                if start == -1:
                    return
                in_str = False
                esc = False
                quote = ""
                depth = 0
                for j in range(start, len(s)):
                    ch = s[j]
                    if in_str:
                        if esc:
                            esc = False
                            continue
                        if ch == "\\":
                            esc = True
                            continue
                        if ch == quote:
                            in_str = False
                            quote = ""
                        continue

                    if ch in {"\"", "'"}:
                        in_str = True
                        quote = ch
                        continue

                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        if depth > 0:
                            depth -= 1
                            if depth == 0:
                                yield s[start : j + 1]
                                break
                # Keep searching even if the first braced block wasn't valid JSON.
                i = start + 1

        payload = str(text or "").strip()
        if not payload:
            return None

        # Prefer JSON inside the first fenced code block that contains '{'.
        if "```" in payload:
            parts = payload.split("```")
            for idx in range(1, len(parts), 2):
                inner = parts[idx].lstrip()
                # Drop an optional language tag on the first line, e.g. "json\n{...}".
                if inner[:4].lower() == "json":
                    # Do not scan to the first newline: some callers (or tests) may include a
                    # literal "\\n" sequence after "json", while the first *real* newline can
                    # appear inside the JSON payload (e.g. from json.dumps(indent=...)).
                    inner = inner[4:]
                    if inner.startswith("\r\n"):
                        inner = inner[2:]
                    elif inner.startswith("\n"):
                        inner = inner[1:]
                    elif inner.startswith("\\n"):
                        inner = inner[2:]
                    inner = inner.lstrip()
                for cand in _iter_candidate_objects(inner):
                    try:
                        json.loads(cand)
                        return cand
                    except Exception:
                        continue

        # Fall back to scanning the full payload.
        for cand in _iter_candidate_objects(payload):
            try:
                json.loads(cand)
                return cand
            except Exception:
                continue

        return None

    def parse_llm_response(
        self, response: str, candidates: List[str]
    ) -> Tuple[List[str], float, str, Dict[str, str], bool, Dict[str, Any], Dict[str, Dict[str, float]]]:
        """Parse the LLM response.

        Returns:
            ranked_actions: candidates sorted by score desc then confidence desc.
            overall_confidence: confidence of the top-ranked action (best after tie-break).
            justification: short free-text rationale (may be empty).
            disruption_risk: map of traffic actions -> low|med|high (may be empty).
            llm_valid: True when the response matches the expected schema for this prompt.
            debug: debug fields for logging/diagnosis.
            per_action_scores: Dict[action_id, {"score": float, "confidence": float}]
        """
        ranked_actions: List[str] = []
        confidence: float = 0.0
        justification: str = ""
        disruption_risk: Dict[str, str] = {}
        llm_valid: bool = False
        per_action_scores: Dict[str, Dict[str, float]] = {}

        debug: Dict[str, Any] = {
            "parse_json_error": True,
            "used_tolerant_extraction": False,
            "extracted_json_present": False,
            "schema": None,
            "invalid_reasons": [],
        }

        data: Any = None
        extracted = self._extract_json_str(response)
        if extracted is not None:
            debug["extracted_json_present"] = True
            try:
                data = json.loads(extracted)
                debug["parse_json_error"] = False
            except Exception:
                data = None
                debug["parse_json_error"] = True

        if isinstance(data, dict):
            # Primary: rank-only schema (new format prompted by current system message).
            # {
            #   "ranked_actions": ["<action_id>", ...],
            #   "confidence": 0-1,
            #   "brief_reason": "..."
            # }
            if "ranked_actions" in data and isinstance(data.get("ranked_actions"), list):
                debug["schema"] = "rank_only"
                ra = data["ranked_actions"]
                conf_raw = data.get("confidence", 0.5)
                reason = data.get("brief_reason", "") or data.get("justification", "")
                if isinstance(reason, str):
                    justification = reason.strip()

                cand_set = set(candidates)
                # Strict permutation check: must be exact permutation of candidates.
                ra_strs = [str(a) for a in ra if isinstance(a, str)]
                if (
                    set(ra_strs) == cand_set
                    and len(ra_strs) == len(candidates)
                    and len(ra_strs) == len(set(ra_strs))
                ):
                    llm_valid = True
                    ranked_actions = ra_strs
                    confidence = float(min(1.0, max(0.0, float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.5)))
                    per_action_scores = {}
                    if ranked_actions:
                        debug["best"] = ranked_actions[0]
                else:
                    llm_valid = False
                    ranked_actions = []
                    confidence = float(min(1.0, max(0.0, float(conf_raw) if isinstance(conf_raw, (int, float)) else 0.5)))
                    per_action_scores = {}
                    debug["invalid_reasons"].append("ranked_actions_not_valid_permutation")

                confidence = float(min(1.0, max(0.0, confidence)))
                return ranked_actions, confidence, justification, disruption_risk, llm_valid, debug, per_action_scores

            # Fallback: scored_actions schema (legacy or backward-compat responses).
            # {
            #   "actions": {"<action_id>": {"score": 1-10, "confidence": 0-1}},
            #   "best": "<action_id>",
            #   "justification": "...",
            #   "disruption_risk": {...}
            # }
            if isinstance(data.get("actions", None), dict):
                debug["schema"] = "scored_actions"

                allowed = set(candidates)
                reason = data.get("justification", "") or data.get("brief_reason", "")
                risk = data.get("disruption_risk", {})
                best_raw = data.get("best", None)

                if isinstance(reason, str):
                    justification = reason.strip()

                if isinstance(risk, dict):
                    for k, v in risk.items():
                        if isinstance(k, str) and isinstance(v, str):
                            disruption_risk[str(k)] = v.lower().strip()

                actions_map = data.get("actions", {})
                key_set = {k for k in actions_map.keys() if isinstance(k, str)}
                missing = sorted(list(allowed - key_set))
                extra = sorted(list(key_set - allowed))
                if missing:
                    debug["invalid_reasons"].append("missing_candidate_actions")
                    debug["missing_action_ids"] = missing
                if extra:
                    debug["invalid_reasons"].append("extra_actions_present")
                    debug["extra_action_ids"] = extra

                # Parse per-action {score, confidence}.
                parse_ok = True
                for cid in candidates:
                    entry = actions_map.get(cid, None)
                    if not isinstance(entry, dict):
                        parse_ok = False
                        debug["invalid_reasons"].append("missing_or_invalid_action_entry")
                        continue
                    sc = entry.get("score", None)
                    cf = entry.get("confidence", None)
                    if not isinstance(sc, (int, float)) or not isinstance(cf, (int, float)):
                        parse_ok = False
                        debug["invalid_reasons"].append("non_numeric_score_or_confidence")
                        continue
                    sc_f = float(sc)
                    cf_f = float(cf)
                    if not (1.0 <= sc_f <= 10.0):
                        parse_ok = False
                        debug["invalid_reasons"].append("score_out_of_range")
                        continue
                    if not (0.0 <= cf_f <= 1.0):
                        parse_ok = False
                        debug["invalid_reasons"].append("confidence_out_of_range")
                        continue
                    per_action_scores[str(cid)] = {"score": sc_f, "confidence": cf_f}

                # Diagnostic checks for degenerate scores/confidences (warn but don't reject).
                # Bug-2 fix: LLM legitimately gives equal scores when it views actions
                # as equivalent.  Hard-rejecting caused excessive GRAPH_FALLBACK.
                if parse_ok and not missing and not extra and len(per_action_scores) == len(candidates):
                    score_vals = [round(float(v["score"]), 3) for v in per_action_scores.values()]
                    conf_vals = [round(float(v["confidence"]), 3) for v in per_action_scores.values()]
                    distinct_scores = len(set(score_vals))
                    distinct_conf = len(set(conf_vals))
                    debug["num_distinct_scores"] = int(distinct_scores)
                    debug["num_distinct_confidences"] = int(distinct_conf)

                    if len(candidates) >= 3 and distinct_scores < 3:
                        debug["invalid_reasons"].append("low_score_distinctness_warning")
                    elif len(candidates) == 2 and distinct_scores < 2:
                        debug["invalid_reasons"].append("low_score_distinctness_warning")

                    if len(candidates) >= 2 and distinct_conf < 2:
                        debug["invalid_reasons"].append("low_confidence_distinctness_warning")

                # Ranking derived from score then confidence (deterministic tie-break by id).
                if parse_ok and len(per_action_scores) == len(candidates) and not missing and not extra:
                    ranked_actions = sorted(
                        list(per_action_scores.keys()),
                        key=lambda aid: (
                            -float(per_action_scores[aid]["score"]),
                            -float(per_action_scores[aid]["confidence"]),
                            str(aid),
                        ),
                    )
                    computed_best = ranked_actions[0] if ranked_actions else None

                    best: Optional[str] = None
                    if isinstance(best_raw, str) and best_raw in allowed:
                        best = str(best_raw)
                        # Enforce best==argmax, but be tolerant: override rather than discarding scores.
                        if computed_best and best != computed_best:
                            debug["invalid_reasons"].append("best_not_argmax_overridden")
                            debug["best_provided"] = best
                            best = str(computed_best)
                    else:
                        if best_raw is not None:
                            debug["invalid_reasons"].append("best_not_in_candidates")
                            debug["best_provided"] = str(best_raw)
                        best = str(computed_best) if computed_best else None

                    if best:
                        debug["best"] = best
                        confidence = float(per_action_scores.get(best, {}).get("confidence", 0.0) or 0.0)

                    llm_valid = True
                else:
                    # Parsed JSON, but schema/contract violation. Do not use tolerant extraction for this case.
                    llm_valid = False

            else:
                # Legacy (ranked_actions) schema:
                # {
                #   "ranked_actions": [...],
                #   "confidence": 0-1,
                #   "justification"/"brief_reason": "...",
                #   "disruption_risk": {...}
                # }
                debug["schema"] = "ranked_actions"
                ra = data.get("ranked_actions", None)
                conf = data.get("confidence", None)
                reason = data.get("justification", "") or data.get("brief_reason", "")
                risk = data.get("disruption_risk", {})

                if isinstance(reason, str):
                    justification = reason.strip()

                if isinstance(risk, dict):
                    for k, v in risk.items():
                        if isinstance(k, str) and isinstance(v, str):
                            disruption_risk[str(k)] = v.lower().strip()

                if isinstance(ra, (list, tuple)) and isinstance(conf, (int, float)):
                    allowed = set(candidates)
                    seen: set[str] = set()
                    for aid in ra:
                        if not isinstance(aid, str):
                            continue
                        if aid not in allowed:
                            continue
                        if aid in seen:
                            continue
                        seen.add(aid)
                        ranked_actions.append(aid)
                    confidence = float(min(1.0, max(0.0, float(conf))))
                    llm_valid = bool(ranked_actions)

                    # Derive per-action scores from rank position (legacy behaviour).
                    for idx, aid in enumerate(ranked_actions):
                        per_action_scores[str(aid)] = {
                            "score": float(max(1.0, 10.0 - float(idx))),
                            "confidence": float(confidence),
                        }
                    if ranked_actions:
                        debug["best"] = ranked_actions[0]

        # Tolerant fallback: extract candidate ids by first appearance order.
        if not llm_valid and bool(debug.get("parse_json_error", True)):
            positions: List[Tuple[int, str]] = []
            text = str(response or "")
            for cid in candidates:
                try:
                    idx = int(text.find(str(cid)))
                except Exception:
                    idx = -1
                if idx >= 0:
                    positions.append((idx, str(cid)))
            positions.sort(key=lambda t: (t[0], t[1]))
            ranked_actions = [cid for _idx, cid in positions]
            if ranked_actions:
                debug["used_tolerant_extraction"] = True
                confidence = 0.40
                justification = ""
                llm_valid = True
                for idx, aid in enumerate(ranked_actions):
                    per_action_scores[str(aid)] = {
                        "score": float(max(1.0, 10.0 - float(idx))),
                        "confidence": float(confidence),
                    }
                debug["best"] = ranked_actions[0]

        # If we have per-action scores, prefer deriving ranking from them to avoid rank-only degeneracy.
        if per_action_scores:
            ranked_actions = sorted(
                list(per_action_scores.keys()),
                key=lambda aid: (
                    -float(per_action_scores[aid]["score"]),
                    -float(per_action_scores[aid]["confidence"]),
                    str(aid),
                ),
            )
            best_dbg = debug.get("best", None)
            if isinstance(best_dbg, str) and best_dbg in per_action_scores:
                confidence = float(per_action_scores[best_dbg]["confidence"])
            elif ranked_actions:
                confidence = float(per_action_scores[ranked_actions[0]]["confidence"])

        confidence = float(min(1.0, max(0.0, float(confidence))))
        return ranked_actions, confidence, justification, disruption_risk, llm_valid, debug, per_action_scores

    @staticmethod
    def _is_flat_scores(scored: Dict[str, float]) -> bool:
        """Return True if all scores are identical (or map is empty)."""
        if not scored:
            return False
        vals = [round(v, 3) for v in scored.values()]
        return len(set(vals)) <= 1
    @staticmethod
    def _is_flat_conf(conf: Dict[str, float]) -> bool:
        if not conf:
            return False
        vals = [round(v, 3) for v in conf.values()]
        return len(set(vals)) <= 1 or all(v <= 0.05 for v in vals)

    def select_top_action(self, final_scores: Dict[str, float], candidates: List[str]) -> str:
        """Pick the best action with a non-deterministic tie-break (avoid first-key bias)."""
        if not final_scores:
            return "defender_sleep"

        usable = dict(final_scores)

        best_score = max(usable.values())
        eps = 1e-6
        top_ids = [aid for aid, sc in usable.items() if abs(float(sc) - float(best_score)) <= eps]

        # Never pick Sleep if there's any non-sleep alternative.
        if "defender_sleep" in top_ids and len(top_ids) > 1:
            top_ids = [aid for aid in top_ids if aid != "defender_sleep"]

        if len(top_ids) == 1 and top_ids[0] == "defender_sleep":
            non_sleep = {k: v for k, v in final_scores.items() if k != "defender_sleep"}
            if not non_sleep:
                return "defender_sleep"
            best_non_sleep = max(non_sleep.values())
            top_ids = [aid for aid, sc in non_sleep.items() if abs(float(sc) - float(best_non_sleep)) <= eps]

        # Prefer not repeating the exact same action node if there's an equally-good alternative.
        if self.last_action_node in top_ids and len(top_ids) > 1:
            without_repeat = [aid for aid in top_ids if aid != self.last_action_node]
            if without_repeat:
                top_ids = without_repeat

        # Random tie-break among equally-scored actions.
        try:
            return str(self.np_random.choice(top_ids))
        except Exception:
            # Fallback if RNG isn't available or top_ids isn't indexable.
            return sorted(top_ids)[0]

    def _is_repeated_scores(self, scored: Dict[str, float]) -> bool:
        if not scored or self._last_llm_scores is None:
            return False
        try:
            return all(
                cid in self._last_llm_scores and round(self._last_llm_scores[cid], 3) == round(score, 3)
                for cid, score in scored.items()
            ) and len(scored) == len(self._last_llm_scores)
        except Exception:
            return False

    def _graph_based_scores(self, candidates: List[str]) -> Dict[str, float]:
        # Fallback; uses simple average of outgoing edge mean_contribution.
        scores: Dict[str, float] = {}
        for cid in candidates:
            outgoing = self.graph.get_outgoing_edges(cid)
            if outgoing:
                avg = sum(float(e[2].get("mean_contribution", 0.0) or 0.0) for e in outgoing) / len(outgoing)
            else:
                avg = 0.0
            scores[cid] = avg
        return scores

    @staticmethod
    def _has_strong_traffic_evidence(observation: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Heuristic, conservative check for traffic-related evidence in the current observation."""
        evidence: Dict[str, Any] = {}
        if not isinstance(observation, dict):
            return False, evidence

        # Check commvector/message strings for traffic keywords.
        traffic_keywords = {
            "traffic",
            "connection",
            "connections",
            "network",
            "subnet",
            "zone",
            "blocked",
            "denied",
            "ddos",
            "scan",
            "exfil",
            "latency",
            "throughput",
        }
        msgs: List[str] = []
        msg_field = observation.get("message")
        if isinstance(msg_field, str):
            msgs.append(msg_field)
        elif isinstance(msg_field, (list, tuple)):
            msgs.extend([m for m in msg_field if isinstance(m, str)])
        for m in msgs:
            ml = m.lower()
            for kw in traffic_keywords:
                if kw in ml:
                    evidence.setdefault("message_keywords", set()).add(kw)

        # Host-level connection anomalies: many repeated remote addresses or any "Connections" field.
        conn_hosts: List[str] = []
        for host, val in observation.items():
            if host in {"success", "action", "phase", "message"} or not isinstance(val, dict):
                continue
            procs = val.get("Processes")
            if isinstance(procs, list):
                for proc in procs:
                    if not isinstance(proc, dict):
                        continue
                    conns = proc.get("Connections")
                    if isinstance(conns, list) and conns:
                        conn_hosts.append(str(host))
                        break
        if conn_hosts:
            evidence["hosts_with_connections"] = conn_hosts

        # Commvector bit activity: treat any non-zero bit in agent messages as potential signal.
        def _has_bit(v: Any) -> bool:
            try:
                return any(bool(x) for x in v)
            except Exception:
                return False

        if isinstance(msg_field, (list, tuple)) and msg_field and _has_bit(msg_field[0]):
            evidence["commvector_activity"] = True

        # Conservative: require at least one concrete signal.
        has_evidence = bool(evidence)
        # Clean set -> list for JSONability
        if "message_keywords" in evidence and isinstance(evidence["message_keywords"], set):
            evidence["message_keywords"] = sorted(evidence["message_keywords"])
        return has_evidence, evidence

    _DISRUPTIVE_ACTIONS: set = {"defender_block_traffic_zone", "defender_allow_traffic_zone"}

    def _should_allow_traffic_action(
        self,
        action_id: str,
        *,
        has_traffic_evidence: bool,
        llm_confidence: float = 0.0,
    ) -> bool:
        """Consolidated traffic gate.  Bug-6 fix.

        Replaces four independent gates with one clear decision:
        allow if (not in cooldown) AND (has_evidence OR llm_confidence >= 0.7).
        """
        if action_id not in self._DISRUPTIVE_ACTIONS:
            return True

        # Cooldown: don't reuse a traffic action within traffic_cooldown_steps.
        if self.config.traffic_cooldown_steps > 0:
            recent = list(self._recent_def_actions)[-int(self.config.traffic_cooldown_steps):]
            if action_id in recent:
                return False

        # Allow if there is concrete traffic evidence OR the LLM is highly confident.
        return has_traffic_evidence or llm_confidence >= 0.7

    @staticmethod
    def _filter_disruptive_by_risk(ranked: List[str], risk: Dict[str, str], justification: str) -> List[str]:
        """Drop traffic actions when risk is high without a concrete trigger in justification."""
        if not ranked:
            return ranked
        safe_ranked: List[str] = []
        trigger_words = {
            "traffic",
            "zone",
            "subnet",
            "connection",
            "network",
            "flow",
            "packet",
            "route",
            "latency",
            "ddos",
        }
        just_lower = (justification or "").lower()
        has_trigger = any(w in just_lower for w in trigger_words)
        for aid in ranked:
            if aid in {"defender_block_traffic_zone", "defender_allow_traffic_zone"}:
                level = str(risk.get(aid, "")).lower()
                if level == "high" and not has_trigger:
                    continue
            safe_ranked.append(aid)
        return safe_ranked or ranked

    def _map_action_id_to_action(
        self,
        action_id: str,
        action_space,
        observation: Dict[str, Any],
        *,
        compromised_hosts: Optional[List[str]] = None,
    ) -> Action:
        target_snake = action_id.replace("defender_", "")
        compromised_hosts = [h for h in (compromised_hosts or []) if isinstance(h, str) and h]
        # If wrapper provided concrete actions, prefer them over reconstructing via action_space.
        if self._actions_cache:
            from CybORG.Agents.LLMAgents.llm_adapter.action_graph import _camel_to_snake

            candidates = [a for a in self._actions_cache if _camel_to_snake(a.__class__.__name__) == target_snake]
            if candidates:
                # Host-targeted actions: pick a concrete target based on the observation instead of
                # always taking the first item (which can lock the agent to host_0).
                if target_snake in {"analyse", "remove", "restore", "deploy_decoy"}:
                    host_candidates = [a for a in candidates if isinstance(getattr(a, "hostname", None), str) and getattr(a, "hostname", None)]

                    if compromised_hosts and target_snake in {"analyse", "remove", "restore"} and host_candidates:
                        # Deterministic: take the first compromised hostname that exists in candidates.
                        for host in compromised_hosts:
                            for act in host_candidates:
                                if getattr(act, "hostname", None) == host:
                                    return act
                        # If compromised_hosts doesn't intersect with our action space, avoid always
                        # picking the same "most suspicious" host and instead round-robin.
                        rr = int(self._rr_index_by_action_id.get(action_id, 0))
                        self._rr_index_by_action_id[action_id] = rr + 1
                        return host_candidates[rr % len(host_candidates)]

                    ranked_hosts = self._rank_hosts_by_suspicion(observation)
                    if ranked_hosts and host_candidates:
                        for host in ranked_hosts:
                            for act in host_candidates:
                                if getattr(act, "hostname", None) == host:
                                    return act

                    # Deterministic fallback: round-robin through available host parameterizations.
                    if host_candidates:
                        rr = int(self._rr_index_by_action_id.get(action_id, 0))
                        self._rr_index_by_action_id[action_id] = rr + 1
                        return host_candidates[rr % len(host_candidates)]

                    return candidates[0]

                if target_snake in {"block_traffic_zone", "allow_traffic_zone"}:
                    traffic_candidates = [
                        a
                        for a in candidates
                        if isinstance(getattr(a, "from_subnet", None), str)
                        and isinstance(getattr(a, "to_subnet", None), str)
                        and getattr(a, "from_subnet", None)
                        and getattr(a, "to_subnet", None)
                    ]
                    if traffic_candidates:
                        pairs = [((getattr(a, "from_subnet"), getattr(a, "to_subnet")), a) for a in traffic_candidates]
                        horizon = int(getattr(self.config, "traffic_flipflop_horizon", 0) or 0)
                        recent_opposite: set[Tuple[str, str]] = set()
                        if horizon > 0:
                            current_step = self._defender_step_counter
                            opposite = "defender_allow_traffic_zone" if target_snake == "block_traffic_zone" else "defender_block_traffic_zone"
                            for aid, pf, pt, step_idx in reversed(self._traffic_history):
                                if current_step - step_idx > horizon:
                                    break
                                if aid == opposite:
                                    recent_opposite.add((pf, pt))
                        preferred = [
                            act for (pair, act) in pairs if pair not in recent_opposite and pair != self._traffic_last_pair
                        ]
                        if not preferred:
                            preferred = [act for (pair, act) in pairs if pair != self._traffic_last_pair]
                        if not preferred:
                            preferred = [act for (_pair, act) in pairs]
                        choice = preferred[0]
                        self._traffic_last_pair = (getattr(choice, "from_subnet"), getattr(choice, "to_subnet"))
                        rr = int(self._rr_index_by_action_id.get(action_id, 0))
                        self._rr_index_by_action_id[action_id] = rr + 1
                        return choice
                    return candidates[0]

                # Other actions: keep the previous behavior (take the first concrete option).
                return candidates[0]

        # Dict action space handling
        if isinstance(action_space, dict):
            options = [cls for cls, valid in action_space.get("action", {}).items() if valid]
            for cls in options:
                from CybORG.Agents.LLMAgents.llm_adapter.action_graph import _camel_to_snake
                if _camel_to_snake(cls.__name__) != target_snake:
                    continue
                params = {}
                for param_name in cls.__init__.__code__.co_varnames:
                    if param_name in ("self",):
                        continue
                    if param_name in ("session", "agent"):
                        params[param_name] = 0 if param_name == "session" else self.name
                        continue
                    if param_name in action_space:
                        choices = [p for p, valid in action_space[param_name].items() if valid]
                        if choices:
                            if param_name == "hostname" and compromised_hosts:
                                for host in compromised_hosts:
                                    if host in choices:
                                        params[param_name] = host
                                        break
                                else:
                                    params[param_name] = choices[0]
                            else:
                                params[param_name] = choices[0]
                try:
                    return cls(**params)
                except Exception:
                    continue
        # Fallback to Sleep
        return Sleep()

    def _rank_hosts_by_suspicion(
        self,
        observation: Dict[str, Any],
        *,
        recency_window: int = 5,
        recency_penalty: int = 30,
        explore_eps: float = 0.15,
        explore_top_n: int = 3,
    ) -> List[str]:
        """Rank hostnames by suspicion, penalizing recently targeted hosts.

        Bug-5 fix: the original static/deterministic version always returned the
        same most-suspicious host, causing Analyse lock-in when the host stays
        suspicious after a read-only Analyse.  This version:
        - Penalizes hosts targeted in the last ``recency_window`` defender steps.
        - With probability ``explore_eps``, shuffles the top-N to break ties.
        """
        if not isinstance(observation, dict):
            return []

        reserved = {"success", "action", "phase", "message"}
        admin_iocs = {"escalate.sh", "escalate.exe"}
        user_iocs = {"cmd.sh", "cmd.exe"}

        # Count how many times each host was targeted recently.
        recent_counts: Dict[str, int] = {}
        cutoff = self._defender_step_counter - recency_window
        for hostname, step_idx in self._recently_targeted_hosts:
            if step_idx >= cutoff:
                recent_counts[hostname] = recent_counts.get(hostname, 0) + 1

        scored: List[Tuple[int, str]] = []
        for key, value in observation.items():
            if key in reserved or not isinstance(value, dict):
                continue

            hostname = str(key)
            sysinfo = value.get("System info")
            if isinstance(sysinfo, dict):
                hn = sysinfo.get("Hostname")
                if hn:
                    hostname = str(hn)

            score = 0

            files = value.get("Files")
            if isinstance(files, list):
                for f in files:
                    if not isinstance(f, dict):
                        continue
                    fname = f.get("File Name")
                    if fname in admin_iocs:
                        score += 100
                    elif fname in user_iocs:
                        score += 50
                    elif fname is not None:
                        score += 5

            procs = value.get("Processes")
            if isinstance(procs, list):
                for proc in procs:
                    if not isinstance(proc, dict):
                        continue
                    if "PID" in proc and "username" not in proc:
                        score += 20
                    conns = proc.get("Connections")
                    if isinstance(conns, list):
                        for conn in conns:
                            if not isinstance(conn, dict):
                                continue
                            if conn.get("remote_address"):
                                score += 1

            # Penalize hosts we already targeted recently.
            score -= recency_penalty * recent_counts.get(hostname, 0)

            if score > 0:
                scored.append((int(score), hostname))

        scored.sort(key=lambda x: (-x[0], x[1]))
        ranked = [hn for _s, hn in scored]

        # Epsilon-greedy exploration: occasionally shuffle top-N to break ties.
        if ranked and len(ranked) >= 2 and random.random() < explore_eps:
            top = ranked[:explore_top_n]
            random.shuffle(top)
            ranked = top + ranked[explore_top_n:]

        return ranked

    def set_actions(self, actions: List[Action], labels: List[str]) -> None:
        self._actions_cache = actions
        self._action_labels_cache = labels

    def _available_action_ids(self, action_space) -> set[str]:
        """Best-effort set of currently-available defender action ids (for candidate filtering)."""
        out: set[str] = set()

        if self._actions_cache:
            for act in self._actions_cache:
                try:
                    out.add(action_to_node_id(self.name, act))
                except Exception:
                    continue
            return out

        if isinstance(action_space, dict):
            action_map = action_space.get("action")
            if isinstance(action_map, dict):
                for cls, valid in action_map.items():
                    if not valid:
                        continue
                    try:
                        out.add(f"defender_{_camel_to_snake(cls.__name__)}")
                    except Exception:
                        continue
        return out

    def _log_llm_eval(
        self,
        prompt: List[Dict[str, str]],
        response: str,
        *,
        candidates: List[str],
        prompt_contains_banned_incident_words: bool,
        prompt_contains_banned_incident_words_pre: Optional[bool] = None,
        prompt_contains_banned_incident_words_post: Optional[bool] = None,
        llm_valid: bool,
        llm_confidence: float,
        llm_ranked_actions: List[str],
        llm_brief_reason: str,
        llm_scores: Dict[str, float],
        graph_prior: Dict[str, float],
        chosen_action: str,
        chosen_from: str,
        gate_g: Optional[float] = None,
        final_scores: Optional[Dict[str, float]] = None,
        visits_evidence: Optional[Dict[str, float]] = None,
        state_signature: Optional[str] = None,
        false_repeat_penalties: Optional[Dict[str, float]] = None,
        last_action_status: str = "",
        llm_debug: Optional[Dict[str, Any]] = None,
        traffic_action_cooldown_active: Optional[bool] = None,
        chosen_traffic_zone: Optional[Tuple[str, str]] = None,
        traffic_flipflop_detected: Optional[bool] = None,
        traffic_gate_applied: Optional[bool] = None,
        traffic_gate_reason: Optional[str] = None,
        traffic_evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        chosen = str(chosen_action)
        chosen_conf = float(llm_confidence)
        chosen_gate = float(gate_g) if gate_g is not None else 0.0
        chosen_evidence = float((visits_evidence or {}).get(chosen, 0.0) or 0.0)
        record = {
            "prompt": prompt,
            "response": response,
            "candidates": list(candidates),
            "prompt_contains_banned_incident_words": bool(prompt_contains_banned_incident_words),
            "prompt_contains_banned_incident_words_pre": bool(
                prompt_contains_banned_incident_words_pre
                if prompt_contains_banned_incident_words_pre is not None
                else False
            ),
            "prompt_contains_banned_incident_words_post": bool(
                prompt_contains_banned_incident_words_post
                if prompt_contains_banned_incident_words_post is not None
                else bool(prompt_contains_banned_incident_words)
            ),
            "llm_valid": bool(llm_valid),
            "llm_confidence": float(llm_confidence),
            "llm_ranked_actions": list(llm_ranked_actions or []),
            "llm_brief_reason": str(llm_brief_reason or ""),
            "llm_scores": llm_scores,
            "graph_prior": graph_prior,
            "gate_g": gate_g,
            "visits_evidence": visits_evidence or {},
            "state_signature": state_signature,
            "llm_debug": llm_debug or {},
            "final_scores": final_scores or {},
            "chosen_action": chosen,
            "chosen_from": str(chosen_from),
            # Backwards-compatible aliases for older analysis scripts.
            "chosen": chosen,
            "chosen_confidence": chosen_conf,
            "chosen_gate": chosen_gate,
            "chosen_visits_evidence": chosen_evidence,
            "last_action_status": last_action_status,
            "false_repeat_penalties": false_repeat_penalties or {},
            "traffic_action_cooldown_active": traffic_action_cooldown_active,
            "chosen_traffic_zone": chosen_traffic_zone,
            "traffic_flipflop_detected": traffic_flipflop_detected,
            "traffic_gate_applied": traffic_gate_applied,
            "traffic_gate_reason": traffic_gate_reason,
            "traffic_evidence": traffic_evidence,
        }
        # Promote compact score diagnostics to top-level fields for easier offline analysis.
        if isinstance(record.get("llm_debug", None), dict):
            dbg = record.get("llm_debug") or {}
            for k in ("llm_action_scores", "score_spread", "num_distinct_scores"):
                if k in dbg:
                    record[k] = dbg.get(k)
        try:
            with (self.log_dir / "self_eval_trace.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _log_step(self, record: Dict[str, Any]) -> None:
        try:
            with self.trace_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _maybe_discover_edge(self, from_id: str, to_id: str) -> None:
        """On-the-fly edge creation for sparse cold-start graphs."""
        if from_id not in self.graph.graph or to_id not in self.graph.graph:
            return
        if self.graph._get_agent_type(from_id) == self.graph._get_agent_type(to_id):
            return
        if self.graph.graph.has_edge(from_id, to_id):
            return
        created = self.graph.ensure_edge(from_id, to_id, description="discovered")
        if created:
            self._new_edges_discovered += 1
            self._log_step({"event": "edge_discovered", "from": from_id, "to": to_id})

    def _visits_evidence(self, action_id: str, *, state_signature: Optional[str] = None) -> float:
        """Evidence proxy in [0,1] from total outgoing visits (state-conditioned with backoff)."""
        if action_id not in self.graph.graph:
            return 0.0
        edges = self.graph.get_outgoing_edges(
            action_id,
            state_signature=state_signature if self.config.use_state_conditioning else None,
            backoff=True,
        )
        total_visits = sum(int(stats.get("visit_count", 0) or 0) for _to, _lbl, stats in edges)
        scale = max(1.0, float(self.config.gate_evidence_visits_scale))
        try:
            return float(min(1.0, max(0.0, math.log1p(total_visits) / math.log1p(scale))))
        except Exception:
            return 0.0

    def _risk_summary(self, action_id: str, *, state_signature: Optional[str] = None) -> Dict[str, Any]:
        """Aggregate, non-leaky risk summary for LLM prompting (no attacker IDs)."""
        if action_id not in self.graph.graph:
            return {
                "visits": 0,
                "evidence": 0.0,
                "responses_seen": 0,
                "resp_concentration": None,
                "resp_entropy": None,
            }
        edges = self.graph.get_outgoing_edges(
            action_id,
            state_signature=state_signature if self.config.use_state_conditioning else None,
            backoff=True,
        )
        counts: List[int] = []
        for _to, _lbl, stats in edges:
            v = int(stats.get("visit_count", 0) or 0)
            if v > 0:
                counts.append(v)

        total_visits = int(sum(counts))
        responses_seen = int(len(counts))
        evidence = float(self._visits_evidence(action_id, state_signature=state_signature))

        if total_visits <= 0 or responses_seen <= 0:
            return {
                "visits": 0,
                "evidence": evidence,
                "responses_seen": 0,
                "resp_concentration": None,
                "resp_entropy": None,
            }

        resp_conc = float(max(counts) / float(total_visits)) if counts else 0.0

        # Normalized entropy over observed response counts (0=concentrated, 1=uniform).
        if responses_seen <= 1:
            resp_entropy = 0.0
        else:
            ps = [c / float(total_visits) for c in counts if c > 0]
            h = -sum(p * math.log(p) for p in ps)
            resp_entropy = float(h / math.log(float(responses_seen)))

        return {
            "visits": total_visits,
            "evidence": evidence,
            "responses_seen": responses_seen,
            "resp_concentration": resp_conc,
            "resp_entropy": resp_entropy,
        }

    @staticmethod
    def _is_low_conf(conf: Dict[str, float]) -> bool:
        """Treat missing or near-zero confidences as invalid for fusion gating."""
        if not conf:
            return True
        vals = [float(v) for v in conf.values()]
        return all(v <= 0.05 for v in vals)

    @staticmethod
    def _clamp01(x: float) -> float:
        try:
            return float(min(1.0, max(0.0, x)))
        except Exception:
            return 0.0

    def _compute_gate(
        self,
        *,
        llm_score: float,
        llm_confidence: float,
        prior: float,
        visits_evidence: float,
        repeat_failures: int,
    ) -> float:
        """Robust fusion gate g in [0,1].

        Design goals:
        - Cold start (low evidence): allow LLM influence (higher g), unless flat/invalid.
        - High evidence: prefer graph unless LLM is very confident and consistent with the prior.
        - Repeat failures: reduce g to break self-reinforcing loops.
        """
        c = self._clamp01(float(llm_confidence))
        e = self._clamp01(float(visits_evidence))
        d = min(1.0, abs(float(llm_score) - float(prior)) / 9.0)
        consistency = 1.0 - d

        # Base: trust LLM more when evidence is low.
        g_base = c * (1.0 - 0.9 * e)
        # Boost: if evidence is high AND LLM is confident AND agrees with prior, allow some influence.
        g_boost = 0.35 * (c**2) * e * (consistency**2)
        g = (g_base + g_boost) * max(0.0, 1.0 - d * e)

        streak = max(0, int(repeat_failures))
        if streak > 0:
            g *= 1.0 / (1.0 + 0.75 * float(streak))

        return self._clamp01(float(g))

    # --- Confidence-aware helpers ---

    def _get_quality_scale(self) -> float:
        """Compute robust spread of mean_contribution across all visited edges.

        Uses a magnitude-aware floor so the scale never collapses to a tiny value
        when the spread is near zero but absolute mean_contributions are large.

        S = max(S_mu, kappa * S_r, epsilon)
        """
        import statistics as _stats

        rma = getattr(self.graph, 'reward_magnitude_anchor', 0.0)

        values = []
        for _, _, data in self.graph.graph.edges(data=True):
            if int(data.get("visit_count", 0) or 0) >= 1:
                values.append(float(data.get("mean_contribution", 0.0) or 0.0))

        kappa = 0.3
        epsilon = 1e-6

        if len(values) < 2:
            median_abs = abs(values[0]) if values else 0.0
            s_r = rma if rma > 0 else median_abs
            return max(kappa * s_r, epsilon)

        values.sort()
        # Use 10th-90th percentile spread for robustness
        lo_idx = max(0, int(len(values) * 0.1))
        hi_idx = min(len(values) - 1, int(len(values) * 0.9))
        s_mu = abs(values[hi_idx] - values[lo_idx])

        median_abs = _stats.median(abs(v) for v in values)
        s_r = rma if rma > 0 else median_abs

        return max(s_mu, kappa * s_r, epsilon)

    def _graph_prior_raw(
        self,
        action_id: str,
        *,
        state_signature: Optional[str] = None,
        bucket_id: Optional[str] = None,
    ) -> Optional[float]:
        """Compute UCB score for an action using its best outgoing edge.

        Fix 2: Replaces edge_combined_score with UCB scoring.
        Returns inf for actions with unvisited edges (n<2) to guarantee exploration.
        Returns None if the action has no outgoing edges at all.

        When bucket_id is provided, uses mixed (global + bucket-conditioned) stats.
        """
        if self.config.disable_graph:
            return None
        if action_id not in self.graph.graph:
            return None

        c = float(getattr(self.config, 'ucb_c', 1.0))

        if bucket_id and hasattr(self.graph, 'get_mixed_edge_stats'):
            lambda_state = float(getattr(self.config, 'lambda_state', 0.3))
            min_bucket_visits = int(getattr(self.config, 'min_bucket_visits', 5))
            edges_out = list(self.graph.graph.edges(action_id, data=True))
            if not edges_out:
                return None
            # N = sum of episode counts across all outgoing edges from this action
            N = sum(
                int(self.graph.get_mixed_edge_stats(
                    action_id, to_id, bucket_id, lambda_state, min_bucket_visits
                ).get("visit_count", 0) or 0)
                for _, to_id, _ in edges_out
            )
            best: Optional[float] = None
            for _, to_id, _ in edges_out:
                stats = self.graph.get_mixed_edge_stats(
                    action_id, to_id, bucket_id, lambda_state, min_bucket_visits
                )
                mc = float(stats.get("mean_contribution", 0.0) or 0.0)
                vc = int(stats.get("visit_count", 0) or 0)
                score = _ucb_score(mc, vc, N, c)
                if best is None or score > best:
                    best = score
            return best
        else:
            edges = self.graph.get_outgoing_edges(
                action_id,
                state_signature=state_signature if self.config.use_state_conditioning else None,
                backoff=True,
            )
            if not edges:
                return None
            # N = sum of episode counts across all outgoing edges from this action
            N = sum(int(s.get("visit_count", 0) or 0) for _, _, s in edges)
            best = None
            for _to, _label, stats in edges:
                mc = float(stats.get("mean_contribution", 0.0) or 0.0)
                vc = int(stats.get("visit_count", 0) or 0)
                score = _ucb_score(mc, vc, N, c)
                if best is None or score > best:
                    best = score
            return best

    def _graph_priors_batch(
        self,
        candidates: List[str],
        *,
        state_signature: Optional[str] = None,
        bucket_id: Optional[str] = None,
    ) -> Dict[str, float]:
        """Map UCB scores to [1, 10] using robust IQR-sigmoid normalization.

        Fix 2: Uses _ucb_score per edge and _map_to_prior_scale for [1,10] mapping.
        - Actions with no edges → inf (unvisited, highest priority).
        - Rank-based fallback when IQR≈0 or <3 candidates guarantees spread during cold-start.
        """
        if self.config.disable_graph:
            return {cid: 5.5 for cid in candidates}

        tau = float(getattr(self.config, 'prior_tau', 1.0))

        # Collect UCB scores; None (no edges) → inf (explore first)
        ucb_scores: List[float] = []
        for cid in candidates:
            rv = self._graph_prior_raw(cid, state_signature=state_signature, bucket_id=bucket_id)
            ucb_scores.append(float('inf') if rv is None else rv)

        priors_list = _map_to_prior_scale(ucb_scores, tau=tau)
        return {cid: priors_list[i] for i, cid in enumerate(candidates)}

    def _graph_prior(self, action_id: str, *, state_signature: Optional[str] = None) -> float:
        """Single-action prior (delegates to batch with one candidate)."""
        priors = self._graph_priors_batch([action_id], state_signature=state_signature)
        return priors.get(action_id, 5.5)

    def _calibrate_confidence(
        self,
        llm_scores: Dict[str, float],
        llm_conf: Dict[str, float],
        priors: Dict[str, float],
        *,
        state_signature: Optional[str] = None,
    ) -> Dict[str, float]:
        calibrated: Dict[str, float] = {}
        for cid, score in llm_scores.items():
            conf = llm_conf.get(cid, 0.0)
            prior = priors.get(cid, 5.5)
            diff = abs(score - prior) / 9.0
            agree_factor = 1.0 - diff
            base = conf * (0.5 + 0.5 * agree_factor)
            edges = self.graph.get_outgoing_edges(
                cid,
                state_signature=state_signature if self.config.use_state_conditioning else None,
                backoff=True,
            )
            if edges:
                avg_log_visit = sum(math.log(1 + int(e[2].get("visit_count", 0) or 0)) for e in edges) / len(edges)
                evidence = min(1.0, max(0.0, avg_log_visit / 3.5))
                base *= (0.5 + 0.5 * evidence)
            calibrated[cid] = min(1.0, max(0.0, base))
        return calibrated

    def _blend_score(self, llm_score: float, confidence: float, graph_prior: float) -> float:
        return confidence * llm_score + (1.0 - confidence) * graph_prior
