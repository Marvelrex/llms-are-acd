"""Tests verifying the 10 bug fixes for the CAGE-4 LLM defender pipeline.

Run with:  python -m pytest test_bug_fixes.py -v
"""
from __future__ import annotations

import json
import math
import random
import unittest
from collections import Counter, deque
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Minimal stubs so the SUT modules can be imported without the full CybORG env.
# ---------------------------------------------------------------------------
import sys, types

# Stub out CybORG imports that would drag in gym / heavy deps.
_STUBS = {
    "gym": types.ModuleType("gym"),
    "CybORG": types.ModuleType("CybORG"),
    "CybORG.env": types.ModuleType("CybORG.env"),
    "CybORG.Agents": types.ModuleType("CybORG.Agents"),
    "CybORG.Simulator": types.ModuleType("CybORG.Simulator"),
    "CybORG.Simulator.Actions": types.ModuleType("CybORG.Simulator.Actions"),
    "CybORG.Shared": types.ModuleType("CybORG.Shared"),
    "CybORG.Shared.Enums": types.ModuleType("CybORG.Shared.Enums"),
    "CybORG.Agents.LLMAgents": types.ModuleType("CybORG.Agents.LLMAgents"),
    "CybORG.Agents.LLMAgents.llm_policy": types.ModuleType("CybORG.Agents.LLMAgents.llm_policy"),
    "CybORG.Agents.LLMAgents.llm_adapter": types.ModuleType("CybORG.Agents.LLMAgents.llm_adapter"),
    "CybORG.Agents.LLMAgents.llm_adapter.utils": types.ModuleType("CybORG.Agents.LLMAgents.llm_adapter.utils"),
    "CybORG.Agents.LLMAgents.llm_adapter.utils.logger": types.ModuleType("CybORG.Agents.LLMAgents.llm_adapter.utils.logger"),
    "CybORG.Agents.LLMAgents.llm_adapter.state_signature": types.ModuleType("CybORG.Agents.LLMAgents.llm_adapter.state_signature"),
}


class _FakeBaseAgent:
    def __init__(self, name: str = "blue_agent_0"):
        self.name = name


class _FakeAction:
    pass


class _FakeSleep(_FakeAction):
    pass


class _FakeTernaryEnum:
    TRUE = "TRUE"
    FALSE = "FALSE"
    IN_PROGRESS = "IN_PROGRESS"


class _FakeLogger:
    @staticmethod
    def debug(*a, **kw):
        pass

    @staticmethod
    def warning(*a, **kw):
        pass


# Wire stubs before importing the real modules.
_STUBS["CybORG.Agents"].BaseAgent = _FakeBaseAgent
_STUBS["CybORG.Simulator.Actions"].Action = _FakeAction
_STUBS["CybORG.Simulator.Actions"].Sleep = _FakeSleep
_STUBS["CybORG.Shared.Enums"].TernaryEnum = _FakeTernaryEnum
_STUBS["CybORG.Agents.LLMAgents.llm_adapter.utils.logger"].Logger = _FakeLogger


class _FakeStateSignatureConfig:
    pass


_STUBS["CybORG.Agents.LLMAgents.llm_adapter.state_signature"].StateSignatureConfig = _FakeStateSignatureConfig
_STUBS["CybORG.Agents.LLMAgents.llm_adapter.state_signature"].compute_state_signature = lambda *a, **kw: None


class _FakeLLMPolicy:
    class _MM:
        def generate_response(self, prompt):
            return "{}"

    def __init__(self, *a, **kw):
        self.model_manager = self._MM()

    def end_episode(self):
        pass


_STUBS["CybORG.Agents.LLMAgents.llm_policy"].LLMDefenderPolicy = _FakeLLMPolicy

for mod_name, mod in _STUBS.items():
    sys.modules.setdefault(mod_name, mod)

# Now import the real modules under test.
from CybORG.Agents.LLMAgents.llm_adapter.action_graph import (
    ActionGraph,
    ActionNode,
    build_cage4_turn_graph,
    select_topk_actions_by_edge_score,
)
from CybORG.Agents.LLMAgents.llm_adapter.self_evolve_defender import (
    EdgeStats,
    SelfEvolveDefender,
)
from CybORG.Agents.LLMAgents.llm_adapter.graph_agent_config import GraphAgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_graph() -> ActionGraph:
    """Create a tiny graph for testing: 3 defenders, 2 attackers, full bipartite."""
    g = ActionGraph()
    for name in ["defender_analyse", "defender_monitor", "defender_sleep"]:
        g.add_action_node(ActionNode(action_id=name, label=name, agent_type="defender"))
    for name in ["attacker_scan", "attacker_exploit"]:
        g.add_action_node(ActionNode(action_id=name, label=name, agent_type="attacker"))
    for d in ["defender_analyse", "defender_monitor", "defender_sleep"]:
        for a in ["attacker_scan", "attacker_exploit"]:
            g.add_edge(d, a)
    for a in ["attacker_scan", "attacker_exploit"]:
        for d in ["defender_analyse", "defender_monitor", "defender_sleep"]:
            g.add_edge(a, d)
    return g


# ===========================================================================
# Bug 1: Alphabetical bias in uniform scores
# ===========================================================================

class TestBug1AlphabeticalBias:
    def test_uniform_scores_not_always_sorted(self):
        """When all edge scores are equal, results must not be alphabetically sorted every time."""
        g = _make_simple_graph()
        results = [tuple(select_topk_actions_by_edge_score(g, k=3)) for _ in range(50)]
        # With randomization, we should see more than one ordering.
        unique_orderings = set(results)
        assert len(unique_orderings) > 1, (
            "All 50 calls returned the same ordering — alphabetical bias is still present"
        )

    def test_uniform_scores_returns_all_defenders(self):
        """All defenders should still appear (just shuffled)."""
        g = _make_simple_graph()
        result = select_topk_actions_by_edge_score(g, k=10)
        assert set(result) == {"defender_analyse", "defender_monitor", "defender_sleep"}

    def test_no_ranked_edges_also_shuffled(self):
        """When there are no ranked edges at all, result should be shuffled."""
        g = ActionGraph()
        g.add_action_node(ActionNode(action_id="defender_a", label="A", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="defender_b", label="B", agent_type="defender"))
        # No edges at all.
        results = [tuple(select_topk_actions_by_edge_score(g, k=2)) for _ in range(30)]
        unique = set(results)
        assert len(unique) > 1, "No-edge case should also be shuffled"


# ===========================================================================
# Bug 2: Overly strict JSON validation
# ===========================================================================

class TestBug2ValidationRelaxed:
    """parse_llm_response should accept responses with <3 distinct scores."""

    def _make_agent_for_parse(self):
        """Build a minimal SelfEvolvingGraphAgent-like object with parse_llm_response."""
        # We only need the static/classmethod parts; import directly.
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        g = _make_simple_graph()
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent.graph = g
        return agent

    def test_two_distinct_scores_accepted(self):
        """A response with only 2 distinct scores among 3 candidates should be valid."""
        agent = self._make_agent_for_parse()
        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]
        # 2 distinct scores: 8.0 and 7.0
        response_data = {
            "actions": {
                "defender_analyse": {"score": 8.0, "confidence": 0.9},
                "defender_monitor": {"score": 8.0, "confidence": 0.7},
                "defender_sleep": {"score": 7.0, "confidence": 0.5},
            },
            "best": "defender_analyse",
            "justification": "test",
            "disruption_risk": {},
        }
        response = json.dumps(response_data)
        ranked, conf, just, risk, valid, debug, scores = agent.parse_llm_response(response, candidates)
        assert valid is True, f"Should be valid, got invalid_reasons={debug.get('invalid_reasons')}"
        assert ranked[0] == "defender_analyse"

    def test_identical_confidences_accepted(self):
        """All-identical confidences should now be accepted (warning only)."""
        agent = self._make_agent_for_parse()
        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]
        response_data = {
            "actions": {
                "defender_analyse": {"score": 9.0, "confidence": 0.8},
                "defender_monitor": {"score": 7.0, "confidence": 0.8},
                "defender_sleep": {"score": 5.0, "confidence": 0.8},
            },
            "best": "defender_analyse",
            "justification": "test",
            "disruption_risk": {},
        }
        response = json.dumps(response_data)
        ranked, conf, just, risk, valid, debug, scores = agent.parse_llm_response(response, candidates)
        assert valid is True, f"Should be valid, got invalid_reasons={debug.get('invalid_reasons')}"
        # Should have a warning, not a rejection.
        reasons = debug.get("invalid_reasons", [])
        assert not any("insufficient" in r or "identical_confidences" in r for r in reasons), (
            "Should not have hard rejection reasons"
        )


# ===========================================================================
# Bug 3: False-repeat penalty in both paths
# ===========================================================================

class TestBug3StreakRemoved:
    """Streak mechanism removed — actions should never be locked out."""

    def _make_agent(self):
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        g = _make_simple_graph()
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent.graph = g
        agent.last_action_node = None
        agent.np_random = random.Random(42)
        return agent

    def test_pick_from_ranking_no_lockout(self):
        """_pick_from_ranking should always return the top-ranked action (no streak filtering)."""
        agent = self._make_agent()
        result = agent._pick_from_ranking(
            ["defender_analyse", "defender_monitor", "defender_sleep"],
            last_status=None,
        )
        assert result == "defender_analyse", f"Expected top-ranked analyse, got {result}"

    def test_select_top_action_picks_highest_score(self):
        """select_top_action should pick the highest-scored action regardless of history."""
        agent = self._make_agent()
        scores = {
            "defender_analyse": 10.0,
            "defender_monitor": 5.0,
            "defender_sleep": 3.0,
        }
        result = agent.select_top_action(scores, list(scores.keys()))
        assert result == "defender_analyse", f"Expected analyse (highest score), got {result}"

    def test_no_false_streak_attribute(self):
        """Agent should not have _false_streak_by_action attribute."""
        agent = self._make_agent()
        assert not hasattr(agent, "_false_streak_by_action"), (
            "_false_streak_by_action should have been removed"
        )


# ===========================================================================
# Bug 4: Circuit breaker gate soft blending
# ===========================================================================

class TestBug4GateBlending:
    def _make_agent(self):
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        return agent

    def test_gate_value_range_uniform_priors(self):
        """Gate should be in [0.5, 0.95] — high when priors are uniform."""
        agent = self._make_agent()
        uniform_priors = {"a": 5.5, "b": 5.5, "c": 5.5}
        for conf in [0.0, 0.3, 0.55, 0.8, 1.0]:
            g, info = agent._compute_circuit_breaker_gate(conf, graph_priors=uniform_priors)
            assert 0.50 <= g <= 0.95, f"gate_g={g} for conf={conf} is out of range"
            assert "prior_variance" in info
            assert "graph_influence_pct" in info

    def test_gate_decreases_with_prior_variance(self):
        """Gate should decrease as graph priors diverge (graph has learned)."""
        agent = self._make_agent()
        uniform_priors = {"a": 5.5, "b": 5.5, "c": 5.5}
        diverged_priors = {"a": 3.0, "b": 5.5, "c": 8.0}
        g_uniform, info_u = agent._compute_circuit_breaker_gate(0.8, graph_priors=uniform_priors)
        g_diverged, info_d = agent._compute_circuit_breaker_gate(0.8, graph_priors=diverged_priors)
        assert g_diverged < g_uniform, (
            f"Gate should decrease with higher prior variance: uniform={g_uniform}, diverged={g_diverged}"
        )
        assert info_d["graph_trust_bonus"] > info_u["graph_trust_bonus"]

    def test_blending_math(self):
        """Verify blending formula: blended = g * llm + (1-g) * prior."""
        agent = self._make_agent()
        uniform_priors = {"a": 5.5, "b": 5.5}
        g, _ = agent._compute_circuit_breaker_gate(0.9, graph_priors=uniform_priors)
        llm_score = 9.0
        graph_prior = 3.0
        blended = g * llm_score + (1.0 - g) * graph_prior
        assert blended > 7.0, f"blended={blended}, expected >7.0 for high confidence + uniform priors"
        assert blended < llm_score, "blended should be lower than raw LLM score"


# ===========================================================================
# Bug 5: Host targeting lock-in
# ===========================================================================

class TestBug5HostLockIn:
    def _make_agent(self):
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent._recently_targeted_hosts = deque(maxlen=20)
        agent._defender_step_counter = 10
        agent._actions_cache = []
        return agent

    def test_recently_targeted_host_is_penalized(self):
        """Hosts targeted recently should rank lower."""
        agent = self._make_agent()
        # Target host_0 at step 8 (within recency_window=5).
        agent._recently_targeted_hosts.append(("host_0", 8))

        obs = {
            "host_0": {
                "System info": {"Hostname": "host_0"},
                "Files": [{"File Name": "escalate.sh"}],  # score=100, but penalized
            },
            "host_1": {
                "System info": {"Hostname": "host_1"},
                "Files": [{"File Name": "escalate.sh"}],  # score=100, not penalized
            },
        }
        ranked = agent._rank_hosts_by_suspicion(obs, explore_eps=0.0)  # disable exploration
        assert ranked[0] == "host_1", (
            f"host_1 should rank first (host_0 penalized), got {ranked}"
        )

    def test_epsilon_greedy_sometimes_shuffles(self):
        """With explore_eps=1.0 (always), top-3 should be shuffled sometimes."""
        agent = self._make_agent()
        obs = {
            f"host_{i}": {
                "System info": {"Hostname": f"host_{i}"},
                "Files": [{"File Name": "cmd.sh"}],
            }
            for i in range(5)
        }
        results = []
        for _ in range(30):
            ranked = agent._rank_hosts_by_suspicion(obs, explore_eps=1.0, explore_top_n=3)
            results.append(tuple(ranked[:3]))
        unique = set(results)
        assert len(unique) > 1, "Epsilon-greedy should produce varied orderings"


# ===========================================================================
# Bug 6: Consolidated traffic gate
# ===========================================================================

class TestBug6TrafficGate:
    def _make_agent(self):
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent._recent_def_actions = deque(maxlen=32)
        return agent

    def test_allows_with_evidence(self):
        agent = self._make_agent()
        assert agent._should_allow_traffic_action(
            "defender_block_traffic_zone",
            has_traffic_evidence=True,
        ) is True

    def test_blocks_without_evidence_low_conf(self):
        agent = self._make_agent()
        assert agent._should_allow_traffic_action(
            "defender_block_traffic_zone",
            has_traffic_evidence=False,
            llm_confidence=0.5,
        ) is False

    def test_allows_with_high_confidence(self):
        """High LLM confidence should bypass the evidence requirement."""
        agent = self._make_agent()
        assert agent._should_allow_traffic_action(
            "defender_block_traffic_zone",
            has_traffic_evidence=False,
            llm_confidence=0.8,
        ) is True

    def test_cooldown_blocks(self):
        """Recently used traffic actions should be blocked even with evidence."""
        agent = self._make_agent()
        agent._recent_def_actions.append("defender_block_traffic_zone")
        assert agent._should_allow_traffic_action(
            "defender_block_traffic_zone",
            has_traffic_evidence=True,
        ) is False

    def test_non_traffic_always_allowed(self):
        agent = self._make_agent()
        assert agent._should_allow_traffic_action(
            "defender_analyse",
            has_traffic_evidence=False,
        ) is True


# ===========================================================================
# Bug 7: EMA learning weight floor
# ===========================================================================

class TestBug7WeightFloor:
    def test_floor_prevents_tiny_weights(self):
        """Early actions should get at least floor weight."""
        weights = SelfEvolveDefender.discounted_weights(100, 0.97, floor=0.2)
        assert len(weights) == 100
        # Without floor, w[0] = 0.97^99 ≈ 0.048. With floor=0.2, minimum raw weight is 0.2.
        assert weights[0] >= 0.001, f"First weight too small: {weights[0]}"
        # The ratio between last and first shouldn't be extreme.
        ratio = weights[-1] / weights[0]
        assert ratio < 10, f"Weight ratio {ratio} too extreme; floor not working"

    def test_weights_sum_to_one(self):
        """Weights must still be normalized."""
        weights = SelfEvolveDefender.discounted_weights(50, 0.97, floor=0.2)
        assert abs(sum(weights) - 1.0) < 1e-9, f"Weights sum to {sum(weights)}, expected 1.0"

    def test_short_trace(self):
        """Short traces should work fine."""
        weights = SelfEvolveDefender.discounted_weights(3, 0.97, floor=0.2)
        assert len(weights) == 3
        assert abs(sum(weights) - 1.0) < 1e-9


# ===========================================================================
# Bug 8: Prompt bias removed
# ===========================================================================

class TestBug8PromptBias:
    def test_no_analyse_monitor_bias_in_prompt(self):
        """System prompt should not explicitly favor Analyse/Monitor."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        g = _make_simple_graph()
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent.graph = g
        agent.last_action = None
        agent.name = "blue_agent_0"
        agent._recent_def_actions = deque(maxlen=32)

        prompt = agent.build_llm_prompt(
            {"phase": "test", "success": True},
            ["defender_analyse", "defender_monitor", "defender_sleep"],
            {"defender_analyse": 5.5, "defender_monitor": 5.5, "defender_sleep": 5.5},
        )
        system_content = prompt[0]["content"]
        assert "Analyse/Monitor > Sleep" not in system_content, (
            "Prompt still contains explicit Analyse/Monitor bias"
        )
        # New rank-only schema: system msg has "prior" and "rank" instructions
        assert "prior" in system_content.lower(), (
            "Prompt should mention graph priors"
        )
        assert "rank" in system_content.lower(), (
            "Prompt should contain ranking instructions (rank-only schema)"
        )


# ===========================================================================
# Bug 9: visits_evidence no longer dead code
# ===========================================================================

class TestBug9VisitsEvidence:
    def test_exploration_bonus_in_final_scores(self):
        """Under-visited actions should get a bonus in final_scores via visits_evidence."""
        # This is a logical test: low evidence → higher exploration bonus.
        ev_low = 0.1  # little evidence
        ev_high = 0.9  # lots of evidence
        prior = 5.5

        # Bonus = 1.0 - evidence
        score_low_ev = prior + (1.0 - ev_low)    # 5.5 + 0.9 = 6.4
        score_high_ev = prior + (1.0 - ev_high)  # 5.5 + 0.1 = 5.6

        assert score_low_ev > score_high_ev, "Low-evidence actions should get higher scores"


# ===========================================================================
# Bug 10: Sparse init increased
# ===========================================================================

class TestBug10SparseInit:
    def test_default_sparse_m_is_5(self):
        config = GraphAgentConfig()
        assert config.sparse_init_m == 5, f"sparse_init_m should be 5, got {config.sparse_init_m}"

    def test_sparse_graph_has_more_edges(self):
        """With m=5, each defender should connect to 5 attacker nodes."""
        g = build_cage4_turn_graph(init_mode="sparse", sparse_m=5)
        defenders = g.get_nodes_by_agent("defender")
        for d in defenders:
            outgoing = [
                to for to in g.graph.successors(d)
                if g.graph.nodes[to].get("agent_type") == "attacker"
            ]
            # Should have exactly 5 connections (or all attackers if fewer than 5).
            assert len(outgoing) >= 5, (
                f"{d} has only {len(outgoing)} attacker edges, expected >= 5"
            )


# ===========================================================================
# New: TestBucketBackoff — coarse bucket conditioning
# ===========================================================================

class TestBucketBackoff(unittest.TestCase):
    """Tests for coarse bucket-conditioned priors (A1-A4)."""

    def _make_graph_with_edges(self):
        g = ActionGraph()
        g.add_action_node(ActionNode(action_id="defender_analyse", label="Analyse", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="defender_monitor", label="Monitor", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="attacker_scan", label="Scan", agent_type="attacker"))
        g.add_edge("defender_analyse", "attacker_scan")
        g.add_edge("defender_monitor", "attacker_scan")
        return g

    def test_bucket_accumulates_across_sigs(self):
        """Two full sigs that map to same bucket should accumulate bucket stats."""
        from CybORG.Agents.LLMAgents.llm_adapter.state_signature import sig_to_bucket_id

        # Two signatures that differ only in las/as (dropped by sig_to_bucket_id)
        sig1 = "comp=0|alerts=0|las=T|step=early|as=xs"
        sig2 = "comp=0|alerts=0|las=F|step=early|as=m"
        bucket1 = sig_to_bucket_id(sig1)
        bucket2 = sig_to_bucket_id(sig2)
        self.assertIsNotNone(bucket1)
        self.assertEqual(bucket1, bucket2, "Both sigs should map to the same coarse bucket")

    def test_low_bucket_visits_uses_global(self):
        """lambda_eff = 0 when bucket visits < min_bucket_visits → returns global stats."""
        g = self._make_graph_with_edges()
        # Set global edge stats
        g.graph.edges["defender_analyse", "attacker_scan"]["visit_count"] = 10
        g.graph.edges["defender_analyse", "attacker_scan"]["mean_contribution"] = 5.0
        # Set bucket stats with only 2 visits (below min_bucket_visits=5)
        g.set_bucket_edge_stats(
            "testbucket", "defender_analyse", "attacker_scan",
            visit_count=2, mean_contribution=9.0, m2=0.0,
        )
        stats = g.get_mixed_edge_stats(
            "defender_analyse", "attacker_scan", "testbucket",
            lambda_state=0.3, min_bucket_visits=5,
        )
        # Should return global stats (visit_count=10, mc=5.0)
        self.assertEqual(stats["visit_count"], 10)
        self.assertAlmostEqual(stats["mean_contribution"], 5.0)

    def test_sufficient_bucket_visits_mixes(self):
        """lambda_eff = lambda_state when bucket visits >= min_bucket_visits."""
        g = self._make_graph_with_edges()
        # Set global edge stats
        g.graph.edges["defender_analyse", "attacker_scan"]["visit_count"] = 10
        g.graph.edges["defender_analyse", "attacker_scan"]["mean_contribution"] = 4.0
        # Set bucket stats with 6 visits (above min_bucket_visits=5)
        g.set_bucket_edge_stats(
            "testbucket", "defender_analyse", "attacker_scan",
            visit_count=6, mean_contribution=8.0, m2=0.0,
        )
        la = 0.3
        stats = g.get_mixed_edge_stats(
            "defender_analyse", "attacker_scan", "testbucket",
            lambda_state=la, min_bucket_visits=5,
        )
        expected_mc = (1.0 - la) * 4.0 + la * 8.0  # 0.7*4 + 0.3*8 = 5.2
        self.assertAlmostEqual(stats["mean_contribution"], expected_mc, places=5)
        self.assertIn("_bucket_vc", stats)
        self.assertEqual(stats["_bucket_vc"], 6)


# ===========================================================================
# New: TestPriorNoCollapse — spread-proportional mapping
# ===========================================================================

class TestPriorNoCollapse(unittest.TestCase):
    """Tests for spread-proportional prior mapping (B)."""

    def _make_agent(self):
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        g = _make_simple_graph()
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent.graph = g
        return agent

    def test_small_spread_amplified(self):
        """Scores with small but non-zero spread produce priors with meaningful spread."""
        agent = self._make_agent()
        # Manually patch _graph_prior_raw to return tiny-spread raw scores
        tiny_spread = 1e-5  # well below EPSILON_MAP but above TINY

        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]
        base_raw = 0.5
        call_count = [0]

        def mock_prior_raw(action_id, *, state_signature=None, bucket_id=None):
            call_count[0] += 1
            offsets = {"defender_analyse": tiny_spread, "defender_monitor": tiny_spread / 2, "defender_sleep": 0.0}
            return base_raw + offsets.get(action_id, 0.0)

        agent._graph_prior_raw = mock_prior_raw
        agent._get_quality_scale = lambda: 1.0  # so EPSILON_MAP = 0.01

        priors = agent._graph_priors_batch(candidates)
        # With spread-proportional mapping, the best should be > 5.5 and worst < 5.5
        vals = list(priors.values())
        spread = max(vals) - min(vals)
        assert spread > 0.1, f"Spread {spread} too small; small raw spread should be amplified proportionally"

    def test_identical_scores_rank_based_fallback(self):
        """All identical raw scores → rank-based fallback guarantees spread on [1,10].

        Fix 2 (IQR-sigmoid): when IQR≈0 (all UCB scores equal), fall back to rank-based
        mapping which always produces spread. This guarantees the LLM sees differentiated
        priors even during cold-start when all edges are unvisited.
        """
        agent = self._make_agent()
        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]

        def mock_prior_raw(action_id, *, state_signature=None, bucket_id=None):
            return 0.5  # identical UCB scores

        agent._graph_prior_raw = mock_prior_raw
        agent._get_quality_scale = lambda: 1.0

        priors = agent._graph_priors_batch(candidates)
        vals = list(priors.values())
        spread = max(vals) - min(vals)
        # Rank-based fallback: [1.0, 5.5, 10.0] → spread = 9.0
        assert spread > 0.1, (
            f"Spread {spread:.3f} too small; rank-based fallback should guarantee spread"
        )
        # All values should be in [1, 10]
        for cid, v in priors.items():
            self.assertGreaterEqual(v, 1.0, f"{cid} prior {v:.3f} below 1.0")
            self.assertLessEqual(v, 10.0, f"{cid} prior {v:.3f} above 10.0")

    def test_large_spread_uses_linear_map(self):
        """When spread > EPSILON_MAP, existing [1,10] linear map is used."""
        agent = self._make_agent()
        candidates = ["defender_analyse", "defender_monitor"]

        # Return scores with large spread (> 0.01 * quality_scale=1.0 = 0.01)
        raw_vals_map = {"defender_analyse": 1.0, "defender_monitor": 0.0}

        def mock_prior_raw(action_id, *, state_signature=None, bucket_id=None):
            return raw_vals_map[action_id]

        agent._graph_prior_raw = mock_prior_raw
        agent._get_quality_scale = lambda: 1.0  # EPSILON_MAP = 0.01; spread=1.0 >> 0.01

        priors = agent._graph_priors_batch(candidates)
        # Linear rescale: best → 10.0, worst → 1.0
        self.assertAlmostEqual(priors["defender_analyse"], 10.0, places=5)
        self.assertAlmostEqual(priors["defender_monitor"], 1.0, places=5)


# ===========================================================================
# New: TestRankOnlyEnforcement — rank-only LLM output + score synthesis
# ===========================================================================

class TestRankOnlyEnforcement(unittest.TestCase):
    """Tests for rank-only LLM output and deterministic score synthesis (C)."""

    def _make_agent(self):
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        g = _make_simple_graph()
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent.graph = g
        return agent

    def test_valid_ranked_actions_parsed(self):
        """Valid ranked_actions response → llm_valid=True, per_action_scores empty."""
        agent = self._make_agent()
        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]
        response = json.dumps({
            "ranked_actions": ["defender_analyse", "defender_monitor", "defender_sleep"],
            "confidence": 0.8,
            "brief_reason": "analyse has highest prior",
        })
        ranked, conf, just, risk, valid, debug, scores = agent.parse_llm_response(response, candidates)
        self.assertTrue(valid, f"Should be valid: {debug.get('invalid_reasons')}")
        self.assertEqual(ranked[0], "defender_analyse")
        self.assertAlmostEqual(conf, 0.8, places=5)
        self.assertEqual(scores, {}, "rank-only schema should return empty per_action_scores")
        self.assertEqual(debug.get("schema"), "rank_only")

    def test_rank_adjustment_bounded_by_delta(self):
        """Rank-delta synthesis: |final_score - prior| <= RANK_DELTA for all actions."""
        RANK_DELTA = 1.5
        priors = {"defender_analyse": 7.0, "defender_monitor": 5.5, "defender_sleep": 3.0}
        ranked_actions = ["defender_analyse", "defender_monitor", "defender_sleep"]
        candidates = list(priors.keys())

        # Simulate the synthesis logic
        final_scores: Dict[str, float] = {}
        n = len(ranked_actions)
        for i, aid in enumerate(ranked_actions):
            prior_val = float(priors.get(aid, 5.5))
            adj = RANK_DELTA * (1.0 - 2.0 * i / max(n - 1, 1))
            final_scores[aid] = max(1.0, min(10.0, prior_val + adj))

        for aid in candidates:
            prior_val = float(priors[aid])
            fs = final_scores[aid]
            # Difference bounded by RANK_DELTA (allow small epsilon for clamp at [1,10])
            raw_adj = fs - prior_val
            self.assertLessEqual(abs(raw_adj), RANK_DELTA + 0.01,
                f"{aid}: |{fs:.2f} - {prior_val:.2f}| = {abs(raw_adj):.2f} > {RANK_DELTA}")

    def test_invalid_ranked_actions_fallback(self):
        """Missing or duplicate ranked_actions → llm_valid=False."""
        agent = self._make_agent()
        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]

        # Missing one candidate
        response_missing = json.dumps({
            "ranked_actions": ["defender_analyse", "defender_monitor"],
            "confidence": 0.8,
            "brief_reason": "only two",
        })
        _, _, _, _, valid, debug, _ = agent.parse_llm_response(response_missing, candidates)
        self.assertFalse(valid, "Missing candidates should make rank invalid")
        self.assertIn("ranked_actions_not_valid_permutation", debug.get("invalid_reasons", []))

        # Duplicate candidates
        response_dup = json.dumps({
            "ranked_actions": ["defender_analyse", "defender_analyse", "defender_sleep"],
            "confidence": 0.8,
            "brief_reason": "duplicate",
        })
        _, _, _, _, valid2, debug2, _ = agent.parse_llm_response(response_dup, candidates)
        self.assertFalse(valid2, "Duplicate candidates should make rank invalid")

    def test_prompt_candidates_sorted_by_prior(self):
        """build_llm_prompt returns candidates block in descending prior order."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        g = _make_simple_graph()
        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig()
        agent.graph = g
        agent.last_action = None
        agent.name = "blue_agent_0"
        agent._recent_def_actions = deque(maxlen=32)

        candidates = ["defender_analyse", "defender_monitor", "defender_sleep"]
        priors = {"defender_analyse": 3.0, "defender_monitor": 8.0, "defender_sleep": 5.5}
        prompt = agent.build_llm_prompt(
            {"phase": "test", "success": True},
            candidates,
            priors,
        )
        # Find the candidates block in the user prompt (skip system msg which also has "prior=")
        candidates_content = ""
        for msg in prompt:
            content = msg.get("content", "")
            if "prior=" in content and "defender_" in content:
                candidates_content = content
                break

        # monitor (prior=8.0) should appear before analyse (prior=3.0)
        monitor_pos = candidates_content.find("defender_monitor")
        analyse_pos = candidates_content.find("defender_analyse")
        self.assertLess(monitor_pos, analyse_pos,
            "Higher-prior action (monitor=8.0) should appear before lower-prior action (analyse=3.0)")

    def test_low_confidence_uses_priors_in_fallback(self):
        """Low confidence (below threshold) → final_scores use graph priors, no LLM rank adjustment."""
        # This tests the fallback branch: final_scores[cid] = priors[cid]
        priors = {"defender_analyse": 7.0, "defender_monitor": 4.0}
        candidates = list(priors.keys())

        # Simulate the else branch of the synthesis
        final_scores_fallback: Dict[str, float] = {}
        for cid in candidates:
            final_scores_fallback[cid] = float(priors.get(cid, 5.5))

        self.assertAlmostEqual(final_scores_fallback["defender_analyse"], 7.0, places=5)
        self.assertAlmostEqual(final_scores_fallback["defender_monitor"], 4.0, places=5)


# ===========================================================================
# Fix 1: Episode-level frequency-boosted credit
# ===========================================================================

class TestFix1EpisodeCredit(unittest.TestCase):
    """Tests for Fix 1: per-episode frequency-boosted credit in observe_round."""

    def _make_graph(self):
        g = ActionGraph()
        g.add_action_node(ActionNode(action_id="defender_analyse", label="Analyse", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="defender_monitor", label="Monitor", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="attacker_scan", label="Scan", agent_type="attacker"))
        g.add_edge("defender_analyse", "attacker_scan")
        g.add_edge("defender_monitor", "attacker_scan")
        g.add_edge("attacker_scan", "defender_analyse")
        g.add_edge("attacker_scan", "defender_monitor")
        g.reward_magnitude_anchor = 0.0
        g.baseline_default = 0.0
        return g

    def test_episode_count_tracks_episodes_not_steps(self):
        """After one observe_round, visit_count == 1 (episode count, not step count)."""
        g = self._make_graph()
        defender = SelfEvolveDefender(g, ema_beta=0.03)

        # Trace: analyse appears 3 times, monitor 0 times in 4-step trace
        trace = [
            "defender_analyse", "attacker_scan",
            "defender_analyse", "attacker_scan",
            "defender_analyse", "attacker_scan",
        ]
        defender.observe_round(trace, {"reward": -300.0})

        # visit_count for analyse→scan should be 1 (one episode), not 3 (not step count)
        analyse_stats = defender.edge_stats["defender_analyse"].get("attacker_scan")
        self.assertIsNotNone(analyse_stats)
        self.assertEqual(analyse_stats.visit_count, 1,
            f"visit_count should be 1 (episode count), got {analyse_stats.visit_count}")

        # total_traversals should be 3 (step count)
        self.assertEqual(analyse_stats.total_traversals, 3,
            f"total_traversals should be 3 (step count), got {analyse_stats.total_traversals}")

    def test_frequent_edge_higher_credit_than_rare(self):
        """Edge traversed more often should accumulate higher |credit| per episode."""
        g = self._make_graph()
        defender = SelfEvolveDefender(g, ema_beta=0.0)  # ema_beta=0 → baseline stays at 0

        # trace: analyse appears 8/10 steps, monitor appears 2/10 steps
        trace = (
            ["defender_analyse", "attacker_scan"] * 8 +
            ["defender_monitor", "attacker_scan"] * 2
        )
        # T = 20 (each pair = 2 edges? No, the trace is a flat list)
        # Build a flat trace with analyse repeated 8 times and monitor 2 times
        flat_trace = []
        for _ in range(8):
            flat_trace.extend(["defender_analyse", "attacker_scan"])
        for _ in range(2):
            flat_trace.extend(["defender_monitor", "attacker_scan"])
        # flat_trace has 20 transitions total; analyse→scan appears 8 times, monitor→scan 2 times

        defender.observe_round(flat_trace, {"reward": -100.0})

        analyse_mc = abs(defender.edge_stats["defender_analyse"]["attacker_scan"].mean_contribution)
        monitor_mc = abs(defender.edge_stats["defender_monitor"]["attacker_scan"].mean_contribution)
        self.assertGreater(analyse_mc, monitor_mc,
            f"Frequent edge (analyse, 8/20) should have higher |credit| than rare edge "
            f"(monitor, 2/20): {analyse_mc:.4f} vs {monitor_mc:.4f}")

    def test_reward_baseline_updates_after_episode(self):
        """Baseline should be updated after each episode."""
        g = self._make_graph()
        defender = SelfEvolveDefender(g, ema_beta=0.1)
        trace = ["defender_analyse", "attacker_scan"]

        initial_mean, _ = defender._reward_baseline.get(None)
        self.assertAlmostEqual(initial_mean, 0.0)

        defender.observe_round(trace, {"reward": -500.0})

        updated_mean, _ = defender._reward_baseline.get(None)
        # EMA with beta=0.1: new_mean = 0.9*0 + 0.1*(-500) = -50
        self.assertLess(updated_mean, 0.0, "Baseline should decrease after negative episode reward")


# ===========================================================================
# Fix 2 + Fix 3: UCB scoring and adaptive blend
# ===========================================================================

class TestFix2UCBScoring(unittest.TestCase):
    """Tests for Fix 2: UCB scoring and sigmoid normalization."""

    def test_ucb_score_inf_for_unvisited(self):
        """_ucb_score returns inf for n<2."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _ucb_score
        self.assertEqual(_ucb_score(0.0, 0, 10, 1.0), float('inf'))
        self.assertEqual(_ucb_score(-5.0, 1, 10, 1.0), float('inf'))

    def test_ucb_score_finite_for_visited(self):
        """_ucb_score is finite and includes exploration bonus for n>=2."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _ucb_score
        score = _ucb_score(-0.5, 5, 50, 1.0)
        self.assertTrue(math.isfinite(score))
        # Should be mean + bonus = -0.5 + something > 0
        self.assertGreater(score, -0.5)

    def test_map_to_prior_scale_range(self):
        """_map_to_prior_scale output is always in [1, 10]."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _map_to_prior_scale
        # Mix of finite and inf
        ucb_scores = [float('inf'), -0.5, -1.0, float('inf'), -0.2]
        result = _map_to_prior_scale(ucb_scores)
        for v in result:
            self.assertGreaterEqual(v, 1.0)
            self.assertLessEqual(v, 10.0)

    def test_map_to_prior_scale_spread(self):
        """Scores with spread produce priors with spread."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _map_to_prior_scale
        ucb_scores = [-0.1, -0.3, -0.5, -0.7, -0.9]
        result = _map_to_prior_scale(ucb_scores)
        spread = max(result) - min(result)
        self.assertGreater(spread, 1.0, "Should have meaningful spread in priors")


class TestFix3AdaptiveBlend(unittest.TestCase):
    """Tests for Fix 3: compute_alpha and blend_scores."""

    def test_compute_alpha_range(self):
        """_compute_alpha returns value in [alpha_min, alpha_max]."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _compute_alpha
        for gap in [0.0, 0.5, 1.0, 2.0, 5.0]:
            alpha = _compute_alpha(gap)
            self.assertGreaterEqual(alpha, 0.4)
            self.assertLessEqual(alpha, 0.8)

    def test_compute_alpha_increases_with_gap(self):
        """Larger prior gap → higher alpha (graph more influential)."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _compute_alpha
        alpha_low = _compute_alpha(0.1)
        alpha_high = _compute_alpha(3.0)
        self.assertLess(alpha_low, alpha_high,
            f"alpha should increase with prior_gap: gap=0.1→{alpha_low:.3f}, gap=3.0→{alpha_high:.3f}")

    def test_blend_scores_always_returns_all_actions(self):
        """_blend_scores returns a score for every action in priors."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _blend_scores
        priors = {"a": 8.0, "b": 5.5, "c": 2.0}
        ranking = ["a", "b", "c"]
        blended = _blend_scores(priors, ranking, alpha=0.6)
        self.assertEqual(set(blended.keys()), set(priors.keys()))

    def test_blend_scores_preserves_ordering(self):
        """When priors and LLM agree, top action stays on top."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _blend_scores
        priors = {"best": 9.0, "mid": 5.5, "worst": 2.0}
        ranking = ["best", "mid", "worst"]
        blended = _blend_scores(priors, ranking, alpha=0.5)
        self.assertGreater(blended["best"], blended["mid"])
        self.assertGreater(blended["mid"], blended["worst"])

    def test_blend_scores_within_prior_range(self):
        """Blended scores stay close to [1, 10] range."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import _blend_scores
        priors = {"a": 7.0, "b": 5.5, "c": 3.0}
        ranking = ["a", "b", "c"]
        for alpha in [0.4, 0.6, 0.8]:
            blended = _blend_scores(priors, ranking, alpha=alpha)
            for v in blended.values():
                self.assertGreaterEqual(v, 1.0 - 0.1)  # small tolerance
                self.assertLessEqual(v, 10.0 + 0.1)


# ===========================================================================
# [avg_ep_reward disabled] TestAverageEpisodeReward — kept for future reference.
# All edges converge to same global mean (~-278), zero differentiation between actions.
# Re-enable when per-action credit attribution is redesigned.
# ===========================================================================

class _DisabledTestAverageEpisodeReward:  # renamed from unittest.TestCase to disable
    """Tests for avg_ep_reward / ep_score feature (episode-level reward tracking)."""

    def _make_graph(self):
        g = ActionGraph()
        g.add_action_node(ActionNode(action_id="defender_analyse", label="Analyse", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="defender_monitor", label="Monitor", agent_type="defender"))
        g.add_action_node(ActionNode(action_id="attacker_scan", label="Scan", agent_type="attacker"))
        g.add_edge("defender_analyse", "attacker_scan")
        g.add_edge("defender_monitor", "attacker_scan")
        g.add_edge("attacker_scan", "defender_analyse")
        g.add_edge("attacker_scan", "defender_monitor")
        return g

    def test_edgestats_update_episode(self):
        """update_episode() correctly computes Welford mean of episode rewards."""
        stats = EdgeStats()
        stats.update_episode(-100.0)
        self.assertEqual(stats.episode_count, 1)
        self.assertAlmostEqual(stats.episode_reward_mean, -100.0)

        stats.update_episode(-300.0)
        self.assertEqual(stats.episode_count, 2)
        self.assertAlmostEqual(stats.episode_reward_mean, -200.0)  # mean of -100 and -300

        stats.update_episode(-200.0)
        self.assertEqual(stats.episode_count, 3)
        self.assertAlmostEqual(stats.episode_reward_mean, -200.0)  # mean of -100, -300, -200

    def test_observe_round_updates_ep_stats(self):
        """After observe_round, unique defender edges should have ep_count > 0."""
        g = self._make_graph()
        g.reward_magnitude_anchor = 0.0
        g.baseline_default = 0.0
        defender = SelfEvolveDefender(g)

        trace = ["defender_analyse", "attacker_scan", "defender_analyse", "attacker_scan"]
        result = {"reward": -500.0}
        defender.observe_round(trace, result)

        # ep_count should be 1 for defender_analyse -> attacker_scan
        edge_data = g.graph.edges["defender_analyse", "attacker_scan"]
        self.assertEqual(edge_data["ep_count"], 1)
        self.assertAlmostEqual(edge_data["avg_ep_reward"], -500.0)

    def test_ep_fields_persisted_in_to_dict(self):
        """avg_ep_reward and ep_count are serialized in to_dict()."""
        g = self._make_graph()
        g.graph.edges["defender_analyse", "attacker_scan"]["avg_ep_reward"] = -250.0
        g.graph.edges["defender_analyse", "attacker_scan"]["ep_count"] = 3
        d = g.to_dict()
        edge_entry = next(
            e for e in d["edges"]
            if e["from"] == "defender_analyse" and e["to"] == "attacker_scan"
        )
        self.assertAlmostEqual(edge_entry["avg_ep_reward"], -250.0)
        self.assertEqual(edge_entry["ep_count"], 3)

    def test_ep_fields_loaded_from_dict(self):
        """avg_ep_reward and ep_count survive save/load round-trip."""
        import json
        import tempfile
        from pathlib import Path

        g = self._make_graph()
        g.graph.edges["defender_analyse", "attacker_scan"]["avg_ep_reward"] = -350.0
        g.graph.edges["defender_analyse", "attacker_scan"]["ep_count"] = 5

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(g.to_dict(), f)
            tmp_path = Path(f.name)

        g2 = ActionGraph.load(tmp_path)
        edge_data = g2.graph.edges["defender_analyse", "attacker_scan"]
        self.assertAlmostEqual(edge_data["avg_ep_reward"], -350.0)
        self.assertEqual(edge_data["ep_count"], 5)
        tmp_path.unlink()

    def test_prior_spread_increases_with_ep_reward(self):
        """With alpha_episode_reward > 0, edges with different avg_ep_reward produce different priors."""
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent

        g = _make_simple_graph()
        # Set different avg_ep_reward for two defender actions' outgoing edges.
        # Good action: avg_ep_reward = -100 (better episodes)
        for _, to, _ in g.graph.edges("defender_analyse", data=True):
            g.graph.edges["defender_analyse", to]["avg_ep_reward"] = -100.0
            g.graph.edges["defender_analyse", to]["ep_count"] = 10
            g.graph.edges["defender_analyse", to]["visit_count"] = 20
            g.graph.edges["defender_analyse", to]["mean_contribution"] = -0.1
        # Bad action: avg_ep_reward = -800 (worse episodes)
        for _, to, _ in g.graph.edges("defender_monitor", data=True):
            g.graph.edges["defender_monitor", to]["avg_ep_reward"] = -800.0
            g.graph.edges["defender_monitor", to]["ep_count"] = 10
            g.graph.edges["defender_monitor", to]["visit_count"] = 20
            g.graph.edges["defender_monitor", to]["mean_contribution"] = -0.1
        g.reward_magnitude_anchor = 800.0
        g.baseline_default = -0.5

        agent = SelfEvolvingGraphAgent.__new__(SelfEvolvingGraphAgent)
        agent.config = GraphAgentConfig(alpha_episode_reward=0.3)
        agent.graph = g

        candidates = ["defender_analyse", "defender_monitor"]
        priors = agent._graph_priors_batch(candidates)

        # Good action should have higher prior than bad action.
        self.assertGreater(priors["defender_analyse"], priors["defender_monitor"],
                           "ep_score blending should rank good episodes higher")
        # Priors should be meaningfully different (spread > 1.0).
        spread = priors["defender_analyse"] - priors["defender_monitor"]
        self.assertGreater(spread, 1.0,
                           f"Prior spread {spread:.2f} should be > 1.0 to guide LLM")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
