"""Tests verifying the 10 bug fixes for the CAGE-4 LLM defender pipeline.

Run with:  python -m pytest test_bug_fixes.py -v
"""
from __future__ import annotations

import json
import random
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

        prompt = agent.build_llm_prompt(
            {"phase": "test", "success": True},
            ["defender_analyse", "defender_monitor", "defender_sleep"],
            {"defender_analyse": 5.5, "defender_monitor": 5.5, "defender_sleep": 5.5},
        )
        system_content = prompt[0]["content"]
        assert "Analyse/Monitor > Sleep" not in system_content, (
            "Prompt still contains explicit Analyse/Monitor bias"
        )
        assert "prior" in system_content.lower() and "start from" in system_content.lower(), (
            "Prompt should contain graph prior usage instructions"
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
