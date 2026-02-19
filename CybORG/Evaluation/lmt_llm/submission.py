from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import wandb

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from CybORG.Agents.CybermonicAgents.cage4 import load as load_cybermonic_agent
from CybORG.Agents.LLMAgents.llm_agent import DefenderAgent
from CybORG.Agents.LLMAgents.llm_policy import LLMDefenderPolicy
from CybORG.Agents.LLMAgents.config.config_vars import (
    BLUE_AGENT_NAME,
    SUB_NAME,
    SUB_TEAM,
    SUB_TECHNIQUE,
)
from CybORG.Agents.Wrappers import BaseWrapper
from CybORG.Agents.Wrappers.BlueFixedActionWrapper import EMPTY_MESSAGE
from CybORG.Agents.Wrappers.CybermonicWrappers.graph_wrapper import GraphWrapper
from CybORG.Simulator.Actions import Action


def _is_truthy(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class Submission:
    # Submission name
    NAME: str = SUB_NAME

    # Name of your team
    TEAM: str = SUB_TEAM

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = SUB_TECHNIQUE

    USE_RL_AGENTS = _is_truthy(os.environ.get("CAGE4_USE_RL_AGENTS"))
    _default_weights = Path(__file__).resolve().parents[1] / "Cybermonics" / "weights"
    _weights_dir = Path(os.environ.get("CAGE4_RL_WEIGHTS_DIR", str(_default_weights)))

    if USE_RL_AGENTS:
        # NOTE: Avoid dict comprehension in class scope (it cannot see class-local names like _weights_dir).
        AGENTS: dict[str, BaseAgent] = {}
        for agent in range(5):
            AGENTS[f"blue_agent_{agent}"] = load_cybermonic_agent(
                str(_weights_dir / f"gnn_ppo-{agent}.pt")
            )
        AGENTS[BLUE_AGENT_NAME] = DefenderAgent(BLUE_AGENT_NAME, LLMDefenderPolicy, [])
    else:
        AGENTS = {BLUE_AGENT_NAME: DefenderAgent(BLUE_AGENT_NAME, LLMDefenderPolicy, [])}

    @classmethod
    def wrap(cls, env: CybORG):
        if getattr(wandb, "run", None) is None:
            wandb.init(mode="disabled")
        if cls.USE_RL_AGENTS:
            return LMTHybridWrapper(env)
        return LMTPhaseWrapper(env)


class LMTPhaseWrapper(BaseWrapper):
    def action_space(self, agent_name: str):
        if hasattr(self.env, "action_space"):
            return self.env.action_space(agent_name)
        return self.env.get_action_space(agent_name)

    def _sync_from_env(self):
        self.agents = [a for a in self.env.agents if "blue" in a]
        return {a: self._decorate_obs(self.env.get_observation(a)) for a in self.agents}

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        return self._sync_from_env(), {}

    def step(
        self,
        actions: dict[str, Action] = {},
        messages: dict[str, Any] = None,
        **kwargs,
    ):
        if messages is None:
            messages = {a: EMPTY_MESSAGE for a in self.agents}

        obs, rews, dones, info = self.env.parallel_step(actions, messages=messages, **kwargs)

        observations = {
            agent: self._decorate_obs(o)
            for agent, o in obs.items()
            if "blue" in agent
        }
        rewards = {
            agent: sum(reward.values())
            for agent, reward in rews.items()
            if "blue" in agent
        }

        terminated = {agent: dones[agent] for agent in observations.keys()}
        truncated = {agent: dones[agent] for agent in observations.keys()}

        self.agents = [agent for agent, done in dones.items() if "blue" in agent and not done]
        return observations, rewards, terminated, truncated, {}

    def _decorate_obs(self, observation: dict) -> dict:
        observation["phase"] = self.env.environment_controller.state.mission_phase
        if "message" not in observation:
            observation["message"] = [EMPTY_MESSAGE for _ in range(4)]
        return observation


class LMTHybridWrapper(BaseWrapper):
    """Hybrid wrapper: one LLM blue agent plus four PPO blue agents."""

    def __init__(self, env: CybORG):
        super().__init__(env)
        self.phase_wrapper = LMTPhaseWrapper(env)
        self.graph_wrapper = GraphWrapper(env)

    def action_space(self, agent_name: str):
        if agent_name == BLUE_AGENT_NAME:
            return self.phase_wrapper.action_space(agent_name)
        return None

    def reset(self, *args, **kwargs):
        graph_obs, graph_info = self.graph_wrapper.reset(*args, **kwargs)
        phase_obs = self.phase_wrapper._sync_from_env()

        observations = {
            agent: (phase_obs[agent] if agent == BLUE_AGENT_NAME and agent in phase_obs else obs)
            for agent, obs in graph_obs.items()
            if "blue" in agent
        }
        self.agents = list(observations.keys())
        return observations, graph_info

    def step(
        self,
        actions: dict[str, Action] = {},
        messages: dict[str, Any] = None,
        **kwargs,
    ):
        graph_obs, rewards, terminated, truncated, info = self.graph_wrapper.step(actions)
        phase_obs = self.phase_wrapper._sync_from_env()

        observations = {
            agent: (phase_obs[agent] if agent == BLUE_AGENT_NAME and agent in phase_obs else obs)
            for agent, obs in graph_obs.items()
            if "blue" in agent
        }
        self.agents = [
            agent
            for agent, done in terminated.items()
            if "blue" in agent and not (done or truncated.get(agent, False))
        ]
        return observations, rewards, terminated, truncated, info
