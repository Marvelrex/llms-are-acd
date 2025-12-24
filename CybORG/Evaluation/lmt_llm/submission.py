from __future__ import annotations

from typing import Any

import wandb

from CybORG import CybORG
from CybORG.Agents import BaseAgent
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
from CybORG.Simulator.Actions import Action


class Submission:
    # Submission name
    NAME: str = SUB_NAME

    # Name of your team
    TEAM: str = SUB_TEAM

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = SUB_TECHNIQUE

    AGENTS: dict[str, BaseAgent] = {
        BLUE_AGENT_NAME: DefenderAgent(BLUE_AGENT_NAME, LLMDefenderPolicy, [])
    }

    @classmethod
    def wrap(cls, env: CybORG):
        if getattr(wandb, "run", None) is None:
            wandb.init(mode="disabled")
        return LMTPhaseWrapper(env)


class LMTPhaseWrapper(BaseWrapper):
    def action_space(self, agent_name: str):
        return self.get_action_space(agent_name)

    def reset(self, *args, **kwargs):
        self.env.reset(*args, **kwargs)
        self.agents = [a for a in self.env.agents if "blue" in a]
        observations = {a: self._decorate_obs(self.env.get_observation(a)) for a in self.agents}
        return observations, {}

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
