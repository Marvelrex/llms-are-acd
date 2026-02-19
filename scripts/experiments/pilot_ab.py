from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any


# Ensure repo root is on sys.path when running as a script file.
# When invoked as `python scripts/experiments/pilot_ab.py`, Python sets sys.path[0]
# to the script directory (not the repo root), so `import CybORG` would fail.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

BLUE_AGENT_NAME = "blue_agent_0"


def _sanitize_path_component(value: str) -> str:
    # Conservative: keep filenames portable and readable.
    return (
        str(value)
        .strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def _run_cmd(argv: list[str], *, cwd: Path | None = None) -> tuple[int, str, str]:
    p = subprocess.run(
        argv,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        capture_output=True,
    )
    return int(p.returncode), str(p.stdout), str(p.stderr)


def _git_sha(repo_root: Path) -> str:
    rc, out, _err = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    return out.strip() if rc == 0 else "UNKNOWN"


def _write_git_txt(path: Path, repo_root: Path) -> None:
    rc1, out1, err1 = _run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_root)
    rc2, out2, err2 = _run_cmd(["git", "status", "--porcelain=v1"], cwd=repo_root)
    lines: list[str] = []
    lines.append("$ git rev-parse HEAD")
    lines.append(out1.strip() if rc1 == 0 else f"ERROR (rc={rc1}): {err1.strip()}")
    lines.append("")
    lines.append("$ git status --porcelain=v1")
    if rc2 == 0:
        lines.append(out2.rstrip("\n"))
    else:
        lines.append(f"ERROR (rc={rc2}): {err2.strip()}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
    except Exception:
        pass


@dataclass(frozen=True)
class ArmResult:
    arm: str
    episode_returns: list[float]
    episode_lengths: list[int]
    reward_mean: float
    reward_stdev: float


def _resolve_dummy_model_config(repo_root: Path) -> Path:
    return repo_root / "CybORG" / "Agents" / "LLMAgents" / "config" / "model" / "dummy.yml"


def _resolve_api_model_config(repo_root: Path) -> Path:
    return (
        repo_root
        / "CybORG"
        / "Agents"
        / "LLMAgents"
        / "config"
        / "model"
        / "gpt-4.1-mini-openai.yml"
    )


def _resolve_openrouter_model_config(repo_root: Path) -> Path:
    # Uses the OpenRouter-compatible backend (see DeepSeekBackend in this repo).
    return repo_root / "CybORG" / "Agents" / "LLMAgents" / "config" / "model" / "gpt-4.1-mini.yml"


def _resolve_llm_identity() -> dict[str, str | None]:
    cfg_path = os.environ.get("CAGE4_MODEL_CONFIG")
    if not cfg_path:
        return {"backend": None, "model_name": None, "model_config": None}
    try:
        from CybORG.Agents.LLMAgents.llm_adapter.config_loader import ConfigLoader

        cfg = ConfigLoader.load_model_configuration(cfg_path)
    except Exception:
        cfg = None

    backend = None
    model_name = None
    if isinstance(cfg, dict):
        backend = str(cfg.get("backend") or "").strip().lower() or None
        model_name = str(cfg.get("model_name") or "").strip() or None

    model_override = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL")
    if backend in {"openai", "new-openai"}:
        if model_override:
            model_name = str(model_override)
        elif not model_name:
            model_name = "gpt-4.1-mini"

    if backend == "dummy" and not model_name:
        model_name = "dummy"

    return {
        "backend": backend,
        "model_name": model_name,
        "model_config": str(Path(cfg_path).resolve()),
    }


def _configure_global_env(*, episodes: int, max_steps: int, llm_mode: str, repo_root: Path) -> None:
    # These are read at import-time by some LLM modules (progress bars, etc).
    os.environ["MAX_EPS"] = str(int(episodes))
    os.environ["CAGE4_EPISODE_LENGTH"] = str(int(max_steps))

    # Ensure wandb never blocks evaluation runs.
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DISABLED", "true")

    if llm_mode == "stub":
        os.environ["CAGE4_MODEL_CONFIG"] = str(_resolve_dummy_model_config(repo_root).resolve())
    elif llm_mode == "api":
        # Default to GPT-5-mini for API-backed runs.
        model_name = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL")
        if not model_name:
            model_name = "gpt-4.1-mini"
            os.environ["OPENAI_MODEL"] = model_name
        else:
            # Normalize on OPENAI_MODEL as the single source of truth.
            os.environ.setdefault("OPENAI_MODEL", str(model_name))

        os.environ["CAGE4_MODEL_CONFIG"] = str(_resolve_api_model_config(repo_root).resolve())
    elif llm_mode == "openrouter":
        os.environ["CAGE4_MODEL_CONFIG"] = str(_resolve_openrouter_model_config(repo_root).resolve())
        # Fail fast with a clear message if the OpenRouter key is missing.
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY is required for --llm-mode openrouter")


def _configure_arm_env(*, arm_log_dir: Path) -> None:
    arm_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CAGE4_LLM_LOG_ROOT"] = str(arm_log_dir.resolve())
    os.environ["OUTPUT_DIR"] = str(arm_log_dir.resolve())
    os.environ["PYG_HOME"] = str((arm_log_dir / ".pyg_cache").resolve())
    os.environ.setdefault("WANDB_DIR", str((arm_log_dir / "wandb").resolve()))


def _actiongraph_cache_paths(
    *,
    cache_root: Path,
    llm_identity: dict[str, str | None],
) -> tuple[Path, Path, Path]:
    backend = llm_identity.get("backend") or "unknown"
    model_name = llm_identity.get("model_name") or "unknown"
    cache_key = f"{_sanitize_path_component(backend)}__{_sanitize_path_component(model_name)}"
    cache_dir = Path(cache_root) / cache_key
    return cache_dir, cache_dir / "graph_scores.json", cache_dir / "meta.json"


def _update_actiongraph_cache(
    *,
    arm_log_dir: Path,
    actiongraph_cache_dir: Path,
    llm_identity: dict[str, str | None],
    episode: int | None = None,
) -> dict[str, Any]:
    """Copy the latest graph_scores.json from the run dir to the shared cache.

    Called after every episode so that subsequent runs (or crash-resumed runs)
    start from the most recently learned graph.
    """
    cache_dir, cache_graph, cache_meta = _actiongraph_cache_paths(
        cache_root=actiongraph_cache_dir,
        llm_identity=llm_identity,
    )
    run_graph = arm_log_dir / "graph_scores.json"
    info: dict[str, Any] = {
        "cache_dir": str(cache_dir),
        "cache_graph_path": str(cache_graph),
        "updated": False,
        "reason": None,
        "episode": episode,
    }
    if not run_graph.is_file():
        info["reason"] = "missing_run_graph"
        return info
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_graph, cache_graph)
    cache_meta.write_text(
        json.dumps(
            {
                "backend": llm_identity.get("backend"),
                "model_name": llm_identity.get("model_name"),
                "model_config": llm_identity.get("model_config"),
                "updated_at": datetime.now().isoformat(),
                "episode": episode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    info["updated"] = True
    return info


def _build_arm_agents(
    *,
    arm: str,
    llm_impl: str,
    llm_mode: str,
    arm_log_dir: Path,
    rl_weights_dir: Path,
    actiongraph_reuse: bool,
    actiongraph_cache_dir: Path | None,
    llm_identity: dict[str, str | None],
) -> tuple[str, dict[str, Any]]:
    """
    Returns (blue_agent_name, agents_dict).

    Agents dict always contains 5 blue agents: blue_agent_0..blue_agent_4,
    where blue_agent_0 is the LLM agent (unless llm_mode == "none").
    """
    _configure_arm_env(arm_log_dir=arm_log_dir)

    from CybORG.Agents.CybermonicAgents.cage4 import load as load_cybermonic_agent

    blue_agent_name = BLUE_AGENT_NAME

    # RL agents: always identical across arms.
    agents: dict[str, Any] = {}
    for i in range(5):
        weight_path = rl_weights_dir / f"gnn_ppo-{i}.pt"
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing RL checkpoint: {weight_path}")
        agents[f"blue_agent_{i}"] = load_cybermonic_agent(str(weight_path))

    if llm_mode == "none":
        # Keep all five PPO agents (debugging / baseline mode).
        return blue_agent_name, agents

    if llm_impl == "actiongraph":
        from CybORG.Agents.LLMAgents.llm_adapter.action_graph import build_cage4_turn_graph
        from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
        from CybORG.Agents.LLMAgents.llm_adapter.action_graph import ActionGraph

        g = None
        resume_info: dict[str, Any] = {
            "enabled": bool(actiongraph_reuse),
            "loaded": False,
            "reason": None,
            "cache_graph_path": None,
        }

        if actiongraph_reuse and actiongraph_cache_dir is not None:
            cache_dir, cache_graph, cache_meta = _actiongraph_cache_paths(
                cache_root=actiongraph_cache_dir,
                llm_identity=llm_identity,
            )
            resume_info["cache_graph_path"] = str(cache_graph)

            if cache_graph.is_file():
                if not cache_meta.is_file():
                    resume_info["reason"] = "missing_cache_meta"
                else:
                    try:
                        meta = json.loads(cache_meta.read_text(encoding="utf-8"))
                    except Exception:
                        meta = None
                    if not isinstance(meta, dict):
                        resume_info["reason"] = "invalid_cache_meta"
                    elif str(meta.get("backend")) != str(llm_identity.get("backend")) or str(meta.get("model_name")) != str(llm_identity.get("model_name")):
                        resume_info["reason"] = "backend_or_model_mismatch"
                    else:
                        try:
                            g = ActionGraph.load(cache_graph)
                            resume_info["loaded"] = True
                        except Exception:
                            resume_info["reason"] = "failed_to_load_cache_graph"
            else:
                resume_info["reason"] = "cache_miss"

        if g is None:
            g = build_cage4_turn_graph(include_terminal=True)

        agent = SelfEvolvingGraphAgent(
            blue_agent_name,
            graph=g,
            log_dir=arm_log_dir,
            persist_path=arm_log_dir / "graph_scores.json",
            snapshot_every=1,
        )
        # Expose resume info to the runner for summary.json (debug aid).
        setattr(agent, "_resume_info", resume_info)
        # Emit resume info for debugging (captured in arm stdout.log).
        print(f"[pilot_ab] actiongraph_resume={resume_info}")
        agents[blue_agent_name] = agent
        return blue_agent_name, agents

    if llm_impl == "lmt":
        from CybORG.Agents.LLMAgents.llm_agent import DefenderAgent
        from CybORG.Agents.LLMAgents.llm_policy import LLMDefenderPolicy

        agents[blue_agent_name] = DefenderAgent(blue_agent_name, LLMDefenderPolicy, [])
        return blue_agent_name, agents

    raise ValueError(f"[{arm}] Unknown llm_impl: {llm_impl}")


def _run_arm(
    *,
    arm: str,
    llm_impl: str,
    llm_mode: str,
    scenario: str,
    episodes: int,
    max_steps: int,
    seed: int,
    arm_dir: Path,
    rl_weights_dir: Path,
    actiongraph_reuse: bool,
    actiongraph_cache_dir: Path | None,
    llm_identity: dict[str, str | None],
) -> ArmResult:
    arm_dir.mkdir(parents=True, exist_ok=True)
    arm_log_dir = arm_dir / "logs"
    arm_log_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = arm_dir / "stdout.log"
    stderr_path = arm_dir / "stderr.log"

    resume_info: dict[str, Any] | None = None

    with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_f, contextlib.redirect_stdout(stdout_f), contextlib.redirect_stderr(stderr_f):
        _seed_everything(seed)

        from CybORG import CybORG, CYBORG_VERSION
        from CybORG.Agents import EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent
        from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
        from CybORG.Agents.Wrappers import BaseWrapper
        from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (
            BlueFixedActionWrapper,
            EMPTY_MESSAGE,
        )
        from CybORG.Agents.Wrappers.CybermonicWrappers.graph_wrapper import GraphWrapper
        from CybORG.Shared.MetricsCallback import MetricsCallback
        from CybORG.Simulator.Actions import Action
        from CybORG.Agents.LLMAgents.llm_adapter.action_graph import _camel_to_snake

        if scenario != "Scenario4":
            raise ValueError(f"Unsupported scenario for pilot_ab: {scenario!r} (expected 'Scenario4')")

        blue_agent_name, agents = _build_arm_agents(
            arm=arm,
            llm_impl=llm_impl,
            llm_mode=llm_mode,
            arm_log_dir=arm_log_dir,
            rl_weights_dir=rl_weights_dir,
            actiongraph_reuse=actiongraph_reuse,
            actiongraph_cache_dir=actiongraph_cache_dir,
            llm_identity=llm_identity,
        )
        if llm_impl == "actiongraph":
            try:
                resume_info = getattr(agents.get(blue_agent_name), "_resume_info", None)
            except Exception:
                resume_info = None

        class PhaseWrapper(BaseWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.metrics_callback = MetricsCallback()
                self.traces: dict[str, list[str]] = {}
                self.rewards: dict[str, float] = {}

            def _decorate_obs(self, agent: str, observation: dict) -> dict:
                state = self.env.get_attr("environment_controller").state
                observation["phase"] = state.mission_phase
                if "message" not in observation:
                    observation["message"] = [EMPTY_MESSAGE for _ in range(4)]
                return observation

            def _sync_from_env(self) -> tuple[dict[str, Any], dict[str, dict]]:
                self.agents = [a for a in self.env.agents if "blue" in a]
                self.traces = {a: [] for a in self.agents}
                self.rewards = {a: 0.0 for a in self.agents}
                observations = {
                    a: self._decorate_obs(a, self.env.get_observation(a)) for a in self.agents
                }
                info: dict[str, dict] = {}
                self.metrics_callback.on_reset(self.env)
                return observations, info

            def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, dict]]:
                self.env.reset(*args, **kwargs)
                return self._sync_from_env()

            def _process_step_results(
                self,
                actions: dict[str, Action],
                obs: dict[str, Any],
                rews: dict[str, Any],
                terminated_env: dict[str, bool],
                truncated_env: dict[str, bool],
                info: dict[str, Any],
            ) -> tuple[
                dict[str, Any],
                dict[str, float],
                dict[str, bool],
                dict[str, bool],
                dict[str, dict],
            ]:
                # Track executed actions for traces (use actions actually submitted)
                for agent_name, action_obj in actions.items():
                    if "blue" not in agent_name:
                        continue
                    agent_obj = agents.get(agent_name)
                    if agent_obj is not None and hasattr(agent_obj, "set_actions"):
                        agent_obj.set_actions(self.env.actions(agent_name), self.env.action_labels(agent_name))
                    if agent_obj is not None and hasattr(agent_obj, "note_transition"):
                        agent_obj.note_transition(agent_name, action_obj)
                    else:
                        node_id = f"defender_{_camel_to_snake(action_obj.__class__.__name__)}"
                        self.traces.setdefault(agent_name, []).append(node_id)

                # Track attacker actions from env controller for pairing
                executed = self.env.unwrapped.environment_controller.action
                for agent_name, act_list in executed.items():
                    if "red" not in agent_name:
                        continue
                    for act in act_list:
                        for b in list(self.traces.keys()):
                            agent_obj = agents.get(b)
                            if agent_obj is not None and hasattr(agent_obj, "note_transition"):
                                agent_obj.note_transition(agent_name, act)
                            else:
                                node_id = f"attacker_{_camel_to_snake(act.__class__.__name__)}"
                                self.traces.setdefault(b, []).append(node_id)

                self.agents = [
                    agent
                    for agent, done in terminated_env.items()
                    if "blue" in agent and not done
                ]

                observations = {
                    agent: self._decorate_obs(agent, o)
                    for agent, o in obs.items()
                    if "blue" in agent
                }

                rewards: dict[str, float] = {}
                for agent, reward in rews.items():
                    if "blue" not in agent:
                        continue
                    if isinstance(reward, dict):
                        rewards[agent] = float(sum(reward.values()))
                    else:
                        rewards[agent] = float(reward)

                for agent, r in rewards.items():
                    self.rewards[agent] = float(self.rewards.get(agent, 0.0) + r)
                    agent_obj = agents.get(agent)
                    if agent == blue_agent_name and agent_obj is not None and hasattr(agent_obj, "_log_step"):
                        agent_obj._log_step({"event": "step_reward", "agent": agent, "reward": r})

                self.metrics_callback.on_step(observations, actions, self.env)

                terminated = {agent: done for agent, done in terminated_env.items() if "blue" in agent}
                truncated = {agent: done for agent, done in truncated_env.items() if "blue" in agent}

                # Finalize traces when an agent finishes
                for agent, done in terminated.items():
                    if done or truncated.get(agent, False):
                        agent_obj = agents.get(agent)
                        if agent_obj is not None and hasattr(agent_obj, "finalize_episode"):
                            agent_obj.finalize_episode(self.rewards.get(agent, 0.0))
                        self.traces[agent] = []
                        self.rewards[agent] = 0.0

                # If the scenario is globally done (e.g., max steps reached) ensure we flush traces.
                if self.env.unwrapped.environment_controller.done:
                    for agent, agent_obj in agents.items():
                        if "blue" not in agent:
                            continue
                        if hasattr(agent_obj, "finalize_episode"):
                            agent_obj.finalize_episode(self.rewards.get(agent, 0.0))
                        self.traces[agent] = []
                        self.rewards[agent] = 0.0

                return observations, rewards, terminated, truncated, {}

            def action_space(self, agent_name: str):
                if hasattr(self.env, "action_space"):
                    return self.env.action_space(agent_name)
                return self.env.get_action_space(agent_name)

        class HybridPhaseGraphWrapper(BaseWrapper):
            """Hybrid wrapper: one BLUE LLM agent plus four PPO graph agents."""

            def __init__(self, env: CybORG):
                super().__init__(env)
                self.phase_wrapper = PhaseWrapper(BlueFixedActionWrapper(env))
                self.graph_wrapper = GraphWrapper(env)

            def reset(self, *args, **kwargs):
                graph_obs, graph_info = self.graph_wrapper.reset(*args, **kwargs)
                phase_obs, phase_info = self.phase_wrapper._sync_from_env()

                observations = {
                    agent: (phase_obs[agent] if agent == blue_agent_name and agent in phase_obs else obs)
                    for agent, obs in graph_obs.items()
                    if "blue" in agent
                }
                info_out = dict(graph_info)
                info_out.update(phase_info)
                self.agents = list(observations.keys())
                return observations, info_out

            def step(self, actions: dict[str, Action] = {}, messages: dict[str, Any] = None, **kwargs):
                graph_obs, rews, terminated_env, truncated_env, info = self.graph_wrapper.step(actions)
                raw_obs = {
                    agent: self.phase_wrapper.env.get_observation(agent)
                    for agent in self.phase_wrapper.env.agents
                    if "blue" in agent
                }
                phase_obs, rewards, terminated, truncated, phase_info = self.phase_wrapper._process_step_results(
                    actions,
                    raw_obs,
                    rews,
                    terminated_env,
                    truncated_env,
                    info,
                )

                observations = {
                    agent: (phase_obs[agent] if agent == blue_agent_name and agent in phase_obs else obs)
                    for agent, obs in graph_obs.items()
                    if "blue" in agent
                }
                if blue_agent_name in phase_obs and blue_agent_name not in observations:
                    observations[blue_agent_name] = phase_obs[blue_agent_name]

                info_out = dict(info)
                info_out.update(phase_info)
                self.agents = [
                    agent
                    for agent, done in terminated.items()
                    if "blue" in agent and not (done or truncated.get(agent, False))
                ]
                return observations, rewards, terminated, truncated, info_out

            def action_space(self, agent_name: str):
                if agent_name == blue_agent_name:
                    return self.phase_wrapper.action_space(agent_name)
                return None

        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=int(max_steps),
        )
        cyborg = CybORG(sg, "sim", seed=int(seed))
        wrapped = HybridPhaseGraphWrapper(cyborg)

        print(f"CybORG v{CYBORG_VERSION}, {scenario}")
        print(f"[pilot_ab] arm={arm} llm_impl={llm_impl} llm_mode={llm_mode}")
        print(f"[pilot_ab] weights_dir={rl_weights_dir}")

        run_start_wall = time.perf_counter()
        run_start_time = datetime.now().isoformat()

        episode_returns: list[float] = []
        episode_lengths: list[int] = []
        episode_reward_series_lengths: list[int] = []
        episode_wall_time_s: list[float] = []

        # RL agent logging: create a subfolder and per-agent JSONL files.
        rl_log_dir = arm_log_dir / "rl_agents"
        rl_log_dir.mkdir(parents=True, exist_ok=True)
        rl_agent_names = sorted(
            name for name in agents if name != BLUE_AGENT_NAME and name.startswith("blue_agent_")
        )
        rl_log_files: dict[str, Path] = {
            name: rl_log_dir / f"{name}.jsonl" for name in rl_agent_names
        }

        def _translate_rl_action(agent_name: str, action_id: int | None) -> dict[str, str | None]:
            """Translate an RL integer action ID to a human-readable action + target."""
            if action_id is None:
                return {"action_name": "Sleep", "target": None}
            try:
                cyborg_action = wrapped.graph_wrapper.action_translator(agent_name, action_id)
                action_name = type(cyborg_action).__name__
                # Extract target from the action object
                target = None
                if hasattr(cyborg_action, "hostname"):
                    target = str(cyborg_action.hostname)
                elif hasattr(cyborg_action, "subnet"):
                    target = str(getattr(cyborg_action, "subnet", ""))
                    from_subnet = getattr(cyborg_action, "from_subnet", None)
                    if from_subnet:
                        target = f"{from_subnet}->{target}"
                return {"action_name": action_name, "target": target}
            except Exception:
                return {"action_name": f"action_id={action_id}", "target": None}

        def _extract_obs_summary(agent_name: str, obs: object) -> dict[str, Any]:
            """Extract a loggable summary from RL observation tensors."""
            import torch as _torch
            summary: dict[str, Any] = {}
            if not isinstance(obs, (tuple, list)) or len(obs) < 2:
                return summary
            state_tensors, is_blocked = obs[0], obs[1]
            summary["is_blocked"] = bool(is_blocked)
            if not isinstance(state_tensors, (tuple, list)) or len(state_tensors) < 3:
                return summary
            # state_tensors = (x, ei, phase, srv, n_srv, usr, n_usr, edges, multi)
            x = state_tensors[0]   # node feature matrix N×d
            phase = state_tensors[2]  # 1×3 one-hot
            try:
                summary["n_nodes"] = int(x.size(0))
                summary["n_features"] = int(x.size(1))
                # Phase: argmax of one-hot vector
                summary["phase"] = int(phase.argmax().item())
                # Summarize node features: mean compromised/scanned from last few cols
                # The tabular features (compromised, scanned) are at specific positions
                # Just log basic stats
                summary["node_feat_mean"] = round(float(x.mean().item()), 4)
                summary["node_feat_nonzero"] = int((x != 0).sum().item())
            except Exception:
                pass
            return summary

        def _log_rl_step(
            agent_name: str,
            episode: int,
            step: int,
            obs: object,
            action: object,
            reward: float | None,
        ) -> None:
            log_path = rl_log_files.get(agent_name)
            if log_path is None:
                return
            is_blocked = False
            if isinstance(obs, (tuple, list)) and len(obs) >= 2:
                is_blocked = bool(obs[1])

            # Translate action ID to human-readable name + target
            action_info = _translate_rl_action(agent_name, action) if action is not None else {}

            # Extract observation summary (only on pre-step log where action is present)
            obs_summary = _extract_obs_summary(agent_name, obs) if action is not None else {}

            record = {
                "episode": int(episode),
                "step": int(step),
                "agent": str(agent_name),
                "is_blocked": bool(is_blocked),
                "action_id": int(action) if action is not None else None,
                "action_name": action_info.get("action_name"),
                "action_target": action_info.get("target"),
                "reward": float(reward) if reward is not None else None,
            }
            if obs_summary:
                record["obs"] = obs_summary

            try:
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

        for ep in range(int(episodes)):
            observations, _info = wrapped.reset()
            ep_start_wall = time.perf_counter()
            steps_taken = 0
            per_step_team = []

            for _t in range(int(max_steps)):
                actions = {
                    agent_name: agent.get_action(
                        observations[agent_name], wrapped.action_space(agent_name)
                    )
                    for agent_name, agent in agents.items()
                    if agent_name in wrapped.agents
                }

                # Capture pre-step observations for RL agents before stepping.
                pre_step_obs = {
                    rl_name: observations[rl_name]
                    for rl_name in rl_agent_names
                    if rl_name in observations
                }

                observations, rew, term, trunc, _info = wrapped.step(actions)
                done = {
                    agent: term.get(agent, False) or trunc.get(agent, False)
                    for agent in wrapped.agents
                }

                # Log RL agents: one record per step with action + reward.
                for rl_name in rl_agent_names:
                    if rl_name in actions or rl_name in rew:
                        _log_rl_step(
                            rl_name, ep, _t,
                            obs=pre_step_obs.get(rl_name),
                            action=actions.get(rl_name),
                            reward=float(rew[rl_name]) if rl_name in rew else None,
                        )

                if rew:
                    per_step_team.append(float(mean(rew.values())))
                else:
                    per_step_team.append(0.0)

                steps_taken += 1
                if done and all(done.values()):
                    break

            ep_return = float(sum(per_step_team))
            episode_returns.append(ep_return)
            episode_lengths.append(int(steps_taken))
            episode_reward_series_lengths.append(int(len(per_step_team)))
            episode_wall_time_s.append(float(time.perf_counter() - ep_start_wall))

            for agent in agents.values():
                if hasattr(agent, "end_episode"):
                    try:
                        agent.end_episode()
                    except Exception:
                        pass

            # Update actiongraph cache after every episode so the next run
            # (or a crash-resumed run) starts from the latest learned graph.
            if llm_impl == "actiongraph" and actiongraph_reuse and actiongraph_cache_dir is not None:
                _update_actiongraph_cache(
                    arm_log_dir=arm_log_dir,
                    actiongraph_cache_dir=actiongraph_cache_dir,
                    llm_identity=llm_identity,
                    episode=ep,
                )

            print(f"[pilot_ab] episode={ep} return={ep_return:.4f} steps={steps_taken}")

        reward_mean = float(mean(episode_returns)) if episode_returns else 0.0
        reward_stdev = float(stdev(episode_returns)) if len(episode_returns) > 1 else 0.0
        run_wall_time_s = float(time.perf_counter() - run_start_wall)
        run_end_time = datetime.now().isoformat()

    # Final cache update (also done per-episode above, but this ensures the
    # summary dict is populated for the run metadata).
    actiongraph_cache_update: dict[str, Any] | None = None
    if llm_impl == "actiongraph" and actiongraph_reuse and actiongraph_cache_dir is not None:
        actiongraph_cache_update = _update_actiongraph_cache(
            arm_log_dir=arm_log_dir,
            actiongraph_cache_dir=actiongraph_cache_dir,
            llm_identity=llm_identity,
            episode=int(episodes) - 1,
        )

    summary = {
        "arm": arm,
        "llm_impl": llm_impl,
        "llm_agent_name": BLUE_AGENT_NAME,
        "llm_identity": dict(llm_identity),
        "parameters": {
            "seed": int(seed),
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "scenario": scenario,
            "llm_mode": llm_mode,
            "rl_weights_dir": str(rl_weights_dir.resolve()),
        },
        "actiongraph_resume": resume_info,
        "actiongraph_cache_update": actiongraph_cache_update,
        "timing": {
            "start_time": run_start_time,
            "end_time": run_end_time,
            "wall_time_s": run_wall_time_s,
            "episode_wall_time_s": episode_wall_time_s,
        },
        "agents": {
            "impl_by_agent": {
                agent_name: f"{agent.__class__.__module__}.{agent.__class__.__name__}"
                for agent_name, agent in agents.items()
            },
        },
        "reward": {
            "episode_returns": episode_returns,
            "episode_lengths": episode_lengths,
            "episode_reward_series_lengths": episode_reward_series_lengths,
            "total_steps": int(sum(episode_lengths)),
            "mean": reward_mean,
            "stdev": reward_stdev,
        },
    }
    if llm_mode == "api":
        summary["model_name"] = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL")
    _write_json(arm_dir / "summary.json", summary)

    return ArmResult(
        arm=arm,
        episode_returns=episode_returns,
        episode_lengths=episode_lengths,
        reward_mean=reward_mean,
        reward_stdev=reward_stdev,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Pilot A/B: ActionGraph vs LMT (1 LLM + 4 RL).")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("outputs/pilot_ab"))
    ap.add_argument(
        "--arms",
        type=str,
        default="both",
        choices=["both", "actiongraph", "lmt"],
        help="Which arm(s) to run. 'actiongraph' runs only A; 'lmt' runs only B; 'both' runs A then B.",
    )
    ap.add_argument(
        "--rl-weights-dir",
        type=Path,
        default=(
            _REPO_ROOT / "CybORG" / "Evaluation" / "Cybermonics" / "weights"
        ),
        help="Directory containing gnn_ppo-{i}.pt checkpoints (i=0..4).",
    )
    ap.add_argument("--scenario", type=str, default=None)
    ap.add_argument("--llm-mode", type=str, default="stub", choices=["stub", "api", "openrouter", "none"])
    ap.add_argument("--notes", type=str, default=None)
    ap.add_argument(
        "--reuse-actiongraph",
        action="store_true",
        help="If set, resume/update the ActionGraph from a cache keyed by LLM backend+model.",
    )
    ap.add_argument(
        "--actiongraph-cache-dir",
        type=Path,
        default=None,
        help="Root directory for ActionGraph resume cache (default: <outdir>/actiongraph_cache).",
    )
    args = ap.parse_args()

    scenario_used = str(args.scenario) if args.scenario else "Scenario4"

    rl_weights_dir = Path(args.rl_weights_dir).resolve()
    if not rl_weights_dir.is_dir():
        raise FileNotFoundError(f"RL weights directory not found: {rl_weights_dir}")

    _configure_global_env(
        episodes=int(args.episodes),
        max_steps=int(args.max_steps),
        llm_mode=str(args.llm_mode),
        repo_root=_REPO_ROOT,
    )

    llm_identity = _resolve_llm_identity()
    actiongraph_cache_dir = (
        Path(args.actiongraph_cache_dir).resolve()
        if args.actiongraph_cache_dir is not None
        else (Path(args.outdir) / "actiongraph_cache").resolve()
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.outdir) / f"run_{timestamp}_seed{int(args.seed)}"
    run_root.mkdir(parents=True, exist_ok=False)

    git_sha = _git_sha(_REPO_ROOT)
    _write_git_txt(run_root / "git.txt", _REPO_ROOT)

    cmdline = " ".join([shlex.quote(sys.executable)] + [shlex.quote(a) for a in sys.argv])
    meta = {
        "timestamp": timestamp,
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "max_steps": int(args.max_steps),
        "scenario": scenario_used,
        "llm_mode": args.llm_mode,
        "arms": str(args.arms),
        "rl_weights_dir": str(Path(args.rl_weights_dir).resolve()),
        "llm_identity": llm_identity,
        "reuse_actiongraph": bool(args.reuse_actiongraph),
        "actiongraph_cache_dir": str(actiongraph_cache_dir) if args.reuse_actiongraph else None,
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "git_sha": git_sha,
        "notes": args.notes,
        "cmdline": cmdline,
    }
    _write_json(run_root / "meta.json", meta)

    a_dir = run_root / "A_actiongraph"
    b_dir = run_root / "B_lmt"

    run_a = str(args.arms) in {"both", "actiongraph"}
    run_b = str(args.arms) in {"both", "lmt"}

    a_res = None
    b_res = None

    if run_a:
        a_res = _run_arm(
            arm="A_actiongraph",
            llm_impl="actiongraph",
            llm_mode=str(args.llm_mode),
            scenario=scenario_used,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            arm_dir=a_dir,
            rl_weights_dir=rl_weights_dir,
            actiongraph_reuse=bool(args.reuse_actiongraph),
            actiongraph_cache_dir=actiongraph_cache_dir,
            llm_identity=llm_identity,
        )
    if run_b:
        b_res = _run_arm(
            arm="B_lmt",
            llm_impl="lmt",
            llm_mode=str(args.llm_mode),
            scenario=scenario_used,
            episodes=int(args.episodes),
            max_steps=int(args.max_steps),
            seed=int(args.seed),
            arm_dir=b_dir,
            rl_weights_dir=rl_weights_dir,
            actiongraph_reuse=False,
            actiongraph_cache_dir=actiongraph_cache_dir,
            llm_identity=llm_identity,
        )

    print(f"DONE. Results in: {run_root}")
    if a_res is not None and b_res is not None:
        delta = a_res.reward_mean - b_res.reward_mean
        print(
            f"A mean={a_res.reward_mean:.4f}  B mean={b_res.reward_mean:.4f}  (A-B)={delta:.4f}"
        )
    elif a_res is not None:
        print(f"A mean={a_res.reward_mean:.4f}")
    elif b_res is not None:
        print(f"B mean={b_res.reward_mean:.4f}")


if __name__ == "__main__":
    main()
