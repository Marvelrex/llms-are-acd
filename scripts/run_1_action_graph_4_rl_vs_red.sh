#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MAX_EPS="${MAX_EPS:-10}"
SEED="${SEED:-42}"
MODEL_CONFIG="${MODEL_CONFIG:-CybORG/Agents/LLMAgents/config/model/gpt-4.1-mini.yml}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
MODEL_TAG="${MODEL_TAG:-$(basename "${MODEL_CONFIG}")}"
MODEL_TAG="${MODEL_TAG%.*}"
BENCH_ROOT="${BENCH_ROOT:-${LOG_ROOT}/benchmark_action_graph_1llm_4rl_vs_fsred/model_${MODEL_TAG}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_ROOT}/run_${RUN_ID}}"
GRAPH_LOG_ROOT="${GRAPH_LOG_ROOT:-${OUTPUT_DIR}/action_graph}"
GRAPH_SHARED_DIR="${GRAPH_SHARED_DIR:-${BENCH_ROOT}/action_graph/shared}"
RL_WEIGHTS_DIR="${RL_WEIGHTS_DIR:-${REPO_ROOT}/cage-challenge-4/CybORG/Evaluation/Cybermonics/weights}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}/cage-challenge-4"

# Hybrid team: 1 SelfEvolvingGraphAgent (blue_agent_4) + 4 PPO agents (blue_agent_0..3) vs FiniteStateRedAgent.
WANDB_MODE=disabled \
MAX_EPS="${MAX_EPS}" \
SEED="${SEED}" \
MODEL_CONFIG="${MODEL_CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
GRAPH_LOG_ROOT="${GRAPH_LOG_ROOT}" \
GRAPH_SHARED_DIR="${GRAPH_SHARED_DIR}" \
RL_WEIGHTS_DIR="${RL_WEIGHTS_DIR}" \
CAGE4_USE_RL_AGENTS=1 \
"${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

from CybORG.Evaluation import evaluation as ev

max_eps = int(os.environ["MAX_EPS"])
seed = int(os.environ["SEED"])
model_config = Path(os.environ["MODEL_CONFIG"]).resolve()
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
graph_log_root = Path(os.environ["GRAPH_LOG_ROOT"]).resolve()
graph_shared_dir = Path(os.environ["GRAPH_SHARED_DIR"]).resolve()
rl_weights_dir = Path(os.environ["RL_WEIGHTS_DIR"]).resolve()

os.environ["CAGE4_MODEL_CONFIG"] = str(model_config)
os.environ["CAGE4_USE_RL_AGENTS"] = "1"
os.environ["CAGE4_RL_WEIGHTS_DIR"] = str(rl_weights_dir)

# Ensure submission uses our benchmark log folder (avoid writing under the repo by default).
os.environ["CAGE4_GRAPH_LOG_ROOT"] = str(graph_log_root)

from CybORG.Agents.CybermonicAgents.cage4 import load as load_cybermonic_agent
from CybORG.Agents.LLMAgents.llm_adapter.self_evolving_agent import SelfEvolvingGraphAgent
from CybORG.Agents.LLMAgents.llm_adapter.action_graph import ActionGraph, build_cage4_turn_graph

sys.path.insert(0, str(Path("CybORG/Evaluation/llamagym").resolve()))
import submission as sub  # type: ignore

graph_shared_dir.mkdir(parents=True, exist_ok=True)

llm_agent = "blue_agent_4"
persist_path = graph_shared_dir / f"graph_scores_{llm_agent}.json"

def load_or_init_graph(p: Path) -> ActionGraph:
    if p.exists():
        try:
            return ActionGraph.load(p)
        except Exception:
            pass
    return build_cage4_turn_graph()

sub.Submission.AGENTS = {
    f"blue_agent_{i}": load_cybermonic_agent(str(rl_weights_dir / f"gnn_ppo-{i}.pt"))
    for i in range(5)
}
sub.Submission.AGENTS[llm_agent] = SelfEvolvingGraphAgent(
    llm_agent,
    graph=load_or_init_graph(persist_path),
    log_dir=graph_log_root / llm_agent,
    persist_path=persist_path,
    snapshot_every=1,
)

ev.rmkdir(str(output_dir) + "/")
ev.run_evaluation(sub.Submission, log_path=str(output_dir), max_eps=max_eps, seed=seed)
PY
