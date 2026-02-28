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
BENCH_ROOT="${BENCH_ROOT:-${LOG_ROOT}/benchmark_llm_lmt_5_vs_fsred/model_${MODEL_TAG}}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${BENCH_ROOT}/run_${RUN_ID}}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

cd "${REPO_ROOT}/cage-challenge-4"

WANDB_MODE=disabled \
MAX_EPS="${MAX_EPS}" \
SEED="${SEED}" \
MODEL_CONFIG="${MODEL_CONFIG}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
"${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

from CybORG.Evaluation import evaluation as ev

max_eps = int(os.environ["MAX_EPS"])
seed = int(os.environ["SEED"])
model_config = Path(os.environ["MODEL_CONFIG"]).resolve()
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()

os.environ["CAGE4_MODEL_CONFIG"] = str(model_config)

from CybORG.Agents.LLMAgents.llm_agent import DefenderAgent
from CybORG.Agents.LLMAgents.llm_policy import LLMDefenderPolicy

sys.path.insert(0, str(Path("CybORG/Evaluation/lmt_llm").resolve()))
import submission as sub  # type: ignore

sub.Submission.AGENTS = {
    f"blue_agent_{i}": DefenderAgent(f"blue_agent_{i}", LLMDefenderPolicy, [])
    for i in range(5)
}

ev.rmkdir(str(output_dir) + "/")
ev.run_evaluation(sub.Submission, log_path=str(output_dir), max_eps=max_eps, seed=seed)
PY
