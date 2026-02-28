#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Keep smoke tests isolated from real benchmark logs/graphs by default.
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/smoke}"
RUN_ID="${RUN_ID:-smoke_$(date +%Y%m%d_%H%M%S)}"

MAX_EPS="${MAX_EPS:-1}"
SEED="${SEED:-123}"
MODEL_CONFIG="${MODEL_CONFIG:-CybORG/Agents/LLMAgents/config/model/dummy.yml}"

# Make the evaluation episode short so this finishes quickly.
CAGE4_EPISODE_LENGTH="${CAGE4_EPISODE_LENGTH:-25}"

export LOG_ROOT RUN_ID MAX_EPS SEED MODEL_CONFIG CAGE4_EPISODE_LENGTH

bash "${REPO_ROOT}/scripts/run_1_action_graph_4_rl_vs_red.sh"
