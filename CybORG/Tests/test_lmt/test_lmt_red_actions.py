from pathlib import Path
import sys

# Ensure repo root is importable when running tests without an editable install.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from CybORG import CybORG
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import (
    AccessRestrictedRemoteDirectory,
    DeployReverseShellAgent,
    DisableMonitoringAndPrepareToolsDirectory,
    ExecuteMimikatzDump,
    PassTheHashAttack,
)
from CybORG.Simulator.Scenarios import LMTScenarioGenerator


AGENT_NAME = "red_agent_0"
IT_DC_HOSTNAME = "LMT-IT-DC01"
LMT_DC_HOSTNAME = "LMTDC01"


def _create_lmt_cyborg(seed: int = 123, steps: int = 20):
    cyborg = CybORG(scenario_generator=LMTScenarioGenerator(steps=steps), seed=seed)
    cyborg.reset(agent=AGENT_NAME)
    return cyborg


def _step_action(cyborg, action):
    result = cyborg.step(agent=AGENT_NAME, action=action)
    assert result.observation["success"] != TernaryEnum.IN_PROGRESS
    return result


def test_lmt_red_action_chain_success():
    cyborg = _create_lmt_cyborg()

    result = _step_action(cyborg, DisableMonitoringAndPrepareToolsDirectory())
    assert result.observation["success"] == TernaryEnum.TRUE
    it_flags = cyborg.environment_controller.state.hosts[IT_DC_HOSTNAME].info["lmt_flags"]
    assert it_flags["tools_dir_created"] is True
    assert it_flags["monitoring_disabled"] is True
    assert it_flags["monitoring_enabled"] is False

    result = _step_action(cyborg, DeployReverseShellAgent())
    assert result.observation["success"] == TernaryEnum.TRUE
    it_flags = cyborg.environment_controller.state.hosts[IT_DC_HOSTNAME].info["lmt_flags"]
    assert it_flags["reverse_shell_deployed"] is True

    result = _step_action(cyborg, ExecuteMimikatzDump())
    assert result.observation["success"] == TernaryEnum.TRUE
    it_flags = cyborg.environment_controller.state.hosts[IT_DC_HOSTNAME].info["lmt_flags"]
    assert it_flags["lmt_admin_hash_dumped"] is True
    assert it_flags["lmt_admin_hash_value"]

    result = _step_action(cyborg, PassTheHashAttack())
    assert result.observation["success"] == TernaryEnum.TRUE
    it_flags = cyborg.environment_controller.state.hosts[IT_DC_HOSTNAME].info["lmt_flags"]
    dc_flags = cyborg.environment_controller.state.hosts[LMT_DC_HOSTNAME].info["lmt_flags"]
    assert it_flags["pth_executed"] is True
    assert dc_flags["lmt_admin_session"] is True

    result = _step_action(cyborg, AccessRestrictedRemoteDirectory())
    assert result.observation["success"] == TernaryEnum.TRUE
    dc_flags = cyborg.environment_controller.state.hosts[LMT_DC_HOSTNAME].info["lmt_flags"]
    assert dc_flags["notes_read"] is True
    assert "notes_path" in dc_flags
    assert "notes_content" in dc_flags
    assert cyborg.environment_controller.done is True


def test_lmt_red_action_requires_reverse_shell():
    cyborg = _create_lmt_cyborg()

    result = _step_action(cyborg, ExecuteMimikatzDump())
    assert result.observation["success"] == TernaryEnum.FALSE
    it_flags = cyborg.environment_controller.state.hosts[IT_DC_HOSTNAME].info["lmt_flags"]
    assert "lmt_admin_hash_dumped" not in it_flags
