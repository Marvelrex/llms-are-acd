# UID: LMT-ACT-20251210-X7Q3
"""
Custom high-level red actions for the LMT Pass-the-Hash scenario.
These are **abstracted steps**, not low-level Windows commands.
They just manipulate CybORG State/Host info so reward calculators
(and your analysis) can see progress.

Steps encoded (matching your scenario reports):
1. Disable AV / Live monitoring and create tools dir
2. Deploy reverse shell (conceptual new session on LMT-IT-DC01)
3. Download Mimikatz
4. Execute Mimikatz & dump hash
5. Create agent.bat launcher
6. Pass-the-Hash to get LMT\\Administrator on LMTDC01
7. Access restricted remote directory (read notes.txt)
8. Remove tools / cleanup
9. Re-enable live monitoring
"""
from typing import Dict, Any, Optional

from CybORG.Shared import Observation
from CybORG.Simulator.Actions.Action import Action
from CybORG.Simulator.State import State


class LMTBaseAction(Action):
    """Base helper class for LMT scenario actions."""

    # Hostnames that MUST match your LMTScenarioGenerator
    HOST_JUMP = "Jump-Server"
    HOST_IT_DC = "LMT-IT-DC01"
    HOST_LMT_DC = "LMTDC01"

    # The "valuable resource" path (for documentation / flags only)
    VALUABLE_SHARE_PATH = r"\\192.168.58.13\AllShares\Domain_Admin_Reserved_Area\notes.txt;"

    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_red_agent_name(state: State) -> Optional[str]:
        """Best-effort: find the red agent name in the state."""
        for name in state.sessions.keys():
            if name.lower().startswith("red"):
                return name
        return None

    @staticmethod
    def _get_flags_dict(host) -> Dict[str, Any]:
        """
        Get (or create) a dict used to store scenario-specific flags
        on a host. We keep them under host.info["lmt_flags"].
        """
        if host.info is None:
            host.info = {}
        if "lmt_flags" not in host.info:
            host.info["lmt_flags"] = {}
        return host.info["lmt_flags"]

    @staticmethod
    def _success_obs(note: str = "") -> Observation:
        obs = Observation(success=True)
        if note:
            obs.data["lmt_note"] = note
        return obs

    @staticmethod
    def _fail_obs(note: str = "") -> Observation:
        obs = Observation(success=False)
        if note:
            obs.data["lmt_note"] = note
        return obs


class DisableMonitoringAndPrepareToolsDirectory(LMTBaseAction):
    """
    Represents:
    - Creating C:\\Users\\Administrator\\Downloads\\tools and tools\\mim
    - Disabling live monitoring / AV in that directory
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        flags["tools_dir_created"] = True
        flags["monitoring_disabled"] = True
        flags["monitoring_enabled"] = False
        # (You could also store exact paths if you want later)
        flags["tools_dir_path"] = r"C:\\Users\\Administrator\\Downloads\\tools"
        flags["mim_dir_path"] = r"C:\\Users\\Administrator\\Downloads\\tools\\mim"
        return self._success_obs("Tools directory created and monitoring disabled.")


class DeployReverseShellAgent(LMTBaseAction):
    """
    Represents:
    - Running Sandcat / reverse shell on LMT-IT-DC01
    - Conceptually, you gain a second agent on that host.
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        # Optional dependency on step 1:
        if not flags.get("tools_dir_created", False):
            return self._fail_obs("Tools directory not prepared yet.")

        # In this simplified version, we don't create a new Session object in `state`.
        # We just mark that a reverse shell has been deployed.
        flags["reverse_shell_deployed"] = True
        return self._success_obs("Reverse shell conceptually deployed on LMT-IT-DC01.")


class DownloadMimikatzTool(LMTBaseAction):
    """
    Represents:
    - Download + extract Mimikatz into tools\\mim\\mim.exe
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        # Optional dependency on reverse shell
        if not flags.get("reverse_shell_deployed", False):
            return self._fail_obs("Reverse shell not yet deployed.")

        # We *could* actually add a File to the host, but for now we just flag it.
        flags["mimikatz_downloaded"] = True
        flags["mimikatz_path"] = r"C:\\Users\\Administrator\\Downloads\\tools\\mim\\mim.exe"
        return self._success_obs("Mimikatz downloaded to tools\\mim\\mim.exe.")


class ExecuteMimikatzDump(LMTBaseAction):
    """
    Represents:
    - Running mim.exe and extracting the NTLM hash for LMT\\Administrator
    """

    LMT_ADMIN_HASH = "cb8a428385459087a76793010d60f5dc"

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        if not flags.get("mimikatz_downloaded", False):
            return self._fail_obs("Mimikatz not downloaded yet.")

        flags["lmt_admin_hash_dumped"] = True
        flags["lmt_admin_hash_value"] = self.LMT_ADMIN_HASH
        return self._success_obs("LMT\\Administrator hash dumped with Mimikatz.")


class CreateAgentBat(LMTBaseAction):
    """
    Represents:
    - Writing agent.bat that contains the Sandcat deployment command to spawn the A113 agent running as LMT\\Administrator.
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        if not flags.get("mimikatz_downloaded", False):
            # Not strictly required in real life, but makes sequence logical
            return self._fail_obs("Mimikatz setup not completed yet.")

        flags["agent_bat_created"] = True
        flags["agent_bat_path"] = r"C:\\Users\\Administrator\\Downloads\\tools\\agent.bat"
        flags["agent_bat_target_group"] = "A113"
        return self._success_obs("agent.bat created for A113 deployment.")


class PassTheHashAttack(LMTBaseAction):
    """
    Represents:
    - Running Mimikatz sekurlsa::pth with the dumped hash
    - Gaining LMT\\Administrator context on LMTDC01 (recorded via flags)
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts or self.HOST_LMT_DC not in state.hosts:
            return self._fail_obs("Required hosts missing (IT DC or LMT DC).")

        it_host = state.hosts[self.HOST_IT_DC]
        it_flags = self._get_flags_dict(it_host)
        if not (it_flags.get("lmt_admin_hash_dumped", False) and it_flags.get("agent_bat_created", False)):
            return self._fail_obs("Hash not dumped or agent.bat not created; cannot PTH.")

        # Instead of adding a real session to the State, we mark on both hosts
        # that the PTH succeeded and LMT\\Administrator is active.
        it_flags["pth_executed"] = True
        lmt_dc_host = state.hosts[self.HOST_LMT_DC]
        dc_flags = self._get_flags_dict(lmt_dc_host)
        dc_flags["lmt_admin_session"] = True
        return self._success_obs("Pass-the-Hash succeeded; LMT\\Administrator context on LMTDC01.")


class AccessRestrictedRemoteDirectory(LMTBaseAction):
    """
    Represents:
    - Using the LMT\\Administrator context to read the protected notes.txt file from the domain admin reserved share.
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_LMT_DC not in state.hosts:
            return self._fail_obs("LMT domain controller not present in state.")

        lmt_dc_host = state.hosts[self.HOST_LMT_DC]
        dc_flags = self._get_flags_dict(lmt_dc_host)
        # Ensure we actually have LMT/Admin context (from step 6)
        if not dc_flags.get("lmt_admin_session", False):
            return self._fail_obs("No LMT\\Administrator session on LMTDC01 yet.")

        # Mark that the notes have been read
        dc_flags["notes_read"] = True
        dc_flags["notes_path"] = self.VALUABLE_SHARE_PATH
        dc_flags["notes_content"] = "You have entered a domain admin reserved area"
        return self._success_obs("Restricted notes.txt successfully read from the share.")


class RemoveToolsDirectory(LMTBaseAction):
    """
    Represents:
    - Deleting tools directory and its contents (mim.exe, agent.bat, etc.)
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        # Mark everything as removed
        flags["tools_removed"] = True
        flags["mimikatz_downloaded"] = False
        flags["agent_bat_created"] = False
        return self._success_obs("Tools directory and contents marked as removed.")


class ReenableLiveMonitoring(LMTBaseAction):
    """
    Represents:
    - Re-enabling live monitoring / AV for the tools directory
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)
        flags["monitoring_disabled"] = False
        flags["monitoring_enabled"] = True
        return self._success_obs("Live monitoring re-enabled on tools directory.")


__all__ = [
    "LMTBaseAction",
    "DisableMonitoringAndPrepareToolsDirectory",
    "DeployReverseShellAgent",
    "DownloadMimikatzTool",
    "ExecuteMimikatzDump",
    "CreateAgentBat",
    "PassTheHashAttack",
    "AccessRestrictedRemoteDirectory",
    "RemoveToolsDirectory",
    "ReenableLiveMonitoring",
]
