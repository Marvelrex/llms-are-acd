# UID: LMT-ACT-20251210-X7Q3

"""
Custom high-level red actions for the LMT Pass-the-Hash scenario.

These are abstracted steps (not low-level Windows commands).
They manipulate CybORG State/Host info so your scenario can track progress.

Kept steps (mapped to CC4 semantics):
  1. Disable monitoring + prepare tools directory   (≈ Stealth prep / folded into stealth)
  2. Deploy reverse shell / foothold                (≈ ExploitRemoteService_cc4)
  3. Execute Mimikatz & dump hash                   (≈ PrivilegeEscalate)
  4. Pass-the-Hash to LMTDC01                       (≈ ExploitRemoteService_cc4 lateral move)
  5. Access restricted remote directory (notes.txt) (≈ Impact)

Removed entirely:
  - DownloadMimikatzTool
  - CreateAgentBat
  - RemoveToolsDirectory
  - ReenableLiveMonitoring
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

    # The “valuable resource” path (for documentation / flags only)
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


# 1) Disable Live Monitoring & AV, create tools dir
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

        # Optional: keep paths for debugging/storytelling
        flags["tools_dir_path"] = r"C:\Users\Administrator\Downloads\tools"
        flags["mim_dir_path"] = r"C:\Users\Administrator\Downloads\tools\mim"

        return self._success_obs("Tools directory created and monitoring disabled.")


# 2) Deploy reverse shell (conceptual foothold)
class DeployReverseShellAgent(LMTBaseAction):
    """
    Represents:
      - Running Sandcat / reverse shell on LMT-IT-DC01
      - Conceptually, you gain a session/foothold on that host.
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)

        # Optional dependency on step 1
        if not flags.get("tools_dir_created", False):
            return self._fail_obs("Tools directory not prepared yet.")

        # In this simplified version, we don't create a new Session object in `state`.
        flags["reverse_shell_deployed"] = True

        return self._success_obs("Reverse shell conceptually deployed on LMT-IT-DC01.")


# 3) Execute Mimikatz & dump LMT\Admin hash  (≈ PrivilegeEscalate)
class ExecuteMimikatzDump(LMTBaseAction):
    """
    Represents:
      - Extracting the NTLM hash for LMT\\Administrator (abstracted)
    """

    LMT_ADMIN_HASH = "cb8a428385459087a76793010d60f5dc"

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts:
            return self._fail_obs("IT DC host not present in state.")

        it_host = state.hosts[self.HOST_IT_DC]
        flags = self._get_flags_dict(it_host)

        if not flags.get("reverse_shell_deployed", False):
            return self._fail_obs("Reverse shell not yet deployed; cannot dump hash.")

        flags["lmt_admin_hash_dumped"] = True
        flags["lmt_admin_hash_value"] = self.LMT_ADMIN_HASH

        return self._success_obs("LMT\\Administrator hash dumped (abstracted).")


# 4) Pass-the-Hash: get LMT\\Administrator on LMTDC01 (≈ lateral ExploitRemoteService_cc4)
class PassTheHashAttack(LMTBaseAction):
    """
    Represents:
      - Using dumped hash to gain LMT\\Administrator context on LMTDC01 (recorded via flags)
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_IT_DC not in state.hosts or self.HOST_LMT_DC not in state.hosts:
            return self._fail_obs("Required hosts missing (IT DC or LMT DC).")

        it_host = state.hosts[self.HOST_IT_DC]
        it_flags = self._get_flags_dict(it_host)

        if not it_flags.get("lmt_admin_hash_dumped", False):
            return self._fail_obs("Hash not dumped; cannot Pass-the-Hash.")

        it_flags["pth_executed"] = True

        lmt_dc_host = state.hosts[self.HOST_LMT_DC]
        dc_flags = self._get_flags_dict(lmt_dc_host)

        dc_flags["lmt_admin_session"] = True

        return self._success_obs("Pass-the-Hash succeeded; LMT\\Administrator context on LMTDC01.")


# 5) Access restricted remote directory (≈ Impact)
class AccessRestrictedRemoteDirectory(LMTBaseAction):
    """
    Represents:
      - Using the LMT\\Administrator context to read the protected notes.txt file
        from the domain admin reserved share.
    """

    def execute(self, state: State) -> Observation:
        if self.HOST_LMT_DC not in state.hosts:
            return self._fail_obs("LMT domain controller not present in state.")

        lmt_dc_host = state.hosts[self.HOST_LMT_DC]
        dc_flags = self._get_flags_dict(lmt_dc_host)

        if not dc_flags.get("lmt_admin_session", False):
            return self._fail_obs("No LMT\\Administrator session on LMTDC01 yet.")

        dc_flags["notes_read"] = True
        dc_flags["notes_path"] = self.VALUABLE_SHARE_PATH
        dc_flags["notes_content"] = "You have entered a domain admin reserved area"

        return self._success_obs("Restricted notes.txt successfully read from the share.")
