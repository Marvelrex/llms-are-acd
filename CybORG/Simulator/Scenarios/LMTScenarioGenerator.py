# LMTScenarioGenerator.py

# Unique scenario code: LMT-SC1-REDONLY-20251210

from ipaddress import IPv4Network
from typing import Dict, List, Type, Optional

from gym.utils.seeding import RandomNumberGenerator

from CybORG.Agents import BaseAgent, SleepAgent
from CybORG.Shared import Scenario
from CybORG.Shared.RewardCalculator import EmptyRewardCalculator
from CybORG.Shared.Scenario import ScenarioAgent
from CybORG.Shared.Session import Session
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator

from CybORG.Simulator.Actions.Action import Sleep
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Interface import Interface
from CybORG.Simulator.Subnet import Subnet
from CybORG.Simulator.User import User

# 🔴 Import your custom LMT red actions (UPDATED: only the 5 kept actions)
from CybORG.Simulator.Actions.LMTAttackActions import (
    DisableMonitoringAndPrepareToolsDirectory,
    DeployReverseShellAgent,
    ExecuteMimikatzDump,
    PassTheHashAttack,
    AccessRestrictedRemoteDirectory,
)

class LMTScenarioGenerator(ScenarioGenerator):
    """
    Minimal custom scenario for the LMT Pass-the-Hash lab.

    - 3 hosts:
        * LMT-IT-DC01   (IT DC, starting foothold)
        * LMTDC01       (main LMT domain controller)
        * FILE-SERVER   (crown-jewel share lives conceptually here)

    - 3 subnets:
        * 192.168.57.0/24  -> LMT-IT-DC01
        * 192.168.56.0/24  -> LMTDC01
        * 192.168.58.0/24  -> FILE-SERVER

    - 1 Red agent:
        * red_agent_0 with a session on LMT-IT-DC01 as it\\administrator

    No Blue or Green agents yet.
    Reward is currently always 0 (EmptyRewardCalculator for Red).
    """

    # SimulationController expects these (see EnterpriseScenarioGenerator)
    MAX_BANDWIDTH = 100
    MESSAGE_LENGTH = 8

    def __init__(
        self,
        red_agent_class: Optional[Type[BaseAgent]] = None,
        steps: int = 50,
    ):
        super().__init__()
        # If no custom red agent class is provided, use SleepAgent as a safe default.
        self.red_agent_class: Type[BaseAgent] = red_agent_class or SleepAgent
        self.steps = steps

    # ------------------------------------------------------------------
    # Core entry point: CybORG calls this to build the Scenario
    # ------------------------------------------------------------------
    def create_scenario(self, np_random: RandomNumberGenerator) -> Scenario:
        """
        Build a minimal LMT scenario and return a Scenario object.
        """
        self.np_random = np_random

        # 1) Create subnets
        subnets: Dict[str, Subnet] = self._create_subnets()

        # 2) Create hosts
        hosts: Dict[str, Host] = self._create_hosts(subnets)

        # 3) Create red agent with a single session on LMT-IT-DC01
        agents: Dict[str, ScenarioAgent] = {}
        self._create_red_agent(hosts, subnets, agents)

        # 4) Map agents to teams (only Red for now)
        team_agents = self._generate_team_agents(agents)

        # 5) Build Scenario
        scenario = Scenario(
            agents=agents,
            team_calcs=None,  # we set scenario.team_calc below
            team_agents=team_agents,
            hosts=hosts,
            subnets=subnets,
            mission_phases=self._generate_mission_phases(self.steps),
            # We don't care about traffic policy yet, so leave these empty:
            allowed_subnets_per_mphase=[[]] * 3,
            predeployed=False,
            max_bandwidth=self.MAX_BANDWIDTH,
        )

        # 6) Attach team reward calculators (Red only, empty reward)
        scenario.team_calc = self._generate_team_calcs()

        return scenario

    # ------------------------------------------------------------------
    # Subnet & host construction
    # ------------------------------------------------------------------
    def _create_subnets(self) -> Dict[str, Subnet]:
        """
        Create three simple subnets for IT-DC, LMT-DC, and FILE-SERVER.
        """
        subnets: Dict[str, Subnet] = {}

        # IT subnet: 192.168.57.0/24
        it_network = IPv4Network("192.168.57.0/24")
        it_subnet = Subnet(
            name="lmt_it_subnet",
            size=0,            # will be updated after adding hosts
            hosts=[],
            nacls={},          # no ACLs for now
            cidr=it_network,
            ip_addresses=[],
        )
        subnets["lmt_it_subnet"] = it_subnet

        # LMT DC subnet: 192.168.56.0/24
        lmt_network = IPv4Network("192.168.56.0/24")
        lmt_subnet = Subnet(
            name="lmt_dc_subnet",
            size=0,
            hosts=[],
            nacls={},
            cidr=lmt_network,
            ip_addresses=[],
        )
        subnets["lmt_dc_subnet"] = lmt_subnet

        # File server subnet: 192.168.58.0/24
        file_network = IPv4Network("192.168.58.0/24")
        file_subnet = Subnet(
            name="lmt_file_subnet",
            size=0,
            hosts=[],
            nacls={},
            cidr=file_network,
            ip_addresses=[],
        )
        subnets["lmt_file_subnet"] = file_subnet

        return subnets

    def _create_hosts(self, subnets: Dict[str, Subnet]) -> Dict[str, Host]:
        """
        Create three Windows-like hosts and attach them to subnets.
        We don't model real services or processes yet – that will come later.
        """
        hosts: Dict[str, Host] = {}

        def make_windows_host(
            hostname: str,
            subnet_key: str,
            desired_ip: str,
            admin_username: str,
        ) -> Host:
            subnet = subnets[subnet_key]

            # Ensure desired_ip is in the subnet CIDR
            ip_addr = IPv4Network(f"{desired_ip}/32").network_address
            if ip_addr not in subnet.cidr:
                # If it’s not inside the CIDR range, just take the first free host IP
                ip_addr = list(subnet.cidr.hosts())[0 + len(subnet.hosts)]

            interface = Interface(
                name="eth0",
                ip_address=ip_addr,
                subnet=subnet.cidr,
                interface_type="wired",
                data_links=[],   # no explicit routing links yet
                swarm=False,
            )

            system_info = {
                "OSType": "WINDOWS",
                "OSDistribution": "Windows",
                "OSVersion": "Server",
                "Architecture": "x64",
            }

            admin_user = User(
                groups=[{"GID": 0, "Group Name": "Administrators"}],
                uid=0,
                username=admin_username,
            )
            normal_user = User(
                groups=[{"GID": 1000, "Group Name": "Users"}],
                uid=1000,
                username="user",
                bruteforceable=True,
            )

            host = Host(
                hostname=hostname,
                host_type="",
                processes=None,
                system_info=system_info,
                interfaces=[interface],
                info=None,
                users=[admin_user, normal_user],
                services=None,
                respond_to_ping=True,
                np_random=self.np_random,
            )

            subnet.hosts.append(hostname)
            subnet.ip_addresses.append(ip_addr)
            subnet.size = len(subnet.hosts)

            return host

        hosts["LMT-IT-DC01"] = make_windows_host(
            hostname="LMT-IT-DC01",
            subnet_key="lmt_it_subnet",
            desired_ip="192.168.57.11",
            admin_username="it\\administrator",
        )

        hosts["LMTDC01"] = make_windows_host(
            hostname="LMTDC01",
            subnet_key="lmt_dc_subnet",
            desired_ip="192.168.56.10",
            admin_username="lmt\\administrator",
        )

        hosts["FILE-SERVER"] = make_windows_host(
            hostname="FILE-SERVER",
            subnet_key="lmt_file_subnet",
            desired_ip="192.168.58.13",
            admin_username="file\\administrator",
        )

        return hosts

    # ------------------------------------------------------------------
    # Red agent setup
    # ------------------------------------------------------------------
    def _create_red_agent(
        self,
        hosts: Dict[str, Host],
        subnets: Dict[str, Subnet],
        agents: Dict[str, ScenarioAgent],
    ) -> None:
        """
        Create a single red agent starting with a session on LMT-IT-DC01.
        """
        agent_name = "red_agent_0"

        red_session = Session(
            name="red_session_0",
            username="it\\administrator",
            session_type="red_session",
            hostname="LMT-IT-DC01",
            pid=None,
            ident=None,
            agent=None,
        )

        sessions: List[Session] = [red_session]

        osint = {
            "Hosts": {
                "LMT-IT-DC01": {"Interfaces": "All", "System info": "All", "User info": "All"},
                "LMTDC01": {"Interfaces": "All", "System info": "All", "User info": "All"},
                "FILE-SERVER": {"Interfaces": "All", "System info": "All", "User info": "All"},
            }
        }

        # 🧨 Available red actions: UPDATED (only 5 kept actions + Sleep)
        red_actions = [
            DisableMonitoringAndPrepareToolsDirectory,
            DeployReverseShellAgent,
            ExecuteMimikatzDump,
            PassTheHashAttack,
            AccessRestrictedRemoteDirectory,
            Sleep,
        ]

        allowed_subnets = list(subnets.keys())

        agent_type = self.red_agent_class(agent_name)
        default_actions = (Sleep, {})

        agents[agent_name] = ScenarioAgent(
            agent_name,
            "Red",
            sessions,
            red_actions,
            osint,
            allowed_subnets,
            agent_type,
            True,
            default_actions,
        )

    # ------------------------------------------------------------------
    # Team + reward helpers
    # ------------------------------------------------------------------
    def _generate_team_agents(self, agents: Dict[str, ScenarioAgent]) -> Dict[str, List[str]]:
        team_agents: Dict[str, List[str]] = {
            "Red": [name for name in agents.keys() if name.startswith("red_agent_")]
        }
        return team_agents

    def _generate_team_calcs(self) -> Dict[str, Dict[str, EmptyRewardCalculator]]:
        team_calcs: Dict[str, Dict[str, EmptyRewardCalculator]] = {
            "Red": {"None": EmptyRewardCalculator("Red")}
        }
        return team_calcs

    def _generate_mission_phases(self, steps: int):
        return (steps, 0, 0)

    # ------------------------------------------------------------------
    # Episode termination
    # ------------------------------------------------------------------
    def determine_done(self, env_controller) -> bool:
        """
        End the episode either when:
          - step limit is reached, OR
          - the restricted notes.txt has been read on LMTDC01
        """
        if env_controller.step_count >= (self.steps - 1):
            return True

        state = env_controller.state
        if "LMTDC01" in state.hosts:
            host = state.hosts["LMTDC01"]
            if host.info and "lmt_flags" in host.info:
                if host.info["lmt_flags"].get("notes_read", False):
                    return True

        return False

    def __str__(self) -> str:
        return "LMTScenarioGenerator"


__all__ = ["LMTScenarioGenerator"]
