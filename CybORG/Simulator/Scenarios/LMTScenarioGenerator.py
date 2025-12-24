# LMTScenarioGenerator.py

# Unique scenario code: LMT-SC1-REDONLY-20251210

from ipaddress import IPv4Network
from typing import Dict, List, Type, Optional

from gym.utils.seeding import RandomNumberGenerator

from CybORG.Agents import BaseAgent, SleepAgent
from CybORG.Shared import Scenario
from CybORG.Shared.RewardCalculator import EmptyRewardCalculator
from CybORG.Shared.Scenario import ScenarioAgent
from CybORG.Shared.Session import Session, VelociraptorServer
from CybORG.Shared.Scenarios.ScenarioGenerator import ScenarioGenerator

from CybORG.Simulator.Actions import (
    Analyse,
    DeployDecoy,
    Monitor,
    Remove,
    Restore,
    Sleep,
)
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import (
    AllowTrafficZone,
    BlockTrafficZone,
)
from CybORG.Simulator.Actions.LMTAttackActions import (
    DisableMonitoringAndPrepareToolsDirectory,
    DeployReverseShellAgent,
    ExecuteMimikatzDump,
    PassTheHashAttack,
    AccessRestrictedRemoteDirectory,
)
from CybORG.Simulator.Host import Host
from CybORG.Simulator.Interface import Interface
from CybORG.Simulator.Subnet import Subnet
from CybORG.Simulator.User import User


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

    - 5 Blue agents:
        * blue_agent_0..4, each with a Velociraptor server session on LMT-IT-DC01

    - 1 Red agent:
        * red_agent_0 with a session on LMT-IT-DC01 as it\\administrator
    """

    MAX_BANDWIDTH = 100
    MESSAGE_LENGTH = 8

    def __init__(
        self,
        blue_agent_class: Optional[Type[BaseAgent]] = None,
        red_agent_class: Optional[Type[BaseAgent]] = None,
        steps: int = 50,
    ):
        super().__init__()
        self.blue_agent_class = blue_agent_class or SleepAgent
        self.red_agent_class = red_agent_class or SleepAgent
        self.steps = steps

    def create_scenario(self, np_random: RandomNumberGenerator) -> Scenario:
        self.np_random = np_random

        subnets: Dict[str, Subnet] = self._create_subnets()
        hosts: Dict[str, Host] = self._create_hosts(subnets)

        agents: Dict[str, ScenarioAgent] = {}
        self._create_blue_agents(hosts, subnets, agents)
        self._create_red_agent(hosts, subnets, agents)

        team_agents = self._generate_team_agents(agents)

        scenario = Scenario(
            agents=agents,
            team_calcs=None,
            team_agents=team_agents,
            hosts=hosts,
            subnets=subnets,
            mission_phases=self._generate_mission_phases(self.steps),
            allowed_subnets_per_mphase=[list(subnets.keys())] * 3,
            predeployed=False,
            max_bandwidth=self.MAX_BANDWIDTH,
        )

        scenario.team_calc = self._generate_team_calcs()
        return scenario

    def _create_subnets(self) -> Dict[str, Subnet]:
        subnets: Dict[str, Subnet] = {}

        it_network = IPv4Network("192.168.57.0/24")
        subnets["lmt_it_subnet"] = Subnet(
            name="lmt_it_subnet",
            size=0,
            hosts=[],
            nacls={},
            cidr=it_network,
            ip_addresses=[],
        )

        lmt_network = IPv4Network("192.168.56.0/24")
        subnets["lmt_dc_subnet"] = Subnet(
            name="lmt_dc_subnet",
            size=0,
            hosts=[],
            nacls={},
            cidr=lmt_network,
            ip_addresses=[],
        )

        file_network = IPv4Network("192.168.58.0/24")
        subnets["lmt_file_subnet"] = Subnet(
            name="lmt_file_subnet",
            size=0,
            hosts=[],
            nacls={},
            cidr=file_network,
            ip_addresses=[],
        )

        return subnets

    def _create_hosts(self, subnets: Dict[str, Subnet]) -> Dict[str, Host]:
        hosts: Dict[str, Host] = {}

        def make_windows_host(
            hostname: str,
            subnet_key: str,
            desired_ip: str,
            admin_username: str,
        ) -> Host:
            subnet = subnets[subnet_key]
            ip_addr = IPv4Network(f"{desired_ip}/32").network_address
            if ip_addr not in subnet.cidr:
                ip_addr = list(subnet.cidr.hosts())[0 + len(subnet.hosts)]

            interface = Interface(
                name="eth0",
                ip_address=ip_addr,
                subnet=subnet.cidr,
                interface_type="wired",
                data_links=[],
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

    def _create_blue_agents(
        self,
        hosts: Dict[str, Host],
        subnets: Dict[str, Subnet],
        agents: Dict[str, ScenarioAgent],
    ) -> None:
        blue_actions = [
            Monitor,
            Analyse,
            Remove,
            Restore,
            DeployDecoy,
            BlockTrafficZone,
            AllowTrafficZone,
            Sleep,
        ]
        osint_hosts = {
            hostname: {"Interfaces": "All", "System info": "All", "User info": "All"}
            for hostname in hosts.keys()
        }
        allowed_subnets = list(subnets.keys())

        for idx in range(5):
            agent_name = f"blue_agent_{idx}"
            session = VelociraptorServer(
                ident=idx,
                hostname="LMT-IT-DC01",
                username="blue\\analyst",
                agent=agent_name,
                pid=None,
                session_type="VelociraptorServer",
            )
            agent_type = self.blue_agent_class(agent_name)
            default_actions = (Monitor, {"session": 0, "agent": agent_name})
            agents[agent_name] = ScenarioAgent(
                agent_name,
                "Blue",
                [session],
                blue_actions,
                {"Hosts": osint_hosts},
                allowed_subnets,
                agent_type,
                True,
                default_actions,
            )

    def _create_red_agent(
        self,
        hosts: Dict[str, Host],
        subnets: Dict[str, Subnet],
        agents: Dict[str, ScenarioAgent],
    ) -> None:
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

        osint = {
            "Hosts": {
                hostname: {"Interfaces": "All", "System info": "All", "User info": "All"}
                for hostname in hosts.keys()
            }
        }

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
            [red_session],
            red_actions,
            osint,
            allowed_subnets,
            agent_type,
            True,
            default_actions,
        )

    def _generate_team_agents(self, agents: Dict[str, ScenarioAgent]) -> Dict[str, List[str]]:
        return {
            "Blue": [name for name in agents.keys() if name.startswith("blue_agent_")],
            "Red": [name for name in agents.keys() if name.startswith("red_agent_")],
        }

    def _generate_team_calcs(self) -> Dict[str, Dict[str, EmptyRewardCalculator]]:
        return {
            "Blue": {"None": EmptyRewardCalculator("Blue")},
            "Red": {"None": EmptyRewardCalculator("Red")},
        }

    def _generate_mission_phases(self, steps: int):
        return (steps, 0, 0)

    def determine_done(self, env_controller) -> bool:
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
