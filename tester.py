# tester_lmt_full_chain.py

from CybORG import CybORG
from CybORG.Simulator.Scenarios import LMTScenarioGenerator

from CybORG.Simulator.Actions.LMTAttackActions import (
    DisableMonitoringAndPrepareToolsDirectory,
    DeployReverseShellAgent,
    ExecuteMimikatzDump,
    PassTheHashAttack,
    AccessRestrictedRemoteDirectory,
)

def print_step_result(step_name, res):
    print(f"\n=== {step_name} ===")
    print("success:", res.observation.get("success", None))
    note = res.observation.get("lmt_note", None)
    if note is not None:
        print("note   :", note)
    print("reward :", res.reward)
    print("done   :", res.done)

if __name__ == "__main__":
    # Build environment with your custom scenario
    sg = LMTScenarioGenerator(steps=50)
    env = CybORG(sg, "sim")

    controller = env.environment_controller

    # We only have one red agent
    agent = controller.get_active_agents()[0]
    print("Active agent:", agent)

    # Reset
    res = env.reset(agent)
    print_step_result("RESET", res)

    # 1) Disable monitoring + create tools dir
    res = env.step(agent, DisableMonitoringAndPrepareToolsDirectory())
    print_step_result("1) DisableMonitoringAndPrepareToolsDirectory", res)

    # 2) Deploy reverse shell
    res = env.step(agent, DeployReverseShellAgent())
    print_step_result("2) DeployReverseShellAgent", res)

    # 3) Execute Mimikatz & dump hash
    res = env.step(agent, ExecuteMimikatzDump())
    print_step_result("3) ExecuteMimikatzDump", res)

    # 4) Pass-the-Hash to LMTDC01
    res = env.step(agent, PassTheHashAttack())
    print_step_result("4) PassTheHashAttack", res)

    # 5) Access restricted remote directory (notes.txt)
    res = env.step(agent, AccessRestrictedRemoteDirectory())
    print_step_result("5) AccessRestrictedRemoteDirectory", res)

    # Inspect final flags on hosts
    state = controller.state
    it_flags = state.hosts["LMT-IT-DC01"].info
    dc_flags = state.hosts["LMTDC01"].info

    print("\n--- Final host flags ---")
    print("IT-DC info:", it_flags)
    print("LMTDC info:", dc_flags)
