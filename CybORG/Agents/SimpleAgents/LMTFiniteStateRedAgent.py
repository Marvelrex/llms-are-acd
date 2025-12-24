from CybORG.Agents.SimpleAgents.BaseAgent import BaseAgent
from CybORG.Shared.Enums import TernaryEnum
from CybORG.Simulator.Actions import Sleep
from CybORG.Simulator.Actions.LMTAttackActions import (
    DisableMonitoringAndPrepareToolsDirectory,
    DeployReverseShellAgent,
    ExecuteMimikatzDump,
    PassTheHashAttack,
    AccessRestrictedRemoteDirectory,
)


class LMTFiniteStateRedAgent(BaseAgent):
    """Finite-state red agent with aggressive-style retries for LMT actions."""

    def __init__(self, name=None, np_random=None):
        super().__init__(name, np_random)
        self._actions = [
            DisableMonitoringAndPrepareToolsDirectory,
            DeployReverseShellAgent,
            ExecuteMimikatzDump,
            PassTheHashAttack,
            AccessRestrictedRemoteDirectory,
        ]
        self._state = 0
        self._last_action = None
        self._transitions_success = {
            0: ([1], [1.0]),
            1: ([2], [1.0]),
            2: ([3], [1.0]),
            3: ([4], [1.0]),
            4: ([0], [1.0]),  # loop after success
        }
        # Aggressive retries: on failure, prefer stepping back to re-establish foothold.
        self._transitions_failure = {
            0: ([0], [1.0]),
            1: ([0, 1], [0.7, 0.3]),
            2: ([1, 2], [0.7, 0.3]),
            3: ([2, 3], [0.7, 0.3]),
            4: ([3, 4], [0.7, 0.3]),
        }

    def get_action(self, observation, action_space):
        if observation.get("success") == TernaryEnum.IN_PROGRESS:
            return Sleep()

        success = observation.get("success")
        if self._last_action is not None and success in (TernaryEnum.TRUE, TernaryEnum.FALSE):
            self._advance_state(success == TernaryEnum.TRUE)

        action = self._actions[self._state]()
        self._last_action = action
        return action

    def end_episode(self):
        self._state = 0
        self._last_action = None

    def set_initial_values(self, action_space, observation):
        pass

    def _advance_state(self, success: bool) -> None:
        if success:
            next_states, weights = self._transitions_success[self._state]
        else:
            next_states, weights = self._transitions_failure[self._state]
        self._state = int(self.np_random.choice(next_states, p=weights))
