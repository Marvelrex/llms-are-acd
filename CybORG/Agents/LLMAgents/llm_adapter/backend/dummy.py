import json
from typing import List, Dict
from CybORG.Agents.LLMAgents.llm_adapter.backend.model_backend import ModelBackend
from CybORG.Agents.LLMAgents.llm_adapter.utils.weave_stub import weave

class DummyBackend(ModelBackend):
    """Dummy model backend for testing."""

    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Return strict ranking JSON so the SelfEvolvingGraphAgent can exercise its parsing
        # logic without any API keys.
        payload = {
            "ranked_actions": [
                "defender_remove",
                "defender_analyse",
                "defender_monitor",
                "defender_deploy_decoy",
                "defender_block_traffic_zone",
                "defender_allow_traffic_zone",
                "defender_restore",
                "defender_sleep",
            ],
            "confidence": 0.85,
        }
        response = json.dumps(payload)
        return self._format_response(response)
