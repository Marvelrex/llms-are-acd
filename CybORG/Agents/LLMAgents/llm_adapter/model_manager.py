import json
import os
from pathlib import Path
from typing import Any, Dict, List

from CybORG.Agents.LLMAgents.llm_adapter.backend.model_backend import ModelBackend


def _load_dotenv_if_present() -> None:
    """Load a .env file (without adding a new dependency) before reading env vars."""
    dotenv_path = None
    model_file_path = Path(__file__).resolve()
    for parent in [model_file_path] + list(model_file_path.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            dotenv_path = candidate
            break
    if not dotenv_path:
        return

    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        # Lightweight fallback parser that keeps existing environment values.
        for line in dotenv_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped.startswith("export "):
                stripped = stripped[len("export ") :].strip()
            if "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    else:
        load_dotenv(dotenv_path, override=False)


_load_dotenv_if_present()

HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

class BackendFactory:
    """Factory class for creating model backend instances."""

    @staticmethod
    def create_backend(backend_name: str, hyperparams: dict) -> ModelBackend:
        if backend_name == "openai":
            from CybORG.Agents.LLMAgents.llm_adapter.backend.openai import OpenAIBackend
            return OpenAIBackend(hyperparams=hyperparams, api_key=OPENAI_API_KEY)
        elif backend_name == "huggingface":
            from CybORG.Agents.LLMAgents.llm_adapter.backend.huggingface import HuggingFaceBackend
            return HuggingFaceBackend(hyperparams=hyperparams, token=HF_TOKEN)
        elif backend_name == "new-openai":
            from CybORG.Agents.LLMAgents.llm_adapter.backend.openai import NewOpenAIBackend
            return NewOpenAIBackend(hyperparams=hyperparams, api_key=OPENAI_API_KEY)
        elif backend_name == "deepseek":
            from CybORG.Agents.LLMAgents.llm_adapter.backend.deepseek import DeepSeekBackend
            if not OPENROUTER_API_KEY:
                raise ValueError("OPENROUTER_API_KEY environment variable is required for DeepSeek models")
            return DeepSeekBackend(hyperparams=hyperparams, api_key=OPENROUTER_API_KEY)
        elif backend_name == "dummy":
            from CybORG.Agents.LLMAgents.llm_adapter.backend.dummy import DummyBackend
            return DummyBackend()
        else:
            raise ValueError(f"Invalid backend: {backend_name}")

class ModelManager:
    """Model manager class.
    
    This class is responsible for managing the model backend instances, sending messages to the model backend,
    handling the responses, and storing the model configurations.
    """
    def __init__(self, hyperparams: dict):
        # Defensive copy: some callers pass a dict that they reuse across policies.
        self.hyperparams = dict(hyperparams or {})
        self.backend_name = str(self.hyperparams.get("backend", "")).lower()
        if not self.backend_name:
            raise ValueError("Model config must define 'backend'")

        # Single source of truth for OpenAI model selection.
        #
        # If set, this overrides the YAML `model_name` for OpenAI-backed backends.
        # This is intentionally *not* applied to the OpenRouter/DeepSeek backend,
        # since those model names use different conventions (e.g., `openai/gpt-4.1-mini`).
        model_override = os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_MODEL")
        if self.backend_name in {"openai", "new-openai"}:
            if model_override:
                self.hyperparams["model_name"] = str(model_override)
            else:
                # Default to GPT-4.1 mini when not explicitly configured.
                self.hyperparams.setdefault("model_name", "gpt-4.1-mini")

        self.model_backend = BackendFactory.create_backend(self.backend_name, self.hyperparams)
        self.log_path = self._resolve_log_path(self.hyperparams)

    @staticmethod
    def _resolve_log_path(hyperparams: dict) -> Path:
        """
        Resolve where to write prompt/response traces.

        Priority:
        1. `log_path` in the model config YAML
        2. `CAGE4_LLM_LOG_PATH` (full file path)
        3. `CAGE4_LLM_LOG_ROOT` or `OUTPUT_DIR` (directory)
        4. Default under this package (`llm_adapter/logs/llm_calls.jsonl`)
        """
        cfg_path = hyperparams.get("log_path")
        if cfg_path:
            return Path(cfg_path)

        env_path = os.environ.get("CAGE4_LLM_LOG_PATH")
        if env_path:
            return Path(env_path)

        env_root = os.environ.get("CAGE4_LLM_LOG_ROOT") or os.environ.get("OUTPUT_DIR")
        if env_root:
            return Path(env_root) / "llm_calls.jsonl"

        return Path(__file__).resolve().parent / "logs" / "llm_calls.jsonl"
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generates a response using the model backend."""
        response = self.model_backend.generate(messages)
        self._log_llm_interaction(messages, response)
        return response

    def _log_llm_interaction(self, messages: List[Dict[str, Any]], response: str) -> None:
        """Append the prompt/response pair to a JSONL file for auditing."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {"messages": messages, "response": response}
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception:
            # Logging must not break inference; ignore failures silently.
            return
