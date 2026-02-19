from typing import Dict, List
from CybORG.Agents.LLMAgents.llm_adapter.backend.model_backend import ModelBackend
from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger
from CybORG.Agents.LLMAgents.llm_adapter.utils.weave_stub import weave
from openai import OpenAI


class DeepSeekBackend(ModelBackend):
    """OpenRouter backend for GPT-4.1-mini, GPT-5-mini, DeepSeek, and other models."""

    def __init__(self, hyperparams: dict, api_key: str):
        self.headers={
            "HTTP-Referer": "https://github.com/cage-paper/cage-4",
            "X-Title": "CAGE Project"
        }

        self.openai_client = OpenAI(base_url="https://openrouter.ai/api/v1",
                                    api_key=api_key)
        self.model_name = hyperparams.get("model_name", "").lower()
        self.temperature = hyperparams["generate"]["temperature"]
        self.max_tokens = hyperparams["generate"]["max_new_tokens"]

    @weave.op
    def generate(self, messages: List[Dict[str, str]]) -> str:
        # Pass structured messages directly — OpenRouter supports the standard
        # OpenAI chat-completions format for all hosted models (GPT-4.1-mini,
        # GPT-5-mini, DeepSeek, etc.).  The old _format_messages_history()
        # approach flattened everything into a single "user" message with custom
        # <|system|>/<|user|> tags that newer models reject or ignore.
        response = self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            extra_headers=self.headers,
            extra_body={}
        ).choices[0].message.content
        Logger.info(f"Generated response: {response}")
        return str(response or "")
