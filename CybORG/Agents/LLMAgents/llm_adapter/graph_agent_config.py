from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Literal

from CybORG.Agents.LLMAgents.llm_adapter.state_signature import StateSignatureConfig


@dataclass
class GraphAgentConfig:
    """Feature flags + lightweight hyperparameters for ablations.

    Defaults enable the requested targeted improvements while keeping the agent API intact.
    """

    # (1) Credit assignment improvements
    use_discounted_credit: bool = True
    credit_gamma: float = 0.97
    use_baseline: bool = True
    baseline_beta: float = 0.05
    baseline_scope: Literal["global", "role"] = "global"

    # (2) State-conditioned priors
    use_state_conditioning: bool = True
    state_signature_config: StateSignatureConfig = field(default_factory=StateSignatureConfig)

    # (3) Cold-start sparsification + discovery
    # Bug-10 fix: increased from 3 to 5 so each defender starts connected to
    # more attacker states, improving candidate diversity during cold start.
    use_sparse_init: bool = True
    sparse_init_m: int = 5
    sparse_init_seed: int = 1337
    use_edge_discovery: bool = True

    # (4) Safer prompting
    use_prompt_risk_summary: bool = True

    # (4b) Prompt neutralization (avoid persistent incident priming)
    # Tokens that must not appear in the prompt text (case-insensitive). These are enforced via
    # a last-mile prompt sanitizer in the agent.
    banned_prompt_tokens: list[str] = field(
        default_factory=lambda: [
            "compromise",
            "compromised",
            "post_compromise",
            "incident",
            "breach",
            "intrusion",
        ]
    )

    # (4c) LLM output format + circuit-breaker gate (LLM dominates when valid)
    # Minimum overall confidence required to accept the LLM ranking.
    llm_conf_min: float = 0.55
    # When LLM is valid/confident, compute g = clamp(gate_base + gate_k*confidence, gate_min, 1.0).
    # This is primarily a diagnostic signal; decisions should overwhelmingly follow the LLM ranking.
    gate_base: float = 0.85
    gate_k: float = 0.10
    gate_min: float = 0.80

    # (5) Deprecated: prior fusion gate stability knobs.
    # The SelfEvolvingGraphAgent selection path is a strict circuit-breaker:
    # - valid/confident LLM ranking => choose LLM action
    # - otherwise => graph prior fallback
    # These knobs are retained for backward compatibility but must not downweight a valid LLM choice.
    use_new_gate: bool = True
    gate_evidence_visits_scale: float = 20.0  # evidence ~= log1p(visits) / log1p(scale)
    # Clip evidence used for gating to reduce saturation (1.0 disables clipping).
    evidence_clip_max: float = 0.7
    # Warmup: during early episodes, de-emphasize graph evidence so LLM can steer exploration.
    warmup_episodes: int = 3
    warmup_gate_floor: float = 0.25
    warmup_ignore_evidence: bool = True

    # (6) Incident-response tuning (restore anti-spam)
    # Keep post-compromise bias only while compromise evidence is recent.
    post_compromise_window_steps: int = 10  # W
    # After any Restore attempt, suppress Restore bias for K defender decisions.
    restore_cooldown_steps: int = 3  # K

    # (7) Candidate diversity / exploration
    candidate_top_k_base: int = 12
    candidate_top_k_boost: int = 6
    candidate_diversity_window: int = 50
    candidate_diversity_floor: int = 8
    candidate_eps: float = 0.15
    candidate_eps_n: int = 4
    traffic_gate_enabled: bool = True

    # (8) Traffic action safety
    traffic_cooldown_steps: int = 3
    traffic_flipflop_horizon: int = 4
