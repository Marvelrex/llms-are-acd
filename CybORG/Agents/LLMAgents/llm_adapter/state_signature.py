from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


_RESERVED_OBS_KEYS = {"success", "action", "phase", "message"}


def _bucket_label(value: int, *, cutoffs: Sequence[int], labels: Sequence[str]) -> str:
    """Return a stable bucket label for a non-negative integer value.

    cutoffs are inclusive upper bounds for each bucket except the final open-ended bucket.
    labels must have length len(cutoffs) + 1.
    """
    v = int(value)
    if v < 0:
        v = 0
    if len(labels) != len(cutoffs) + 1:
        raise ValueError("labels must have length len(cutoffs) + 1")
    for idx, cutoff in enumerate(cutoffs):
        if v <= int(cutoff):
            return str(labels[idx])
    return str(labels[-1])


def _to_tri_state(value: Any) -> str:
    """Map a best-effort action outcome into a compact tri-state token."""
    if value is True:
        return "T"
    if value is False:
        return "F"

    # Common cases in this codebase use TernaryEnum.*; we avoid importing it here.
    s = str(value).upper()
    if "TRUE" in s:
        return "T"
    if "FALSE" in s:
        return "F"
    return "U"


def _iter_host_obs(observation: Mapping[str, Any]) -> Iterable[Tuple[str, Mapping[str, Any]]]:
    for key, value in observation.items():
        if key in _RESERVED_OBS_KEYS:
            continue
        if isinstance(value, Mapping):
            yield str(key), value


def _count_compromised_hosts(
    observation: Mapping[str, Any],
    *,
    user_ioc_filenames: Sequence[str],
    admin_ioc_filenames: Sequence[str],
) -> int:
    """Best-effort compromise count from observation IOCs (deterministic, no ML)."""
    user_iocs = set(map(str, user_ioc_filenames))
    admin_iocs = set(map(str, admin_ioc_filenames))
    compromised = 0

    for _host, host_obs in _iter_host_obs(observation):
        files = host_obs.get("Files")
        if not isinstance(files, list):
            continue
        hit = False
        for f in files:
            if not isinstance(f, Mapping):
                continue
            fname = f.get("File Name")
            if fname in admin_iocs or fname in user_iocs:
                hit = True
                break
        compromised += int(hit)
    return int(compromised)


def _count_alert_hosts(observation: Mapping[str, Any]) -> int:
    """Best-effort alert count from suspicious process / IOC presence (deterministic)."""
    alerts = 0
    for _host, host_obs in _iter_host_obs(observation):
        hit = False

        procs = host_obs.get("Processes")
        if isinstance(procs, list):
            for proc in procs:
                if not isinstance(proc, Mapping):
                    continue
                # Match existing obs_formatter heuristic for "suspicious process".
                if "PID" in proc and "username" not in proc:
                    hit = True
                    break

        if not hit:
            files = host_obs.get("Files")
            if isinstance(files, list):
                for f in files:
                    if not isinstance(f, Mapping):
                        continue
                    if f.get("File Name") is not None:
                        # Presence of any file record is a weak-but-stable alert feature.
                        hit = True
                        break

        alerts += int(hit)
    return int(alerts)


def _count_available_actions(action_space: Any) -> int:
    """Best-effort count of currently-available action *types* (not parameterizations)."""
    if isinstance(action_space, Mapping):
        action_map = action_space.get("action")
        if isinstance(action_map, Mapping):
            return int(sum(1 for _cls, valid in action_map.items() if bool(valid)))
    try:
        return int(len(action_space))
    except Exception:
        return 0


@dataclass(frozen=True)
class StateSignatureConfig:
    """Configurable, deterministic state signature for lightweight state-conditioned priors.

    Extend by:
    - Adding a new feature function to FEATURE_REGISTRY.
    - Adding the feature key to `features` here (order matters for determinism).
    """

    features: Tuple[str, ...] = (
        "compromised_hosts",
        "alert_hosts",
        "last_action_success",
        "step_bucket",
        "action_space_bucket",
    )

    compromised_host_cutoffs: Tuple[int, ...] = (0, 1, 2)
    compromised_host_labels: Tuple[str, ...] = ("0", "1", "2", "3p")

    alert_host_cutoffs: Tuple[int, ...] = (0, 1, 2, 3)
    alert_host_labels: Tuple[str, ...] = ("0", "1", "2", "3", "4p")

    step_cutoffs: Tuple[int, ...] = (3, 7)
    step_labels: Tuple[str, ...] = ("early", "mid", "late")

    action_space_cutoffs: Tuple[int, ...] = (4, 6, 8, 12)
    action_space_labels: Tuple[str, ...] = ("xs", "s", "m", "l", "xl")

    user_ioc_filenames: Tuple[str, ...] = ("cmd.sh", "cmd.exe")
    admin_ioc_filenames: Tuple[str, ...] = ("escalate.sh", "escalate.exe")


@dataclass(frozen=True)
class _SignatureContext:
    observation: Mapping[str, Any]
    last_action_status: Any
    step_idx: int
    action_space_size: int


_FeatureFn = Callable[[_SignatureContext, StateSignatureConfig], str]


def _feature_compromised_hosts(ctx: _SignatureContext, cfg: StateSignatureConfig) -> str:
    n = _count_compromised_hosts(
        ctx.observation,
        user_ioc_filenames=cfg.user_ioc_filenames,
        admin_ioc_filenames=cfg.admin_ioc_filenames,
    )
    b = _bucket_label(n, cutoffs=cfg.compromised_host_cutoffs, labels=cfg.compromised_host_labels)
    return f"comp={b}"


def _feature_alert_hosts(ctx: _SignatureContext, cfg: StateSignatureConfig) -> str:
    n = _count_alert_hosts(ctx.observation)
    b = _bucket_label(n, cutoffs=cfg.alert_host_cutoffs, labels=cfg.alert_host_labels)
    return f"alerts={b}"


def _feature_last_action_success(ctx: _SignatureContext, _cfg: StateSignatureConfig) -> str:
    return f"las={_to_tri_state(ctx.last_action_status)}"


def _feature_step_bucket(ctx: _SignatureContext, cfg: StateSignatureConfig) -> str:
    b = _bucket_label(ctx.step_idx, cutoffs=cfg.step_cutoffs, labels=cfg.step_labels)
    return f"step={b}"


def _feature_action_space_bucket(ctx: _SignatureContext, cfg: StateSignatureConfig) -> str:
    b = _bucket_label(ctx.action_space_size, cutoffs=cfg.action_space_cutoffs, labels=cfg.action_space_labels)
    return f"as={b}"


FEATURE_REGISTRY: Dict[str, _FeatureFn] = {
    "compromised_hosts": _feature_compromised_hosts,
    "alert_hosts": _feature_alert_hosts,
    "last_action_success": _feature_last_action_success,
    "step_bucket": _feature_step_bucket,
    "action_space_bucket": _feature_action_space_bucket,
}


def sig_to_bucket_id(sig: Optional[str]) -> Optional[str]:
    """Extract 3-feature coarse bucket from full state signature.

    Keeps only comp=X, alerts=Y, step=Z (drops las=Z and as=Z).
    Returns None if sig is None or 'default'.
    Max ~36 unique buckets (4 comp × 3 alerts × 3 step).
    """
    if not sig or sig == "default":
        return None
    parts = {k: v for kv in sig.split("|") for k, v in [kv.split("=", 1)] if "=" in kv}
    kept = {k: parts[k] for k in ("comp", "alerts", "step") if k in parts}
    # Coarsen alerts: collapse 3/4/4p → "3p"
    if kept.get("alerts") in ("3", "4", "4p"):
        kept["alerts"] = "3p"
    if len(kept) < 3:
        return None
    return f"comp={kept['comp']}|alerts={kept['alerts']}|step={kept['step']}"


def compute_state_signature(
    observation: Any,
    *,
    last_action_status: Any = None,
    step_idx: int = 0,
    action_space: Any = None,
    action_space_size: Optional[int] = None,
    config: Optional[StateSignatureConfig] = None,
) -> str:
    """Compute a deterministic, bucketed state signature from an observation.

    This is intentionally lightweight and stable across runs. It avoids heavy ML and keeps
    the representation small so it can be used as a key for state-conditioned priors.
    """
    cfg = config or StateSignatureConfig()
    obs_map: Mapping[str, Any] = observation if isinstance(observation, Mapping) else {}
    as_size = int(action_space_size) if action_space_size is not None else _count_available_actions(action_space)

    ctx = _SignatureContext(
        observation=obs_map,
        last_action_status=last_action_status,
        step_idx=int(step_idx) if step_idx is not None else 0,
        action_space_size=as_size,
    )

    parts: List[str] = []
    for key in cfg.features:
        fn = FEATURE_REGISTRY.get(key)
        if fn is None:
            continue
        token = fn(ctx, cfg)
        if token:
            parts.append(token)
    return "|".join(parts) if parts else "default"

