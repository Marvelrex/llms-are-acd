from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List

from CybORG.Agents.LLMAgents.llm_adapter.utils.logger import Logger

# NOTE: Keep phrasing neutral to avoid permanently priming the LLM into an "incident narrative".
WARNING_MSG = "WARNING: Unusual process pattern detected"
ALERT_MSG = "ALERT: High-risk indicator observed"
MAXIMUM_ALERT_MSG = "CRITICAL: Strong risk indicator observed"


def _to_binary_list(bits: Any) -> List[int]:
    try:
        return [1 if bool(x) else 0 for x in bits]
    except Exception:
        return []


def _format_comm_vector_message(agent_name: str, commvectors: Any) -> List[str]:
    """Format the commvector messages for the LLM (best-effort)."""
    if not isinstance(commvectors, (list, tuple)):
        return []
    try:
        agent_number = int(str(agent_name)[-1])
        agent_indices = [i for i in range(5) if i != agent_number]
    except Exception:
        agent_indices = list(range(len(commvectors)))

    text_obs: List[str] = []
    for idx, c in zip(agent_indices, commvectors):
        text_obs.append(f"Commvector Blue Agent {idx} Message: {_to_binary_list(c)}")
    return text_obs


def _format_suspicious_activity(observation: Dict[str, Any]) -> List[str]:
    """Return a compact list of suspicious host summaries (do not list every host)."""
    suspicious_activity: List[str] = []

    for key, value in observation.items():
        if key in {"success", "action", "phase", "message"}:
            continue
        if not isinstance(value, dict):
            continue

        hostname = key
        sysinfo = value.get("System info")
        if isinstance(sysinfo, dict):
            hostname = sysinfo.get("Hostname", key) or key

        ip = None
        iface = value.get("Interface")
        if isinstance(iface, list) and iface:
            ip = iface[0].get("ip_address")

        events: List[str] = []

        # Processes / connections
        conn_counter: Counter[str] = Counter()
        procs = value.get("Processes")
        if isinstance(procs, list):
            for proc in procs:
                if not isinstance(proc, dict):
                    continue
                if "PID" in proc and "username" not in proc:
                    events.append(WARNING_MSG)
                conns = proc.get("Connections")
                if isinstance(conns, list):
                    for conn in conns:
                        if not isinstance(conn, dict):
                            continue
                        remote_addr = conn.get("remote_address")
                        if remote_addr:
                            conn_counter[str(remote_addr)] += 1

        if conn_counter:
            # Show up to a few distinct remotes; keep the prompt small.
            max_addrs = 3
            entries = sorted(conn_counter.items(), key=lambda kv: (-kv[1], kv[0]))
            shown = entries[:max_addrs]
            rest = len(entries) - len(shown)
            parts = [f"{addr} (x{cnt})" if cnt > 1 else addr for addr, cnt in shown]
            if rest > 0:
                parts.append(f"+{rest} more")
            events.append(f"INFO: Connections to {', '.join(parts)}")

        # File IOCs
        ioc_files_user = {"cmd.sh", "cmd.exe"}
        ioc_files_admin = {"escalate.sh", "escalate.exe"}
        files = value.get("Files")
        if isinstance(files, list):
            for f in files:
                if not isinstance(f, dict):
                    continue
                fname = f.get("File Name")
                if fname in ioc_files_admin:
                    events.append(MAXIMUM_ALERT_MSG)
                elif fname in ioc_files_user:
                    events.append(ALERT_MSG)

        # Only include hosts with actual events (avoid listing all hosts with just hostname/IP).
        if not events:
            continue

        host_parts = [f"Hostname: {hostname}"]
        if ip:
            host_parts.append(f"IP: {ip}")
        host_parts.extend(events)
        suspicious_activity.append(" | ".join(host_parts))

    # Hard cap to avoid blowing up prompt size if many hosts are noisy.
    max_hosts = 12
    if len(suspicious_activity) > max_hosts:
        extra = len(suspicious_activity) - max_hosts
        suspicious_activity = suspicious_activity[:max_hosts] + [f"... (+{extra} more hosts)"]

    return suspicious_activity


def format_observation(observation: Dict[str, Any], last_action: Any, agent_name: str) -> str:
    """Format the observation for the LLM."""
    if observation is None:
        return "No observation available. Choose a defensive action."

    Logger.debug("Received observation")
    text_obs: List[str] = []

    phase = observation.get("phase", "unknown")
    success = observation.get("success", "unknown")
    text_obs.append(f"Mission Phase: {phase}")
    text_obs.append(f"Last Action : {last_action}")
    text_obs.append(f"Last Action Status: {success}")

    text_obs.append("Communication Vectors:")
    text_obs.extend(_format_comm_vector_message(agent_name, observation.get("message")))

    suspicious_activity = _format_suspicious_activity(observation)
    if suspicious_activity:
        text_obs.append(f"Risk Signals: {len(suspicious_activity)} host(s) flagged")
        text_obs.extend(f"- {activity}" for activity in suspicious_activity)
    else:
        text_obs.append("Risk Signals: None")
    text_obs.append(
        "Note: Traffic block/allow actions can disrupt normal user operations and incur heavy penalties; only choose them when clear evidence of malicious or risky traffic is present."
    )

    formatted_obs = "# OBSERVATION\n\n" + "\n".join(text_obs)
    Logger.debug(f"Formatted observation:\n{formatted_obs}\n")
    return formatted_obs
