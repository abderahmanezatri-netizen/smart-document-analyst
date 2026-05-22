from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any

LOG_PATH = Path("outputs/logs/agent_actions.jsonl")


def log_action(actor: str, action: str, status: str, metadata: dict[str, Any] | None = None, log_path: Path = LOG_PATH) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "status": status,
        "metadata": metadata or {},
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_log_tail(log_path: Path = LOG_PATH, tail: int = 20) -> str:
    if not log_path.exists():
        return "No log file found yet."
    lines = log_path.read_text(encoding="utf-8").splitlines()[-tail:]
    return "\n".join(lines)
