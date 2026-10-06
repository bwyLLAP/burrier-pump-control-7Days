from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


SENSITIVE_KEY_FRAGMENTS = ("api_key", "secret", "token", "authorization", "password")


def _redact(value: Any, key: str = "") -> Any:
    if any(fragment in key.lower() for fragment in SENSITIVE_KEY_FRAGMENTS):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _redact(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, (datetime, Path)):
        return str(value)
    return value


class AuditLog:
    def __init__(self, path: Path, session_id: str) -> None:
        self.path = Path(path)
        self.session_id = session_id

    def record(
        self,
        *,
        event_type: str,
        actor: str,
        payload: dict[str, Any],
        prior_fingerprint: str | None = None,
        current_fingerprint: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "event_id": str(uuid4()),
            "session_id": self.session_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "actor": actor,
            "prior_fingerprint": prior_fingerprint,
            "current_fingerprint": current_fingerprint,
            "payload": _redact(payload),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event
