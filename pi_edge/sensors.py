from datetime import datetime, timezone
from typing import Any


def semantic_event(
    event_type: str,
    payload: dict[str, Any],
    session_id: str | None = None,
    source: str = "raspberry_pi_camera",
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "source": source,
        "event_type": event_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": payload,
    }


def presence_started(confidence: float = 0.94, source: str = "raspberry_pi_camera") -> dict[str, Any]:
    return semantic_event("presence_started", {"confidence": confidence}, source=source)


def posture_window(payload: dict[str, Any], source: str = "raspberry_pi_camera") -> dict[str, Any]:
    return semantic_event("posture_window", payload, source=source)


def object_capture_snapshot(payload: dict[str, Any], source: str = "raspberry_pi_camera") -> dict[str, Any]:
    return semantic_event("object_capture_snapshot", payload, source=source)
