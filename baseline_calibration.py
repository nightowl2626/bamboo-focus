"""Run a calibration session and save the result as baseline.json.

This script expects laptop_calibration_server.py to already be running and the
Raspberry Pi edge loop to be polling it. It starts a timed calibration session,
waits for completion, asks Qwen to classify detected objects, and writes a
baseline file containing the raw calibration data plus Qwen's object policy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qwen_config import qwen_model_for

DEFAULT_SERVER_BASE = os.getenv("FLOWPILOT_CALIBRATION_SERVER", "http://127.0.0.1:8000")
DEFAULT_BASELINE_PATH = "baseline.json"
DEFAULT_QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
OBJECT_EVENT_TYPES = {
    "object_capture_snapshot",
    "object_calibration_snapshot",
    "object_surveillance_snapshot",
}
LOCAL_WHITELIST_LABELS = {
    "person",
    "laptop",
    "computer",
    "screen",
    "monitor",
    "keyboard",
    "mouse",
    "desk",
    "chair",
    "table",
    "tv",
    "bookcase",
    "shelf",
    "plant",
    "potted plant",
    "clock",
}
LOCAL_MONITOR_LABELS = {
    "cell phone",
    "phone",
    "mobile phone",
    "cup",
    "mug",
    "bottle",
    "water bottle",
    "paper",
    "book",
    "notebook",
    "plate",
    "bowl",
    "snack",
    "remote",
    "headphones",
    "backpack",
    "bag",
}


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 15,
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_headers = {"Accept": "application/json"}
    if body is not None:
        request_headers["Content-Type"] = "application/json"
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def start_calibration(server_base: str, seconds: int, token: str | None) -> dict[str, Any]:
    payload: dict[str, Any] = {"seconds": seconds}
    if token:
        payload["token"] = token
    return request_json("POST", f"{server_base.rstrip('/')}/api/start", payload)


def fetch_status(server_base: str) -> dict[str, Any]:
    return request_json("GET", f"{server_base.rstrip('/')}/api/status")


def wait_for_completion(
    server_base: str,
    seconds: int,
    poll_interval: float,
    settle_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + seconds + settle_seconds + 30
    snapshot = fetch_status(server_base)
    while time.monotonic() < deadline:
        snapshot = fetch_status(server_base)
        session = snapshot.get("session") if isinstance(snapshot.get("session"), dict) else None
        if session and not session.get("active"):
            time.sleep(settle_seconds)
            return fetch_status(server_base)
        time.sleep(poll_interval)
    raise RuntimeError("Calibration did not finish before timeout")


def object_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for event in events if event.get("event_type") in OBJECT_EVENT_TYPES]


def compact_objects(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, dict[str, Any]] = {}
    for event in object_events(events):
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        for item in payload.get("objects", []):
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "unknown")).strip() or "unknown"
            score = as_float(item.get("score"))
            current = by_label.get(label)
            if current is None:
                by_label[label] = {
                    "label": label,
                    "count": 1,
                    "best_score": score,
                    "bboxes": [item.get("bbox")],
                    "event_sequences": [event.get("_sequence")],
                }
            else:
                current["count"] += 1
                current["best_score"] = max(current["best_score"], score)
                current["bboxes"].append(item.get("bbox"))
                current["event_sequences"].append(event.get("_sequence"))
    return sorted(by_label.values(), key=lambda item: (-item["best_score"], item["label"]))


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().split())


def classify_objects_locally(objects: list[dict[str, Any]]) -> dict[str, Any]:
    whitelisted = []
    monitorable = []
    uncertain = []
    for item in objects:
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        normalized = normalize_label(label)
        if normalized in LOCAL_WHITELIST_LABELS:
            whitelisted.append({"label": label, "reason": "Common fixed or expected work-setting object."})
        elif normalized in LOCAL_MONITOR_LABELS:
            monitorable.append(
                {
                    "label": label,
                    "reason": "Potential distraction or left-behind desk item.",
                    "priority": "medium" if normalized in {"cell phone", "phone", "mobile phone"} else "low",
                }
            )
        else:
            uncertain.append({"label": label, "reason": "Not recognized by local offline policy."})
    return {
        "whitelisted_objects": whitelisted,
        "monitorable_objects": monitorable,
        "uncertain_objects": uncertain,
        "notes": "Deterministic local object policy. Qwen was skipped or unavailable.",
    }


def classify_objects_with_qwen(objects: list[dict[str, Any]], context: str | None) -> dict[str, Any]:
    api_base = os.getenv("QWEN_API_BASE", DEFAULT_QWEN_API_BASE).rstrip("/")
    api_key = os.getenv("QWEN_API_KEY")
    model = qwen_model_for("calibration", DEFAULT_QWEN_MODEL)
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set")

    object_lines = [
        {
            "label": item["label"],
            "count": item["count"],
            "best_score": round(item["best_score"], 3),
            "bboxes": item["bboxes"][:3],
        }
        for item in objects
    ]
    user_context = context or "Desk/work environment calibration for focus and distraction monitoring."
    messages = [
        {
            "role": "system",
            "content": (
                "You classify desk object detections for a focus-monitoring baseline. "
                "Return only valid JSON. No markdown, no prose outside JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Given detected object labels from a baseline desk calibration, decide which objects "
                "are normal background/work-setting objects to whitelist and which objects should be "
                "monitored because they may indicate left-behind items or distractions.\n\n"
                "Whitelist examples: person, laptop, screen, monitor, keyboard, mouse, desk, chair, "
                "wall decorations, plants, normal room decor, fixed furniture.\n"
                "Monitor examples: cup, mug, bottle, phone, paper, scrap paper, book, snack, plate, "
                "remote, headphones, miscellaneous clutter.\n\n"
                "Use the labels as detector labels; do not invent objects not present. If a generic "
                "label could be either background or distraction, choose the most useful monitoring "
                "policy for a work desk and explain briefly.\n\n"
                f"Context: {user_context}\n\n"
                "Return this exact JSON shape:\n"
                "{\n"
                '  "whitelisted_objects": [{"label": "string", "reason": "string"}],\n'
                '  "monitorable_objects": [{"label": "string", "reason": "string", "priority": "low|medium|high"}],\n'
                '  "uncertain_objects": [{"label": "string", "reason": "string"}],\n'
                '  "notes": "string"\n'
                "}\n\n"
                f"Detected objects:\n{json.dumps(object_lines, indent=2)}"
            ),
        },
    ]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
    }
    response = request_json(
        "POST",
        f"{api_base}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Qwen response shape: {response}") from exc
    policy = parse_json_content(content)
    validate_policy(policy)
    return policy


def parse_json_content(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise RuntimeError(f"Qwen did not return JSON: {content}")
        parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise RuntimeError("Qwen JSON response must be an object")
    return parsed


def validate_policy(policy: dict[str, Any]) -> None:
    for key in ("whitelisted_objects", "monitorable_objects", "uncertain_objects"):
        if key not in policy or not isinstance(policy[key], list):
            policy[key] = []
    if "notes" not in policy or not isinstance(policy["notes"], str):
        policy["notes"] = ""


def build_baseline(snapshot: dict[str, Any], qwen_policy: dict[str, Any], qwen_error: str | None = None) -> dict[str, Any]:
    events = snapshot.get("events") if isinstance(snapshot.get("events"), list) else []
    objects = compact_objects(events)
    baseline = {
        "kind": "flowpilot_baseline",
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "calibration": {
            "session": snapshot.get("session"),
            "summary": snapshot.get("summary"),
            "commands": snapshot.get("commands"),
            "events": events,
        },
        "object_detection": {
            "objects": objects,
            "source_event_count": len(object_events(events)),
        },
        "object_policy": qwen_policy,
    }
    if qwen_error:
        baseline["object_policy_error"] = qwen_error
    return baseline


def write_baseline(path: str | Path, baseline: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibration and save baseline data with Qwen object policy.")
    parser.add_argument("--server", default=DEFAULT_SERVER_BASE, help="Calibration server base URL.")
    parser.add_argument("--seconds", type=int, default=30, help="Calibration duration in seconds.")
    parser.add_argument("--token", default=os.getenv("FLOWPILOT_PI_TOKEN"), help="Token to assign to the session.")
    parser.add_argument("--output", default=DEFAULT_BASELINE_PATH, help="Baseline JSON output path.")
    parser.add_argument("--context", help="Extra context for Qwen about the room/desk setup.")
    parser.add_argument("--poll-interval", type=float, default=1.0, help="Seconds between status polls.")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="Extra wait after calibration ends.")
    parser.add_argument("--skip-qwen", action="store_true", help="Save baseline without calling Qwen.")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    if args.seconds < 1:
        raise SystemExit("--seconds must be at least 1")

    server_base = args.server.rstrip("/")
    print(f"Starting calibration for {args.seconds}s via {server_base}")
    start_calibration(server_base, args.seconds, args.token)
    snapshot = wait_for_completion(server_base, args.seconds, args.poll_interval, args.settle_seconds)
    events = snapshot.get("events") if isinstance(snapshot.get("events"), list) else []
    objects = compact_objects(events)
    print(f"Calibration complete: {len(events)} events, {len(objects)} object labels")

    qwen_policy: dict[str, Any] = classify_objects_locally(objects)
    qwen_error = None
    if not args.skip_qwen:
        try:
            qwen_policy = classify_objects_with_qwen(objects, args.context)
            print(
                "Qwen policy complete: "
                f"{len(qwen_policy['whitelisted_objects'])} whitelisted, "
                f"{len(qwen_policy['monitorable_objects'])} monitorable"
            )
        except Exception as exc:
            qwen_error = str(exc)
            print(f"Qwen policy failed: {qwen_error}")
            print(
                "Using local policy: "
                f"{len(qwen_policy['whitelisted_objects'])} whitelisted, "
                f"{len(qwen_policy['monitorable_objects'])} monitorable"
            )

    baseline = build_baseline(snapshot, qwen_policy, qwen_error)
    output_path = write_baseline(args.output, baseline)
    print(f"Saved baseline: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
