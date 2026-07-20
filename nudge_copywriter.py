"""Turn factual nudge decisions into user-facing notification copy.

This is intentionally not an agent. It makes one deterministic Qwen call from a
saved factual decision record and writes the notification text separately.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from nudge import (
    DEFAULT_QWEN_API_BASE,
    DEFAULT_QWEN_MODEL,
    append_jsonl,
    load_dotenv,
    parse_json_content,
    read_json,
    request_json,
    utc_now,
)
from qwen_config import qwen_model_for


DEFAULT_COPY = {
    "title": "Quick check-in",
    "message": "Hey, a quick reset might help right now.",
    "tone": "friendly",
    "category": "other",
}


def normalize_copy(copy: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_COPY)
    normalized.update({key: value for key, value in copy.items() if key in normalized})
    for key in ("title", "message", "tone", "category"):
        if not isinstance(normalized[key], str):
            normalized[key] = str(normalized[key])
        normalized[key] = normalized[key].strip()
    if not normalized["message"]:
        normalized["message"] = DEFAULT_COPY["message"]
    if len(normalized["message"]) > 180:
        normalized["message"] = normalized["message"][:177].rstrip() + "..."
    return normalized


def fallback_copy(decision: dict[str, Any], user_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    category = decision.get("category")
    focus = str(decision.get("recommended_focus") or "").strip()
    tone = str((user_settings or {}).get("tone") or "friendly")
    if category == "posture":
        message = "Hey, looks like your posture could use a reset. How about sitting up and relaxing your shoulders?"
    elif category == "break":
        message = "Hey, looks like you may be locked in. How about a quick break before jumping back in?"
    elif category == "restlessness":
        message = "Hey, there seems to be a lot of shifting. Want to pause for a short reset?"
    elif category in {"cleanup", "object"}:
        message = f"Hey, quick desk check: {focus or 'there may be something worth clearing'}."
    else:
        message = f"Hey, quick check-in: {focus or 'a small reset might help right now'}."
    return normalize_copy(
        {
            "title": "Quick check-in",
            "message": message,
            "tone": tone,
            "category": category or "other",
        }
    )


def call_qwen_copywriter(
    decision_record: dict[str, Any],
    user_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_base = os.getenv("QWEN_API_BASE", DEFAULT_QWEN_API_BASE).rstrip("/")
    api_key = os.getenv("QWEN_API_KEY")
    model = qwen_model_for("copywriter", DEFAULT_QWEN_MODEL)
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set")

    decision = decision_record.get("decision") if isinstance(decision_record.get("decision"), dict) else {}
    settings = user_settings or decision_record.get("user_settings") or {}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a friendly ADHD productivity notification copywriter. "
                    "Convert factual nudge decisions into one short, kind, user-facing notification. "
                    "Do not mention sensors, data, detection, Qwen, logs, confidence, or raw metrics. "
                    "Do not shame, diagnose, or overstate. Return only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write the actual notification copy from this factual nudge decision. "
                    "Keep it brief, warm, and practical. It should be one notification, not a paragraph. "
                    "Use the category and recommended_focus, but do not repeat the factual rationale mechanically.\n\n"
                    "Apply these user preferences if present:\n"
                    "- tone controls voice: neutral, funny, strict, calm, encouraging.\n"
                    "- intent gives work-session context.\n"
                    "- focus_areas indicate what the user explicitly cares about.\n\n"
                    "Return this exact JSON shape:\n"
                    "{\n"
                    '  "title": "short notification title",\n'
                    '  "message": "friendly notification under 180 characters",\n'
                    '  "tone": "friendly|calm|encouraging|direct",\n'
                    '  "category": "same category as decision"\n'
                    "}\n\n"
                    f"User settings:\n{json.dumps(settings, indent=2)}\n\n"
                    f"Decision record:\n{json.dumps(decision_record, indent=2)}"
                ),
            },
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    response = request_json(
        "POST",
        f"{api_base}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Qwen copywriter response shape: {response}") from exc
    copy = normalize_copy(parse_json_content(content))
    if decision.get("category") and copy.get("category") != decision.get("category"):
        copy["category"] = str(decision.get("category"))
    return copy


def write_notification(
    decision_record: dict[str, Any],
    output_dir: str | Path = "nudge_agent_data",
    user_settings: dict[str, Any] | None = None,
    use_qwen: bool = True,
) -> dict[str, Any]:
    decision = decision_record.get("decision") if isinstance(decision_record.get("decision"), dict) else {}
    if not decision.get("should_nudge"):
        raise ValueError("decision_record does not request a nudge")

    copy_error = None
    if use_qwen:
        try:
            copy = call_qwen_copywriter(decision_record, user_settings)
        except Exception as exc:
            copy_error = str(exc)
            copy = fallback_copy(decision, user_settings)
    else:
        copy = fallback_copy(decision, user_settings)

    record = {
        "created_at": utc_now(),
        "decision_created_at": decision_record.get("created_at"),
        "decision": decision,
        "notification": copy,
        "user_settings": user_settings or decision_record.get("user_settings") or {},
        "copywriter_error": copy_error,
    }
    out_dir = Path(output_dir)
    append_jsonl(out_dir / "notification_messages.jsonl", record)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latest_notification.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert latest factual nudge decision into notification copy.")
    parser.add_argument("--decision", default="nudge_agent_data/latest_nudge_decision.json", help="Decision record path.")
    parser.add_argument("--output-dir", default="nudge_agent_data", help="Where notification copy records are written.")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    decision_record = read_json(Path(args.decision))
    if decision_record is None:
        raise SystemExit(f"Could not read decision record: {args.decision}")
    try:
        notification = write_notification(decision_record, args.output_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(notification, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
