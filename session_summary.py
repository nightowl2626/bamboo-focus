"""Deterministic end-of-session summary for FlowPilot.

Reads recent posture Qwen analyses and nudge decisions, filters them to the
session window, and writes a short paragraph in the user's selected tone.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value)
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def tail_jsonl(path: Path, max_lines: int = 500) -> list[dict[str, Any]]:
    if not path.exists() or max_lines <= 0:
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    records = []
    for line in lines:
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 90,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json", **headers},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def in_window(record: dict[str, Any], started_at: datetime | None, ended_at: datetime | None) -> bool:
    created = parse_time(record.get("created_at") or record.get("received_at") or record.get("_received_at"))
    if created is None:
        return False
    if started_at and created < started_at:
        return False
    if ended_at and created > ended_at:
        return False
    return True


def count_label(items: list[dict[str, Any]], path: tuple[str, ...], label: str) -> int:
    count = 0
    for item in items:
        value: Any = item
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if value == label:
            count += 1
    return count


def summarize_session(
    posture_analyses: list[dict[str, Any]],
    nudge_decisions: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    true_nudges = [
        item
        for item in nudge_decisions
        if isinstance(item.get("decision"), dict) and item["decision"].get("should_nudge")
    ]
    categories: dict[str, int] = {}
    for item in true_nudges:
        category = str(item.get("decision", {}).get("category") or "other")
        categories[category] = categories.get(category, 0) + 1

    slouching = count_label(posture_analyses, ("analysis", "posture"), "slouching")
    restless = count_label(posture_analyses, ("analysis", "stillness_or_restlessness"), "restless")
    too_still = count_label(posture_analyses, ("analysis", "stillness_or_restlessness"), "too_still")
    hyperfocus = count_label(posture_analyses, ("analysis", "behaviour"), "possibly_hyperfocused")
    baseline_like = count_label(posture_analyses, ("analysis", "posture"), "baseline_like")

    focus_minutes = int(settings.get("pomodoro_minutes") or 25)
    break_minutes = int(settings.get("break_minutes") or 5)
    ratio = round(break_minutes / focus_minutes, 2) if focus_minutes else 0

    if restless >= 2 and focus_minutes > 30:
        ratio_note = "The focus block may be a bit long for a restless stretch; a shorter focus block or a slightly longer break could fit better."
    elif too_still + hyperfocus >= 2 and ratio < 0.2:
        ratio_note = "The break looks a little short for how still the session became; a longer movement break would probably help."
    elif not true_nudges and baseline_like >= max(1, len(posture_analyses) // 2):
        ratio_note = "The focus and break length looked reasonable for this session."
    else:
        ratio_note = "The focus and break length looked usable, but keep an eye on whether the same pattern repeats."

    return {
        "analysis_count": len(posture_analyses),
        "nudge_count": len(true_nudges),
        "nudge_categories": categories,
        "slouching": slouching,
        "restless": restless,
        "too_still": too_still,
        "hyperfocus": hyperfocus,
        "baseline_like": baseline_like,
        "focus_minutes": focus_minutes,
        "break_minutes": break_minutes,
        "break_focus_ratio": ratio,
        "ratio_note": ratio_note,
    }


def build_paragraph(stats: dict[str, Any], settings: dict[str, Any]) -> str:
    tone = str(settings.get("tone") or "neutral")
    intent = str(settings.get("intent") or "").strip()
    focus_text = f" on {intent}" if intent else ""
    nudge_count = stats["nudge_count"]
    analysis_count = stats["analysis_count"]

    if analysis_count == 0:
        base = (
            f"Session wrap-up{focus_text}: there was not enough posture analysis to say much yet. "
            f"{nudge_count} nudge{'s' if nudge_count != 1 else ''} came up, and the chosen "
            f"{stats['focus_minutes']}/{stats['break_minutes']} minute focus-break rhythm can stay as a starting point."
        )
    else:
        signals = []
        if stats["slouching"]:
            signals.append("some slouching")
        if stats["restless"]:
            signals.append("restlessness")
        if stats["too_still"] or stats["hyperfocus"]:
            signals.append("very still focus")
        if not signals:
            signals.append("mostly steady posture")
        base = (
            f"Session wrap-up{focus_text}: concentration looked {'fairly steady' if nudge_count <= 1 else 'interrupted in a few places'}, "
            f"with {', '.join(signals)} showing up. You received {nudge_count} useful nudge"
            f"{'s' if nudge_count != 1 else ''}. {stats['ratio_note']}"
        )

    if tone == "calm":
        return f"{base} For the next round, keep the adjustment small and easy."
    if tone == "encouraging":
        return f"{base} Overall, this is useful feedback: adjust one thing next time and keep going."
    if tone == "funny":
        return f"{base} Tiny debrief: your desk session left a few clues, and the next round gets to be slightly smarter."
    if tone == "strict":
        return f"{base} Next session, act on the main pattern early instead of waiting for it to repeat."
    return base


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


def compact_posture_analyses(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in items[-20:]:
        analysis = item.get("analysis") if isinstance(item.get("analysis"), dict) else {}
        compact.append(
            {
                "created_at": item.get("created_at"),
                "judgement": analysis.get("judgement"),
                "posture": analysis.get("posture"),
                "behaviour": analysis.get("behaviour"),
                "stillness_or_restlessness": analysis.get("stillness_or_restlessness"),
                "observations": analysis.get("observations", []),
            }
        )
    return compact


def compact_nudges(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for item in items[-30:]:
        decision = item.get("decision") if isinstance(item.get("decision"), dict) else {}
        compact.append(
            {
                "created_at": item.get("created_at"),
                "should_nudge": decision.get("should_nudge"),
                "category": decision.get("category"),
                "recommended_focus": decision.get("recommended_focus"),
                "rationale": decision.get("rationale"),
                "suppress_reason": decision.get("suppress_reason"),
            }
        )
    return compact


def call_qwen_for_session_summary(
    stats: dict[str, Any],
    settings: dict[str, Any],
    posture_analyses: list[dict[str, Any]],
    nudge_decisions: list[dict[str, Any]],
    started_at: str | None,
    ended_at: str | None,
) -> str:
    api_base = os.getenv("QWEN_API_BASE", DEFAULT_QWEN_API_BASE).rstrip("/")
    api_key = os.getenv("QWEN_API_KEY")
    model = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set")

    tone = str(settings.get("tone") or "neutral")
    payload_for_model = {
        "session": {
            "started_at": started_at,
            "ended_at": ended_at,
            "intent": settings.get("intent", ""),
            "tone": tone,
            "notification_level": settings.get("notification_level"),
            "focus_areas": settings.get("focus_areas", []),
            "work_struggles": settings.get("work_struggles", {}),
            "focus_minutes": settings.get("pomodoro_minutes"),
            "break_minutes": settings.get("break_minutes"),
        },
        "stats": stats,
        "posture_analyses": compact_posture_analyses(posture_analyses),
        "nudge_decisions": compact_nudges(nudge_decisions),
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You write concise end-of-focus-session debriefs for an ADHD/productivity assistant. "
                "You are not making a nudge decision and you are not diagnosing. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Write one short paragraph, 70-120 words, in the user's chosen tone. "
                "Summarize how the session went, concentration level, posture/restlessness/hyperfocus patterns, "
                "what to watch out for next time, and whether the chosen focus/break ratio seemed appropriate. "
                "Use plain language. Do not mention raw data, JSON, metrics, sensor internals, or that you are reading analyses. "
                "Do not overstate weak evidence. If there was not enough information, say that gently.\n\n"
                "Tone guidance: neutral = direct and factual; calm = soft and grounding; encouraging = supportive; "
                "funny = lightly witty but still useful; strict = concise and no-nonsense.\n\n"
                'Return exactly: {"paragraph": "string"}\n\n'
                f"Session information:\n{json.dumps(payload_for_model, indent=2)}"
            ),
        },
    ]
    response = request_json(
        "POST",
        f"{api_base}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Qwen response shape: {response}") from exc
    parsed = parse_json_content(content)
    paragraph = str(parsed.get("paragraph") or "").strip()
    if not paragraph:
        raise RuntimeError("Qwen summary paragraph was empty")
    return paragraph


def build_session_summary(
    monitor_data_dir: Path,
    agent_data_dir: Path,
    settings: dict[str, Any],
    started_at: str | None,
    ended_at: str | None,
    use_qwen: bool = True,
) -> dict[str, Any]:
    started = parse_time(started_at)
    ended = parse_time(ended_at) or datetime.now(timezone.utc)
    posture = [
        item
        for item in tail_jsonl(monitor_data_dir / "qwen_analyses.jsonl", 800)
        if in_window(item, started, ended)
    ]
    nudges = [
        item
        for item in tail_jsonl(agent_data_dir / "nudge_decisions.jsonl", 800)
        if in_window(item, started, ended)
    ]
    stats = summarize_session(posture, nudges, settings)
    qwen_error = None
    if use_qwen:
        try:
            paragraph = call_qwen_for_session_summary(
                stats=stats,
                settings=settings,
                posture_analyses=posture,
                nudge_decisions=nudges,
                started_at=started_at,
                ended_at=ended_at,
            )
        except Exception as exc:
            qwen_error = str(exc)
            paragraph = build_paragraph(stats, settings)
    else:
        qwen_error = "Qwen session summary skipped in local mode."
        paragraph = build_paragraph(stats, settings)
    return {
        "created_at": utc_now(),
        "session": {
            "started_at": started_at,
            "ended_at": ended_at,
        },
        "settings": settings,
        "stats": stats,
        "paragraph": paragraph,
        "qwen_error": qwen_error,
    }


def write_session_summary(summary: dict[str, Any], agent_data_dir: Path) -> Path:
    agent_data_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = agent_data_dir / "session_summaries.jsonl"
    latest_path = agent_data_dir / "latest_session_summary.json"
    with summaries_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary) + "\n")
    latest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return latest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write a deterministic FlowPilot session summary.")
    parser.add_argument("--monitor-data-dir", default="monitor_data")
    parser.add_argument("--agent-data-dir", default="nudge_agent_data")
    parser.add_argument("--settings", default="nudge_agent_data/session_settings.json")
    parser.add_argument("--started-at")
    parser.add_argument("--ended-at")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    settings_path = Path(args.settings)
    settings = json.loads(settings_path.read_text(encoding="utf-8")) if settings_path.exists() else {}
    summary = build_session_summary(
        monitor_data_dir=Path(args.monitor_data_dir),
        agent_data_dir=Path(args.agent_data_dir),
        settings=settings,
        started_at=args.started_at or settings.get("started_at"),
        ended_at=args.ended_at or utc_now(),
    )
    output_path = write_session_summary(summary, Path(args.agent_data_dir))
    print(json.dumps({"ok": True, "path": str(output_path), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
