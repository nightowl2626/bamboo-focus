"""Privacy ledger, explainability, and memory helpers for Bamboo Focus.

These helpers intentionally persist only compact derived records. They never
store images, video frames, or full sensor streams in the agent memory layer.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def tail_jsonl(path: Path, max_lines: int = 100) -> list[dict[str, Any]]:
    if not path.exists() or max_lines <= 0:
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-max_lines:]
    except OSError:
        return []
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload) + "\n")


def _path_string(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_privacy_ledger(paths: Any, nudge_mode: str, runtime_status: dict[str, Any] | None = None) -> dict[str, Any]:
    latest_decision = read_json(paths.latest_decision_path) or {}
    qwen_error = latest_decision.get("qwen_error")
    retrieval = latest_decision.get("retrieval") if isinstance(latest_decision.get("retrieval"), dict) else {}
    dynamic_calls = retrieval.get("dynamic_tool_calls") if isinstance(retrieval.get("dynamic_tool_calls"), list) else []
    cloud_attempted = nudge_mode != "local" and latest_decision != {}
    cloud_succeeded = cloud_attempted and qwen_error is None and bool(dynamic_calls)
    return {
        "updated_at": utc_now(),
        "track_positioning": "EdgeAgent: hardware perception stays on the Pi/laptop edge; cloud reasoning sees compact JSON only.",
        "hardware_first": {
            "edge_devices": ["Raspberry Pi camera", "laptop webcam fallback"],
            "local_processing": [
                "MediaPipe pose landmarks",
                "MediaPipe object detection",
                "motion metrics",
                "object whitelist filtering",
            ],
            "raw_video_sent_off_device": False,
            "raw_frames_persisted_by_backend": False,
        },
        "data_boundaries": {
            "edge_to_backend": [
                "posture window metrics",
                "object labels, scores, and bounding boxes",
                "presence and command polling metadata",
            ],
            "backend_to_cloud": (
                [
                    "bounded posture summaries",
                    "filtered object dwell summaries",
                    "baseline object policy",
                    "recent nudge history",
                    "session preferences",
                ]
                if nudge_mode != "local"
                else []
            ),
            "cloud_provider": "Alibaba Cloud DashScope/Qwen when mode is auto or qwen",
            "cloud_attempted_for_latest_decision": cloud_attempted,
            "cloud_succeeded_for_latest_decision": cloud_succeeded,
            "latest_cloud_error": qwen_error,
        },
        "local_runtime_paths": {
            "baseline": _path_string(paths.baseline),
            "posture_events": _path_string(paths.long_events_path),
            "posture_analyses": _path_string(paths.long_analyses_path),
            "object_events": _path_string(paths.object_events_path),
            "nudge_decisions": _path_string(paths.decisions_path),
            "agent_memory": _path_string(paths.agent_data_dir / "agent_memory.json"),
            "privacy_ledger": _path_string(paths.agent_data_dir / "privacy_ledger.json"),
        },
        "retention_model": {
            "source_of_truth": "local JSON/JSONL files under gitignored runtime data directories",
            "memory_scope": "compact session-level patterns only",
            "excluded_from_memory": ["raw video", "camera frames", "full calibration event streams"],
        },
        "runtime_status": runtime_status or {},
    }


def write_privacy_ledger(paths: Any, nudge_mode: str, runtime_status: dict[str, Any] | None = None) -> dict[str, Any]:
    ledger = build_privacy_ledger(paths, nudge_mode, runtime_status)
    write_json(paths.agent_data_dir / "privacy_ledger.json", ledger)
    return ledger


def write_decision_trace(
    paths: Any,
    record: dict[str, Any],
    context: dict[str, Any],
    nudge_mode: str,
    qwen_error: str | None,
) -> dict[str, Any]:
    decision = record.get("decision") if isinstance(record.get("decision"), dict) else {}
    retrieval = record.get("retrieval") if isinstance(record.get("retrieval"), dict) else {}
    posture_context = context.get("posture_context") if isinstance(context.get("posture_context"), dict) else {}
    object_context = context.get("object_context") if isinstance(context.get("object_context"), dict) else {}
    raw_summary = context.get("raw_posture_summary") if isinstance(context.get("raw_posture_summary"), dict) else {}
    memory_profile = context.get("memory_profile") if isinstance(context.get("memory_profile"), dict) else {}
    guidance = memory_profile.get("adaptive_guidance") if isinstance(memory_profile.get("adaptive_guidance"), dict) else {}
    memory_adjustments = []
    local_eval = context.get("local_rule_evaluation") if isinstance(context.get("local_rule_evaluation"), dict) else {}
    if isinstance(local_eval.get("memory_adjustments"), list):
        memory_adjustments.extend(str(item) for item in local_eval["memory_adjustments"])
    for signal in decision.get("supporting_signals", []):
        if isinstance(signal, str) and "session memory" in signal:
            memory_adjustments.append(signal)
    memory_influenced = bool(memory_adjustments) or (
        memory_profile.get("session_count", 0) >= 2
        and any(item in str(decision.get("rationale", "")).lower() for item in ("memory", "previous", "recurring"))
    )
    trace = {
        "created_at": utc_now(),
        "decision_created_at": record.get("created_at"),
        "decision": decision,
        "why": {
            "rationale": decision.get("rationale"),
            "supporting_signals": decision.get("supporting_signals", []),
            "suppress_reason": decision.get("suppress_reason"),
            "cooldown_key": decision.get("cooldown_key"),
        },
        "agent_path": {
            "mode": nudge_mode,
            "local_fallback_used": nudge_mode == "local" or (bool(qwen_error) and nudge_mode != "qwen"),
            "qwen_error": qwen_error,
            "tool_calls": retrieval.get("dynamic_tool_calls", []),
            "retrieval_strategy": retrieval.get("strategy") or context.get("retrieval_strategy"),
        },
        "edge_evidence": {
            "posture_analysis_count": len(posture_context.get("analyses", [])) if isinstance(posture_context.get("analyses"), list) else retrieval.get("posture_analysis_count", 0),
            "object_snapshot_count": len(object_context.get("snapshots", [])) if isinstance(object_context.get("snapshots"), list) else retrieval.get("object_snapshot_count", 0),
            "object_dwell_candidate_count": len(object_context.get("dwell_candidates", [])) if isinstance(object_context.get("dwell_candidates"), list) else retrieval.get("dwell_candidate_count", 0),
            "raw_posture_event_count": raw_summary.get("event_count", 0),
        },
        "memory_influence": {
            "used": memory_influenced,
            "session_count": memory_profile.get("session_count", 0),
            "adjustments": sorted(set(memory_adjustments)),
            "guidance_summary": guidance.get("summary", "No memory guidance available yet."),
            "recommendations": guidance.get("recommendations", []),
        },
        "privacy_guards": {
            "raw_video_sent_off_device": False,
            "raw_frames_sent_to_cloud": False,
            "cloud_input_shape": "compact JSON summaries and bounded tool results only",
            "decision_record_contains_raw_video": False,
        },
    }
    write_json(paths.agent_data_dir / "latest_decision_trace.json", trace)
    append_jsonl(paths.agent_data_dir / "decision_traces.jsonl", trace)
    return trace


def latest_decision_trace(paths: Any) -> dict[str, Any]:
    return read_json(paths.agent_data_dir / "latest_decision_trace.json") or {
        "created_at": None,
        "decision": None,
        "why": {"suppress_reason": "No nudge decision has been recorded yet."},
        "privacy_guards": {
            "raw_video_sent_off_device": False,
            "raw_frames_sent_to_cloud": False,
        },
    }


def remember_session_summary(paths: Any, summary: dict[str, Any]) -> dict[str, Any]:
    stats = summary.get("stats") if isinstance(summary.get("stats"), dict) else {}
    settings = summary.get("settings") if isinstance(summary.get("settings"), dict) else {}
    record = {
        "created_at": utc_now(),
        "session": summary.get("session", {}),
        "intent": settings.get("intent", ""),
        "tone": settings.get("tone", "neutral"),
        "notification_level": settings.get("notification_level", "minimal"),
        "focus_areas": settings.get("focus_areas", []),
        "stats": {
            "analysis_count": stats.get("analysis_count", 0),
            "nudge_count": stats.get("nudge_count", 0),
            "nudge_categories": stats.get("nudge_categories", {}),
            "slouching": stats.get("slouching", 0),
            "restless": stats.get("restless", 0),
            "too_still": stats.get("too_still", 0),
            "hyperfocus": stats.get("hyperfocus", 0),
            "break_focus_ratio": stats.get("break_focus_ratio"),
        },
        "paragraph": summary.get("paragraph", ""),
    }
    append_jsonl(paths.agent_data_dir / "agent_memory_events.jsonl", record)
    return build_memory_profile(paths)


def build_memory_profile(paths: Any, max_sessions: int = 30) -> dict[str, Any]:
    sessions = tail_jsonl(paths.agent_data_dir / "agent_memory_events.jsonl", max_sessions)
    category_counts: dict[str, int] = {}
    focus_area_counts: dict[str, int] = {}
    totals = {
        "analysis_count": 0,
        "nudge_count": 0,
        "slouching": 0,
        "restless": 0,
        "too_still": 0,
        "hyperfocus": 0,
    }
    for session in sessions:
        stats = session.get("stats") if isinstance(session.get("stats"), dict) else {}
        for key in totals:
            try:
                totals[key] += int(stats.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
        categories = stats.get("nudge_categories") if isinstance(stats.get("nudge_categories"), dict) else {}
        for key, value in categories.items():
            try:
                category_counts[str(key)] = category_counts.get(str(key), 0) + int(value or 0)
            except (TypeError, ValueError):
                pass
        for area in session.get("focus_areas", []):
            focus_area_counts[str(area)] = focus_area_counts.get(str(area), 0) + 1
    adaptive_guidance = build_adaptive_guidance(len(sessions), totals, category_counts)
    profile = {
        "updated_at": utc_now(),
        "session_count": len(sessions),
        "memory_scope": "Compact session summaries only; no camera frames or raw video.",
        "totals": totals,
        "nudge_category_counts": dict(sorted(category_counts.items())),
        "focus_area_counts": dict(sorted(focus_area_counts.items())),
        "adaptive_guidance": adaptive_guidance,
        "latest_sessions": sessions[-5:],
    }
    write_json(paths.agent_data_dir / "agent_memory.json", profile)
    return profile


def build_adaptive_guidance(
    session_count: int,
    totals: dict[str, int],
    category_counts: dict[str, int],
) -> dict[str, Any]:
    sensitivity = {
        "posture": "normal",
        "restlessness": "normal",
        "breaks": "normal",
        "cleanup": "normal",
    }
    recommendations = []
    if session_count < 2:
        return {
            "summary": "Not enough completed sessions to adapt coaching yet.",
            "sensitivity": sensitivity,
            "recommendations": recommendations,
            "evidence": {"session_count": session_count},
        }

    if totals.get("slouching", 0) >= max(2, session_count):
        sensitivity["posture"] = "higher"
        recommendations.append("Treat live posture evidence as more important because slouching has recurred across sessions.")
    if totals.get("restless", 0) >= max(2, session_count):
        sensitivity["restlessness"] = "higher"
        recommendations.append("Act slightly earlier on live restlessness signals because restlessness has recurred across sessions.")
    if totals.get("too_still", 0) + totals.get("hyperfocus", 0) >= max(2, session_count):
        sensitivity["breaks"] = "higher"
        recommendations.append("Prefer movement-break nudges when live stillness suggests hyperfocus.")
    if category_counts.get("cleanup", 0) + category_counts.get("object", 0) >= max(2, session_count // 2):
        sensitivity["cleanup"] = "higher"
        recommendations.append("Treat persistent monitorable objects as more actionable because cleanup nudges have recurred.")

    active = [key for key, value in sensitivity.items() if value == "higher"]
    summary = (
        "Memory can adapt coaching for: " + ", ".join(active)
        if active
        else "Memory has enough sessions, but no strong recurring pattern yet."
    )
    return {
        "summary": summary,
        "sensitivity": sensitivity,
        "recommendations": recommendations,
        "evidence": {
            "session_count": session_count,
            "totals": totals,
            "nudge_category_counts": dict(sorted(category_counts.items())),
        },
    }
