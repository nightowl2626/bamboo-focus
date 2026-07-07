"""Deterministic local fallback rules for offline nudging.

These rules intentionally stay conservative. They use the same bounded data
tools as the Qwen agent, but they never call a model or the network.
"""

from __future__ import annotations

from typing import Any


FOCUS_AREAS = {
    "posture",
    "restlessness",
    "movement",
    "declutter",
    "breaks",
    "focus",
}


def notification_level(settings: dict[str, Any] | None) -> str:
    level = str((settings or {}).get("notification_level") or "minimal")
    return level if level in {"minimal", "balanced", "active"} else "minimal"


def selected_focus(settings: dict[str, Any] | None) -> set[str]:
    raw = (settings or {}).get("focus_areas")
    if not isinstance(raw, list):
        return set()
    return {str(item) for item in raw if str(item) in FOCUS_AREAS}


def metric(raw_summary: dict[str, Any], name: str, stat: str = "mean") -> float | None:
    metrics = raw_summary.get("metrics") if isinstance(raw_summary.get("metrics"), dict) else {}
    item = metrics.get(name) if isinstance(metrics.get(name), dict) else {}
    value = item.get(stat)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def posture_term_hits(posture_context: dict[str, Any], terms: set[str]) -> int:
    hits = 0
    analyses = posture_context.get("analyses") if isinstance(posture_context.get("analyses"), list) else []
    for item in analyses:
        text = " ".join(
            str(value)
            for value in (
                item.get("judgement"),
                item.get("posture"),
                item.get("behaviour"),
                item.get("stillness_or_restlessness"),
                " ".join(str(x) for x in item.get("significant_changes", []) if x is not None)
                if isinstance(item.get("significant_changes"), list)
                else "",
                " ".join(str(x) for x in item.get("observations", []) if x is not None)
                if isinstance(item.get("observations"), list)
                else "",
            )
        ).lower()
        if any(term in text for term in terms):
            hits += 1
    return hits


def build_local_context(
    tools: Any,
    user_settings: dict[str, Any] | None = None,
    include_raw: bool = True,
) -> dict[str, Any]:
    context = {
        "retrieval_strategy": {
            "summary": "Deterministic local fallback over bounded JSONL/state retrieval. No Qwen calls.",
            "lookback_seconds": tools.lookback_seconds,
            "mode": "local_fallback",
        },
        "baseline_policy": tools.baseline_policy(),
        "posture_context": tools.latest_posture_context(),
        "object_context": tools.latest_object_context(),
        "nudge_history": tools.recent_nudge_history(),
        "user_settings": user_settings or {},
    }
    if include_raw:
        context["raw_posture_summary"] = tools.recent_raw_posture_summary(max_events=30)
    dwell = tools.object_dwell_report(min_duration_minutes=0, min_seen_count=1)
    context["object_context"]["dwell_candidates"] = dwell.get("candidates", [])
    context["local_rule_evaluation"] = {}
    return context


def local_decision_from_context(context: dict[str, Any]) -> dict[str, Any]:
    settings = context.get("user_settings") if isinstance(context.get("user_settings"), dict) else {}
    level = notification_level(settings)
    focus = selected_focus(settings)
    posture_context = context.get("posture_context") if isinstance(context.get("posture_context"), dict) else {}
    raw = context.get("raw_posture_summary") if isinstance(context.get("raw_posture_summary"), dict) else {}
    object_context = context.get("object_context") if isinstance(context.get("object_context"), dict) else {}

    min_repeats = {"minimal": 2, "balanced": 2, "active": 1}[level]
    object_minutes = {"minimal": 60, "balanced": 30, "active": 15}[level]
    if "declutter" in focus:
        object_minutes = max(10, object_minutes // 2)

    recurring = posture_context.get("recurring_flags") if isinstance(posture_context.get("recurring_flags"), dict) else {}
    slouch_hits = int(recurring.get("slouching", 0) or 0) + posture_term_hits(
        posture_context, {"slouch", "forward head", "too far forward", "leaning"}
    )
    restless_hits = int(recurring.get("restless", 0) or 0) + posture_term_hits(
        posture_context, {"restless", "shifting", "more movement"}
    )
    still_hits = (
        int(recurring.get("too_still", 0) or 0)
        + int(recurring.get("possibly_hyperfocused", 0) or 0)
        + posture_term_hits(posture_context, {"too still", "very still", "hyperfocused", "locked in"})
    )

    forward_mean = metric(raw, "forward_head_ratio")
    forward_max = metric(raw, "forward_head_ratio", "max")
    torso_mean = metric(raw, "torso_lean_ratio")
    shoulder_mean = metric(raw, "shoulder_tilt_ratio")
    motion_mean = metric(raw, "motion_score")
    motion_max = metric(raw, "motion_score", "max")
    raw_events = int(raw.get("event_count", 0) or 0)

    if raw_events >= 2:
        if (forward_mean is not None and forward_mean >= 0.42) or (forward_max is not None and forward_max >= 0.5):
            slouch_hits += 2
        if torso_mean is not None and torso_mean >= 0.24:
            slouch_hits += 1
        if shoulder_mean is not None and shoulder_mean >= 0.12:
            slouch_hits += 1
        if motion_mean is not None and motion_mean <= 1.0:
            still_hits += 2
        if motion_mean is not None and motion_mean >= 8.0:
            restless_hits += 2
        if motion_max is not None and motion_max >= 14.0:
            restless_hits += 1

    context["local_rule_evaluation"] = {
        "notification_level": level,
        "focus_areas": sorted(focus),
        "min_repeats": min_repeats,
        "object_minutes_threshold": object_minutes,
        "slouch_hits": slouch_hits,
        "restless_hits": restless_hits,
        "still_hits": still_hits,
        "raw_events": raw_events,
        "raw_motion_mean": motion_mean,
        "raw_forward_head_mean": forward_mean,
    }

    candidates: list[tuple[int, dict[str, Any]]] = []

    dwell_candidates = object_context.get("dwell_candidates")
    if isinstance(dwell_candidates, list):
        for candidate in dwell_candidates:
            if not isinstance(candidate, dict):
                continue
            duration = int(candidate.get("observed_duration_seconds", 0) or 0)
            seen_count = int(candidate.get("seen_count", 0) or 0)
            if duration < object_minutes * 60 or seen_count < 2:
                continue
            label = str(candidate.get("label") or "object")
            score = 40 + min(20, duration // 900) + min(10, seen_count)
            if "declutter" in focus:
                score += 15
            candidates.append(
                (
                    score,
                    {
                        "should_nudge": True,
                        "category": "cleanup",
                        "urgency": "low",
                        "rationale": (
                            f"A monitorable object has remained visible for about {duration // 60} minutes."
                        ),
                        "recommended_focus": f"check or clear the {label}",
                        "supporting_signals": [
                            f"{label} persisted across {seen_count} object snapshots",
                            f"local threshold is {object_minutes} minutes for {level} notification level",
                        ],
                        "suppress_reason": None,
                        "cooldown_key": f"object_{label}",
                        "suggested_recheck_minutes": 30,
                    },
                )
            )

    if slouch_hits >= min_repeats:
        score = 50 + slouch_hits * 5 + (15 if "posture" in focus else 0)
        candidates.append(
            (
                score,
                {
                    "should_nudge": True,
                    "category": "posture",
                    "urgency": "low" if slouch_hits < 4 else "medium",
                    "rationale": "Local rules found repeated slouching or forward-head posture signals.",
                    "recommended_focus": "sit up straighter and reset shoulder position",
                    "supporting_signals": [
                        f"slouching/forward-head hit count: {slouch_hits}",
                        f"recent forward-head mean: {forward_mean}" if forward_mean is not None else "Qwen analysis labels flagged posture",
                    ],
                    "suppress_reason": None,
                    "cooldown_key": "posture_slouching",
                    "suggested_recheck_minutes": 20,
                },
            )
        )

    if restless_hits >= min_repeats:
        score = 45 + restless_hits * 5 + (15 if "restlessness" in focus else 0)
        candidates.append(
            (
                score,
                {
                    "should_nudge": True,
                    "category": "restlessness",
                    "urgency": "low" if restless_hits < 4 else "medium",
                    "rationale": "Local rules found repeated restlessness or elevated motion signals.",
                    "recommended_focus": "take a short reset for restlessness",
                    "supporting_signals": [
                        f"restlessness hit count: {restless_hits}",
                        f"recent motion mean: {motion_mean}" if motion_mean is not None else "analysis labels flagged restlessness",
                    ],
                    "suppress_reason": None,
                    "cooldown_key": "behaviour_restlessness",
                    "suggested_recheck_minutes": 20,
                },
            )
        )

    if still_hits >= min_repeats and (level != "minimal" or {"breaks", "movement", "focus"} & focus or still_hits >= 3):
        score = 42 + still_hits * 5 + (15 if {"breaks", "movement", "focus"} & focus else 0)
        candidates.append(
            (
                score,
                {
                    "should_nudge": True,
                    "category": "break",
                    "urgency": "low",
                    "rationale": "Local rules found repeated too-still or possible hyperfocus signals.",
                    "recommended_focus": "take a quick movement break",
                    "supporting_signals": [
                        f"too-still/hyperfocus hit count: {still_hits}",
                        f"recent motion mean: {motion_mean}" if motion_mean is not None else "analysis labels flagged stillness",
                    ],
                    "suppress_reason": None,
                    "cooldown_key": "break_stillness",
                    "suggested_recheck_minutes": 20,
                },
            )
        )

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        decision = dict(candidates[0][1])
        context["local_rule_evaluation"]["candidate_count"] = len(candidates)
        context["local_rule_evaluation"]["selected_score"] = candidates[0][0]
        return decision

    return {
        "should_nudge": False,
        "category": "none",
        "urgency": "low",
        "rationale": "No repeated or persistent local signal is strong enough for a nudge.",
        "recommended_focus": "",
        "supporting_signals": [
            f"slouch_hits={slouch_hits}",
            f"restless_hits={restless_hits}",
            f"still_hits={still_hits}",
        ],
        "suppress_reason": "No necessary local nudge found.",
        "cooldown_key": "none",
        "suggested_recheck_minutes": 15,
    }


def local_decision(
    tools: Any,
    user_settings: dict[str, Any] | None = None,
    include_raw: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = build_local_context(tools, user_settings=user_settings, include_raw=include_raw)
    return local_decision_from_context(context), context
