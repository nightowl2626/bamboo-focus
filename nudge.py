"""ADHD/productivity nudge decision agent.

This script reads the monitoring artifacts produced by long_monitor.py and
object_monitor.py through bounded retrieval helpers, asks Qwen for a conservative
decision, and writes a factual decision record. It does not produce the final
nudge message; it decides whether a nudge is warranted and what it should be
about.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_fallback import local_decision, local_decision_from_context


DEFAULT_QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
DEFAULT_DECISION = {
    "should_nudge": False,
    "category": "none",
    "urgency": "low",
    "rationale": "No clear need for a nudge.",
    "recommended_focus": "",
    "supporting_signals": [],
    "suppress_reason": None,
    "cooldown_key": "none",
    "suggested_recheck_minutes": 15,
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
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def seconds_between(start: datetime | None, end: datetime | None) -> int:
    if start is None or end is None:
        return 0
    return max(0, int((end - start).total_seconds()))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item) + "\n")


def tail_jsonl(path: Path, max_lines: int = 100, block_size: int = 8192) -> list[dict[str, Any]]:
    if not path.exists() or max_lines <= 0:
        return []
    lines: list[bytes] = []
    with path.open("rb") as file:
        file.seek(0, os.SEEK_END)
        position = file.tell()
        buffer = b""
        while position > 0 and len(lines) <= max_lines:
            read_size = min(block_size, position)
            position -= read_size
            file.seek(position)
            buffer = file.read(read_size) + buffer
            parts = buffer.splitlines()
            if position > 0:
                buffer = parts[0]
                lines = parts[1:] + lines
            else:
                lines = parts + lines
    items = []
    for line in lines[-max_lines:]:
        if not line.strip():
            continue
        try:
            item = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(item, dict):
            items.append(item)
    return items


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


def normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().split())


def labels_from_policy(policy: dict[str, Any], key: str) -> list[str]:
    entries = policy.get(key) if isinstance(policy.get(key), list) else []
    labels = []
    for entry in entries:
        if isinstance(entry, dict):
            label = str(entry.get("label", "")).strip()
        else:
            label = str(entry).strip()
        if label:
            labels.append(label)
    return sorted(set(labels), key=str.lower)


def stringify_for_search(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {stringify_for_search(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(stringify_for_search(item) for item in value)
    return str(value)


def search_terms(query: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9_]+", query.lower()) if len(term) > 1]


def lexical_score(record: dict[str, Any], query: str) -> int:
    terms = search_terms(query)
    if not terms:
        return 1
    haystack = stringify_for_search(record).lower()
    return sum(haystack.count(term) for term in terms)


def compact_record(record: dict[str, Any], max_chars: int = 6000) -> dict[str, Any]:
    text = json.dumps(record)
    if len(text) <= max_chars:
        return record
    return {
        "truncated": True,
        "preview": text[:max_chars],
    }


@dataclass
class AgentPaths:
    baseline: Path
    long_monitor_dir: Path
    object_monitor_dir: Path
    agent_data_dir: Path

    @property
    def long_analyses_path(self) -> Path:
        return self.long_monitor_dir / "qwen_analyses.jsonl"

    @property
    def long_events_path(self) -> Path:
        return self.long_monitor_dir / "monitor_events.jsonl"

    @property
    def long_state_path(self) -> Path:
        return self.long_monitor_dir / "monitor_state.json"

    @property
    def object_events_path(self) -> Path:
        return self.object_monitor_dir / "object_events.jsonl"

    @property
    def object_state_path(self) -> Path:
        return self.object_monitor_dir / "object_monitor_state.json"

    @property
    def decisions_path(self) -> Path:
        return self.agent_data_dir / "nudge_decisions.jsonl"

    @property
    def latest_decision_path(self) -> Path:
        return self.agent_data_dir / "latest_nudge_decision.json"

    @property
    def session_settings_path(self) -> Path:
        return self.agent_data_dir / "session_settings.json"


class DataTools:
    """Bounded retrieval helpers over monitor artifacts."""

    def __init__(self, paths: AgentPaths, lookback_hours: float) -> None:
        self.paths = paths
        self.lookback_seconds = int(lookback_hours * 3600)

    def baseline_policy(self) -> dict[str, Any]:
        baseline = read_json(self.paths.baseline) or {}
        policy = baseline.get("object_policy") if isinstance(baseline.get("object_policy"), dict) else {}
        return {
            "baseline_created_at": baseline.get("created_at"),
            "whitelisted_objects": labels_from_policy(policy, "whitelisted_objects"),
            "monitorable_objects": labels_from_policy(policy, "monitorable_objects"),
            "notes": policy.get("notes", ""),
        }

    def user_profile(self) -> dict[str, Any]:
        settings = read_json(self.paths.session_settings_path) or {}
        struggles = settings.get("work_struggles") if isinstance(settings.get("work_struggles"), dict) else {}
        return {
            "tool": "user_profile",
            "intent": settings.get("intent", ""),
            "notification_level": settings.get("notification_level", "minimal"),
            "focus_areas": settings.get("focus_areas", []),
            "work_struggles": {
                "selected": struggles.get("selected", []),
                "notes": struggles.get("notes", ""),
                "support_preferences": struggles.get("support_preferences", ""),
            },
            "pomodoro_minutes": settings.get("pomodoro_minutes"),
            "break_minutes": settings.get("break_minutes"),
        }

    def monitor_overview(self) -> dict[str, Any]:
        return {
            "tool": "monitor_overview",
            "baseline_policy": self.baseline_policy(),
            "user_profile_summary": self.user_profile(),
            "posture_state": read_json(self.paths.long_state_path) or {},
            "object_state": read_json(self.paths.object_state_path) or {},
            "latest_nudge_decision": read_json(self.paths.latest_decision_path) or {},
            "available_sources": {
                "posture_analyses": str(self.paths.long_analyses_path),
                "raw_posture_windows": str(self.paths.long_events_path),
                "object_snapshots": str(self.paths.object_events_path),
                "nudge_history": str(self.paths.decisions_path),
                "baseline": str(self.paths.baseline),
            },
        }

    def baseline_raw(self, include_events: bool = False, max_events: int = 10) -> dict[str, Any]:
        baseline = read_json(self.paths.baseline) or {}
        if not include_events:
            baseline = dict(baseline)
            calibration = baseline.get("calibration")
            if isinstance(calibration, dict) and "events" in calibration:
                calibration = dict(calibration)
                calibration["events"] = {
                    "omitted": True,
                    "event_count": len(calibration.get("events", [])) if isinstance(calibration.get("events"), list) else 0,
                    "hint": "call baseline_raw with include_events=true only if baseline raw events are needed",
                }
                baseline["calibration"] = calibration
            return baseline
        baseline = dict(baseline)
        calibration = baseline.get("calibration")
        if isinstance(calibration, dict) and isinstance(calibration.get("events"), list):
            calibration = dict(calibration)
            calibration["events"] = calibration["events"][-max_events:]
            baseline["calibration"] = calibration
        return baseline

    def latest_posture_context(self, max_analyses: int = 12) -> dict[str, Any]:
        analyses = self._recent_by_time(tail_jsonl(self.paths.long_analyses_path, max_analyses * 3))[-max_analyses:]
        state = read_json(self.paths.long_state_path) or {}
        compact = []
        category_counts: dict[str, int] = {}
        for item in analyses:
            analysis = item.get("analysis") if isinstance(item.get("analysis"), dict) else {}
            posture = str(analysis.get("posture", "unclear"))
            behaviour = str(analysis.get("behaviour", "unclear"))
            stillness = str(analysis.get("stillness_or_restlessness", "unclear"))
            for key in (posture, behaviour, stillness):
                if key and key not in {"baseline_like", "normal", "unclear"}:
                    category_counts[key] = category_counts.get(key, 0) + 1
            compact.append(
                {
                    "created_at": item.get("created_at"),
                    "judgement": analysis.get("judgement"),
                    "posture": posture,
                    "behaviour": behaviour,
                    "stillness_or_restlessness": stillness,
                    "confidence": analysis.get("confidence"),
                    "significant_changes": analysis.get("significant_changes", []),
                    "observations": analysis.get("observations", []),
                }
            )
        return {
            "tool": "latest_posture_context",
            "state": state,
            "analyses": compact,
            "recurring_flags": category_counts,
        }

    def search_posture_analyses(
        self,
        query: str = "",
        limit: int = 8,
        lookback_hours: float | None = None,
        posture: str | None = None,
        behaviour: str | None = None,
        stillness_or_restlessness: str | None = None,
    ) -> dict[str, Any]:
        items = tail_jsonl(self.paths.long_analyses_path, 800)
        items = self._recent_by_time_for(items, lookback_hours)
        matches = []
        for item in items:
            analysis = item.get("analysis") if isinstance(item.get("analysis"), dict) else {}
            if posture and analysis.get("posture") != posture:
                continue
            if behaviour and analysis.get("behaviour") != behaviour:
                continue
            if stillness_or_restlessness and analysis.get("stillness_or_restlessness") != stillness_or_restlessness:
                continue
            score = lexical_score(item, query)
            if query and score <= 0:
                continue
            matches.append((score, item))
        matches.sort(
            key=lambda pair: (
                pair[0],
                parse_time(pair[1].get("created_at")).timestamp() if parse_time(pair[1].get("created_at")) else 0,
            ),
            reverse=True,
        )
        return {
            "tool": "search_posture_analyses",
            "query": query,
            "filters": {
                "posture": posture,
                "behaviour": behaviour,
                "stillness_or_restlessness": stillness_or_restlessness,
                "lookback_hours": lookback_hours if lookback_hours is not None else self.lookback_seconds / 3600,
            },
            "matches": [
                {
                    "created_at": item.get("created_at"),
                    "interval": item.get("interval"),
                    "analysis": item.get("analysis"),
                }
                for _, item in matches[:limit]
            ],
        }

    def recent_raw_posture_summary(self, max_events: int = 20) -> dict[str, Any]:
        events = self._recent_by_time(tail_jsonl(self.paths.long_events_path, max_events * 3))[-max_events:]
        values: dict[str, list[float]] = {
            "forward_head_ratio": [],
            "torso_lean_ratio": [],
            "shoulder_tilt_ratio": [],
            "motion_score": [],
            "confidence": [],
        }
        for event in events:
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            series = payload.get("series") if isinstance(payload.get("series"), dict) else {}
            for key in values:
                for value in series.get(key, []):
                    if value is None:
                        continue
                    try:
                        values[key].append(float(value))
                    except (TypeError, ValueError):
                        continue
        return {
            "tool": "recent_raw_posture_summary",
            "event_count": len(events),
            "metrics": {
                key: {
                    "mean": round(sum(nums) / len(nums), 3) if nums else None,
                    "min": round(min(nums), 3) if nums else None,
                    "max": round(max(nums), 3) if nums else None,
                    "count": len(nums),
                }
                for key, nums in values.items()
            },
        }

    def latest_object_context(self, max_snapshots: int = 12) -> dict[str, Any]:
        snapshots = self._recent_by_time(tail_jsonl(self.paths.object_events_path, max_snapshots * 4))[-max_snapshots:]
        state = read_json(self.paths.object_state_path) or {}
        compact = []
        for item in snapshots:
            compact.append(
                {
                    "received_at": item.get("received_at"),
                    "monitorable_count": item.get("monitorable_count"),
                    "objects": [
                        {
                            "label": obj.get("label"),
                            "score": obj.get("score"),
                        }
                        for obj in item.get("monitorable_objects", [])
                        if isinstance(obj, dict)
                    ],
                }
            )
        return {
            "tool": "latest_object_context",
            "state": state,
            "snapshots": compact,
            "dwell_candidates": self.object_dwell_candidates(snapshots),
        }

    def search_object_snapshots(
        self,
        query: str = "",
        label: str | None = None,
        limit: int = 12,
        lookback_hours: float | None = None,
    ) -> dict[str, Any]:
        snapshots = tail_jsonl(self.paths.object_events_path, 1000)
        snapshots = self._recent_by_time_for(snapshots, lookback_hours)
        normalized_label = normalize_label(label) if label else None
        matches = []
        for snapshot in snapshots:
            objects = snapshot.get("monitorable_objects") if isinstance(snapshot.get("monitorable_objects"), list) else []
            if normalized_label and not any(
                normalize_label(str(obj.get("label", ""))) == normalized_label
                for obj in objects
                if isinstance(obj, dict)
            ):
                continue
            score = lexical_score(snapshot, query)
            if query and score <= 0:
                continue
            matches.append((score, snapshot))
        matches.sort(
            key=lambda pair: (
                pair[0],
                parse_time(pair[1].get("received_at")).timestamp() if parse_time(pair[1].get("received_at")) else 0,
            ),
            reverse=True,
        )
        return {
            "tool": "search_object_snapshots",
            "query": query,
            "label": label,
            "matches": [
                {
                    "received_at": item.get("received_at"),
                    "monitorable_count": item.get("monitorable_count"),
                    "monitorable_objects": item.get("monitorable_objects", []),
                    "whitelist_labels": item.get("whitelist_labels", []),
                }
                for _, item in matches[:limit]
            ],
        }

    def object_dwell_report(
        self,
        min_duration_minutes: int = 0,
        min_seen_count: int = 1,
        lookback_hours: float | None = None,
    ) -> dict[str, Any]:
        snapshots = self._recent_by_time_for(tail_jsonl(self.paths.object_events_path, 1200), lookback_hours)
        candidates = self.object_dwell_candidates(snapshots)
        filtered = [
            item
            for item in candidates
            if item.get("observed_duration_seconds", 0) >= min_duration_minutes * 60
            and item.get("seen_count", 0) >= min_seen_count
        ]
        return {
            "tool": "object_dwell_report",
            "filters": {
                "min_duration_minutes": min_duration_minutes,
                "min_seen_count": min_seen_count,
                "lookback_hours": lookback_hours if lookback_hours is not None else self.lookback_seconds / 3600,
            },
            "candidates": filtered,
        }

    def object_dwell_candidates(self, snapshots: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        if snapshots is None:
            snapshots = self._recent_by_time(tail_jsonl(self.paths.object_events_path, 200))
        by_label: dict[str, dict[str, Any]] = {}
        for snapshot in snapshots:
            seen_at = parse_time(snapshot.get("received_at"))
            for obj in snapshot.get("monitorable_objects", []):
                if not isinstance(obj, dict):
                    continue
                label = normalize_label(str(obj.get("label", "")))
                if not label:
                    continue
                current = by_label.setdefault(
                    label,
                    {
                        "label": label,
                        "first_seen": seen_at,
                        "last_seen": seen_at,
                        "seen_count": 0,
                        "best_score": 0.0,
                    },
                )
                current["seen_count"] += 1
                if seen_at and (current["first_seen"] is None or seen_at < current["first_seen"]):
                    current["first_seen"] = seen_at
                if seen_at and (current["last_seen"] is None or seen_at > current["last_seen"]):
                    current["last_seen"] = seen_at
                try:
                    current["best_score"] = max(current["best_score"], float(obj.get("score", 0) or 0))
                except (TypeError, ValueError):
                    pass
        candidates = []
        for item in by_label.values():
            first_seen = item["first_seen"]
            last_seen = item["last_seen"]
            candidates.append(
                {
                    "label": item["label"],
                    "seen_count": item["seen_count"],
                    "first_seen": first_seen.isoformat() if first_seen else None,
                    "last_seen": last_seen.isoformat() if last_seen else None,
                    "observed_duration_seconds": seconds_between(first_seen, last_seen),
                    "best_score": round(item["best_score"], 3),
                }
            )
        return sorted(candidates, key=lambda item: (-item["observed_duration_seconds"], -item["seen_count"], item["label"]))

    def recent_nudge_history(self, max_decisions: int = 20) -> dict[str, Any]:
        decisions = self._recent_by_time(tail_jsonl(self.paths.decisions_path, max_decisions * 2))[-max_decisions:]
        compact = []
        for item in decisions:
            decision = item.get("decision") if isinstance(item.get("decision"), dict) else item
            compact.append(
                {
                    "created_at": item.get("created_at"),
                    "should_nudge": decision.get("should_nudge"),
                    "category": decision.get("category"),
                    "cooldown_key": decision.get("cooldown_key"),
                    "recommended_focus": decision.get("recommended_focus"),
                    "suppress_reason": decision.get("suppress_reason"),
                }
            )
        return {
            "tool": "recent_nudge_history",
            "decisions": compact,
        }

    def search_nudge_history(
        self,
        query: str = "",
        category: str | None = None,
        cooldown_key: str | None = None,
        limit: int = 12,
        lookback_hours: float | None = None,
    ) -> dict[str, Any]:
        items = self._recent_by_time_for(tail_jsonl(self.paths.decisions_path, 500), lookback_hours)
        matches = []
        for item in items:
            decision = item.get("decision") if isinstance(item.get("decision"), dict) else item
            if category and decision.get("category") != category:
                continue
            if cooldown_key and decision.get("cooldown_key") != cooldown_key:
                continue
            score = lexical_score(item, query)
            if query and score <= 0:
                continue
            matches.append((score, item))
        matches.sort(
            key=lambda pair: (
                pair[0],
                parse_time(pair[1].get("created_at")).timestamp() if parse_time(pair[1].get("created_at")) else 0,
            ),
            reverse=True,
        )
        return {
            "tool": "search_nudge_history",
            "query": query,
            "category": category,
            "cooldown_key": cooldown_key,
            "matches": [
                {
                    "created_at": item.get("created_at"),
                    "decision": item.get("decision", item),
                }
                for _, item in matches[:limit]
            ],
        }

    def _recent_by_time(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self._recent_by_time_for(items, None)

    def _recent_by_time_for(self, items: list[dict[str, Any]], lookback_hours: float | None) -> list[dict[str, Any]]:
        lookback_seconds = self.lookback_seconds if lookback_hours is None else int(lookback_hours * 3600)
        if lookback_seconds <= 0:
            return items
        cutoff = datetime.now(timezone.utc).timestamp() - lookback_seconds
        kept = []
        for item in items:
            dt = parse_time(item.get("created_at") or item.get("received_at") or item.get("_received_at"))
            if dt is None or dt.timestamp() >= cutoff:
                kept.append(item)
        return kept


def build_context(tools: DataTools, include_raw: bool, user_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    context = {
        "retrieval_strategy": {
            "summary": (
                "Bounded RAG-like retrieval over state files and JSONL tails. "
                "The agent reads latest analyses, compact object dwell candidates, baseline policy, "
                "and recent nudge history instead of scanning full logs linearly."
            ),
            "lookback_seconds": tools.lookback_seconds,
        },
        "baseline_policy": tools.baseline_policy(),
        "posture_context": tools.latest_posture_context(),
        "object_context": tools.latest_object_context(),
        "nudge_history": tools.recent_nudge_history(),
        "user_settings": user_settings or {},
    }
    if include_raw:
        context["raw_posture_summary"] = tools.recent_raw_posture_summary()
    return context


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


def normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_DECISION)
    normalized.update({key: value for key, value in decision.items() if key in normalized})
    normalized["should_nudge"] = bool(normalized["should_nudge"])
    if normalized["category"] not in {"posture", "break", "restlessness", "cleanup", "object", "focus", "none", "other"}:
        normalized["category"] = "other" if normalized["should_nudge"] else "none"
    if normalized["urgency"] not in {"low", "medium", "high"}:
        normalized["urgency"] = "low"
    if not isinstance(normalized["supporting_signals"], list):
        normalized["supporting_signals"] = []
    if not normalized["cooldown_key"]:
        normalized["cooldown_key"] = normalized["category"]
    try:
        normalized["suggested_recheck_minutes"] = max(1, int(normalized["suggested_recheck_minutes"]))
    except (TypeError, ValueError):
        normalized["suggested_recheck_minutes"] = 15
    return normalized


def qwen_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_base = os.getenv("QWEN_API_BASE", DEFAULT_QWEN_API_BASE).rstrip("/")
    api_key = os.getenv("QWEN_API_KEY")
    model = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set")
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
    }
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice or "auto"
    else:
        payload["response_format"] = {"type": "json_object"}
    return request_json(
        "POST",
        f"{api_base}/chat/completions",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )


def tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_monitor_overview",
                "description": "Get current state, baseline object policy, latest nudge decision, and available data sources.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_user_profile",
                "description": "Fetch the user's work-session questionnaire, struggles, support preferences, intent, focus areas, and notification level.",
                "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recent_posture_analyses",
                "description": "Fetch compact recent Qwen posture/behaviour analyses from the long monitor.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 12}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_posture_analyses",
                "description": "Search recent posture analyses by text and/or structured labels such as slouching, restless, too_still, focused.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "default": ""},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
                        "lookback_hours": {"type": "number", "minimum": 0, "default": 4},
                        "posture": {"type": "string", "description": "Optional exact posture label."},
                        "behaviour": {"type": "string", "description": "Optional exact behaviour label."},
                        "stillness_or_restlessness": {"type": "string", "description": "Optional exact stillness/restlessness label."},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recent_raw_posture_summary",
                "description": "Fetch compact raw posture metrics summary for recent posture windows when high-level analyses are missing or contradictory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "max_events": {"type": "integer", "minimum": 1, "maximum": 80, "default": 20}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recent_object_snapshots",
                "description": "Fetch compact recent object monitor snapshots after whitelist filtering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_object_dwell_report",
                "description": "Aggregate monitorable objects over time to find objects that persisted on the desk.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_duration_minutes": {"type": "integer", "minimum": 0, "default": 0},
                        "min_seen_count": {"type": "integer", "minimum": 1, "default": 1},
                        "lookback_hours": {"type": "number", "minimum": 0, "default": 4},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_object_snapshots",
                "description": "Search recent object snapshots by label or text, for example mug, cup, phone, paper, clutter.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "default": ""},
                        "label": {"type": "string", "description": "Optional exact object label."},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                        "lookback_hours": {"type": "number", "minimum": 0, "default": 4},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_recent_nudge_history",
                "description": "Fetch recent nudge decisions to avoid repeated nudges.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_nudge_history",
                "description": "Search prior nudge decisions by category, cooldown key, or text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "default": ""},
                        "category": {"type": "string"},
                        "cooldown_key": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
                        "lookback_hours": {"type": "number", "minimum": 0, "default": 4},
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_baseline_raw",
                "description": "Optionally inspect baseline JSON. Raw calibration events are omitted unless requested.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "include_events": {"type": "boolean", "default": False},
                        "max_events": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                    },
                    "additionalProperties": False,
                },
            },
        },
    ]


def parse_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def execute_tool(tools: DataTools, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    try:
        if name == "get_monitor_overview":
            return tools.monitor_overview()
        if name == "get_user_profile":
            return tools.user_profile()
        if name == "get_recent_posture_analyses":
            return tools.latest_posture_context(max_analyses=int(arguments.get("limit", 12)))
        if name == "search_posture_analyses":
            return tools.search_posture_analyses(
                query=str(arguments.get("query", "")),
                limit=int(arguments.get("limit", 8)),
                lookback_hours=arguments.get("lookback_hours"),
                posture=arguments.get("posture"),
                behaviour=arguments.get("behaviour"),
                stillness_or_restlessness=arguments.get("stillness_or_restlessness"),
            )
        if name == "get_recent_raw_posture_summary":
            return tools.recent_raw_posture_summary(max_events=int(arguments.get("max_events", 20)))
        if name == "get_recent_object_snapshots":
            return tools.latest_object_context(max_snapshots=int(arguments.get("limit", 12)))
        if name == "get_object_dwell_report":
            return tools.object_dwell_report(
                min_duration_minutes=int(arguments.get("min_duration_minutes", 0)),
                min_seen_count=int(arguments.get("min_seen_count", 1)),
                lookback_hours=arguments.get("lookback_hours"),
            )
        if name == "search_object_snapshots":
            return tools.search_object_snapshots(
                query=str(arguments.get("query", "")),
                label=arguments.get("label"),
                limit=int(arguments.get("limit", 12)),
                lookback_hours=arguments.get("lookback_hours"),
            )
        if name == "get_recent_nudge_history":
            return tools.recent_nudge_history(max_decisions=int(arguments.get("limit", 20)))
        if name == "search_nudge_history":
            return tools.search_nudge_history(
                query=str(arguments.get("query", "")),
                category=arguments.get("category"),
                cooldown_key=arguments.get("cooldown_key"),
                limit=int(arguments.get("limit", 12)),
                lookback_hours=arguments.get("lookback_hours"),
            )
        if name == "get_baseline_raw":
            return tools.baseline_raw(
                include_events=bool(arguments.get("include_events", False)),
                max_events=int(arguments.get("max_events", 10)),
            )
        return {"error": f"unknown tool: {name}"}
    except Exception as exc:
        return {"error": str(exc), "tool": name}


def extract_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return tool_calls
    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        return [
            {
                "id": "function_call",
                "type": "function",
                "function": function_call,
            }
        ]
    return []


def call_qwen_tool_agent(
    tools: DataTools,
    max_rounds: int = 8,
    user_settings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    overview = compact_record(tools.monitor_overview(), max_chars=12000)
    history = compact_record(tools.recent_nudge_history(), max_chars=12000)
    calls_made: list[dict[str, Any]] = [
        {"name": "get_monitor_overview", "arguments": {}, "forced": True},
        {"name": "get_recent_nudge_history", "arguments": {}, "forced": True},
    ]
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "You are a conservative ADHD and productivity decision agent with dynamic data tools. "
                "Your job is to decide whether the user needs a nudge right now and what topic it should cover. "
                "You do not write the actual nudge message. You return a factual decision record. "
                "Aim for as few nudges as possible. Prefer no nudge when signals are weak, stale, already handled, "
                "or recently nudged."
            ),
        },
        {
            "role": "user",
            "content": (
                "Autonomously choose which tools to call. Start by inspecting monitor overview and recent nudge history. "
                "Then query the user profile/questionnaire, posture analyses, object dwell, object snapshots, raw posture, "
                "or baseline only if useful. "
                "Use search tools instead of reading raw logs linearly. Avoid repeated nudges unless the issue is persistent "
                "or high urgency.\n\n"
                "Decision guidance:\n"
                "- Apply user settings when present: work intent, focus areas, desired notification level, and tone. "
                "Use the questionnaire as preference context, not as proof that a nudge is needed. "
                "A minimal notification level should require stronger evidence; active notification level can act on lighter signals. "
                "Prioritize selected focus areas when several possible nudges compete.\n"
                "- posture/slouching: nudge only if posture is meaningfully worse than baseline or repeatedly flagged.\n"
                "- break/hyperfocus: nudge only if sustained stillness/stability suggests a break would be useful.\n"
                "- restlessness: nudge only if elevated movement is persistent enough to act on.\n"
                "- cleanup/object: nudge only if monitorable objects such as mug, cup, phone, scrap paper, plate, or clutter "
                "have persisted long enough to matter. Never nudge for whitelisted objects.\n\n"
                "When finished, return only valid JSON with this exact shape:\n"
                "{\n"
                '  "should_nudge": true,\n'
                '  "category": "posture|break|restlessness|cleanup|object|focus|none|other",\n'
                '  "urgency": "low|medium|high",\n'
                '  "rationale": "factual reason, not a user-facing nudge",\n'
                '  "recommended_focus": "what the nudge should be about, not the actual message",\n'
                '  "supporting_signals": ["concise factual signals"],\n'
                '  "suppress_reason": null,\n'
                '  "cooldown_key": "stable key such as posture_slouching or object_mug",\n'
                '  "suggested_recheck_minutes": 15\n'
                "}\n\n"
                "If no nudge is needed, set should_nudge false, category none, recommended_focus empty, "
                "and explain why in suppress_reason."
            ),
        },
        {
            "role": "user",
            "content": f"Current user work-session settings:\n{json.dumps(user_settings or {}, indent=2)}",
        },
        {
            "role": "user",
            "content": (
                "Initial required tool context has already been retrieved. Use it as grounded data, "
                "then call deeper tools if needed.\n\n"
                f"get_monitor_overview result:\n{json.dumps(overview, indent=2)}\n\n"
                f"get_recent_nudge_history result:\n{json.dumps(history, indent=2)}"
            ),
        },
    ]
    specs = tool_specs()
    for _ in range(max_rounds):
        response = qwen_chat(messages, tools=specs, tool_choice="auto")
        message = response["choices"][0]["message"]
        tool_calls = extract_tool_calls(message)
        if not tool_calls:
            decision = normalize_decision(parse_json_content(message.get("content") or "{}"))
            return decision, {"tool_calls": calls_made}
        messages.append(message)
        for call in tool_calls:
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            name = str(function.get("name", ""))
            arguments = parse_tool_arguments(function.get("arguments"))
            result = compact_record(execute_tool(tools, name, arguments), max_chars=12000)
            calls_made.append({"name": name, "arguments": arguments})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", name),
                    "name": name,
                    "content": json.dumps(result),
                }
            )
    messages.append(
        {
            "role": "user",
            "content": "Stop calling tools and return the final decision JSON now.",
        }
    )
    response = qwen_chat(messages)
    message = response["choices"][0]["message"]
    return normalize_decision(parse_json_content(message.get("content") or "{}")), {"tool_calls": calls_made}


def call_qwen_for_decision(context: dict[str, Any]) -> dict[str, Any]:
    api_base = os.getenv("QWEN_API_BASE", DEFAULT_QWEN_API_BASE).rstrip("/")
    api_key = os.getenv("QWEN_API_KEY")
    model = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a conservative ADHD and productivity decision agent. "
                "Your job is to decide if a person needs a nudge right now. "
                "You do not write the nudge message. You return a factual decision record. "
                "Aim to send as few nudges as possible. Only recommend a nudge when it is useful, "
                "timely, and supported by repeated or meaningful signals."
            ),
        },
        {
            "role": "user",
            "content": (
                "Use the retrieved tool context below. Prefer latest Qwen monitor analyses and object dwell "
                "summaries. Consult raw summaries only when the high-level analyses are missing, stale, or contradictory. "
                "Take recent nudge history into account and avoid repeating a topic unless the issue is clearly persistent "
                "or high urgency.\n\n"
                "Examples of valid factual decisions:\n"
                "- posture/slouching: nudge if posture has clearly degraded from baseline or repeated slouching appears.\n"
                "- break/hyperfocus: nudge if the person has been unusually still and stable for a sustained period.\n"
                "- restlessness: nudge if movement/shifting is clearly elevated and persistent.\n"
                "- cleanup/object: nudge if a monitorable object such as mug, cup, phone, scrap paper, plate, or clutter "
                "has persisted long enough to be worth addressing. Do not nudge for whitelisted environment objects.\n\n"
                "Return only valid JSON with this exact shape:\n"
                "{\n"
                '  "should_nudge": true,\n'
                '  "category": "posture|break|restlessness|cleanup|object|focus|none|other",\n'
                '  "urgency": "low|medium|high",\n'
                '  "rationale": "factual reason, not a user-facing nudge",\n'
                '  "recommended_focus": "what the nudge should be about, not the actual message",\n'
                '  "supporting_signals": ["concise factual signals"],\n'
                '  "suppress_reason": null,\n'
                '  "cooldown_key": "stable key such as posture_slouching or object_mug",\n'
                '  "suggested_recheck_minutes": 15\n'
                "}\n\n"
                "If no nudge is needed, set should_nudge false, category none, recommended_focus empty, "
                "and explain why in suppress_reason.\n\n"
                f"Retrieved tool context:\n{json.dumps(context, indent=2)}"
            ),
        },
    ]
    response = request_json(
        "POST",
        f"{api_base}/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        content = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected Qwen response shape: {response}") from exc
    return normalize_decision(parse_json_content(content))


def fallback_decision(context: dict[str, Any]) -> dict[str, Any]:
    return normalize_decision(local_decision_from_context(context))


def apply_cooldown(decision: dict[str, Any], history: dict[str, Any], cooldown_minutes: int) -> dict[str, Any]:
    if not decision.get("should_nudge"):
        return decision
    key = decision.get("cooldown_key") or decision.get("category")
    now = datetime.now(timezone.utc)
    for item in reversed(history.get("decisions", [])):
        if not item.get("should_nudge"):
            continue
        if item.get("cooldown_key") != key and item.get("category") != decision.get("category"):
            continue
        created_at = parse_time(item.get("created_at"))
        age_seconds = seconds_between(created_at, now)
        if created_at and age_seconds < cooldown_minutes * 60:
            suppressed = dict(decision)
            suppressed["should_nudge"] = False
            suppressed["suppress_reason"] = (
                f"Suppressed by cooldown: a similar {key} nudge decision was made "
                f"{age_seconds // 60} minutes ago."
            )
            return normalize_decision(suppressed)
    return decision


def save_decision(paths: AgentPaths, decision: dict[str, Any], context: dict[str, Any], qwen_error: str | None) -> dict[str, Any]:
    record = {
        "created_at": utc_now(),
        "decision": decision,
        "qwen_error": qwen_error,
        "user_settings": context.get("user_settings", {}),
        "retrieval": {
            "strategy": context.get("retrieval_strategy"),
            "baseline_policy": context.get("baseline_policy"),
            "posture_analysis_count": len(context.get("posture_context", {}).get("analyses", [])),
            "object_snapshot_count": len(context.get("object_context", {}).get("snapshots", [])),
            "dwell_candidate_count": len(context.get("object_context", {}).get("dwell_candidates", [])),
            "history_count": len(context.get("nudge_history", {}).get("decisions", [])),
            "dynamic_tool_calls": context.get("dynamic_tool_calls", []),
            "local_rule_evaluation": context.get("local_rule_evaluation"),
        },
    }
    append_jsonl(paths.decisions_path, record)
    paths.latest_decision_path.parent.mkdir(parents=True, exist_ok=True)
    paths.latest_decision_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide whether an ADHD/productivity nudge is needed.")
    parser.add_argument("--baseline", default="baseline.json", help="Baseline JSON path.")
    parser.add_argument("--long-monitor-dir", default="monitor_data", help="Directory from long_monitor.py.")
    parser.add_argument("--object-monitor-dir", default="object_monitor_data", help="Directory from object_monitor.py.")
    parser.add_argument("--agent-data-dir", default="nudge_agent_data", help="Directory for nudge decisions.")
    parser.add_argument("--lookback-hours", type=float, default=4.0, help="Retrieval lookback window.")
    parser.add_argument("--cooldown-minutes", type=int, default=45, help="Suppress repeated similar nudges.")
    parser.add_argument("--include-raw", action="store_true", help="Include compact raw posture summary.")
    parser.add_argument(
        "--mode",
        choices=("auto", "qwen", "local"),
        default="auto",
        help="auto tries Qwen then local rules; qwen disables local fallback; local never calls Qwen.",
    )
    parser.add_argument("--local", action="store_true", help="Alias for --mode local.")
    parser.add_argument("--rule-based", action="store_true", help="Deprecated alias for --mode local.")
    parser.add_argument("--skip-qwen", action="store_true", help="Deprecated alias for --mode local.")
    parser.add_argument("--max-tool-rounds", type=int, default=8, help="Maximum Qwen tool-calling rounds.")
    parser.add_argument("--print-context", action="store_true", help="Print retrieved context with the decision.")
    parser.add_argument("--watch", action="store_true", help="Run continuously instead of making one decision.")
    parser.add_argument("--interval-seconds", type=int, default=30, help="Seconds between decisions in --watch mode.")
    return parser


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    paths = AgentPaths(
        baseline=Path(args.baseline),
        long_monitor_dir=Path(args.long_monitor_dir),
        object_monitor_dir=Path(args.object_monitor_dir),
        agent_data_dir=Path(args.agent_data_dir),
    )
    tools = DataTools(paths, args.lookback_hours)
    mode = "local" if args.local or args.rule_based or args.skip_qwen else args.mode
    user_settings = read_json(paths.session_settings_path) or {}

    qwen_error = None
    if mode == "local":
        decision, context = local_decision(tools, user_settings=user_settings, include_raw=True)
    else:
        try:
            decision, retrieval = call_qwen_tool_agent(tools, max_rounds=args.max_tool_rounds, user_settings=user_settings)
            context = {
                "retrieval_strategy": {
                    "summary": "Dynamic Qwen tool-calling over bounded JSONL RAG tools.",
                    "lookback_seconds": tools.lookback_seconds,
                },
                "baseline_policy": tools.baseline_policy(),
                "posture_context": {},
                "object_context": {},
                "nudge_history": tools.recent_nudge_history(),
                "dynamic_tool_calls": retrieval.get("tool_calls", []),
                "user_settings": user_settings,
            }
        except Exception as exc:
            qwen_error = str(exc)
            if mode == "qwen":
                context = build_context(tools, args.include_raw, user_settings)
                decision = normalize_decision(
                    {
                        "should_nudge": False,
                        "category": "none",
                        "rationale": "Qwen nudge agent failed and strict qwen mode does not permit local fallback.",
                        "recommended_focus": "",
                        "supporting_signals": [],
                        "suppress_reason": "Qwen unavailable in qwen-only mode.",
                        "cooldown_key": "none",
                        "suggested_recheck_minutes": 15,
                    }
                )
            else:
                decision, context = local_decision(tools, user_settings=user_settings, include_raw=True)
                context["retrieval_strategy"]["fallback_reason"] = qwen_error

    decision = apply_cooldown(decision, context["nudge_history"], args.cooldown_minutes)
    record = save_decision(paths, decision, context, qwen_error)
    output = record if args.print_context else {"created_at": record["created_at"], "decision": record["decision"], "qwen_error": qwen_error}
    return output


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    if args.watch:
        print(
            "nudge_agent watch mode "
            f"interval={args.interval_seconds}s "
            f"mode={'local' if args.local or args.rule_based or args.skip_qwen else args.mode}",
            flush=True,
        )
        try:
            while True:
                output = run_once(args)
                decision = output["decision"]
                print(
                    "[nudge_agent] "
                    f"{output['created_at']} "
                    f"should_nudge={decision.get('should_nudge')} "
                    f"category={decision.get('category')} "
                    f"focus={decision.get('recommended_focus')} "
                    f"suppress_reason={decision.get('suppress_reason')}",
                    flush=True,
                )
                time.sleep(args.interval_seconds)
        except KeyboardInterrupt:
            print("\nStopping nudge_agent watch mode.")
            return 130

    print(json.dumps(run_once(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
