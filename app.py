"""Unified laptop app for monitoring and nudge orchestration.

Start this on the laptop, then run pi_start.py on the Raspberry Pi. This single
server receives posture windows, schedules object captures, runs Qwen posture
analysis, and periodically runs the nudge decision agent.
"""

from __future__ import annotations

import argparse
import hmac
import json
import mimetypes
import os
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from baseline_calibration import (
    build_baseline as build_calibration_baseline,
    classify_objects_locally,
    classify_objects_with_qwen,
    compact_objects,
    write_baseline,
)
from long_monitor import (
    MonitorStore,
    POSTURE_EVENT_TYPE,
    discover_laptop_ip,
    load_dotenv,
    read_json,
)
from local_fallback import local_decision
from edge_privacy import (
    build_privacy_ledger,
    build_memory_profile,
    latest_decision_trace,
    remember_session_summary,
    write_decision_trace,
    write_privacy_ledger,
)
from nudge import (
    AgentPaths,
    DataTools,
    apply_cooldown,
    build_context,
    call_qwen_tool_agent,
    normalize_decision,
    save_decision,
)
from nudge_copywriter import write_notification
from object_monitor import (
    OBJECT_EVENT_TYPES,
    ObjectMonitorStore,
    baseline_labels,
)
from qwen_config import qwen_model_config
from session_summary import build_session_summary, write_session_summary


DEFAULT_TOKEN = os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token")
WEB_APP_DIR = Path(__file__).resolve().parent / "web_app"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_SESSION_SETTINGS = {
    "active": False,
    "intent": "",
    "tone": "neutral",
    "notification_level": "minimal",
    "focus_areas": ["posture", "restlessness", "movement", "declutter"],
    "pomodoro_minutes": 25,
    "break_minutes": 5,
    "work_struggles": {
        "selected": [],
        "notes": "",
        "support_preferences": "",
    },
    "started_at": None,
    "ends_at": None,
}

QUESTIONNAIRE_CHOICES = {
    "getting_started",
    "staying_focused",
    "hyperfocus",
    "restlessness",
    "forgetting_breaks",
    "desk_clutter",
    "posture",
    "task_switching",
}


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def utc_now_after_seconds(seconds: int) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def read_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_SESSION_SETTINGS)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_SESSION_SETTINGS)
    settings = dict(DEFAULT_SESSION_SETTINGS)
    if isinstance(loaded, dict):
        settings.update(loaded)
    return normalize_session_settings(settings)


def normalize_session_settings(settings: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(DEFAULT_SESSION_SETTINGS)
    normalized.update(settings)
    if normalized["tone"] not in {"neutral", "funny", "strict", "calm", "encouraging"}:
        normalized["tone"] = "neutral"
    if normalized["notification_level"] not in {"minimal", "balanced", "active"}:
        normalized["notification_level"] = "minimal"
    if not isinstance(normalized.get("focus_areas"), list):
        normalized["focus_areas"] = []
    normalized["focus_areas"] = [
        str(item)
        for item in normalized["focus_areas"]
        if str(item) in {"posture", "restlessness", "movement", "declutter", "breaks", "focus"}
    ]
    for key in ("pomodoro_minutes", "break_minutes"):
        try:
            normalized[key] = max(1, int(normalized[key]))
        except (TypeError, ValueError):
            normalized[key] = DEFAULT_SESSION_SETTINGS[key]
    normalized["active"] = bool(normalized.get("active"))
    normalized["intent"] = str(normalized.get("intent") or "")[:500]
    raw_struggles = normalized.get("work_struggles")
    if not isinstance(raw_struggles, dict):
        raw_struggles = {}
    selected = raw_struggles.get("selected")
    if not isinstance(selected, list):
        selected = []
    normalized["work_struggles"] = {
        "selected": [
            str(item)
            for item in selected
            if str(item) in QUESTIONNAIRE_CHOICES
        ],
        "notes": str(raw_struggles.get("notes") or "")[:1000],
        "support_preferences": str(raw_struggles.get("support_preferences") or "")[:1000],
    }
    return normalized


def empty_baseline() -> dict[str, Any]:
    return {
        "kind": "flowpilot_baseline",
        "version": 1,
        "created_at": None,
        "calibration": {"events": []},
        "object_detection": {"objects": [], "source_event_count": 0},
        "object_policy": {
            "whitelisted_objects": [],
            "monitorable_objects": [],
            "uncertain_objects": [],
            "notes": "No calibration baseline has been created yet.",
        },
    }


def summarize_calibration_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
    return {
        "total_events": len(events),
        "event_counts": counts,
        "object_labels": [item["label"] for item in compact_objects(events)],
    }


def baseline_is_ready(baseline: dict[str, Any]) -> bool:
    if not baseline.get("created_at"):
        return False
    calibration = baseline.get("calibration") if isinstance(baseline.get("calibration"), dict) else {}
    events = calibration.get("events") if isinstance(calibration.get("events"), list) else []
    return bool(events)


class AppStore:
    def __init__(
        self,
        token: str,
        baseline_path: Path,
        baseline: dict[str, Any],
        monitor_data_dir: Path,
        object_monitor_data_dir: Path,
        nudge_agent_data_dir: Path,
        posture_analysis_interval_seconds: int,
        object_capture_interval_seconds: int,
        nudge_interval_seconds: int,
        nudge_cooldown_minutes: int,
        nudge_lookback_hours: float,
        include_raw_for_nudge: bool,
        nudge_mode: str,
        calibration_seconds: int,
        public_base_url: str | None = None,
    ) -> None:
        whitelist_labels = baseline_labels(baseline, "whitelisted_objects")
        monitor_labels = baseline_labels(baseline, "monitorable_objects")
        self.token = token
        self.public_base_url = public_base_url.rstrip("/") if public_base_url else None
        self.baseline_path = baseline_path
        self.baseline = baseline
        self.posture_store = MonitorStore(
            token=token,
            baseline=baseline,
            data_dir=monitor_data_dir,
            analysis_interval_seconds=posture_analysis_interval_seconds,
            use_qwen=nudge_mode != "local",
        )
        self.object_store = ObjectMonitorStore(
            token=token,
            baseline=baseline,
            data_dir=object_monitor_data_dir,
            capture_interval_seconds=object_capture_interval_seconds,
            whitelist_labels=whitelist_labels,
            monitor_labels=monitor_labels,
        )
        self.posture_store.analysis_enabled = self.is_focus_session_active
        self.agent_paths = AgentPaths(
            baseline=baseline_path,
            long_monitor_dir=monitor_data_dir,
            object_monitor_dir=object_monitor_data_dir,
            agent_data_dir=nudge_agent_data_dir,
        )
        self.nudge_interval_seconds = nudge_interval_seconds
        self.nudge_cooldown_minutes = nudge_cooldown_minutes
        self.nudge_lookback_hours = nudge_lookback_hours
        self.include_raw_for_nudge = include_raw_for_nudge
        self.nudge_mode = nudge_mode
        self.settings_path = nudge_agent_data_dir / "session_settings.json"
        self.session_settings = read_settings(self.settings_path)
        if self.session_settings.get("active"):
            self.session_settings["active"] = False
            self.session_settings["ended_at"] = utc_now()
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(json.dumps(self.session_settings, indent=2) + "\n", encoding="utf-8")
        self.calibration_seconds = calibration_seconds
        self.calibration_lock = threading.Lock()
        baseline_ready = baseline_is_ready(baseline)
        self.calibration: dict[str, Any] = {
            "active": False,
            "status": "complete" if baseline_ready else "idle",
            "reason": None,
            "started_at": None,
            "ends_at": None,
            "seconds": calibration_seconds,
            "events": [],
            "pending_commands": [],
            "dispatched_commands": [],
            "midpoint_capture_queued": False,
            "last_error": None,
            "completed_at": baseline.get("created_at") if baseline_ready else None,
            "latest_baseline_path": str(baseline_path),
        }
        self._stop = threading.Event()

    def start_background_tasks(self) -> None:
        threading.Thread(target=self.posture_store.analysis_loop, daemon=True).start()
        threading.Thread(target=self.nudge_loop, daemon=True).start()
        print(
            "[app] background tasks started: "
            f"posture_analysis_every={self.posture_store.analysis_interval_seconds}s "
            f"object_capture_every={self.object_store.capture_interval_seconds}s "
            f"nudge_every={self.nudge_interval_seconds}s",
            flush=True,
        )

    def start_calibration(self, seconds: int | None = None, reason: str = "manual") -> dict[str, Any]:
        duration = max(1, int(seconds or self.calibration_seconds))
        now_monotonic = time.monotonic()
        with self.calibration_lock:
            if self.calibration.get("status") in {"collecting", "classifying_objects"}:
                return {
                    "ok": False,
                    "reason": "calibration already active",
                    "calibration": self.calibration_status_locked(),
                }
            self.calibration = {
                "active": True,
                "status": "collecting",
                "reason": reason,
                "started_at": utc_now(),
                "ends_at": utc_now_after_seconds(duration),
                "seconds": duration,
                "events": [],
                "pending_commands": [],
                "dispatched_commands": [],
                "midpoint_capture_queued": False,
                "midpoint_capture_at": now_monotonic + duration / 2,
                "last_error": None,
                "latest_baseline_path": str(self.baseline_path),
            }
            status = self.calibration_status_locked()
        threading.Thread(target=self._finish_calibration_after, args=(duration,), daemon=True).start()
        print(f"[app] calibration started reason={reason} seconds={duration}", flush=True)
        return {"ok": True, "calibration": status}

    def _finish_calibration_after(self, seconds: int) -> None:
        if self._stop.wait(seconds + 2):
            return
        self.finish_calibration()

    def finish_calibration(self) -> None:
        with self.calibration_lock:
            if not self.calibration.get("active"):
                return
            self.calibration["active"] = False
            self.calibration["status"] = "classifying_objects"
            snapshot = self.calibration_snapshot_locked()

        events = snapshot.get("events") if isinstance(snapshot.get("events"), list) else []
        objects = compact_objects(events)
        qwen_policy: dict[str, Any] = classify_objects_locally(objects)
        qwen_error = None
        if self.nudge_mode != "local":
            try:
                qwen_policy = classify_objects_with_qwen(
                    objects,
                    context=json.dumps(
                        {
                            "intent": self.session_settings.get("intent"),
                            "work_struggles": self.session_settings.get("work_struggles"),
                            "focus_areas": self.session_settings.get("focus_areas"),
                        },
                        indent=2,
                    ),
                )
            except Exception as exc:
                qwen_error = str(exc)
        else:
            qwen_error = "Qwen object classification skipped in local mode."

        baseline = build_calibration_baseline(snapshot, qwen_policy, qwen_error)
        write_baseline(self.baseline_path, baseline)
        self.refresh_baseline(baseline)
        with self.calibration_lock:
            self.calibration["status"] = "complete"
            self.calibration["completed_at"] = utc_now()
            self.calibration["last_error"] = qwen_error
            self.calibration["latest_baseline_path"] = str(self.baseline_path)
        print(
            "[app] calibration complete "
            f"events={len(events)} object_labels={len(objects)} "
            f"baseline={self.baseline_path} qwen_error={qwen_error}",
            flush=True,
        )

    def refresh_baseline(self, baseline: dict[str, Any]) -> None:
        self.baseline = baseline
        self.posture_store.baseline = baseline
        whitelist_labels = baseline_labels(baseline, "whitelisted_objects")
        monitor_labels = baseline_labels(baseline, "monitorable_objects")
        with self.object_store._lock:
            self.object_store.baseline = baseline
            self.object_store.whitelist_labels = whitelist_labels
            self.object_store.monitor_labels = monitor_labels
            self.object_store.write_state_locked()

    def stop(self) -> None:
        self._stop.set()
        self.posture_store.stop()

    def add_event(self, event: dict[str, Any]) -> dict[str, Any]:
        if self.is_calibration_collecting():
            sequence = self.add_calibration_event(event)
            print(
                f"[app] stored calibration event #{sequence} type={event.get('event_type')}",
                flush=True,
            )
            return {"ok": True, "stored_as": "calibration_event", "sequence": sequence}
        event_type = event.get("event_type")
        if not self.is_focus_session_active():
            return {
                "ok": True,
                "ignored": True,
                "reason": "monitoring inactive; start a focus session to store posture/object events",
                "event_type": event_type,
            }
        if event_type == POSTURE_EVENT_TYPE:
            sequence = self.posture_store.add_event(event)
            print(f"[app] stored posture window #{sequence}", flush=True)
            return {"ok": True, "stored_as": "posture_window", "sequence": sequence}
        if event_type in OBJECT_EVENT_TYPES:
            sequence = self.object_store.add_event(event)
            print(f"[app] stored object snapshot #{sequence}", flush=True)
            return {"ok": True, "stored_as": "object_snapshot", "sequence": sequence}
        return {"ok": True, "ignored": True, "reason": f"unsupported event_type {event_type}"}

    def pop_command(self) -> dict[str, Any] | None:
        command = self.pop_calibration_command()
        if command is not None:
            return command
        if self.is_calibration_busy():
            return None
        if not self.is_focus_session_active():
            return None
        command = self.object_store.pop_command()
        return command

    def is_calibration_collecting(self) -> bool:
        with self.calibration_lock:
            return bool(self.calibration.get("active") and self.calibration.get("status") == "collecting")

    def is_calibration_busy(self) -> bool:
        with self.calibration_lock:
            return self.calibration.get("status") in {"collecting", "classifying_objects"}

    def is_focus_session_active(self) -> bool:
        return bool(self.session_settings.get("active")) and not self.is_calibration_busy()

    def add_calibration_event(self, event: dict[str, Any]) -> int:
        with self.calibration_lock:
            events = self.calibration.setdefault("events", [])
            event = dict(event)
            event["_sequence"] = len(events) + 1
            event["_received_at"] = utc_now()
            events.append(event)
            return int(event["_sequence"])

    def pop_calibration_command(self) -> dict[str, Any] | None:
        with self.calibration_lock:
            if not self.calibration.get("active"):
                return None
            self.queue_midpoint_capture_locked()
            pending = self.calibration.get("pending_commands")
            if not isinstance(pending, list) or not pending:
                return None
            command = pending.pop(0)
            dispatched = dict(command)
            dispatched["dispatched_at"] = utc_now()
            dispatched_commands = self.calibration.setdefault("dispatched_commands", [])
            dispatched_commands.append(dispatched)
            self.calibration["dispatched_commands"] = dispatched_commands[-20:]
            return command

    def queue_midpoint_capture_locked(self) -> None:
        if not self.calibration.get("active"):
            return
        if self.calibration.get("midpoint_capture_queued"):
            return
        midpoint = self.calibration.get("midpoint_capture_at")
        if not isinstance(midpoint, (int, float)) or time.monotonic() < midpoint:
            return
        command = {
            "id": str(uuid.uuid4()),
            "command_type": "object_capture",
            "seconds": 0,
            "created_at": utc_now(),
            "reason": "baseline_calibration_midpoint",
            "exclude_labels": [],
            "monitor_labels": [],
        }
        pending = self.calibration.setdefault("pending_commands", [])
        pending.append(command)
        self.calibration["midpoint_capture_queued"] = True
        print(f"[app] queued calibration object capture id={command['id']}", flush=True)

    def calibration_snapshot_locked(self) -> dict[str, Any]:
        events = list(self.calibration.get("events", []))
        return {
            "version": 1,
            "session": self.calibration_status_locked(),
            "commands": {
                "pending": list(self.calibration.get("pending_commands", [])),
                "dispatched": list(self.calibration.get("dispatched_commands", [])),
            },
            "events": events,
            "summary": summarize_calibration_events(events),
        }

    def calibration_status(self) -> dict[str, Any]:
        with self.calibration_lock:
            return self.calibration_status_locked()

    def calibration_status_locked(self) -> dict[str, Any]:
        self.queue_midpoint_capture_locked()
        event_count = len(self.calibration.get("events", []))
        return {
            "active": bool(self.calibration.get("active")),
            "status": self.calibration.get("status"),
            "reason": self.calibration.get("reason"),
            "started_at": self.calibration.get("started_at"),
            "ends_at": self.calibration.get("ends_at"),
            "seconds": self.calibration.get("seconds"),
            "event_count": event_count,
            "pending_commands": len(self.calibration.get("pending_commands", [])),
            "dispatched_commands": len(self.calibration.get("dispatched_commands", [])),
            "midpoint_capture_queued": bool(self.calibration.get("midpoint_capture_queued")),
            "last_error": self.calibration.get("last_error"),
            "completed_at": self.calibration.get("completed_at"),
            "latest_baseline_path": self.calibration.get("latest_baseline_path"),
        }

    def status(self) -> dict[str, Any]:
        return {
            "monitoring": {
                "focus_session_active": self.is_focus_session_active(),
                "accepting_monitor_events": self.is_focus_session_active(),
                "accepting_calibration_events": self.is_calibration_collecting(),
                "qwen_background_allowed": self.is_focus_session_active(),
            },
            "posture_monitor": self.posture_store.status(),
            "object_monitor": self.object_status(),
            "calibration": self.calibration_status(),
            "nudge_agent": {
                "mode": self.nudge_mode,
                "qwen_models": qwen_model_config() if self.nudge_mode != "local" else {},
                "interval_seconds": self.nudge_interval_seconds,
                "cooldown_minutes": self.nudge_cooldown_minutes,
                "effective_cooldown_minutes": self.effective_cooldown_minutes(),
                "latest_decision_path": str(self.agent_paths.latest_decision_path),
                "latest_notification_path": str(self.agent_paths.agent_data_dir / "latest_notification.json"),
                "latest_trace_path": str(self.agent_paths.agent_data_dir / "latest_decision_trace.json"),
                "memory_path": str(self.agent_paths.agent_data_dir / "agent_memory.json"),
            },
            "privacy": self.privacy_ledger(compact=True),
            "session_settings": self.session_settings,
        }

    def privacy_ledger(self, compact: bool = False) -> dict[str, Any]:
        runtime_status = {
            "focus_session_active": self.is_focus_session_active(),
            "calibration_status": self.calibration.get("status"),
            "nudge_mode": self.nudge_mode,
        }
        ledger = (
            build_privacy_ledger(self.agent_paths, self.nudge_mode, runtime_status)
            if compact
            else write_privacy_ledger(self.agent_paths, self.nudge_mode, runtime_status)
        )
        if compact:
            return {
                "raw_video_sent_off_device": ledger["hardware_first"]["raw_video_sent_off_device"],
                "raw_frames_persisted_by_backend": ledger["hardware_first"]["raw_frames_persisted_by_backend"],
                "cloud_provider": ledger["data_boundaries"]["cloud_provider"],
                "cloud_attempted_for_latest_decision": ledger["data_boundaries"]["cloud_attempted_for_latest_decision"],
                "memory_scope": ledger["retention_model"]["memory_scope"],
            }
        return ledger

    def explainability_trace(self) -> dict[str, Any]:
        return latest_decision_trace(self.agent_paths)

    def memory_profile(self) -> dict[str, Any]:
        return build_memory_profile(self.agent_paths)

    def history_rag_search(self, query: str, limit: int = 6, lookback_days: float = 30) -> dict[str, Any]:
        tools = DataTools(self.agent_paths, self.nudge_lookback_hours)
        return tools.search_history_rag(query=query, limit=limit, lookback_days=lookback_days)

    def connect_info(self, port: int) -> dict[str, Any]:
        api_base = self.public_base_url or f"http://{discover_laptop_ip()}:{port}"
        return {
            "api_base": api_base,
            "token": self.token,
            "pi_command": f"python pi_start.py --laptop-api-base {api_base} --token {self.token} --download-object-model --download-pose-model",
            "webcam_command": (
                f"python webcam_edge.py --laptop-api-base {api_base} "
                f"--token {self.token} --download-object-model --download-pose-model --debug-stream"
            ),
        }

    def object_status(self) -> dict[str, Any]:
        if self.is_focus_session_active():
            return self.object_store.status()
        with self.object_store._lock:
            return {
                "snapshots": len(self.object_store.snapshots),
                "pending_commands": len(self.object_store.pending_commands),
                "dispatched_commands": len(self.object_store.dispatched_commands),
                "capture_interval_seconds": self.object_store.capture_interval_seconds,
                "seconds_until_next_capture": max(0, int(self.object_store._next_capture_at - time.monotonic())),
                "whitelist_labels": self.object_store.whitelist_labels,
                "monitor_labels": self.object_store.monitor_labels,
                "latest_snapshot": self.object_store.snapshots[-1] if self.object_store.snapshots else None,
                "paused_for_calibration": self.is_calibration_busy(),
                "paused_until_focus_session": not self.session_settings.get("active"),
            }

    def reset_monitoring_window(self, *, clear_object_commands: bool) -> None:
        with self.posture_store._lock:
            self.posture_store._last_analysis_at = time.monotonic()
            self.posture_store._interval_started_at = utc_now()
            self.posture_store._last_analysis_index = len(self.posture_store.events)
        with self.object_store._lock:
            if clear_object_commands:
                self.object_store.pending_commands.clear()
            self.object_store._next_capture_at = time.monotonic()
            self.object_store.write_state_locked()

    def update_session_settings(self, updates: dict[str, Any]) -> dict[str, Any]:
        was_active = bool(self.session_settings.get("active"))
        settings = dict(self.session_settings)
        settings.update(updates)
        if updates.get("active") is True and not settings.get("started_at"):
            settings["started_at"] = utc_now()
        if updates.get("active") is False:
            settings["ended_at"] = utc_now()
        self.session_settings = normalize_session_settings(settings)
        is_active = bool(self.session_settings.get("active"))
        if is_active != was_active:
            self.reset_monitoring_window(clear_object_commands=True)
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        self.settings_path.write_text(json.dumps(self.session_settings, indent=2) + "\n", encoding="utf-8")
        print(f"[app] session settings updated {self.session_settings}", flush=True)
        return self.session_settings

    def nudge_loop(self) -> None:
        while not self._stop.wait(self.nudge_interval_seconds):
            self.run_nudge_once()

    def run_nudge_once(self) -> None:
        if self.is_calibration_busy():
            print("[app] skipped nudge cycle while calibration is active", flush=True)
            return
        if not self.session_settings.get("active"):
            return
        print("[app] running nudge agent decision cycle", flush=True)
        tools = DataTools(self.agent_paths, self.nudge_lookback_hours)
        context = build_context(tools, self.include_raw_for_nudge, self.session_settings)
        qwen_error = None
        if self.nudge_mode == "local":
            decision, context = local_decision(tools, self.session_settings, include_raw=True)
        else:
            try:
                decision, retrieval = call_qwen_tool_agent(tools, user_settings=self.session_settings)
                context = {
                    "retrieval_strategy": {
                        "summary": "Dynamic Qwen tool-calling over bounded JSONL RAG tools.",
                        "lookback_seconds": tools.lookback_seconds,
                    },
                    "baseline_policy": tools.baseline_policy(),
                    "posture_context": {},
                    "object_context": {},
                    "nudge_history": tools.recent_nudge_history(),
                    "memory_profile": tools.memory_profile(),
                    "dynamic_tool_calls": retrieval.get("tool_calls", []),
                    "user_settings": self.session_settings,
                }
            except Exception as exc:
                qwen_error = str(exc)
                if self.nudge_mode == "qwen":
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
                    decision, context = local_decision(tools, self.session_settings, include_raw=True)
                    context["retrieval_strategy"]["fallback_reason"] = qwen_error
        decision = apply_cooldown(decision, context["nudge_history"], self.effective_cooldown_minutes())
        record = save_decision(self.agent_paths, decision, context, qwen_error)
        write_decision_trace(self.agent_paths, record, context, self.nudge_mode, qwen_error)
        self.privacy_ledger(compact=False)
        print(
            "[app] nudge decision "
            f"should_nudge={record['decision'].get('should_nudge')} "
            f"category={record['decision'].get('category')} "
            f"focus={record['decision'].get('recommended_focus')}",
            flush=True,
        )
        if qwen_error:
            if self.nudge_mode == "qwen":
                print(f"[app] nudge agent skipped local fallback in qwen mode after error: {qwen_error}", flush=True)
            else:
                print(f"[app] nudge agent used local fallback because Qwen failed: {qwen_error}", flush=True)
        if record["decision"].get("should_nudge"):
            notification = write_notification(
                record,
                self.agent_paths.agent_data_dir,
                self.session_settings,
                use_qwen=self.nudge_mode != "local" and qwen_error is None,
            )
            print(
                "[app] notification copy "
                f"title={notification['notification'].get('title')} "
                f"message={notification['notification'].get('message')}",
                flush=True,
            )

    def create_session_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else self.session_settings
        started_at = payload.get("started_at") or settings.get("started_at")
        ended_at = payload.get("ended_at") or utc_now()
        summary = build_session_summary(
            monitor_data_dir=self.agent_paths.long_monitor_dir,
            agent_data_dir=self.agent_paths.agent_data_dir,
            settings=normalize_session_settings(settings),
            started_at=started_at,
            ended_at=ended_at,
            use_qwen=self.nudge_mode != "local",
        )
        write_session_summary(summary, self.agent_paths.agent_data_dir)
        remember_session_summary(self.agent_paths, summary)
        print(
            "[app] session summary created "
            f"qwen_error={summary.get('qwen_error')} paragraph={summary.get('paragraph')}",
            flush=True,
        )
        return summary

    def effective_cooldown_minutes(self) -> int:
        level = self.session_settings.get("notification_level")
        if level == "active":
            return min(self.nudge_cooldown_minutes, 10)
        if level == "balanced":
            return min(self.nudge_cooldown_minutes, 20)
        return self.nudge_cooldown_minutes


class AppHandler(BaseHTTPRequestHandler):
    server: "AppHTTPServer"

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/app", "/app/"}:
            self._send_static_file(WEB_APP_DIR / "index.html", WEB_APP_DIR)
        elif path.startswith("/app/"):
            relative = unquote(path.removeprefix("/app/")) or "index.html"
            self._send_static_file(WEB_APP_DIR / relative, WEB_APP_DIR)
        elif path.startswith("/assets/"):
            relative = unquote(path.removeprefix("/assets/"))
            self._send_static_file(ASSETS_DIR / relative, ASSETS_DIR)
        elif path in {"/", "/health"}:
            self._send_json({"ok": True, "status": self.server.store.status()})
        elif path == "/api/status":
            self._send_json(self.server.store.status())
        elif path == "/api/connect-info":
            self._send_json({"ok": True, "connect_info": self.server.store.connect_info(self.server.server_address[1])})
        elif path == "/api/session-settings":
            self._send_json({"ok": True, "settings": self.server.store.session_settings})
        elif path == "/api/calibration":
            self._send_json({"ok": True, "calibration": self.server.store.calibration_status()})
        elif path == "/api/latest-notification":
            notification_path = self.server.store.agent_paths.agent_data_dir / "latest_notification.json"
            payload = read_json(notification_path) if notification_path.exists() else None
            self._send_json({"ok": True, "notification": payload})
        elif path == "/api/latest-session-summary":
            summary_path = self.server.store.agent_paths.agent_data_dir / "latest_session_summary.json"
            payload = read_json(summary_path) if summary_path.exists() else None
            self._send_json({"ok": True, "summary": payload})
        elif path == "/api/privacy-ledger":
            self._send_json({"ok": True, "privacy": self.server.store.privacy_ledger()})
        elif path == "/api/explainability":
            self._send_json({"ok": True, "trace": self.server.store.explainability_trace()})
        elif path == "/api/memory-profile":
            self._send_json({"ok": True, "memory": self.server.store.memory_profile()})
        elif path == "/api/history-rag":
            query = parse_qs(urlparse(self.path).query)
            search_query = (query.get("q") or query.get("query") or [""])[0]
            try:
                limit = int((query.get("limit") or ["6"])[0])
            except ValueError:
                limit = 6
            try:
                lookback_days = float((query.get("lookback_days") or ["30"])[0])
            except ValueError:
                lookback_days = 30
            self._send_json(
                {
                    "ok": True,
                    "history_rag": self.server.store.history_rag_search(
                        search_query,
                        limit=limit,
                        lookback_days=lookback_days,
                    ),
                }
            )
        elif path == "/pi/commands":
            self._send_next_command()
        elif path == "/sessions/active":
            self._send_json({"active": True, "commands_url": "/pi/commands", "events_url": "/events"})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/events":
            self._receive_event()
        elif path == "/api/nudge-now":
            self.server.store.run_nudge_once()
            self._send_json({"ok": True, "status": self.server.store.status()})
        elif path == "/api/session-settings":
            payload = self._read_json()
            if payload is None:
                return
            settings = self.server.store.update_session_settings(payload)
            self._send_json({"ok": True, "settings": settings})
        elif path == "/api/recalibrate":
            payload = self._read_json()
            if payload is None:
                return
            seconds = payload.get("seconds")
            result = self.server.store.start_calibration(seconds=seconds, reason="manual")
            status = HTTPStatus.OK if result.get("ok") else HTTPStatus.CONFLICT
            self._send_json(result, status=status)
        elif path == "/api/session-summary":
            payload = self._read_json()
            if payload is None:
                return
            summary = self.server.store.create_session_summary(payload)
            self._send_json({"ok": True, "summary": summary})
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/pi/commands"):
            return
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return None
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return None
        if not isinstance(payload, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "payload must be a JSON object")
            return None
        return payload

    def _check_auth(self) -> bool:
        expected = f"Bearer {self.server.store.token}"
        received = self.headers.get("Authorization", "")
        if not hmac.compare_digest(received, expected):
            self.send_error(HTTPStatus.UNAUTHORIZED, "invalid bearer token")
            return False
        return True

    def _receive_event(self) -> None:
        if not self._check_auth():
            return
        event = self._read_json()
        if event is None:
            return
        self._send_json(self.server.store.add_event(event))

    def _send_next_command(self) -> None:
        if not self._check_auth():
            return
        command = self.server.store.pop_command()
        if command is not None:
            print(
                "[app] dispatched object capture "
                f"id={command.get('id')} excluded={command.get('exclude_labels')}",
                flush=True,
            )
        self._send_json({"command": command})

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            print(f"[app] client disconnected before response finished path={self.path}", flush=True)

    def _send_static_file(self, path: Path, root: Path) -> None:
        try:
            resolved_root = root.resolve()
            resolved_path = path.resolve()
            if resolved_root != resolved_path and resolved_root not in resolved_path.parents:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not resolved_path.exists() or not resolved_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            body = resolved_path.read_bytes()
            content_type = mimetypes.guess_type(str(resolved_path))[0] or "application/octet-stream"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if resolved_path.name in {"index.html", "service-worker.js", "app.js", "styles.css"}:
                self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            print(f"[app] client disconnected before static response finished path={self.path}", flush=True)


class AppHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        store: AppStore,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store = store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the unified FlowPilot monitoring app.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Unified laptop app port.")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token expected from the Pi.")
    parser.add_argument(
        "--public-base-url",
        default=os.getenv("FLOWPILOT_PUBLIC_BASE_URL"),
        help="Public URL sensors and browsers should use, for example http://your-ecs-public-ip.",
    )
    parser.add_argument("--baseline", default="baseline.json", help="Baseline JSON path.")
    parser.add_argument("--monitor-data-dir", default="monitor_data", help="Posture monitor output directory.")
    parser.add_argument("--object-monitor-data-dir", default="object_monitor_data", help="Object monitor output directory.")
    parser.add_argument("--nudge-agent-data-dir", default="nudge_agent_data", help="Nudge decision output directory.")
    parser.add_argument("--posture-analysis-interval", type=int, default=120, help="Seconds between Qwen posture analyses.")
    parser.add_argument("--object-capture-interval", type=int, default=120, help="Seconds between object captures.")
    parser.add_argument("--nudge-interval", type=int, default=30, help="Seconds between nudge decisions.")
    parser.add_argument("--nudge-cooldown-minutes", type=int, default=45, help="Suppress repeated similar nudges.")
    parser.add_argument("--nudge-lookback-hours", type=float, default=4.0, help="Retrieval lookback for nudge decisions.")
    parser.add_argument("--include-raw-for-nudge", action="store_true", help="Include compact raw posture fallback.")
    parser.add_argument(
        "--nudge-mode",
        choices=("auto", "qwen", "local"),
        default="auto",
        help="auto tries Qwen then local rules; qwen disables local fallback; local never calls Qwen for orchestration.",
    )
    parser.add_argument("--local", action="store_true", help="Alias for --nudge-mode local.")
    parser.add_argument("--skip-qwen-for-nudge", action="store_true", help="Deprecated alias for --local.")
    parser.add_argument("--calibration-seconds", type=int, default=120, help="Seconds for manual baseline calibration.")
    parser.add_argument("--startup-calibration", action="store_true", help="Run baseline calibration when app.py starts.")
    parser.add_argument("--skip-startup-calibration", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"[app] baseline not found: {baseline_path}; click Calibrate in the app to create it.", flush=True)
        baseline = empty_baseline()
    else:
        baseline = read_json(baseline_path)
        if baseline is None:
            print(f"[app] could not read baseline: {baseline_path}; click Calibrate in the app to replace it.", flush=True)
            baseline = empty_baseline()

    nudge_mode = "local" if args.local or args.skip_qwen_for_nudge else args.nudge_mode

    store = AppStore(
        token=args.token,
        baseline_path=baseline_path,
        baseline=baseline,
        monitor_data_dir=Path(args.monitor_data_dir),
        object_monitor_data_dir=Path(args.object_monitor_data_dir),
        nudge_agent_data_dir=Path(args.nudge_agent_data_dir),
        posture_analysis_interval_seconds=args.posture_analysis_interval,
        object_capture_interval_seconds=args.object_capture_interval,
        nudge_interval_seconds=args.nudge_interval,
        nudge_cooldown_minutes=args.nudge_cooldown_minutes,
        nudge_lookback_hours=args.nudge_lookback_hours,
        include_raw_for_nudge=args.include_raw_for_nudge,
        nudge_mode=nudge_mode,
        calibration_seconds=args.calibration_seconds,
        public_base_url=args.public_base_url,
    )
    server = AppHTTPServer((args.host, args.port), AppHandler, store)
    if args.startup_calibration and not args.skip_startup_calibration:
        store.start_calibration(seconds=args.calibration_seconds, reason="startup")
    store.start_background_tasks()

    laptop_ip = discover_laptop_ip()
    api_base = args.public_base_url.rstrip("/") if args.public_base_url else f"http://{laptop_ip}:{args.port}"
    app_url = f"{api_base}/app/" if args.public_base_url else f"http://127.0.0.1:{args.port}/app/"
    print(f"FlowPilot app: {app_url}")
    print(f"Pi API base:  {api_base}")
    print(f"Token:        {args.token}")
    print("")
    print("Orchestration:")
    print(f"  posture windows:       received from Pi and stored in {args.monitor_data_dir}/monitor_events.jsonl")
    if nudge_mode == "local":
        print(f"  posture local analysis: every {args.posture_analysis_interval}s into {args.monitor_data_dir}/qwen_analyses.jsonl")
    else:
        print(f"  posture Qwen analysis: every {args.posture_analysis_interval}s into {args.monitor_data_dir}/qwen_analyses.jsonl")
    print(f"  object captures:       every {args.object_capture_interval}s into {args.object_monitor_data_dir}/object_events.jsonl")
    print(f"  nudge agent:           mode={nudge_mode} every {args.nudge_interval}s into {args.nudge_agent_data_dir}/nudge_decisions.jsonl")
    if args.startup_calibration and not args.skip_startup_calibration:
        print(f"  startup calibration:   first {args.calibration_seconds}s will replace {baseline_path}")
    else:
        print("  calibration:           manual from the app")
    print("")
    print("Run the Pi with:")
    print(f"  python pi_start.py --laptop-api-base {api_base} --token {args.token} --download-object-model --download-pose-model")
    print("Or run the laptop webcam demo with:")
    print(f"  python webcam_edge.py --laptop-api-base {api_base} --token {args.token} --download-object-model --download-pose-model --debug-stream")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping app.")
    finally:
        store.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
