"""Laptop-side calibration receiver for FlowPilot Pi edge events.

Run this on the laptop, open the served HTML page, choose a capture duration,
and optionally launch the Raspberry Pi command over SSH. The Pi should send its
existing semantic events to this server's /events endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlparse


DEFAULT_TOKEN = os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token")
DEFAULT_PI_TARGET = os.getenv("FLOWPILOT_PI_SSH_TARGET", "pi@raspberrypi.local")
DEFAULT_REMOTE_TEMPLATE = (
    "cd ~/adhd-coach-new && "
    "timeout ${run_seconds}s "
    "python pi_start.py "
    "--laptop-api-base ${api_base} "
    "--token ${token} "
    "--download-object-model "
    "--download-pose-model "
    "--interval 1 "
    "--window-seconds ${run_seconds} "
    "--command-poll-interval 1"
)


@dataclass
class CalibrationSession:
    id: str
    seconds: int
    started_at: str
    ends_at_monotonic: float
    api_base: str
    token: str
    midpoint_capture_at_monotonic: float | None = None
    midpoint_capture_queued: bool = False
    pi_target: str | None = None
    command: str | None = None
    process_started: bool = False
    process_returncode: int | None = None
    process_error: str | None = None
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)

    def is_active(self) -> bool:
        return self.ends_at_monotonic > time.monotonic()

    def status(self) -> dict[str, Any]:
        remaining = max(0, int(round(self.ends_at_monotonic - time.monotonic())))
        return {
            "id": self.id,
            "seconds": self.seconds,
            "started_at": self.started_at,
            "active": self.is_active(),
            "remaining_seconds": remaining,
            "api_base": self.api_base,
            "token": self.token,
            "midpoint_capture_queued": self.midpoint_capture_queued,
            "pi_target": self.pi_target,
            "command": self.command,
            "process_started": self.process_started,
            "process_returncode": self.process_returncode,
            "process_error": self.process_error,
            "stdout_tail": self.stdout_tail[-20:],
            "stderr_tail": self.stderr_tail[-20:],
        }


class CalibrationStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._subscribers: list[threading.Condition] = []
        self._version = 0
        self.session: CalibrationSession | None = None
        self.pending_commands: list[dict[str, Any]] = []
        self.dispatched_commands: list[dict[str, Any]] = []

    def has_active_session(self) -> bool:
        with self._lock:
            return self.session is not None and self.session.is_active()

    def reset_session(self, session: CalibrationSession) -> None:
        with self._lock:
            self.session = session
            self._events = []
            self.pending_commands = []
            self.dispatched_commands = []
            self._version += 1
            self._notify_locked()

    def enqueue_object_capture(self, seconds: int) -> dict[str, Any]:
        command = {
            "id": str(uuid.uuid4()),
            "command_type": "object_capture",
            "seconds": seconds,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            if self.session is None or not self.session.is_active():
                raise RuntimeError("start calibration before requesting object capture")
            self.pending_commands.append(command)
            self._version += 1
            self._notify_locked()
            return command

    def pop_command(self) -> dict[str, Any] | None:
        with self._lock:
            if self.session is None or not self.session.is_active():
                return None
            self._queue_midpoint_capture_if_due_locked()
            if not self.pending_commands:
                return None
            command = self.pending_commands.pop(0)
            dispatched = dict(command)
            dispatched["dispatched_at"] = datetime.now(timezone.utc).isoformat()
            self.dispatched_commands.append(dispatched)
            self.dispatched_commands = self.dispatched_commands[-20:]
            self._version += 1
            self._notify_locked()
            return command

    def _queue_midpoint_capture_if_due_locked(self) -> None:
        if self.session is None:
            return
        if self.session.midpoint_capture_queued:
            return
        if self.session.midpoint_capture_at_monotonic is None:
            return
        if time.monotonic() < self.session.midpoint_capture_at_monotonic:
            return

        command = {
            "id": str(uuid.uuid4()),
            "command_type": "object_capture",
            "seconds": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "reason": "midpoint_calibration",
        }
        self.pending_commands.append(command)
        self.session.midpoint_capture_queued = True
        print(
            "[calibration] queued midpoint object capture "
            f"id={command.get('id')}",
            flush=True,
        )

    def add_event(self, event: dict[str, Any]) -> int:
        with self._lock:
            event["_received_at"] = datetime.now(timezone.utc).isoformat()
            event["_sequence"] = len(self._events) + 1
            self._events.append(event)
            self._version += 1
            self._notify_locked()
            return event["_sequence"]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self.session is not None and self.session.is_active():
                self._queue_midpoint_capture_if_due_locked()
            return self._snapshot_locked()

    def update_process(
        self,
        returncode: int | None = None,
        error: str | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
    ) -> None:
        with self._lock:
            if self.session is None:
                return
            if returncode is not None:
                self.session.process_returncode = returncode
            if error is not None:
                self.session.process_error = error
            if stdout:
                self.session.stdout_tail.extend(stdout.splitlines())
                self.session.stdout_tail = self.session.stdout_tail[-50:]
            if stderr:
                self.session.stderr_tail.extend(stderr.splitlines())
                self.session.stderr_tail = self.session.stderr_tail[-50:]
            self._version += 1
            self._notify_locked()

    def subscribe(self) -> threading.Condition:
        condition = threading.Condition(self._lock)
        with self._lock:
            self._subscribers.append(condition)
        return condition

    def unsubscribe(self, condition: threading.Condition) -> None:
        with self._lock:
            if condition in self._subscribers:
                self._subscribers.remove(condition)

    def wait_for_change(self, condition: threading.Condition, version: int, timeout: float) -> dict[str, Any]:
        with condition:
            if self._version == version:
                condition.wait(timeout=timeout)
            return self._snapshot_locked()

    def _notify_locked(self) -> None:
        for condition in self._subscribers:
            condition.notify_all()

    def _snapshot_locked(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "session": self.session.status() if self.session else None,
            "commands": {
                "pending": list(self.pending_commands),
                "dispatched": list(self.dispatched_commands),
            },
            "events": list(self._events),
            "summary": summarize_events(self._events),
        }


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    objects: dict[str, dict[str, Any]] = {}
    latest_posture: dict[str, Any] | None = None

    for event in events:
        event_type = str(event.get("event_type", "unknown"))
        counts[event_type] = counts.get(event_type, 0) + 1
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}

        if event_type in {"object_capture_snapshot", "object_calibration_snapshot", "object_surveillance_snapshot"}:
            for item in payload.get("objects", []):
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label", "unknown"))
                score = float(item.get("score", 0) or 0)
                current = objects.get(label)
                if current is None or score > current["score"]:
                    objects[label] = {
                        "label": label,
                        "score": score,
                        "bbox": item.get("bbox"),
                        "event_sequence": event.get("_sequence"),
                    }

        if event_type == "posture_window":
            series = payload.get("series") if isinstance(payload.get("series"), dict) else {}
            latest_posture = {
                "event_sequence": event.get("_sequence"),
                "window_start": payload.get("window_start"),
                "forward_head_ratio": last_value(series.get("forward_head_ratio")),
                "shoulder_tilt_ratio": last_value(series.get("shoulder_tilt_ratio")),
                "torso_lean_ratio": last_value(series.get("torso_lean_ratio")),
                "motion_score": last_value(series.get("motion_score")),
                "confidence": last_value(series.get("confidence")),
            }

    return {
        "counts": counts,
        "objects": sorted(objects.values(), key=lambda item: (-item["score"], item["label"])),
        "latest_posture": latest_posture,
        "total_events": len(events),
    }


def last_value(value: Any) -> Any:
    if isinstance(value, list) and value:
        return value[-1]
    return None


def discover_laptop_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def render_command(template: str, values: dict[str, str]) -> str:
    return Template(template).safe_substitute(values)


def launch_pi_over_ssh(store: CalibrationStore, target: str, command: str) -> None:
    def run() -> None:
        try:
            process = subprocess.Popen(
                ["ssh", target, command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            store.update_process(error=f"Could not start ssh: {exc}")
            return

        stdout, stderr = process.communicate()
        store.update_process(
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    threading.Thread(target=run, daemon=True).start()


def html_path() -> Path:
    return Path(__file__).with_name("calibration.html")


class CalibrationHandler(BaseHTTPRequestHandler):
    server: "CalibrationHTTPServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/calibration.html"}:
            self._send_html()
        elif path == "/api/status":
            self._send_json(self._with_config(self.server.store.snapshot()))
        elif path == "/api/stream":
            self._send_stream()
        elif path == "/pi/commands":
            self._send_next_pi_command()
        elif path == "/sessions/active":
            self._send_active_session()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/events":
            self._receive_event()
        elif path == "/api/start":
            self._start_session()
        elif path == "/api/object-capture":
            self._request_object_capture()
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            return None
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return None

    def _receive_event(self) -> None:
        session = self.server.store.session
        token = session.token if session is not None else self.server.token
        expected = f"Bearer {token}"
        if self.server.require_auth and self.headers.get("Authorization") != expected:
            self.send_error(HTTPStatus.UNAUTHORIZED, "invalid bearer token")
            return

        event = self._read_json()
        if event is None:
            return
        if not isinstance(event, dict):
            self.send_error(HTTPStatus.BAD_REQUEST, "event must be a JSON object")
            return

        if not self.server.store.has_active_session():
            print(
                "[calibration] ignored event outside active calibration "
                f"type={event.get('event_type', 'unknown')} source={event.get('source', 'unknown')}",
                flush=True,
            )
            self._send_json({"ok": True, "ignored": True, "reason": "no active calibration session"})
            return

        sequence = self.server.store.add_event(event)
        print(
            "[calibration] received event "
            f"#{sequence} type={event.get('event_type', 'unknown')} "
            f"source={event.get('source', 'unknown')}",
            flush=True,
        )
        self._send_json({"ok": True, "sequence": sequence})

    def _send_next_pi_command(self) -> None:
        session = self.server.store.session
        token = session.token if session is not None else self.server.token
        expected = f"Bearer {token}"
        if self.server.require_auth and self.headers.get("Authorization") != expected:
            self.send_error(HTTPStatus.UNAUTHORIZED, "invalid bearer token")
            return
        command = self.server.store.pop_command()
        if command is not None:
            print(
                "[calibration] dispatched command "
                f"id={command.get('id')} type={command.get('command_type')} seconds={command.get('seconds')}",
                flush=True,
            )
        self._send_json({"command": command})

    def _send_active_session(self) -> None:
        snapshot = self.server.store.snapshot()
        session = snapshot.get("session")
        active = bool(session and session.get("active"))
        self._send_json(
            {
                "active": active,
                "session": session,
                "commands_url": "/pi/commands",
                "events_url": "/events",
            }
        )

    def _start_session(self) -> None:
        body = self._read_json()
        if body is None:
            return

        try:
            seconds = int(body.get("seconds", 30))
        except (TypeError, ValueError):
            self.send_error(HTTPStatus.BAD_REQUEST, "seconds must be an integer")
            return
        if seconds < 1 or seconds > 3600:
            self.send_error(HTTPStatus.BAD_REQUEST, "seconds must be between 1 and 3600")
            return

        token = str(body.get("token") or self.server.token)
        api_base = str(body.get("api_base") or self.server.api_base)
        pi_target = str(body.get("pi_target") or "").strip() or None
        use_ssh = bool(body.get("use_ssh"))
        template = str(body.get("command_template") or DEFAULT_REMOTE_TEMPLATE)
        run_seconds = max(seconds + 10, 15)
        command = render_command(
            template,
            {
                "api_base": api_base,
                "token": token,
                "seconds": str(seconds),
                "run_seconds": str(run_seconds),
            },
        )

        session = CalibrationSession(
            id=str(uuid.uuid4()),
            seconds=seconds,
            started_at=datetime.now(timezone.utc).isoformat(),
            ends_at_monotonic=time.monotonic() + seconds,
            api_base=api_base,
            token=token,
            midpoint_capture_at_monotonic=time.monotonic() + (seconds / 2),
            pi_target=pi_target,
            command=command,
            process_started=bool(use_ssh and pi_target),
        )
        self.server.store.reset_session(session)

        if use_ssh and pi_target:
            launch_pi_over_ssh(self.server.store, pi_target, command)

        self._send_json(self._with_config(self.server.store.snapshot()))

    def _request_object_capture(self) -> None:
        body = self._read_json()
        if body is None:
            return
        try:
            seconds = int(body.get("seconds", 5))
        except (TypeError, ValueError):
            self.send_error(HTTPStatus.BAD_REQUEST, "seconds must be an integer")
            return
        if seconds < 0 or seconds > 3600:
            self.send_error(HTTPStatus.BAD_REQUEST, "seconds must be between 0 and 3600")
            return
        try:
            command = self.server.store.enqueue_object_capture(seconds)
        except RuntimeError as exc:
            self.send_error(HTTPStatus.CONFLICT, str(exc))
            return
        print(
            "[calibration] queued object capture "
            f"id={command.get('id')} seconds={seconds}",
            flush=True,
        )
        self._send_json({"ok": True, "command": command, "snapshot": self._with_config(self.server.store.snapshot())})

    def _send_html(self) -> None:
        try:
            content = html_path().read_bytes()
        except OSError:
            self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "calibration.html not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _with_config(self, snapshot: dict[str, Any]) -> dict[str, Any]:
        snapshot = dict(snapshot)
        snapshot["config"] = {
            "api_base": self.server.api_base,
            "token": self.server.token,
            "pi_target": DEFAULT_PI_TARGET,
            "command_template": DEFAULT_REMOTE_TEMPLATE,
        }
        return snapshot

    def _send_stream(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        condition = self.server.store.subscribe()
        version = -1
        try:
            while True:
                snapshot = self.server.store.wait_for_change(condition, version, timeout=15)
                version = int(snapshot["version"])
                data = json.dumps(snapshot)
                self.wfile.write(f"event: snapshot\ndata: {data}\n\n".encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.server.store.unsubscribe(condition)


class CalibrationHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        store: CalibrationStore,
        token: str,
        api_base: str,
        require_auth: bool,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store = store
        self.token = token
        self.api_base = api_base
        self.require_auth = require_auth


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the FlowPilot Pi calibration receiver UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind. Use 0.0.0.0 for Pi access.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port for the laptop receiver.")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token expected from the Pi.")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Accept /events without checking the Authorization header.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    laptop_ip = discover_laptop_ip()
    api_base = f"http://{laptop_ip}:{args.port}"
    store = CalibrationStore()
    server = CalibrationHTTPServer(
        (args.host, args.port),
        CalibrationHandler,
        store=store,
        token=args.token,
        api_base=api_base,
        require_auth=not args.no_auth,
    )
    print(f"Calibration UI: http://127.0.0.1:{args.port}")
    print(f"Pi API base:    {api_base}")
    print(f"Pi token:       {args.token}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping calibration server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
