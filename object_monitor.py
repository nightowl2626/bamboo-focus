"""Periodic laptop receiver/scheduler for object-only monitoring.

Run this on the laptop and point the Pi object loop at it. It reads
baseline.json, extracts Qwen's whitelisted objects, and includes those labels
in each object_capture command so the Pi filters them before sending results.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_TOKEN = os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token")
OBJECT_EVENT_TYPES = {
    "object_capture_snapshot",
    "object_calibration_snapshot",
    "object_surveillance_snapshot",
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


def discover_laptop_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item) + "\n")


def normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().split())


def baseline_labels(baseline: dict[str, Any], key: str) -> list[str]:
    policy = baseline.get("object_policy") if isinstance(baseline.get("object_policy"), dict) else {}
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


def filter_payload_objects(objects: list[dict[str, Any]], exclude_labels: list[str]) -> list[dict[str, Any]]:
    excluded = {normalize_label(label) for label in exclude_labels}
    return [
        item
        for item in objects
        if isinstance(item, dict) and normalize_label(str(item.get("label", ""))) not in excluded
    ]


@dataclass
class ObjectMonitorStore:
    token: str
    baseline: dict[str, Any]
    data_dir: Path
    capture_interval_seconds: int
    whitelist_labels: list[str]
    monitor_labels: list[str]
    pending_commands: list[dict[str, Any]] = field(default_factory=list)
    dispatched_commands: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _sequence: int = 0
    _next_capture_at: float = field(default_factory=time.monotonic)

    @property
    def events_path(self) -> Path:
        return self.data_dir / "object_events.jsonl"

    @property
    def commands_path(self) -> Path:
        return self.data_dir / "object_commands.jsonl"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "object_monitor_state.json"

    def queue_due_capture_locked(self) -> None:
        if self.pending_commands or time.monotonic() < self._next_capture_at:
            return
        command = {
            "id": str(uuid.uuid4()),
            "command_type": "object_capture",
            "seconds": 0,
            "reason": "periodic_object_monitor",
            "created_at": utc_now(),
            "exclude_labels": self.whitelist_labels,
            "monitor_labels": self.monitor_labels,
        }
        self.pending_commands.append(command)
        self._next_capture_at = time.monotonic() + self.capture_interval_seconds
        append_jsonl(self.commands_path, {"queued": command})
        print(
            "[object_monitor] queued object capture "
            f"id={command['id']} excluded={self.whitelist_labels}",
            flush=True,
        )

    def pop_command(self) -> dict[str, Any] | None:
        with self._lock:
            self.queue_due_capture_locked()
            if not self.pending_commands:
                return None
            command = self.pending_commands.pop(0)
            dispatched = dict(command)
            dispatched["dispatched_at"] = utc_now()
            self.dispatched_commands.append(dispatched)
            self.dispatched_commands = self.dispatched_commands[-20:]
            append_jsonl(self.commands_path, {"dispatched": dispatched})
            self.write_state_locked()
            return command

    def add_event(self, event: dict[str, Any]) -> int:
        with self._lock:
            self._sequence += 1
            event["_sequence"] = self._sequence
            event["_received_at"] = utc_now()
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
            filtered_objects = filter_payload_objects(objects, self.whitelist_labels)
            record = {
                "sequence": self._sequence,
                "received_at": event["_received_at"],
                "event": event,
                "whitelist_labels": self.whitelist_labels,
                "monitor_labels": self.monitor_labels,
                "monitorable_objects": filtered_objects,
                "monitorable_count": len(filtered_objects),
            }
            self.snapshots.append(record)
            self.snapshots = self.snapshots[-100:]
            append_jsonl(self.events_path, record)
            self.write_state_locked()
            return self._sequence

    def status(self) -> dict[str, Any]:
        with self._lock:
            self.queue_due_capture_locked()
            return {
                "snapshots": len(self.snapshots),
                "pending_commands": len(self.pending_commands),
                "dispatched_commands": len(self.dispatched_commands),
                "capture_interval_seconds": self.capture_interval_seconds,
                "seconds_until_next_capture": max(0, int(self._next_capture_at - time.monotonic())),
                "whitelist_labels": self.whitelist_labels,
                "monitor_labels": self.monitor_labels,
                "latest_snapshot": self.snapshots[-1] if self.snapshots else None,
            }

    def write_state_locked(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "updated_at": utc_now(),
            "snapshots": len(self.snapshots),
            "pending_commands": len(self.pending_commands),
            "dispatched_commands": len(self.dispatched_commands),
            "capture_interval_seconds": self.capture_interval_seconds,
            "whitelist_labels": self.whitelist_labels,
            "monitor_labels": self.monitor_labels,
            "latest_snapshot": self.snapshots[-1] if self.snapshots else None,
        }
        self.state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


class ObjectMonitorHandler(BaseHTTPRequestHandler):
    server: "ObjectMonitorHTTPServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/api/status", "/health"}:
            self._send_json({"ok": True, "status": self.server.store.status()})
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

    def _send_next_command(self) -> None:
        expected = f"Bearer {self.server.store.token}"
        if self.headers.get("Authorization") != expected:
            self.send_error(HTTPStatus.UNAUTHORIZED, "invalid bearer token")
            return
        command = self.server.store.pop_command()
        if command is not None:
            print(
                "[object_monitor] dispatched object capture "
                f"id={command['id']}",
                flush=True,
            )
        self._send_json({"command": command})

    def _receive_event(self) -> None:
        expected = f"Bearer {self.server.store.token}"
        if self.headers.get("Authorization") != expected:
            self.send_error(HTTPStatus.UNAUTHORIZED, "invalid bearer token")
            return
        event = self._read_json()
        if event is None:
            return
        if event.get("event_type") not in OBJECT_EVENT_TYPES:
            self._send_json({"ok": True, "ignored": True, "reason": "not an object event"})
            return
        sequence = self.server.store.add_event(event)
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        objects = payload.get("objects") if isinstance(payload.get("objects"), list) else []
        kept = filter_payload_objects(objects, self.server.store.whitelist_labels)
        print(
            "[object_monitor] stored object snapshot "
            f"#{sequence} kept={len(kept)} total={len(objects)}",
            flush=True,
        )
        self._send_json({"ok": True, "sequence": sequence})

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ObjectMonitorHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        store: ObjectMonitorStore,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store = store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run periodic object-only monitoring receiver.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=8020, help="HTTP port for Pi object events.")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token expected from the Pi.")
    parser.add_argument("--baseline", default="baseline.json", help="Baseline JSON with Qwen object policy.")
    parser.add_argument("--data-dir", default="object_monitor_data", help="Directory for object monitor output.")
    parser.add_argument(
        "--capture-interval",
        type=int,
        default=1800,
        help="Seconds between object captures. Default is 1800, i.e. 30 minutes.",
    )
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found: {baseline_path}")
    baseline = read_json(baseline_path)
    whitelist_labels = baseline_labels(baseline, "whitelisted_objects")
    monitor_labels = baseline_labels(baseline, "monitorable_objects")
    store = ObjectMonitorStore(
        token=args.token,
        baseline=baseline,
        data_dir=Path(args.data_dir),
        capture_interval_seconds=args.capture_interval,
        whitelist_labels=whitelist_labels,
        monitor_labels=monitor_labels,
    )
    server = ObjectMonitorHTTPServer((args.host, args.port), ObjectMonitorHandler, store)
    laptop_ip = discover_laptop_ip()
    api_base = f"http://{laptop_ip}:{args.port}"
    print(f"Object monitor receiver: http://127.0.0.1:{args.port}")
    print(f"Pi API base:             {api_base}")
    print(f"Capture interval:        {args.capture_interval}s")
    print(f"Whitelisted labels:      {whitelist_labels}")
    print(f"Monitor labels:          {monitor_labels}")
    print("Run the Pi with:")
    print(
        f"  python pi_start.py --laptop-api-base {api_base} "
        f"--token {args.token} --download-object-model --download-pose-model --command-poll-interval 1"
    )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping object monitor.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
