"""Long-running laptop receiver for Pi posture/behaviour monitoring.

Run this on the laptop, point the Raspberry Pi edge loop at this server, and it
will persist incoming posture windows. On a fixed interval it sends the recent
posture windows plus baseline.json to Qwen for a concise observational summary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from qwen_config import qwen_model_for

DEFAULT_TOKEN = os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token")
DEFAULT_QWEN_API_BASE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_QWEN_MODEL = "qwen-plus"
POSTURE_EVENT_TYPE = "posture_window"


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


def discover_laptop_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 90,
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **headers,
    }
    request = urllib.request.Request(url, data=body, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc


def number_values(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    numbers = []
    for value in values:
        if value is None:
            continue
        try:
            numbers.append(float(value))
        except (TypeError, ValueError):
            continue
    return numbers


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def minmax(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "max": None}
    return {"min": round(min(values), 3), "max": round(max(values), 3)}


def summarize_posture_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "forward_head_ratio",
        "shoulder_tilt_ratio",
        "torso_lean_ratio",
        "motion_score",
        "confidence",
        "shoulder_offset_ratio",
        "elbow_angle_degrees",
        "arm_elevation_ratio",
    )
    all_values: dict[str, list[float]] = {name: [] for name in metric_names}
    sample_count = 0
    present_sample_count = 0
    windows: list[dict[str, Any]] = []

    for event in events:
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        series = payload.get("series") if isinstance(payload.get("series"), dict) else {}
        sample_interval = payload.get("sample_interval_seconds")
        window_sample_count = max(
            (len(value) for value in series.values() if isinstance(value, list)),
            default=0,
        )
        sample_count += window_sample_count
        confidence_values = number_values(series.get("confidence"))
        present_sample_count += len(confidence_values)
        window_summary: dict[str, Any] = {
            "sequence": event.get("_sequence"),
            "received_at": event.get("_received_at"),
            "window_start": payload.get("window_start"),
            "sample_interval_seconds": sample_interval,
            "sample_count": window_sample_count,
        }
        for name in metric_names:
            values = number_values(series.get(name))
            all_values[name].extend(values)
            window_summary[name] = {
                "mean": mean(values),
                **minmax(values),
            }
        windows.append(window_summary)

    metrics = {
        name: {
            "mean": mean(values),
            **minmax(values),
            "count": len(values),
        }
        for name, values in all_values.items()
    }
    return {
        "window_count": len(events),
        "sample_count": sample_count,
        "person_detected_sample_ratio": (
            round(present_sample_count / sample_count, 3) if sample_count else None
        ),
        "metrics": metrics,
        "windows": windows,
    }


def baseline_posture_events(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    calibration = baseline.get("calibration") if isinstance(baseline.get("calibration"), dict) else {}
    events = calibration.get("events") if isinstance(calibration.get("events"), list) else []
    return [event for event in events if event.get("event_type") == POSTURE_EVENT_TYPE]


def compact_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": baseline.get("created_at"),
        "object_policy": baseline.get("object_policy"),
        "object_detection": baseline.get("object_detection"),
        "baseline_posture_summary": summarize_posture_events(baseline_posture_events(baseline)),
    }


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


def normalize_analysis(analysis: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "judgement": "",
        "significant_changes": [],
        "posture": "unclear",
        "behaviour": "unclear",
        "stillness_or_restlessness": "unclear",
        "confidence": "low",
        "observations": [],
    }
    for key, value in defaults.items():
        if key not in analysis:
            analysis[key] = value
    for key in ("significant_changes", "observations"):
        if not isinstance(analysis[key], list):
            analysis[key] = []
    return analysis


def deterministic_monitor_analysis(recent_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = recent_summary.get("metrics") if isinstance(recent_summary.get("metrics"), dict) else {}

    def mean_metric(name: str) -> float | None:
        item = metrics.get(name) if isinstance(metrics.get(name), dict) else {}
        value = item.get("mean")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    forward = mean_metric("forward_head_ratio")
    torso = mean_metric("torso_lean_ratio")
    shoulder = mean_metric("shoulder_tilt_ratio")
    motion = mean_metric("motion_score")
    window_count = int(recent_summary.get("window_count", 0) or 0)

    posture = "baseline_like"
    behaviour = "baseline_like"
    stillness = "normal"
    changes = []
    observations = []

    if forward is not None and forward >= 0.42:
        posture = "slouching"
        changes.append("head position appears further forward than ideal")
        observations.append("The head is sitting forward, suggesting slouching.")
    elif torso is not None and torso >= 0.24:
        posture = "leaning"
        changes.append("torso appears to be leaning")
        observations.append("The torso appears to be leaning away from a neutral seated position.")
    elif shoulder is not None and shoulder >= 0.12:
        posture = "leaning"
        changes.append("shoulders appear uneven")
        observations.append("The shoulders look uneven, which may be worth resetting.")
    else:
        observations.append("Posture looks broadly steady compared with the available baseline.")

    if motion is not None and motion <= 1.0:
        stillness = "too_still"
        behaviour = "possibly_hyperfocused"
        changes.append("the person is sitting very still")
        observations.append("The person is sitting very still, which may mean they are locked in.")
    elif motion is not None and motion >= 8.0:
        stillness = "restless"
        behaviour = "restless"
        changes.append("there is more movement than usual")
        observations.append("There is more shifting and movement than a steady seated baseline.")

    if not changes:
        judgement = "Nothing stands out strongly; the person looks broadly steady."
    else:
        judgement = " ".join(observations[:2])

    confidence = "medium" if window_count >= 2 else "low"
    return normalize_analysis(
        {
            "judgement": judgement,
            "significant_changes": changes,
            "posture": posture,
            "behaviour": behaviour,
            "stillness_or_restlessness": stillness,
            "confidence": confidence,
            "observations": observations,
        }
    )


def call_qwen_for_monitoring(
    baseline: dict[str, Any],
    recent_summary: dict[str, Any],
    recent_events: list[dict[str, Any]],
    interval_started_at: str,
    interval_ended_at: str,
) -> dict[str, Any]:
    api_base = os.getenv("QWEN_API_BASE", DEFAULT_QWEN_API_BASE).rstrip("/")
    api_key = os.getenv("QWEN_API_KEY")
    model = qwen_model_for("posture", DEFAULT_QWEN_MODEL)
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is not set")

    payload_for_model = {
        "interval": {
            "started_at": interval_started_at,
            "ended_at": interval_ended_at,
        },
        "baseline": compact_baseline(baseline),
        "recent_posture_summary": recent_summary,
        "recent_raw_posture_windows": [
            {
                "sequence": event.get("_sequence"),
                "received_at": event.get("_received_at"),
                "payload": event.get("payload"),
            }
            for event in recent_events
        ],
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You are an observational posture and behaviour monitor for a desk setup. "
                "Do not diagnose. Do not infer mental state beyond the provided sensor data. "
                "Write in natural human observation language. Do not mention raw data, metrics, "
                "ratios, samples, windows, confidence values, or sensor internals. Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Compare the recent posture/motion windows against the baseline. Give a short, "
                "accurate judgement about significant changes. Phrase it like a coach observing "
                "the person, not like a report of measurements. For example: 'Compared to baseline, "
                "the head is sitting too far forward, suggesting slouching,' or 'The person is sitting "
                "very still,' or 'There is more shifting than usual.' Look for slouching/forward head, "
                "torso leaning, shoulder tilt, too-still behaviour, restlessness, and possible focused "
                "or hyperfocused patterns only when supported by sustained low movement plus stable posture.\n\n"
                "Avoid saying 'the data shows', 'metrics indicate', 'ratio', 'window', 'sample', or numeric values. "
                "If nothing meaningful changed, say that plainly.\n\n"
                "Return this exact JSON shape:\n"
                "{\n"
                '  "judgement": "one or two short sentences",\n'
                '  "significant_changes": ["string"],\n'
                '  "posture": "baseline_like|slouching|leaning|upright|unclear",\n'
                '  "behaviour": "focused|possibly_hyperfocused|restless|too_still|baseline_like|unclear",\n'
                '  "stillness_or_restlessness": "too_still|restless|normal|unclear",\n'
                '  "confidence": "low|medium|high",\n'
                '  "observations": ["brief plain-language observations"]\n'
                "}\n\n"
                f"Data:\n{json.dumps(payload_for_model, indent=2)}"
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
    return normalize_analysis(parse_json_content(content))


@dataclass
class MonitorStore:
    token: str
    baseline: dict[str, Any]
    data_dir: Path
    analysis_interval_seconds: int
    use_qwen: bool = True
    analysis_enabled: Callable[[], bool] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    analyses: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _sequence: int = 0
    _last_analysis_index: int = 0
    _last_analysis_at: float = field(default_factory=time.monotonic)
    _interval_started_at: str = field(default_factory=utc_now)
    _stop: threading.Event = field(default_factory=threading.Event)

    @property
    def events_path(self) -> Path:
        return self.data_dir / "monitor_events.jsonl"

    @property
    def analyses_path(self) -> Path:
        return self.data_dir / "qwen_analyses.jsonl"

    @property
    def state_path(self) -> Path:
        return self.data_dir / "monitor_state.json"

    def add_event(self, event: dict[str, Any]) -> int:
        with self._lock:
            self._sequence += 1
            event["_sequence"] = self._sequence
            event["_received_at"] = utc_now()
            self.events.append(event)
            append_jsonl(self.events_path, event)
            self.write_state_locked()
            return self._sequence

    def status(self) -> dict[str, Any]:
        with self._lock:
            posture_count = sum(1 for event in self.events if event.get("event_type") == POSTURE_EVENT_TYPE)
            return {
                "events": len(self.events),
                "posture_windows": posture_count,
                "analyses": len(self.analyses),
                "analysis_interval_seconds": self.analysis_interval_seconds,
                "analysis_mode": "qwen" if self.use_qwen else "local",
                "seconds_until_next_analysis": max(
                    0,
                    int(self.analysis_interval_seconds - (time.monotonic() - self._last_analysis_at)),
                ),
                "latest_analysis": self.analyses[-1] if self.analyses else None,
            }

    def stop(self) -> None:
        self._stop.set()

    def analysis_loop(self) -> None:
        while not self._stop.wait(1):
            if self.analysis_enabled is not None and not self.analysis_enabled():
                with self._lock:
                    self._last_analysis_at = time.monotonic()
                    self._interval_started_at = utc_now()
                    self._last_analysis_index = len(self.events)
                continue
            if time.monotonic() - self._last_analysis_at < self.analysis_interval_seconds:
                continue
            self.run_analysis_if_needed()

    def run_analysis_if_needed(self) -> None:
        if self.analysis_enabled is not None and not self.analysis_enabled():
            with self._lock:
                self._last_analysis_at = time.monotonic()
                self._interval_started_at = utc_now()
                self._last_analysis_index = len(self.events)
            return
        with self._lock:
            recent = [
                event
                for event in self.events[self._last_analysis_index :]
                if event.get("event_type") == POSTURE_EVENT_TYPE
            ]
            start_index = self._last_analysis_index
            self._last_analysis_index = len(self.events)
            interval_started_at = self._interval_started_at
            interval_ended_at = utc_now()
            self._interval_started_at = interval_ended_at
            self._last_analysis_at = time.monotonic()

        if not recent:
            analysis = {
                "created_at": interval_ended_at,
                "interval": {
                    "started_at": interval_started_at,
                    "ended_at": interval_ended_at,
                    "start_event_index": start_index,
                    "end_event_index": self._last_analysis_index,
                },
                "analysis": {
                    "judgement": "No posture windows were received during this interval.",
                    "significant_changes": [],
                    "posture": "unclear",
                    "behaviour": "unclear",
                    "stillness_or_restlessness": "unclear",
                    "confidence": "low",
                    "observations": [],
                },
                "posture_summary": summarize_posture_events([]),
            }
        else:
            if self.analysis_enabled is not None and not self.analysis_enabled():
                return
            summary = summarize_posture_events(recent)
            if not self.use_qwen:
                qwen_analysis = deterministic_monitor_analysis(summary)
                error = None
            else:
                try:
                    qwen_analysis = call_qwen_for_monitoring(
                        self.baseline,
                        summary,
                        recent,
                        interval_started_at,
                        interval_ended_at,
                    )
                    error = None
                except Exception as exc:
                    qwen_analysis = deterministic_monitor_analysis(summary)
                    error = str(exc)
            analysis = {
                "created_at": utc_now(),
                "interval": {
                    "started_at": interval_started_at,
                    "ended_at": interval_ended_at,
                    "start_event_index": start_index,
                    "end_event_index": self._last_analysis_index,
                },
                "analysis": qwen_analysis,
                "posture_summary": summary,
            }
            if error:
                analysis["error"] = error

        with self._lock:
            self.analyses.append(analysis)
            append_jsonl(self.analyses_path, analysis)
            self.write_state_locked()
        print(
            "[monitor] analysis "
            f"{len(self.analyses)}: {analysis['analysis'].get('judgement', '')}",
            flush=True,
        )

    def write_state_locked(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "updated_at": utc_now(),
            "events": len(self.events),
            "analyses": len(self.analyses),
            "analysis_mode": "qwen" if self.use_qwen else "local",
            "latest_analysis": self.analyses[-1] if self.analyses else None,
        }
        self.state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


class MonitorHandler(BaseHTTPRequestHandler):
    server: "MonitorHTTPServer"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self._send_json({"ok": True, "status": self.server.store.status()})
        elif path == "/api/status":
            self._send_json(self.server.store.status())
        elif path in {"/pi/commands", "/sessions/active"}:
            self._send_json({"active": True, "command": None})
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

    def _receive_event(self) -> None:
        expected = f"Bearer {self.server.store.token}"
        if self.headers.get("Authorization") != expected:
            self.send_error(HTTPStatus.UNAUTHORIZED, "invalid bearer token")
            return
        event = self._read_json()
        if event is None:
            return
        sequence = self.server.store.add_event(event)
        if event.get("event_type") == POSTURE_EVENT_TYPE:
            print(f"[monitor] stored posture window #{sequence}", flush=True)
        self._send_json({"ok": True, "sequence": sequence})

    def _send_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MonitorHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        store: MonitorStore,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store = store


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run long posture/behaviour monitoring receiver.")
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    parser.add_argument("--port", type=int, default=8010, help="HTTP port for Pi events.")
    parser.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token expected from the Pi.")
    parser.add_argument("--baseline", default="baseline.json", help="Baseline JSON file.")
    parser.add_argument("--data-dir", default="monitor_data", help="Directory for monitor JSONL output.")
    parser.add_argument(
        "--analysis-interval",
        type=int,
        default=120,
        help="Seconds between Qwen analyses. Use 900 for 15 minutes later.",
    )
    parser.add_argument("--local-analysis", action="store_true", help="Use deterministic local posture analysis instead of Qwen.")
    return parser


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise SystemExit(f"Baseline not found: {baseline_path}")
    baseline = read_json(baseline_path)
    data_dir = Path(args.data_dir)
    store = MonitorStore(
        token=args.token,
        baseline=baseline,
        data_dir=data_dir,
        analysis_interval_seconds=args.analysis_interval,
        use_qwen=not args.local_analysis,
    )
    server = MonitorHTTPServer((args.host, args.port), MonitorHandler, store)
    analysis_thread = threading.Thread(target=store.analysis_loop, daemon=True)
    analysis_thread.start()

    laptop_ip = discover_laptop_ip()
    api_base = f"http://{laptop_ip}:{args.port}"
    print(f"Monitor receiver: http://127.0.0.1:{args.port}")
    print(f"Pi API base:      {api_base}")
    print(f"Data directory:   {data_dir}")
    print(f"Analysis every:   {args.analysis_interval}s ({'local' if args.local_analysis else 'qwen'})")
    print("Run the Pi with:")
    print(
        f"  FLOWPILOT_API_BASE={api_base} FLOWPILOT_PI_TOKEN={args.token} "
        "python -m pi_edge.camera_loop --interval 10 --window-seconds 60"
    )
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping monitor.")
    finally:
        store.stop()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
