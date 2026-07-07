from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class DebugMetrics:
    forward_head_ratio: float | None = None
    shoulder_tilt_ratio: float | None = None
    torso_lean_ratio: float | None = None
    motion_score: float | None = None
    confidence: float | None = None
    object_labels: list[str] | None = None
    capture_active: bool = False


class DebugFrameStore:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._sequence = 0

    def update(self, jpeg: bytes) -> None:
        with self._condition:
            self._jpeg = jpeg
            self._sequence += 1
            self._condition.notify_all()

    def wait_for_frame(self, last_sequence: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        with self._condition:
            if self._sequence == last_sequence:
                self._condition.wait(timeout)
            return self._sequence, self._jpeg


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _landmark_point(landmark: Any, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def _draw_landmark_line(frame, landmarks: Any, width: int, height: int, start: int, end: int, color: tuple[int, int, int]) -> None:
    import cv2

    if not landmarks:
        return
    if start >= len(landmarks) or end >= len(landmarks):
        return
    a = landmarks[start]
    b = landmarks[end]
    if getattr(a, "visibility", 0.0) < 0.5 or getattr(b, "visibility", 0.0) < 0.5:
        return
    pa = _landmark_point(a, width, height)
    pb = _landmark_point(b, width, height)
    cv2.line(frame, pa, pb, color, 2, cv2.LINE_AA)
    cv2.circle(frame, pa, 4, color, -1, cv2.LINE_AA)
    cv2.circle(frame, pb, 4, color, -1, cv2.LINE_AA)


def _draw_pose_overlay(frame, landmarks: Any, selected_indices: tuple[int, int] | None) -> None:
    import cv2

    if not landmarks:
        return
    height, width = frame.shape[:2]
    body_color = (92, 214, 255)
    arm_color = (125, 255, 159)
    selected_color = (77, 77, 255)
    for start, end in (
        (7, 11),
        (8, 12),
        (11, 12),
        (11, 23),
        (12, 24),
        (23, 24),
    ):
        _draw_landmark_line(frame, landmarks, width, height, start, end, body_color)
    for start, end in ((11, 13), (13, 15), (12, 14), (14, 16)):
        _draw_landmark_line(frame, landmarks, width, height, start, end, arm_color)
    if selected_indices is not None:
        _draw_landmark_line(frame, landmarks, width, height, selected_indices[0], selected_indices[1], selected_color)
        ear = landmarks[selected_indices[0]]
        shoulder = landmarks[selected_indices[1]]
        if getattr(ear, "visibility", 0.0) >= 0.5 and getattr(shoulder, "visibility", 0.0) >= 0.5:
            ear_point = _landmark_point(ear, width, height)
            shoulder_point = _landmark_point(shoulder, width, height)
            cv2.line(frame, shoulder_point, (ear_point[0], shoulder_point[1]), (255, 170, 80), 1, cv2.LINE_AA)


def _draw_object_boxes(frame, object_detections: list[Any] | None) -> None:
    import cv2

    if not object_detections:
        return
    height, width = frame.shape[:2]
    for detection in object_detections:
        x1, y1, x2, y2 = detection.bbox
        x1 = max(0, min(width - 1, int(x1)))
        y1 = max(0, min(height - 1, int(y1)))
        x2 = max(0, min(width - 1, int(x2)))
        y2 = max(0, min(height - 1, int(y2)))
        label = f"{detection.label} {detection.score:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
        text_y = max(18, y1 - 6)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 255, 220), 1, cv2.LINE_AA)


def encode_debug_jpeg(
    frame_rgb,
    metrics: DebugMetrics,
    landmarks: Any = None,
    selected_indices: tuple[int, int] | None = None,
    object_detections: list[Any] | None = None,
) -> bytes:
    import cv2

    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    _draw_pose_overlay(frame, landmarks, selected_indices)
    _draw_object_boxes(frame, object_detections)

    lines = [
        f"confidence: {_fmt(metrics.confidence)}",
        f"neck: {_fmt(metrics.forward_head_ratio)}",
        f"shoulders: {_fmt(metrics.shoulder_tilt_ratio)}",
        f"torso: {_fmt(metrics.torso_lean_ratio)}",
        f"motion: {_fmt(metrics.motion_score)}",
        f"object capture: {'active' if metrics.capture_active else 'idle'}",
    ]
    if metrics.object_labels:
        lines.append("objects: " + ", ".join(metrics.object_labels[:6]))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (min(width, 430), min(height, 26 + len(lines) * 24)), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (12, 26 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError("could not encode debug frame")
    return encoded.tobytes()


def start_debug_server(host: str, port: int) -> tuple[ThreadingHTTPServer, DebugFrameStore]:
    store = DebugFrameStore()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def do_GET(self) -> None:
            if self.path in {"/", "/index.html"}:
                body = (
                    "<!doctype html><html><head><title>FlowPilot Pi Debug</title>"
                    "<style>body{margin:0;background:#111;color:#eee;font-family:sans-serif}"
                    "main{display:grid;place-items:center;min-height:100vh}img{max-width:100vw;max-height:100vh}</style>"
                    "</head><body><main><img src='/stream'></main></body></html>"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path != "/stream":
                self.send_error(404)
                return

            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            while True:
                sequence, jpeg = store.wait_for_frame(sequence)
                if jpeg is None:
                    time.sleep(0.1)
                    continue
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

    server = ThreadingHTTPServer((host, port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, store
