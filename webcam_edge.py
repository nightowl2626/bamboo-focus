"""Laptop webcam demo edge process for FlowPilot.

This is intentionally protocol-compatible with the Raspberry Pi edge process:
it sends posture_window/object_capture_snapshot events to app.py and polls the
same /pi/commands endpoint for object captures. It is meant for judging/demo
setups where no Raspberry Pi camera is available.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from pi_edge.client import default_client
from pi_edge.metrics_window import MetricSample, MetricsWindowBuilder
from pi_edge.objects import (
    ObjectCaptureCollector,
    ObjectDetector,
    ObjectDetectorUnavailable,
    filter_detections_by_label,
)
from pi_edge.sensors import object_capture_snapshot, posture_window, presence_started
from pi_start import (
    default_model_path,
    default_pose_model_path,
    download_object_model,
    download_pose_model,
    find_object_model,
    find_pose_model,
)

SOURCE = "laptop_webcam_demo"


def _log(message: str) -> None:
    print(f"[webcam_edge {datetime.now(timezone.utc):%H:%M:%S}] {message}", flush=True)


def _fmt_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _fmt_detections(detections) -> str:
    if not detections:
        return "none"
    return ", ".join(
        f"{item.label}:{item.score:.2f}@{list(item.bbox)}"
        for item in sorted(detections, key=lambda detection: detection.score, reverse=True)[:8]
    )


def resolve_object_model(path: str, download: bool) -> Path | None:
    requested = Path(path)
    resolved = find_object_model(requested)
    if resolved is None and download:
        if download_object_model(requested):
            resolved = requested
    if resolved is None:
        _log(f"object model not found: {requested}")
        _log("object captures will be disabled unless you add --download-object-model or place the model in ./models/")
    return resolved


def resolve_pose_model(path: str, download: bool) -> Path | None:
    requested = Path(path)
    resolved = find_pose_model(requested)
    if resolved is None and download:
        if download_pose_model(requested):
            resolved = requested
    if resolved is None:
        _log(f"pose model not found: {requested}")
        _log("posture will need legacy mediapipe solutions or --download-pose-model")
    return resolved


def run_webcam(
    api_base: str,
    token: str,
    camera_index: int = 0,
    interval_seconds: float = 2.0,
    window_seconds: float = 60.0,
    object_model_path: str | None = None,
    pose_model_path: str | None = None,
    object_score_threshold: float = 0.35,
    command_poll_interval_seconds: float = 1.0,
    width: int = 640,
    height: int = 480,
    mirror: bool = False,
    debug_stream: bool = False,
    debug_host: str = "127.0.0.1",
    debug_port: int = 8766,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Webcam mode requires OpenCV. Install opencv-python in this environment.") from exc

    os.environ["FLOWPILOT_API_BASE"] = api_base.rstrip("/")
    os.environ["FLOWPILOT_PI_TOKEN"] = token
    os.environ["FLOWPILOT_PI_QUEUE_PATH"] = "webcam_event_queue.jsonl"
    if object_model_path:
        os.environ["FLOWPILOT_OBJECT_DETECTION_MODEL_PATH"] = object_model_path
    if pose_model_path:
        os.environ["FLOWPILOT_POSE_LANDMARKER_MODEL_PATH"] = pose_model_path

    _log(
        "webcam mode starting "
        f"camera_index={camera_index} interval={interval_seconds}s window={window_seconds}s "
        f"command_poll_interval={command_poll_interval_seconds}s debug_stream={'on' if debug_stream else 'off'}"
    )
    client = default_client()
    client.clear_queue()

    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open webcam index {camera_index}.")

    from pi_edge.posture import PostureAnalyzer, PostureUnavailable

    analyzer = None
    try:
        analyzer = PostureAnalyzer(model_path=pose_model_path)
        _log("posture detection on (mediapipe)")
    except PostureUnavailable as exc:
        _log(f"posture detection off: {exc}")

    object_detector = None
    try:
        object_detector = ObjectDetector(
            model_path=object_model_path,
            score_threshold=object_score_threshold,
        )
        _log("object detection on (mediapipe); waiting for laptop capture commands")
    except ObjectDetectorUnavailable as exc:
        _log(f"object detection off: {exc}")

    debug_store = None
    if debug_stream:
        from pi_edge.debug_view import start_debug_server

        _, debug_store = start_debug_server(debug_host, debug_port)
        _log(f"debug stream on http://{debug_host}:{debug_port}/")

    window_builder = MetricsWindowBuilder(window_seconds, interval_seconds)
    previous_gray = None
    presence_sent = False
    last_command_poll_at = 0.0
    manual_object_capture = None
    frame_count = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                _log("webcam frame unavailable")
                time.sleep(interval_seconds)
                continue
            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_count += 1
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            now = datetime.now(timezone.utc)
            motion_score = None
            sample = None
            observation = None
            latest_object_detections = []

            if not presence_sent:
                sent = client.send_event(presence_started(source=SOURCE))
                _log(f"presence_started {'sent' if sent else 'queued (hub unreachable)'}")
                presence_sent = True

            if previous_gray is not None:
                motion_score = float(cv2.absdiff(gray, previous_gray).mean())

            if analyzer is not None:
                observation = analyzer.observe(frame_rgb)
                sample = observation.sample if observation is not None else None
                if sample is None:
                    _log(f"frame={frame_count} posture: no person / low confidence motion={_fmt_metric(motion_score)}")
                else:
                    _log(
                        f"frame={frame_count} posture neck={_fmt_metric(sample.forward_head_ratio)}"
                        f" shoulders={_fmt_metric(sample.shoulder_tilt_ratio)}"
                        f" torso={_fmt_metric(sample.torso_lean_ratio)}"
                        f" motion={_fmt_metric(motion_score)}"
                    )

            if object_detector is not None:
                if time.monotonic() - last_command_poll_at >= command_poll_interval_seconds:
                    last_command_poll_at = time.monotonic()
                    command = client.poll_command()
                    if command and command.get("command_type") == "object_capture":
                        try:
                            capture_seconds = max(0, int(command.get("seconds", 0)))
                        except (TypeError, ValueError):
                            capture_seconds = 0
                        manual_object_capture = {
                            "id": command.get("id"),
                            "reason": command.get("reason"),
                            "exclude_labels": command.get("exclude_labels")
                            if isinstance(command.get("exclude_labels"), list)
                            else [],
                            "seconds": capture_seconds,
                            "started_at": time.monotonic(),
                            "collector": ObjectCaptureCollector(),
                        }
                        _log(
                            "manual object capture requested "
                            f"id={manual_object_capture['id']} seconds={capture_seconds}"
                        )

                if manual_object_capture is not None:
                    frame_height, frame_width = frame_rgb.shape[:2]
                    frame_size = (frame_width, frame_height)
                    raw_detections = object_detector.detect(frame_rgb)
                    detections = filter_detections_by_label(raw_detections, manual_object_capture["exclude_labels"])
                    latest_object_detections = detections
                    manual_object_capture["collector"].add(detections)
                    elapsed = time.monotonic() - manual_object_capture["started_at"]
                    _log(
                        "manual object capture observed "
                        f"{len(detections)} objects ({int(elapsed)}/{manual_object_capture['seconds']}s): "
                        f"{_fmt_detections(detections)}"
                    )
                    if elapsed >= manual_object_capture["seconds"]:
                        payload = manual_object_capture["collector"].snapshot(frame_size, now)
                        payload["mode"] = "webcam_demo"
                        payload["request_id"] = manual_object_capture["id"]
                        payload["reason"] = manual_object_capture["reason"]
                        payload["excluded_labels"] = manual_object_capture["exclude_labels"]
                        payload["raw_object_count"] = len(raw_detections)
                        payload["excluded_object_count"] = len(raw_detections) - len(detections)
                        sent = client.send_event(
                            object_capture_snapshot(payload, source=SOURCE),
                            queue_on_failure=False,
                        )
                        _log(
                            "object_capture_snapshot "
                            f"{'sent' if sent else 'failed (not queued)'} ({len(payload['objects'])} objects)"
                        )
                        manual_object_capture = None

            metric_sample = MetricSample(
                forward_head_ratio=sample.forward_head_ratio if sample is not None else None,
                shoulder_tilt_ratio=sample.shoulder_tilt_ratio if sample is not None else None,
                torso_lean_ratio=sample.torso_lean_ratio if sample is not None else None,
                motion_score=motion_score,
                confidence=sample.confidence if sample is not None else None,
                shoulder_offset_ratio=sample.shoulder_offset_ratio if sample is not None else None,
                elbow_angle_degrees=sample.elbow_angle_degrees if sample is not None else None,
                arm_elevation_ratio=sample.arm_elevation_ratio if sample is not None else None,
            )
            window_payload = window_builder.add(metric_sample, now)
            if window_payload is not None:
                sent = client.send_event(posture_window(window_payload, source=SOURCE))
                samples_count = len(window_payload["series"]["confidence"])
                _log(f"posture_window {'sent' if sent else 'queued (hub unreachable)'} ({samples_count} samples)")

            if debug_store is not None:
                from pi_edge.debug_view import DebugMetrics, encode_debug_jpeg

                try:
                    debug_store.update(
                        encode_debug_jpeg(
                            frame_rgb,
                            DebugMetrics(
                                forward_head_ratio=sample.forward_head_ratio if sample is not None else None,
                                shoulder_tilt_ratio=sample.shoulder_tilt_ratio if sample is not None else None,
                                torso_lean_ratio=sample.torso_lean_ratio if sample is not None else None,
                                motion_score=motion_score,
                                confidence=sample.confidence if sample is not None else None,
                                object_labels=[item.label for item in latest_object_detections],
                                capture_active=manual_object_capture is not None,
                            ),
                            landmarks=observation.landmarks if observation is not None else None,
                            selected_indices=observation.selected_indices if observation is not None else None,
                            object_detections=latest_object_detections,
                        )
                    )
                except RuntimeError as exc:
                    _log(f"debug stream frame skipped: {exc}")

            previous_gray = gray
            time.sleep(interval_seconds)
    finally:
        capture.release()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FlowPilot with this laptop's webcam instead of a Raspberry Pi.")
    parser.add_argument("--laptop-api-base", default="http://127.0.0.1:8000", help="Local app.py URL.")
    parser.add_argument("--token", default=os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token"))
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV webcam index.")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between posture samples.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Seconds per posture window.")
    parser.add_argument(
        "--object-model-path",
        default=os.getenv("FLOWPILOT_OBJECT_DETECTION_MODEL_PATH", str(default_model_path())),
        help="Path to efficientdet_lite0.tflite.",
    )
    parser.add_argument(
        "--pose-model-path",
        default=os.getenv("FLOWPILOT_POSE_LANDMARKER_MODEL_PATH", str(default_pose_model_path())),
        help="Path to pose_landmarker_lite.task.",
    )
    parser.add_argument("--download-object-model", action="store_true", help="Download the object model if missing.")
    parser.add_argument("--download-pose-model", action="store_true", help="Download the pose model if missing.")
    parser.add_argument("--object-score-threshold", type=float, default=0.35)
    parser.add_argument("--command-poll-interval", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam frame before analysis.")
    parser.add_argument("--debug-stream", action="store_true", help="Serve a local MJPEG debug stream.")
    parser.add_argument("--debug-host", default="127.0.0.1")
    parser.add_argument("--debug-port", type=int, default=8766)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model_path = resolve_object_model(args.object_model_path, args.download_object_model)
    pose_model_path = resolve_pose_model(args.pose_model_path, args.download_pose_model)
    run_webcam(
        api_base=args.laptop_api_base,
        token=args.token,
        camera_index=args.camera_index,
        interval_seconds=args.interval,
        window_seconds=args.window_seconds,
        object_model_path=str(model_path) if model_path is not None else None,
        pose_model_path=str(pose_model_path) if pose_model_path is not None else None,
        object_score_threshold=args.object_score_threshold,
        command_poll_interval_seconds=args.command_poll_interval,
        width=args.width,
        height=args.height,
        mirror=args.mirror,
        debug_stream=args.debug_stream,
        debug_host=args.debug_host,
        debug_port=args.debug_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
