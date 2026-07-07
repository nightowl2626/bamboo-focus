import argparse
import os
import time
from datetime import datetime, timezone

from .client import default_client
from .sensors import object_capture_snapshot, posture_window, presence_started


def _log(message: str) -> None:
    print(f"[pi_edge {datetime.now(timezone.utc):%H:%M:%S}] {message}", flush=True)


def _fmt_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def _fmt_detections(detections) -> str:
    if not detections:
        return "none"
    return ", ".join(
        f"{item.label}:{item.score:.2f}@{list(item.bbox)}"
        for item in sorted(detections, key=lambda detection: detection.score, reverse=True)[:8]
    )


def run_camera(
    interval_seconds: float,
    window_seconds: float = 60.0,
    object_model_path: str | None = None,
    pose_model_path: str | None = None,
    object_score_threshold: float = 0.35,
    command_poll_interval_seconds: float = 1.0,
    debug_stream: bool = False,
    debug_host: str = "127.0.0.1",
    debug_port: int = 8765,
) -> None:
    _log(
        "camera mode starting "
        f"interval={interval_seconds}s window={window_seconds}s "
        f"command_poll_interval={command_poll_interval_seconds}s"
        f" debug_stream={'on' if debug_stream else 'off'}"
    )
    missing_camera_dependencies = []
    try:
        _log("importing camera dependency cv2")
        import cv2
    except ImportError:
        missing_camera_dependencies.append("cv2 / python3-opencv")
    try:
        _log("importing camera dependency Picamera2")
        from picamera2 import Picamera2
    except ImportError:
        missing_camera_dependencies.append("picamera2 / python3-picamera2")
    if missing_camera_dependencies:
        raise SystemExit(
            "Camera mode requires Picamera2 and OpenCV on the Raspberry Pi. "
            "Missing: "
            f"{', '.join(missing_camera_dependencies)}. "
            "Install the apt packages and run from a Python environment created "
            "with --system-site-packages."
        )

    _log("creating hub client")
    client = default_client()
    client.clear_queue()
    _log("manual object capture mode: discarded queued events")

    _log("initializing Picamera2")
    camera = Picamera2()
    _log("configuring Picamera2 preview size=640x480 format=RGB888")
    camera.configure(camera.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
    _log("starting Picamera2")
    camera.start()
    _log("Picamera2 started")

    from .posture import PostureAnalyzer, PostureUnavailable

    analyzer = None
    try:
        _log("initializing posture detector (mediapipe pose)")
        analyzer = PostureAnalyzer(model_path=pose_model_path)
        _log("posture detection on (mediapipe)")
    except PostureUnavailable as exc:
        _log(f"posture detection off: {exc}")

    from .metrics_window import MetricSample, MetricsWindowBuilder
    from .objects import (
        ObjectCaptureCollector,
        ObjectDetector,
        ObjectDetectorUnavailable,
        filter_detections_by_label,
    )

    object_detector = None
    _log(
        "initializing object detector "
        f"model={object_model_path or os.getenv('FLOWPILOT_OBJECT_DETECTION_MODEL_PATH') or 'not configured'} "
        f"score_threshold={object_score_threshold}"
    )
    try:
        object_detector = ObjectDetector(
            model_path=object_model_path,
            score_threshold=object_score_threshold,
        )
        _log("object detection on (mediapipe); waiting for laptop capture commands")
    except ObjectDetectorUnavailable as exc:
        _log(f"object detection off: {exc}")

    window_builder = MetricsWindowBuilder(window_seconds, interval_seconds)
    debug_store = None
    if debug_stream:
        from .debug_view import start_debug_server

        _, debug_store = start_debug_server(debug_host, debug_port)
        _log(f"debug stream on http://{debug_host}:{debug_port}/")

    previous_gray = None
    presence_sent = False
    frame_count = 0
    last_command_poll_at = 0.0
    manual_object_capture = None
    _log("camera loop entering capture cycle")

    while True:
        if frame_count == 0:
            _log("waiting for first camera frame")
        frame = camera.capture_array()
        if frame_count == 0:
            _log(f"first camera frame captured shape={getattr(frame, 'shape', 'unknown')}")
        frame_count += 1
        frame_rgb = frame[:, :, :3]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        now = datetime.now(timezone.utc)
        motion_score = None
        sample = None
        observation = None
        latest_object_detections = []

        if not presence_sent:
            sent = client.send_event(presence_started())
            _log(f"presence_started {'sent' if sent else 'queued (hub unreachable)'}")
            presence_sent = True

        if previous_gray is not None:
            delta = cv2.absdiff(gray, previous_gray)
            motion_score = float(delta.mean())
            _log(f"motion={motion_score:.2f}")

        if analyzer is not None:
            observation = analyzer.observe(frame_rgb)
            sample = observation.sample if observation is not None else None
            if sample is None:
                _log("posture: no person / low confidence")
            else:
                _log(
                    f"posture neck={_fmt_metric(sample.forward_head_ratio)}"
                    f" shoulders={_fmt_metric(sample.shoulder_tilt_ratio)}"
                    f" torso={_fmt_metric(sample.torso_lean_ratio)}"
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
                        "exclude_labels": command.get("exclude_labels") if isinstance(command.get("exclude_labels"), list) else [],
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
                elapsed_manual_capture = time.monotonic() - manual_object_capture["started_at"]
                _log(
                    "manual object capture observed "
                    f"{len(detections)} objects "
                    f"({int(elapsed_manual_capture)}/{manual_object_capture['seconds']}s): "
                    f"{_fmt_detections(detections)}"
                )
                if elapsed_manual_capture >= manual_object_capture["seconds"]:
                    payload = manual_object_capture["collector"].snapshot(frame_size, now)
                    payload["mode"] = "manual"
                    payload["request_id"] = manual_object_capture["id"]
                    payload["reason"] = manual_object_capture["reason"]
                    payload["excluded_labels"] = manual_object_capture["exclude_labels"]
                    payload["raw_object_count"] = len(raw_detections)
                    payload["excluded_object_count"] = len(raw_detections) - len(detections)
                    sent = client.send_event(object_capture_snapshot(payload), queue_on_failure=False)
                    _log(
                        "manual object_capture_snapshot "
                        f"{'sent' if sent else 'failed (not queued)'} "
                        f"({len(payload['objects'])} objects)"
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
            sent = client.send_event(posture_window(window_payload))
            samples_count = len(window_payload["series"]["confidence"])
            _log(f"posture_window {'sent' if sent else 'queued (hub unreachable)'} ({samples_count} samples)")

        if debug_store is not None:
            from .debug_view import DebugMetrics, encode_debug_jpeg

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FlowPilot Pi edge loop.")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between loop iterations.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=float(os.getenv("FLOWPILOT_WINDOW_SECONDS", "60")),
        help="Seconds of samples per posture_window event.",
    )
    parser.add_argument(
        "--object-model-path",
        default=os.getenv("FLOWPILOT_OBJECT_DETECTION_MODEL_PATH"),
        help="Path to a MediaPipe object detector .tflite model.",
    )
    parser.add_argument(
        "--pose-model-path",
        default=os.getenv("FLOWPILOT_POSE_LANDMARKER_MODEL_PATH"),
        help="Path to a MediaPipe pose_landmarker_lite.task model.",
    )
    parser.add_argument(
        "--object-score-threshold",
        type=float,
        default=float(os.getenv("FLOWPILOT_OBJECT_SCORE_THRESHOLD", "0.35")),
        help="Minimum MediaPipe object detection score.",
    )
    parser.add_argument(
        "--command-poll-interval",
        type=float,
        default=float(os.getenv("FLOWPILOT_COMMAND_POLL_INTERVAL_SECONDS", "1")),
        help="Seconds between laptop command polls.",
    )
    parser.add_argument("--debug-stream", action="store_true", help="Serve a live MJPEG debug stream.")
    parser.add_argument("--debug-host", default="127.0.0.1", help="Host/interface for --debug-stream.")
    parser.add_argument("--debug-port", type=int, default=8765, help="Port for --debug-stream.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_camera(
        args.interval,
        window_seconds=args.window_seconds,
        object_model_path=args.object_model_path,
        pose_model_path=args.pose_model_path,
        object_score_threshold=args.object_score_threshold,
        command_poll_interval_seconds=args.command_poll_interval,
        debug_stream=args.debug_stream,
        debug_host=args.debug_host,
        debug_port=args.debug_port,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
