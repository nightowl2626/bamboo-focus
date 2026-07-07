"""Start the single Raspberry Pi camera process for the unified laptop app."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from urllib.error import URLError
from urllib.request import urlretrieve
from pathlib import Path


MODEL_FILENAME = "efficientdet_lite0.tflite"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
)
POSE_MODEL_FILENAME = "pose_landmarker_lite.task"
POSE_MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)


def default_model_path() -> Path:
    return Path.cwd() / "models" / MODEL_FILENAME


def default_pose_model_path() -> Path:
    return Path.cwd() / "models" / POSE_MODEL_FILENAME


def find_model_file(filename: str, preferred: Path) -> Path | None:
    candidates = [
        preferred,
        Path.cwd() / filename,
        Path.cwd() / "models" / filename,
        Path(__file__).resolve().parent / "models" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in Path.cwd().glob(f"**/{filename}"):
        if ".venv" in candidate.parts or "venv" in candidate.parts:
            continue
        return candidate
    return None


def find_object_model(preferred: Path) -> Path | None:
    return find_model_file(MODEL_FILENAME, preferred)


def find_pose_model(preferred: Path) -> Path | None:
    return find_model_file(POSE_MODEL_FILENAME, preferred)


def download_model(url: str, path: Path, label: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} model to {path}", flush=True)
    try:
        urlretrieve(url, path)
    except (OSError, URLError) as exc:
        print(f"WARNING: could not download {label} model: {exc}", flush=True)
        return False
    return path.exists() and path.stat().st_size > 0


def download_object_model(path: Path) -> bool:
    return download_model(MODEL_DOWNLOAD_URL, path, "object")


def download_pose_model(path: Path) -> bool:
    return download_model(POSE_MODEL_DOWNLOAD_URL, path, "pose")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start FlowPilot Pi collection for app.py.")
    parser.add_argument(
        "--laptop-api-base",
        required=True,
        help="Unified laptop app URL, for example http://192.168.1.20:8000.",
    )
    parser.add_argument("--token", default=os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token"))
    parser.add_argument(
        "--object-model-path",
        default=os.getenv("FLOWPILOT_OBJECT_DETECTION_MODEL_PATH", str(default_model_path())),
        help="Path to efficientdet_lite0.tflite.",
    )
    parser.add_argument(
        "--pose-model-path",
        default=os.getenv("FLOWPILOT_POSE_LANDMARKER_MODEL_PATH", str(default_pose_model_path())),
        help="Path to pose_landmarker_lite.task for MediaPipe Tasks posture.",
    )
    parser.add_argument(
        "--download-object-model",
        action="store_true",
        help="Download efficientdet_lite0.tflite to --object-model-path if it is missing.",
    )
    parser.add_argument(
        "--download-pose-model",
        action="store_true",
        help="Download pose_landmarker_lite.task to --pose-model-path if it is missing.",
    )
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between posture samples.")
    parser.add_argument("--window-seconds", type=float, default=60.0, help="Seconds per posture window.")
    parser.add_argument("--command-poll-interval", type=float, default=1.0, help="Seconds between object command polls.")
    parser.add_argument("--debug-stream", action="store_true", help="Enable Pi live MJPEG debug stream.")
    parser.add_argument("--debug-host", default="0.0.0.0", help="Host/interface for --debug-stream.")
    parser.add_argument("--debug-port", type=int, default=8765, help="Port for --debug-stream.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    requested_model_path = Path(args.object_model_path)
    model_path = find_object_model(requested_model_path)
    if model_path is None and args.download_object_model:
        if download_object_model(requested_model_path):
            model_path = requested_model_path
    requested_pose_model_path = Path(args.pose_model_path)
    pose_model_path = find_pose_model(requested_pose_model_path)
    if pose_model_path is None and args.download_pose_model:
        if download_pose_model(requested_pose_model_path):
            pose_model_path = requested_pose_model_path

    if model_path is None:
        print(f"WARNING: object model not found: {requested_model_path}", flush=True)
        print(
            "Object captures will be disabled. To enable them, run once with "
            "--download-object-model, or put efficientdet_lite0.tflite in ./models/.",
            flush=True,
        )
    if pose_model_path is None:
        print(f"WARNING: pose model not found: {requested_pose_model_path}", flush=True)
        print(
            "Tasks-based posture detection will need --download-pose-model, "
            "or pose_landmarker_lite.task in ./models/.",
            flush=True,
        )

    env = os.environ.copy()
    env["FLOWPILOT_API_BASE"] = args.laptop_api_base.rstrip("/")
    env["FLOWPILOT_PI_TOKEN"] = args.token
    if model_path is not None:
        env["FLOWPILOT_OBJECT_DETECTION_MODEL_PATH"] = str(model_path)
    if pose_model_path is not None:
        env["FLOWPILOT_POSE_LANDMARKER_MODEL_PATH"] = str(pose_model_path)
    env["FLOWPILOT_PI_QUEUE_PATH"] = "pi_app_event_queue.jsonl"

    command = [
        sys.executable,
        "-m",
        "pi_edge.camera_loop",
        "--interval",
        str(args.interval),
        "--window-seconds",
        str(args.window_seconds),
        "--command-poll-interval",
        str(args.command_poll_interval),
    ]
    if model_path is not None:
        command.extend(["--object-model-path", str(model_path)])
    if pose_model_path is not None:
        command.extend(["--pose-model-path", str(pose_model_path)])
    if args.debug_stream:
        command.extend(["--debug-stream", "--debug-host", args.debug_host, "--debug-port", str(args.debug_port)])

    print("Starting Pi camera loop:")
    print(" ".join(command))
    print(f"FLOWPILOT_API_BASE={env['FLOWPILOT_API_BASE']}")
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        print("\nStopping Pi camera loop.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
