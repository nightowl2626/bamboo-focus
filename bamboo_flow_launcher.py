"""Local FlowPilot launcher.

This is the desktop-app entry point before packaging. It starts app.py, opens
the local PWA, and shuts the backend down when the launcher exits.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import urlopen


def wait_for_app(url: str, timeout_seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=1.0):
                return True
        except OSError:
            time.sleep(0.4)
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start FlowPilot and open the local app UI.")
    parser.add_argument("--port", type=int, default=8000, help="Local FlowPilot port.")
    parser.add_argument("--host", default="0.0.0.0", help="Host passed to app.py.")
    parser.add_argument("--local", action="store_true", help="Start app.py in offline/local mode.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser window.")
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="Extra arguments passed to app.py after --.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    app_command = [
        sys.executable,
        str(root / "app.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]
    if args.local:
        app_command.append("--local")
    if args.app_args:
        extra = args.app_args[1:] if args.app_args[0] == "--" else args.app_args
        app_command.extend(extra)

    print("Starting FlowPilot backend...")
    process = subprocess.Popen(app_command, cwd=root)
    app_url = f"http://127.0.0.1:{args.port}/app/"
    try:
        if wait_for_app(app_url) and not args.no_browser:
            webbrowser.open(app_url)
        print(f"FlowPilot is running at {app_url}")
        print("Close this window or press Ctrl+C to stop the local backend.")
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping FlowPilot...")
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
