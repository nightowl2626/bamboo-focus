"""Read-only MCP server for Bamboo Focus edge-agent data.

This stdio server exposes privacy-first tools over local JSON/JSONL artifacts.
It does not send data over the network and it never exposes raw camera frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from edge_privacy import build_memory_profile, build_privacy_ledger, latest_decision_trace
from nudge import AgentPaths, DataTools, execute_tool


def json_text(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, indent=2),
            }
        ]
    }


def tool_schema() -> list[dict[str, Any]]:
    return [
        {
            "name": "get_current_focus_state",
            "description": "Read current local focus, posture, object, baseline, and latest nudge state. No raw video is exposed.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_recent_posture_summary",
            "description": "Read recent compact posture analyses and local posture metrics from edge-derived JSON.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 12},
                    "include_raw_metrics": {"type": "boolean", "default": False},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_object_dwell_report",
            "description": "Read compact object dwell summaries after local whitelist filtering.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "min_duration_minutes": {"type": "integer", "minimum": 0, "default": 0},
                    "min_seen_count": {"type": "integer", "minimum": 1, "default": 1},
                    "lookback_hours": {"type": "number", "minimum": 0, "default": 4},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_recent_nudge_history",
            "description": "Read compact recent nudge decisions and suppressions.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20}
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "explain_last_nudge",
            "description": "Explain the last nudge decision, including evidence counts, tool path, and privacy guards.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_privacy_ledger",
            "description": "Read the hardware/privacy boundary ledger for the EdgeAgent demo.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_memory_profile",
            "description": "Read compact cross-session memory. Contains no raw video, frames, or full sensor streams.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    ]


class BambooMcpServer:
    def __init__(self, paths: AgentPaths, nudge_mode: str, lookback_hours: float) -> None:
        self.paths = paths
        self.nudge_mode = nudge_mode
        self.tools = DataTools(paths, lookback_hours)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_current_focus_state":
            return json_text(execute_tool(self.tools, "get_monitor_overview", {}))
        if name == "get_recent_posture_summary":
            payload = execute_tool(
                self.tools,
                "get_recent_posture_analyses",
                {"limit": int(arguments.get("limit", 12))},
            )
            if bool(arguments.get("include_raw_metrics", False)):
                payload["raw_metrics"] = execute_tool(self.tools, "get_recent_raw_posture_summary", {"max_events": 30})
            return json_text(payload)
        if name == "get_object_dwell_report":
            return json_text(
                execute_tool(
                    self.tools,
                    "get_object_dwell_report",
                    {
                        "min_duration_minutes": int(arguments.get("min_duration_minutes", 0)),
                        "min_seen_count": int(arguments.get("min_seen_count", 1)),
                        "lookback_hours": arguments.get("lookback_hours"),
                    },
                )
            )
        if name == "get_recent_nudge_history":
            return json_text(execute_tool(self.tools, "get_recent_nudge_history", {"limit": int(arguments.get("limit", 20))}))
        if name == "explain_last_nudge":
            return json_text(latest_decision_trace(self.paths))
        if name == "get_privacy_ledger":
            return json_text(build_privacy_ledger(self.paths, self.nudge_mode))
        if name == "get_memory_profile":
            return json_text(build_memory_profile(self.paths))
        raise ValueError(f"unknown tool: {name}")

    def handle(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = message.get("method")
        request_id = message.get("id")
        if method == "notifications/initialized":
            return None
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "bamboo-focus-edge", "version": "0.1.0"},
                }
            elif method == "tools/list":
                result = {"tools": tool_schema()}
            elif method == "tools/call":
                params = message.get("params") if isinstance(message.get("params"), dict) else {}
                name = str(params.get("name", ""))
                arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
                result = self.call_tool(name, arguments)
            else:
                raise ValueError(f"unsupported method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32000, "message": str(exc)},
            }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Bamboo Focus read-only MCP server.")
    parser.add_argument("--baseline", default="baseline.json")
    parser.add_argument("--monitor-data-dir", default="monitor_data")
    parser.add_argument("--object-monitor-dir", default="object_monitor_data")
    parser.add_argument("--agent-data-dir", default="nudge_agent_data")
    parser.add_argument("--nudge-mode", choices=("auto", "qwen", "local"), default="auto")
    parser.add_argument("--lookback-hours", type=float, default=4.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    server = BambooMcpServer(
        AgentPaths(
            baseline=Path(args.baseline),
            long_monitor_dir=Path(args.monitor_data_dir),
            object_monitor_dir=Path(args.object_monitor_dir),
            agent_data_dir=Path(args.agent_data_dir),
        ),
        nudge_mode=args.nudge_mode,
        lookback_hours=args.lookback_hours,
    )
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
        else:
            response = server.handle(message)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
