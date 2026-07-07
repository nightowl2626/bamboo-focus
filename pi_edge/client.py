import os
from typing import Any

import httpx

from .queue import EventQueue


class PiEventClient:
    def __init__(self, api_base: str, token: str, queue: EventQueue):
        self.api_base = api_base.rstrip("/")
        self.token = token
        self.queue = queue

    def send_event(self, event: dict[str, Any], queue_on_failure: bool = True) -> bool:
        try:
            response = httpx.post(
                f"{self.api_base}/events",
                headers={"Authorization": f"Bearer {self.token}"},
                json=event,
                timeout=5,
            )
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            if queue_on_failure:
                self.queue.push(event)
            return False

    def poll_command(self) -> dict[str, Any] | None:
        try:
            response = httpx.get(
                f"{self.api_base}/pi/commands",
                headers={"Authorization": f"Bearer {self.token}"},
                timeout=2,
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        command = payload.get("command")
        return command if isinstance(command, dict) else None

    def flush_queue(self) -> int:
        sent = 0
        failures = []
        pending = self.queue.peek_all()
        for event in pending:
            if self.send_event(event):
                sent += 1
            else:
                failures.append(event)
        self.queue.replace(failures)
        return sent

    def clear_queue(self) -> None:
        self.queue.replace([])


def default_client() -> PiEventClient:
    return PiEventClient(
        api_base=os.getenv("FLOWPILOT_API_BASE", "http://127.0.0.1:8000"),
        token=os.getenv("FLOWPILOT_PI_TOKEN", "dev-local-token"),
        queue=EventQueue(os.getenv("FLOWPILOT_PI_QUEUE_PATH", "pi_event_queue.jsonl")),
    )
