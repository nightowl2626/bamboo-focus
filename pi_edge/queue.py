import json
import os
from pathlib import Path
from typing import Any


class EventQueue:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def push(self, event: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event) + "\n")

    def peek_all(self) -> list[dict[str, Any]]:
        """Read all events without truncating the queue file.

        Non-destructive: callers are responsible for calling `replace()`
        once events have been handled. Malformed/truncated lines (e.g. from
        a crash mid-write) are skipped rather than raising, so a single
        poison-pill line can't permanently wedge the queue.
        """
        if not self.path.exists():
            return []
        events = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    def replace(self, events: list[dict[str, Any]]) -> None:
        """Atomically rewrite the queue file to contain exactly `events`.

        Writes to a temp file and uses os.replace so a crash mid-write
        never leaves the queue file partially written or corrupted.
        """
        tmp_path = self.path.with_name(self.path.name + ".tmp")
        content = "".join(json.dumps(event) + "\n" for event in events)
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, self.path)

