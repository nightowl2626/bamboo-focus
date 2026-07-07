"""Accumulates per-frame posture/motion metrics into fixed windows.

The Pi does no posture judgment: it collects raw metric samples and ships
them to the hub as posture_window events. All reasoning happens hub-side.
Stdlib only so it imports on any machine without mediapipe or OpenCV.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

METRIC_NAMES = (
    "forward_head_ratio",
    "shoulder_tilt_ratio",
    "torso_lean_ratio",
    "motion_score",
    "confidence",
    "shoulder_offset_ratio",
    "elbow_angle_degrees",
    "arm_elevation_ratio",
)


@dataclass
class MetricSample:
    forward_head_ratio: float | None = None
    shoulder_tilt_ratio: float | None = None
    torso_lean_ratio: float | None = None
    motion_score: float | None = None
    confidence: float | None = None
    shoulder_offset_ratio: float | None = None
    elbow_angle_degrees: float | None = None
    arm_elevation_ratio: float | None = None


class MetricsWindowBuilder:
    def __init__(self, window_seconds: float, sample_interval_seconds: float) -> None:
        self.window_seconds = window_seconds
        self.sample_interval_seconds = sample_interval_seconds
        self._window_start: datetime | None = None
        self._samples: list[MetricSample] = []

    def add(self, sample: MetricSample, now: datetime) -> dict[str, Any] | None:
        if self._window_start is None:
            self._window_start = now
        self._samples.append(sample)
        elapsed = (now - self._window_start).total_seconds()
        if elapsed + self.sample_interval_seconds < self.window_seconds:
            return None
        payload = self._build_payload()
        self._window_start = None
        self._samples = []
        return payload

    def _build_payload(self) -> dict[str, Any]:
        series = {
            name: [
                None if getattr(sample, name) is None else round(getattr(sample, name), 2)
                for sample in self._samples
            ]
            for name in METRIC_NAMES
        }
        return {
            "window_start": self._window_start.isoformat(),
            "sample_interval_seconds": self.sample_interval_seconds,
            "series": series,
        }
