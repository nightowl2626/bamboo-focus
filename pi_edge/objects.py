"""MediaPipe object detection helpers for Pi-side desk snapshots.

The Pi only detects and compresses objects. Qwen reasoning stays in the
hub/backend, matching the posture-window architecture.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ObjectDetectorUnavailable(Exception):
    pass


@dataclass(frozen=True)
class ObjectDetection:
    label: str
    score: float
    bbox: tuple[int, int, int, int]

    def as_payload(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "score": round(self.score, 3),
            "bbox": list(self.bbox),
        }


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return 0.0 if union == 0 else intersection / union


def dedupe_detections(
    detections: list[ObjectDetection],
    iou_threshold: float = 0.55,
) -> list[ObjectDetection]:
    """Keep the highest-confidence detection for near-identical label+bbox hits."""
    ordered = sorted(detections, key=lambda item: item.score, reverse=True)
    kept: list[ObjectDetection] = []
    for detection in ordered:
        if any(
            detection.label == existing.label
            and bbox_iou(detection.bbox, existing.bbox) >= iou_threshold
            for existing in kept
        ):
            continue
        kept.append(detection)
    return kept


def normalize_label(label: str) -> str:
    return " ".join(label.lower().strip().split())


def filter_detections_by_label(
    detections: list[ObjectDetection],
    exclude_labels: list[str] | tuple[str, ...] | set[str] | None,
) -> list[ObjectDetection]:
    if not exclude_labels:
        return detections
    excluded = {normalize_label(label) for label in exclude_labels}
    return [item for item in detections if normalize_label(item.label) not in excluded]


def build_object_snapshot_payload(
    detections: list[ObjectDetection],
    frame_size: tuple[int, int],
    captured_at: datetime,
    directives_version: str | None = None,
) -> dict[str, Any]:
    payload = {
        "captured_at": captured_at.isoformat(),
        "frame_size": list(frame_size),
        "objects": [item.as_payload() for item in dedupe_detections(detections)],
    }
    if directives_version is not None:
        payload["directives_version"] = directives_version
    return payload


class ObjectCaptureCollector:
    def __init__(self) -> None:
        self._detections: list[ObjectDetection] = []

    def add(self, detections: list[ObjectDetection]) -> None:
        self._detections.extend(detections)

    def snapshot(
        self,
        frame_size: tuple[int, int],
        captured_at: datetime,
    ) -> dict[str, Any]:
        payload = build_object_snapshot_payload(self._detections, frame_size, captured_at)
        payload["mode"] = "manual"
        return payload


class ObjectDetector:
    def __init__(
        self,
        model_path: str | None = None,
        score_threshold: float = 0.35,
        max_results: int = 25,
    ) -> None:
        model_path = model_path or os.getenv("FLOWPILOT_OBJECT_DETECTION_MODEL_PATH")
        if not model_path:
            raise ObjectDetectorUnavailable(
                "FLOWPILOT_OBJECT_DETECTION_MODEL_PATH is not configured"
            )
        if not os.path.exists(model_path):
            raise ObjectDetectorUnavailable(f"object detector model not found: {model_path}")
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise ObjectDetectorUnavailable("mediapipe is not installed") from exc

        try:
            model_buffer = Path(model_path).read_bytes()
        except OSError as exc:
            raise ObjectDetectorUnavailable(f"could not read object detector model: {model_path}") from exc

        base_options = python.BaseOptions(model_asset_buffer=model_buffer)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            max_results=max_results,
            score_threshold=score_threshold,
            running_mode=vision.RunningMode.IMAGE,
        )
        self._mp = mp
        try:
            self._detector = vision.ObjectDetector.create_from_options(options)
        except Exception as exc:
            raise ObjectDetectorUnavailable(f"could not initialize object detector: {exc}") from exc

    def detect(self, frame_rgb) -> list[ObjectDetection]:
        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._detector.detect(image)
        except Exception:
            return []
        detections: list[ObjectDetection] = []
        for item in result.detections:
            if not item.categories:
                continue
            category = max(item.categories, key=lambda cat: cat.score)
            label = category.category_name or category.display_name or "unknown"
            box = item.bounding_box
            detections.append(
                ObjectDetection(
                    label=label,
                    score=float(category.score),
                    bbox=(
                        int(box.origin_x),
                        int(box.origin_y),
                        int(box.origin_x + box.width),
                        int(box.origin_y + box.height),
                    ),
                )
            )
        return detections
