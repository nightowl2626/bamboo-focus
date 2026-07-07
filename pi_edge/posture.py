"""Forward-head posture detection for a side-profile camera.

Pure math lives here so it is testable without mediapipe or camera hardware;
mediapipe is imported lazily by PostureAnalyzer only.
"""

import math
import os
from dataclasses import dataclass
from pathlib import Path

MIN_VISIBILITY = 0.5
LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER = 7, 8, 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = 13, 14, 15, 16


class PostureUnavailable(Exception):
    pass


@dataclass
class PostureSample:
    forward_head_ratio: float | None
    confidence: float
    shoulder_tilt_ratio: float | None = None
    torso_lean_ratio: float | None = None
    shoulder_offset_ratio: float | None = None
    elbow_angle_degrees: float | None = None
    arm_elevation_ratio: float | None = None


@dataclass
class PostureObservation:
    sample: PostureSample | None
    landmarks: object
    selected_indices: tuple[int, int] | None


@dataclass
class LandmarkAdapter:
    x: float
    y: float
    visibility: float = 1.0


def forward_head_ratio(ear_x: float, ear_y: float, shoulder_x: float, shoulder_y: float) -> float:
    dx = ear_x - shoulder_x
    dy = ear_y - shoulder_y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance == 0:
        return 0.0
    return abs(dx) / distance


def shoulder_tilt_ratio(left_x: float, left_y: float, right_x: float, right_y: float) -> float:
    dx = left_x - right_x
    dy = left_y - right_y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance == 0:
        return 0.0
    return abs(dy) / distance


def torso_lean_ratio(left_shoulder, right_shoulder, left_hip, right_hip) -> float:
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2
    hip_mid_y = (left_hip[1] + right_hip[1]) / 2
    dx = shoulder_mid_x - hip_mid_x
    dy = shoulder_mid_y - hip_mid_y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance == 0:
        return 0.0
    return abs(dx) / distance


def shoulder_offset_ratio(left_shoulder, right_shoulder, left_hip, right_hip) -> float:
    """Shoulder separation normalized by torso length: ~0 side view, 0.5+ frontal."""
    sep_dx = left_shoulder[0] - right_shoulder[0]
    sep_dy = left_shoulder[1] - right_shoulder[1]
    separation = (sep_dx * sep_dx + sep_dy * sep_dy) ** 0.5
    shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_mid_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_mid_x = (left_hip[0] + right_hip[0]) / 2
    hip_mid_y = (left_hip[1] + right_hip[1]) / 2
    t_dx = shoulder_mid_x - hip_mid_x
    t_dy = shoulder_mid_y - hip_mid_y
    torso = (t_dx * t_dx + t_dy * t_dy) ** 0.5
    if torso == 0:
        return 0.0
    return separation / torso


def elbow_angle_degrees(shoulder, elbow, wrist) -> float:
    """Inner elbow angle (0-180) between elbow->shoulder and elbow->wrist."""
    upper = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    fore = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    upper_len = (upper[0] * upper[0] + upper[1] * upper[1]) ** 0.5
    fore_len = (fore[0] * fore[0] + fore[1] * fore[1]) ** 0.5
    if upper_len == 0 or fore_len == 0:
        return 0.0
    cos_angle = (upper[0] * fore[0] + upper[1] * fore[1]) / (upper_len * fore_len)
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.degrees(math.acos(cos_angle))


def arm_elevation_ratio(shoulder_x: float, shoulder_y: float, elbow_x: float, elbow_y: float) -> float:
    """-1 = upper arm hanging down, 0 = horizontal, +1 = elbow above shoulder
    (image y grows downward)."""
    dx = elbow_x - shoulder_x
    dy = elbow_y - shoulder_y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance == 0:
        return 0.0
    return (shoulder_y - elbow_y) / distance


def sample_from_landmarks(landmarks) -> PostureSample | None:
    observation = observe_from_landmarks(landmarks)
    if observation is None:
        return None
    return observation.sample


def observe_from_landmarks(landmarks) -> PostureObservation | None:
    pairs = (
        (LEFT_EAR, LEFT_SHOULDER),
        (RIGHT_EAR, RIGHT_SHOULDER),
    )
    selected = max(
        pairs,
        key=lambda pair: landmarks[pair[0]].visibility + landmarks[pair[1]].visibility,
    )
    ear = landmarks[selected[0]]
    shoulder = landmarks[selected[1]]
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]
    left_hip = landmarks[LEFT_HIP]
    right_hip = landmarks[RIGHT_HIP]
    left_elbow = landmarks[LEFT_ELBOW]
    right_elbow = landmarks[RIGHT_ELBOW]
    left_wrist = landmarks[LEFT_WRIST]
    right_wrist = landmarks[RIGHT_WRIST]

    forward = None
    shoulder_tilt = None
    torso_lean = None
    shoulder_offset = None
    elbow_angle = None
    arm_elevation = None
    confidence_landmarks = []

    if ear.visibility < MIN_VISIBILITY or shoulder.visibility < MIN_VISIBILITY:
        forward = None
    else:
        forward = forward_head_ratio(ear.x, ear.y, shoulder.x, shoulder.y)
        confidence_landmarks.extend([ear, shoulder])

    if left_shoulder.visibility >= MIN_VISIBILITY and right_shoulder.visibility >= MIN_VISIBILITY:
        shoulder_tilt = shoulder_tilt_ratio(
            left_shoulder.x,
            left_shoulder.y,
            right_shoulder.x,
            right_shoulder.y,
        )
        confidence_landmarks.extend([left_shoulder, right_shoulder])

    if (
        left_shoulder.visibility >= MIN_VISIBILITY
        and right_shoulder.visibility >= MIN_VISIBILITY
        and left_hip.visibility >= MIN_VISIBILITY
        and right_hip.visibility >= MIN_VISIBILITY
    ):
        torso_lean = torso_lean_ratio(
            left_shoulder=(left_shoulder.x, left_shoulder.y),
            right_shoulder=(right_shoulder.x, right_shoulder.y),
            left_hip=(left_hip.x, left_hip.y),
            right_hip=(right_hip.x, right_hip.y),
        )
        shoulder_offset = shoulder_offset_ratio(
            left_shoulder=(left_shoulder.x, left_shoulder.y),
            right_shoulder=(right_shoulder.x, right_shoulder.y),
            left_hip=(left_hip.x, left_hip.y),
            right_hip=(right_hip.x, right_hip.y),
        )
        confidence_landmarks.extend([left_hip, right_hip])

    arm_sides = (
        (left_shoulder, left_elbow, left_wrist),
        (right_shoulder, right_elbow, right_wrist),
    )
    arm_shoulder, arm_elbow, arm_wrist = max(
        arm_sides,
        key=lambda side: sum(landmark.visibility for landmark in side),
    )
    if arm_shoulder.visibility >= MIN_VISIBILITY and arm_elbow.visibility >= MIN_VISIBILITY:
        arm_elevation = arm_elevation_ratio(arm_shoulder.x, arm_shoulder.y, arm_elbow.x, arm_elbow.y)
        confidence_landmarks.extend([arm_shoulder, arm_elbow])
        if arm_wrist.visibility >= MIN_VISIBILITY:
            elbow_angle = elbow_angle_degrees(
                (arm_shoulder.x, arm_shoulder.y),
                (arm_elbow.x, arm_elbow.y),
                (arm_wrist.x, arm_wrist.y),
            )
            confidence_landmarks.append(arm_wrist)

    if (
        forward is None
        and shoulder_tilt is None
        and torso_lean is None
        and shoulder_offset is None
        and elbow_angle is None
        and arm_elevation is None
    ):
        sample = None
    else:
        unique = {(id(landmark)): landmark for landmark in confidence_landmarks}
        confidence = sum(landmark.visibility for landmark in unique.values()) / len(unique)
        sample = PostureSample(
            forward_head_ratio=forward,
            shoulder_tilt_ratio=shoulder_tilt,
            torso_lean_ratio=torso_lean,
            shoulder_offset_ratio=shoulder_offset,
            elbow_angle_degrees=elbow_angle,
            arm_elevation_ratio=arm_elevation,
            confidence=confidence,
        )
    return PostureObservation(
        sample=sample,
        landmarks=landmarks,
        selected_indices=selected,
    )


def is_bad(
    sample: PostureSample,
    threshold: float | None = None,
    shoulder_threshold: float | None = None,
    torso_threshold: float | None = None,
) -> bool:
    if threshold is None:
        threshold = float(os.getenv("FLOWPILOT_POSTURE_RATIO", "0.35"))
    if shoulder_threshold is None:
        shoulder_threshold = float(os.getenv("FLOWPILOT_SHOULDER_TILT_RATIO", "0.08"))
    if torso_threshold is None:
        torso_threshold = float(os.getenv("FLOWPILOT_TORSO_LEAN_RATIO", "0.18"))
    return any(
        (
            sample.forward_head_ratio is not None and sample.forward_head_ratio > threshold,
            sample.shoulder_tilt_ratio is not None and sample.shoulder_tilt_ratio > shoulder_threshold,
            sample.torso_lean_ratio is not None and sample.torso_lean_ratio > torso_threshold,
        )
    )


class PostureAnalyzer:
    def __init__(self, model_path: str | None = None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise PostureUnavailable("mediapipe is not installed") from exc
        pose_module = None
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
            pose_module = mp.solutions.pose
        else:
            try:
                from mediapipe.python.solutions import pose as pose_module
            except (ImportError, AttributeError) as exc:
                pose_module = None
        if pose_module is not None:
            self._mode = "solutions"
            self._mp = mp
            self._pose = pose_module.Pose(model_complexity=0)
            return

        model_path = model_path or os.getenv("FLOWPILOT_POSE_LANDMARKER_MODEL_PATH")
        if not model_path:
            raise PostureUnavailable(
                "installed mediapipe package needs FLOWPILOT_POSE_LANDMARKER_MODEL_PATH for Tasks pose"
            )
        if not os.path.exists(model_path):
            raise PostureUnavailable(f"pose landmarker model not found: {model_path}")
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise PostureUnavailable("mediapipe Tasks pose API is not installed") from exc
        try:
            model_buffer = Path(model_path).read_bytes()
        except OSError as exc:
            raise PostureUnavailable(f"could not read pose landmarker model: {model_path}") from exc

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_buffer=model_buffer),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except Exception as exc:
            raise PostureUnavailable(f"could not initialize pose landmarker: {exc}") from exc
        self._mode = "tasks"
        self._mp = mp

    def analyze(self, frame_rgb) -> PostureSample | None:
        observation = self.observe(frame_rgb)
        if observation is None:
            return None
        return observation.sample

    def observe(self, frame_rgb) -> PostureObservation | None:
        if self._mode == "solutions":
            try:
                result = self._pose.process(frame_rgb)
            except Exception:
                return None
            if not result.pose_landmarks:
                return None
            return observe_from_landmarks(result.pose_landmarks.landmark)

        try:
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._landmarker.detect(image)
        except Exception:
            return None
        if not result.pose_landmarks:
            return None
        landmarks = [
            LandmarkAdapter(
                x=landmark.x,
                y=landmark.y,
                visibility=getattr(landmark, "visibility", getattr(landmark, "presence", 1.0)),
            )
            for landmark in result.pose_landmarks[0]
        ]
        return observe_from_landmarks(landmarks)
