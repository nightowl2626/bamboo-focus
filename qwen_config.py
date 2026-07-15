"""Qwen model role selection.

Keep the OpenAI-compatible request path, but allow expensive analysis calls and
cheap copy/summary calls to use different model IDs.
"""

from __future__ import annotations

import os


DEFAULT_QWEN_MODEL = "qwen-plus"

ROLE_ENV = {
    "nudge": ("QWEN_NUDGE_MODEL", "QWEN_ANALYSIS_MODEL"),
    "posture": ("QWEN_POSTURE_MODEL", "QWEN_ANALYSIS_MODEL"),
    "calibration": ("QWEN_CALIBRATION_MODEL", "QWEN_ANALYSIS_MODEL"),
    "copywriter": ("QWEN_COPY_MODEL", "QWEN_FAST_MODEL"),
    "summary": ("QWEN_SUMMARY_MODEL", "QWEN_FAST_MODEL"),
}


def qwen_model_for(role: str, default: str = DEFAULT_QWEN_MODEL) -> str:
    for env_name in ROLE_ENV.get(role, ()):
        value = os.getenv(env_name)
        if value:
            return value
    return os.getenv("QWEN_MODEL", default)


def qwen_model_config() -> dict[str, str]:
    return {
        role: qwen_model_for(role)
        for role in ("nudge", "posture", "calibration", "copywriter", "summary")
    }
