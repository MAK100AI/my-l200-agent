"""Root package initializer exposing ADK `root_agent` and `app` for the CAL GRAD system."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from agent import (
    APP_NAME,
    COORDINATOR_MODEL,
    SPECIALIST_FLASH_MODEL,
    app,
    build_cal_grad_system,
    cal_grad_coordinator,
    create_production_runner,
    evidence_extractor_agent,
    impact_classifier_agent,
    root_agent,
)

__all__ = [
    "APP_NAME",
    "COORDINATOR_MODEL",
    "SPECIALIST_FLASH_MODEL",
    "app",
    "build_cal_grad_system",
    "cal_grad_coordinator",
    "create_production_runner",
    "evidence_extractor_agent",
    "impact_classifier_agent",
    "root_agent",
]
