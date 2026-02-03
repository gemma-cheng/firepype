# firepype/__init__.py
from __future__ import annotations

__version__ = "0.0.1"

from .config import (
    PipelineConfig,
    RunSpec,
    QASettings,
    WavecalSettings,
    ExtractionSettings,
    SlitDetectionSettings,
)
from .pipeline import run_ab_pairs
from .telluric import apply_telluric_correction

__all__ = [
    "__version__",
    "PipelineConfig",
    "RunSpec",
    "QASettings",
    "WavecalSettings",
    "ExtractionSettings",
    "SlitDetectionSettings",
    "run_ab_pairs",
    "apply_telluric_correction",
]
