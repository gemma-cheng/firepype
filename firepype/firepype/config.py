# firepype/config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence, Tuple, Optional, List

@dataclass
class QASettings:
    save_figs: bool = True
    verbose: bool = True
    out_dir: Path = Path("./output")
    qa_dir: Path = field(default_factory=lambda: Path("./output/qa"))

@dataclass
class WavecalSettings:
    wl_range: Tuple[float, float] = (0.83, 2.45)
    grid_step: float = 0.0008
    band_anchors_global: Sequence[Tuple[int, float]] = (
        (100, 0.83),
        (1000, 1.15),
        (1600, 1.79),
        (1950, 2.43),
    )
    max_sep: float = 0.012
    deg: int = 3
    force_global_orient: str = "fwd"

@dataclass
class ExtractionSettings:
    ap_radius: int = 5
    bg_in: int = 8
    bg_out: int = 18
    footprint_half: int = 1
    row_fraction: Tuple[float, float] = (0.35, 0.85)

@dataclass
class SlitDetectionSettings:
    slit_x_hint: Tuple[int, int] = (900, 1300)
    slit_hint_expand: int = 150
    row_fraction: Tuple[float, float] = (0.35, 0.85)

@dataclass
class RunSpec:
    raw_dir: Path
    out_dir: Path
    arc_path: Path
    ref_list_path: Path
    user_spec: str  # e.g., "XXXX-XXXX"

@dataclass
class PipelineConfig:
    run: RunSpec
    qa: QASettings = field(default_factory=QASettings)
    wavecal: WavecalSettings = field(default_factory=WavecalSettings)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    slit: SlitDetectionSettings = field(default_factory=SlitDetectionSettings)
