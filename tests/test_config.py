# tests/test_config.py
from pathlib import Path

from firepype.config import (
    PipelineConfig,
    RunSpec,
    QASettings,
    WavecalSettings,
    ExtractionSettings,
    SlitDetectionSettings,
)


def test_cfg_construct_defaults(tmp_path: Path):
    cfg = PipelineConfig(
        run=RunSpec(
            raw_dir=tmp_path,
            out_dir=tmp_path / "out",
            arc_path=tmp_path / "arc.fits",
            ref_list_path=tmp_path / "lines.lst",
            user_spec="0001-0002",
        )
    )
    assert isinstance(cfg.qa, QASettings)
    assert isinstance(cfg.wavecal, WavecalSettings)
    assert isinstance(cfg.extraction, ExtractionSettings)
    assert isinstance(cfg.slit, SlitDetectionSettings)
    assert cfg.wavecal.wl_range[0] < cfg.wavecal.wl_range[1]
