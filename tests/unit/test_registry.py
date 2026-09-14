from pathlib import Path

import pytest
import torch

from video_action_mlops.config.loader import load_config
from video_action_mlops.models.registry import build_model
from video_action_mlops.models.spatial import SpatialBackbone
from video_action_mlops.models.temporal import TemporalAggregatorExport, TemporalAggregatorTrainable

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def _load_base_cfg():
    return load_config(CONFIGS_DIR / "base.yaml")


def test_build_phase1_returns_spatial_backbone():
    cfg = _load_base_cfg()
    model = build_model(cfg, stage="phase1")
    assert isinstance(model, SpatialBackbone)


def test_build_phase1_uses_embed_dim_out_from_config():
    cfg = _load_base_cfg()
    model = build_model(cfg, stage="phase1")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, cfg.phase1.embed_dim_out)


def test_build_phase2_trainable_variant():
    cfg = _load_base_cfg()
    model = build_model(cfg, stage="phase2", variant="trainable")
    assert isinstance(model, TemporalAggregatorTrainable)


def test_build_phase2_export_variant():
    cfg = _load_base_cfg()
    model = build_model(cfg, stage="phase2", variant="export")
    assert isinstance(model, TemporalAggregatorExport)


def test_build_phase2_reads_dims_from_config():
    cfg = _load_base_cfg()
    model = build_model(cfg, stage="phase2", variant="trainable")
    x = torch.randn(2, 5, cfg.phase2.input_dim)
    out = model(x)
    assert out.shape == (2, cfg.data.num_classes)


def test_build_model_rejects_invalid_stage():
    cfg = _load_base_cfg()
    with pytest.raises(ValueError, match="stage"):
        build_model(cfg, stage="phase99")  # type: ignore[arg-type]


def test_build_model_rejects_invalid_variant():
    cfg = _load_base_cfg()
    with pytest.raises(ValueError, match="variant"):
        build_model(cfg, stage="phase2", variant="bad")  # type: ignore[arg-type]
