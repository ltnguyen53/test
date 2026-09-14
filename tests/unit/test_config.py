from pathlib import Path

import pytest
from pydantic import ValidationError

from video_action_mlops.config.loader import ConfigError, load_config
from video_action_mlops.config.schema import DataConfig

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"


def test_load_valid_config():
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.phase1.embed_dim_out == cfg.phase2.input_dim


def test_dim_contract_violation_raises():
    with pytest.raises(ConfigError, match="Contract vi phạm"):
        load_config(CONFIGS_DIR / "broken_dim_mismatch.yaml")


def test_num_heads_not_divisible_raises():
    with pytest.raises(ConfigError, match="chia hết"):
        load_config(CONFIGS_DIR / "broken_num_heads.yaml")


def test_missing_file_raises_config_error():
    with pytest.raises(ConfigError, match="Không tìm thấy"):
        load_config(CONFIGS_DIR / "does_not_exist.yaml")


# ---------- _apply_env_overrides() qua load_config() — hoàn toàn chưa có
# test nào trước đây, dù được ghi rõ trong docstring loader.py (vd
# "APP__phase1__epochs=5") và có thể dùng thật trong CI để đổi
# hyperparameter mà không sửa file YAML. Phát hiện khi kiểm tra coverage
# (loader.py chỉ 79%, thiếu đúng khối logic này). ----------


def test_env_override_changes_single_field(monkeypatch):
    monkeypatch.setenv("APP__phase1__epochs", "99")
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.phase1.epochs == 99


def test_env_override_parses_yaml_scalar_types_not_just_strings(monkeypatch):
    """yaml.safe_load(value) phải ép đúng kiểu (float, không phải string
    "0.5") -- nếu thiếu bước ép kiểu này, pydantic có thể vô tình accept
    string rồi coerce ngầm, che giấu lỗi thật nếu sau này đổi kiểu field."""
    monkeypatch.setenv("APP__phase1__learning_rate", "0.5")
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.phase1.learning_rate == 0.5
    assert isinstance(cfg.phase1.learning_rate, float)


def test_env_override_ignores_env_vars_without_app_prefix(monkeypatch):
    monkeypatch.setenv("PHASE1__EPOCHS", "99")  # thieu tien to "APP__"
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.phase1.epochs != 99


def test_env_override_ignores_wrong_number_of_parts(monkeypatch):
    """Chỉ hỗ trợ đúng "APP__<section>__<field>" (3 phần khi split theo
    "__") -- key thiếu hoặc thừa phần phải bị bỏ qua an toàn, không crash
    và không áp dụng nhầm."""
    monkeypatch.setenv("APP__phase1", "99")  # thieu field, chi 2 phan
    monkeypatch.setenv("APP__phase1__epochs__extra", "99")  # thua 1 phan, 4 phan
    cfg = load_config(CONFIGS_DIR / "base.yaml")
    assert cfg.phase1.epochs != 99


def test_env_override_ignores_unknown_section(monkeypatch):
    monkeypatch.setenv("APP__khong_ton_tai__field", "99")
    cfg = load_config(CONFIGS_DIR / "base.yaml")  # KHONG duoc raise
    assert cfg is not None


def test_env_override_still_goes_through_validation(monkeypatch):
    """Override qua env KHÔNG được phép né validation -- 1 giá trị làm vi
    phạm contract (num_heads không chia hết input_dim) vẫn phải raise
    ConfigError giống hệt khi sửa trực tiếp trong YAML."""
    monkeypatch.setenv("APP__phase2__num_heads", "3")  # base.yaml: input_dim=32, 32 % 3 != 0
    with pytest.raises(ConfigError, match="chia hết"):
        load_config(CONFIGS_DIR / "base.yaml")


def test_data_config_rejects_non_positive_frame_size():
    """DataConfig.check_frame_size_positive() -- nhánh chưa từng được
    test (phát hiện qua coverage report: schema.py 98%, thiếu đúng dòng
    này)."""
    with pytest.raises(ValidationError, match="frame_size"):
        DataConfig(
            frames_per_video=16, frame_size=(0, 224), use_motion_channel=False, num_classes=10
        )
    with pytest.raises(ValidationError, match="frame_size"):
        DataConfig(
            frames_per_video=16, frame_size=(224, -1), use_motion_channel=False, num_classes=10
        )
