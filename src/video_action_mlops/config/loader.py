"""Load + validate AppConfig từ YAML, có override qua biến môi trường.

Đây là ĐIỂM VÀO DUY NHẤT để lấy config trong toàn bộ codebase. Không
script/module nào khác được gọi thẳng yaml.safe_load() rồi tự dùng dict —
luôn phải đi qua load_config() để validation fail-fast chạy trước khi
dùng bất kỳ giá trị nào.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from video_action_mlops.config.schema import AppConfig


class ConfigError(RuntimeError):
    """Raised khi config không load được hoặc không hợp lệ."""


def _apply_env_overrides(raw: dict) -> dict:
    """Override field dạng APP__<section>__<field>=value, vd APP__phase1__epochs=5.

    Chỉ hỗ trợ override nông (đúng 1 section, 1 field) — đủ dùng cho quy mô
    project này (vd đổi epochs khi debug CI mà không sửa file YAML).
    """
    prefix = "APP__"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = key.split("__")
        if len(parts) != 3:
            continue
        _, section, field = parts
        if section not in raw:
            continue
        raw[section][field] = yaml.safe_load(value)  # "5" -> int, "true" -> bool, ...
    return raw


def load_config(path: str | Path, env_file: str | Path | None = None) -> AppConfig:
    """Load, override bằng env, và validate config. Raise ConfigError ngay nếu sai."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Không tìm thấy file config: {path}")

    load_dotenv(env_file) if env_file is not None else load_dotenv()

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    raw = _apply_env_overrides(raw)

    try:
        return AppConfig.model_validate(raw)
    except ValidationError as exc:
        # Bọc lại thành ConfigError để nơi gọi (scripts/, training/, ...) không
        # cần biết chi tiết pydantic, nhưng message gốc vẫn giữ nguyên để debug.
        raise ConfigError(f"Config không hợp lệ ở {path}:\n{exc}") from exc
