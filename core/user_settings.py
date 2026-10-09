"""Ghi nhớ lựa chọn của người dùng giữa các lần mở app (thư mục xuất, thư mục mở file...).

Lưu ở `%APPDATA%\\LogoEraser\\settings.json` — không cạnh exe, vì thư mục cài
(ví dụ Program Files) có thể không ghi được và mỗi tài khoản Windows nên có
lựa chọn riêng. Khác `config.json` (cấu hình do người dùng tự sửa tay).
File hỏng/thiếu ⇒ dùng mặc định, không bao giờ làm app không mở được.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Khoá được nhớ. Thêm khoá mới thì thêm vào đây (giá trị mặc định = "").
KEYS = (
    "output_dir",  # ô "Thư mục xuất"
    "input_dir",  # nơi mở hộp thoại Thêm tệp / Thêm thư mục lần trước
    "preset_dir",  # nơi mở/lưu preset lần trước
    "overlay_dir",  # nơi chọn ảnh phủ PNG lần trước
)


def settings_path() -> Path:
    env = os.environ.get("LOGOERASER_SETTINGS")
    if env:
        return Path(env)
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "LogoEraser" / "settings.json"


def load_settings() -> dict[str, str]:
    data: dict[str, str] = {k: "" for k in KEYS}
    p = settings_path()
    try:
        raw: Any = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for k in KEYS:
                if isinstance(raw.get(k), str):
                    data[k] = raw[k]
    except FileNotFoundError:
        pass
    except Exception as e:  # file hỏng ⇒ bỏ qua, lần lưu sau sẽ ghi lại
        logger.warning("Cannot read settings '%s': %s", p, e)
    return data


def save_settings(data: dict[str, str]) -> None:
    p = settings_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        clean = {k: str(data.get(k, "") or "") for k in KEYS}
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception as e:  # không ghi được (ổ chỉ đọc...) ⇒ app vẫn chạy bình thường
        logger.warning("Cannot save settings '%s': %s", p, e)


def existing_dir(path: str) -> str:
    """Trả lại path nếu thư mục còn tồn tại, không thì "" (ổ USB đã rút, thư mục bị đổi tên...)."""
    return path if path and os.path.isdir(path) else ""
