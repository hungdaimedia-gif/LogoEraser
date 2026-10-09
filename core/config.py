"""Configuration management for LogoEraser.

Loads external tool paths (ffmpeg, ffprobe) and default output directories from config.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import TypedDict

logger = logging.getLogger(__name__)


def app_dir() -> Path:
    """Thư mục gốc của ứng dụng.

    Bản exe (PyInstaller) thì là thư mục chứa LogoEraser.exe — để người dùng
    sửa được config.json và đặt ffmpeg cạnh exe; bản chạy từ mã nguồn thì là
    thư mục dự án.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


DEFAULT_CONFIG_PATH = str(app_dir() / "config.json")


class AppConfig(TypedDict, total=False):
    ffmpeg: str
    ffprobe: str
    default_output_dir: str


DEFAULT_CONFIG: AppConfig = {
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    "default_output_dir": "",
}


def load_config(path: str | None = None) -> AppConfig:
    """Load configuration from JSON file or return default configuration.

    If file does not exist, returns defaults without error.
    Defaults to config.json located at the project root (Review 1b-r2 N16).
    """
    config_file = path if path is not None else DEFAULT_CONFIG_PATH
    if not os.path.isfile(config_file):
        return dict(DEFAULT_CONFIG)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cfg: AppConfig = dict(DEFAULT_CONFIG)
            for k in ("ffmpeg", "ffprobe", "default_output_dir"):
                if k in data and isinstance(data[k], str):
                    cfg[k] = data[k]
            return cfg
    except Exception as e:
        logger.warning("Failed to parse config file '%s': %s", config_file, e)

    return dict(DEFAULT_CONFIG)


def tool_path(cfg: AppConfig, name: str) -> str:
    """Resolve ffmpeg/ffprobe: giá trị trong config → PATH → bản đặt cạnh ứng dụng.

    Thứ tự tìm bản đặt cạnh: `<app>/ffmpeg/bin/<name>.exe`, `<app>/ffmpeg/<name>.exe`,
    `<app>/<name>.exe`. Không tìm thấy thì trả nguyên giá trị để lỗi báo rõ tên.
    """
    import shutil

    value = str(cfg.get(name) or name)
    if os.path.isfile(value) or shutil.which(value):
        return value
    exe = name + (".exe" if sys.platform == "win32" else "")
    base = app_dir()
    for cand in (base / "ffmpeg" / "bin" / exe, base / "ffmpeg" / exe, base / exe):
        if cand.is_file():
            return str(cand)
    return value


def check_tools(ffmpeg: str, ffprobe: str) -> str | None:
    """Return a Vietnamese error message if ffmpeg/ffprobe cannot be run, else None."""
    import shutil

    missing = [t for t in (ffmpeg, ffprobe) if not (os.path.isfile(t) or shutil.which(t))]
    if not missing:
        return None
    return (
        "Không tìm thấy " + ", ".join(missing) + ".\n"
        "Xử lý video cần FFmpeg: cài FFmpeg và thêm vào PATH, hoặc chép thư mục "
        f"ffmpeg (có bin\\ffmpeg.exe, bin\\ffprobe.exe) vào '{app_dir()}', "
        "hoặc ghi đường dẫn đầy đủ vào config.json."
    )


def save_config(cfg: AppConfig, path: str | None = None) -> None:
    """Save configuration dictionary to JSON file."""
    config_file = path if path is not None else DEFAULT_CONFIG_PATH
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
