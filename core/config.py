"""Configuration management for LogoEraser.

Loads external tool paths (ffmpeg, ffprobe) and default output directories from config.json.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TypedDict

logger = logging.getLogger(__name__)


class AppConfig(TypedDict, total=False):
    ffmpeg: str
    ffprobe: str
    default_output_dir: str


DEFAULT_CONFIG: AppConfig = {
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffprobe",
    "default_output_dir": "",
}


def load_config(path: str = "config.json") -> AppConfig:
    """Load configuration from JSON file or return default configuration.

    If file does not exist, returns defaults without error.
    """
    if not os.path.isfile(path):
        return dict(DEFAULT_CONFIG)

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            cfg: AppConfig = dict(DEFAULT_CONFIG)
            for k in ("ffmpeg", "ffprobe", "default_output_dir"):
                if k in data and isinstance(data[k], str):
                    cfg[k] = data[k]
            return cfg
    except Exception as e:
        logger.warning("Failed to parse config file '%s': %s", path, e)

    return dict(DEFAULT_CONFIG)


def save_config(cfg: AppConfig, path: str = "config.json") -> None:
    """Save configuration dictionary to JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
