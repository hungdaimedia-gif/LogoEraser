"""Preset configuration module for LogoEraser.

Defines schemas, conversion routines between relative and absolute coordinates,
validation, and JSON persistence used by both GUI and CLI workflows.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

VALID_METHODS = {
    "inpaint_fast",
    "inpaint_hq",
    "clone_smooth",
    "blur",
    "mosaic",
    "solid",
    "overlay",
}


class Box(TypedDict):
    x: float | int
    y: float | int
    w: float | int
    h: float | int
    unit: str


class PresetParams(TypedDict, total=False):
    dilate: int
    feather: int
    radius: int
    blur_ksize: int
    mosaic_block: int
    solid_color: str
    overlay_path: str | None
    overlay_opacity: float


class VideoSettings(TypedDict, total=False):
    mode: str
    encoder: str
    crf: int
    copy_audio: bool


class Preset(TypedDict):
    version: int
    boxes: list[Box]
    method: str
    params: PresetParams
    video: VideoSettings


DEFAULT_PARAMS: PresetParams = {
    "dilate": 3,
    "feather": 4,
    "radius": 5,
    "blur_ksize": 31,
    "mosaic_block": 12,
    "solid_color": "#000000",
    "overlay_path": None,
    "overlay_opacity": 1.0,
}

DEFAULT_VIDEO: VideoSettings = {
    "mode": "fast",
    "encoder": "auto",
    "crf": 20,
    "copy_audio": True,
}


def get_default_preset() -> Preset:
    """Return a fresh default preset structure."""
    return {
        "version": 1,
        "boxes": [
            {
                "x": 0.86,
                "y": 0.90,
                "w": 0.12,
                "h": 0.07,
                "unit": "relative",
            }
        ],
        "method": "inpaint_hq",
        "params": dict(DEFAULT_PARAMS),
        "video": dict(DEFAULT_VIDEO),
    }


def to_abs_box(box: Box | dict[str, Any], frame_w: int, frame_h: int) -> Box:
    """Convert a box (relative or absolute) to pixel coordinates clamped within the frame.

    Ensures width and height are at least 2 pixels.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Frame dimensions must be positive integers.")

    unit = str(box.get("unit", "relative")).lower()
    raw_x = float(box["x"])
    raw_y = float(box["y"])
    raw_w = float(box["w"])
    raw_h = float(box["h"])

    if unit == "relative":
        px_x = int(round(raw_x * frame_w))
        px_y = int(round(raw_y * frame_h))
        px_w = int(round(raw_w * frame_w))
        px_h = int(round(raw_h * frame_h))
    else:
        px_x = int(round(raw_x))
        px_y = int(round(raw_y))
        px_w = int(round(raw_w))
        px_h = int(round(raw_h))

    # Clamp coordinates to frame bounds
    px_x = max(0, min(frame_w - 2, px_x))
    px_y = max(0, min(frame_h - 2, px_y))

    max_w = frame_w - px_x
    max_h = frame_h - px_y

    px_w = max(2, min(max_w, px_w))
    px_h = max(2, min(max_h, px_h))

    return {
        "x": px_x,
        "y": px_y,
        "w": px_w,
        "h": px_h,
        "unit": "absolute",
    }


def to_rel_box(box: Box | dict[str, Any], frame_w: int, frame_h: int) -> Box:
    """Convert an absolute or relative box into a normalized (0..1) relative box."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError("Frame dimensions must be positive integers.")

    unit = str(box.get("unit", "relative")).lower()
    if unit == "relative":
        rx = max(0.0, min(1.0, float(box["x"])))
        ry = max(0.0, min(1.0, float(box["y"])))
        rw = max(0.001, min(1.0 - rx, float(box["w"])))
        rh = max(0.001, min(1.0 - ry, float(box["h"])))
        return {
            "x": round(rx, 4),
            "y": round(ry, 4),
            "w": round(rw, 4),
            "h": round(rh, 4),
            "unit": "relative",
        }

    abs_box = to_abs_box(box, frame_w, frame_h)
    rx = float(abs_box["x"]) / float(frame_w)
    ry = float(abs_box["y"]) / float(frame_h)
    rw = float(abs_box["w"]) / float(frame_w)
    rh = float(abs_box["h"]) / float(frame_h)

    return {
        "x": round(max(0.0, min(1.0, rx)), 4),
        "y": round(max(0.0, min(1.0, ry)), 4),
        "w": round(max(0.001, min(1.0, rw)), 4),
        "h": round(max(0.001, min(1.0, rh)), 4),
        "unit": "relative",
    }


def validate_preset(data: Any) -> Preset:
    """Validate and sanitize a preset structure, filling defaults for missing keys."""
    if not isinstance(data, dict):
        raise TypeError(f"Preset must be a dict, got {type(data).__name__}")

    known_top_keys = {"version", "boxes", "method", "params", "video"}
    for k in data:
        if k not in known_top_keys:
            logger.warning("Ignoring unknown key in preset: %s", k)

    version = int(data.get("version", 1))

    raw_method = data.get("method", "inpaint_hq")
    if not isinstance(raw_method, str) or raw_method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method: '{raw_method}'. Must be one of: {sorted(VALID_METHODS)}"
        )
    method = raw_method

    raw_boxes = data.get("boxes", [])
    if not isinstance(raw_boxes, list):
        raise TypeError(f"Preset 'boxes' must be a list, got {type(raw_boxes).__name__}")

    validated_boxes: list[Box] = []
    for idx, b in enumerate(raw_boxes):
        if not isinstance(b, dict):
            raise TypeError(f"Box at index {idx} must be a dict")
        for coord in ("x", "y", "w", "h"):
            if coord not in b:
                raise ValueError(f"Box at index {idx} is missing '{coord}'")
            if not isinstance(b[coord], (int, float)):
                raise TypeError(f"Box at index {idx} '{coord}' must be numeric")
            if float(b[coord]) < 0:
                raise ValueError(f"Box at index {idx} '{coord}' cannot be negative")

        unit = str(b.get("unit", "relative")).lower()
        if unit not in ("relative", "absolute"):
            raise ValueError(
                f"Box at index {idx} unit must be 'relative' or 'absolute', got '{unit}'"
            )

        validated_boxes.append({
            "x": b["x"],
            "y": b["y"],
            "w": b["w"],
            "h": b["h"],
            "unit": unit,
        })

    # Validate parameters
    params_dict = data.get("params", {})
    if not isinstance(params_dict, dict):
        raise TypeError("Preset 'params' must be a dict")

    clean_params: PresetParams = dict(DEFAULT_PARAMS)
    for pk, pv in params_dict.items():
        if pk in clean_params:
            if pk in ("dilate", "feather", "radius", "blur_ksize", "mosaic_block"):
                clean_params[pk] = int(pv)  # type: ignore[literal-required]
            elif pk == "overlay_opacity":
                clean_params[pk] = float(pv)  # type: ignore[literal-required]
            elif pk == "solid_color":
                clean_params[pk] = str(pv)  # type: ignore[literal-required]
            elif pk == "overlay_path":
                clean_params[pk] = str(pv) if pv is not None else None  # type: ignore[literal-required]
        else:
            logger.warning("Ignoring unknown parameter: %s", pk)

    # Validate video settings
    video_dict = data.get("video", {})
    if not isinstance(video_dict, dict):
        raise TypeError("Preset 'video' must be a dict")

    clean_video: VideoSettings = dict(DEFAULT_VIDEO)
    for vk, vv in video_dict.items():
        if vk in clean_video:
            if vk == "crf":
                clean_video[vk] = int(vv)  # type: ignore[literal-required]
            elif vk == "copy_audio":
                clean_video[vk] = bool(vv)  # type: ignore[literal-required]
            elif vk in ("mode", "encoder"):
                clean_video[vk] = str(vv)  # type: ignore[literal-required]
        else:
            logger.warning("Ignoring unknown video setting: %s", vk)

    return {
        "version": version,
        "boxes": validated_boxes,
        "method": method,
        "params": clean_params,
        "video": clean_video,
    }


def load_preset(path: str) -> Preset:
    """Load and validate a preset from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return validate_preset(data)


def save_preset(preset: Preset, path: str) -> None:
    """Validate and write a preset to a JSON file."""
    sanitized = validate_preset(preset)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)
