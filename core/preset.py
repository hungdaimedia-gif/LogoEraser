"""Preset configuration module for LogoEraser.

Defines schemas, conversion routines between relative and absolute coordinates,
strict validation, and JSON persistence used by both GUI and CLI workflows.
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

VALID_VIDEO_MODES = {"fast", "quality"}
VALID_VIDEO_ENCODERS = {"auto", "h264_nvenc", "h264_qsv", "h264_amf", "libx264"}


def parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB' into BGR uint8 tuple.

    Raises:
        TypeError: If hex_str is not a string.
        ValueError: If hex_str does not match '#RRGGBB' format.
    """
    if not isinstance(hex_str, str):
        raise TypeError(f"solid_color must be a string, got {type(hex_str).__name__}")
    clean = hex_str.strip()
    if not clean.startswith("#") or len(clean) != 7:
        raise ValueError(f"Invalid solid_color '{hex_str}'. Expected format #RRGGBB.")
    try:
        r = int(clean[1:3], 16)
        g = int(clean[3:5], 16)
        b = int(clean[5:7], 16)
    except ValueError as exc:
        raise ValueError(f"Invalid hex characters in solid_color '{hex_str}'.") from exc
    return (b, g, r)  # OpenCV BGR format


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
    """Validate and sanitize a preset structure strictly against PLAN specification."""
    if not isinstance(data, dict):
        raise TypeError(f"Preset must be a dict, got {type(data).__name__}")

    known_top_keys = {"version", "boxes", "method", "params", "video"}
    for k in data:
        if k not in known_top_keys:
            logger.warning("Ignoring unknown key in preset: %s", k)

    # 1. Version check (strict integer, reject bool or float)
    if "version" in data:
        raw_version = data["version"]
        if type(raw_version) is not int or raw_version != 1:
            raise ValueError(
                f"Unsupported preset version '{raw_version}'. Only version 1 (int) is supported."
            )
        version = raw_version
    else:
        version = 1

    # 2. Method check
    raw_method = data.get("method", "inpaint_hq")
    if not isinstance(raw_method, str) or raw_method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method: '{raw_method}'. Must be one of: {sorted(VALID_METHODS)}"
        )
    method = raw_method

    # 3. Boxes check
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
            if type(b[coord]) not in (int, float):
                raise TypeError(f"Box at index {idx} '{coord}' must be numeric (not {type(b[coord]).__name__})")

        w_val = float(b["w"])
        h_val = float(b["h"])
        x_val = float(b["x"])
        y_val = float(b["y"])

        if w_val <= 0 or h_val <= 0:
            raise ValueError(f"Box at index {idx} w and h must be positive > 0 (got w={w_val}, h={h_val})")

        unit = str(b.get("unit", "relative")).lower()
        if unit not in ("relative", "absolute"):
            raise ValueError(
                f"Box at index {idx} unit must be 'relative' or 'absolute', got '{unit}'"
            )

        if unit == "relative":
            if x_val < 0.0 or x_val > 1.0 or y_val < 0.0 or y_val > 1.0:
                raise ValueError(f"Box at index {idx} relative coordinates must be in [0.0, 1.0]")
            if w_val > 1.0 or h_val > 1.0:
                raise ValueError(f"Box at index {idx} relative dimensions must be <= 1.0")
            if (x_val + w_val) > 1.0001 or (y_val + h_val) > 1.0001:
                raise ValueError(
                    f"Box at index {idx} exceeds bounds: x+w={x_val+w_val:.3f}, y+h={y_val+h_val:.3f}"
                )
        else:
            if x_val < 0 or y_val < 0:
                raise ValueError(f"Box at index {idx} absolute coordinates cannot be negative")

        validated_boxes.append({
            "x": b["x"],
            "y": b["y"],
            "w": b["w"],
            "h": b["h"],
            "unit": unit,
        })

    # 4. Params check (N7: strict type check, reject bool or float for integer parameters)
    params_dict = data.get("params", {})
    if not isinstance(params_dict, dict):
        raise TypeError("Preset 'params' must be a dict")

    clean_params: PresetParams = dict(DEFAULT_PARAMS)
    for pk, pv in params_dict.items():
        if pk in ("dilate", "feather", "radius", "blur_ksize", "mosaic_block"):
            if type(pv) is not int:
                raise TypeError(f"Parameter '{pk}' must be an int, got {type(pv).__name__}")
            if pk == "dilate" and pv < 0:
                raise ValueError(f"Parameter 'dilate' cannot be negative (got {pv})")
            if pk == "feather" and pv < 0:
                raise ValueError(f"Parameter 'feather' cannot be negative (got {pv})")
            if pk == "radius" and pv < 1:
                raise ValueError(f"Parameter 'radius' must be >= 1 (got {pv})")
            if pk == "blur_ksize" and pv < 3:
                raise ValueError(f"Parameter 'blur_ksize' must be >= 3 (got {pv})")
            if pk == "mosaic_block" and pv < 2:
                raise ValueError(f"Parameter 'mosaic_block' must be >= 2 (got {pv})")
            clean_params[pk] = pv  # type: ignore[literal-required]

        elif pk == "overlay_opacity":
            if type(pv) not in (int, float):
                raise TypeError(f"Parameter 'overlay_opacity' must be a float/int, got {type(pv).__name__}")
            val_f = float(pv)
            if val_f < 0.0 or val_f > 1.0:
                raise ValueError(f"Parameter 'overlay_opacity' must be in [0.0, 1.0] (got {val_f})")
            clean_params["overlay_opacity"] = val_f

        elif pk == "solid_color":
            parse_hex_color(pv)  # Validates string and #RRGGBB strictly
            clean_params["solid_color"] = str(pv)

        elif pk == "overlay_path":
            clean_params["overlay_path"] = str(pv) if pv is not None else None
        else:
            logger.warning("Ignoring unknown parameter: %s", pk)

    # 5. Video check (strict types)
    video_dict = data.get("video", {})
    if not isinstance(video_dict, dict):
        raise TypeError("Preset 'video' must be a dict")

    clean_video: VideoSettings = dict(DEFAULT_VIDEO)
    for vk, vv in video_dict.items():
        if vk == "mode":
            v_mode = str(vv).lower()
            if v_mode not in VALID_VIDEO_MODES:
                raise ValueError(
                    f"Invalid video.mode '{vv}'. Must be one of: {sorted(VALID_VIDEO_MODES)}"
                )
            clean_video["mode"] = v_mode
        elif vk == "encoder":
            v_enc = str(vv).lower()
            if v_enc not in VALID_VIDEO_ENCODERS:
                raise ValueError(
                    f"Invalid video.encoder '{vv}'. Must be one of: {sorted(VALID_VIDEO_ENCODERS)}"
                )
            clean_video["encoder"] = v_enc
        elif vk == "crf":
            if type(vv) is not int:
                raise TypeError(f"video.crf must be an integer, got {type(vv).__name__}")
            if vv < 0 or vv > 51:
                raise ValueError(f"video.crf must be in range [0, 51] (got {vv})")
            clean_video["crf"] = vv
        elif vk == "copy_audio":
            if type(vv) is not bool:
                raise TypeError(
                    f"video.copy_audio must be a strict boolean (got {type(vv).__name__})"
                )
            clean_video["copy_audio"] = vv
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
