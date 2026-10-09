"""LogoEraser core package."""

from .preset import (
    Box,
    Preset,
    get_default_preset,
    load_preset,
    save_preset,
    to_abs_box,
    to_rel_box,
    validate_preset,
)
from .image_engine import apply, build_mask, detect_logo_boxes

__all__ = [
    "Box",
    "Preset",
    "get_default_preset",
    "load_preset",
    "save_preset",
    "to_abs_box",
    "to_rel_box",
    "validate_preset",
    "apply",
    "build_mask",
    "detect_logo_boxes",
]
