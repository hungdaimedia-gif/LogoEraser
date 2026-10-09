"""Tests for LogoEraser preset and image engine (Phase 1a Gate)."""

from __future__ import annotations

import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import pytest

from core.preset import (
    get_default_preset,
    load_preset,
    save_preset,
    to_abs_box,
    to_rel_box,
    validate_preset,
)
from core.image_engine import apply, build_mask, detect_logo_boxes
from tests.make_samples import create_synthetic_image


def test_preset_defaults_and_validation(tmp_path):
    preset = get_default_preset()
    assert preset["version"] == 1
    assert preset["method"] == "inpaint_hq"
    assert len(preset["boxes"]) == 1

    # Save and reload
    preset_file = str(tmp_path / "preset.json")
    save_preset(preset, preset_file)
    reloaded = load_preset(preset_file)

    assert reloaded["method"] == "inpaint_hq"
    assert reloaded["params"]["dilate"] == 3


def test_coordinate_conversions():
    frame_w, frame_h = 1920, 1080

    rel_box = {"x": 0.5, "y": 0.25, "w": 0.1, "h": 0.1, "unit": "relative"}
    abs_box = to_abs_box(rel_box, frame_w, frame_h)
    assert abs_box["unit"] == "absolute"
    assert abs_box["x"] == 960
    assert abs_box["y"] == 270
    assert abs_box["w"] == 192
    assert abs_box["h"] == 108

    # Convert back to relative
    rel_back = to_rel_box(abs_box, frame_w, frame_h)
    assert rel_back["unit"] == "relative"
    assert pytest.approx(rel_back["x"], abs=0.01) == 0.5
    assert pytest.approx(rel_back["y"], abs=0.01) == 0.25


def test_invalid_method_raises():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = {"x": 10, "y": 10, "w": 20, "h": 20, "unit": "absolute"}
    with pytest.raises(ValueError, match="Invalid method"):
        apply(img, [box], "non_existent_method", {})


def test_all_seven_methods_execute(tmp_path):
    clean, corrupted, box = create_synthetic_image(400, 300, (200, 100, 60, 40))

    # Create dummy PNG for overlay test
    ov_file = str(tmp_path / "logo.png")
    pil_ov = Image.new("RGBA", (50, 50), (255, 0, 0, 180))
    pil_ov.save(ov_file)

    params = {
        "dilate": 3,
        "feather": 4,
        "radius": 5,
        "blur_ksize": 25,
        "mosaic_block": 8,
        "solid_color": "#00FF00",
        "overlay_path": ov_file,
        "overlay_opacity": 0.9,
    }

    methods = [
        "inpaint_fast",
        "inpaint_hq",
        "clone_smooth",
        "blur",
        "mosaic",
        "solid",
        "overlay",
    ]

    for m in methods:
        out = apply(corrupted, [box], m, params)
        assert isinstance(out, np.ndarray), f"Method {m} did not return ndarray"
        assert out.shape == corrupted.shape, f"Method {m} altered image shape"
        assert out.dtype == np.uint8, f"Method {m} returned non-uint8 dtype"


def test_inpaint_hq_error_reduction():
    """Gate test: inpaint_hq average error in logo region compared to clean background

    must be strictly less than 50% of the initial error caused by the logo.
    """
    clean, corrupted, box = create_synthetic_image(400, 300, (150, 100, 70, 45))
    lx, ly, lw, lh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])

    params = {
        "dilate": 2,
        "feather": 3,
        "radius": 5,
    }

    erased = apply(corrupted, [box], "inpaint_hq", params)

    roi_clean = clean[ly:ly + lh, lx:lx + lw].astype(np.float64)
    roi_corrupted = corrupted[ly:ly + lh, lx:lx + lw].astype(np.float64)
    roi_erased = erased[ly:ly + lh, lx:lx + lw].astype(np.float64)

    initial_error = float(np.mean(np.abs(roi_corrupted - roi_clean)))
    restored_error = float(np.mean(np.abs(roi_erased - roi_clean)))

    error_ratio = restored_error / initial_error
    print(f"\n[inpaint_hq Gate] Initial error: {initial_error:.2f}, Restored error: {restored_error:.2f}, Ratio: {error_ratio:.2%}")

    # Gate: Restored error must be less than 50% of initial error
    assert error_ratio < 0.50, f"Error ratio {error_ratio:.2%} is not < 50%"


def test_detect_logo_boxes():
    # Less than 8 frames -> returns empty
    frames_few = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    assert detect_logo_boxes(frames_few) == []

    # 10 frames with dynamic background and a static logo in top-left
    frames = []
    for i in range(12):
        f = np.full((200, 200, 3), (i * 20) % 255, dtype=np.uint8)
        # Static logo
        cv2.rectangle(f, (20, 20), (60, 50), (255, 255, 255), -1)
        cv2.putText(f, "X", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        frames.append(f)

    detected = detect_logo_boxes(frames, max_boxes=2)
    assert len(detected) >= 1
    d_box = detected[0]
    assert d_box["unit"] == "relative"
    assert d_box["w"] > 0
    assert d_box["h"] > 0
