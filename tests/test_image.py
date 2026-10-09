"""Comprehensive tests for LogoEraser preset and image engine (Phase 1a Gate)."""

from __future__ import annotations

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
from core.image_engine import apply, build_mask, detect_logo_boxes, _compute_iou
from tests.make_samples import create_synthetic_image, create_striped_sample


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


@pytest.mark.parametrize(
    "bad_patch, expected_exc",
    [
        ({"version": 2}, ValueError),
        ({"method": "unknown_algo"}, ValueError),
        ({"params": {"dilate": -1}}, ValueError),
        ({"params": {"feather": -2}}, ValueError),
        ({"params": {"radius": 0}}, ValueError),
        ({"params": {"blur_ksize": 2}}, ValueError),
        ({"params": {"mosaic_block": 1}}, ValueError),
        ({"params": {"overlay_opacity": 1.5}}, ValueError),
        ({"params": {"overlay_opacity": -0.1}}, ValueError),
        ({"params": {"solid_color": "red"}}, ValueError),
        ({"params": {"solid_color": "#12345"}}, ValueError),
        ({"boxes": [{"x": 0.5, "y": 0.5, "w": 0, "h": 0.1, "unit": "relative"}]}, ValueError),
        ({"boxes": [{"x": 0.5, "y": 0.5, "w": 0.1, "h": -0.1, "unit": "relative"}]}, ValueError),
        ({"boxes": [{"x": 1.2, "y": 0.5, "w": 0.1, "h": 0.1, "unit": "relative"}]}, ValueError),
        ({"boxes": [{"x": 0.8, "y": 0.5, "w": 0.3, "h": 0.1, "unit": "relative"}]}, ValueError),
        ({"video": {"mode": "turbo"}}, ValueError),
        ({"video": {"encoder": "cuda"}}, ValueError),
        ({"video": {"crf": 60}}, ValueError),
        ({"video": {"crf": -1}}, ValueError),
        ({"video": {"copy_audio": "false"}}, TypeError),
        ({"video": {"copy_audio": 1}}, TypeError),
    ],
)
def test_preset_validation_errors(bad_patch, expected_exc):
    """Test B3: validate_preset must strictly reject invalid parameters."""
    base = get_default_preset()
    for k, v in bad_patch.items():
        if isinstance(v, dict) and k in base:
            base[k].update(v)
        else:
            base[k] = v

    with pytest.raises(expected_exc):
        validate_preset(base)


def test_coordinate_conversions():
    frame_w, frame_h = 1920, 1080

    rel_box = {"x": 0.5, "y": 0.25, "w": 0.1, "h": 0.1, "unit": "relative"}
    abs_box = to_abs_box(rel_box, frame_w, frame_h)
    assert abs_box["unit"] == "absolute"
    assert abs_box["x"] == 960
    assert abs_box["y"] == 270
    assert abs_box["w"] == 192
    assert abs_box["h"] == 108

    rel_back = to_rel_box(abs_box, frame_w, frame_h)
    assert rel_back["unit"] == "relative"
    assert pytest.approx(rel_back["x"], abs=0.01) == 0.5
    assert pytest.approx(rel_back["y"], abs=0.01) == 0.25


def test_invalid_method_raises():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    box = {"x": 10, "y": 10, "w": 20, "h": 20, "unit": "absolute"}
    with pytest.raises(ValueError, match="Invalid method"):
        apply(img, [box], "non_existent_method", {})


def test_all_seven_methods_execute_and_verify_boundaries(tmp_path):
    """Test 1: Verify all 7 methods execute, change pixels inside box,

    and leave pixels strictly outside (margin > dilate + feather + 6) UNCHANGED.
    """
    clean, corrupted, box = create_synthetic_image(400, 300, (150, 100, 70, 45))
    h, w = corrupted.shape[:2]
    lx, ly, lw, lh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])

    # Create dummy PNG for overlay test
    ov_file = str(tmp_path / "logo.png")
    pil_ov = Image.new("RGBA", (lw, lh), (255, 0, 0, 200))
    pil_ov.save(ov_file)

    dilate = 2
    feather = 3
    params = {
        "dilate": dilate,
        "feather": feather,
        "radius": 5,
        "blur_ksize": 25,
        "mosaic_block": 8,
        "solid_color": "#00FF00",
        "overlay_path": ov_file,
        "overlay_opacity": 0.9,
    }

    # For clone_smooth, expansion margin is up to dilate + 8 px
    outer_margin = dilate + feather + 10
    outer_mask = np.ones((h, w), dtype=bool)
    outer_mask[
        max(0, ly - outer_margin) : min(h, ly + lh + outer_margin),
        max(0, lx - outer_margin) : min(w, lx + lw + outer_margin),
    ] = False

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

        # 1. Pixels inside box MUST change
        inner_corrupted = corrupted[ly : ly + lh, lx : lx + lw]
        inner_out = out[ly : ly + lh, lx : lx + lw]
        assert np.any(inner_out != inner_corrupted), f"Method {m} did not modify pixels inside box!"

        # 2. Pixels far outside box MUST NOT change
        assert np.array_equal(
            out[outer_mask], corrupted[outer_mask]
        ), f"Method {m} modified pixels outside the allowed bounding margin!"


def test_overlay_errors_and_unicode_filename(tmp_path):
    """Test B2: Overlay must raise FileNotFoundError on missing files and handle Unicode filenames."""
    clean, corrupted, box = create_synthetic_image(400, 300, (150, 100, 70, 45))

    # 1. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        apply(corrupted, [box], "overlay", {"overlay_path": "non_existent_file.png"})

    # 2. Empty path raises ValueError
    with pytest.raises(ValueError):
        apply(corrupted, [box], "overlay", {"overlay_path": None})

    # 3. Unicode Vietnamese filename succeeds
    unicode_png = str(tmp_path / "logo_ảnh_xoá.png")
    Image.new("RGBA", (30, 30), (0, 255, 0, 255)).save(unicode_png)
    out = apply(corrupted, [box], "overlay", {"overlay_path": unicode_png, "overlay_opacity": 1.0})
    assert isinstance(out, np.ndarray)


def test_inpaint_hq_error_reduction():
    """Gate test: inpaint_hq average error in logo region compared to clean background

    must be strictly less than 50% of the initial error caused by the logo.
    """
    clean, corrupted, box = create_synthetic_image(400, 300, (150, 100, 70, 45))
    lx, ly, lw, lh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])

    params = {"dilate": 2, "feather": 3, "radius": 5}

    erased_fast = apply(corrupted, [box], "inpaint_fast", params)
    erased_hq = apply(corrupted, [box], "inpaint_hq", params)

    roi_clean = clean[ly : ly + lh, lx : lx + lw].astype(np.float64)
    roi_corrupted = corrupted[ly : ly + lh, lx : lx + lw].astype(np.float64)
    roi_fast = erased_fast[ly : ly + lh, lx : lx + lw].astype(np.float64)
    roi_hq = erased_hq[ly : ly + lh, lx : lx + lw].astype(np.float64)

    initial_error = float(np.mean(np.abs(roi_corrupted - roi_clean)))
    fast_error = float(np.mean(np.abs(roi_fast - roi_clean)))
    hq_error = float(np.mean(np.abs(roi_hq - roi_clean)))

    print(
        f"\n[Comparison] Initial: {initial_error:.2f} | Fast: {fast_error:.2f} | HQ: {hq_error:.2f} (Ratio: {hq_error/initial_error:.2%})"
    )

    assert hq_error / initial_error < 0.50, f"inpaint_hq ratio {hq_error/initial_error:.2%} >= 50%"


def test_clone_smooth_error_reduction():
    """Test 2: Quality gate for clone_smooth on patterned/striped background.

    Must achieve error reduction < 50% of initial logo error.
    """
    clean, corrupted, box = create_striped_sample(400, 300, (150, 100, 70, 45))
    lx, ly, lw, lh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])

    params = {"dilate": 2, "feather": 3}
    cloned = apply(corrupted, [box], "clone_smooth", params)

    roi_clean = clean[ly : ly + lh, lx : lx + lw].astype(np.float64)
    roi_corrupted = corrupted[ly : ly + lh, lx : lx + lw].astype(np.float64)
    roi_cloned = cloned[ly : ly + lh, lx : lx + lw].astype(np.float64)

    initial_error = float(np.mean(np.abs(roi_corrupted - roi_clean)))
    cloned_error = float(np.mean(np.abs(roi_cloned - roi_clean)))

    ratio = cloned_error / initial_error
    print(f"\n[clone_smooth Gate] Initial: {initial_error:.2f}, Cloned: {cloned_error:.2f}, Ratio: {ratio:.2%}")

    assert ratio < 0.50, f"clone_smooth error ratio {ratio:.2%} is not < 50%"


def test_detect_logo_boxes():
    """Test 5: detect_logo_boxes must identify logo with IoU >= 0.5 and avoid duplicate boxes."""
    # Less than 8 frames -> returns empty
    frames_few = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    assert detect_logo_boxes(frames_few) == []

    # 12 frames with varying background and a single static logo at (30, 20, 50, 30)
    w, h = 200, 200
    true_box = {"x": 30 / w, "y": 20 / h, "w": 50 / w, "h": 30 / h}
    frames = []
    for i in range(12):
        # Varying background across frames
        f = np.full((h, w, 3), (i * 22) % 255, dtype=np.uint8)
        # Static logo
        cv2.rectangle(f, (30, 20), (80, 50), (255, 255, 255), -1)
        cv2.putText(f, "LOGO", (32, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        frames.append(f)

    detected = detect_logo_boxes(frames, max_boxes=3)
    assert len(detected) >= 1, "detect_logo_boxes failed to detect any box"

    # Verify IoU of the top box is >= 0.5 with ground truth
    top_box = detected[0]
    iou = _compute_iou(top_box, true_box)
    print(f"\n[detect_logo_boxes IoU] Detected: {top_box}, True: {true_box}, IoU: {iou:.3f}")
    assert iou >= 0.50, f"Top box IoU {iou:.3f} is less than 0.50"

    # Verify no duplicate boxes overlapping each other
    if len(detected) > 1:
        for idx1 in range(len(detected)):
            for idx2 in range(idx1 + 1, len(detected)):
                pair_iou = _compute_iou(detected[idx1], detected[idx2])
                assert pair_iou < 0.3, f"Duplicate boxes found with IoU {pair_iou}"
