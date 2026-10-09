"""Unit tests for GuiState in LogoEraser GUI.

Verifies pure-Python state management without requiring a display or tkinter window.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest

from core.preset import get_default_preset, save_preset
from gui.state import GuiState


def test_gui_state_initial_defaults():
    state = GuiState()
    assert state.method == "inpaint_hq"
    assert state.video_mode == "fast"
    assert state.video_encoder == "auto"
    assert state.video_crf == 20
    assert state.video_copy_audio is True
    assert len(state.boxes) == 0
    assert len(state.files) == 0
    assert state.selected_box_idx is None
    assert state.selected_file_idx is None


def test_gui_state_file_management(tmp_path):
    f1 = tmp_path / "img1.png"
    f2 = tmp_path / "img2.jpg"
    f3 = tmp_path / "vid1.mp4"
    f4 = tmp_path / "ignored.txt"
    for f in (f1, f2, f3, f4):
        f.write_text("data", encoding="utf-8")

    state = GuiState()
    added = state.add_files([str(f1), str(f2), str(f3), str(f4)])
    assert added == 3
    assert len(state.files) == 3
    assert state.selected_file_idx == 0
    assert state.get_selected_file() == str(f1.resolve())

    # Selecting file
    state.select_file(1)
    assert state.get_selected_file() == str(f2.resolve())

    # Removing file: does NOT delete on disk
    removed = state.remove_file(1)
    assert removed == str(f2.resolve())
    assert f2.exists()  # Ensure file on disk is untouched
    assert len(state.files) == 2

    # Clear files
    state.clear_files()
    assert len(state.files) == 0
    assert state.selected_file_idx is None


def test_gui_state_box_management():
    state = GuiState()
    img_w, img_h = 1000, 500

    idx0 = state.add_box_abs(100, 50, 200, 100, img_w, img_h)
    assert idx0 == 0
    assert len(state.boxes) == 1
    assert state.selected_box_idx == 0

    rel_box = state.boxes[0]
    assert rel_box["unit"] == "relative"
    assert rel_box["x"] == pytest.approx(0.1)
    assert rel_box["y"] == pytest.approx(0.1)
    assert rel_box["w"] == pytest.approx(0.2)
    assert rel_box["h"] == pytest.approx(0.2)

    # Conversion back to absolute
    abs_boxes = state.get_abs_boxes(img_w, img_h)
    assert abs_boxes[0]["x"] == 100
    assert abs_boxes[0]["y"] == 50
    assert abs_boxes[0]["w"] == 200
    assert abs_boxes[0]["h"] == 100

    # Updating box
    state.update_box_abs(0, 150, 75, 250, 150, img_w, img_h)
    abs_boxes = state.get_abs_boxes(img_w, img_h)
    assert abs_boxes[0]["x"] == 150
    assert abs_boxes[0]["y"] == 75

    # Adding second box and removing
    idx1 = state.add_box_abs(400, 200, 100, 50, img_w, img_h)
    assert idx1 == 1
    assert len(state.boxes) == 2

    state.remove_box(0)
    assert len(state.boxes) == 1
    abs_boxes = state.get_abs_boxes(img_w, img_h)
    assert abs_boxes[0]["x"] == 400

    state.clear_boxes()
    assert len(state.boxes) == 0
    assert state.selected_box_idx is None


def test_gui_state_preset_export_import_roundtrip(tmp_path):
    state = GuiState()
    state.set_method("blur")
    state.set_param("blur_ksize", 33)
    state.set_video_mode("quality")
    state.video_encoder = "libx264"
    state.video_crf = 18
    state.video_copy_audio = False
    state.add_box_abs(10, 10, 50, 50, 100, 100)

    p_dict = state.to_preset_dict()
    assert p_dict["method"] == "blur"
    assert p_dict["params"]["blur_ksize"] == 33
    assert p_dict["video"]["mode"] == "quality"
    assert p_dict["video"]["encoder"] == "libx264"
    assert p_dict["video"]["crf"] == 18
    assert p_dict["video"]["copy_audio"] is False

    # Save to file and reload in fresh state
    p_file = str(tmp_path / "exported_preset.json")
    state.save_preset_file(p_file)

    new_state = GuiState()
    new_state.load_preset_file(p_file)
    assert new_state.method == "blur"
    assert new_state.params["blur_ksize"] == 33
    assert new_state.video_mode == "quality"
    assert len(new_state.boxes) == 1


def test_gui_state_active_param_names():
    state = GuiState()
    state.set_method("inpaint_fast")
    assert state.get_active_param_names() == ["dilate", "feather", "radius"]

    state.set_method("clone_smooth")
    assert state.get_active_param_names() == ["dilate", "feather"]

    state.set_method("blur")
    assert state.get_active_param_names() == ["dilate", "feather", "blur_ksize"]

    state.set_method("mosaic")
    assert state.get_active_param_names() == ["dilate", "feather", "mosaic_block"]

    state.set_method("solid")
    assert state.get_active_param_names() == ["dilate", "feather", "solid_color"]

    # dilate/feather định hình mask của blur/mosaic/solid nên phải hiện trên GUI

    state.set_method("overlay")
    assert state.get_active_param_names() == ["overlay_path", "overlay_opacity"]


def test_gui_state_validation_for_run(tmp_path):
    state = GuiState()

    # 1. Empty files & boxes
    errs = state.validate_for_run()
    assert any("Danh sách tệp tin đang trống" in e for e in errs)
    assert any("Chưa chọn vùng logo" in e for e in errs)

    # 2. Add an image file and box
    img_file = tmp_path / "sample.png"
    img_file.write_text("dummy", encoding="utf-8")
    state.add_files([str(img_file)])
    state.add_box_abs(10, 10, 40, 40, 100, 100)

    # Now valid for image
    errs = state.validate_for_run()
    assert len(errs) == 0

    # 3. Add video file + clone_smooth + fast mode (Forbidden check)
    vid_file = tmp_path / "sample.mp4"
    vid_file.write_text("dummy", encoding="utf-8")
    state.add_files([str(vid_file)])
    state.set_method("clone_smooth")
    state.set_video_mode("fast")

    errs = state.validate_for_run()
    assert any("clone_smooth" in e and "fast" in e for e in errs)

    # Switch to quality mode -> valid
    state.set_video_mode("quality")
    errs = state.validate_for_run()
    assert len(errs) == 0

    # 4. Overlay missing file check
    state.set_method("overlay")
    state.set_param("overlay_path", str(tmp_path / "non_existent.png"))
    errs = state.validate_for_run()
    assert any("Không tìm thấy tệp ảnh phủ" in e for e in errs)


def test_set_param_rejects_unknown_key():
    state = GuiState()
    with pytest.raises(KeyError):
        state.set_param("inpaint_radius", 5)
    state.set_param("radius", 7)
    assert state.to_preset_dict()["params"]["radius"] == 7
