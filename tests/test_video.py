"""Integration and unit tests for VideoEngine and CLI in LogoEraser.

Covers Phase 1b gates:
- Video probing and encoder detection.
- Fast mode with all boundary conditions and clone_smooth rejection.
- Quality mode with >= 300 frames to prove no stderr pipe deadlock.
- 3-second sample video gate with audio preservation.
- Cancellation cleanup and error containment.
- CLI execution, exit codes (0, 1, 2), and Unicode Vietnamese directory handling.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from core.preset import get_default_preset, save_preset
from core.video_engine import detect_encoder, probe, process_video
from tests.make_samples import create_synthetic_video

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def test_probe_utility(tmp_path):
    """Test probe extracts accurate video metadata and audio detection."""
    vid_path = str(tmp_path / "probe_test.mp4")
    create_synthetic_video(vid_path, width=320, height=240, fps=25, duration=1.0, with_audio=True)

    info = probe(vid_path)
    assert info.width == 320
    assert info.height == 240
    assert abs(info.fps - 25.0) < 0.1
    assert abs(info.duration - 1.0) < 0.2
    assert abs(info.nb_frames - 25) <= 1
    assert info.has_audio is True


def test_detect_encoder_reports_valid_choice():
    """Test detect_encoder returns an available H.264 encoder."""
    enc = detect_encoder()
    print(f"\n[detect_encoder] Selected hardware/software encoder: {enc}")
    assert enc in ("h264_nvenc", "h264_qsv", "h264_amf", "libx264")


def test_fast_mode_clone_smooth_fails(tmp_path):
    """Review 1b Note 1: clone_smooth in fast mode must raise ValueError asking for quality mode."""
    src = str(tmp_path / "fast_fail_src.mp4")
    dst = str(tmp_path / "fast_fail_dst.mp4")
    box = create_synthetic_video(src, width=320, height=240, fps=25, duration=0.5, with_audio=False)

    preset = get_default_preset()
    preset["method"] = "clone_smooth"
    preset["video"]["mode"] = "fast"
    preset["boxes"] = [box]

    with pytest.raises(ValueError, match="clone_smooth is not supported in fast mode"):
        process_video(src, dst, preset)

    assert not os.path.exists(dst)


@pytest.mark.parametrize(
    "pos_name,box_t",
    [
        ("center", (100, 80, 50, 30)),
        ("right_edge", (270, 80, 50, 30)),
        ("bottom_right_corner", (270, 210, 50, 30)),
        ("top_edge", (100, 0, 50, 30)),
    ],
)
def test_fast_mode_delogo_boundaries(tmp_path, pos_name, box_t):
    """Review 1b Note 2: delogo in video_engine must clamp boxes >= 1px away from borders without error."""
    src = str(tmp_path / f"delogo_{pos_name}_src.mp4")
    dst = str(tmp_path / f"delogo_{pos_name}_dst.mp4")
    box = create_synthetic_video(src, width=320, height=240, fps=25, duration=0.5, logo_box=box_t, with_audio=False)

    preset = get_default_preset()
    preset["method"] = "inpaint_hq"
    preset["video"]["mode"] = "fast"
    preset["boxes"] = [box]

    process_video(src, dst, preset)
    assert os.path.exists(dst)

    info = probe(dst)
    assert info.width == 320
    assert info.height == 240
    assert info.nb_frames > 0


def test_quality_mode_long_video_no_deadlock(tmp_path):
    """Review 1b Note 3: Quality mode must process >= 300 frames with separate stderr threads without deadlocking."""
    src = str(tmp_path / "long_video_src.mp4")
    dst = str(tmp_path / "long_video_dst.mp4")
    # 320 frames at 25 fps = 12.8 seconds
    total_frames = 320
    fps = 25
    box = create_synthetic_video(
        src,
        width=160,
        height=120,
        fps=fps,
        duration=total_frames / fps,
        with_audio=False,
        logo_box=(40, 30, 40, 20),
    )

    preset = get_default_preset()
    preset["method"] = "inpaint_fast"
    preset["video"]["mode"] = "quality"
    preset["boxes"] = [box]

    progress_records = []

    def on_progress(cur, total):
        progress_records.append(cur)

    process_video(src, dst, preset, progress_cb=on_progress)
    assert os.path.exists(dst)

    info = probe(dst)
    print(f"\n[Long Video Quality Test] Input frames: {total_frames}, Output probed frames: {info.nb_frames}")
    assert abs(info.nb_frames - total_frames) <= 2
    assert len(progress_records) > 0


def test_sample_video_3s_with_audio_gate(tmp_path):
    """PLAN §8 Gate Test: 3-second sample video with audio in fast and quality modes.

    Both must produce valid video files with identical w, h, fps, frame count ±1, and audio intact.
    """
    src = str(tmp_path / "sample_3s_src.mp4")
    dst_fast = str(tmp_path / "sample_3s_fast.mp4")
    dst_qual = str(tmp_path / "sample_3s_qual.mp4")

    box = create_synthetic_video(
        src,
        width=320,
        height=240,
        fps=25,
        duration=3.0,
        with_audio=True,
        logo_box=(120, 90, 60, 40),
    )

    # 1. Fast mode
    preset_fast = get_default_preset()
    preset_fast["method"] = "inpaint_hq"
    preset_fast["video"]["mode"] = "fast"
    preset_fast["video"]["copy_audio"] = True
    preset_fast["boxes"] = [box]

    process_video(src, dst_fast, preset_fast)
    assert os.path.exists(dst_fast)
    info_fast = probe(dst_fast)

    # 2. Quality mode
    preset_qual = get_default_preset()
    preset_qual["method"] = "clone_smooth"
    preset_qual["video"]["mode"] = "quality"
    preset_qual["video"]["copy_audio"] = True
    preset_qual["boxes"] = [box]

    process_video(src, dst_qual, preset_qual)
    assert os.path.exists(dst_qual)
    info_qual = probe(dst_qual)

    detected_enc = detect_encoder()

    print("\n" + "=" * 60)
    print("PHASE 1B GATE RESULTS:")
    print(f"Detected Encoder: {detected_enc}")
    print(f"Fast Mode Output: {info_fast.width}x{info_fast.height} @ {info_fast.fps:.2f}fps, "
          f"{info_fast.nb_frames} frames, audio={info_fast.has_audio}")
    print(f"Quality Mode Output: {info_qual.width}x{info_qual.height} @ {info_qual.fps:.2f}fps, "
          f"{info_qual.nb_frames} frames, audio={info_qual.has_audio}")
    print("=" * 60)

    # Verify matching dimensions, fps, audio
    assert info_fast.width == 320 and info_fast.height == 240
    assert abs(info_fast.fps - 25.0) < 0.2
    assert abs(info_fast.nb_frames - 75) <= 1
    assert info_fast.has_audio is True

    assert info_qual.width == 320 and info_qual.height == 240
    assert abs(info_qual.fps - 25.0) < 0.2
    assert abs(info_qual.nb_frames - 75) <= 1
    assert info_qual.has_audio is True


def test_cancel_video_processing(tmp_path):
    """Test cancellation cleans up incomplete output file."""
    src = str(tmp_path / "cancel_src.mp4")
    dst = str(tmp_path / "cancel_dst.mp4")
    box = create_synthetic_video(src, width=320, height=240, fps=25, duration=2.0, with_audio=False)

    preset = get_default_preset()
    preset["boxes"] = [box]

    cancel_evt = threading.Event()
    cancel_evt.set()  # Cancel immediately

    with pytest.raises(RuntimeError, match="cancelled"):
        process_video(src, dst, preset, cancel=cancel_evt)

    assert not os.path.exists(dst)


def test_apply_error_during_quality_processing(tmp_path):
    """Review 1b Note 4: Error in apply() halts job, cleans output, and raises exception."""
    src = str(tmp_path / "apply_err_src.mp4")
    dst = str(tmp_path / "apply_err_dst.mp4")
    box = create_synthetic_video(src, width=320, height=240, fps=25, duration=1.0, with_audio=False)

    preset = get_default_preset()
    preset["method"] = "overlay"
    preset["params"]["overlay_path"] = str(tmp_path / "non_existent_file.png")
    preset["video"]["mode"] = "quality"
    preset["boxes"] = [box]

    with pytest.raises(FileNotFoundError):
        process_video(src, dst, preset)

    assert not os.path.exists(dst)


def test_cli_execution_and_unicode_paths(tmp_path):
    """Review 1b Note 5 & PLAN §6: CLI end-to-end test including Vietnamese path 'mẫu thử/' and exit codes."""
    # 1. Create a valid preset JSON
    p_path = str(tmp_path / "test_preset.json")
    preset = get_default_preset()
    preset["method"] = "solid"
    preset["params"]["solid_color"] = "#00FF00"
    preset["boxes"] = [{"x": 10, "y": 10, "w": 40, "h": 20, "unit": "absolute"}]
    save_preset(preset, p_path)

    # 2. Test Vietnamese directory: 'mẫu thử/'
    vietnamese_dir = tmp_path / "mẫu thử"
    vietnamese_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "kết quả"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_img = np.full((100, 100, 3), 150, dtype=np.uint8)
    img_in_path = str(vietnamese_dir / "ảnh_gốc.png")
    success, enc = cv2.imencode(".png", test_img)
    assert success
    enc.tofile(img_in_path)

    # Run CLI on directory: should exit 0
    cmd_cli_dir = [
        sys.executable,
        "cli.py",
        "--preset",
        p_path,
        "--input",
        str(vietnamese_dir),
        "--output",
        str(out_dir),
    ]
    res_dir = subprocess.run(
        cmd_cli_dir,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=CREATE_NO_WINDOW,
    )
    assert res_dir.returncode == 0, f"CLI directory failed: {res_dir.stderr}"

    expected_out = out_dir / "ảnh_gốc_clean.png"
    assert expected_out.exists()

    # 3. Test video input via CLI: should exit 0
    vid_in = str(tmp_path / "cli_vid.mp4")
    vid_out = str(tmp_path / "cli_vid_out.mp4")
    create_synthetic_video(vid_in, width=160, height=120, fps=25, duration=0.5, with_audio=False)

    cmd_cli_vid = [
        sys.executable,
        "cli.py",
        "--preset",
        p_path,
        "--input",
        vid_in,
        "--output",
        vid_out,
        "--mode",
        "fast",
    ]
    res_vid = subprocess.run(
        cmd_cli_vid,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=CREATE_NO_WINDOW,
    )
    assert res_vid.returncode == 0, f"CLI video failed: {res_vid.stderr}"
    assert os.path.exists(vid_out)

    # 4. Test error handling (missing input): should exit 1
    cmd_cli_err = [
        sys.executable,
        "cli.py",
        "--preset",
        p_path,
        "--input",
        str(tmp_path / "does_not_exist.mp4"),
        "--output",
        str(tmp_path / "out.mp4"),
    ]
    res_err = subprocess.run(
        cmd_cli_err,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=CREATE_NO_WINDOW,
    )
    assert res_err.returncode == 1

    # 5. Test invalid preset: should exit 2
    cmd_cli_preset_err = [
        sys.executable,
        "cli.py",
        "--preset",
        str(tmp_path / "invalid_preset.json"),
        "--input",
        vid_in,
        "--output",
        vid_out,
    ]
    res_p_err = subprocess.run(
        cmd_cli_preset_err,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=CREATE_NO_WINDOW,
    )
    assert res_p_err.returncode == 2
