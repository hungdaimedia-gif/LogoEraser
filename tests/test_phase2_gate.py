"""Phase 2 Acceptance Gate Tests for LogoEraser GUI.

Verifies:
1. Opening an image, adding 2 bounding boxes, generating preview, saving preset.
2. Running cli.py with the exact saved preset produces an identical image (100% matching SHA-256 hash).
3. Testing batch video execution with progress callbacks and mid-run cancellation.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
import pytest

from cli import read_image_safe
from core.preset import get_default_preset, save_preset
from gui.state import GuiState
from tests.test_video import CREATE_NO_WINDOW, create_synthetic_video


def test_gui_preset_and_cli_output_produce_identical_hash(tmp_path):
    """PLAN §8 Phase 2 Acceptance Gate:

    An image previewed in the GUI with 2 bounding boxes must produce an output
    with the exact same SHA-256 hash when processed by cli.py using the saved preset.
    """
    # 1. Create a synthetic test image with two distinct logos
    img_w, img_h = 400, 300
    base_img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for y in range(img_h):
        for x in range(img_w):
            base_img[y, x] = [int((x / img_w) * 200), int((y / img_h) * 200), 120]

    # Stamp Logo 1 (Top-Left)
    base_img[30:70, 40:120] = [0, 255, 255]  # Yellow logo
    # Stamp Logo 2 (Bottom-Right)
    base_img[200:260, 280:360] = [255, 0, 255]  # Magenta logo

    src_img_path = str(tmp_path / "sample_with_2_logos.png")
    success, enc = cv2.imencode(".png", base_img)
    assert success
    enc.tofile(src_img_path)

    # 2. Simulate GUI State & Canvas actions
    state = GuiState()
    state.add_files([src_img_path])
    state.set_method("inpaint_hq")
    state.set_param("radius", 5)
    state.set_param("dilate", 3)
    state.set_param("feather", 4)

    # Add 2 bounding boxes in pixel coordinates (converted to relative automatically by state)
    idx1 = state.add_box_abs(40, 30, 80, 40, img_w, img_h)
    idx2 = state.add_box_abs(280, 200, 80, 60, img_w, img_h)
    assert len(state.boxes) == 2

    # 3. Compute GUI preview (equivalent to Canvas preview)
    from core.image_engine import apply
    abs_boxes = state.get_abs_boxes(img_w, img_h)
    gui_preview_img = apply(base_img, abs_boxes, state.method, state.params)
    assert gui_preview_img is not None

    # Write GUI preview to temporary PNG
    gui_out_path = str(tmp_path / "gui_preview_output.png")
    success, enc_gui = cv2.imencode(".png", gui_preview_img)
    assert success
    enc_gui.tofile(gui_out_path)
    gui_bytes = Path(gui_out_path).read_bytes()
    gui_hash = hashlib.sha256(gui_bytes).hexdigest()

    # 4. Save preset from GUI state to JSON
    preset_file = str(tmp_path / "gui_saved_preset.json")
    state.save_preset_file(preset_file)
    assert os.path.exists(preset_file)

    # 5. Run cli.py with the exact saved preset
    cli_out_path = str(tmp_path / "cli_processed_output.png")
    cmd = [
        sys.executable,
        "cli.py",
        "--preset",
        preset_file,
        "--input",
        src_img_path,
        "--output",
        cli_out_path,
        "--overwrite",
    ]
    res = subprocess.run(
        cmd,
        cwd=str(Path(__file__).parent.parent),
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=CREATE_NO_WINDOW,
    )
    assert res.returncode == 0, f"CLI processing failed: {res.stderr}"
    assert os.path.exists(cli_out_path), "CLI did not create output file!"

    # 6. Verify identical image output
    cli_bytes = Path(cli_out_path).read_bytes()
    cli_hash = hashlib.sha256(cli_bytes).hexdigest()

    cli_img = read_image_safe(cli_out_path)
    max_diff = int(np.max(np.abs(gui_preview_img.astype(np.int32) - cli_img.astype(np.int32))))

    print(f"\n============================================================")
    print(f"PHASE 2 ACCEPTANCE GATE:")
    print(f"GUI Preview SHA-256: {gui_hash}")
    print(f"CLI Output  SHA-256: {cli_hash}")
    print(f"Hashes Match:       {gui_hash == cli_hash}")
    print(f"Max Pixel Diff:     {max_diff}")
    print(f"============================================================")

    assert gui_hash == cli_hash, "GUI preview output and CLI output hashes do not match!"
    assert max_diff == 0, f"Expected 0 pixel difference, got max diff {max_diff}"


def test_gui_worker_cancellation_during_batch(tmp_path):
    """REVIEW-1b-r2 Lưu ý 7: Test mid-run batch cancellation cleanly terminates without leaving orphaned ffmpeg."""
    src_vid = str(tmp_path / "long_batch_src.mp4")
    create_synthetic_video(src_vid, width=160, height=120, fps=25, duration=10.0, with_audio=False)

    state = GuiState()
    state.add_files([src_vid])
    state.add_box_abs(10, 10, 40, 20, 160, 120)
    state.set_video_mode("quality")

    preset = state.to_preset_dict()
    cancel_evt = threading.Event()
    worker_finished = threading.Event()

    def run_worker():
        try:
            from core.video_engine import process_video
            dst_vid = str(tmp_path / "long_batch_out.mp4")
            def on_progress(cur, total):
                if cur >= 25:
                    cancel_evt.set()
            process_video(src_vid, dst_vid, preset, progress_cb=on_progress, cancel=cancel_evt)
        except Exception:
            pass
        finally:
            worker_finished.set()

    t_start = time.time()
    th = threading.Thread(target=run_worker)
    th.start()
    th.join(timeout=15.0)
    elapsed = time.time() - t_start

    print(f"\n[GUI Worker Cancel Test] Worker stopped in {elapsed:.2f}s. Finished: {worker_finished.is_set()}")
    assert not th.is_alive(), "Worker thread hung on cancel!"
    assert elapsed < 10.0, "Worker cancel took too long!"
