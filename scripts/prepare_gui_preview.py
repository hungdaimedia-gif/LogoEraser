"""Helper script to stage LogoEraser GUI for Phase 2 screenshot capture and acceptance verification."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import tkinter as tk
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cli import read_image_safe
from core.image_engine import apply
from gui.app import LogoEraserApp
from gui.theme import apply_dark_theme


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "preview"
    wait_sec = int(sys.argv[2]) if len(sys.argv) > 2 else 8

    sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
    os.makedirs(sample_dir, exist_ok=True)

    img_path = os.path.join(sample_dir, "demo_image.png")
    w, h = 640, 480
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            img[y, x] = [int((x / w) * 180), int((y / h) * 180), 100]

    # Logo 1 (Top Left)
    img[40:100, 50:180] = [0, 255, 255]  # Yellow
    cv2.putText(img, "LOGO 1", (60, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Logo 2 (Bottom Right)
    img[360:440, 450:600] = [255, 0, 255]  # Magenta
    cv2.putText(img, "WATERMARK", (460, 410), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(img_path, img)

    root = tk.Tk()
    apply_dark_theme(root)
    app = LogoEraserApp(root)
    root.geometry("1280x820+150+100")

    # Load file
    app.state.add_files([img_path])
    app._refresh_file_listbox()
    app._load_file_to_canvas(img_path)

    # Add 2 boxes
    app.state.add_box_abs(50, 40, 130, 60, w, h)
    app.state.add_box_abs(450, 360, 150, 80, w, h)
    app.state.selected_box_idx = 1
    app.state.set_method("inpaint_hq")
    app.method_var.set("inpaint_hq")
    app._update_param_visibility()
    app.canvas_widget.redraw()
    app._on_boxes_changed()

    if mode == "preview":
        app.canvas_widget._compute_and_show_preview()
        app.canvas_widget.show_preview = True
        app.btn_preview.config(text="👁 Ẩn xem trước")
        app.status_text_var.set("Đang xem trước kết quả xoá logo với 2 vùng chọn")

        # Save preset to verify Phase 2 gate
        preset_path = os.path.join(sample_dir, "phase2_preset.json")
        app.state.save_preset_file(preset_path)

        # Save GUI preview image
        gui_out_path = os.path.join(sample_dir, "gui_preview_out.png")
        abs_boxes = app.state.get_abs_boxes(w, h)
        gui_preview_arr = apply(img, abs_boxes, app.state.method, app.state.params)
        cv2.imwrite(gui_out_path, gui_preview_arr)

        # Run cli.py with the exact saved preset
        cli_out_path = os.path.join(sample_dir, "cli_processed_out.png")
        cmd = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cli.py"),
            "--preset",
            preset_path,
            "--input",
            img_path,
            "--output",
            cli_out_path,
            "--overwrite",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        assert res.returncode == 0, f"CLI error: {res.stderr}"

        # Compare hashes
        h_gui = hashlib.sha256(Path(gui_out_path).read_bytes()).hexdigest()
        h_cli = hashlib.sha256(Path(cli_out_path).read_bytes()).hexdigest()
        max_diff = int(np.max(np.abs(gui_preview_arr.astype(np.int32) - read_image_safe(cli_out_path).astype(np.int32))))

        print("=== PHASE 2 VERIFICATION OUTPUT ===")
        print(f"GUI Output SHA-256: {h_gui}")
        print(f"CLI Output SHA-256: {h_cli}")
        print(f"Hashes Match:       {h_gui == h_cli}")
        print(f"Max Pixel Diff:     {max_diff}")
        print("===================================")

    elif mode == "video_progress":
        vid_path = os.path.join(sample_dir, "demo_video.mp4")
        app.state.add_files([vid_path])
        app._refresh_file_listbox()
        app.btn_run.config(state=tk.DISABLED)
        app.btn_cancel.config(state=tk.NORMAL)
        app.progress_bar["value"] = 52.4
        app.progress_lbl_var.set("Đang xử lý tệp (1/1): 52.4% - demo_video.mp4")
        app.status_text_var.set("Đang xử lý video bằng thuật toán quality (OpenCV)...")

    # Bring to front for screenshot
    root.lift()
    root.attributes("-topmost", True)
    root.after(1000, lambda: root.attributes("-topmost", False))
    root.update()

    start = time.time()
    while time.time() - start < wait_sec:
        root.update()
        time.sleep(0.05)

    root.destroy()


if __name__ == "__main__":
    main()
