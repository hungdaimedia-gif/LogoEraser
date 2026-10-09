"""End-to-end tests that drive the REAL Tkinter app (REVIEW-2 C2).

Dựng `LogoEraserApp` thật, giả lập chuột bằng `event_generate`, chạy job qua nút
Bắt đầu / Huỷ thật — không gọi thẳng image_engine. Tự skip khi máy không có
màn hình (Tk không khởi tạo được).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

tk = pytest.importorskip("tkinter")

from core.preset import save_preset  # noqa: E402
from tests.make_samples import create_synthetic_image, create_synthetic_video  # noqa: E402

pytestmark = pytest.mark.gui

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def app(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as e:  # không có display
        pytest.skip(f"Tk unavailable: {e}")

    import gui.app as app_mod
    from gui.theme import apply_dark_theme

    # Hộp thoại chặn luồng Tk — ghi lại thay vì hiện ra.
    shown: list[tuple[str, str]] = []
    for name in ("showinfo", "showwarning", "showerror"):
        monkeypatch.setattr(app_mod.messagebox, name, lambda t, m, _n=name: shown.append((_n, m)))
    monkeypatch.setattr(app_mod.messagebox, "askyesno", lambda *a, **k: True)

    apply_dark_theme(root)
    a = app_mod.LogoEraserApp(root)
    a.shown_dialogs = shown
    root.update()
    yield a
    try:
        root.destroy()
    except tk.TclError:
        pass


def pump(app, cond, timeout: float = 30.0) -> bool:
    """Run the Tk event loop until cond() is true or timeout."""
    end = time.time() + timeout
    while time.time() < end:
        app.root.update()
        if cond():
            return True
        time.sleep(0.02)
    return False


def drag(app, ix0, iy0, ix1, iy1) -> None:
    cw = app.canvas_widget
    x0, y0 = cw.img_to_canvas(ix0, iy0)
    x1, y1 = cw.img_to_canvas(ix1, iy1)
    c = cw.canvas
    c.event_generate("<ButtonPress-1>", x=int(x0), y=int(y0))
    for t in (0.33, 0.66, 1.0):
        c.event_generate("<B1-Motion>", x=int(x0 + (x1 - x0) * t), y=int(y0 + (y1 - y0) * t))
    c.event_generate("<ButtonRelease-1>", x=int(x1), y=int(y1))
    app.root.update()


def sha(p: str) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_gui_image_job_matches_cli(app, tmp_path):
    _, corrupted, _ = create_synthetic_image(640, 400, (470, 330, 140, 50))
    src = tmp_path / "mẫu ảnh.png"
    cv2.imencode(".png", corrupted)[1].tofile(str(src))

    app.state.add_files([str(src)])
    app._refresh_file_listbox()
    app._load_file_to_canvas(str(src))
    assert pump(app, lambda: app.canvas_widget.original_image is not None, 5)

    drag(app, 465, 325, 615, 385)
    drag(app, 20, 20, 120, 70)
    assert len(app.state.boxes) == 2

    app._on_toggle_preview()
    assert pump(app, lambda: app.canvas_widget.preview_image is not None, 5)

    preset_path = tmp_path / "gui_preset.json"
    save_preset(app.state.to_preset_dict(), str(preset_path))

    gui_out = tmp_path / "gui_out"
    app.out_dir_var.set(str(gui_out))
    app._on_start_processing()
    assert pump(app, lambda: not app.is_processing, 30)
    gui_file = gui_out / "mẫu ảnh_clean.png"
    assert gui_file.is_file()

    cli_out = tmp_path / "cli_out"
    r = subprocess.run(
        [sys.executable, "cli.py", "--preset", str(preset_path), "--input", str(src), "--output", str(cli_out) + os.sep],
        cwd=PROJECT_ROOT,
        capture_output=True,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert sha(str(gui_file)) == sha(str(cli_out / "mẫu ảnh_clean.png"))

    # Logo trong box #1 phải thật sự bị xoá
    out = cv2.imdecode(np.fromfile(str(gui_file), np.uint8), cv2.IMREAD_COLOR)
    roi_before = corrupted[330:380, 470:610].astype(float)
    roi_after = out[330:380, 470:610].astype(float)
    assert np.abs(roi_after - roi_before).mean() > 20


def test_gui_video_cancel_mid_run(app, tmp_path):
    src = tmp_path / "dài.mp4"
    create_synthetic_video(str(src), width=640, height=400, duration=30.0, logo_box=(470, 330, 140, 50))

    app.state.add_files([str(src)])
    app._refresh_file_listbox()
    app._load_file_to_canvas(str(src))
    # Khung đầu đọc ở luồng phụ: giao diện vẫn phải phản hồi trong lúc chờ
    assert pump(app, lambda: app.canvas_widget.original_image is not None, 15)

    drag(app, 465, 325, 615, 385)
    app.video_mode_var.set("quality")
    app._on_video_mode_changed()
    out_dir = tmp_path / "xuat"
    app.out_dir_var.set(str(out_dir))
    app._on_start_processing()

    assert pump(app, lambda: app.progress_bar["value"] > 3, 30), "progress never moved"
    t0 = time.time()
    app._on_cancel_processing()
    assert pump(app, lambda: not app.is_processing, 15), "cancel did not stop the worker"
    assert time.time() - t0 < 15
    assert not app.worker_thread.is_alive()
    assert not (out_dir / "dài_clean.mp4").exists()
    assert not list(out_dir.glob("*.part*")) if out_dir.exists() else True
    assert any("Đã dừng" in m for _, m in app.shown_dialogs)


def test_close_window_while_running_waits_for_worker(app, tmp_path):
    src = tmp_path / "close.mp4"
    create_synthetic_video(str(src), width=640, height=400, duration=30.0, logo_box=(470, 330, 140, 50))
    app.state.add_files([str(src)])
    app._load_file_to_canvas(str(src))
    assert pump(app, lambda: app.canvas_widget.original_image is not None, 15)
    drag(app, 465, 325, 615, 385)
    app.video_mode_var.set("quality")
    app._on_video_mode_changed()
    app.out_dir_var.set(str(tmp_path / "xuat"))
    app._on_start_processing()
    assert pump(app, lambda: app.progress_bar["value"] > 3, 30)

    worker = app.worker_thread
    app._on_close_window()

    def destroyed() -> bool:
        try:
            app.root.winfo_exists()
            return False
        except tk.TclError:
            return True

    end = time.time() + 20
    while time.time() < end and not destroyed():
        try:
            app.root.update()
        except tk.TclError:
            break
        time.sleep(0.02)
    assert destroyed(), "window did not close"
    assert not worker.is_alive(), "window closed before worker stopped (orphan ffmpeg risk)"


def test_method_order_is_stable(app):
    from core.preset import SUPPORTED_METHODS

    assert tuple(app.method_combo["values"]) == SUPPORTED_METHODS


def test_param_controls_reach_preset_without_focus_out(app):
    # Bấm mũi tên Spinbox chỉ đổi biến, không có FocusOut — giá trị vẫn phải vào preset.
    app.var_radius.set(9)
    app.var_ksize.set(15)
    app.video_crf_var.set(28)
    preset = app.state.to_preset_dict()
    assert preset["params"]["radius"] == 9
    assert preset["params"]["blur_ksize"] == 15
    assert preset["video"]["crf"] == 28
