"""LogoEraser - Open-source watermark and logo removal application.

Main entry point for launching the desktop GUI.
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk

from gui.app import LogoEraserApp
from gui.theme import apply_dark_theme


def enable_dpi_awareness() -> None:
    """Khai báo DPI-aware trước khi tạo cửa sổ Tk.

    Không khai báo thì Windows phóng to cửa sổ kiểu ảnh bitmap ở mức scale
    150–200% (màn 4K) ⇒ chữ nhoè (REVIEW-2 C3). Tk tự đổi cỡ chữ theo DPI thật;
    kích thước tính bằng pixel được nhân hệ số trong LogoEraserApp.
    """
    if sys.platform != "win32":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # PROCESS_SYSTEM_DPI_AWARE
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def ai_check(report_path: str) -> int:
    """`LogoEraser.exe --ai-check <file>`: ghi ra file AI đang chạy GPU hay CPU.

    Bản exe không có console nên kết quả ghi ra file — dùng để kiểm bản đóng gói
    có nạp được thư viện CUDA hay đã lùi về CPU.
    """
    import time

    import numpy as np

    from core.ai_inpaint import active_provider, inpaint_lama, model_path

    lines = [f"model: {model_path()}"]
    try:
        img = np.full((300, 400, 3), 128, np.uint8)
        mask = np.zeros((300, 400), np.uint8)
        mask[100:160, 150:250] = 255
        box = [{"x": 150, "y": 100, "w": 100, "h": 60}]
        inpaint_lama(img, mask, box)  # khởi động
        t0 = time.perf_counter()
        inpaint_lama(img, mask, box)
        lines += [f"provider: {active_provider()}", f"ms_per_region: {(time.perf_counter() - t0) * 1000:.0f}"]
        code = 0
    except Exception as e:
        lines.append(f"error: {e}")
        code = 1
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return code


def main() -> int:
    """Launch LogoEraser desktop application."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--ai-check":
        return ai_check(sys.argv[2])
    enable_dpi_awareness()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    root = tk.Tk()
    apply_dark_theme(root)
    app = LogoEraserApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
