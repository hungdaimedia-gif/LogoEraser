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


def main() -> int:
    """Launch LogoEraser desktop application."""
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
