"""File helpers that tolerate short-lived Windows file locks.

Ngay sau khi ffmpeg đóng file, Windows Defender / trình tạo thumbnail / ổ đồng bộ
đám mây có thể giữ file vài trăm ms ⇒ os.replace báo WinError 32 và job thất bại
dù video đã mã hoá xong (gặp thật trên ổ H: của người dùng, 2026-10-09).
"""

from __future__ import annotations

import os
import time

_LOCK_ERRORS = (PermissionError,)


def replace_retry(src: str, dst: str, timeout: float = 15.0) -> None:
    """os.replace, retrying while the file is locked by another process."""
    end = time.monotonic() + timeout
    delay = 0.1
    while True:
        try:
            os.replace(src, dst)
            return
        except _LOCK_ERRORS:
            if time.monotonic() >= end:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 1.0)


def remove_part_file(path: str, timeout: float = 10.0) -> None:
    """Dọn file tạm `.part` do chính job tạo ra; chịu được khoá file ngắn hạn.

    Chỉ được gọi với đường dẫn `.part` — không bao giờ với file kết quả của người dùng.
    """
    if ".part" not in os.path.basename(path):
        raise ValueError(f"refusing to remove non-.part file: '{path}'")
    end = time.monotonic() + timeout
    while os.path.exists(path):
        try:
            os.remove(path)
            return
        except FileNotFoundError:
            return
        except _LOCK_ERRORS:
            if time.monotonic() >= end:
                return  # để lại file tạm còn hơn làm hỏng job; lần chạy sau ghi đè
            time.sleep(0.25)
