"""Lock-tolerant rename/cleanup (WinError 32 seen on the user's H: drive)."""

from __future__ import annotations

import sys
import threading
import time

import pytest

from core.fsutil import remove_part_file, replace_retry


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locking semantics")
def test_replace_retry_waits_for_transient_lock(tmp_path):
    src = tmp_path / "out.part.mp4"
    dst = tmp_path / "out.mp4"
    src.write_bytes(b"video")

    handle = open(src, "rb")  # giữ file như Defender/thumbnailer — chặn rename trên Windows

    def release():
        time.sleep(0.6)
        handle.close()

    threading.Thread(target=release).start()
    replace_retry(str(src), str(dst), timeout=5)
    assert dst.read_bytes() == b"video"
    assert not src.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="Windows file locking semantics")
def test_replace_retry_gives_up_after_timeout(tmp_path):
    src = tmp_path / "a.part.png"
    src.write_bytes(b"x")
    with open(src, "rb"):
        with pytest.raises(PermissionError):
            replace_retry(str(src), str(tmp_path / "a.png"), timeout=0.5)


def test_remove_part_file_refuses_non_part_files(tmp_path):
    real = tmp_path / "ket_qua.mp4"
    real.write_bytes(b"keep")
    with pytest.raises(ValueError):
        remove_part_file(str(real))
    assert real.exists()
