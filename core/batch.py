"""Shared batch helpers used by both cli.py and the GUI worker.

Một đường duy nhất cho: nhận diện loại file, đặt tên file xuất, đọc/ghi ảnh
đường dẫn tiếng Việt, và xử lý một file (ảnh hoặc video) qua file tạm `.part`.
Trước đây GUI tự viết lại vòng lặp của CLI nên hai bên lệch nhau (REVIEW-2 N19).
"""

from __future__ import annotations

import os
import threading
from typing import Callable

import cv2
import numpy as np

from core.fsutil import remove_part_file, replace_retry
from core.image_engine import apply
from core.preset import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, to_abs_box
from core.video_engine import process_video

CLEAN_SUFFIX = "_clean"


def media_kind(path: str) -> str | None:
    """Return 'image', 'video' or None based on file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def is_clean_output(path: str) -> bool:
    """True if the file looks like a previous LogoEraser output (*_clean.*)."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.endswith(CLEAN_SUFFIX)


def clean_name(src: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(src))
    return f"{stem}{CLEAN_SUFFIX}{ext}"


def output_path_for(src: str, output_dir: str | None) -> str:
    """Output file path: `<output_dir or folder of src>/<stem>_clean<ext>`."""
    folder = output_dir if output_dir else os.path.dirname(os.path.abspath(src))
    return os.path.join(folder, clean_name(src))


def part_path_for(dst: str) -> str:
    stem, ext = os.path.splitext(dst)
    return f"{stem}.part{ext}"


def read_image_safe(path: str) -> np.ndarray:
    """Read image supporting Unicode and Vietnamese filesystem paths."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: '{path}'")
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty image file: '{path}'")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image from '{path}'")
    return img


def write_image_safe(path: str, img: np.ndarray) -> None:
    """Write image supporting Unicode and Vietnamese filesystem paths."""
    ext = os.path.splitext(path)[1].lower() or ".png"
    success, encoded = cv2.imencode(ext, img)
    if not success:
        raise ValueError(f"Failed to encode image to format '{ext}'")
    encoded.tofile(path)


def process_image_file(src: str, dst: str, preset: dict) -> None:
    """Process a single image file via .part temporary file (Review 1b N12)."""
    img = read_image_safe(src)
    h, w = img.shape[:2]
    abs_boxes = [to_abs_box(b, w, h) for b in preset["boxes"]]
    out = apply(img, abs_boxes, preset["method"], preset["params"])
    out_dir = os.path.dirname(dst)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    part_dst = part_path_for(dst)
    try:
        write_image_safe(part_dst, out)
        replace_retry(part_dst, dst)
    finally:
        # Chỉ dọn file .part do chính job này vừa tạo, không đụng dst cũ.
        remove_part_file(part_dst)


def process_file(
    src: str,
    dst: str,
    preset: dict,
    *,
    overwrite: bool = False,
    progress_cb: Callable[[float, float], None] | None = None,
    cancel: threading.Event | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> None:
    """Process one image or video. Raises on any failure; never leaves partial output.

    Raises:
        FileExistsError: dst exists and overwrite is False.
        ValueError: unsupported file type.
    """
    kind = media_kind(src)
    if kind is None:
        raise ValueError(f"Unsupported file format: '{os.path.splitext(src)[1]}'")
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(f"Destination file already exists (use overwrite): '{dst}'")

    if kind == "image":
        process_image_file(src, dst, preset)
        if progress_cb:
            progress_cb(1.0, 1.0)
        return

    try:
        process_video(
            src,
            dst,
            preset,
            progress_cb=progress_cb,
            cancel=cancel,
            ffmpeg=ffmpeg,
            ffprobe=ffprobe,
        )
    finally:
        remove_part_file(part_path_for(dst))
