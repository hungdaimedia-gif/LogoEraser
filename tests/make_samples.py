"""Synthetic sample generation for testing LogoEraser."""

from __future__ import annotations

import cv2
import numpy as np


def create_synthetic_image(
    w: int = 400, h: int = 300, logo_box: tuple[int, int, int, int] = (150, 100, 70, 45)
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a synthetic test image with a known logo.

    Returns:
        clean_img: Original image without logo (BGR uint8).
        corrupted_img: Image with logo inserted in logo_box (BGR uint8).
        box_dict: Bounding box dictionary with unit='absolute'.
    """
    # Vectorized gradient generation using np.meshgrid
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)

    b = np.clip(120 + 80 * np.sin(xx / 40.0), 0, 255).astype(np.uint8)
    g = np.clip(100 + 70 * np.cos(yy / 35.0), 0, 255).astype(np.uint8)
    r = np.clip(140 + 60 * np.sin((xx + yy) / 50.0), 0, 255).astype(np.uint8)

    clean = np.dstack((b, g, r))

    # Add geometric background elements
    cv2.circle(clean, (120, 100), 50, (180, 120, 90), -1)
    cv2.rectangle(clean, (20, 180), (160, 260), (90, 160, 130), -1)

    # Corrupt with a prominent yellow/red logo in the specified box
    corrupted = clean.copy()
    lx, ly, lw, lh = logo_box
    cv2.rectangle(corrupted, (lx + 2, ly + 2), (lx + lw - 2, ly + lh - 2), (0, 255, 255), -1)
    cv2.putText(
        corrupted,
        "LOGO",
        (lx + 6, ly + lh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    box_dict = {
        "x": lx,
        "y": ly,
        "w": lw,
        "h": lh,
        "unit": "absolute",
    }

    return clean, corrupted, box_dict


def create_striped_sample(
    w: int = 400, h: int = 300, logo_box: tuple[int, int, int, int] = (150, 100, 70, 45)
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create a repetitive striped test image ideal for testing clone_smooth.

    Cloning from adjacent horizontal stripes preserves the periodic pattern.
    """
    clean = np.zeros((h, w, 3), dtype=np.uint8)
    # Repeating horizontal stripes of period 16 px
    for y in range(h):
        val = 180 if ((y // 16) % 2 == 0) else 60
        clean[y, :] = (val, 120, 220 - val)

    corrupted = clean.copy()
    lx, ly, lw, lh = logo_box
    cv2.rectangle(corrupted, (lx + 2, ly + 2), (lx + lw - 2, ly + lh - 2), (0, 255, 255), -1)
    cv2.putText(
        corrupted,
        "LOGO",
        (lx + 6, ly + lh - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )

    box_dict = {
        "x": lx,
        "y": ly,
        "w": lw,
        "h": lh,
        "unit": "absolute",
    }

    return clean, corrupted, box_dict


def create_synthetic_video(
    output_path: str,
    width: int = 320,
    height: int = 240,
    fps: int = 25,
    duration: float = 3.0,
    with_audio: bool = True,
    logo_box: tuple[int, int, int, int] | None = None,
    ffmpeg: str = "ffmpeg",
) -> dict:
    """Generate a synthetic test MP4 video using FFmpeg lavfi filters.

    Returns a bounding box dict with unit='absolute'.
    """
    import subprocess
    import sys

    create_no_window = 0x08000000 if sys.platform == "win32" else 0

    vf = "null"
    box_dict = {"x": 20, "y": 20, "w": 60, "h": 40, "unit": "absolute"}
    if logo_box is not None:
        lx, ly, lw, lh = logo_box
        vf = f"drawbox=x={lx}:y={ly}:w={lw}:h={lh}:color=yellow:t=fill"
        box_dict = {"x": lx, "y": ly, "w": lw, "h": lh, "unit": "absolute"}

    cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={fps}:duration={duration}",
    ]

    if with_audio:
        cmd.extend([
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration}",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ])
    else:
        cmd.extend([
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-an",
            "-pix_fmt",
            "yuv420p",
            output_path,
        ])

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=create_no_window,
    )
    return box_dict
