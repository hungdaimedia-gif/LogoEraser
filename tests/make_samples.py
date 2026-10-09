"""Synthetic sample generation for testing LogoEraser."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def create_synthetic_image(
    w: int = 400, h: int = 300, logo_box: tuple[int, int, int, int] = (250, 180, 80, 40)
) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """Create a synthetic test image with a known logo.

    Returns:
        clean_img: Original image without logo (BGR uint8).
        corrupted_img: Image with logo inserted in logo_box (BGR uint8).
        box_dict: Bounding box dictionary.
        overlay_png_path: Path or creates a temporary overlay PNG.
    """
    # 1. Generate smooth color gradient background with diagonal texture
    clean = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            b = int(120 + 80 * np.sin(x / 40.0))
            g = int(100 + 70 * np.cos(y / 35.0))
            r = int(140 + 60 * np.sin((x + y) / 50.0))
            clean[y, x] = (np.clip(b, 0, 255), np.clip(g, 0, 255), np.clip(r, 0, 255))

    # Add subtle geometric background elements
    cv2.circle(clean, (120, 100), 50, (180, 120, 90), -1)
    cv2.rectangle(clean, (20, 180), (160, 260), (90, 160, 130), -1)

    # 2. Corrupt with a prominent solid/textured logo in the specified box
    corrupted = clean.copy()
    lx, ly, lw, lh = logo_box
    # Draw a distinct bright logo (e.g. white/yellow text-like graphic)
    cv2.rectangle(corrupted, (lx + 4, ly + 4), (lx + lw - 4, ly + lh - 4), (0, 255, 255), -1)
    cv2.putText(
        corrupted,
        "LOGO",
        (lx + 8, ly + lh - 12),
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
