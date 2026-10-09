"""Pure image processing engine for LogoEraser.

All functions are pure: they accept and return np.ndarray (BGR uint8)
without performing file or network I/O.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from .preset import VALID_METHODS, to_abs_box


def _odd(n: int, minimum: int = 3) -> int:
    """Ensure integer n is odd and at least minimum."""
    v = max(int(n), minimum)
    return v if (v % 2 == 1) else (v + 1)


def parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    """Parse '#RRGGBB' or 'RRGGBB' into BGR uint8 tuple."""
    clean = hex_str.strip().lstrip("#")
    if len(clean) != 6:
        return (0, 0, 0)
    try:
        r = int(clean[0:2], 16)
        g = int(clean[2:4], 16)
        b = int(clean[4:6], 16)
        return (b, g, r)  # OpenCV BGR format
    except ValueError:
        return (0, 0, 0)


def build_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    boxes_abs: list[dict],
    dilate: int = 3,
    feather: int = 4,
) -> np.ndarray:
    """Build a grayscale uint8 mask (0..255) from absolute bounding boxes.

    Applies dilation and soft Gaussian boundary feathering.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for b in boxes_abs:
        bx = max(0, min(w - 1, int(b["x"])))
        by = max(0, min(h - 1, int(b["y"])))
        bw = max(1, min(w - bx, int(b["w"])))
        bh = max(1, min(h - by, int(b["h"])))
        cv2.rectangle(mask, (bx, by), (bx + bw, by + bh), 255, -1)

    if dilate > 0:
        ksize = _odd(dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        mask = cv2.dilate(mask, kernel)

    if feather > 0:
        k_feather = _odd(feather * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k_feather, k_feather), 0)

    return mask


def _inpaint_fast(img: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Fast Telea inpainting."""
    bin_mask = (mask > 10).astype(np.uint8) * 255
    r = max(1, int(radius))
    return cv2.inpaint(img, bin_mask, r, cv2.INPAINT_TELEA)


def _inpaint_hq(
    img: np.ndarray, mask: np.ndarray, boxes_abs: list[dict], radius: int
) -> np.ndarray:
    """Multi-scale dual-algorithm (Telea + Navier-Stokes) high quality inpainting.

    Combines weighted blends of NS and Telea, and performs multi-scale downscaled
    reconstruction for large regions to prevent directional streak artifacts.
    """
    bin_mask = (mask > 10).astype(np.uint8) * 255
    r = max(1, int(radius))

    # Base blend of Telea and Navier-Stokes
    base_telea = cv2.inpaint(img, bin_mask, r, cv2.INPAINT_TELEA)
    base_ns = cv2.inpaint(img, bin_mask, r, cv2.INPAINT_NS)
    blended = cv2.addWeighted(base_telea, 0.5, base_ns, 0.5, 0.0)

    # Check if any box is large enough to benefit from multi-scale downscaled inpaint
    has_large_box = any(
        (b.get("w", 0) >= 30 and b.get("h", 0) >= 30) or max(b.get("w", 0), b.get("h", 0)) >= 45
        for b in boxes_abs
    )

    if not has_large_box:
        return blended

    # Multi-scale pass at half-resolution
    h, w = img.shape[:2]
    small_w = max(16, w // 2)
    small_h = max(16, h // 2)

    small_img = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small_mask = cv2.resize(bin_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    small_mask = cv2.dilate(small_mask, np.ones((3, 3), np.uint8))

    small_inpaint = cv2.inpaint(small_img, small_mask, max(2, r // 2), cv2.INPAINT_TELEA)
    upscaled = cv2.resize(small_inpaint, (w, h), interpolation=cv2.INTER_CUBIC)

    # Use distance transform to blend upscaled core into the base inpaint
    dist = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 5)
    span = float(max(1.0, float(dist.max())))
    core_weight = np.clip(dist / span, 0.0, 1.0).astype(np.float32)
    core_weight = cv2.GaussianBlur(core_weight, (15, 15), 0)
    core_weight = np.expand_dims(core_weight, axis=2)

    out = (blended.astype(np.float32) * (1.0 - core_weight) +
           upscaled.astype(np.float32) * core_weight)
    return np.clip(out, 0, 255).astype(np.uint8)


def _clone_smooth(img: np.ndarray, boxes_abs: list[dict], feather: int) -> np.ndarray:
    """Seamless patch cloning from adjacent background regions.

    Selects the neighbor direction with lowest variance and blends using cv2.seamlessClone.
    """
    out = img.copy()
    h, w = img.shape[:2]

    for b in boxes_abs:
        bx = max(0, min(w - 2, int(b["x"])))
        by = max(0, min(h - 2, int(b["y"])))
        bw = max(2, min(w - bx, int(b["w"])))
        bh = max(2, min(h - by, int(b["h"])))

        # Evaluate candidate neighbor patches: top, bottom, left, right
        candidates: list[tuple[float, int, int]] = []

        # Top neighbor
        if by >= bh:
            patch = out[by - bh:by, bx:bx + bw]
            var = float(np.var(patch))
            candidates.append((var, bx, by - bh))

        # Bottom neighbor
        if by + bh + bh <= h:
            patch = out[by + bh:by + bh + bh, bx:bx + bw]
            var = float(np.var(patch))
            candidates.append((var, bx, by + bh))

        # Left neighbor
        if bx >= bw:
            patch = out[by:by + bh, bx - bw:bx]
            var = float(np.var(patch))
            candidates.append((var, bx - bw, by))

        # Right neighbor
        if bx + bw + bw <= w:
            patch = out[by:by + bh, bx + bw:bx + bw + bw]
            var = float(np.var(patch))
            candidates.append((var, bx + bw, by))

        if not candidates:
            # Fallback to Telea if no candidate fits inside frame
            box_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.rectangle(box_mask, (bx, by), (bx + bw, by + bh), 255, -1)
            out = cv2.inpaint(out, box_mask, 5, cv2.INPAINT_TELEA)
            continue

        # Choose patch with minimum color variance
        candidates.sort(key=lambda item: item[0])
        _, src_x, src_y = candidates[0]

        patch_src = out[src_y:src_y + bh, src_x:src_x + bw].copy()
        patch_mask = np.full((bh, bw), 255, dtype=np.uint8)
        center = (bx + bw // 2, by + bh // 2)

        try:
            out = cv2.seamlessClone(patch_src, out, patch_mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            # Boundary or dimension error fallback
            k_feather = _odd(feather * 2 + 1)
            soft_mask = cv2.GaussianBlur(patch_mask, (k_feather, k_feather), 0)
            alpha = np.expand_dims(soft_mask.astype(np.float32) / 255.0, axis=2)
            roi = out[by:by + bh, bx:bx + bw].astype(np.float32)
            blended = roi * (1.0 - alpha) + patch_src.astype(np.float32) * alpha
            out[by:by + bh, bx:bx + bw] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def _apply_blur(img: np.ndarray, mask: np.ndarray, blur_ksize: int) -> np.ndarray:
    """Gaussian blur blended via feathered mask."""
    k = _odd(blur_ksize)
    blurred = cv2.GaussianBlur(img, (k, k), 0)
    alpha = np.expand_dims(mask.astype(np.float32) / 255.0, axis=2)
    out = img.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_mosaic(
    img: np.ndarray, mask: np.ndarray, boxes_abs: list[dict], block_size: int
) -> np.ndarray:
    """Pixelate/mosaic transformation inside target boxes."""
    out = img.copy()
    h, w = img.shape[:2]
    b_size = max(2, int(block_size))

    mosaic_full = img.copy()
    for b in boxes_abs:
        bx = max(0, min(w - 1, int(b["x"])))
        by = max(0, min(h - 1, int(b["y"])))
        bw = max(1, min(w - bx, int(b["w"])))
        bh = max(1, min(h - by, int(b["h"])))

        roi = mosaic_full[by:by + bh, bx:bx + bw]
        small_w = max(1, bw // b_size)
        small_h = max(1, bh // b_size)
        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
        mosaic_full[by:by + bh, bx:bx + bw] = pix

    alpha = np.expand_dims(mask.astype(np.float32) / 255.0, axis=2)
    res = img.astype(np.float32) * (1.0 - alpha) + mosaic_full.astype(np.float32) * alpha
    return np.clip(res, 0, 255).astype(np.uint8)


def _apply_solid(img: np.ndarray, mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    """Solid color fill blended via feathered mask."""
    solid_layer = np.full_like(img, color_bgr, dtype=np.uint8)
    alpha = np.expand_dims(mask.astype(np.float32) / 255.0, axis=2)
    out = img.astype(np.float32) * (1.0 - alpha) + solid_layer.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_overlay(
    img: np.ndarray, boxes_abs: list[dict], overlay_path: str | None, opacity: float
) -> np.ndarray:
    """Overlay PNG with alpha channel scaled into target boxes."""
    if not overlay_path:
        return img.copy()

    try:
        pil_ov = Image.open(overlay_path).convert("RGBA")
    except Exception:
        return img.copy()

    out = img.copy()
    h, w = img.shape[:2]
    user_opacity = max(0.0, min(1.0, float(opacity)))

    for b in boxes_abs:
        bx = max(0, min(w - 1, int(b["x"])))
        by = max(0, min(h - 1, int(b["y"])))
        bw = max(1, min(w - bx, int(b["w"])))
        bh = max(1, min(h - by, int(b["h"])))

        resized_ov = pil_ov.resize((bw, bh), Image.Resampling.LANCZOS)
        arr_ov = np.array(resized_ov)

        fg_bgr = arr_ov[:, :, :3][:, :, ::-1]  # RGB to BGR
        fg_alpha = (arr_ov[:, :, 3].astype(np.float32) / 255.0) * user_opacity
        fg_alpha = np.expand_dims(fg_alpha, axis=2)

        bg_roi = out[by:by + bh, bx:bx + bw].astype(np.float32)
        blended = fg_bgr.astype(np.float32) * fg_alpha + bg_roi * (1.0 - fg_alpha)
        out[by:by + bh, bx:bx + bw] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


def apply(img: np.ndarray, boxes_abs: list[dict], method: str, params: dict) -> np.ndarray:
    """Apply one of the 7 logo erasure/covering methods to the image.

    Raises:
        ValueError: If method is not one of the 7 supported methods.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method: '{method}'. Supported methods: {sorted(VALID_METHODS)}"
        )

    if not boxes_abs:
        return img.copy()

    h, w = img.shape[:2]
    # Ensure all boxes are absolute
    clean_boxes = [to_abs_box(b, w, h) for b in boxes_abs]

    dilate = int(params.get("dilate", 3))
    feather = int(params.get("feather", 4))
    mask = build_mask(img.shape, clean_boxes, dilate=dilate, feather=feather)

    if method == "inpaint_fast":
        radius = int(params.get("radius", 5))
        return _inpaint_fast(img, mask, radius=radius)

    if method == "inpaint_hq":
        radius = int(params.get("radius", 5))
        return _inpaint_hq(img, mask, clean_boxes, radius=radius)

    if method == "clone_smooth":
        return _clone_smooth(img, clean_boxes, feather=feather)

    if method == "blur":
        blur_ksize = int(params.get("blur_ksize", 31))
        return _apply_blur(img, mask, blur_ksize=blur_ksize)

    if method == "mosaic":
        block_size = int(params.get("mosaic_block", 12))
        return _apply_mosaic(img, mask, clean_boxes, block_size=block_size)

    if method == "solid":
        color_hex = str(params.get("solid_color", "#000000"))
        bgr = parse_hex_color(color_hex)
        return _apply_solid(img, mask, bgr)

    if method == "overlay":
        ov_path = params.get("overlay_path")
        opacity = float(params.get("overlay_opacity", 1.0))
        return _apply_overlay(img, clean_boxes, ov_path, opacity=opacity)

    raise ValueError(f"Unhandled method '{method}'")


def detect_logo_boxes(frames: list[np.ndarray], max_boxes: int = 3) -> list[dict]:
    """Detect static watermark/logo bounding boxes across >= 8 video frames.

    Uses temporal variance combined with Canny edge density to pinpoint
    persistent static graphics or subtitles.
    """
    if len(frames) < 8:
        return []

    h, w = frames[0].shape[:2]
    total_area = float(w * h)

    # Convert frames to grayscale
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    stack = np.stack(grays, axis=0).astype(np.float32)

    # Compute temporal standard deviation
    temporal_std = np.std(stack, axis=0)

    # Static areas have low temporal standard deviation
    static_mask = (temporal_std < 12.0).astype(np.uint8) * 255

    # Edge density from average frame
    avg_gray = np.mean(stack, axis=0).astype(np.uint8)
    edges = cv2.Canny(avg_gray, 50, 150)

    # Candidate regions: static AND containing edge details
    candidates = cv2.bitwise_and(static_mask, edges)

    # Morphological grouping
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9))
    grouped = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, k_close)
    k_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    grouped = cv2.dilate(grouped, k_dilate, iterations=2)

    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[dict] = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)

        # Ignore boxes that are > 25% of the frame (background/borders)
        if area > 0.25 * total_area:
            continue
        # Ignore tiny noise artifacts
        if bw < 12 or bh < 12 or area < 150:
            continue

        boxes.append({
            "x": round(float(bx) / float(w), 4),
            "y": round(float(by) / float(h), 4),
            "w": round(float(bw) / float(w), 4),
            "h": round(float(bh) / float(h), 4),
            "unit": "relative",
            "area": area,
        })

    # Sort candidates by area descending
    boxes.sort(key=lambda b: b["area"], reverse=True)

    result = []
    for b in boxes[:max_boxes]:
        result.append({
            "x": b["x"],
            "y": b["y"],
            "w": b["w"],
            "h": b["h"],
            "unit": "relative",
        })

    return result
