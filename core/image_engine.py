"""Pure image processing engine for LogoEraser.

All functions are pure: they accept and return np.ndarray (BGR uint8)
without performing file or network I/O (except reading overlay PNG via standard file system).
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from PIL import Image

from .preset import VALID_METHODS, parse_hex_color, to_abs_box


def _odd(n: int, minimum: int = 3) -> int:
    """Ensure integer n is odd and at least minimum."""
    v = max(int(n), minimum)
    return v if (v % 2 == 1) else (v + 1)


def build_mask(
    shape: tuple[int, int] | tuple[int, int, int],
    boxes_abs: list[dict],
    dilate: int = 3,
    feather: int = 4,
) -> np.ndarray:
    """Build a grayscale uint8 mask (0..255) from absolute bounding boxes.

    Uses exact numpy slicing to prevent off-by-one pixel overflow.
    Applies dilation and soft Gaussian boundary feathering.
    """
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for b in boxes_abs:
        bx = max(0, min(w - 1, int(b["x"])))
        by = max(0, min(h - 1, int(b["y"])))
        bw = max(1, min(w - bx, int(b["w"])))
        bh = max(1, min(h - by, int(b["h"])))
        # Exact numpy slicing: avoids OpenCV cv2.rectangle inclusive coordinate overflow
        mask[by:by + bh, bx:bx + bw] = 255

    if dilate > 0:
        ksize = _odd(dilate * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        mask = cv2.dilate(mask, kernel)

    if feather > 0:
        k_feather = _odd(feather * 2 + 1)
        mask = cv2.GaussianBlur(mask, (k_feather, k_feather), 0)

    return mask


def _inpaint_fast(img: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Fast Telea inpainting using cv2.INPAINT_TELEA."""
    bin_mask = (mask > 10).astype(np.uint8) * 255
    r = max(1, int(radius))
    return cv2.inpaint(img, bin_mask, r, cv2.INPAINT_TELEA)


def _inpaint_hq(
    img: np.ndarray, mask: np.ndarray, boxes_abs: list[dict], radius: int
) -> np.ndarray:
    """Dual-algorithm inpainting combining Telea and Navier-Stokes.

    Runs cv2.INPAINT_TELEA and cv2.INPAINT_NS, blending them with weights
    (0.45 Telea / 0.55 NS). For large boxes (>= 900 px or dimension >= 40 px),
    executes an additional half-resolution pass to provide smooth low-frequency fill.
    """
    bin_mask = (mask > 10).astype(np.uint8) * 255
    r = max(1, int(radius))

    # Base dual-inpaint blend: NS preserves smooth gradients; Telea preserves edge sharpness
    base_telea = cv2.inpaint(img, bin_mask, r, cv2.INPAINT_TELEA)
    base_ns = cv2.inpaint(img, bin_mask, r, cv2.INPAINT_NS)
    blended = cv2.addWeighted(base_telea, 0.45, base_ns, 0.55, 0.0)

    # For regions with area >= 900 px or dimension >= 40px, use multi-scale low-frequency fill
    has_large_box = any(
        (b.get("w", 0) * b.get("h", 0) >= 900) or max(b.get("w", 0), b.get("h", 0)) >= 40
        for b in boxes_abs
    )

    if not has_large_box:
        return blended

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


def _boxes_intersect(b1: dict, b2: dict) -> bool:
    """Check if two bounding boxes intersect."""
    x1, y1, w1, h1 = b1["x"], b1["y"], b1["w"], b1["h"]
    x2, y2, w2, h2 = b2["x"], b2["y"], b2["w"], b2["h"]
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or y1 + h1 <= y2 or y2 + h2 <= y1)


def _clone_smooth(
    img: np.ndarray, boxes_abs: list[dict], dilate: int = 3, feather: int = 4
) -> np.ndarray:
    """Seamless patch cloning from adjacent clean background regions.

    Addresses Review B1 & B4:
    Pre-inpaints target logo regions to ensure boundary conditions and reflective
    padding (copyMakeBorder) remain free of logo artifacts even for corner/border logos.
    Candidates for patch cloning are sourced strictly from uncorrupted regions within
    the original image frame.
    """
    if not boxes_abs:
        return img.copy()

    h, w = img.shape[:2]
    margin = max(6, int(dilate) + 4)

    # Determine maximum dimension to ensure sufficient padding for candidate patches
    max_dim = max([max(int(b["w"]), int(b["h"])) for b in boxes_abs], default=30)
    pad = margin + max_dim + 4

    # Pre-inpaint logo boxes so that boundary conditions at image borders and
    # reflective padding do not mirror corrupt logo pixels.
    pre_mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes_abs:
        bx = max(0, min(w - 1, int(b["x"])))
        by = max(0, min(h - 1, int(b["y"])))
        bw = max(1, min(w - bx, int(b["w"])))
        bh = max(1, min(h - by, int(b["h"])))
        pre_mask[by : by + bh, bx : bx + bw] = 255

    if dilate > 0:
        k_d = _odd(dilate * 2 + 1)
        pre_mask = cv2.dilate(pre_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (k_d, k_d)))

    base_img = cv2.inpaint(img, pre_mask, 5, cv2.INPAINT_TELEA)

    # Pad base image with reflection so that border/corner logos have clean surrounding margin
    padded = cv2.copyMakeBorder(base_img, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)

    for i, b in enumerate(boxes_abs):
        bx = int(b["x"])
        by = int(b["y"])
        bw = int(b["w"])
        bh = int(b["h"])

        # Coordinates inside padded image
        pbx = bx + pad
        pby = by + pad

        # Expanded box bounds inside padded image
        ebx = pbx - margin
        eby = pby - margin
        ebw = bw + 2 * margin
        ebh = bh + 2 * margin

        other_boxes = [ob for j, ob in enumerate(boxes_abs) if j != i]

        # Candidates MUST be located strictly inside the original unpadded image area:
        # Original area inside padded is: X in [pad, pad + w], Y in [pad, pad + h]
        candidates: list[tuple[float, int, int]] = []

        # 1. Top candidate: (ebx, eby - ebh)
        if eby - ebh >= pad:
            cand_orig = {"x": bx - margin, "y": by - margin - ebh, "w": ebw, "h": ebh}
            if not any(_boxes_intersect(cand_orig, ob) for ob in other_boxes):
                patch = padded[eby - ebh : eby, ebx : ebx + ebw]
                var = float(np.var(patch))
                candidates.append((var, ebx, eby - ebh))

        # 2. Bottom candidate: (ebx, eby + ebh)
        if eby + 2 * ebh <= pad + h:
            cand_orig = {"x": bx - margin, "y": by - margin + ebh, "w": ebw, "h": ebh}
            if not any(_boxes_intersect(cand_orig, ob) for ob in other_boxes):
                patch = padded[eby + ebh : eby + 2 * ebh, ebx : ebx + ebw]
                var = float(np.var(patch))
                candidates.append((var, ebx, eby + ebh))

        # 3. Left candidate: (ebx - ebw, eby)
        if ebx - ebw >= pad:
            cand_orig = {"x": bx - margin - ebw, "y": by - margin, "w": ebw, "h": ebh}
            if not any(_boxes_intersect(cand_orig, ob) for ob in other_boxes):
                patch = padded[eby : eby + ebh, ebx - ebw : ebx]
                var = float(np.var(patch))
                candidates.append((var, ebx - ebw, eby))

        # 4. Right candidate: (ebx + ebw, eby)
        if ebx + 2 * ebw <= pad + w:
            cand_orig = {"x": bx - margin + ebw, "y": by - margin, "w": ebw, "h": ebh}
            if not any(_boxes_intersect(cand_orig, ob) for ob in other_boxes):
                patch = padded[eby : eby + ebh, ebx + ebw : ebx + 2 * ebw]
                var = float(np.var(patch))
                candidates.append((var, ebx + ebw, eby))

        if not candidates:
            # Already pre-inpainted in base_img, so padded already has clean inpaint fallback
            continue

        # Choose patch with lowest variance
        candidates.sort(key=lambda item: item[0])
        _, src_x, src_y = candidates[0]

        patch_src = padded[src_y : src_y + ebh, src_x : src_x + ebw].copy()

        # Build mask for seamlessClone inside expanded box
        # Mask shape is (ebh, ebw). Outer boundary is 0, inner box is 255
        clone_mask = np.zeros((ebh, ebw), dtype=np.uint8)
        inner_x = margin
        inner_y = margin
        clone_mask[inner_y : inner_y + bh, inner_x : inner_x + bw] = 255

        if dilate > 0:
            ksize = _odd(dilate * 2 + 1)
            clone_mask = cv2.dilate(
                clone_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            )

        center = (ebx + ebw // 2, eby + ebh // 2)

        try:
            padded = cv2.seamlessClone(patch_src, padded, clone_mask, center, cv2.NORMAL_CLONE)
        except Exception:
            # Fallback to soft alpha blend with feathered mask
            k_f = _odd(feather * 2 + 1)
            soft = cv2.GaussianBlur(clone_mask, (k_f, k_f), 0).astype(np.float32) / 255.0
            soft = np.expand_dims(soft, axis=2)
            roi = padded[eby : eby + ebh, ebx : ebx + ebw].astype(np.float32)
            if roi.shape == patch_src.shape:
                blended = roi * (1.0 - soft) + patch_src.astype(np.float32) * soft
                padded[eby : eby + ebh, ebx : ebx + ebw] = np.clip(blended, 0, 255).astype(np.uint8)

    # Crop padded image back to original dimensions
    out = padded[pad : pad + h, pad : pad + w].copy()
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
    """Overlay PNG with alpha channel scaled into target boxes.

    Raises:
        ValueError: If overlay_path is None/empty or opacity is out of range [0, 1].
        FileNotFoundError: If overlay_path does not exist on disk.
    """
    if not overlay_path:
        raise ValueError("overlay_path must be specified for 'overlay' method.")

    if not os.path.isfile(overlay_path):
        raise FileNotFoundError(f"Overlay image file not found: '{overlay_path}'")

    user_opacity = float(opacity)
    if user_opacity < 0.0 or user_opacity > 1.0:
        raise ValueError(f"overlay_opacity must be in range [0.0, 1.0], got {user_opacity}")

    pil_ov = Image.open(overlay_path).convert("RGBA")

    out = img.copy()
    h, w = img.shape[:2]

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
        ValueError: If method is not one of the 7 supported methods or params are invalid.
        FileNotFoundError: If overlay image file is not found.
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
    return apply_with_mask(img, mask, clean_boxes, method, params)


def apply_with_mask(
    img: np.ndarray,
    mask: np.ndarray,
    clean_boxes: list[dict],
    method: str,
    params: dict,
) -> np.ndarray:
    """Apply erasure method using a precomputed binary mask.

    Used by the quality video processing pipeline to avoid recomputing masks per frame.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid method '{method}'. Supported methods: {sorted(list(VALID_METHODS))}"
        )

    if not clean_boxes:
        return img.copy()

    if method == "inpaint_fast":
        radius = int(params.get("radius", 5))
        return _inpaint_fast(img, mask, radius=radius)

    if method == "inpaint_hq":
        radius = int(params.get("radius", 5))
        return _inpaint_hq(img, mask, clean_boxes, radius=radius)

    if method == "clone_smooth":
        dilate = int(params.get("dilate", 3))
        feather = int(params.get("feather", 4))
        return _clone_smooth(img, clean_boxes, dilate=dilate, feather=feather)

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


def _compute_iou(b1: dict, b2: dict) -> float:
    """Compute Intersection over Union (IoU) of two boxes."""
    x1 = max(b1["x"], b2["x"])
    y1 = max(b1["y"], b2["y"])
    x2 = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
    y2 = min(b1["y"] + b1["h"], b2["y"] + b2["h"])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = b1["w"] * b1["h"]
    area2 = b2["w"] * b2["h"]
    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def detect_logo_boxes(frames: list[np.ndarray], max_boxes: int = 3) -> list[dict]:
    """Detect static watermark/logo bounding boxes across >= 8 video frames.

    Uses temporal variance combined with Canny edge density to pinpoint
    persistent static graphics or subtitles.
    Includes Non-Maximum Suppression (NMS) to eliminate duplicate/nested bounding boxes.
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

    # Static areas have low temporal standard deviation (< 10.0)
    static_mask = (temporal_std < 10.0).astype(np.uint8) * 255

    # Edge density from average frame
    avg_gray = np.mean(stack, axis=0).astype(np.uint8)
    edges = cv2.Canny(avg_gray, 50, 150)

    # Candidate regions: static AND containing edge details
    candidates = cv2.bitwise_and(static_mask, edges)

    # Tight morphological grouping (5x5 closing, no excessive dilation)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    grouped = cv2.morphologyEx(candidates, cv2.MORPH_CLOSE, k_close)

    contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes: list[dict] = []
    for cnt in contours:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)

        # Ignore boxes that are > 25% of the frame (background/borders)
        if area > 0.25 * total_area:
            continue
        # Ignore tiny noise artifacts
        if bw < 8 or bh < 8 or area < 80:
            continue

        raw_boxes.append({
            "x": round(float(bx) / float(w), 4),
            "y": round(float(by) / float(h), 4),
            "w": round(float(bw) / float(w), 4),
            "h": round(float(bh) / float(h), 4),
            "unit": "relative",
            "area": area,
        })

    # Sort raw boxes by area descending
    raw_boxes.sort(key=lambda b: b["area"], reverse=True)

    # Non-Maximum Suppression (NMS) to merge overlapping or nested boxes
    filtered_boxes: list[dict] = []
    for b in raw_boxes:
        overlap = False
        for kept in filtered_boxes:
            iou = _compute_iou(b, kept)
            if iou > 0.2:
                overlap = True
                break
            # Check containment
            inter_x = max(b["x"], kept["x"])
            inter_y = max(b["y"], kept["y"])
            inter_w = max(0.0, min(b["x"] + b["w"], kept["x"] + kept["w"]) - inter_x)
            inter_h = max(0.0, min(b["y"] + b["h"], kept["y"] + kept["h"]) - inter_y)
            if (inter_w * inter_h) / (b["w"] * b["h"]) > 0.6:
                overlap = True
                break

        if not overlap:
            filtered_boxes.append(b)

    result = []
    for b in filtered_boxes[:max_boxes]:
        result.append({
            "x": b["x"],
            "y": b["y"],
            "w": b["w"],
            "h": b["h"],
            "unit": "relative",
        })

    return result
