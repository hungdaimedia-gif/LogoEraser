"""Command Line Interface (CLI) for LogoEraser.

Processes images and videos using LogoEraser presets.
Supports batch directory processing, recursive traversal, Unicode paths, and external config.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Force UTF-8 encoding on Windows to support Unicode and Vietnamese logging
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from core.config import load_config
from core.image_engine import apply
from core.preset import load_preset, to_abs_box, validate_preset
from core.video_engine import process_video

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


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
    ext = os.path.splitext(path)[1].lower()
    if not ext:
        ext = ".png"
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

    stem, ext = os.path.splitext(dst)
    part_dst = f"{stem}.part{ext}"
    try:
        write_image_safe(part_dst, out)
        os.replace(part_dst, dst)
    finally:
        if os.path.exists(part_dst):
            try:
                os.remove(part_dst)
            except OSError:
                pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="LogoEraser - Open-source watermark and logo removal tool",
    )
    parser.add_argument("--preset", required=True, help="Path to JSON preset configuration file")
    parser.add_argument("--input", required=True, help="Input file or directory to process")
    parser.add_argument("--output", required=True, help="Output destination file or directory")
    parser.add_argument(
        "--mode",
        choices=["fast", "quality"],
        help="Override video processing mode (fast or quality)",
    )
    parser.add_argument("--method", help="Override logo removal method")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recursively process files in input directory",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as e:
        return 2 if e.code != 0 else 0

    # Review 1b N13: Load config.json for custom ffmpeg/ffprobe paths
    cfg = load_config()
    ffmpeg_bin = cfg.get("ffmpeg", "ffmpeg")
    ffprobe_bin = cfg.get("ffprobe", "ffprobe")

    # 1. Load and validate preset
    try:
        preset_data = load_preset(args.preset)
        if args.mode:
            if "video" not in preset_data or not isinstance(preset_data["video"], dict):
                preset_data["video"] = {}
            preset_data["video"]["mode"] = args.mode
        if args.method:
            preset_data["method"] = args.method
        valid_preset = validate_preset(preset_data)
    except Exception as e:
        print(f"Error loading/validating preset: {e}", file=sys.stderr)
        return 2

    # 2. Check input
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        print(f"Error: Input path does not exist: '{args.input}'", file=sys.stderr)
        return 1

    # 3. Discover files
    tasks: list[tuple[str, str, str]] = []  # (src, dst, file_type: 'image' | 'video')
    output_path = os.path.abspath(args.output)

    if os.path.isfile(input_path):
        ext = os.path.splitext(input_path)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            ftype = "image"
        elif ext in VIDEO_EXTENSIONS:
            ftype = "video"
        else:
            print(f"Error: Unsupported input file format '{ext}'", file=sys.stderr)
            return 1

        # Determine dst
        if os.path.isdir(output_path) or args.output.endswith(("/", "\\")):
            stem, fext = os.path.splitext(os.path.basename(input_path))
            dst = os.path.join(output_path, f"{stem}_clean{fext}")
        else:
            dst = output_path
        tasks.append((input_path, dst, ftype))

    elif os.path.isdir(input_path):
        os.makedirs(output_path, exist_ok=True)
        for root, dirs, files in os.walk(input_path):
            if not args.recursive and os.path.abspath(root) != input_path:
                continue
            for f in files:
                f_ext = os.path.splitext(f)[1].lower()
                if f_ext in IMAGE_EXTENSIONS:
                    ftype = "image"
                elif f_ext in VIDEO_EXTENSIONS:
                    ftype = "video"
                else:
                    continue

                stem, ext_curr = os.path.splitext(f)
                # Review 1b N15: Avoid processing previously created clean files (*_clean.*)
                if stem.endswith("_clean"):
                    continue

                src_file = os.path.join(root, f)
                rel_dir = os.path.relpath(root, input_path)
                out_name = f"{stem}_clean{ext_curr}"
                if rel_dir == ".":
                    dst_file = os.path.join(output_path, out_name)
                else:
                    dst_file = os.path.join(output_path, rel_dir, out_name)
                tasks.append((src_file, dst_file, ftype))

    if not tasks:
        print(f"Warning: No supported image or video files found in '{args.input}'", file=sys.stderr)
        return 0

    # 4. Process files
    failed_files: list[tuple[str, str]] = []
    success_count = 0

    for src, dst, ftype in tasks:
        if os.path.exists(dst) and not args.overwrite:
            failed_files.append((src, f"Destination file already exists (use --overwrite): '{dst}'"))
            continue

        try:
            if ftype == "image":
                process_image_file(src, dst, valid_preset)
            else:
                process_video(src, dst, valid_preset, ffmpeg=ffmpeg_bin, ffprobe=ffprobe_bin)
            success_count += 1
            print(f"[OK] {src} -> {dst}")
        except Exception as e:
            # Review 1b N12: Do not delete existing dst on failure; clean up partial .part file only
            stem, ext = os.path.splitext(dst)
            part_file = f"{stem}.part{ext}"
            if os.path.exists(part_file):
                try:
                    os.remove(part_file)
                except OSError:
                    pass
            failed_files.append((src, str(e)))
            print(f"[FAIL] {src}: {e}", file=sys.stderr)

    if failed_files:
        print("\nSummary of failures:", file=sys.stderr)
        for f_path, reason in failed_files:
            print(f" - {f_path}: {reason}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
