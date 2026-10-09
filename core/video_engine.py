"""Video processing engine for LogoEraser.

Supports both fast mode (FFmpeg single-pass filtergraph) and quality mode
(piped frame processing via image_engine algorithms).
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Callable

import cv2
import numpy as np

from core.image_engine import apply_with_mask, build_mask
from core.preset import to_abs_box, validate_preset

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    nb_frames: int
    duration: float
    has_audio: bool
    r_frame_rate: str = "25/1"


def _parse_fps(fps_str: str) -> float:
    """Parse frame rate string such as '25/1' or '29.97' into float."""
    if not fps_str or fps_str == "0/0":
        return 25.0
    if "/" in fps_str:
        num, den = fps_str.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d != 0.0 else 25.0
        except ValueError:
            return 25.0
    try:
        return float(fps_str)
    except ValueError:
        return 25.0


def probe(path: str, ffprobe: str = "ffprobe") -> VideoInfo:
    """Probe video file using ffprobe to obtain stream dimensions, frame rate, and audio status.

    Raises:
        FileNotFoundError: If path does not exist on disk.
        ValueError: If file contains no video streams or cannot be parsed.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Video file not found: '{path}'")

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        path,
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            creationflags=CREATE_NO_WINDOW,
        )
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"ffprobe failed to probe '{path}': {err_msg}") from e

    try:
        data = json.loads(res.stdout.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse ffprobe JSON output for '{path}'") from e

    streams = data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not v_stream:
        raise ValueError(f"No video stream found in '{path}'")

    width = int(v_stream.get("width", 0))
    height = int(v_stream.get("height", 0))

    r_frame_rate = v_stream.get("r_frame_rate") or v_stream.get("avg_frame_rate") or "25/1"
    if not r_frame_rate or r_frame_rate == "0/0":
        r_frame_rate = "25/1"
    fps = _parse_fps(r_frame_rate)

    # Duration from stream or container format
    dur_str = v_stream.get("duration") or data.get("format", {}).get("duration", "0")
    try:
        duration = float(dur_str)
    except ValueError:
        duration = 0.0

    # Number of frames from stream header or estimated from duration * fps
    nb_str = v_stream.get("nb_frames")
    nb_frames = 0
    if nb_str:
        try:
            nb_frames = int(nb_str)
        except ValueError:
            nb_frames = 0

    if nb_frames <= 0 and duration > 0.0 and fps > 0.0:
        nb_frames = int(round(duration * fps))

    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        nb_frames=nb_frames,
        duration=duration,
        has_audio=has_audio,
        r_frame_rate=r_frame_rate,
    )


@functools.lru_cache(maxsize=1)
def detect_encoder(ffmpeg: str = "ffmpeg") -> str:
    """Detect the best available H.264 encoder on the current hardware/OS.

    Tests candidate encoders in order:
    h264_nvenc -> h264_qsv -> h264_amf -> libx264.
    Returns the first encoder that successfully executes a test encode.
    """
    candidates = ["h264_nvenc", "h264_qsv", "h264_amf", "libx264"]
    for enc in candidates:
        cmd = [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=black:s=256x256:d=0.2",
            "-c:v",
            enc,
            "-f",
            "null",
            "-",
        ]
        try:
            res = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=CREATE_NO_WINDOW,
            )
            if res.returncode == 0:
                logger.info("Detected video encoder: %s", enc)
                return enc
        except Exception:
            continue

    logger.info("Defaulting video encoder to libx264")
    return "libx264"


def _build_encoder_args(encoder: str, crf: int) -> list[str]:
    """Map encoder and CRF quality setting to appropriate FFmpeg encoder arguments.

    Mapping (addresses Review 1b N9):
        - libx264: -c:v libx264 -crf <crf> -preset medium
        - h264_nvenc: -c:v h264_nvenc -rc vbr -cq <crf>
        - h264_qsv: -c:v h264_qsv -global_quality <crf>
        - h264_amf: -c:v h264_amf -rc cqp -qp_i <crf> -qp_p <crf>
    """
    crf_str = str(int(crf))
    if encoder == "libx264":
        return ["-c:v", "libx264", "-crf", crf_str, "-preset", "medium"]
    elif encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-rc", "vbr", "-cq", crf_str]
    elif encoder == "h264_qsv":
        return ["-c:v", "h264_qsv", "-global_quality", crf_str]
    elif encoder == "h264_amf":
        return ["-c:v", "h264_amf", "-rc", "cqp", "-qp_i", crf_str, "-qp_p", crf_str]
    return ["-c:v", encoder]


def process_video(
    src: str,
    dst: str,
    preset: dict,
    progress_cb: Callable[[float, float], None] | None = None,
    cancel: threading.Event | None = None,
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> None:
    """Process a video file according to the given preset.

    Args:
        src: Path to input video file.
        dst: Path to output video destination.
        preset: Validated preset dictionary or raw dictionary to validate.
        progress_cb: Optional callback receiving (current_progress, total_progress).
        cancel: Optional threading.Event to signal job cancellation.
        ffmpeg: FFmpeg binary executable name/path.
        ffprobe: FFprobe binary executable name/path.

    Raises:
        FileNotFoundError: If source video or required external assets do not exist.
        ValueError: If preset parameters are invalid or unsupported in chosen mode.
        RuntimeError: If FFmpeg encoding fails or job is cancelled.
    """
    valid_preset = validate_preset(preset)

    if not os.path.isfile(src):
        raise FileNotFoundError(f"Input video file not found: '{src}'")

    info = probe(src, ffprobe=ffprobe)

    # Convert preset boxes to absolute pixel coordinates for this video resolution
    abs_boxes = [to_abs_box(b, info.width, info.height) for b in valid_preset["boxes"]]

    mode = valid_preset["video"]["mode"]
    method = valid_preset["method"]
    params = valid_preset["params"]
    copy_audio = valid_preset["video"]["copy_audio"] and info.has_audio

    # Resolve encoder
    enc_setting = valid_preset["video"]["encoder"]
    encoder = detect_encoder(ffmpeg=ffmpeg) if enc_setting == "auto" else enc_setting
    crf = valid_preset["video"]["crf"]

    # Ensure output parent directory exists
    out_dir = os.path.dirname(dst)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if mode == "fast":
        _process_video_fast(
            src=src,
            dst=dst,
            info=info,
            abs_boxes=abs_boxes,
            method=method,
            params=params,
            encoder=encoder,
            crf=crf,
            copy_audio=copy_audio,
            progress_cb=progress_cb,
            cancel=cancel,
            ffmpeg=ffmpeg,
        )
    else:
        _process_video_quality(
            src=src,
            dst=dst,
            info=info,
            abs_boxes=abs_boxes,
            method=method,
            params=params,
            encoder=encoder,
            crf=crf,
            copy_audio=copy_audio,
            progress_cb=progress_cb,
            cancel=cancel,
            ffmpeg=ffmpeg,
        )


def _process_video_fast(
    src: str,
    dst: str,
    info: VideoInfo,
    abs_boxes: list[dict],
    method: str,
    params: dict,
    encoder: str,
    crf: int,
    copy_audio: bool,
    progress_cb: Callable[[float, float], None] | None,
    cancel: threading.Event | None,
    ffmpeg: str,
) -> None:
    """Execute fast mode via a single FFmpeg process with filtergraphs."""
    # Review 1b Note 1: clone_smooth is not supported in fast mode
    if method == "clone_smooth":
        raise ValueError("clone_smooth is not supported in fast mode. Please use quality mode.")

    # Review 1b N12: Write to temporary .part file to avoid corrupting/overwriting existing dst on failure
    stem, ext = os.path.splitext(dst)
    part_path = f"{stem}.part{ext}"

    cmd = [ffmpeg, "-y", "-v", "error", "-i", src]
    filter_args: list[str] = []

    if method in ("inpaint_fast", "inpaint_hq"):
        # Review 1b Note 2: Clamp boxes at least 1px away from borders for FFmpeg delogo filter
        clamped_filters = []
        for b in abs_boxes:
            dx = max(1, min(info.width - 3, int(b["x"])))
            dy = max(1, min(info.height - 3, int(b["y"])))
            dw = max(2, min(info.width - 1 - dx, int(b["w"])))
            dh = max(2, min(info.height - 1 - dy, int(b["h"])))
            clamped_filters.append(f"delogo=x={dx}:y={dy}:w={dw}:h={dh}")
        filter_str = ",".join(clamped_filters) if clamped_filters else "null"
        filter_args = ["-vf", filter_str]

    elif method == "solid":
        color_hex = params.get("solid_color", "#000000").replace("#", "0x")
        draw_filters = []
        for b in abs_boxes:
            bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            draw_filters.append(f"drawbox=x={bx}:y={by}:w={bw}:h={bh}:color={color_hex}:t=fill")
        filter_str = ",".join(draw_filters) if draw_filters else "null"
        filter_args = ["-vf", filter_str]

    elif method == "blur":
        ksize = max(3, int(params.get("blur_ksize", 31)))
        fc_parts = []
        cur_v = "0:v"
        for i, b in enumerate(abs_boxes):
            bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            r = max(1, min(int(ksize // 2), min(bw, bh) // 4))
            m_out = f"m{i}"
            c_out = f"c{i}"
            b_out = f"b{i}"
            v_next = f"v{i+1}"
            # Review 1b B5: Use [{c_out}] instead of literal [c_out]
            fc_parts.append(f"[{cur_v}]split[{m_out}][{c_out}]")
            fc_parts.append(f"[{c_out}]crop={bw}:{bh}:{bx}:{by},boxblur={r}:1:{r}:1[{b_out}]")
            fc_parts.append(f"[{m_out}][{b_out}]overlay={bx}:{by}[{v_next}]")
            cur_v = v_next
        fc_str = ";".join(fc_parts)
        filter_args = ["-filter_complex", fc_str, "-map", f"[{cur_v}]"]

    elif method == "mosaic":
        block = max(2, int(params.get("mosaic_block", 12)))
        fc_parts = []
        cur_v = "0:v"
        for i, b in enumerate(abs_boxes):
            bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            m_out = f"m{i}"
            c_out = f"c{i}"
            mos_out = f"mos{i}"
            v_next = f"v{i+1}"
            # Review 1b B5: Use [{c_out}] instead of literal [c_out]
            fc_parts.append(f"[{cur_v}]split[{m_out}][{c_out}]")
            scale_down = f"scale=max(2\\,iw/{block}):max(2\\,ih/{block})"
            scale_up = f"scale={bw}:{bh}:flags=neighbor"
            fc_parts.append(f"[{c_out}]crop={bw}:{bh}:{bx}:{by},{scale_down},{scale_up}[{mos_out}]")
            fc_parts.append(f"[{m_out}][{mos_out}]overlay={bx}:{by}[{v_next}]")
            cur_v = v_next
        fc_str = ";".join(fc_parts)
        filter_args = ["-filter_complex", fc_str, "-map", f"[{cur_v}]"]

    elif method == "overlay":
        ov_path = params.get("overlay_path")
        if not ov_path or not os.path.isfile(ov_path):
            raise FileNotFoundError(f"Overlay image file not found: '{ov_path}'")
        cmd.extend(["-i", ov_path])
        opacity = float(params.get("overlay_opacity", 1.0))
        fc_parts = []
        cur_v = "0:v"
        for i, b in enumerate(abs_boxes):
            bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
            ov_scaled = f"ov{i}"
            v_next = f"v{i+1}"
            fc_parts.append(
                f"[1:v]format=rgba,colorchannelmixer=aa={opacity:.2f},scale={bw}:{bh}[{ov_scaled}]"
            )
            fc_parts.append(f"[{cur_v}][{ov_scaled}]overlay={bx}:{by}[{v_next}]")
            cur_v = v_next
        fc_str = ";".join(fc_parts)
        filter_args = ["-filter_complex", fc_str, "-map", f"[{cur_v}]"]

    cmd.extend(filter_args)

    # Video encoder settings with mapped CRF (Review 1b N9)
    cmd.extend(_build_encoder_args(encoder, crf))
    cmd.extend(["-pix_fmt", "yuv420p"])

    # Audio handling
    if copy_audio:
        if "-filter_complex" in filter_args:
            cmd.extend(["-map", "0:a?", "-c:a", "copy"])
        else:
            cmd.extend(["-c:a", "copy"])
    else:
        cmd.append("-an")

    cmd.extend(["-progress", "pipe:1", part_path])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )

    stderr_lines: list[str] = []

    def read_stderr():
        if proc.stderr:
            for line in proc.stderr:
                stderr_lines.append(line.decode("utf-8", errors="replace"))

    t_err = threading.Thread(target=read_stderr, daemon=True)
    t_err.start()

    total_sec = info.duration if info.duration > 0.0 else (info.nb_frames / info.fps if info.fps > 0 else 1.0)
    cancelled = False

    try:
        if proc.stdout:
            for line_bytes in proc.stdout:
                if cancel and cancel.is_set():
                    cancelled = True
                    break
                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                # Review 1b N11: out_time_us is microseconds, out_time_ms in FFmpeg progress is also microseconds
                if line_str.startswith("out_time_us=") or line_str.startswith("out_time_ms="):
                    try:
                        us = int(line_str.split("=", 1)[1])
                        current_sec = us / 1_000_000.0
                        if progress_cb and total_sec > 0.0:
                            progress_cb(current_sec, total_sec)
                    except ValueError:
                        pass
    finally:
        if cancelled or (cancel and cancel.is_set()):
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
        else:
            proc.wait()
        t_err.join(timeout=2.0)

    if cancelled or (cancel and cancel.is_set()):
        if os.path.exists(part_path):
            os.remove(part_path)
        raise RuntimeError("Video processing was cancelled.")

    if proc.returncode != 0:
        if os.path.exists(part_path):
            os.remove(part_path)
        last_err = "".join(stderr_lines[-20:])
        raise RuntimeError(f"FFmpeg fast mode failed (exit code {proc.returncode}):\n{last_err}")

    # Success: atomically move part file to final destination (Review 1b N12)
    os.replace(part_path, dst)


def _process_video_quality(
    src: str,
    dst: str,
    info: VideoInfo,
    abs_boxes: list[dict],
    method: str,
    params: dict,
    encoder: str,
    crf: int,
    copy_audio: bool,
    progress_cb: Callable[[float, float], None] | None,
    cancel: threading.Event | None,
    ffmpeg: str,
) -> None:
    """Execute quality mode via piped raw video frames processed by image_engine."""
    # Pre-validate overlay asset exists before opening FFmpeg pipes (Review 1b Note 4)
    if method == "overlay":
        ov_path = params.get("overlay_path")
        if not ov_path or not os.path.isfile(ov_path):
            raise FileNotFoundError(f"Overlay image file not found: '{ov_path}'")

    # Review 1b N12: Write to temporary .part file to avoid corrupting/overwriting existing dst on failure
    stem, ext = os.path.splitext(dst)
    part_path = f"{stem}.part{ext}"

    dec_cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        src,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]

    # Review 1b N14: Preserve exact frame rate string (e.g. '30000/1001') to avoid audio desync
    enc_cmd = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{info.width}x{info.height}",
        "-r",
        info.r_frame_rate,
        "-i",
        "-",
    ]

    if copy_audio:
        enc_cmd.extend(["-i", src, "-map", "0:v", "-map", "1:a?", "-c:a", "copy"])
    else:
        enc_cmd.append("-an")

    # Review 1b N9: Map CRF to encoder quality parameters
    enc_cmd.extend(_build_encoder_args(encoder, crf))
    enc_cmd.extend(["-pix_fmt", "yuv420p", part_path])

    dec_proc = subprocess.Popen(
        dec_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )

    enc_proc = subprocess.Popen(
        enc_cmd,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )

    dec_err_lines: list[str] = []
    enc_err_lines: list[str] = []

    # Review 1b Note 3: Separate stderr reader threads to prevent pipe deadlock on long videos
    def read_dec_err():
        if dec_proc.stderr:
            for l in dec_proc.stderr:
                dec_err_lines.append(l.decode("utf-8", errors="replace"))

    def read_enc_err():
        if enc_proc.stderr:
            for l in enc_proc.stderr:
                enc_err_lines.append(l.decode("utf-8", errors="replace"))

    t_dec_err = threading.Thread(target=read_dec_err, daemon=True)
    t_enc_err = threading.Thread(target=read_enc_err, daemon=True)
    t_dec_err.start()
    t_enc_err.start()

    # Review 1b N10: Precompute binary mask once to avoid recomputing build_mask on every frame
    dilate = int(params.get("dilate", 3))
    feather = int(params.get("feather", 4))
    mask = build_mask((info.height, info.width), abs_boxes, dilate=dilate, feather=feather)

    frame_size = info.width * info.height * 3
    frame_idx = 0
    total_frames = info.nb_frames if info.nb_frames > 0 else int(round(info.duration * info.fps))
    cancelled = False
    aborted_early = False

    try:
        while True:
            if cancel and cancel.is_set():
                cancelled = True
                aborted_early = True
                break

            assert dec_proc.stdout is not None
            raw_bytes = dec_proc.stdout.read(frame_size)
            if not raw_bytes or len(raw_bytes) < frame_size:
                # End of stream reached cleanly
                break

            frame = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((info.height, info.width, 3))

            # Review 1b Note 4: apply error halts job and cleans up corrupt output
            try:
                processed = apply_with_mask(frame, mask, abs_boxes, method, params)
            except Exception as apply_exc:
                aborted_early = True
                raise apply_exc

            assert enc_proc.stdin is not None
            try:
                enc_proc.stdin.write(processed.tobytes())
            except (BrokenPipeError, OSError):
                aborted_early = True
                break

            frame_idx += 1
            if progress_cb and total_frames > 0:
                progress_cb(frame_idx, total_frames)

    except Exception:
        aborted_early = True
        raise

    finally:
        # Review 1b B6: If aborted early (cancel/error/broken pipe), terminate decoder first to unblock pipes
        if aborted_early or cancelled or (cancel and cancel.is_set()):
            try:
                dec_proc.terminate()
            except OSError:
                pass
            try:
                enc_proc.terminate()
            except OSError:
                pass
            if dec_proc.stdout:
                try:
                    dec_proc.stdout.close()
                except OSError:
                    pass
            if enc_proc.stdin:
                try:
                    enc_proc.stdin.close()
                except OSError:
                    pass

            for p in (dec_proc, enc_proc):
                try:
                    p.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()

            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass
        else:
            # Normal completion
            if enc_proc.stdin:
                try:
                    enc_proc.stdin.close()
                except OSError:
                    pass
            if dec_proc.stdout:
                try:
                    dec_proc.stdout.close()
                except OSError:
                    pass

            dec_proc.wait()
            enc_proc.wait()

        t_dec_err.join(timeout=2.0)
        t_enc_err.join(timeout=2.0)

    if cancelled or (cancel and cancel.is_set()):
        if os.path.exists(part_path):
            os.remove(part_path)
        raise RuntimeError("Video processing was cancelled.")

    # Review 1b N8: Verify both decoder and encoder exit codes
    if dec_proc.returncode not in (0, None):
        if os.path.exists(part_path):
            os.remove(part_path)
        err_msg = "".join(dec_err_lines[-20:])
        raise RuntimeError(f"FFmpeg decoder failed (code {dec_proc.returncode}):\n{err_msg}")

    if enc_proc.returncode != 0:
        if os.path.exists(part_path):
            os.remove(part_path)
        err_msg = "".join(enc_err_lines[-20:])
        raise RuntimeError(f"FFmpeg quality encoder failed (code {enc_proc.returncode}):\n{err_msg}")

    # Review 1b N8: Warn if processed frame count deviates by > 2% from stream header
    if total_frames > 50 and abs(frame_idx - total_frames) > 0.02 * total_frames:
        logger.warning(
            "Processed frame count (%d) deviates by more than 2%% from expected (%d)",
            frame_idx,
            total_frames,
        )

    # Success: atomically move part file to final destination (Review 1b N12)
    os.replace(part_path, dst)
