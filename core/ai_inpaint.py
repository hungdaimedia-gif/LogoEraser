"""AI inpainting with the LaMa model (ONNX, chạy offline).

LaMa (Suvorov et al., "Resolution-robust Large Mask Inpainting with Fourier
Convolutions", Apache-2.0) dựng lại vân/cạnh của nền thay vì kéo màu từ mép
vào như Telea/NS — khác biệt thấy rõ trên vải, chiếu cói, mép bàn.

Model: `models/lama_fp32.onnx` cạnh ứng dụng (bản ONNX của Carve/LaMa-ONNX,
đầu vào 512×512: image RGB 0..1 + mask 0/1, đầu ra RGB 0..255).
Có GPU NVIDIA + onnxruntime-gpu thì chạy CUDA, không thì CPU (~1.3 s/vùng).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from core.config import app_dir

logger = logging.getLogger(__name__)

MODEL_FILENAME = "lama_fp32.onnx"
MODEL_URL = "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
MODEL_SIZE = 512

_session = None
_session_lock = threading.Lock()


def model_path() -> Path:
    env = os.environ.get("LOGOERASER_LAMA_MODEL")
    return Path(env) if env else app_dir() / "models" / MODEL_FILENAME


def model_available() -> bool:
    return model_path().is_file()


def missing_model_message() -> str:
    return (
        f"Thiếu model AI '{MODEL_FILENAME}'.\n"
        f"Đặt file vào: {model_path().parent}\n"
        f"Tải (khoảng 200 MB, một lần): chạy download_model.bat hoặc mở {MODEL_URL}"
    )


def _get_session():
    """Create (once) an ONNX Runtime session, preferring CUDA when usable."""
    global _session
    with _session_lock:
        if _session is not None:
            return _session
        path = model_path()
        if not path.is_file():
            raise FileNotFoundError(missing_model_message())

        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            try:
                # onnxruntime-gpu[cuda,cudnn] cài thư viện CUDA qua pip; cần nạp trước.
                if hasattr(ort, "preload_dlls"):
                    ort.preload_dlls()
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except Exception as e:  # thiếu driver/cuDNN ⇒ CPU
                logger.warning("CUDA unavailable for LaMa, using CPU: %s", e)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(str(path), so, providers=providers)
        except Exception as e:
            if providers[0] == "CPUExecutionProvider":
                raise
            logger.warning("CUDA session failed, falling back to CPU: %s", e)
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        logger.info("LaMa providers: %s", sess.get_providers())
        _session = sess
        return _session


def active_provider() -> str:
    """'CUDAExecutionProvider' or 'CPUExecutionProvider' (loads the model if needed)."""
    return _get_session().get_providers()[0]


def _crop_rect(w: int, h: int, box: dict) -> tuple[int, int, int, int]:
    """Square-ish context window around a box: box + one box-size of context each side.

    LaMa cần nhìn đủ nền xung quanh để đoán vân; vùng ngữ cảnh tối thiểu 256 px.
    """
    bx, by, bw, bh = int(box["x"]), int(box["y"]), int(box["w"]), int(box["h"])
    side = max(256, 3 * max(bw, bh))
    cw, ch = min(side, w), min(side, h)
    # Hộp dài hơn khung nhìn (ví dụ dải chữ ngang) ⇒ nới theo hộp
    cw = max(cw, min(w, bw + 32))
    ch = max(ch, min(h, bh + 32))
    cx, cy = bx + bw // 2, by + bh // 2
    x0 = min(max(0, cx - cw // 2), w - cw)
    y0 = min(max(0, cy - ch // 2), h - ch)
    return x0, y0, cw, ch


def inpaint_lama(img: np.ndarray, mask: np.ndarray, boxes_abs: list[dict]) -> np.ndarray:
    """Inpaint every box region of `img` with LaMa.

    Args:
        img: BGR uint8.
        mask: uint8 0..255 (đã nới + làm mềm bởi build_mask). >10 là vùng cần lấp;
            giá trị mềm ở mép dùng làm alpha khi ghép lại.
    """
    sess = _get_session()
    h, w = img.shape[:2]
    out = img.copy()
    alpha_full = mask.astype(np.float32) / 255.0
    hole_full = (mask > 10).astype(np.float32)

    for box in boxes_abs:
        x0, y0, cw, ch = _crop_rect(w, h, box)
        hole = hole_full[y0:y0 + ch, x0:x0 + cw]
        if not hole.any():
            continue
        crop = out[y0:y0 + ch, x0:x0 + cw]

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(rgb, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        m = cv2.resize(hole, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_NEAREST)
        # Lỗ nhỏ có thể biến mất khi thu nhỏ ⇒ nới 1 px ở không gian model
        m = cv2.dilate((m > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)

        res = sess.run(None, {"image": inp.transpose(2, 0, 1)[None], "mask": m[None, None]})[0][0]
        res = res.transpose(1, 2, 0)
        if res.max() <= 1.5:  # một số bản export trả 0..1
            res = res * 255.0
        res = cv2.resize(np.clip(res, 0, 255).astype(np.uint8), (cw, ch), interpolation=cv2.INTER_CUBIC)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

        a = alpha_full[y0:y0 + ch, x0:x0 + cw][..., None]
        blended = res.astype(np.float32) * a + crop.astype(np.float32) * (1.0 - a)
        out[y0:y0 + ch, x0:x0 + cw] = np.clip(blended, 0, 255).astype(np.uint8)

    return out


class TemporalStabilizer:
    """Giảm rung/nhấp nháy khi inpaint từng khung video.

    AI dựng lại vùng logo hơi khác nhau ở mỗi khung ⇒ nhấp nháy. Khi nền quanh
    logo gần như đứng yên, trộn kết quả với khung trước; khi nền chuyển động
    (camera lia, vật đi qua) thì không trộn để tránh bóng ma.
    """

    def __init__(self, mask: np.ndarray, still_threshold: float = 6.0, weight_prev: float = 0.6) -> None:
        hole = (mask > 10).astype(np.uint8)
        ring = cv2.dilate(hole, np.ones((15, 15), np.uint8)) - hole
        self.ring = ring.astype(bool)
        self.alpha = (mask.astype(np.float32) / 255.0)[..., None]
        self.still_threshold = still_threshold
        self.weight_prev = weight_prev
        self.prev_in: np.ndarray | None = None
        self.prev_out: np.ndarray | None = None

    def step(self, frame_in: np.ndarray, frame_out: np.ndarray) -> np.ndarray:
        result = frame_out
        if self.prev_in is not None and self.ring.any():
            motion = float(np.abs(frame_in[self.ring].astype(np.int16) - self.prev_in[self.ring].astype(np.int16)).mean())
            if motion < self.still_threshold:
                # nền đứng yên ⇒ càng yên càng trộn nhiều với khung trước
                wp = self.weight_prev * (1.0 - motion / self.still_threshold)
                mixed = frame_out.astype(np.float32) * (1.0 - wp) + self.prev_out.astype(np.float32) * wp
                result = (mixed * self.alpha + frame_out.astype(np.float32) * (1.0 - self.alpha)).astype(np.uint8)
        self.prev_in = frame_in
        self.prev_out = result
        return result
