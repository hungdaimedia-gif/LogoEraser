"""Tests for the LaMa AI inpainting method (inpaint_ai).

Tự skip khi chưa có model hoặc onnxruntime — các test còn lại của dự án không phụ thuộc AI.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest

from core.ai_inpaint import TemporalStabilizer, model_available
from core.image_engine import apply, build_mask
from core.preset import get_default_preset
from core.video_engine import probe, process_video
from tests.make_samples import create_synthetic_video

ort = pytest.importorskip("onnxruntime")
needs_model = pytest.mark.skipif(not model_available(), reason="models/lama_fp32.onnx chưa được tải")


def _woven_texture(w: int = 480, h: int = 320) -> np.ndarray:
    """Nền vân đan (giống chiếu cói/vải) — trường hợp Telea/NS làm nhoè rõ nhất."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    weave = 0.5 + 0.25 * np.sin(xx / 3.0) * np.sign(np.sin(yy / 6.0)) + 0.25 * np.sin(yy / 3.0) * np.sign(np.sin(xx / 6.0))
    base = np.stack([110 + 60 * weave, 140 + 50 * weave, 170 + 40 * weave], axis=-1)
    return np.clip(base, 0, 255).astype(np.uint8)


def _detail_error(out: np.ndarray, clean: np.ndarray, box: tuple[int, int, int, int]) -> tuple[float, float]:
    """(sai lệch màu, độ hụt chi tiết) trong vùng box so với nền sạch."""
    x, y, w, h = box
    a = out[y:y + h, x:x + w].astype(np.float32)
    b = clean[y:y + h, x:x + w].astype(np.float32)
    color_err = float(np.abs(a - b).mean())
    lap = lambda im: float(np.abs(cv2.Laplacian(cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY), cv2.CV_32F)).mean())
    detail_loss = 1.0 - lap(a) / max(lap(b), 1e-6)  # 0 = giữ đủ chi tiết, 1 = phẳng lì
    return color_err, detail_loss


@needs_model
def test_inpaint_ai_keeps_texture_better_than_opencv():
    clean = _woven_texture()
    box = (300, 220, 120, 60)
    x, y, w, h = box
    corrupted = clean.copy()
    cv2.rectangle(corrupted, (x + 6, y + 6), (x + w - 6, y + h - 6), (255, 255, 255), -1)
    cv2.putText(corrupted, "Veo", (x + 18, y + h - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2)
    b = {"x": x, "y": y, "w": w, "h": h, "unit": "absolute"}

    ai = apply(corrupted, [b], "inpaint_ai", {"dilate": 3, "feather": 4})
    hq = apply(corrupted, [b], "inpaint_hq", {"dilate": 3, "feather": 4, "radius": 5})

    ai_err, ai_loss = _detail_error(ai, clean, box)
    hq_err, hq_loss = _detail_error(hq, clean, box)
    print(f"\n[inpaint_ai] color err {ai_err:.1f} detail loss {ai_loss:.0%} | inpaint_hq {hq_err:.1f} / {hq_loss:.0%}")

    # Logo phải mất và vân phải còn: AI giữ chi tiết tốt hơn hẳn inpaint cổ điển
    assert ai_err < 0.5 * float(np.abs(corrupted[y:y + h, x:x + w].astype(np.float32) - clean[y:y + h, x:x + w]).mean())
    assert ai_loss < hq_loss - 0.2


@needs_model
def test_inpaint_ai_only_changes_masked_area():
    img = _woven_texture()
    b = {"x": 50, "y": 40, "w": 80, "h": 40, "unit": "absolute"}
    out = apply(img, [b], "inpaint_ai", {"dilate": 3, "feather": 4})
    mask = build_mask(img.shape, [b], dilate=3, feather=4)
    assert np.array_equal(out[mask == 0], img[mask == 0])


@needs_model
@pytest.mark.parametrize("box", [(0, 0, 60, 40), (420, 280, 60, 40), (200, 0, 80, 30)], ids=["top_left", "bottom_right", "top_edge"])
def test_inpaint_ai_handles_frame_edges(box):
    img = _woven_texture()
    x, y, w, h = box
    img[y:y + h, x:x + w] = 255
    out = apply(img, [{"x": x, "y": y, "w": w, "h": h, "unit": "absolute"}], "inpaint_ai", {"dilate": 2, "feather": 2})
    assert out.shape == img.shape
    assert out[y:y + h, x:x + w].mean() < 230  # khối trắng đã được lấp


def test_inpaint_ai_rejected_in_fast_mode(tmp_path):
    src = str(tmp_path / "in.mp4")
    box = create_synthetic_video(src, duration=1.0, logo_box=(240, 190, 70, 40))
    p = get_default_preset()
    p["boxes"] = [box]
    p["method"] = "inpaint_ai"
    p["video"]["mode"] = "fast"
    with pytest.raises(ValueError, match="quality"):
        process_video(src, str(tmp_path / "out.mp4"), p)


def test_missing_model_reported_before_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setenv("LOGOERASER_LAMA_MODEL", str(tmp_path / "khong_co.onnx"))
    src = str(tmp_path / "in.mp4")
    box = create_synthetic_video(src, duration=1.0, logo_box=(240, 190, 70, 40))
    p = get_default_preset()
    p["boxes"] = [box]
    p["method"] = "inpaint_ai"
    p["video"]["mode"] = "quality"
    with pytest.raises(FileNotFoundError, match="lama_fp32.onnx"):
        process_video(src, str(tmp_path / "out.mp4"), p)
    assert not os.path.exists(tmp_path / "out.mp4")


@needs_model
def test_inpaint_ai_video_quality_mode(tmp_path):
    src = str(tmp_path / "in.mp4")
    dst = str(tmp_path / "out.mp4")
    box = create_synthetic_video(src, duration=1.0, logo_box=(240, 190, 70, 40))
    p = get_default_preset()
    p["boxes"] = [box]
    p["method"] = "inpaint_ai"
    p["video"]["mode"] = "quality"
    process_video(src, dst, p)
    a, b = probe(src), probe(dst)
    assert (b.width, b.height) == (a.width, a.height)
    assert abs(b.nb_frames - a.nb_frames) <= 1


def test_temporal_stabilizer_smooths_static_scene_but_not_motion():
    h, w = 120, 160
    mask = np.zeros((h, w), np.uint8)
    mask[40:80, 60:100] = 255
    stab = TemporalStabilizer(mask)
    still = np.full((h, w, 3), 100, np.uint8)

    out_a = still.copy(); out_a[40:80, 60:100] = 50
    out_b = still.copy(); out_b[40:80, 60:100] = 150  # AI "nhấp nháy" giữa 2 khung
    stab.step(still, out_a)
    r = stab.step(still, out_b)
    assert 50 < int(r[60, 80, 0]) < 150  # nền đứng yên ⇒ đã trộn

    moving = np.full((h, w, 3), 200, np.uint8)  # nền đổi hẳn ⇒ không trộn (tránh bóng ma)
    out_c = moving.copy(); out_c[40:80, 60:100] = 30
    r2 = stab.step(moving, out_c)
    assert int(r2[60, 80, 0]) == 30


def _panning_clip(n: int = 24, w: int = 320, h: int = 240, speed: int = 3) -> np.ndarray:
    """Nền có vân trượt ngang đều (camera lia) — dữ liệu có đáp án cho từng khung."""
    big = _woven_texture(w + n * speed + 8, h)
    return np.stack([big[:, i * speed:i * speed + w] for i in range(n)])


def test_flow_stabilizer_reduces_flicker_on_moving_background():
    from core.ai_inpaint import FlowStabilizer

    gt = _panning_clip()
    n, h, w, _ = gt.shape
    box = [{"x": 120, "y": 90, "w": 80, "h": 50, "unit": "absolute"}]
    mask = build_mask((h, w), box, dilate=3, feather=4)
    hole = mask > 10
    rng = np.random.default_rng(0)

    # Giả lập AI: đúng nội dung nhưng mỗi khung lệch ngẫu nhiên một chút ⇒ nhấp nháy
    frames_in, raw = [], []
    for g in gt:
        logo = g.copy()
        logo[hole] = 255
        noisy = g.astype(np.int16)
        noisy[hole] += rng.integers(-25, 26, size=(int(hole.sum()), 3), dtype=np.int16)
        frames_in.append(logo)
        raw.append(np.clip(noisy, 0, 255).astype(np.uint8))

    st = FlowStabilizer(mask, box)
    out = [st.step(a, b) for a, b in zip(frames_in, raw)]

    def flicker(seq):
        s = np.stack(seq).astype(np.float32)
        return float(np.abs(np.diff(s, axis=0) - np.diff(gt.astype(np.float32), axis=0))[:, hole].mean())

    def error(seq):
        return float(np.abs(np.stack(seq).astype(np.float32) - gt.astype(np.float32))[:, hole].mean())

    print(f"\n[flow] flicker {flicker(raw):.1f} -> {flicker(out):.1f} | error {error(raw):.1f} -> {error(out):.1f}")
    assert flicker(out) < 0.6 * flicker(raw)
    assert error(out) <= error(raw)  # kéo theo chuyển động đúng ⇒ không tạo bóng ma
    assert np.array_equal(out[5][mask == 0], raw[5][mask == 0])  # ngoài vùng logo không đổi


def test_flow_stabilizer_resets_on_scene_cut():
    from core.ai_inpaint import FlowStabilizer

    h, w = 240, 320
    box = [{"x": 120, "y": 90, "w": 80, "h": 50, "unit": "absolute"}]
    mask = build_mask((h, w), box, dilate=3, feather=4)
    st = FlowStabilizer(mask, box)
    a = _woven_texture(w, h)
    b = np.full((h, w, 3), 30, np.uint8)  # cắt cảnh sang khung tối hoàn toàn khác
    st.step(a, a)
    out = st.step(b, b)
    assert st.last_weights == [0.0]
    assert np.array_equal(out, b)
