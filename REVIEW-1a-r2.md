# Review Phase 1a — lượt 2 — commit ec5d5fb

**Kết luận: GẦN ĐẠT, còn 1 lỗi chặn (B4).** Sửa B4 + thêm test là đạt 1a, không
cần bàn giao đầy đủ lại — chỉ cần commit + output pytest.

Chạy lại trên máy: 30 passed. B1, B2, B3, N2–N5 đã sửa đúng như yêu cầu.

## B4 (chặn) — `clone_smooth` hỏng khi logo chạm mép khung hình

Đo bằng `create_synthetic_image(400,300, box)`, `dilate=3, feather=4`:

| vị trí logo | inpaint_fast | inpaint_hq | clone_smooth |
|---|---|---|---|
| giữa (150,100,70,45) | 13.7% | 13.7% | 11.2% |
| cách mép 8px (340,10,52,30) | 10.5% | 9.5% | 4.6% |
| **chạm góc dưới-phải (330,255,70,45)** | 21.7% | 20.9% | **59.4% — còn vệt vàng** |

Logo dính mép/góc là trường hợp phổ biến nhất ngoài thực tế, nên đây là lỗi chặn.

Nguyên nhân (`core/image_engine.py`, `_clone_smooth`):
1. Box nới rộng bị cắt ở mép khung ⇒ cạnh đó của vùng clone **vẫn là pixel
   logo**, `seamlessClone` lại lấy điều kiện biên từ logo (đúng lỗi B1, nhưng ở mép).
2. `ix1 = min(ebw - 2, …)` / `iy1 = min(ebh - 2, …)` chừa 2px sát mép **không
   được phủ mask** ⇒ dải logo 2px còn nguyên.
3. `seamlessClone` đặt vùng theo **tâm boundingRect của mask**, không phải tâm
   patch. Khi lề hai bên không đều (do bị cắt ở mép), vùng dán bị lệch so với
   chỗ logo.

Cách sửa đề xuất (một cách duy nhất giải cả 3): đệm ảnh trước khi xử lý —
`cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REFLECT_101)` với
`m = margin + max(bw, bh)` (đủ chỗ cho patch nguồn), dịch toạ độ box thêm `m`,
chạy clone trên ảnh đã đệm, rồi cắt về `[m:m+h, m:m+w]`. Khi đó lề luôn đều,
không cần chừa 2px, và biên luôn nằm ngoài logo. Ứng viên patch nguồn phải
**nằm hoàn toàn trong vùng ảnh gốc** (không lấy từ phần đệm phản chiếu), nếu
không có ứng viên nào thì rơi về inpaint như hiện tại.

Test bắt buộc: tham số hoá `test_clone_smooth_error_reduction` (và cổng
`inpaint_hq`) cho 4 vị trí: giữa, chạm mép phải, chạm góc dưới-phải, chạm mép
trên. Cổng < 50% cho mọi vị trí.

## Không chặn

- **N1 chưa xử lý.** `inpaint_hq` vẫn ngang `inpaint_fast` (13.21 so với 13.19
  — hq còn kém hơn chút) nhưng chậm ~1.5–3 lần. Docstring mới viết "eliminate
  directional streak artifacts" mà không có số đo nào chứng minh. Chọn một:
  (a) thêm test vùng lớn (≥120×80 trên nền có vân) cho thấy hq tốt hơn fast
  ≥ 10%; hoặc (b) sửa docstring thành mô tả trung thực (trộn Telea+NS, thêm
  một lượt nửa độ phân giải cho vùng lớn) và bỏ câu khẳng định. Bàn giao ghi
  "Lệch so với PLAN: Không có" trong khi N1 chưa xong — lần sau ghi rõ mục nào
  chưa làm.
- **N7** `validate_preset`: `int(pv)` nhận `True` thành 1 và `3.9` thành 3 mà
  không báo. Nên chỉ nhận `int` thật (loại `bool`), giống cách đã làm với
  `copy_audio`.

## Đạt

- B1 (vùng giữa ảnh): clone_smooth giờ tốt nhất nhóm (11.2%).
- B2: overlay/solid_color sai ⇒ raise, có test đường dẫn tiếng Việt.
- B3: 21 preset sai bị từ chối.
- N2: detect IoU 0.949, không còn box lồng nhau. N3/N4/N5 xong.
