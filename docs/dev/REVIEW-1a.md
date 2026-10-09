# Review Phase 1a — commit 65cd50b

**Kết luận: CHƯA ĐẠT.** Phải sửa 3 lỗi chặn rồi bàn giao lại 1a. Chưa được làm 1b.

`pytest` chạy lại trên máy: 7 passed. Nhưng test hiện tại chỉ kiểm shape/dtype
nên bỏ lọt 2 method hỏng. Số liệu bên dưới đo bằng ảnh mẫu
`create_synthetic_image(400,300,(150,100,70,45))`, `dilate=2, feather=3, radius=5`.

| method | sai lệch vùng logo so với nền sạch (ban đầu 90.42) |
|---|---|
| inpaint_fast | 13.96 |
| inpaint_hq | 13.81 |
| clone_smooth | **91.46 — tệ hơn không làm gì** |
| overlay (đường dẫn PNG sai) | **90.42 — ảnh không đổi, không báo lỗi** |

## Lỗi chặn (bắt buộc sửa)

### B1. `clone_smooth` không xoá được logo — `core/image_engine.py:124-191`
Kết quả vẫn là khối vàng. Nguyên nhân: `patch_mask` phủ kín 255 đúng bằng box,
nên `seamlessClone` lấy điều kiện biên từ **chính viền logo** ở ảnh đích ⇒ màu
logo bị lan ngược vào trong. Hàm này cũng bỏ qua tham số `dilate`.
Cách sửa: vùng clone phải là box **đã nới rộng** (theo `dilate` + tối thiểu
vài px) để biên nằm trên nền sạch; patch nguồn cũng lấy theo kích thước đã nới
rộng; mask truyền vào `seamlessClone` có lề 0 ít nhất 1px. Khi patch nguồn
trùng một box logo khác thì bỏ ứng viên đó.

### B2. Lỗi âm thầm — vi phạm PLAN §3 ("không âm thầm rơi về mặc định")
- `_apply_overlay` (`:242-248`): `overlay_path` rỗng hoặc không đọc được ⇒ trả
  ảnh gốc. CLI sẽ báo thành công mà file y nguyên. Phải raise `ValueError`/
  `FileNotFoundError` có tên file.
- `parse_hex_color` (`:22-33`): `"red"` ⇒ âm thầm thành đen. Phải raise.
- `Image.open(overlay_path)` không đọc được đường dẫn có dấu tiếng Việt?
  (PIL đọc được — giữ nguyên, nhưng thêm test với tên file `logo_ảnh.png`.)

### B3. `validate_preset` chấp nhận dữ liệu sai — `core/preset.py:575-666`
Các preset sau hiện đều được **chấp nhận**, phải bị từ chối (hoặc kẹp có warning
ghi rõ):
- `dilate`, `feather`, `radius`, `blur_ksize`, `mosaic_block` âm ⇒ lỗi
  (`mosaic_block` ≥ 2, `radius` ≥ 1).
- `overlay_opacity` ngoài [0,1] ⇒ lỗi.
- `video.mode` không thuộc `{fast, quality}`; `video.encoder` không thuộc
  `{auto, h264_nvenc, h264_qsv, h264_amf, libx264}` ⇒ lỗi.
- `copy_audio: "false"` đang thành `True` (do `bool("false")`). Chỉ nhận bool thật.
- `crf` ngoài [0,51] ⇒ lỗi.
- `version` ≠ 1 ⇒ lỗi "preset version không hỗ trợ".
- Box `w`/`h` = 0, hoặc box `relative` có giá trị > 1 ⇒ lỗi.
- `solid_color` phải đúng `#RRGGBB` ngay từ lúc validate (dùng chung hàm với B2).

## Test bắt buộc thêm (để B1–B3 không tái phát)

1. `test_all_seven_methods_execute` → với **mỗi** method, kiểm thêm:
   vùng box **có thay đổi**, và pixel cách box > `dilate+feather+2` px
   **không đổi**.
2. Cổng chất lượng riêng cho `clone_smooth`: sai lệch < 50% ban đầu, giống
   cổng `inpaint_hq` (dùng nền có vân lặp — ví dụ sọc ngang — vì đó là trường
   hợp clone mạnh hơn inpaint).
3. `overlay` với đường dẫn không tồn tại ⇒ `pytest.raises`.
4. Một test tham số hoá cho từng preset sai ở B3 ⇒ `pytest.raises`.
5. `test_detect_logo_boxes`: kiểm box đầu tiên có IoU ≥ 0.5 với logo thật
   (hiện chỉ kiểm `len >= 1`).

## Không chặn, nên sửa trong lượt này

- **N1 `inpaint_hq` gần như không hơn `inpaint_fast`** (13.81 so với 13.96)
  nhưng tốn ~3 lần thời gian. Hoặc cải thiện thật, hoặc ghi rõ trong docstring
  khác biệt là gì. Thêm một dòng in so sánh hq/fast trong test cổng để theo dõi.
- **N2 `detect_logo_boxes` trả box lồng nhau**: với logo hình chữ nhật đơn
  (20,20)-(60,50) trên khung 200×200, trả về 2 box chồng nhau
  (`0.06,0.06,0.285,0.235` và `0.06,0.06,0.085,0.085`), box đầu bị phình ~8px
  mỗi cạnh. Cần gộp box chồng nhau (IoU/chứa nhau) và bớt `dilate iterations=2`.
- **N3** `build_mask` (`:54`): `cv2.rectangle(..., (bx+bw, by+bh))` vẽ thừa 1px
  (toạ độ cuối là inclusive). Dùng `bx+bw-1, by+bh-1` hoặc gán slice numpy.
- **N4** `tests/make_samples.py`: docstring nói trả 4 giá trị, thực tế trả 3;
  import `Image` thừa; vòng lặp pixel Python thuần ⇒ vector hoá bằng `np.meshgrid`.
- **N5** `requirements.txt` chưa ghim phiên bản. Ghim mức tối thiểu
  (`Image.Resampling` cần Pillow ≥ 9.1). Thêm `requirements-dev.txt` cho `pytest`.
- **N6** Thư mục `.venv/` nằm trong `G:\LogoEraser` nhưng PLAN §2 không liệt kê
  — chấp nhận (đã có trong `.gitignore`), ghi chú vào README ở Phase 3.

## Đạt

- Đúng phạm vi file, không có import mạng, không có `powershell`/`wmic`.
- Hàm engine thuần, không I/O ngoài `overlay` đọc PNG.
- Không method nào làm thay đổi pixel ngoài vùng box.
- Chuyển đổi relative↔absolute và kẹp vào khung hình đúng.

## Bàn giao lại

Dùng mẫu PLAN §10, kèm bảng sai lệch 7 method như bảng trên (đo lại sau khi sửa).
