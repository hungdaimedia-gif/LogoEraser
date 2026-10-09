# Review Phase 1b — lượt 2 — commit f4577ef

**Kết luận: ĐẠT. Mở Phase 2.**

Kiểm lại trên máy: 66 passed. Script thử độc lập của Claude:

| Thử | Trước (b973097) | Nay (f4577ef) |
|---|---|---|
| fast · blur / mosaic | FAIL filtergraph | OK, 75 khung |
| fast · inpaint / solid / overlay | OK | OK |
| quality · huỷ ở khung 30 (video 20s) | treo > 25s, còn file dở | kết thúc 0.7s, raise "cancelled", không còn file |

Đã xem diff: B5, B6 sửa đúng gốc (terminate trước wait, có timeout → kill);
N8 kiểm decoder + cảnh báo lệch > 2%; N9 ánh xạ crf cho nvenc/qsv/amf;
N10 `apply_with_mask` (đường `apply()` cho ảnh không đổi hành vi); N11–N15 xong.

## Ghi nhận, không chặn (gộp vào Phase 2)

- **N16** `core/config.py: load_config(path="config.json")` đọc theo thư mục
  hiện hành ⇒ chạy `python G:\LogoEraser\cli.py` từ thư mục khác sẽ không thấy
  `config.json`. Đổi mặc định thành đường dẫn cạnh file mã nguồn
  (`Path(__file__).resolve().parent.parent / "config.json"`).
- `core/config.py` là file ngoài danh sách PLAN §2 — chấp nhận (cần cho N13);
  lần sau ghi vào mục "Lệch so với PLAN".

## Lưu ý khi làm Phase 2 (GUI) — theo PLAN §7

1. **Không treo UI**: mọi xử lý (preview video khung đầu, chạy batch, dò
   encoder) chạy trong worker thread; UI chỉ nhận tin qua `queue.Queue` +
   `after(50)`. Không gọi `subprocess` trên luồng Tk.
2. **Huỷ & đóng cửa sổ**: nút Huỷ set `threading.Event`; đóng cửa sổ khi đang
   chạy ⇒ hỏi xác nhận, set cancel, chờ worker xong rồi mới `destroy()` — không
   để ffmpeg mồ côi.
3. **Preset GUI ↔ CLI**: GUI chỉ được lưu/mở bằng `save_preset`/`load_preset`.
   Box vẽ trên canvas lưu dạng `relative` (dùng `to_rel_box`).
4. **Không nút trang trí**: control nào chỉ áp dụng cho một method (radius,
   blur_ksize, mosaic_block, màu, overlay) thì ẩn/disable khi method khác.
   `clone_smooth` + mode fast ⇒ báo ngay trên UI, không đợi lúc chạy.
5. **Lỗi hiện cho người dùng**: thiếu ffmpeg, preset sai, file lỗi ⇒ hộp thoại /
   dòng trạng thái rõ ràng, không chỉ ghi log.
6. **Test được không cần màn hình**: tách logic trạng thái (danh sách box,
   method, params → preset) ra lớp thuần không import tkinter, có pytest riêng.
   Phần tkinter nghiệm thu bằng ảnh chụp.
7. **Cổng Phase 2 (PLAN §8)**: ảnh chụp mở ảnh → vẽ 2 box → xem trước → lưu
   preset; dùng chính preset đó chạy `cli.py` ⇒ hash ảnh ra **giống hệt** ảnh
   GUI xuất. Thêm ảnh chụp đang chạy video có thanh tiến độ và bấm Huỷ.
