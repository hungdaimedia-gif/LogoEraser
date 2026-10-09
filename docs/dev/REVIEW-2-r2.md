# Phase 2 + 3 — hoàn thiện bởi Claude (2026-10-09)

Người dùng giao Claude tự hoàn thiện thay vì trả lại AG3. Các mục của REVIEW-2:

| Mục | Xử lý |
|---|---|
| C1 chưa commit | Commit cùng bản sửa |
| C2 cổng test vòng tròn | `tests/test_gui_e2e.py`: dựng `LogoEraserApp` thật, vẽ box bằng `event_generate`, chạy/huỷ job qua nút thật, so hash với `cli.py`; đóng cửa sổ khi đang chạy phải chờ worker dừng |
| C3 mờ trên màn 4K | `main.enable_dpi_awareness()` + `LogoEraserApp.ui_scale` nhân kích thước pixel; kẹp cửa sổ theo kích thước màn hình |
| N17 chữ "Kéo thả" | Bỏ |
| N18 đơ khi mở video | `extract_frame` chạy luồng phụ, poll bằng `after` |
| N19 hai đường batch | `core/batch.py` dùng chung cho CLI và GUI |
| N20 đóng cửa sổ trần 5s | Chờ worker dừng hẳn rồi mới `destroy()` |
| N21 hằng số trùng | Đuôi file + danh sách method chỉ định nghĩa ở `core/preset.py` |

Lỗi phát hiện thêm khi sửa:
- Ô "Bán kính inpaint" ghi vào key `inpaint_radius` — preset không có key này nên
  `validate_preset` bỏ qua ⇒ ô không có tác dụng. Đổi về `radius`; `GuiState.set_param`
  giờ từ chối key lạ.
- Tham số chỉ cập nhật khi `<FocusOut>` ⇒ bấm mũi tên Spinbox rồi Chạy ngay thì giá trị
  mới không vào preset. Đổi sang `trace_add("write")` (cả ô CRF).
- `dilate`/`feather` bị ẩn với blur/mosaic/solid dù ba method này dùng mask.
- Ô chọn method lấy từ `set` ⇒ thứ tự đổi mỗi lần mở app. Đổi sang tuple cố định.
- "Đặt lại về mặc định" xoá luôn danh sách file và không cập nhật các ô tham số.
- `config.json` với bản exe: đọc cạnh `LogoEraser.exe`; ffmpeg tự tìm trong
  PATH → `ffmpeg\bin\` cạnh ứng dụng; thiếu ffmpeg báo tiếng Việt rõ ràng.
  `default_output_dir` trước đây không được dùng — nay là thư mục xuất mặc định.

Phase 3: `README.md`, `run.bat`, `build_exe.bat` (chạy test trước khi đóng gói).
