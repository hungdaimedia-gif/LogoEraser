# Review Phase 1a — lượt 3 — commit ac7bed4

**Kết luận: ĐẠT. Mở Phase 1b.**

Kiểm lại trên máy: 44 passed. Logo chạm góc dưới-phải đã xoá sạch, không còn
vệt vàng (clone_smooth 20.7%, ngang inpaint). N1 (docstring trung thực), N7
(chỉ nhận `int` thật) đã xong.

Ghi nhận, không cần sửa: logo cách mép 8px, clone_smooth từ 4.6% lên 9.2% do
bước pre-inpaint — vẫn tốt nhất nhóm, chấp nhận.

## Lưu ý khi làm 1b (theo PLAN §5–§6)

1. Chế độ fast với `clone_smooth` ⇒ báo lỗi "dùng chế độ quality", không tự đổi method.
2. `delogo` của FFmpeg báo lỗi khi box chạm mép khung ⇒ kẹp box cách mép ≥ 1px
   **trong video_engine**, không sửa `to_abs_box`. Test với box chạm góc
   (cùng 4 vị trí như cổng ảnh).
3. Quality pipe: đọc stderr bằng thread riêng; cổng 1b phải có 1 test video
   ≥ 300 khung để chứng minh không bị treo khi buffer stderr đầy.
4. Lỗi ở `apply()` (ví dụ overlay thiếu file) phải dừng job, dọn file output dở,
   và CLI trả exit code 1 — **không** được ra video rỗng/hỏng mà báo thành công.
5. Đọc/ghi ảnh trong CLI bằng `np.fromfile` + `imdecode` / `imencode().tofile`
   (đường dẫn tiếng Việt). Test với thư mục `mẫu thử/`.
6. Mẫu bàn giao: thêm dòng "Mục REVIEW chưa làm: …" (để trống nếu không có).
