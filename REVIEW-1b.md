# Review Phase 1b — commit b973097

**Kết luận: CHƯA ĐẠT — 2 lỗi chặn.** Chưa mở Phase 2.

Chạy lại trên máy: 56 passed (ffmpeg gyan.dev 2026-03, encoder dò ra
`h264_nvenc`). Nhưng bộ test không phủ 2 đường hỏng dưới đây. Script thử của
Claude: video mẫu 320×240, 3s, logo (240,190,70,40).

| Thử | Kết quả |
|---|---|
| fast · inpaint_fast / solid / overlay | OK, 75 khung |
| **fast · blur** | **FAIL** `Error binding filtergraph inputs/outputs: Invalid argument` |
| **fast · mosaic** | **FAIL** cùng lỗi |
| **quality · huỷ ở khung 30 (video 20s)** | **TREO > 25s**, file dở còn trên đĩa |

## B5 (chặn) — fast `blur`/`mosaic` luôn lỗi — `core/video_engine.py:312`, `:329`

```python
fc_parts.append(f"[{cur_v}]split[{m_out}][c_out]")   # thiếu {} quanh c_out
```
Nhãn sinh ra là chữ `c_out` theo nghĩa đen, còn dòng sau đọc `[c0]` ⇒ FFmpeg
không nối được graph. Sửa thành `[{c_out}]` ở cả hai chỗ.

Test bắt buộc: tham số hoá test fast cho **đủ 6 method** (trừ `clone_smooth`
đã có test riêng), kiểm ra file + đủ số khung; với 2 box cùng lúc cho
blur/mosaic/overlay (để kiểm chuỗi nhãn `v1 → v2`).

## B6 (chặn) — huỷ giữa chừng ở chế độ quality làm treo vĩnh viễn — `:526-572`

Khi `cancel` được set (hoặc `BrokenPipeError` ở `:555`), vòng lặp `break` rồi
`finally` gọi `dec_proc.wait()`. Nhưng decoder vẫn đang ghi khung vào
`stdout` mà không ai đọc nữa ⇒ pipe đầy ⇒ decoder chặn ⇒ `wait()` không bao
giờ trả về. GUI ở Phase 2 bấm Huỷ sẽ treo luôn ứng dụng.

`test_cancel_video_processing` không bắt được vì set `cancel` **trước** khi
chạy (vòng lặp thoát trước khi decoder kịp ghi).

Cách sửa: trong `finally`, nếu thoát sớm (huỷ / lỗi / broken pipe) thì
`dec_proc.terminate()` (và `enc_proc.terminate()` khi huỷ) **trước** `wait()`;
`wait(timeout=10)` rồi `kill()` nếu còn sống. Đóng `dec_proc.stdout`.

Test bắt buộc: video ≥ 10s, huỷ từ `progress_cb` ở khung 30, chạy trong
thread với `join(timeout=15)` ⇒ phải kết thúc, raise "cancelled", file dst
không còn, không còn tiến trình ffmpeg con. Làm tương tự cho chế độ fast
(huỷ khi đang chạy, không phải trước).

## Không chặn, nên sửa trong lượt này

- **N8 Decoder lỗi bị bỏ qua** (`:579`): chỉ kiểm `enc_proc.returncode`. Nếu
  decoder chết giữa chừng (file hỏng phần giữa), vòng lặp gặp EOF, encoder đóng
  file sạch ⇒ báo **thành công** với video bị cụt. Kiểm thêm
  `dec_proc.returncode != 0` ⇒ lỗi, và cảnh báo khi `frame_idx` lệch
  `nb_frames` quá 2%.
- **N9 `crf` vô tác dụng với encoder phần cứng** (`:364`, `:486`): máy có NVIDIA
  thì `auto` chọn nvenc và `crf` trong preset bị bỏ qua — đúng kiểu "nút trang
  trí". Ánh xạ: nvenc `-rc vbr -cq <crf>`, qsv `-global_quality <crf>`, amf
  `-rc cqp -qp_i <crf> -qp_p <crf>`. Ghi vào docstring.
- **N10 Mask không tính trước** ở quality: `apply()` dựng lại mask + `to_abs_box`
  mỗi khung (PLAN §5: "Mask tính một lần"). Tách hàm `apply_with_mask` hoặc cache.
- **N11 Tiến độ fast**: `out_time_ms` của FFmpeg thực chất là micro-giây; heuristic
  `ms > 1_000_000` sai trong giây đầu. Đọc `out_time_us` (hoặc luôn chia 1e6).
- **N12 Ghi đè làm mất file cũ khi lỗi**: với `--overwrite`, nếu job lỗi thì
  `os.remove(dst)` xoá luôn kết quả cũ của người dùng (`cli.py:190`,
  `video_engine.py:416/421/549/576/581`). Ghi ra `dst + ".part"` rồi
  `os.replace` khi thành công; khi lỗi chỉ dọn file `.part` do chính job tạo.
- **N13 `config.json` chưa có** (PLAN §0.4, §2): CLI chỉ dùng `ffmpeg` trong
  PATH. Thêm đọc `config.json` (`ffmpeg`, `ffprobe`) + truyền xuống
  `process_video`. Bàn giao ghi "Lệch PLAN: Không có" — mục này là lệch.
- **N14** `-r str(info.fps)` (`:473`): truyền nguyên chuỗi `r_frame_rate`
  (ví dụ `30000/1001`) thay vì float để không trôi tiếng với video 29.97.
- **N15** Chạy CLI hai lần vào cùng thư mục: file `*_clean.*` lần trước bị xử lý
  thành `*_clean_clean.*`. Bỏ qua file có hậu tố `_clean` khi input == output.

## Đạt

- `clone_smooth` ở fast báo lỗi đúng; delogo kẹp mép đúng ở 4 vị trí.
- Quality 320 khung không treo (khi chạy hết, không huỷ); 3s giữ audio, đúng
  w/h/fps/số khung.
- Lỗi `apply()` giữa chừng dừng job; CLI exit 0/1/2 đúng; đường dẫn tiếng Việt OK.
- Không có import mạng.

## Bàn giao lại

Mẫu PLAN §10 + dòng "Mục REVIEW chưa làm". Kèm output của script/test huỷ giữa
chừng (thời gian kết thúc) và bảng fast cho đủ 6 method.
