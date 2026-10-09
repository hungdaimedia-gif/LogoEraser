# Review Phase 2 (GUI) — bản chưa commit trên f4577ef

**Kết luận: CHƯA ĐẠT — chức năng chạy được, nhưng 3 điểm chặn về quy trình/hiển thị.**

## Claude đã tự mở và điều khiển GUI thật

AG3 báo không mở được cửa sổ từ runner nền. Claude chạy được: script gọi
`LogoEraserApp` thật, dùng `root.after(...)` để lần lượt nạp file, giả lập
chuột bằng `canvas.event_generate("<ButtonPress-1>"/"<B1-Motion>"/"<ButtonRelease-1>")`,
chụp cửa sổ bằng `PIL.ImageGrab` rồi tự `destroy()`. Không cần người bấm.
(Script mẫu: `F:\Temp\Temp\claude\G--DSHarness-hungdaitool\faedd3b5-b651-43da-a4ae-81d7e09434a0\scratchpad\gui_e2e.py`
và `gui_parity.py`.)

| Bước trên GUI thật | Kết quả |
|---|---|
| Mở cửa sổ, không lỗi callback Tk | ✅ |
| Nạp ảnh tên có dấu `mẫu ảnh.png`, vẽ 2 box bằng chuột | ✅ state có 2 box |
| Xem trước | ✅ logo biến mất trong box #1 |
| Chạy video 30s chế độ quality | ✅ thanh tiến độ chạy, UI không đơ |
| Bấm Huỷ ở ~10% | ✅ worker dừng, hộp thoại "Đã huỷ", không còn file dở |
| Lưu preset từ GUI → chạy `cli.py` với preset đó | ✅ hash ảnh GUI xuất = CLI xuất (`0292a7be…`) |

## Chặn

### C1. Chưa commit
`gui/`, `main.py`, `tests/test_gui_*.py`, `scripts/` đều untracked;
`core/config.py`, `core/preset.py`, `core/video_engine.py` đang sửa dở. Bàn giao
phải trỏ tới một commit. Lưu ý `scripts/samples/*.png` là file sinh ra — cho vào
`.gitignore`, không commit.

### C2. Cổng Phase 2 trong test là vòng tròn
`tests/test_phase2_gate.py` không import `gui.app` hay `gui.canvas`; "ảnh GUI"
được tính bằng `image_engine.apply` trực tiếp (dòng ~65) rồi so với CLI — tức là
so `apply` với chính nó. 1.500 dòng tkinter không có dòng nào được test chạy.
Sửa: thêm `tests/test_gui_e2e.py` dựng `LogoEraserApp` thật như cách ở trên
(nạp ảnh → vẽ 2 box bằng `event_generate` → lưu preset → chạy job qua
`_on_start_processing` → so hash với `cli.py`; và một ca video + Huỷ giữa chừng).
Đánh dấu `@pytest.mark.gui`, tự skip nếu không có display.

### C3. Mờ trên màn hình của người dùng (4K, scale 192%)
App không khai báo DPI-aware ⇒ Windows phóng to kiểu bitmap, chữ nhoè (thấy rõ ở
ảnh chụp không bật DPI). Trong `main.py`, trước `tk.Tk()`:
`ctypes.windll.shcore.SetProcessDpiAwareness(1)` (bọc try/except), sau đó
`root.tk.call("tk", "scaling", dpi/72)` để chữ/nút có kích thước đúng. Kiểm bằng
ảnh chụp ở 192%.

## Không chặn

- **N17 Chữ "Kéo thả" là trang trí**: `gui/canvas.py:166` ghi "(Kéo thả hoặc nhấn
  'Thêm tệp'…)" nhưng không có code kéo-thả nào (tkinter không hỗ trợ sẵn). Bỏ
  chữ "Kéo thả" (hoặc làm thật — nhưng sẽ phải thêm dependency, ngoài PLAN).
- **N18 Đơ UI khi mở video**: `_load_file_to_canvas` gọi `extract_frame`
  (ffprobe + ffmpeg) ngay trên luồng Tk — vi phạm lưu ý 1 của REVIEW-1b-r2.
  Video lớn/ổ mạng sẽ đơ vài giây. Đưa vào worker, hiện "Đang đọc khung đầu…".
- **N19 Hai đường batch**: `_worker_run` tự viết lại vòng lặp của CLI nên lệch:
  không bỏ qua file `*_clean.*` (N15), mặc định xuất cạnh file nguồn còn CLI bắt
  buộc `--output`. Tách hàm batch chung trong `core/` (ví dụ `core/batch.py`)
  để CLI và GUI cùng gọi.
- **N20** Đóng cửa sổ khi đang chạy: `join(timeout=5.0)` rồi `destroy()` —
  `process_video` có thể cần tới 5s/tiến trình để terminate→kill; nếu worker
  chưa xong thì ffmpeg có thể còn sống. Chờ tới khi worker xong (hiện trạng thái
  "Đang dừng…", poll bằng `after`), không đặt trần 5s.
- **N21** `core/preset.py` thêm `SUPPORTED_METHODS = VALID_METHODS` (bí danh thừa)
  và `IMAGE/VIDEO_EXTENSIONS` trùng với `cli.py`. Giữ một nơi định nghĩa.

## Bàn giao lại

Mẫu PLAN §10 + "Mục REVIEW chưa làm" + "Lệch so với PLAN" (ghi `gui/state.py`,
`core/config.py`, `scripts/`). Kèm ảnh chụp ở scale 192% sau khi sửa C3.
