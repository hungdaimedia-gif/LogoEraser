# LogoEraser

Công cụ xoá / che logo và watermark **nhìn thấy được** trên ảnh và video. Chạy 100% offline trên Windows: không tài khoản, không quảng cáo, không gửi dữ liệu đi đâu.

Có hai cách dùng, dùng chung một file preset `.json`:
- **Giao diện (GUI):** vẽ vùng logo bằng chuột, xem trước, xử lý hàng loạt.
- **Dòng lệnh (CLI):** xử lý cả thư mục theo preset.

> Công cụ chỉ xoá phần logo nhìn thấy được. Watermark ẩn như SynthID hay chữ ký C2PA nằm trong dữ liệu ảnh nên không xoá được. Chỉ dùng cho nội dung bạn có quyền chỉnh sửa.

## Cài đặt

### Cách 1: Bản exe (không cần Python)
1. Tải `LogoEraser-<phiên bản>-win64.zip` ở mục Releases rồi giải nén.
2. Chạy `LogoEraser.exe`.
3. Muốn xử lý **video** thì cần thêm FFmpeg (xem mục [FFmpeg](#ffmpeg)). Ảnh thì không cần.

### Cách 2: Chạy từ mã nguồn
Cần Python 3.10 trở lên.

```bat
git clone https://github.com/hungdaimedia-gif/LogoEraser.git
cd LogoEraser
run.bat
```

Lần chạy đầu, `run.bat` tự tạo môi trường ảo `.venv` và cài `opencv-python`, `numpy`, `pillow`. Đây là lần duy nhất cần mạng.

### FFmpeg
Cần cho xử lý video. Ứng dụng tìm theo thứ tự sau:
1. Đường dẫn ghi trong `config.json` (`"ffmpeg"`, `"ffprobe"`).
2. `ffmpeg` / `ffprobe` có trong `PATH`.
3. Thư mục `ffmpeg\bin\` (hoặc `ffmpeg\`) đặt cạnh `LogoEraser.exe` hoặc cạnh `main.py`.

Bản build Windows tải ở https://www.gyan.dev/ffmpeg/builds/ hoặc https://github.com/BtbN/FFmpeg-Builds/releases.

## Dùng giao diện

1. Bấm **Thêm tệp...** (Ctrl+O) hoặc **Thêm thư mục...** để đưa ảnh/video vào danh sách. Với video, khung hình đầu tiên được hiện ra để vẽ vùng.
2. **Kéo chuột trái** trên ảnh để vẽ vùng logo. Có thể vẽ nhiều vùng.
   - Kéo 8 tay nắm để chỉnh kích thước, kéo phần giữa để di chuyển.
   - Nhấn **Delete** để xoá vùng đang chọn.
   - Lăn chuột để phóng to/thu nhỏ. Kéo chuột giữa (hoặc giữ Space rồi kéo) để di chuyển khung nhìn.
3. Chọn **phương thức** và chỉnh tham số. Chỉ những tham số dùng được cho phương thức đang chọn mới hiện ra.
4. Bấm **👁 Xem trước** để xem kết quả ngay trên ảnh.
5. Chọn **Thư mục xuất**. Để trống thì file kết quả được ghi cạnh file gốc với tên `<tên>_clean.<đuôi>`.
6. Bấm **▶ BẮT ĐẦU XỬ LÝ**. Bấm **⏹ HUỶ** để dừng; file đang xử lý dở sẽ bị dọn, không để lại file hỏng.

Menu **Preset** dùng để lưu/mở cấu hình (Ctrl+S / Ctrl+L). File preset này dùng được luôn cho CLI.

## Các phương thức

| Phương thức | Cách làm | Hợp với |
|---|---|---|
| `inpaint_hq` | Trộn hai thuật toán inpaint của OpenCV (Telea + Navier–Stokes). Vùng lớn có thêm một lượt xử lý ở nửa độ phân giải. | Mặc định, logo trên nền mượt |
| `inpaint_fast` | Inpaint Telea, nhanh hơn | Logo nhỏ, video dài |
| `clone_smooth` | Lấy mảng nền bên cạnh dán đè (seamless clone) | Nền có vân lặp (tường, vải, cỏ) |
| `blur` | Làm mờ Gaussian trong vùng | Che nhanh |
| `mosaic` | Làm vỡ hạt (pixelate) | Che kiểu kiểm duyệt |
| `solid` | Tô một màu đặc `#RRGGBB` | Khung viền, nền đơn sắc |
| `overlay` | Dán ảnh PNG có nền trong suốt đè lên | Thay bằng logo của bạn |

Tham số chung: `dilate` nới rộng vùng ra thêm vài pixel, `feather` làm mềm mép vùng.

## Video: hai chế độ

| Chế độ | Cách chạy | Ghi chú |
|---|---|---|
| **fast** | Một lệnh FFmpeg với bộ lọc (`delogo`, `boxblur`, `drawbox`, `overlay`) | Rất nhanh. `inpaint_*` dùng `delogo` của FFmpeg. **Không hỗ trợ `clone_smooth`.** |
| **quality** | Đọc từng khung hình, xử lý bằng đúng thuật toán của ảnh, rồi mã hoá lại | Chậm hơn nhưng kết quả giống hệt xử lý ảnh |

Bộ mã hoá `auto` tự dò theo thứ tự NVIDIA NVENC → Intel QSV → AMD AMF → libx264, chạy thử thật trước khi chọn. Âm thanh được giữ nguyên (copy). `crf` (0–51, càng nhỏ càng đẹp) được áp dụng cho cả bộ mã hoá phần cứng.

## Dòng lệnh

```bat
.venv\Scripts\python cli.py --preset preset.json --input <file hoặc thư mục> --output <thư mục>\
```

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--preset` | File preset `.json` (bắt buộc) |
| `--input` | Một file, hoặc một thư mục |
| `--output` | Thư mục xuất (thêm `\` ở cuối khi input là một file), hoặc đường dẫn file đích |
| `--mode fast\|quality` | Ghi đè chế độ video trong preset |
| `--method <tên>` | Ghi đè phương thức trong preset |
| `--recursive`, `-r` | Quét cả thư mục con |
| `--overwrite` | Cho phép ghi đè file kết quả đã có |

Mã thoát: `0` tất cả thành công · `1` có file lỗi (in danh sách) · `2` preset sai.
File `*_clean.*` từ lần chạy trước được tự bỏ qua.

## File preset

```json
{
  "version": 1,
  "boxes": [{"x": 0.86, "y": 0.90, "w": 0.12, "h": 0.07, "unit": "relative"}],
  "method": "inpaint_hq",
  "params": {
    "dilate": 3, "feather": 4, "radius": 5,
    "blur_ksize": 31, "mosaic_block": 12,
    "solid_color": "#000000",
    "overlay_path": null, "overlay_opacity": 1.0
  },
  "video": {"mode": "fast", "encoder": "auto", "crf": 20, "copy_audio": true}
}
```

- `unit: "relative"`: toạ độ là tỉ lệ 0–1 theo khung hình, nên một preset dùng được cho mọi độ phân giải. `"absolute"`: toạ độ tính bằng pixel.
- Preset sai (giá trị âm, sai kiểu, phương thức không tồn tại…) sẽ bị từ chối kèm thông báo rõ ràng, không âm thầm dùng giá trị mặc định.

## config.json

Đặt cạnh `main.py`, hoặc cạnh `LogoEraser.exe` với bản exe.

```json
{ "ffmpeg": "ffmpeg", "ffprobe": "ffprobe", "default_output_dir": "" }
```

`default_output_dir` là thư mục xuất mặc định hiện sẵn trên giao diện.

## Phát triển

```bat
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
build_exe.bat
```

- `tests/test_gui_e2e.py` dựng giao diện Tkinter thật và giả lập chuột, nên sẽ thấy một cửa sổ chớp lên trong lúc test. Máy không có màn hình thì test này tự bỏ qua.
- `tests/test_no_network.py` quét toàn bộ mã nguồn, báo lỗi nếu có import thư viện mạng (`socket`, `urllib`, `http`, `requests`…).
- `build_exe.bat` chạy test trước, test đỏ thì không đóng gói. Kết quả nằm ở `release\LogoEraser\`. Muốn đóng gói kèm FFmpeg thì đặt FFmpeg vào `tools\ffmpeg\bin\` trước khi build.

Cấu trúc:

```
core/      preset, image_engine (7 phương thức), video_engine (FFmpeg), batch (dùng chung CLI + GUI), config
gui/       app (cửa sổ chính), canvas (vẽ vùng, zoom, xem trước), state (logic thuần, không phụ thuộc Tk), theme
cli.py     dòng lệnh
main.py    mở giao diện
docs/dev/  kế hoạch và biên bản review của quá trình phát triển
```
