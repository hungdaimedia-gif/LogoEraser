# LogoEraser — Kế hoạch thi công (Claude lập, AG3 thi công)

> Công cụ xoá/che logo trên ảnh & video, chạy 100% offline, viết **mới từ đầu**
> bằng OpenCV + FFmpeg. Đây **không phải** bản port của DeLogo.

## 0. Luật phòng sạch (clean-room) — đọc trước tiên

1. **Cấm mở, đọc, copy** bất cứ thứ gì trong:
   - `C:\Users\Admim\.gemini\antigravity-ide\brain\...\DeLogo.exe_extracted`
   - `G:\DeLogo\` (kể cả `_internal/`, icon, chuỗi giao diện)
2. Nguồn được phép tham khảo: **chỉ file PLAN.md này** + tài liệu công khai
   (docs OpenCV, docs FFmpeg filters, docs Tkinter).
3. Không dùng tên "DeLogo", tên hàm/biến, chuỗi UI, icon hay layout của bản gốc.
4. FFmpeg: dùng bản trong `PATH`, hoặc cấu hình đường dẫn trong `config.json`.
   Không trỏ cứng vào `G:\DeLogo\tools`.
5. Không có mã mạng nào: cấm `socket`, `urllib`, `http`, `requests`, `webbrowser`,
   và cấm gọi `powershell`/`wmic`. `subprocess` chỉ được gọi `ffmpeg`/`ffprobe`.

Nếu điều 1 không giữ được (ví dụ đã đọc bản decompile để "đối chiếu"), thì
dừng lại: kế hoạch này không còn hiệu lực.

## 1. Mục tiêu đo được

- 7 phương thức xử lý ảnh (mục 3), dùng chung cho ảnh và video chế độ Quality.
- Video có 2 chế độ: **fast** (filtergraph FFmpeg) và **quality** (pipe từng
  khung qua OpenCV), tự dò encoder phần cứng.
- Preset JSON dùng được cho cả GUI lẫn CLI, không cần sửa gì.
- Chạy được bằng `python main.py` (GUI) và `python cli.py` (hàng loạt).

## 2. Phạm vi file (chỉ được tạo/sửa những file này)

```
G:\LogoEraser\
├── core/__init__.py
├── core/preset.py         # schema + load/save/validate preset, chuyển box relative↔absolute
├── core/image_engine.py   # 7 phương thức + build_mask + detect_logo_boxes
├── core/video_engine.py   # ffprobe, dò encoder, fast filtergraph, quality pipe
├── gui/__init__.py
├── gui/theme.py
├── gui/canvas.py
├── gui/app.py
├── cli.py
├── main.py
├── config.json            # đường dẫn ffmpeg/ffprobe (tuỳ chọn), thư mục output mặc định
├── requirements.txt       # opencv-python, numpy, pillow — KHÔNG thêm gì khác
├── tests/make_samples.py  # sinh ảnh/video mẫu có logo giả
├── tests/test_*.py        # pytest (pytest chỉ là dev-dependency)
├── README.md
└── run.bat
```

## 3. Lõi ảnh — `core/image_engine.py`

Mọi hàm đều **thuần**: nhận `np.ndarray` BGR uint8, trả mảng mới, không I/O.

```python
def build_mask(shape, boxes_abs, dilate: int, feather: int) -> np.ndarray  # uint8 0..255
def apply(img, boxes_abs, method: str, params: dict) -> np.ndarray
def detect_logo_boxes(frames: list[np.ndarray], max_boxes=3) -> list[Box]
```

| `method` | Cách làm |
|---|---|
| `inpaint_fast` | `cv2.inpaint(..., cv2.INPAINT_TELEA)`, bán kính `radius` |
| `inpaint_hq` | Chạy NS + Telea, trộn theo trọng số; vùng lớn thì inpaint ở 1/2 độ phân giải rồi phóng lên và trộn bằng mask có feather |
| `clone_smooth` | Lấy dải ảnh kế bên (ưu tiên phía có phương sai thấp nhất trong 4 phía), đắp vào bằng `cv2.seamlessClone(NORMAL_CLONE)` |
| `blur` | `GaussianBlur` kích thước `blur_ksize` (ép lẻ), trộn bằng mask có feather |
| `mosaic` | Thu nhỏ theo `mosaic_block` (INTER_LINEAR), phóng lại (INTER_NEAREST) |
| `solid` | Tô `solid_color` (#RRGGBB) |
| `overlay` | Dán PNG `overlay_path` có alpha, co giãn vừa box, nhân `overlay_opacity` |

`detect_logo_boxes`: lấy độ lệch chuẩn theo thời gian của ≥8 khung (ảnh xám) →
vùng có std thấp **và** mật độ cạnh Canny cao → morphology close → contour →
lấy top-N theo diện tích, bỏ box > 25% khung hình. Với ảnh đơn thì không dò,
trả `[]`.

Mọi `method` không hợp lệ ⇒ `ValueError`, không âm thầm rơi về mặc định.

## 4. Preset — `core/preset.py`

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

- `unit`: `relative` (0..1 theo khung hình) hoặc `absolute` (pixel).
  `to_abs(box, w, h)` kẹp box vào trong khung hình, w/h ≥ 2.
- Key lạ ⇒ bỏ qua và ghi warning; thiếu key ⇒ lấy mặc định; sai kiểu ⇒ lỗi rõ ràng.
- GUI lưu preset bằng **đúng** hàm `save_preset` này (để đảm bảo GUI/CLI tương thích).

## 5. Lõi video — `core/video_engine.py`

```python
def probe(path) -> VideoInfo          # w, h, fps, nb_frames, has_audio  (ffprobe -of json)
def detect_encoder(ffmpeg) -> str     # cache theo phiên
def process_video(src, dst, preset, progress_cb, cancel: threading.Event)
```

**Dò encoder**: thứ tự `h264_nvenc` → `h264_qsv` → `h264_amf` → `libx264`. Với
mỗi ứng viên, chạy thử mã hoá
`-f lavfi -i color=black:s=256x256:d=0.2 -c:v <enc> -f null -`; ứng viên nào
exit 0 đầu tiên thì chọn. Không chỉ dựa vào danh sách `-encoders`, vì có trong
danh sách chưa chắc có driver.

**Chế độ fast** (một lệnh ffmpeg duy nhất, ghép filter cho từng box):

| method | filter |
|---|---|
| `inpaint_fast`, `inpaint_hq` | `delogo=x:y:w:h` (box phải cách mép ≥1px; kẹp lại nếu cần) |
| `blur` | `split` → `crop` → `boxblur` → `overlay` |
| `mosaic` | `crop` → `scale=iw/N:ih/N` → `scale=…:flags=neighbor` → `overlay` |
| `solid` | `drawbox=…:t=fill` |
| `overlay` | input PNG thứ 2 → `scale` → `overlay` |
| `clone_smooth` | **không hỗ trợ** ⇒ báo lỗi "dùng chế độ quality". Không tự đổi method |

Tiến độ: `-progress pipe:1`, đọc `out_time_ms` chia cho thời lượng.

**Chế độ quality**:
- Đọc: `ffmpeg -i src -f rawvideo -pix_fmt bgr24 -` → đọc đúng `w*h*3` byte/khung.
- Ghi: `ffmpeg -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i - [-i src -map 0:v -map 1:a? -c:a copy] -c:v ENC -pix_fmt yuv420p dst`.
- Mask tính **một lần** trước vòng lặp.
- stderr của cả 2 process phải đọc bằng thread riêng (hoặc ghi ra file tạm),
  nếu không sẽ treo khi buffer đầy.
- Hủy: `cancel.is_set()` ⇒ đóng stdin, `terminate()` cả 2, xoá file output dở
  (đây là file do chính tool vừa tạo trong phiên, không đụng file nào khác).
- Hết khung trước `nb_frames` thì không coi là lỗi. Exit code ≠ 0 thì lỗi kèm
  20 dòng stderr cuối.
- Windows: `creationflags=CREATE_NO_WINDOW`.

## 6. CLI — `cli.py`

```
python cli.py --preset p.json --input <file|thư mục> --output <thư mục>
              [--mode fast|quality] [--method ...] [--recursive] [--overwrite]
```
- Ảnh: `.png .jpg .jpeg .webp .bmp`. Video: `.mp4 .mov .mkv .webm .avi`.
- Đọc/ghi ảnh bằng `cv2.imdecode(np.fromfile(...))` / `imencode(...).tofile(...)`
  để không vỡ khi đường dẫn có tiếng Việt.
- Không ghi đè nếu thiếu `--overwrite`; output mặc định là `<tên>_clean.<ext>`.
- Exit code: 0 = tất cả OK, 1 = có file lỗi (in danh sách), 2 = preset sai.

## 7. GUI — `gui/`

- `theme.py`: bảng màu tối dùng `ttk.Style`; thanh tiêu đề tối qua
  `DwmSetWindowAttribute(DWMWA_USE_IMMERSIVE_DARK_MODE=20)` (`ctypes`); lỗi thì bỏ qua.
- `canvas.py`: hiển thị ảnh/khung video đầu tiên, zoom bằng con lăn (giữ điểm
  dưới con trỏ), pan bằng chuột giữa/Space+kéo, vẽ box mới bằng kéo, 8 tay nắm
  resize, Delete xoá **box đang chọn trên canvas** (chỉ là dữ liệu trong bộ nhớ),
  nút "Xem trước" gọi `image_engine.apply` trên ảnh hiển thị (debounce 150ms).
- `app.py`: danh sách file (thêm/bớt khỏi danh sách, **không** xoá file trên
  đĩa), ô chọn method + tham số, chọn mode video, Lưu/Mở preset, nút Chạy và Hủy,
  thanh tiến độ. Worker chạy trong `threading.Thread`, gửi tiến độ qua
  `queue.Queue`, GUI poll bằng `after(50)`.
- Mọi control phải gắn với một tham số có thật trong preset (không có nút trang trí).
  Không có màn hình đăng nhập, popup hay banner.

## 8. Thứ tự thi công & cổng nghiệm thu

| Phase | Giao | Cổng (AG3 dán output thật vào bàn giao) |
|---|---|---|
| 1a | `preset.py`, `image_engine.py`, `tests/make_samples.py`, `tests/test_image.py` | `pytest -q` pass: đủ 7 method chạy được; với `inpaint_hq`, sai lệch trung bình trong vùng logo so với ảnh nền sạch nhỏ hơn 50% so với ảnh có logo |
| 1b | `video_engine.py`, `cli.py`, `tests/test_video.py` | Video mẫu 3 giây có âm thanh: fast + quality đều ra file, `probe(out)` cùng w/h/fps, số khung ±1, còn audio. In ra encoder đã dò được |
| 2 | `gui/*`, `main.py` | Chụp màn hình: mở ảnh, vẽ 2 box, xem trước, lưu preset; dùng chính preset đó chạy `cli.py` ra kết quả giống hệt (so hash ảnh) |
| 3 | `README.md`, `run.bat` | Máy chưa có venv: `run.bat` tạo venv, cài requirements, mở GUI |

**Cổng offline** (chạy ở mọi phase): `tests/test_no_network.py` dùng `ast` quét
toàn bộ `.py`, fail nếu có import `socket|urllib|http|requests|webbrowser`
hoặc chuỗi `powershell|wmic`.

## 9. Ranh giới cấm

- Không ghi/xoá bất cứ thứ gì ngoài `G:\LogoEraser\` và thư mục output người dùng chọn.
- Không đụng `G:\DeLogo\` và thư mục decompile (xem mục 0).
- Không lệnh xoá ngoài trường hợp hủy job ở mục 5.
- Không thêm dependency ngoài `requirements.txt`.

## 10. Mẫu bàn giao (AG3 → Claude)

```
Phase: 1a | Commit/thư mục: ...
File đã tạo/sửa: ...
Output cổng (dán nguyên văn): ...
Lệch so với PLAN (nếu có) + lý do: ...
Cam kết clean-room: Không mở thư mục decompile / G:\DeLogo trong phase này: CÓ/KHÔNG
```
