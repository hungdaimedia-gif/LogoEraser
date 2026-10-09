@echo off
rem Dong goi LogoEraser thanh LogoEraser.exe (PyInstaller, dang thu muc).
rem Ket qua: release\LogoEraser\LogoEraser.exe  (gui ca thu muc release\LogoEraser cho nguoi dung)
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Chua co .venv - hay chay run.bat mot lan truoc.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller --version >nul 2>nul || ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt

rem Chay test truoc khi dong goi - test do thi khong build.
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 (
    echo Test that bai - huy dong goi.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -m PyInstaller main.py ^
    --name LogoEraser ^
    --onedir --windowed --noconfirm ^
    --collect-binaries onnxruntime --collect-binaries nvidia ^
    --exclude-module onnxruntime.transformers --exclude-module onnxruntime.quantization ^
    --distpath release --workpath build --specpath build
if errorlevel 1 (
    echo PyInstaller that bai.
    pause
    exit /b 1
)

rem config.json dat CANH exe de nguoi dung sua duoc duong dan ffmpeg.
copy /y config.json release\LogoEraser\config.json >nul
copy /y README.md release\LogoEraser\README.md >nul
copy /y download_model.bat release\LogoEraser\download_model.bat >nul
copy /y LICENSE release\LogoEraser\LICENSE >nul

rem Model AI (inpaint_ai) dat canh exe, neu da tai.
if exist "models\lama_fp32.onnx" (
    if not exist "release\LogoEraser\models" mkdir "release\LogoEraser\models"
    copy /y "models\lama_fp32.onnx" "release\LogoEraser\models\" >nul
    echo Da dong goi kem model AI.
)

rem Neu co thu muc tools\ffmpeg (chua bin\ffmpeg.exe, bin\ffprobe.exe) thi dong goi kem.
if exist "tools\ffmpeg\bin\ffmpeg.exe" (
    xcopy /e /i /y "tools\ffmpeg" "release\LogoEraser\ffmpeg" >nul
    echo Da dong goi kem FFmpeg.
)

echo.
echo Xong: release\LogoEraser\LogoEraser.exe
endlocal
