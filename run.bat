@echo off
rem LogoEraser - chay giao dien tu ma nguon.
rem Lan dau: tao moi truong ao .venv va cai thu vien (can Python 3.10+ va mang de tai thu vien).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [LogoEraser] Tao moi truong ao .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Khong tao duoc .venv. Hay cai Python 3.10+ tu https://www.python.org/ roi chay lai.
        pause
        exit /b 1
    )
    echo [LogoEraser] Cai thu vien ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Cai thu vien that bai.
        pause
        exit /b 1
    )
)

where ffmpeg >nul 2>nul || echo [Canh bao] Khong thay ffmpeg trong PATH - xu ly anh van chay, video can FFmpeg. Xem README.md.

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" main.py
) else (
    ".venv\Scripts\python.exe" main.py
)
endlocal
