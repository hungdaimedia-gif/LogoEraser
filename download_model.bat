@echo off
rem Tai model AI LaMa (khoang 200 MB, mot lan) cho phuong thuc inpaint_ai.
rem Sau khi tai, ung dung chay hoan toan offline.
setlocal
cd /d "%~dp0"
if not exist models mkdir models
if exist "models\lama_fp32.onnx" (
    echo Da co models\lama_fp32.onnx - khong can tai lai.
    goto :done
)
echo Dang tai model LaMa tu HuggingFace (Carve/LaMa-ONNX, giay phep Apache-2.0)...
curl.exe -L --fail -o "models\lama_fp32.onnx.part" "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
if errorlevel 1 (
    echo Tai that bai. Kiem tra mang roi chay lai.
    goto :done
)
move /y "models\lama_fp32.onnx.part" "models\lama_fp32.onnx" >nul
echo Xong: models\lama_fp32.onnx
:done
pause
endlocal
