@echo off
setlocal
set "GRADCAM_PYTHON=%~dp0..\BurnerEye Trainer\.venv\Scripts\python.exe"
if not exist "%GRADCAM_PYTHON%" (
    echo BurnerEye Trainer Python environment was not found.
    echo Expected: "%GRADCAM_PYTHON%"
    pause
    exit /b 1
)
"%GRADCAM_PYTHON%" "%~dp0gradcam_gui.py"
if errorlevel 1 pause
