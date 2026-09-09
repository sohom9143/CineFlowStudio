@echo off
title CineFlow AI Studio - Enterprise C# Desktop Edition
echo ========================================================
echo   CineFlow AI Studio: Enterprise C# Desktop Edition
echo ========================================================
echo.

cd /d "%~dp0"

:: 1. Check if local Python API service is running on port 7860
netstat -ano | findstr 127.0.0.1:7860 >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [1/2] Starting local Python AI Bridge in background...
    start /b python app.py --port 7860 --config configs/colab_t4_config.yaml > nul 2>&1
    timeout /t 3 /nobreak > nul
) else (
    echo [1/2] Local Python AI Bridge already running on port 7860.
)

:: 2. Launch Native C# WPF Desktop Application (.exe)
echo [2/2] Launching Native C# Desktop Studio (.exe)...
start "" "dist\CineFlowStudio-Desktop\CineFlowStudio.Desktop.exe"

echo.
echo Enjoy CineFlow-AI Studio!
