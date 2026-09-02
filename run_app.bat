@echo off
title CineFlow-AI Studio - Live Backend Launcher
cd /d "%~dp0"
echo ================================================================
echo   CineFlow-AI Studio: Starting Desktop App with Live Backend...
echo   Auto-linking with Google Colab T4 Cloud Environment...
echo ================================================================

:: Start live Python backend with native Desktop window and Colab auto-open
python desktop_app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ----------------------------------------------------------------
    echo [Fallback] Launching compiled desktop executable...
    echo ----------------------------------------------------------------
    if exist "dist\CineFlow-AI-Studio\CineFlow-AI-Studio.exe" (
        start "" "dist\CineFlow-AI-Studio\CineFlow-AI-Studio.exe"
    )
)

pause

