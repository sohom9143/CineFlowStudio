@echo off
title CineFlow-AI Studio Launcher
cd /d "%~dp0"
echo ====================================================
echo   Starting CineFlow-AI Studio Desktop Application...
echo   (Auto-linking with Google Colab Cloud Session)
echo ====================================================

if exist "dist\CineFlow-AI-Studio\CineFlow-AI-Studio.exe" (
    start "" "dist\CineFlow-AI-Studio\CineFlow-AI-Studio.exe"
) else (
    python desktop_app.py
)

