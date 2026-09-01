@echo off
title Build CineFlow-AI Studio Executable
cd /d "%~dp0"
echo ====================================================
echo   Building CineFlow-AI Standalone .exe...
echo ====================================================
python build_exe.py
echo.
echo Build complete. Check the dist\ folder.
pause
