@echo off
title CineFlow-AI - Model Weights Downloader
cd /d "%~dp0"
echo ==========================================================
echo   Downloading CineFlow-AI Neural Network Weights...
echo ==========================================================
python download_models.py
echo.
echo Model verification finished.
pause
