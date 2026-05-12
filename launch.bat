@echo off
title StudioLite
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1"
if errorlevel 1 (
    echo.
    echo Launcher exited with errorlevel %errorlevel%.
    pause
)
