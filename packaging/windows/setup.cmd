@echo off
title StudioLite Setup
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if errorlevel 1 (
  echo Setup failed. Review the error above.
  pause
  exit /b 1
)
echo Setup complete. Use the StudioLite desktop shortcut to start.
pause
