@echo off
setlocal
cd /d "%~dp0"
title XTTS-v2 Easy GUI Installer
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"

pause
