@echo off
setlocal
cd /d "%~dp0"

if not exist "Med\myenv\Scripts\python.exe" (
  echo ERROR: Python not found at Med\myenv\Scripts\python.exe
  pause
  exit /b 1
)

"Med\myenv\Scripts\python.exe" -m streamlit run "Med\myenv\lawchatbot.py"
