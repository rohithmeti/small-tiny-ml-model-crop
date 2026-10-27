@echo off
rem ============================================================================
rem  Crop recommendation (CPU only) - double-click me.
rem  First run : builds the Python environment (.venv) - needs internet, a few minutes.
rem  Next runs : starts immediately.
rem  Optional  : START_HERE_windows.bat train     (or: bench, test, all, live ... see README.md)
rem ============================================================================
setlocal
cd /d "%~dp0"
title Crop recommendation (CPU)

rem --- find a supported Python (3.10 - 3.13)
set "PY="
for %%V in (3.12 3.11 3.13 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY=python"
)
if not defined PY (
  echo.
  echo Python 3.10 - 3.13 was not found on this computer.
  echo Install it from https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^) and run this file again.
  pause
  exit /b 1
)
echo Using Python: %PY%

rem --- first run only: build the environment
if not exist ".venv\Scripts\python.exe" (
  echo.
  echo First run: building the environment. This downloads about 350 MB and takes a few minutes.
  %PY% setup_env.py
  if errorlevel 1 (
    echo.
    echo Setup failed - see the messages above.
    pause
    exit /b 1
  )
)

rem --- run: no argument = demo; otherwise pass the arguments to run.py
if "%~1"=="" (
  ".venv\Scripts\python.exe" run.py demo
  echo.
  pause
) else (
  ".venv\Scripts\python.exe" run.py %*
)
