@echo off
setlocal
REM ============================================================
REM  nn-mind launcher - native Electron app (no browser).
REM  Self-checks every dependency and tells you what is wrong
REM  instead of silently closing.
REM ============================================================
cd /d "%~dp0"
title nn-mind launcher
echo.
echo  [nn-mind] starting from %CD%
echo.

REM ---- 1. Node / npm available? ------------------------------
where npm >/dev/null 2>&1
if errorlevel 1 (
  echo  [FAIL] npm not found. Install Node.js from https://nodejs.org
  goto :fail
)

REM ---- 2. Electron binary present? ---------------------------
if not exist "node_modules\electron\dist\electron.exe" (
  echo  [....] Electron not installed yet - running npm install ^(one-time^)...
  call npm install
  if not exist "node_modules\electron\dist\electron.exe" (
    echo  [FAIL] npm install did not produce node_modules\electron\dist\electron.exe
    echo         Check your network/proxy and re-run this launcher.
    goto :fail
  )
)
echo  [ ok ] Electron binary present

REM ---- 3. Python available? ----------------------------------
set "PY=%NN_MIND_PYTHON%"
if not defined PY set "PY=%LOCALAPPDATA%\Python\bin\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" --version >/dev/null 2>&1
if errorlevel 1 (
  echo  [FAIL] Python not found. Install Python 3.10+ or set NN_MIND_PYTHON
  echo         to your python.exe path.
  goto :fail
)
echo  [ ok ] Python found: %PY%

REM ---- 4. Python deps present? (auto-install if missing) -----
"%PY%" -c "import numpy, scipy, torch" >/dev/null 2>&1
if errorlevel 1 (
  echo  [....] Installing Python packages: numpy scipy torch ^(one-time, ~2 min^)...
  "%PY%" -m pip install numpy scipy torch
  "%PY%" -c "import numpy, scipy, torch" >/dev/null 2>&1
  if errorlevel 1 (
    echo  [FAIL] Python packages still missing after install.
    echo         Run by hand:  "%PY%" -m pip install numpy scipy torch
    goto :fail
  )
)
echo  [ ok ] Python packages ready ^(numpy scipy torch^)

REM ib_insync is optional - only needed for a real IBKR connection
"%PY%" -c "import ib_insync" >/dev/null 2>&1
if errorlevel 1 (
  echo  [note] ib_insync not installed - Test feed works, real IBKR will not.
  echo         To enable:  "%PY%" -m pip install ib_insync
) else (
  echo  [ ok ] ib_insync ready ^(real IBKR enabled^)
)

REM ---- 5. Launch ---------------------------------------------
echo.
echo  [nn-mind] launching native window...
set "NN_MIND_PYTHON=%PY%"
"node_modules\electron\dist\electron.exe" .
if errorlevel 1 (
  echo  [FAIL] Electron exited with an error ^(see messages above^).
  goto :fail
)
exit /b 0

:fail
echo.
echo  ============================================================
echo   Launch failed - read the [FAIL] line above for the reason.
echo  ============================================================
pause
exit /b 1
