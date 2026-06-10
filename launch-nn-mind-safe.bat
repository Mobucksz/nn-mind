@echo off
REM ============================================================
REM  launch-nn-mind-safe.bat
REM  Safe launcher for the electron-only (Node/TF.js) build.
REM
REM  USE THIS instead of any pasted "git pull gracious-fermi" block.
REM
REM  - Stays on / switches to the electron-only branch
REM    (claude/3d-surface-lab).
REM  - Syncs with --ff-only: fast-forward ONLY. If the branch has
REM    diverged it aborts safely and merges NOTHING. It never
REM    touches the Python-sidecar branch (gracious-fermi-w8ATB).
REM  - Refuses to open a second window if nn-mind is already up.
REM ============================================================
setlocal
cd /d "C:\Users\Bryce\nn-mind"

REM --- ensure we are on the electron-only branch ---
for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH=%%b"
if not "%BRANCH%"=="claude/3d-surface-lab" (
    echo [nn-mind] On %BRANCH% - switching to claude/3d-surface-lab ...
    git checkout claude/3d-surface-lab || goto :end
)

REM --- safe sync: fast-forward only, never a merge ---
echo [nn-mind] Syncing origin/claude/3d-surface-lab (fast-forward only) ...
git pull --ff-only origin claude/3d-surface-lab

REM --- do not open a second window if already running ---
tasklist /FI "WINDOWTITLE eq nn-mind*" 2>nul | find /I "electron.exe" >nul
if not errorlevel 1 (
    echo [nn-mind] Already running - not opening a second window.
    goto :end
)

echo [nn-mind] Launching electron-only build ...
"%CD%\node_modules\.bin\electron.cmd" .

:end
endlocal
