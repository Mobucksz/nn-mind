@echo off
REM Launches nn-mind as a desktop app via Electron (native window, no browser).
cd /d "C:\Users\Bryce\nn-mind"
call npm run build >nul 2>&1
"%CD%\node_modules\.bin\electron.cmd" .
