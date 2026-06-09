@echo off
REM Launches nn-mind as a desktop app via Electron (native window).
cd /d "C:\Users\Bryce\nn-mind"
"%CD%\node_modules\.bin\electron.cmd" .
