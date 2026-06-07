#!/usr/bin/env pwsh
<#
.SYNOPSIS
  Record one trading day of IBKR option data for OpVis on Windows.
  Fires each weekday just after market open, records until the close,
  and updates public/data/sessions/ + public/data/manifest.json.

  Automatically launches TWS paper if not already running, waits for it
  to connect, then records the full session. When you open OpVis after
  work, you'll see the entire day you missed.
#>

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
$Symbol = if ($env:OPTIONSCOPE_SYMBOL) { $env:OPTIONSCOPE_SYMBOL } else { 'SPY' }
$Python = 'C:\Users\Bryce\AppData\Local\Python\bin\python.exe'
$Script = Join-Path $RepoRoot 'tools\record_ibkr_session.py'
$LogDir = Join-Path $RepoRoot 'logs'
$LogFile = Join-Path $LogDir "record-$(Get-Date -Format 'yyyy-MM-dd').log"

# Create log directory if needed
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }

# Skip weekends
$DayOfWeek = (Get-Date).DayOfWeek
if ($DayOfWeek -eq 'Saturday' -or $DayOfWeek -eq 'Sunday') {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - weekend, skipping." | Out-File -FilePath $LogFile -Append
    exit 0
}

function Write-Log {
    param([string]$Msg)
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $Msg" | Out-File -FilePath $LogFile -Append
}

Write-Log "=== Starting daily recording for $Symbol ==="

# Ensure TWS Paper is running
$twsRunning = $false
try {
    $procs = Get-Process -Name 'tws','ibgateway','java' -ErrorAction SilentlyContinue
    $twsRunning = ($procs.Count -gt 0)
} catch {}

if (-not $twsRunning) {
    Write-Log "TWS/Gateway not running. Launching TWS Paper..."
    $twsPath = "$env:LOCALAPPDATA\IBKR\TWS\tws.exe"
    $twsAlt = "C:\Program Files\Interactive Brokers\TWS\tws.exe"
    $launcher = $null
    if (Test-Path $twsPath) { $launcher = $twsPath }
    elseif (Test-Path $twsAlt) { $launcher = $twsAlt }
    elseif (Test-Path "C:\Program Files\Interactive Brokers\Gateway\ibgateway.exe") {
        $launcher = "C:\Program Files\Interactive Brokers\Gateway\ibgateway.exe"
    }

    if ($launcher) {
        Write-Log "Launching $launcher..."
        Start-Process -FilePath $launcher
        Write-Log "Waiting 60s for TWS/Gateway to start and log in..."
        Start-Sleep -Seconds 60
    } else {
        Write-Log "WARNING: Could not find TWS or Gateway executable. Recording will likely fail."
    }
} else {
    Write-Log "TWS/Gateway already running."
}

# Record the session
Write-Log "Recording $Symbol until 16:00 ET close..."
try {
    $env:PYTHONIOENCODING = 'utf-8'
    $output = & $Python $Script --symbol $Symbol --auto-rth 2>&1
    $output | Out-File -FilePath $LogFile -Append
    Write-Log "Recording completed successfully."
} catch {
    Write-Log "Recording failed: $_"
    exit 1
}

Write-Log "=== Done ==="
