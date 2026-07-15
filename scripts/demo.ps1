# One-command local demo launcher (Windows / PowerShell).
#
# Creates a virtualenv, installs the dashboard dependencies, seeds the local
# database from the bundled Starlink snapshot (so it works even with no
# network), and launches the Streamlit dashboard at http://localhost:8501.
#
# Usage:
#   .\scripts\demo.ps1
#
# Safe to re-run: the venv and seeded DB are reused if they already exist.

$ErrorActionPreference = "Stop"

# Move to the project root regardless of where the script is invoked from.
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "==> Creating virtual environment (.venv)"
    python -m venv .venv
}

Write-Host "==> Installing dependencies (this may take a minute the first time)"
& $VenvPython -m pip install --upgrade pip | Out-Null
& $VenvPython -m pip install -e ".[dashboard]" | Out-Null

Write-Host "==> Seeding the local database from the bundled Starlink snapshot"
& $VenvPython scripts\seed_demo.py

Write-Host ""
Write-Host "==> Launching the dashboard at http://localhost:8501  (Ctrl+C to stop)"
Write-Host ""
& $VenvPython -m spacetrack.cli dashboard
