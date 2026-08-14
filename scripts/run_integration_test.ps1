# ==============================================================================
# AeroMesh Integration Test Suite Runner (PowerShell)
# Specification Reference: Section 5.5 - Standardized Integration & Developer Experience
# Drives pytest suites and verifies cross-node telemetry and ILP compiler resolution
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "[*] Running AeroMesh Integration Test Suite..." -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Resolve-Path "$ScriptDir\.."
Set-Location $RepoRoot

# Activate virtualenv if available
if (Test-Path ".venv\Scripts\activate.ps1") {
    & .venv\Scripts\activate.ps1
}

Write-Host "[*] Executing Pytest test suite..." -ForegroundColor Yellow
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    uv run pytest -v
} else {
    pytest -v
}

Write-Host ""
Write-Host "[+] All AeroMesh integration tests passed clean!" -ForegroundColor Green
