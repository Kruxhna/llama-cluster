# ==============================================================================
# AeroMesh Native Windows Host Setup Script (PowerShell)
# Specification Reference: Section 5.1 Pillar 1 - Native Windows Host Execution
# Installs dependencies via uv, sets up .env, and builds llama.cpp
# ==============================================================================

$ErrorActionPreference = "Stop"

Write-Host "🔥 [1/4] Setting up AeroMesh environment on Native Windows Host... 🔥" -ForegroundColor Cyan

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Resolve-Path "$ScriptDir\.."
Set-Location $RepoRoot

# Verify Python
if (-not (Get-Command "python" -ErrorAction SilentlyContinue)) {
    Write-Error "❌ Error: Python is required but not installed or not added to PATH."
    exit 1
}

# Create virtualenv via uv if installed, else python venv
if (Get-Command "uv" -ErrorAction SilentlyContinue) {
    Write-Host "⚡ Using 'uv' package manager for ultra-fast setup..." -ForegroundColor Green
    uv venv
    & .venv\Scripts\activate.ps1
    uv pip install -e ".[dev]"
} else {
    Write-Host "📦 Using standard Python venv setup..." -ForegroundColor Yellow
    python -m venv .venv
    & .venv\Scripts\activate.ps1
    pip install -e ".[dev]"
}

# Create .env from template if missing
if (-not (Test-Path ".env")) {
    Write-Host "📄 Copying .env from .env.example..." -ForegroundColor Green
    Copy-Item ".env.example" ".env"
}

# Create models directory
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}

# Build llama.cpp binaries via CMake
if (Test-Path "llama.cpp") {
    Write-Host "🛠️ [2/4] Compiling native llama.cpp binaries (-DGGML_RPC=ON -DGGML_CUDA=ON)..." -ForegroundColor Cyan
    if (-not (Test-Path "llama.cpp\build")) {
        New-Item -ItemType Directory -Path "llama.cpp\build" | Out-Null
    }
    Set-Location "llama.cpp\build"
    cmake .. -DLLAMA_BUILD_SERVER=ON -DLLAMA_RPC=ON
    cmake --build . --config Release
    Set-Location $RepoRoot
    Write-Host "✅ Native llama.cpp build complete!" -ForegroundColor Green
} else {
    Write-Host "⚠️ Warning: llama.cpp directory missing. Clone submodules with: git submodule update --init --recursive" -ForegroundColor Yellow
}

Write-Host "🚀 [3/4] Initializing AeroMesh CLI..." -ForegroundColor Cyan
llama-cluster init

Write-Host ""
Write-Host "🎉 AeroMesh Windows setup complete! We cookin now fr fr." -ForegroundColor Green
Write-Host "👉 Run '.venv\Scripts\activate' and then 'aeromesh status' to get started!" -ForegroundColor White
