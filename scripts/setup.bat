@echo off
REM ==============================================================================
REM llama-cluster Quick Setup Script (Windows Command Prompt / PowerShell)
REM Installs dependencies, sets up .env, and builds llama.cpp
REM ==============================================================================

echo 🔥 [1/4] Setting up llama-cluster environment on Windows... 🔥

cd /d "%~dp0\.."

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Error: Python is not installed or not added to PATH.
    exit /b 1
)

where uv >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚡ Using 'uv' for fast environment setup...
    uv venv
    call .venv\Scripts\activate.bat
    uv pip install -e .[dev]
) else (
    echo 📦 Using standard python venv...
    python -m venv .venv
    call .venv\Scripts\activate.bat
    pip install -e .[dev]
)

if not exist .env (
    echo 📄 Creating default .env from .env.example...
    copy .env.example .env
)

if not exist models (
    mkdir models
)

if exist llama.cpp (
    echo 🛠️ [2/4] Building llama.cpp binaries via CMake...
    if not exist llama.cpp\build mkdir llama.cpp\build
    cd llama.cpp\build
    cmake .. -DLLAMA_BUILD_SERVER=ON -DLLAMA_RPC=ON
    cmake --build . --config Release
    cd ..\..
    echo ✅ llama.cpp build complete!
) else (
    echo ⚠️ Warning: llama.cpp directory not found. Clone submodules with: git submodule update --init --recursive
)

echo 🚀 [3/4] Initializing llama-cluster CLI...
llama-cluster init

echo.
echo 🎉 Windows setup complete! We cookin now.
echo 👉 Run '.venv\Scripts\activate' and then 'llama-cluster status' to get started!
