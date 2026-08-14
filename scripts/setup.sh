#!/usr/bin/env bash
# ==============================================================================
# llama-cluster Quick Setup Script (Linux / macOS)
# Installs dependencies, sets up .env, and builds llama.cpp
# ==============================================================================

set -e

echo "🔥 [1/4] Setting up llama-cluster environment... 🔥"

# Move to repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is required but not installed."
    exit 1
fi

# Initialize .venv using uv or venv
if command -v uv &> /dev/null; then
    echo "⚡ Using 'uv' for ultra-fast environment setup..."
    uv venv
    source .venv/bin/activate
    uv pip install -e ".[dev]"
else
    echo "📦 Using standard 'pip' setup..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
fi

# Copy default .env if missing
if [ ! -f ".env" ]; then
    echo "📄 Creating default .env file from .env.example..."
    cp .env.example .env
fi

# Create models directory
mkdir -p models

# Build llama.cpp if present
if [ -d "llama.cpp" ]; then
    echo "🛠️ [2/4] Building llama.cpp binaries via CMake..."
    mkdir -p llama.cpp/build
    cd llama.cpp/build
    cmake .. -DLLAMA_BUILD_SERVER=ON -DLLAMA_RPC=ON
    cmake --build . --config Release -j $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd "$REPO_ROOT"
    echo "✅ llama.cpp build complete!"
else
    echo "⚠️ Warning: llama.cpp directory not found. Clone submodules with: git submodule update --init --recursive"
fi

echo "🚀 [3/4] Initializing llama-cluster CLI..."
llama-cluster init

echo ""
echo "🎉 Setup complete! You're locked in fr fr."
echo "👉 Run 'source .venv/bin/activate' and then 'llama-cluster status' to get started!"
