# Contributing to AeroMesh (llama-cluster)

Thank you for your interest in contributing to **AeroMesh**! We're building an open-source, topology-aware P2P orchestration engine to run 30B+ parameter models across consumer hardware.

Whether you're fixing a telemetry edge case, improving the ILP graph compiler, adding new chaos tests, or enhancing documentation, your contributions are appreciated.

---

## 📋 Table of Contents

- [Code of Conduct](#-code-of-conduct)
- [Development Setup](#-development-setup)
- [Architecture & Core Modules](#-architecture--core-modules)
- [Development Guidelines](#-development-guidelines)
- [Testing & Quality Assurance](#-testing--quality-assurance)
- [Git & Pull Request Workflow](#-git--pull-request-workflow)
- [Commit Message Conventions](#-commit-message-conventions)

---

## 📜 Code of Conduct

We are committed to providing an open, welcoming, and inclusive environment for all contributors. Please keep discussions constructive, respectful, and focused on technical excellence.

---

## 🛠️ Development Setup

### 1. Prerequisites
- **Python**: Version 3.10+
- **Package Manager**: [uv](https://docs.astral.sh/uv/) (recommended) or standard `pip`
- **Build Tools**: CMake 3.14+ and a C++17 compiler (MSVC on Windows, GCC/Clang on Linux/macOS)
- **Git**: Configured with submodule support

### 2. Clone & Initialize Submodules
```bash
git clone --recursive https://github.com/Krushna/llama-cluster.git
cd llama-cluster
```

If already cloned without submodules:
```bash
git submodule update --init --recursive
```

### 3. Setup Virtual Environment
```bash
# Using uv (fastest)
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 4. Build `llama.cpp` Binaries
AeroMesh relies on native `llama-server` and `llama-rpc-server` binaries compiled with RPC support:

#### Windows:
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

#### Linux / macOS:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

## 🏛️ Architecture & Core Modules

When contributing, ensure your changes align with the modular system design:

```text
src/llama_cluster/
├── canary_validator.py    # Byzantine Tensor Validator (Canary Trap L2 distance)
├── chaos.py               # Toxiproxy network perturbation & chaos testing
├── cli.py                 # CLI commands (status, rebalance, chaos, start, node)
├── config.py              # Dynamic root-relative path & environment management
├── downloader.py          # Hugging Face stream downloader
├── graph_compiler.py      # Dynamic Graph Compiler (PuLP ILP layer solver)
├── node.py                # Worker node daemon & 200ms telemetry stream
├── orchestrator.py        # Master Coordinator Control Plane
└── telemetry.py           # Dual-tier telemetry (pynvml + psutil + network RTT)
```

---

## 📐 Development Guidelines

### 1. Cross-Platform Path Handling
- Always use Python's standard `pathlib.Path`.
- Never hardcode absolute system paths (e.g., `C:\...`, `/Users/...`, `/home/...`).
- Use `REPO_ROOT` from `llama_cluster.config` to resolve paths relative to the project root.

### 2. Defensive Telemetry & Fallbacks
- Hardware environments vary widely across consumer laptops.
- Any GPU-specific calls (`pynvml`) must be encapsulated within defensive try/except blocks and degrade gracefully to CPU/RAM (`psutil`) telemetry.
- Telemetry payloads must adhere to the 200ms JSON stream specification defined in Section 6.2 of the Master Architecture Report.

### 3. Dynamic Graph Compiler (ILP)
- Integer Linear Programming formulations in `graph_compiler.py` must maintain layer conservation ($\sum l_i = L_{\text{total}}$).
- Respect the thermal throttling derating rules ($50\%$ compute throughput penalty when $T > 85^\circ\text{C}$) and network latency eviction thresholds ($>300\text{ms}$ RTT).

### 4. Byzantine Validator (Canary Trap)
- Verification scores must follow the relative $L_2$ Euclidean distance metric ($S_{\text{mismatch}}$).
- Respect the defined threshold boundaries:
  - $S_{\text{mismatch}} \le 0.003$: Normal quantization noise
  - $0.003 < S_{\text{mismatch}} \le 0.005$: Numerical drift / thermal warning
  - $S_{\text{mismatch}} > 0.005$: Hardware arithmetic corruption / trigger eviction

### 5. Git Hygiene & Artifacts
- **Never commit model weights**: Files matching `*.gguf`, `*.bin`, or `*.safetensors` belong in `models/` and are ignored by `.gitignore`.
- **Never commit secrets**: Do not commit active `.env` files or API tokens.

---

## 🧪 Testing & Quality Assurance

Before submitting any code, verify that the entire test suite passes locally.

### Running Pytest
```bash
uv run pytest -v
```

### Running Linter & Formatter
```bash
uv run ruff check .
uv run ruff format --check .
```

### Running Integration Tests
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_integration_test.ps1
```

---

## 🔄 Git & Pull Request Workflow

1. **Fork the repository** on GitHub.
2. **Create a topic branch** from `main`:
   ```bash
   git checkout -b feature/dynamic-ilp-optimizations
   # or
   git checkout -b fix/nvml-telemetry-fallback
   ```
3. **Make focused, atomic commits** with clear commit messages.
4. **Push your branch** to your fork:
   ```bash
   git push origin feature/dynamic-ilp-optimizations
   ```
5. **Open a Pull Request** against `main` with:
   - A clear summary of the changes and motivation
   - Reproduction or test steps demonstrating that the changes work
   - Associated issue numbers (e.g., `Closes #12`)

---

## ✍️ Commit Message Conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat: add adaptive KV-cache migration during between-turn rebalancing`
- `fix: handle missing NVML DLL gracefully on non-NVIDIA laptops`
- `perf: optimize ILP solver execution time to <15ms`
- `test: add unit tests for relative L2 canary distance calculation`
- `docs: update 3-laptop benchmark topology in README`
- `refactor: clean up node daemon socket lifecycle`

---

## 💬 Questions & Support

If you encounter bugs, have design suggestions, or need help setting up the cluster, please open an issue on GitHub.
