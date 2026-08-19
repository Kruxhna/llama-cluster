# AeroMesh (`llama-cluster`)

A distributed peer-to-peer (P2P) orchestration engine that runs 30B+ parameter LLMs (like Qwen 2.5 32B or DeepSeek 14B) across mismatched consumer laptops over standard Wi-Fi and Tailscale networks. Built on native `llama.cpp` RPC workers, real-time GPU telemetry (`pynvml`), an Integer Linear Programming (ILP) layer balancer (`pulp`), and zero-weight local model slicing.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![llama.cpp RPC](https://img.shields.io/badge/backend-llama.cpp--RPC-orange.svg)](https://github.com/ggerganov/llama.cpp)

---

## Why AeroMesh?

Running local 30B+ models usually requires 24GB+ VRAM (an RTX 3090/4090 or datacenter GPU). Most people only have gaming laptops with 4GB to 8GB VRAM (RTX 3050, 4060, etc.).

AeroMesh aggregates consumer laptops into a unified cluster:
1. **Dynamic Layer Slicing**: An ILP solver partitions model layers across nodes based on each machine's actual free VRAM, GPU compute speed, and network ping.
2. **Zero-Weight Local Slicing**: Laptops load their assigned layer slice directly from their local `models/` directory using NVMe memory-mapping (`mmap`) with zero weight transfers over Wi-Fi.
3. **Thermal & Latency Fault Tolerance**: Every 200ms, worker nodes report GPU temperature and ping. If a laptop thermal throttles (>85°C) or Wi-Fi latency spikes, layers are automatically shifted to other nodes between turns.
4. **Canary Trap Validation**: Periodically passes reference prompts through the pipeline to catch silent hardware math errors or bitflips.

---

## Quickstart

### 1. Clone Repository
```bash
git clone --recursive https://github.com/Kruxhna/llama-cluster.git
cd llama-cluster
```

### 2. Install & Build

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

---

## How to Run the Cluster

### Option A: Rust Engine (`cargo run`)

#### On Worker Laptops:
```powershell
cargo run --bin aeromesh -- worker --port 50052
```

#### On Coordinator Laptop:
```powershell
cargo run --bin aeromesh -- coordinator `
    --model "models/DS.gguf" `
    --peers "100.101.147.24:50052" `
    --ngl -1 `
    --prompt "Explain distributed GPU clustering in one short sentence."
```

---

### Option B: Python Control Plane & Web Dashboard

#### On Worker Laptop:
```powershell
.\.venv\Scripts\Activate.ps1
python -m llama_cluster.cli node --name Laptop_A --port 50052 --mode local-pipeline --model DS.gguf
```

#### On Coordinator Laptop:
```powershell
.\.venv\Scripts\Activate.ps1
python -m llama_cluster.cli start --model DS.gguf --port 8080
python -m llama_cluster.cli dashboard --port 3000
```
Open **`http://localhost:3000`** in your browser for real-time telemetry, layer sliders, and chat console.

---

## Project Structure

```text
llama-cluster/
├── Cargo.toml                # Rust workspace configuration
├── crates/                   # Rust native engine & CLI crates
│   ├── aeromesh-core/        # Core domain types & Tailscale prober
│   ├── aeromesh-engine/      # Windows Job Object supervisor & process manager
│   └── aeromesh-cli/         # Unified 'aeromesh' CLI binary
├── src/llama_cluster/        # Python orchestration engine
│   ├── canary_validator.py   # Canary Trap tensor validator (L2 distance)
│   ├── cli.py                # CLI commands
│   ├── dashboard.py          # Web Control Dashboard REST & Static backend
│   ├── graph_compiler.py     # ILP layer solver (PuLP)
│   ├── node.py               # Worker node daemon
│   ├── orchestrator.py       # Master control plane & layer rebalancer
│   ├── pipeline_bridge.py    # Zero-Weight P2P activation bridge
│   ├── telemetry.py          # Hardware collector (pynvml + psutil)
│   └── web/                  # Web Dashboard UI
├── tests/                    # Unit & integration test suite
├── config/                   # Cluster topology profiles
└── models/                   # Local .gguf model repository
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
