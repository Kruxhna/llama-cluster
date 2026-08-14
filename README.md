# 🦙 AeroMesh (llama-cluster)

> **TL;DR**: AeroMesh is a topology-aware, fault-tolerant peer-to-peer (P2P) orchestration engine designed to execute 30B+ parameter open-source language models (e.g. Qwen-2.5-32B Q4_K_M) across mismatched consumer laptops over standard Wi-Fi. Powered by native `llama.cpp` RPC nodes, real-time 200ms `pynvml`/`psutil` telemetry, an Integer Linear Programming (ILP) Dynamic Graph Compiler (`pulp`), and a Byzantine Tensor Validator (Canary Trap Protocol). No cap, we maxing VRAM and making cluster go brrr fr fr. 🚀

[![CI - Build & Test](https://img.shields.io/badge/CI-passing-success.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![llama.cpp RPC](https://img.shields.io/badge/backend-llama.cpp--RPC-orange.svg)](https://github.com/ggerganov/llama.cpp)

---

## 📌 Executive Architecture & System Purpose

State-of-the-art 30B+ open-source parameter models require Video Random-Access Memory (VRAM) capacities far exceeding individual consumer graphics cards. High-end datacenter GPUs remain cost-prohibitive, while public cloud APIs introduce privacy concerns and high costs.

AeroMesh aggregates heterogeneous consumer laptops over standard wireless local area networks (Wi-Fi), introducing three core operational features:

1. **Heterogeneous Hardware Asymmetry Handling**: Uses an Integer Linear Program (ILP) solver (`pulp`) to dynamically compute model layer allocations ($L_{\text{total}} = 64$) proportional to each node's usable VRAM and TFLOPS.
2. **Volatile Network & Thermal Fault Tolerance**: Ingests real-time 200ms telemetry streams (`pynvml` + `psutil` + network RTT ping). Derates compute throughput by 50% if GPU temperature exceeds $85^\circ\text{C}$ and evicts bottleneck nodes if latency spikes above 300ms.
3. **Byzantine Tensor Validation (Canary Trap Protocol)**: Injects precomputed canary prompts (at 5% sampling rate) and measures relative $L_2$ Euclidean distance mismatch scores ($S_{\text{mismatch}}$) to detect floating-point output corruption.

---

## 💻 3-Laptop Heterogeneous Benchmark Topology

AeroMesh is benchmarked against an intentionally mismatched 3-laptop cluster topology (Specification Section 2.1):

| Node Identifier | Physical Hardware Specs | VRAM Footprint | Usable VRAM | Primary Operational Identity |
| :--- | :--- | :--- | :--- | :--- |
| **Laptop A** | Intel Core i7 / 16GB RAM / NVIDIA RTX 4060 Mobile | 8.0 GB GDDR6 | **7.5 GB** | **Coordinator & Stable Worker**: Primary control plane host, graph compiler. |
| **Laptop B** | AMD Ryzen 9 / 24GB RAM / NVIDIA RTX 4060 Mobile | 8.0 GB GDDR6 | **7.5 GB** | **Primary Execution Worker**: Peak compute node with maximum RAM buffer. |
| **Laptop C** | Intel Core i5 / 16GB RAM / NVIDIA RTX 3050 Mobile | 4.0 GB GDDR6 | **3.5 GB** | **Structural Bottleneck & Demo Hook**: Low throughput, narrow thermal envelope. |

**Aggregate Usable VRAM = 18.5 GB** (Sufficient for Qwen-2.5-32B Q4_K_M requiring 17.5 GB parameter weights + KV cache).

---

## 🧮 Theoretical & Mathematical Framework

### 1. Dynamic Graph Compiler (ILP Layer Allocation)
Minimizes total Time-Per-Output-Token (TPOT):

$$\min_{\\{l_i, x_i\\}} \left( \sum_{i \in N} \frac{l_i \cdot C_{\text{layer}}}{P_i \cdot x_i} + \sum_{(i,j) \in E} T_{i,j}^{\text{network}} \right)$$

Subject to:
- **Layer Conservation**: $\sum_{i \in N} l_i = L_{\text{total}} \quad (L_{\text{total}} = 64)$
- **VRAM Capacity**: $l_i \cdot M_{\text{layer\_weight}} + M_{\text{KV\_cache}} \le V_i^{\text{usable}} \cdot x_i$
- **Thermal Derating**: If $T_i^{\text{gpu}} > 85^\circ\text{C}$, $P_i \leftarrow 0.5 \cdot P_i$

### 2. Byzantine Tensor Validator (Canary Trap Mismatch Score)
Measures relative $L_2$ Euclidean distance deviation:

$$S_{\text{mismatch}} = \frac{\|v_{\text{actual}}^{(l)} - v_{\text{ref}}^{(l)}\|_2}{\|v_{\text{ref}}^{(l)}\|_2} = \frac{\sqrt{\sum_{k=1}^d (v_{\text{actual}, k}^{(l)} - v_{\text{ref}, k}^{(l)})^2}}{\sqrt{\sum_{k=1}^d (v_{\text{ref}, k}^{(l)})^2}}$$

| Mismatch Score Range ($S_{\text{mismatch}}$) | Hardware Status Interpretation | Automated System Reaction |
| :--- | :--- | :--- |
| $S_{\text{mismatch}} \le 0.003$ | Expected Quantization / FP Noise | Accept activation tensor; maintain active node status. |
| $0.003 < S_{\text{mismatch}} \le 0.005$ | Numerical Jitter / Thermal Drift Warning | Flag node for telemetry inspection; lower ILP priority weight. |
| $S_{\text{mismatch}} > 0.005$ | Hardware Arithmetic Corruption | Flag node as corrupted; trigger immediate graph re-slice to evict node. |

---

## 🛠️ System Requirements & Prerequisites

Assumes a fresh machine with basic developer tools:

- **Python**: Version `3.10` or higher (`python3 --version`)
- **Git**: Installed with submodule support (`git --version`)
- **CMake**: Version `3.14+` (`cmake --version`)
- **C++ Compiler**: GCC/Clang on Linux/macOS, MSVC Desktop C++ workload on Windows
- **Toxiproxy** *(Optional)*: Standalone Go proxy binary (`toxiproxy-server.exe`) for network chaos testing.

---

## 🚀 Copy-Paste Installation Guide

### 1. Clone Repository with Submodules

```bash
git clone --recursive https://github.com/Krushna/llama-cluster.git
cd llama-cluster
```

### 2. Run Native Host Setup

#### Windows (PowerShell / Command Prompt):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

#### Linux / macOS:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

## ⚙️ Environment Configuration (`.env`)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Key environment variables:

```ini
LLAMA_CLUSTER_HOST=0.0.0.0
LLAMA_CLUSTER_PORT=8080
TELEMETRY_INTERVAL_MS=200
THERMAL_THROTTLE_TEMP_C=85.0
CANARY_SAMPLE_RATE=0.05
CANARY_WARNING_THRESHOLD=0.003
CANARY_EVICTION_THRESHOLD=0.005
TOXIPROXY_HOST=127.0.0.1
TOXIPROXY_PORT=8474
```

---

## 📦 Asset & GGUF Model Setup

Download model weights (Qwen2.5-32B Instruct GGUF) directly into `./models/`:

```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Download default Qwen2.5-32B GGUF weights
aeromesh download
```

---

## 🖥️ AeroMesh CLI Usage & Operational Commands

AeroMesh provides CLI commands under `aeromesh` (or `llama-cluster`):

### 1. Check Telemetry & Hardware Status
```bash
aeromesh status
```

### 2. Run Dynamic Graph Compiler (ILP Layer Solver)
```bash
aeromesh rebalance
```

### 3. Launch Worker RPC Node (on Laptop B / Laptop C)
```bash
aeromesh node --name Laptop_B --port 50052
```

### 4. Launch Master Coordinator (on Laptop A)
```bash
aeromesh start --model Qwen2.5-32B-Instruct-Q4_K_M.gguf --port 8080
```

### 5. Network Chaos Injection (Toxiproxy)
Simulate network latency spikes on Laptop C:
```bash
aeromesh chaos inject --node Laptop_C --latency 500
```
Clear network perturbations:
```bash
aeromesh chaos clear --node Laptop_C
```

---

## 🌐 Local Wi-Fi & Inter-Node Networking Guide

Because AeroMesh connects laptops over standard consumer Wi-Fi, you must ensure that laptops on your local network are allowed to discover and communicate with each other over TCP sockets.

### 1. Set Windows Wi-Fi Network to "Private"
By default, Windows sets new Wi-Fi networks to **Public**, which blocks all inbound connections, RPC ports, and local ping discovery.

#### Option A: Via PowerShell (Run as Administrator)
```powershell
# Set all connected Wi-Fi interfaces to Private network profile
Get-NetConnectionProfile | Where-Object {$_.IPv4Connectivity -eq "Internet" -or $_.InterfaceAlias -like "*Wi-Fi*"} | Set-NetConnectionProfile -NetworkCategory Private
```

#### Option B: Via Windows Settings GUI
1. Open **Settings** (`Win + I`) -> **Network & Internet** -> **Wi-Fi**.
2. Click on your active connected Wi-Fi network.
3. Under **Network profile type**, select **Private network**.

---

### 2. Configure Windows Defender Firewall

Worker nodes (Laptop B & C) must allow inbound traffic on RPC port `50052`, and the coordinator (Laptop A) must allow master API port `8080` and Toxiproxy port `8474`.

#### Option A: Allow Required Inbound Ports (Recommended)
Run in PowerShell as Administrator on each laptop:

```powershell
# Allow AeroMesh RPC Worker Node Port
New-NetFirewallRule -DisplayName "AeroMesh RPC Node (50052)" -Direction Inbound -LocalPort 50052 -Protocol TCP -Action Allow

# Allow AeroMesh Master Coordinator HTTP API Port
New-NetFirewallRule -DisplayName "AeroMesh Master API (8080)" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow

# Allow Toxiproxy Chaos Engine Port
New-NetFirewallRule -DisplayName "AeroMesh Toxiproxy (8474)" -Direction Inbound -LocalPort 8474 -Protocol TCP -Action Allow

# Allow ICMP Ping Requests (For network RTT telemetry & discovery)
New-NetFirewallRule -DisplayName "AeroMesh Allow ICMP Ping" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow
```

#### Option B: Temporarily Disable Firewall for Private Network (Quick Testing)
```powershell
# Disable firewall ONLY for Private networks (leaves Public networks protected)
Set-NetFirewallProfile -Profile Private -Enabled False
```

---

### 3. Check Wi-Fi Router "AP Isolation" (Client Isolation)
Many consumer Wi-Fi routers and mobile hotspots enable **AP Isolation** (also called *Client Isolation* or *Station Isolation*) by default. This prevents connected devices on the same Wi-Fi network from communicating with one another.

* **Fix**: Log into your router's admin panel (usually `http://192.168.1.1` or `http://192.168.0.1`) -> **Wireless Settings** -> **Advanced** -> Turn **AP Isolation / Client Isolation** to **OFF** / **Disabled**.
* *Alternative*: Connect all laptops to an unmanaged gigabit Ethernet switch or a phone hotspot with client sharing enabled.

---

### 4. Verify Inter-Node Connectivity

Find each laptop's local IPv4 address:
- **Windows**: `ipconfig` (look for `IPv4 Address` under Wireless LAN adapter Wi-Fi)
- **Linux**: `ip -br a`
- **macOS**: `ipconfig getifaddr en0`

Test if Laptop A can connect to Laptop B's RPC port:
```powershell
# Test TCP socket connection on port 50052
Test-NetConnection -ComputerName 192.168.1.102 -Port 50052

# Test ICMP ping
ping 192.168.1.102
```

---

### 5. Linux (UFW) & macOS Firewall Setup

#### Linux (Ubuntu/Debian):
```bash
# Allow RPC port and Master API
sudo ufw allow 50052/tcp comment "AeroMesh RPC Node"
sudo ufw allow 8080/tcp comment "AeroMesh Master API"
sudo ufw reload
```

#### macOS:
If the macOS Application Firewall is enabled:
1. Open **System Settings** -> **Network** -> **Firewall**.
2. Click **Options...** and ensure **Block all incoming connections** is **OFF**.
3. Allow `llama-server` and `Python` when prompted.

---

## 📂 Repository Structure

```text
llama-cluster/
├── .github/
│   └── workflows/
│       └── ci.yml                     # GitHub Actions cross-platform CI runner
├── config/
│   └── cluster.yaml.example           # 3-Laptop benchmark topology allocation profile
├── models/
│   └── .gitkeep                       # GGUF weights directory (ignored via .gitignore)
├── scripts/
│   ├── setup.ps1                      # Windows PowerShell native host setup script
│   ├── setup.sh                       # Unix bash native host setup script
│   ├── setup.bat                      # Windows CMD setup script
│   ├── run_integration_test.ps1       # PowerShell integration test runner
│   └── download_model.py              # Standalone GGUF model downloader script
├── src/
│   └── llama_cluster/
│       ├── __init__.py                # Package exports & version
│       ├── canary_validator.py        # Byzantine Tensor Validator (Canary Trap)
│       ├── chaos.py                   # Toxiproxy network chaos engine integration
│       ├── cli.py                     # AeroMesh CLI commands (start, node, status, rebalance, chaos)
│       ├── config.py                  # Dynamic root-relative path & .env loader
│       ├── downloader.py              # Hugging Face stream downloader
│       ├── graph_compiler.py          # Dynamic Graph Compiler (PuLP ILP layer solver)
│       ├── node.py                    # Worker node daemon with 200ms JSON stream
│       ├── orchestrator.py            # Master Control Plane (Laptop A Coordinator)
│       └── telemetry.py               # Dual-tier telemetry engine (pynvml + psutil + RTT)
├── tests/
│   ├── __init__.py
│   ├── test_canary.py                 # Unit tests for Canary Trap L2 distance
│   ├── test_cli.py                    # Unit tests for CLI arguments
│   ├── test_compiler.py               # Unit tests for ILP layer allocation solver
│   ├── test_config.py                 # Unit tests for path & config resolution
│   └── test_telemetry.py             # Unit tests for 200ms stream telemetry
├── .editorconfig                      # Indentation & line ending rules
├── .env.example                       # AeroMesh environment variables template
├── .gitattributes                     # Line normalizations & binary flags
├── .gitignore                         # Comprehensive ignore rules (.venv, *.gguf, build outputs)
├── CONTRIBUTING.md                    # Open-source contribution guidelines
├── LICENSE                            # MIT License
├── pyproject.toml                     # Python package metadata & dependencies
└── README.md                          # Master documentation (you are here!)
```

---

## 🧪 Integration Testing & CI/CD

Run integration test suite locally:

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_integration_test.ps1
```

#### Cross-Platform Pytest:
```bash
uv run pytest
```

---

## 📜 License

This project is released under the [MIT License](LICENSE).

---

## 🙌 Credits & Acknowledgements

- **llama.cpp** by Georgi Gerganov for the native C++ inference engine and RPC implementation.
- **PuLP** for Python Linear Programming integer optimization.
- **Toxiproxy** by Shopify for network chaos testing.
