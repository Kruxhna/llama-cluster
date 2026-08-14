# AeroMesh (`llama-cluster`)

A distributed peer-to-peer (P2P) orchestration engine that runs 30B+ parameter LLMs (like Qwen 2.5 32B) across mismatched consumer laptops over standard Wi-Fi. Built on native `llama.cpp` RPC workers, real-time GPU telemetry (`pynvml`), an Integer Linear Programming (ILP) layer balancer (`pulp`), and runtime tensor validation.

[![CI](https://img.shields.io/badge/CI-passing-success.svg)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![llama.cpp RPC](https://img.shields.io/badge/backend-llama.cpp--RPC-orange.svg)](https://github.com/ggerganov/llama.cpp)

---

## Why AeroMesh?

Running local 30B+ models usually requires 24GB+ VRAM (an RTX 3090/4090 or datacenter GPU). Most people only have gaming laptops with 4GB to 8GB VRAM (RTX 3050, 4060, etc.).

Tensor Parallelism (TP) doesn't work well over Wi-Fi because every single layer needs an `All-Reduce` collective synchronization across the network, causing massive latency.

**AeroMesh uses Pipeline Parallelism (PP) with dynamic topology awareness:**
1. **Dynamic Layer Slicing**: An ILP solver partitions model layers across nodes based on each machine's actual free VRAM, GPU compute speed, and network ping.
2. **Thermal & Latency Fault Tolerance**: Every 200ms, worker nodes report GPU temperature and ping. If a laptop thermal throttles (>85°C) or Wi-Fi latency spikes, layers are automatically shifted to other nodes between turns.
3. **Canary Trap Validation**: Periodically passes reference prompts through the pipeline to catch silent hardware math errors or bitflips caused by thermal stress or aggressive memory overclocks.

---

## Benchmark Cluster Setup

Tested on an intentionally mismatched 3-laptop setup:

| Node | Specs | Total VRAM | Usable VRAM | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Laptop A** | Intel i7 / 16GB RAM / RTX 4060 Mobile | 8.0 GB | **7.5 GB** | Coordinator & Stable Worker (Hosts API & ILP solver) |
| **Laptop B** | AMD Ryzen 9 / 24GB RAM / RTX 4060 Mobile | 8.0 GB | **7.5 GB** | Primary Execution Worker (High RAM buffer) |
| **Laptop C** | Intel i5 / 16GB RAM / RTX 3050 Mobile | 4.0 GB | **3.5 GB** | Bottleneck Worker (Low VRAM, narrow thermals) |

* **Total Usable VRAM across cluster**: **18.5 GB**
* **Target Model**: `Qwen-2.5-32B-Instruct` (Q4_K_M quantization = ~17.5 GB weights + 1 GB KV cache at 2k context).
* **Layer Allocation**: 64 total layers distributed dynamically across all 3 devices.

---

## How It Works

```
                     ┌──────────────────────────────────────────────┐
                     │          Laptop A (Coordinator)              │
                     │  - OpenAI API Endpoint (Port 8080)           │
                     │  - Dynamic Graph Compiler (PuLP ILP Solver)  │
                     │  - Byzantine Canary Validator                │
                     └───────────────┬──────────────┬───────────────┘
                                     │              │
                    200ms Telemetry  │              │  200ms Telemetry
                    & Activation Ring│              │  & Activation Ring
                                     ▼              ▼
                     ┌──────────────────┐        ┌──────────────────┐
                     │     Laptop B     │        │     Laptop C     │
                     │  (RTX 4060 8GB)  │◄──────►│  (RTX 3050 4GB)  │
                     │  RPC Port 50052  │        │  RPC Port 50052  │
                     └──────────────────┘        └──────────────────┘
```

### 1. Dynamic Layer Partitioning (ILP Formulation)
Minimizes total Time-Per-Output-Token (TPOT):

$$\min_{\{l_i, x_i\}} \left( \sum_{i \in N} \frac{l_i \cdot C_{\text{layer}}}{P_i \cdot x_i} + \sum_{(i,j) \in E} T_{i,j}^{\text{network}} \right)$$

* **Layer Conservation**: $\sum_{i \in N} l_i = 64$
* **VRAM Constraint**: $l_i \cdot M_{\text{layer}} + M_{\text{KV}} \le V_i^{\text{usable}} \cdot x_i$
* **Thermal Throttling**: If $T_{\text{GPU}} > 85^\circ\text{C}$, node throughput $P_i$ is penalized by 50%.
* **High Latency Drop**: If network RTT > 300ms, the node is evicted and remaining nodes re-slice the model.

### 2. Canary Trap Tensor Verification
Measures relative $L_2$ Euclidean distance mismatch:

$$S_{\text{mismatch}} = \frac{\|v_{\text{actual}} - v_{\text{ref}}\|_2}{\|v_{\text{ref}}\|_2}$$

* $S_{\text{mismatch}} \le 0.003$: Normal floating-point / quantization noise.
* $0.003 < S_{\text{mismatch}} \le 0.005$: Thermal drift warning $\rightarrow$ reduce layer allocation on node.
* $S_{\text{mismatch}} > 0.005$: Hardware arithmetic corruption $\rightarrow$ immediately evict node and rebalance.

---

## Quickstart

### 1. Clone with Submodules
```bash
git clone --recursive https://github.com/tex1ure/AeroMesh-vap-2026.git AeroMesh
cd AeroMesh
```

### 2. Install & Build

#### Windows (PowerShell):
```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

#### Linux / macOS:
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

---

## Networking: Same Wi-Fi vs Different Wi-Fi Networks

### Scenario A: Laptops on the Same Wi-Fi

1. **Set Wi-Fi Profile to "Private" (Windows)**:
   ```powershell
   Get-NetConnectionProfile | Set-NetConnectionProfile -NetworkCategory Private
   ```
2. **Allow Inbound Firewall Ports (Run on each laptop)**:
   ```powershell
   # Allow RPC Worker Node (Port 50052)
   New-NetFirewallRule -DisplayName "AeroMesh RPC Node" -Direction Inbound -LocalPort 50052 -Protocol TCP -Action Allow

   # Allow Coordinator Master API (Port 8080)
   New-NetFirewallRule -DisplayName "AeroMesh Master API" -Direction Inbound -LocalPort 8080 -Protocol TCP -Action Allow

   # Allow ICMP Ping (for latency measurement)
   New-NetFirewallRule -DisplayName "AeroMesh Allow Ping" -Protocol ICMPv4 -IcmpType 8 -Direction Inbound -Action Allow
   ```
3. **Router Client Isolation**: If laptops cannot ping each other on the same Wi-Fi, open your router admin page (`192.168.1.1`) and disable **AP Isolation** / **Client Isolation**.

---

### Scenario B: Laptops on Different Wi-Fi Networks (Tailscale Mesh VPN)

If your laptops are on different Wi-Fi networks (e.g. home vs college dorm vs phone hotspot), they sit behind separate NAT firewalls and cannot reach `192.168.x.x` addresses directly.

**Use Tailscale (Zero-Config WireGuard Mesh VPN):**

1. Install [Tailscale](https://tailscale.com/download) on all laptops (Windows, macOS, Linux).
2. Sign in with the same account so all machines join your secure virtual network (**Tailnet**).
3. Find each laptop's virtual Tailscale IP (e.g. `100.x.y.z`):
   ```bash
   tailscale ip -4
   ```
4. Put the Tailscale IP addresses directly into `config/cluster.yaml`:
   ```yaml
   nodes:
     - name: "Laptop_A"
       ip: "100.85.12.34"
       rpc_port: 50052
     - name: "Laptop_B"
       ip: "100.85.12.56"
       rpc_port: 50052
   ```
5. Start your nodes and coordinator normally. Tailscale automatically handles encrypted NAT traversal and direct peer-to-peer UDP hole punching across different Wi-Fi networks with zero router port forwarding required.

---

## CLI Usage

### Check Node Status & VRAM
```bash
aeromesh status
```

### Run ILP Layer Slicer
Calculates optimal layer distribution across active nodes:
```bash
aeromesh rebalance
```

### Start a Worker Node (e.g. on Laptop B or C)
```bash
aeromesh node --name Laptop_B --port 50052
```

### Start the Coordinator (on Laptop A)
```bash
aeromesh start --model Qwen2.5-32B-Instruct-Q4_K_M.gguf --port 8080
```

### Network Chaos Testing (via Toxiproxy)
Inject 500ms latency onto Laptop C to test dynamic rebalancing:
```bash
aeromesh chaos inject --node Laptop_C --latency 500
aeromesh chaos clear --node Laptop_C
```

---

## Project Structure

```text
AeroMesh/
├── config/
│   └── cluster.yaml.example     # 3-Laptop benchmark topology configuration
├── models/
│   └── .gitkeep                 # Local model weights directory (GGUFs gitignored)
├── scripts/
│   ├── setup.ps1                # Windows PowerShell automated setup
│   ├── setup.sh                 # Linux/macOS bash setup
│   ├── setup.bat                # Windows CMD setup script
│   └── run_integration_test.ps1 # Integration test suite
├── src/llama_cluster/
│   ├── canary_validator.py      # Canary Trap tensor validator (L2 distance)
│   ├── chaos.py                 # Toxiproxy network chaos client
│   ├── cli.py                   # CLI commands (status, rebalance, start, node)
│   ├── config.py                # Root-relative path and environment loader
│   ├── graph_compiler.py        # Integer Linear Programming layer solver (PuLP)
│   ├── node.py                  # Worker node daemon (spawns llama-rpc-server)
│   ├── orchestrator.py          # Master control plane & between-turn rebalancer
│   └── telemetry.py             # Dual-tier hardware collector (pynvml + psutil)
├── tests/                       # Pytest unit & integration test suite
├── .env.example                 # Safe environment configuration template
├── CONTRIBUTING.md              # Contributor guidelines & architecture rules
├── LICENSE                      # MIT License
└── pyproject.toml               # Package dependencies & tool configs
```

---

## Tests

Run the test suite:
```bash
uv run pytest
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
