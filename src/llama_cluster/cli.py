"""
AeroMesh CLI Interface.
Entry point for managing distributed LLM inference clusters, hardware telemetry,
ILP graph compiler rebalancing, and network chaos testing.
"""

import argparse
from pathlib import Path
import sys
from typing import List, Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from llama_cluster import __version__
from llama_cluster.config import get_config
from llama_cluster.telemetry import get_telemetry
from llama_cluster.downloader import download_gguf_model
from llama_cluster.node import run_node
from llama_cluster.orchestrator import AeroMeshOrchestrator, run_orchestrator
from llama_cluster.chaos import ToxiproxyChaosEngine


def show_banner():
    """Prints banner for AeroMesh / llama-cluster."""
    banner = f"""
 [LLAMA] ============================================================= [LLAMA]
    AEROMESH (llama-cluster) v{__version__} - P2P Topology-Aware Engine
    "Dynamic Graph Slicing & Byzantine Fault Tolerance over Wi-Fi"
 [LLAMA] ============================================================= [LLAMA]
"""
    print(banner)


def cmd_status(args: argparse.Namespace):
    """Displays hardware, VRAM, ILP topology, and model status."""
    show_banner()
    cfg = get_config()
    telemetry = get_telemetry("Local_Node")
    payload = telemetry.get_telemetry_payload()
    metrics = payload["metrics"]

    print(f"[*] Cluster Name           : {cfg.topology.get('cluster_name', 'AeroMesh-Cluster')}")
    print(f"[*] Repository Root        : {cfg.repo_root}")
    print(f"[*] Model Storage Path     : {cfg.model_dir}")
    print(f"[*] Telemetry Target       : {'NVIDIA NVML Active' if telemetry.nvml_initialized else 'CPU / psutil Fallback'}")
    print(f"[*] Telemetry Interval     : {cfg.telemetry_interval_ms} ms")
    
    print("\n--- Local Machine Metrics (200ms Telemetry Stream) ---")
    print(f"  - GPU Temp        : {metrics['gpu_temp_celsius']} C")
    print(f"  - GPU Power       : {metrics['gpu_power_draw_watts']} W")
    print(f"  - VRAM Free       : {metrics['vram_free_bytes'] / (1024*1024*1024):.2f} GB")
    print(f"  - System RAM Free : {metrics['ram_free_bytes'] / (1024*1024*1024):.2f} GB")
    print(f"  - CPU Load        : {metrics['cpu_utilization_percent']} %")

    # Display configured cluster nodes & live reachability
    nodes = cfg.topology.get("nodes", [])
    print(f"\n--- Cluster Topology ({len(nodes)} Configured Nodes) ---")
    for n in nodes:
        name = n.get("name", "Node")
        ip = n.get("ip", "127.0.0.1")
        port = n.get("rpc_port", 50052)
        vram = n.get("usable_vram_gb", 0.0)
        gpu = n.get("gpu_model", "GPU")
        
        # Ping target node to check connection latency
        rtt = telemetry.measure_network_rtt(ip, port=port, timeout=0.8)
        if rtt < 900:
            link_status = f"ONLINE (RTT: {rtt:.1f} ms)"
        else:
            link_status = "READY (Awaiting node daemon start)"

        print(f"  * [{name}] @ {ip}:{port}")
        print(f"      GPU Model     : {gpu}")
        print(f"      Usable VRAM   : {vram} GB")
        print(f"      Link Status   : {link_status}")

    # List GGUF models
    models = list(cfg.model_dir.glob("*.gguf")) if cfg.model_dir.exists() else []
    print("\n--- Local GGUF Model Weights ---")
    if models:
        for m in models:
            size_gb = m.stat().st_size / (1024 * 1024 * 1024)
            print(f"  - {m.name} ({size_gb:.2f} GB)")
    else:
        print("  (No GGUF models downloaded yet. Run `aeromesh download` to fetch weights!)")

    telemetry.close()


def cmd_rebalance(args: argparse.Namespace):
    """Triggers Dynamic Graph Compiler ILP solver manually."""
    show_banner()
    print("[*] Running Dynamic Graph Compiler (ILP PuLP Solver)...")
    orchestrator = AeroMeshOrchestrator()
    result = orchestrator.rebalance_cluster_layers()

    print(f"[+] ILP Solver Status: {result['status']}")
    print(f"[+] Target Model Layers: {result['total_layers']}")
    print("--- Calculated Layer Slices per Node ---")
    for nid, layers in result["allocations"].items():
        print(f"  - Node {nid}: {layers} layers assigned")

    if result["evicted_nodes"]:
        print(f"[!] Bottleneck Evicted Nodes: {', '.join(result['evicted_nodes'])}")


def cmd_chaos(args: argparse.Namespace):
    """Triggers Toxiproxy network chaos injection."""
    show_banner()
    chaos = ToxiproxyChaosEngine()
    if not chaos.is_available():
        print("[!] Toxiproxy server not detected at http://127.0.0.1:8474.")
        print("[!] Make sure toxiproxy-server.exe is running for chaos testing!")
        return

    if args.action == "inject":
        print(f"[*] Injecting {args.latency}ms latency spike onto node {args.node}...")
        success = chaos.inject_latency_toxic(args.node, latency_ms=args.latency)
        if success:
            print(f"[+] Chaos injected! Laptop C simulated latency spike active. We cookin.")
        else:
            print("[!] Failed to inject toxic.")
    elif args.action == "clear":
        print(f"[*] Clearing network perturbations on node {args.node}...")
        chaos.remove_toxics(args.node)
        print("[+] Network link restored to normal.")


def cmd_start(args: argparse.Namespace):
    show_banner()
    run_orchestrator(model_name=args.model, port=args.port)


def cmd_node(args: argparse.Namespace):
    show_banner()
    run_node(node_name=args.name, port=args.port)


def cmd_download(args: argparse.Namespace):
    show_banner()
    download_gguf_model(repo_id=args.repo, filename=args.file)


def cmd_init(args: argparse.Namespace):
    show_banner()
    cfg = get_config()
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    config_dir = cfg.repo_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)

    env_dst = cfg.repo_root / ".env"
    env_example = cfg.repo_root / ".env.example"
    if not env_dst.exists() and env_example.exists():
        env_dst.write_text(env_example.read_text(encoding="utf-8"), encoding="utf-8")
        print("[+] Created .env from .env.example! Lock in your settings bro.")

    yaml_dst = config_dir / "cluster.yaml"
    yaml_example = config_dir / "cluster.yaml.example"
    if not yaml_dst.exists() and yaml_example.exists():
        yaml_dst.write_text(yaml_example.read_text(encoding="utf-8"), encoding="utf-8")
        print("[+] Created config/cluster.yaml 3-laptop benchmark profile! No cap.")

    print("[+] AeroMesh environment initialization complete. We ready to cook!")


def cmd_dashboard(args: argparse.Namespace):
    show_banner()
    from llama_cluster.dashboard import run_dashboard
    run_dashboard(host=args.host, port=args.port)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(
        prog="aeromesh",
        description="AeroMesh (llama-cluster): Fault-tolerant P2P LLM orchestration engine over native llama.cpp RPC."
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="AeroMesh subcommands")

    # Dashboard
    parser_dashboard = subparsers.add_parser("dashboard", help="Start Web Control Dashboard & Topology Manager")
    parser_dashboard.add_argument("--host", default="0.0.0.0", help="Binding host (default: 0.0.0.0)")
    parser_dashboard.add_argument("-p", "--port", type=int, default=3000, help="Dashboard web port (default: 3000)")
    parser_dashboard.set_defaults(func=cmd_dashboard)

    # Status
    parser_status = subparsers.add_parser("status", help="Show 200ms telemetry, VRAM, and model status")
    parser_status.set_defaults(func=cmd_status)

    # Rebalance
    parser_rebalance = subparsers.add_parser("rebalance", help="Run ILP Dynamic Graph Compiler layer solver")
    parser_rebalance.set_defaults(func=cmd_rebalance)

    # Chaos
    parser_chaos = subparsers.add_parser("chaos", help="Toxiproxy network perturbation injection")
    parser_chaos.add_argument("action", choices=["inject", "clear"], help="Chaos action")
    parser_chaos.add_argument("-n", "--node", default="Laptop_C", help="Target node proxy name")
    parser_chaos.add_argument("-l", "--latency", type=int, default=500, help="Latency spike in ms")
    parser_chaos.set_defaults(func=cmd_chaos)

    # Start
    parser_start = subparsers.add_parser("start", help="Start master coordinator (Laptop A)")
    parser_start.add_argument("-m", "--model", help="GGUF model filename in ./models/")
    parser_start.add_argument("-p", "--port", type=int, help="Master HTTP port (default: 8080)")
    parser_start.set_defaults(func=cmd_start)

    # Node
    parser_node = subparsers.add_parser("node", help="Start worker node daemon (Laptop B/C)")
    parser_node.add_argument("-n", "--name", default="Laptop_B", help="Worker node identifier")
    parser_node.add_argument("-p", "--port", type=int, default=50052, help="RPC worker port (default: 50052)")
    parser_node.set_defaults(func=cmd_node)

    # Download
    parser_download = subparsers.add_parser("download", help="Download GGUF model weights from HF")
    parser_download.add_argument("-r", "--repo", default="Qwen/Qwen2.5-32B-Instruct-GGUF", help="HF Repo ID")
    parser_download.add_argument("-f", "--file", default="Qwen2.5-32B-Instruct-Q4_K_M.gguf", help="GGUF Filename")
    parser_download.set_defaults(func=cmd_download)

    # Init
    parser_init = subparsers.add_parser("init", help="Initialize local environment & config")
    parser_init.set_defaults(func=cmd_init)

    args = parser.parse_args(argv)
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
