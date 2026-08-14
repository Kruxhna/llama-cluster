"""
Worker Node Daemon for llama-cluster.
Runs on each machine in the cluster, exposes telemetry, and spawns llama-rpc-server instances.
"""

from pathlib import Path
import subprocess
from typing import Dict, Any, Optional
import time
import sys
from llama_cluster.config import Config, get_config
from llama_cluster.telemetry import get_telemetry


class NodeDaemon:
    """Manages worker node state, RPC server lifecycle, and telemetry reporting."""

    def __init__(self, node_name: str = "worker-01", rpc_port: int = 50052, cfg: Optional[Config] = None):
        self.config = cfg or get_config()
        self.node_name = node_name
        self.rpc_port = rpc_port
        self.telemetry = get_telemetry()
        self.rpc_process: Optional[subprocess.Popen] = None

    def get_status(self) -> Dict[str, Any]:
        """Returns node health and hardware stats."""
        gpus = self.telemetry.get_gpu_stats()
        return {
            "node_name": self.node_name,
            "rpc_port": self.rpc_port,
            "rpc_running": self.rpc_process is not None and self.rpc_process.poll() is None,
            "gpus": gpus,
            "timestamp": time.time(),
        }

    def find_rpc_binary(self) -> Optional[Path]:
        """Looks up llama-rpc-server, ggml-rpc-server, or rpc-server binary in build directories."""
        candidates = [
            self.config.llama_cpp_dir / "build" / "bin" / "Release" / "ggml-rpc-server.exe",
            self.config.llama_cpp_dir / "build" / "bin" / "Release" / "llama-rpc-server.exe",
            self.config.llama_cpp_dir / "build" / "bin" / "Release" / "rpc-server.exe",
            self.config.llama_cpp_dir / "build" / "bin" / "ggml-rpc-server.exe",
            self.config.llama_cpp_dir / "build" / "bin" / "llama-rpc-server.exe",
            self.config.llama_cpp_dir / "build" / "bin" / "rpc-server.exe",
            self.config.llama_cpp_dir / "build" / "bin" / "Release" / "ggml-rpc-server",
            self.config.llama_cpp_dir / "build" / "bin" / "ggml-rpc-server",
            self.config.llama_cpp_dir / "build" / "bin" / "llama-rpc-server",
            self.config.llama_cpp_dir / "build" / "bin" / "rpc-server",
            self.config.llama_cpp_dir / "rpc-server.exe",
            self.config.llama_cpp_dir / "rpc-server",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def start_rpc_server(self, host: str = "0.0.0.0"):
        """Spawns local llama-rpc-server backend process."""
        binary = self.find_rpc_binary()
        if not binary:
            print(f"[!] Warning: llama-rpc-server binary not found under {self.config.llama_cpp_dir}/build/bin")
            print(f"[!] Please build llama.cpp first (e.g. via CMake or bash scripts/setup.sh).")
            return False

        cmd = [
            str(binary),
            "-H", host,
            "-p", str(self.rpc_port),
        ]
        print(f"[*] Spawning RPC node daemon on {host}:{self.rpc_port}...")
        print(f"[*] Executable: {binary}")

        try:
            self.rpc_process = subprocess.Popen(cmd)
            time.sleep(1.0)
            if self.rpc_process.poll() is not None:
                print(f"[!] RPC process terminated immediately with exit code {self.rpc_process.returncode}")
                return False

            print(f"[+] RPC worker node {self.node_name} active on port {self.rpc_port}! Ready for incoming tensor allocations.")
            return True

        except Exception as e:
            print(f"[!] Failed to spawn RPC server process: {e}")
            return False

    def stop_rpc_server(self):
        """Stops running RPC server process."""
        if self.rpc_process and self.rpc_process.poll() is None:
            print(f"[*] Shutting down RPC node process on port {self.rpc_port}...")
            self.rpc_process.terminate()
            try:
                self.rpc_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rpc_process.kill()
            print(f"[+] Node process stopped cleanly.")
            self.rpc_process = None


def run_node(node_name: str = "worker-node-1", port: int = 50052):
    """CLI runner function for launching a worker node."""
    daemon = NodeDaemon(node_name=node_name, rpc_port=port)
    print(f"=== llama-cluster Worker Node: {node_name} ===")
    status = daemon.get_status()
    print(f"[*] Telemetry initialized. Detected GPUs: {len(status['gpus'])}")
    for gpu in status["gpus"]:
        print(f"    - {gpu['name']} | VRAM: {gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB ({gpu['vram_utilization_percent']}%)")

    success = daemon.start_rpc_server()
    if not success:
        print("[!] Running in telemetry-only node mode (no active llama-rpc-server binary binary built yet).")

    print("[*] Press Ctrl+C to exit node daemon.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[*] Exiting node daemon...")
        daemon.stop_rpc_server()
        daemon.telemetry.close()
