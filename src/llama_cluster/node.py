"""
Worker Node Daemon for llama-cluster.
Runs on each machine in the cluster, exposes telemetry, and spawns llama-rpc-server instances.
"""

from pathlib import Path
import subprocess
import threading
from typing import Dict, Any, Optional
import time
import sys
from llama_cluster.config import Config, get_config
from llama_cluster.telemetry import get_telemetry
from llama_cluster.pipeline_bridge import ActivationPipelineServer


class NodeDaemon:
    """Manages worker node state, RPC server lifecycle, and telemetry reporting."""

    def __init__(
        self,
        node_name: str = "worker-01",
        rpc_port: int = 50052,
        mode: str = "auto",
        model_name: Optional[str] = None,
        cfg: Optional[Config] = None
    ):
        self.config = cfg or get_config()
        self.node_name = node_name
        self.rpc_port = rpc_port
        self.mode = mode
        self.model_name = model_name
        self.telemetry = get_telemetry()
        self.rpc_process: Optional[subprocess.Popen] = None
        self.pipeline_server: Optional[ActivationPipelineServer] = None
        self.pipeline_thread: Optional[threading.Thread] = None

    def get_status(self) -> Dict[str, Any]:
        """Returns node health and hardware stats."""
        gpus = self.telemetry.get_gpu_stats()
        is_running = (self.rpc_process is not None and self.rpc_process.poll() is None) or (self.pipeline_server is not None)
        return {
            "node_name": self.node_name,
            "rpc_port": self.rpc_port,
            "rpc_running": is_running,
            "gpus": gpus,
            "timestamp": time.time(),
        }

    def find_rpc_binary(self) -> Optional[Path]:
        """Looks up ggml-rpc-server, llama-rpc-server, or rpc-server binary in build directories."""
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

    def start_pipeline_server(self, host: str = "0.0.0.0"):
        """Starts Zero-Weight Activation Pipeline server for direct local SSD model execution."""
        try:
            model_path = self.config.get_model_path(self.model_name)
            print(f"[*] Starting Zero-Weight Activation Pipeline server on {host}:{self.rpc_port}...")
            print(f"[*] Local Model SSD Path: {model_path}")
            self.pipeline_server = ActivationPipelineServer(
                host=host,
                port=self.rpc_port,
                layer_start=24,
                layer_end=48,
                local_model_path=model_path
            )
            self.pipeline_thread = threading.Thread(target=self.pipeline_server.serve_forever, daemon=True)
            self.pipeline_thread.start()
            print(f"[+] Activation Pipeline server ACTIVE! Zero-weight transfer enabled.")
            return True
        except Exception as e:
            print(f"[!] Failed to start activation pipeline server: {e}")
            return False

    def start_rpc_server(self, host: str = "0.0.0.0"):
        """Spawns local llama-rpc-server backend process with local disk cache (-c)."""
        binary = self.find_rpc_binary()
        if not binary:
            print(f"[!] Note: llama-rpc-server binary not found, using pure Zero-Weight Pipeline server.")
            return self.start_pipeline_server(host=host)

        cmd = [
            str(binary),
            "-H", host,
            "-p", str(self.rpc_port),
            "-c",  # Enable local disk tensor caching to eliminate network transfers on subsequent runs
        ]
        print(f"[*] Spawning RPC node daemon on {host}:{self.rpc_port} (with local cache -c)...")
        print(f"[*] Executable: {binary}")

        try:
            self.rpc_process = subprocess.Popen(cmd)
            time.sleep(1.0)
            if self.rpc_process.poll() is not None:
                print(f"[!] RPC process terminated immediately with exit code {self.rpc_process.returncode}")
                return self.start_pipeline_server(host=host)

            print(f"[+] RPC worker node {self.node_name} active on port {self.rpc_port}! Ready for incoming tensor allocations.")
            return True

        except Exception as e:
            print(f"[!] Failed to spawn RPC server process: {e}")
            return self.start_pipeline_server(host=host)

    def stop(self):
        """Stops all running worker server processes."""
        if self.rpc_process and self.rpc_process.poll() is None:
            print(f"[*] Shutting down RPC node process on port {self.rpc_port}...")
            self.rpc_process.terminate()
            try:
                self.rpc_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rpc_process.kill()
            print(f"[+] Node process stopped cleanly.")
            self.rpc_process = None

        if self.pipeline_server:
            print("[*] Shutting down Zero-Weight Activation Pipeline server...")
            self.pipeline_server.shutdown()
            self.pipeline_server.server_close()
            self.pipeline_server = None


def run_node(node_name: str = "worker-node-1", port: int = 50052, mode: str = "auto", model: Optional[str] = None):
    """CLI runner function for launching a worker node."""
    daemon = NodeDaemon(node_name=node_name, rpc_port=port, mode=mode, model_name=model)
    print(f"=== AeroMesh Worker Node: {node_name} ===")
    status = daemon.get_status()
    print(f"[*] Telemetry initialized. Detected GPUs: {len(status['gpus'])}")
    for gpu in status["gpus"]:
        print(f"    - {gpu['name']} | VRAM: {gpu['vram_used_mb']}/{gpu['vram_total_mb']} MB ({gpu['vram_utilization_percent']}%)")

    if mode == "local-pipeline":
        daemon.start_pipeline_server()
    else:
        daemon.start_rpc_server()

    print("[*] Press Ctrl+C to exit node daemon.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[*] Exiting node daemon...")
        daemon.stop()
        daemon.telemetry.close()

