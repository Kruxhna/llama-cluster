"""
AeroMesh Configuration Manager.
Resolves root-relative paths dynamically for cross-platform portability across Windows, Linux, and macOS.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

ENV_FILE = REPO_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    load_dotenv()


class Config:
    """Centralized AeroMesh configuration container."""

    def __init__(self, config_path: Optional[Path] = None):
        self.repo_root: Path = REPO_ROOT
        
        # Network & Master Control Plane
        self.host: str = os.getenv("LLAMA_CLUSTER_HOST", "0.0.0.0")
        self.port: int = int(os.getenv("LLAMA_CLUSTER_PORT", "8080"))
        self.secret: str = os.getenv("LLAMA_CLUSTER_SECRET", "super_secret_aeromesh_token_2026")

        # System Paths
        raw_llama_cpp = os.getenv("LLAMA_CPP_DIR", "./llama.cpp")
        self.llama_cpp_dir: Path = (REPO_ROOT / raw_llama_cpp).resolve() if not Path(raw_llama_cpp).is_absolute() else Path(raw_llama_cpp)

        raw_model_dir = os.getenv("MODEL_DIR", "./models")
        self.model_dir: Path = (REPO_ROOT / raw_model_dir).resolve() if not Path(raw_model_dir).is_absolute() else Path(raw_model_dir)

        self.default_model: str = os.getenv("DEFAULT_MODEL", "Qwen2.5-32B-Instruct-Q4_K_M.gguf")
        self.hf_token: Optional[str] = os.getenv("HF_TOKEN")

        # AeroMesh Telemetry & Health Monitoring
        self.telemetry_interval_ms: int = int(os.getenv("TELEMETRY_INTERVAL_MS", "200"))
        self.thermal_throttle_temp_c: float = float(os.getenv("THERMAL_THROTTLE_TEMP_C", "85.0"))
        self.max_vram_percent: float = float(os.getenv("MAX_VRAM_USAGE_PERCENT", "90.0"))

        # Byzantine Tensor Validator (Canary Trap)
        self.canary_sample_rate: float = float(os.getenv("CANARY_SAMPLE_RATE", "0.05"))
        self.canary_warning_threshold: float = float(os.getenv("CANARY_WARNING_THRESHOLD", "0.003"))
        self.canary_eviction_threshold: float = float(os.getenv("CANARY_EVICTION_THRESHOLD", "0.005"))

        # Toxiproxy Chaos Engine
        self.toxiproxy_host: str = os.getenv("TOXIPROXY_HOST", "127.0.0.1")
        self.toxiproxy_port: int = int(os.getenv("TOXIPROXY_PORT", "8474"))

        # Topology File
        self.config_path: Path = config_path or (REPO_ROOT / "config" / "cluster.yaml")
        self.topology: Dict[str, Any] = self._load_topology()

    def _load_topology(self) -> Dict[str, Any]:
        """Loads cluster allocation profile from cluster.yaml or returns 3-laptop default."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[!] Warning: Failed to parse topology config at {self.config_path}: {e}")

        example_path = self.config_path.with_name("cluster.yaml.example")
        if example_path.exists():
            try:
                with open(example_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                pass

        # Default 3-laptop benchmark topology (Section 2.1)
        return {
            "cluster_name": "aeromesh-3-laptop-mismatched-cluster",
            "model_spec": {
                "name": "Qwen2.5-32B-Instruct",
                "total_layers": 64,
                "file": "Qwen2.5-32B-Instruct-Q4_K_M.gguf"
            },
            "nodes": [
                {
                    "name": "Laptop_A",
                    "ip": "127.0.0.1",
                    "rpc_port": 50052,
                    "usable_vram_gb": 7.5,
                    "compute_tflops": 15.0
                },
                {
                    "name": "Laptop_B",
                    "ip": "192.168.1.102",
                    "rpc_port": 50052,
                    "usable_vram_gb": 7.5,
                    "compute_tflops": 15.0
                },
                {
                    "name": "Laptop_C",
                    "ip": "192.168.1.103",
                    "rpc_port": 50052,
                    "usable_vram_gb": 3.5,
                    "compute_tflops": 9.0
                }
            ]
        }

    def get_model_path(self, model_name: Optional[str] = None) -> Path:
        """Returns root-relative path to requested GGUF weight file."""
        target_name = model_name or self.default_model
        return self.model_dir / target_name


def get_config(config_path: Optional[Path] = None) -> Config:
    return Config(config_path=config_path)
