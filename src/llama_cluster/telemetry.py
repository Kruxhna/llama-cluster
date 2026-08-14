"""
AeroMesh Defensive Telemetry Engine.
Dual-tier hardware telemetry collector using pynvml, psutil, and network RTT ping.
Generates 200ms JSON telemetry payloads per AeroMesh Specification Section 6.2.
"""

import time
import sys
import socket
from typing import Dict, Any, Optional

import warnings

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import pynvml
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class TelemetryCollector:
    """Dual-tier hardware telemetry collector with NVML and psutil fallback."""

    def __init__(self, node_id: str = "Laptop_A"):
        self.node_id = node_id
        self.nvml_initialized = False
        if HAS_NVML:
            try:
                pynvml.nvmlInit()
                self.nvml_initialized = True
            except Exception:
                # Defensive fallback for non-NVIDIA or missing driver nodes
                self.nvml_initialized = False

    def measure_network_rtt(self, target_host: str, port: int = 8080, timeout: float = 1.0) -> float:
        """Measures network RTT latency to target host in milliseconds."""
        if target_host in ("127.0.0.1", "localhost", "0.0.0.0"):
            return 0.5  # Local loopback RTT

        start = time.perf_counter()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((target_host, port))
            sock.close()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return round(elapsed_ms, 2)
        except Exception:
            return 999.9  # Latency penalty for unreachable nodes

    def get_telemetry_payload(self, coordinator_host: str = "127.0.0.1") -> Dict[str, Any]:
        """
        Generates 200ms JSON telemetry stream payload.
        Complies with AeroMesh Specification Section 6.2.
        """
        timestamp_ms = int(time.time() * 1000)

        # Host RAM & CPU metrics via psutil
        ram_free_bytes = 0
        cpu_util_pct = 0.0
        if HAS_PSUTIL:
            try:
                ram_info = psutil.virtual_memory()
                ram_free_bytes = ram_info.available
                cpu_util_pct = psutil.cpu_percent(interval=None)
            except Exception:
                pass

        # GPU metrics via pynvml
        gpu_temp_celsius = 0.0
        gpu_power_watts = 0.0
        vram_free_bytes = 0

        if self.nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                vram_free_bytes = mem.free

                try:
                    gpu_temp_celsius = float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    gpu_temp_celsius = 0.0

                try:
                    gpu_power_watts = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
                except Exception:
                    gpu_power_watts = 0.0
            except Exception:
                pass
        else:
            # CPU/System RAM fallback
            vram_free_bytes = ram_free_bytes

        # Network RTT measurement
        rtt_ms = self.measure_network_rtt(coordinator_host)

        return {
            "node_id": self.node_id,
            "timestamp_ms": timestamp_ms,
            "metrics": {
                "gpu_temp_celsius": round(gpu_temp_celsius, 1),
                "gpu_power_draw_watts": round(gpu_power_watts, 1),
                "vram_free_bytes": vram_free_bytes,
                "ram_free_bytes": ram_free_bytes,
                "cpu_utilization_percent": round(cpu_util_pct, 1),
                "network_rtt_to_coordinator_ms": rtt_ms,
            }
        }

    def close(self):
        """Cleanly releases NVML resources."""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
                self.nvml_initialized = False
            except Exception:
                pass


def get_telemetry(node_id: str = "Laptop_A") -> TelemetryCollector:
    return TelemetryCollector(node_id=node_id)
