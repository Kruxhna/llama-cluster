"""
AeroMesh Chaos Testing Infrastructure.
Integrates with Toxiproxy (Shopify's standalone network proxy) for injecting latency spikes,
bandwidth restrictions, and packet loss routines during fault-tolerance test cycles.
Specification Reference: Section 5.3 & Section 7.2
"""

from typing import Dict, Any, Optional
import requests
from llama_cluster.config import Config, get_config


class ToxiproxyChaosEngine:
    """Controls network perturbation experiments via Toxiproxy REST API."""

    def __init__(self, cfg: Optional[Config] = None):
        self.config = cfg or get_config()
        self.api_url = f"http://{self.config.toxiproxy_host}:{self.config.toxiproxy_port}"

    def is_available(self) -> bool:
        """Checks if toxiproxy-server is active."""
        try:
            res = requests.get(f"{self.api_url}/version", timeout=1.0)
            return res.status_code == 200
        except Exception:
            return False

    def create_proxy(self, name: str, listen_addr: str, upstream_addr: str) -> bool:
        """Creates a proxy link for a worker node."""
        payload = {
            "name": name,
            "listen": listen_addr,
            "upstream": upstream_addr,
            "enabled": True
        }
        try:
            res = requests.post(f"{self.api_url}/proxies", json=payload, timeout=2.0)
            return res.status_code in (200, 201)
        except Exception as e:
            print(f"[!] Warning: Failed to create toxiproxy link {name}: {e}")
            return False

    def inject_latency_toxic(self, proxy_name: str, latency_ms: int = 500, jitter_ms: int = 50) -> bool:
        """Injects network latency onto node link (e.g. 500ms latency on Laptop C)."""
        payload = {
            "name": "latency_spike",
            "type": "latency",
            "stream": "downstream",
            "toxicity": 1.0,
            "attributes": {
                "latency": latency_ms,
                "jitter": jitter_ms
            }
        }
        try:
            url = f"{self.api_url}/proxies/{proxy_name}/toxics"
            res = requests.post(url, json=payload, timeout=2.0)
            return res.status_code in (200, 201)
        except Exception as e:
            print(f"[!] Failed to inject latency toxic: {e}")
            return False

    def remove_toxics(self, proxy_name: str) -> bool:
        """Removes all injected network perturbations restoring normal link bandwidth."""
        try:
            res = requests.get(f"{self.api_url}/proxies/{proxy_name}/toxics", timeout=2.0)
            if res.status_code == 200:
                toxics = res.json()
                for toxic in toxics:
                    tname = toxic.get("name")
                    requests.delete(f"{self.api_url}/proxies/{proxy_name}/toxics/{tname}", timeout=2.0)
            return True
        except Exception:
            return False
