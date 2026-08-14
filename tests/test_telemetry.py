"""
Tests for AeroMesh 200ms JSON stream telemetry collector and dual-tier fallback logic.
Specification Reference: Section 5.2 & Section 6.2
"""

from llama_cluster.telemetry import TelemetryCollector, get_telemetry


def test_telemetry_payload_structure():
    """Verify 200ms JSON telemetry payload structure per AeroMesh Spec Section 6.2."""
    collector = get_telemetry("Laptop_A_Test")
    payload = collector.get_telemetry_payload("127.0.0.1")

    assert "node_id" in payload
    assert payload["node_id"] == "Laptop_A_Test"
    assert "timestamp_ms" in payload
    assert "metrics" in payload

    metrics = payload["metrics"]
    assert "gpu_temp_celsius" in metrics
    assert "gpu_power_draw_watts" in metrics
    assert "vram_free_bytes" in metrics
    assert "ram_free_bytes" in metrics
    assert "cpu_utilization_percent" in metrics
    assert "network_rtt_to_coordinator_ms" in metrics

    collector.close()
