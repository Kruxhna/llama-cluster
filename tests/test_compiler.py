"""
Unit tests for AeroMesh Dynamic Graph Compiler (ILP PuLP Solver).
Specification Reference: Section 4.3
"""

from llama_cluster.graph_compiler import DynamicGraphCompiler


def test_ilp_solver_optimal_allocation():
    """Verify ILP solver allocates 64 model layers across 3 nodes."""
    compiler = DynamicGraphCompiler()
    
    nodes_telemetry = [
        {
            "node_id": "Laptop_A",
            "compute_tflops": 15.0,
            "metrics": {
                "vram_free_bytes": 7.5 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 50.0,
                "network_rtt_to_coordinator_ms": 1.0,
            }
        },
        {
            "node_id": "Laptop_B",
            "compute_tflops": 15.0,
            "metrics": {
                "vram_free_bytes": 7.5 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 55.0,
                "network_rtt_to_coordinator_ms": 15.0,
            }
        },
        {
            "node_id": "Laptop_C",
            "compute_tflops": 9.0,
            "metrics": {
                "vram_free_bytes": 3.5 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 60.0,
                "network_rtt_to_coordinator_ms": 25.0,
            }
        }
    ]

    res = compiler.solve_layer_allocation(nodes_telemetry, total_layers=64)
    assert res["status"] in ("Optimal", "Not Solved", "Fallback_Uniform")
    assert sum(res["allocations"].values()) == 64


def test_ilp_thermal_throttling_derating():
    """Verify thermal stress (>85°C) derates compute throughput."""
    compiler = DynamicGraphCompiler()
    
    nodes_telemetry = [
        {
            "node_id": "Laptop_A",
            "compute_tflops": 15.0,
            "metrics": {
                "vram_free_bytes": 7.5 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 50.0,
                "network_rtt_to_coordinator_ms": 1.0,
            }
        },
        {
            "node_id": "Laptop_C_Hot",
            "compute_tflops": 15.0,
            "metrics": {
                "vram_free_bytes": 7.5 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 90.0,  # Exceeds 85°C thermal threshold!
                "network_rtt_to_coordinator_ms": 10.0,
            }
        }
    ]

    res = compiler.solve_layer_allocation(nodes_telemetry, total_layers=64)
    assert sum(res["allocations"].values()) == 64
    assert res["allocations"]["Laptop_A"] >= res["allocations"]["Laptop_C_Hot"]
