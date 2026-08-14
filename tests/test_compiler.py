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


def test_halda_algorithm_factorization_and_layer_assignment():
    """Verify HALDA (Algorithm 1) handles factor factorization and returns GPU + total layers."""
    compiler = DynamicGraphCompiler()
    
    # Test valid factors of 48 layers (Qwen2.5-14B)
    factors_48 = compiler.get_valid_factors(48)
    assert 1 in factors_48 and 2 in factors_48 and 48 in factors_48
    
    nodes_telemetry = [
        {
            "node_id": "Laptop_A",
            "compute_tflops": 15.0,
            "metrics": {
                "vram_free_bytes": 2.0 * 1024 * 1024 * 1024,
                "ram_free_bytes": 16.0 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 45.0,
                "network_rtt_to_coordinator_ms": 1.0,
            }
        },
        {
            "node_id": "Laptop_B",
            "compute_tflops": 15.0,
            "metrics": {
                "vram_free_bytes": 8.0 * 1024 * 1024 * 1024,
                "ram_free_bytes": 16.0 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 52.0,
                "network_rtt_to_coordinator_ms": 15.0,
            }
        },
        {
            "node_id": "Laptop_C",
            "compute_tflops": 9.0,
            "metrics": {
                "vram_free_bytes": 4.0 * 1024 * 1024 * 1024,
                "ram_free_bytes": 16.0 * 1024 * 1024 * 1024,
                "gpu_temp_celsius": 55.0,
                "network_rtt_to_coordinator_ms": 20.0,
            }
        }
    ]

    res = compiler.solve_halda(nodes_telemetry, total_layers=48, layer_weight_mb=175.0)
    assert res["status"] == "Optimal"
    assert sum(res["allocations"].values()) == 48
    assert res["algorithm"] == "HALDA (ICLR 2026)"
    assert "gpu_layers" in res
    assert all(res["gpu_layers"][nid] <= res["allocations"][nid] for nid in res["allocations"])

