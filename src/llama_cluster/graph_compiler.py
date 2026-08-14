"""
AeroMesh Dynamic Graph Compiler - Heterogeneity-Aware Layer-to-Device Allocation (HALDA).
Specification Reference: ICLR 2026 Paper "Heterogeneity-Aware Layer-to-Device Allocation for Distributed LLM Inference"
Section 3.3 HALDA: Automatic Layer Partitioning and Device Selection (Algorithm 1)
"""

import math
from typing import Dict, Any, List, Tuple, Optional, Set
from llama_cluster.config import Config, get_config

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class DynamicGraphCompiler:
    """
    Heterogeneity-Aware Layer-to-Device Allocation (HALDA) ILP Solver.
    Solves the Layer-to-Device Assignment (LDA) Problem minimizing Time-Per-Output-Token (TPOT):
    
        min_{w, n}  L * (a^T * w + b^T * n + e^T * c) / (e^T * w) + kappa
        
    where:
      - w_m: Total layer window allocated to device m (w_m in Z_>=0)
      - n_m: Number of GPU-accelerated layers within w_m (0 <= n_m <= w_m <= L)
      - a_m: CPU compute / memory access latency coefficient per layer
      - b_m: GPU acceleration differential per layer (b_m < 0 representing GPU speedup)
      - c_m: Interconnect network latency per activation transmission
      - kappa: Constant latency offset
      - k: Factor of L dividing the pipeline windows (W = L / k, sum(w_m) = W)
    """

    def __init__(self, cfg: Optional[Config] = None):
        self.config = cfg or get_config()

    def get_valid_factors(self, L: int) -> List[int]:
        """Calculates valid factors K_L dividing total layers L."""
        factors = []
        for i in range(1, int(math.isqrt(L)) + 1):
            if L % i == 0:
                factors.append(i)
                if i * i != L:
                    factors.append(L // i)
        factors.sort()
        return factors

    def calculate_platform_coefficients(
        self,
        node: Dict[str, Any],
        layer_weight_mb: float,
        kv_cache_mb: float,
    ) -> Dict[str, float]:
        """
        Computes device platform-specific coefficients a_m, b_m, c_m, z_m, z_m_gpu per Definition 1.
        """
        metrics = node.get("metrics", {})
        usable_vram_mb = (metrics.get("vram_free_bytes", 8 * 1024 * 1024 * 1024)) / (1024 * 1024)
        usable_ram_mb = (metrics.get("ram_free_bytes", 16 * 1024 * 1024 * 1024)) / (1024 * 1024)
        temp_c = metrics.get("gpu_temp_celsius", 50.0)
        rtt_ms = metrics.get("network_rtt_to_coordinator_ms", 10.0)
        
        base_tflops = float(node.get("compute_tflops", 15.0))
        
        # Thermal throttling derating (50% penalty if temp > 85°C)
        if temp_c > self.config.thermal_throttle_temp_c:
            base_tflops *= 0.5

        # Maximum layers fitting in VRAM (z_m_gpu)
        z_gpu = max(0, int((usable_vram_mb - kv_cache_mb) / max(1.0, layer_weight_mb)))
        
        # Maximum layers fitting in Total System Memory (RAM + VRAM) (z_m)
        z_ram = max(z_gpu, int((usable_vram_mb + usable_ram_mb - kv_cache_mb) / max(1.0, layer_weight_mb)))

        # CPU latency per layer: a_m (ms per layer on CPU)
        a_m = 12.0 / max(0.5, (base_tflops / 3.0))

        # GPU acceleration differential: b_m (ms speedup when layer runs on GPU)
        # GPU time per layer is ~10.0 / base_tflops -> b_m < 0
        gpu_time_per_layer = 10.0 / max(0.1, base_tflops)
        b_m = gpu_time_per_layer - a_m

        # Network communication cost: c_m
        c_m = rtt_ms

        return {
            "a_m": a_m,
            "b_m": b_m,
            "c_m": c_m,
            "z_gpu": z_gpu,
            "z_ram": z_ram,
            "rtt_ms": rtt_ms,
            "usable_vram_mb": usable_vram_mb,
            "usable_ram_mb": usable_ram_mb,
        }

    def solve_single_ilp(
        self,
        node_ids: List[str],
        node_coeffs: Dict[str, Dict[str, float]],
        W: int,
        k: int,
        forced_m4_nodes: Set[str],
    ) -> Optional[Tuple[Dict[str, int], Dict[str, int], float]]:
        """
        Solves the ILP for a fixed factor k and window W (Eq 6-10 in HALDA):
            min_{w, n}  k * sum(a_m * w_m + b_m * n_m + c_m * x_m) + kappa
            s.t.
              sum(w_m) == W
              0 <= n_m <= w_m <= W
              n_m <= z_gpu_m * x_m
              w_m <= z_ram_m * x_m
        """
        if not HAS_PULP:
            return None

        prob = pulp.LpProblem(f"HALDA_ILP_k{k}_W{W}", pulp.LpMinimize)

        # Decision variables:
        # w_m: total layers on device m
        # n_m: GPU layers on device m (n_m <= w_m)
        # x_m: binary active indicator
        w_vars = {nid: pulp.LpVariable(f"w_{nid}", lowBound=0, upBound=W, cat=pulp.LpInteger) for nid in node_ids}
        n_vars = {nid: pulp.LpVariable(f"n_{nid}", lowBound=0, upBound=W, cat=pulp.LpInteger) for nid in node_ids}
        x_vars = {nid: pulp.LpVariable(f"x_{nid}", cat=pulp.LpBinary) for nid in node_ids}

        # Constraint (8): sum(w_m) == W
        prob += pulp.lpSum([w_vars[nid] for nid in node_ids]) == W, "Window_Conservation"

        cost_terms = []

        for nid in node_ids:
            c = node_coeffs[nid]
            a_m = c["a_m"]
            b_m = c["b_m"]
            c_m = c["c_m"]
            z_gpu = c["z_gpu"]
            z_ram = c["z_ram"]

            # High latency eviction (>300ms RTT forces node inactive)
            if c["rtt_ms"] > 300.0:
                prob += x_vars[nid] == 0, f"Evict_{nid}_Latency"

            # Constraint (7): n_m <= w_m
            prob += n_vars[nid] <= w_vars[nid], f"GPU_Leq_Window_{nid}"

            # Constraint (10): VRAM bound n_m <= z_gpu * x_m
            if nid in forced_m4_nodes:
                prob += w_vars[nid] == n_vars[nid], f"Force_M4_{nid}"

            prob += n_vars[nid] <= max(0, z_gpu) * x_vars[nid], f"VRAM_Limit_{nid}"
            prob += w_vars[nid] <= max(1, z_ram) * x_vars[nid], f"RAM_Limit_{nid}"
            prob += w_vars[nid] <= W * x_vars[nid], f"Active_Bound_{nid}"

            # Objective cost term: k * (a_m * w_m + b_m * n_m + c_m * x_m)
            cost_terms.append(a_m * w_vars[nid] + b_m * n_vars[nid] + (c_m / 1000.0) * x_vars[nid])

        prob += k * pulp.lpSum(cost_terms), "Minimize_TPOT"

        # Suppress solver output
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        if pulp.LpStatus[prob.status] == "Optimal":
            w_res = {nid: int(round(pulp.value(w_vars[nid]))) for nid in node_ids}
            n_res = {nid: int(round(pulp.value(n_vars[nid]))) for nid in node_ids}
            obj_val = float(pulp.value(prob.objective))
            return w_res, n_res, obj_val

        return None

    def solve_halda(
        self,
        nodes_telemetry: List[Dict[str, Any]],
        total_layers: int = 48,
        layer_weight_mb: float = 265.0,
        kv_cache_mb: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Full implementation of Algorithm 1: Heterogeneity-Aware Layer-to-Device Allocation (HALDA).
        Iterative optimization over device sets M1-M4, factorizing L into sub-problems.
        """
        node_ids = [n["node_id"] for n in nodes_telemetry]
        if not node_ids:
            return {"status": "No_Nodes", "allocations": {}, "gpu_layers": {}, "active_nodes": []}

        # Dynamically scale layer weight if default was passed for smaller models
        if layer_weight_mb == 270.0 and total_layers <= 48:
            layer_weight_mb = 175.0

        # 1. Initialize platform coefficients
        node_coeffs = {
            n["node_id"]: self.calculate_platform_coefficients(n, layer_weight_mb, kv_cache_mb)
            for n in nodes_telemetry
        }

        # Initial memory-proportional allocation
        total_vram = sum(c["usable_vram_mb"] for c in node_coeffs.values())
        w = {}
        alloc_so_far = 0
        for i, nid in enumerate(node_ids):
            if i == len(node_ids) - 1:
                w[nid] = total_layers - alloc_so_far
            else:
                ratio = node_coeffs[nid]["usable_vram_mb"] / max(1.0, total_vram)
                layers_i = int(round(ratio * total_layers))
                w[nid] = layers_i
                alloc_so_far += layers_i
        n_gpu = {nid: min(w[nid], node_coeffs[nid]["z_gpu"]) for nid in node_ids}

        # 3. Calculate valid factors K_L of L
        K_L = self.get_valid_factors(total_layers)
        if 1 in K_L:
            K_L = [1] + [k for k in K_L if k != 1]

        forced_m4_nodes: Set[str] = set()
        best_w = dict(w)
        best_n = dict(n_gpu)
        best_cost = float("inf")
        solver_status = "Optimal"

        # 4. HALDA Outer Loop: Iterative Optimization for Device Sets
        prev_signature = None
        max_iterations = 10

        for _ in range(max_iterations):
            # 6. Reassign devices to sets M1..M4 based on memory headroom
            current_signature = tuple(
                (nid, w[nid] > node_coeffs[nid]["z_gpu"], nid in forced_m4_nodes)
                for nid in node_ids
            )

            if current_signature == prev_signature:
                break
            prev_signature = current_signature

            # 10. Foreach k in K_L: solve ILP
            found_solution_in_round = False
            for k in K_L:
                W = total_layers // k
                ilp_res = self.solve_single_ilp(
                    node_ids=node_ids,
                    node_coeffs=node_coeffs,
                    W=W,
                    k=k,
                    forced_m4_nodes=forced_m4_nodes
                )
                if ilp_res is not None:
                    w_sol, n_sol, cost = ilp_res
                    if k > 1:
                        w_full = {nid: w_sol[nid] * k for nid in node_ids}
                        n_full = {nid: n_sol[nid] * k for nid in node_ids}
                    else:
                        w_full = w_sol
                        n_full = n_sol

                    if cost < best_cost:
                        best_cost = cost
                        best_w = w_full
                        best_n = n_full
                        found_solution_in_round = True

            # 13. Calibration check: If a device has free VRAM while another device is overloaded
            has_free_vram = any(best_n[nid] < node_coeffs[nid]["z_gpu"] for nid in node_ids)
            has_overloaded = any(best_w[nid] > node_coeffs[nid]["z_gpu"] for nid in node_ids)

            if has_free_vram and has_overloaded:
                overloaded = [nid for nid in node_ids if best_w[nid] > node_coeffs[nid]["z_gpu"]]
                slowest_node = max(
                    overloaded,
                    key=lambda nid: node_coeffs[nid]["rtt_ms"],
                    default=None
                )
                if slowest_node and slowest_node not in forced_m4_nodes:
                    forced_m4_nodes.add(slowest_node)
                    continue

            if found_solution_in_round:
                w = dict(best_w)
                n_gpu = dict(best_n)
            else:
                break

        # Fallback if unfeasible
        if best_cost == float("inf"):
            solver_status = "Fallback_Uniform"
            best_w = dict(w)
            best_n = dict(n_gpu)

        active_nodes = [nid for nid in node_ids if best_w.get(nid, 0) > 0]
        evicted_nodes = [nid for nid in node_ids if nid not in active_nodes]

        return {
            "status": solver_status,
            "total_layers": total_layers,
            "allocations": best_w,
            "gpu_layers": best_n,
            "active_nodes": active_nodes,
            "evicted_nodes": evicted_nodes,
            "estimated_tpot_cost": round(best_cost, 4) if best_cost < float("inf") else None,
            "algorithm": "HALDA (ICLR 2026)",
        }

    def solve_layer_allocation(
        self,
        nodes_telemetry: List[Dict[str, Any]],
        total_layers: int = 64,
        layer_weight_mb: float = 265.0,
        kv_cache_mb: float = 300.0,
    ) -> Dict[str, Any]:
        """Entry point compatible with AeroMesh orchestration pipeline."""
        return self.solve_halda(
            nodes_telemetry=nodes_telemetry,
            total_layers=total_layers,
            layer_weight_mb=layer_weight_mb,
            kv_cache_mb=kv_cache_mb,
        )

