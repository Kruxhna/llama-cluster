"""
AeroMesh Dynamic Graph Compiler (The Brain).
Solves an Integer Linear Program (ILP) using PuLP to compute optimal model layer boundaries
and dynamic between-turn rebalancing across heterogeneous worker nodes.
Specification Reference: Section 4.3 & Section 7.1
"""

from typing import Dict, Any, List, Tuple, Optional
from llama_cluster.config import Config, get_config

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False


class DynamicGraphCompiler:
    """Integer Linear Programming (ILP) solver for layer slicing and rebalancing."""

    def __init__(self, cfg: Optional[Config] = None):
        self.config = cfg or get_config()

    def solve_layer_allocation(
        self,
        nodes_telemetry: List[Dict[str, Any]],
        total_layers: int = 64,
        layer_weight_mb: float = 265.0,
        kv_cache_mb: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Solves ILP for layer assignment l_i across nodes N minimizing Time-Per-Output-Token (TPOT).
        
        Constraints:
          1. Layer conservation: sum(l_i) == L_total (64 layers for Qwen-2.5-32B)
          2. VRAM constraint: l_i * layer_weight_mb + kv_cache_mb <= Usable_VRAM_i * x_i
          3. Thermal throttle derating: If temp > 85°C, derate throughput P_i by 50%
          4. Network eviction: If RTT > 300ms, set x_i = 0 (evict bottleneck node)
        """
        node_ids = [n["node_id"] for n in nodes_telemetry]

        if not HAS_PULP:
            # Fallback uniform layer allocation when pulp is not installed
            layers_per_node = total_layers // len(node_ids)
            allocations = {}
            for i, nid in enumerate(node_ids):
                allocations[nid] = layers_per_node if i < len(node_ids) - 1 else total_layers - (layers_per_node * i)
            return {
                "status": "Fallback_Uniform",
                "total_layers": total_layers,
                "allocations": allocations,
                "active_nodes": node_ids,
                "evicted_nodes": [],
            }

        prob = pulp.LpProblem("AeroMesh_Layer_Allocation", pulp.LpMinimize)

        # Decision variables: l_i (integer layer count), x_i (binary active indicator)
        l_vars = {nid: pulp.LpVariable(f"l_{nid}", lowBound=0, upBound=total_layers, cat=pulp.LpInteger) for nid in node_ids}
        x_vars = {nid: pulp.LpVariable(f"x_{nid}", cat=pulp.LpBinary) for nid in node_ids}

        # 1. Layer Conservation Constraint
        prob += pulp.lpSum([l_vars[nid] for nid in node_ids]) == total_layers, "Layer_Conservation"

        # Dynamically calculate layer weight in MB from model size or default
        if layer_weight_mb == 270.0 and total_layers <= 48:
            layer_weight_mb = 175.0  # Appropriate for 14B models (~8.5GB / 48 layers)

        total_vram_available = sum((n.get("metrics", {}).get("vram_free_bytes", 8*1024*1024*1024)) / (1024*1024) for n in nodes_telemetry)
        
        cost_terms = []

        for node in nodes_telemetry:
            nid = node["node_id"]
            metrics = node.get("metrics", {})
            
            usable_vram_mb = (metrics.get("vram_free_bytes", 8 * 1024 * 1024 * 1024)) / (1024 * 1024)
            temp_c = metrics.get("gpu_temp_celsius", 50.0)
            rtt_ms = metrics.get("network_rtt_to_coordinator_ms", 10.0)

            base_p = node.get("compute_tflops", 15.0)
            
            # Thermal derating (50% penalty if temp > 85°C)
            if temp_c > self.config.thermal_throttle_temp_c:
                base_p *= 0.5

            # High latency eviction (>300ms RTT forces node eviction)
            if rtt_ms > 300.0:
                prob += x_vars[nid] == 0, f"Evict_{nid}_Latency"

            # 2. VRAM Capacity Constraint per node
            max_allocable_layers = max(1, int((usable_vram_mb - kv_cache_mb) / max(1.0, layer_weight_mb)))
            # If total available VRAM is sufficient, apply individual upper bound
            if total_vram_available >= total_layers * layer_weight_mb:
                prob += l_vars[nid] <= max_allocable_layers * x_vars[nid], f"VRAM_Limit_{nid}"

            # Objective terms: compute time per layer + network RTT cost
            compute_cost = 1.0 / max(0.1, base_p)
            network_cost = rtt_ms / 1000.0
            
            cost_terms.append(l_vars[nid] * compute_cost + x_vars[nid] * network_cost)

        # Minimize Total Time-Per-Output-Token (TPOT)
        prob += pulp.lpSum(cost_terms), "Minimize_TPOT"

        # Solve ILP
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        status = pulp.LpStatus[prob.status]
        allocations = {}
        active_nodes = []

        if status == "Optimal":
            for nid in node_ids:
                alloc_layers = int(pulp.value(l_vars[nid]))
                is_active = bool(pulp.value(x_vars[nid]) > 0.5 and alloc_layers > 0)
                allocations[nid] = alloc_layers
                if is_active:
                    active_nodes.append(nid)
        else:
            # Proportional distribution based on usable VRAM
            vram_weights = [max(1.0, (n.get("metrics", {}).get("vram_free_bytes", 1) / (1024*1024*1024))) for n in nodes_telemetry]
            total_w = sum(vram_weights)
            allocated_so_far = 0
            for i, nid in enumerate(node_ids):
                if i == len(node_ids) - 1:
                    allocations[nid] = total_layers - allocated_so_far
                else:
                    layers_for_node = int(round((vram_weights[i] / total_w) * total_layers))
                    allocations[nid] = layers_for_node
                    allocated_so_far += layers_for_node
                active_nodes.append(nid)

        return {
            "status": status,
            "total_layers": total_layers,
            "allocations": allocations,
            "active_nodes": active_nodes,
            "evicted_nodes": [nid for nid in node_ids if nid not in active_nodes],
        }
