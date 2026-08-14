"""
AeroMesh Byzantine Tensor Validator (The Data Guard).
Implements Canary Trap verification protocol measuring relative L2 Euclidean distance mismatch scores.
Specification Reference: Section 4.4 & Section 7.1
"""

import math
from typing import Dict, Any, List, Optional
from llama_cluster.config import Config, get_config


class ByzantineTensorValidator:
    """Canary Trap validator for hardware arithmetic corruption and thermal drift detection."""

    def __init__(self, cfg: Optional[Config] = None):
        self.config = cfg or get_config()

    def compute_relative_l2_distance(self, v_actual: List[float], v_ref: List[float]) -> float:
        """
        Computes S_mismatch = ||v_actual - v_ref||_2 / ||v_ref||_2
        Formula: Section 4.4 Canary Trap Protocol
        """
        if len(v_actual) != len(v_ref) or not v_ref:
            return 1.0  # Mismatch error penalty

        sq_diff_sum = sum((a - r) ** 2 for a, r in zip(v_actual, v_ref))
        sq_ref_sum = sum(r ** 2 for r in v_ref)

        if sq_ref_sum == 0:
            return 0.0

        dist = math.sqrt(sq_diff_sum) / math.sqrt(sq_ref_sum)
        return round(dist, 6)

    def validate_activation_tensor(
        self,
        node_id: str,
        v_actual: List[float],
        v_ref: List[float]
    ) -> Dict[str, Any]:
        """
        Evaluates returned activation tensor against Canary Trap reference vector.
        Dictates continuous system responses according to Section 4.4 Table.
        """
        s_mismatch = self.compute_relative_l2_distance(v_actual, v_ref)

        warn_thresh = self.config.canary_warning_threshold  # 0.003
        evict_thresh = self.config.canary_eviction_threshold  # 0.005

        if s_mismatch <= warn_thresh:
            status = "ACCEPTED"
            interpretation = "Expected Quantization / FP Noise"
            action = "Accept activation tensor; maintain active node status."
            corrupted = False
            warning = False

        elif s_mismatch <= evict_thresh:
            status = "WARNING"
            interpretation = "Numerical Jitter / Thermal Drift Warning"
            action = "Flag node for telemetry inspection; lower ILP prioritization weight."
            corrupted = False
            warning = True

        else:
            status = "CORRUPTED"
            interpretation = "Hardware Arithmetic Corruption / Thermal Failure"
            action = "Flag node as corrupted; trigger immediate graph re-slice to evict node."
            corrupted = True
            warning = True

        return {
            "node_id": node_id,
            "s_mismatch": s_mismatch,
            "status": status,
            "interpretation": interpretation,
            "action": action,
            "is_corrupted": corrupted,
            "is_warning": warning,
        }
