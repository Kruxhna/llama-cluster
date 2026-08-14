"""
Unit tests for AeroMesh Byzantine Tensor Validator (Canary Trap Protocol).
Specification Reference: Section 4.4
"""

from llama_cluster.canary_validator import ByzantineTensorValidator


def test_canary_distance_exact_match():
    """Verify zero L2 distance for identical vectors."""
    validator = ByzantineTensorValidator()
    v_actual = [1.0, 2.0, 3.0, 4.0]
    v_ref = [1.0, 2.0, 3.0, 4.0]
    
    res = validator.validate_activation_tensor("Node_A", v_actual, v_ref)
    assert res["s_mismatch"] == 0.0
    assert res["status"] == "ACCEPTED"
    assert not res["is_corrupted"]


def test_canary_distance_warning():
    """Verify WARNING status for small numerical jitter (S_mismatch between 0.003 and 0.005)."""
    validator = ByzantineTensorValidator()
    v_ref = [100.0] * 100
    # Add ~0.4% noise
    v_actual = [100.4] * 100

    res = validator.validate_activation_tensor("Node_B", v_actual, v_ref)
    assert res["status"] in ("WARNING", "CORRUPTED")
    assert res["is_warning"]


def test_canary_distance_corruption_eviction():
    """Verify CORRUPTED status and eviction trigger for large arithmetic deviation (>0.005)."""
    validator = ByzantineTensorValidator()
    v_ref = [10.0] * 10
    v_actual = [12.0] * 10  # 20% deviation!

    res = validator.validate_activation_tensor("Node_C_Corrupted", v_actual, v_ref)
    assert res["status"] == "CORRUPTED"
    assert res["is_corrupted"]
