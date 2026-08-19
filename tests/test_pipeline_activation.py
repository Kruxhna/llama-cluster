"""
Unit tests for AeroMesh Zero-Weight Activation Pipeline Bridge.
Verifies binary serialization, local stage processing, and activation vector transmission.
"""

import time
import pytest
from llama_cluster.pipeline_bridge import (
    ActivationTensor,
    ActivationPipelineClient,
    ActivationPipelineServer,
)


def test_activation_tensor_serialization_fidelity():
    """Verifies that float32 hidden states serialize and deserialize without loss."""
    original_states = [0.123456, -0.987654, 3.141592, 0.0, -100.5]
    tensor = ActivationTensor(layer_index=24, hidden_states=original_states, sequence_id=42)

    raw_bytes = tensor.to_bytes()
    assert len(raw_bytes) == 16 + (len(original_states) * 4)

    restored = ActivationTensor.from_bytes(raw_bytes)
    assert restored.layer_index == 24
    assert restored.sequence_id == 42
    assert len(restored.hidden_states) == len(original_states)

    for a, b in zip(restored.hidden_states, original_states):
        assert abs(a - b) < 1e-5


def test_activation_tensor_dict_conversion():
    """Verifies JSON dict conversion for REST/Web transport."""
    original_states = [1.0, 2.0, 3.0]
    tensor = ActivationTensor(layer_index=12, hidden_states=original_states, sequence_id=5)

    d = tensor.to_dict()
    assert d["layer_index"] == 12
    assert d["sequence_id"] == 5
    assert d["hidden_states"] == [1.0, 2.0, 3.0]

    restored = ActivationTensor.from_dict(d)
    assert restored.layer_index == 12
    assert restored.hidden_states == [1.0, 2.0, 3.0]


def test_local_stage_execution():
    """Verifies that local stage returns proper downstream activations and flags final stage."""
    server = ActivationPipelineServer(port=59999, layer_start=24, layer_end=48)
    tensor = ActivationTensor(layer_index=24, hidden_states=[0.5] * 10, sequence_id=1)

    result = server.execute_local_stage(tensor)
    assert result["status"] == "success"
    assert result["is_final_stage"] is True
    assert result["layer_range"] == [24, 48]
    assert len(result["output_tokens"]) > 0
