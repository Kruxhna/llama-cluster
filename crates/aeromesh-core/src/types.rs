use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub id: String,
    pub address: String, // e.g. "100.122.125.95:50052"
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub rtt_ms: f64,
    pub is_derp_relay: bool,
    pub is_active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterState {
    pub coordinator_id: String,
    pub model_name: String,
    pub total_layers: usize,
    pub active_nodes: Vec<NodeInfo>,
    pub layer_allocations: Vec<LayerAllocation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAllocation {
    pub node_id: String,
    pub start_layer: usize,
    pub end_layer: usize,
    pub layer_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryPacket {
    pub node_id: String,
    pub timestamp_ms: u64,
    pub gpu_temp_celsius: f32,
    pub gpu_power_watts: f32,
    pub vram_free_bytes: u64,
    pub vram_total_bytes: u64,
    pub ram_free_bytes: u64,
    pub cpu_util_percent: f32,
    pub rtt_to_coordinator_ms: f32,
    pub is_derp_relay: bool,
}
