use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use anyhow::{bail, Context, Result};
use sha2::{Digest, Sha256};
use tracing::info;

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub file_path: String,
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub file_size_bytes: u64,
    pub fast_checksum: String,
}

pub fn inspect_gguf_file<P: AsRef<Path>>(path: P) -> Result<GgufMetadata> {
    let path_ref = path.as_ref();
    let mut file = File::open(path_ref).context("Failed to open GGUF model file")?;
    let file_size_bytes = file.metadata()?.len();

    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("Invalid GGUF header magic. Expected 'GGUF', got {:?}", magic);
    }

    let mut ver_bytes = [0u8; 4];
    file.read_exact(&mut ver_bytes)?;
    let version = u32::from_le_bytes(ver_bytes);
    if !(2..=3).contains(&version) {
        bail!("Unsupported GGUF version: {}", version);
    }

    let mut count_bytes = [0u8; 8];
    file.read_exact(&mut count_bytes)?;
    let tensor_count = u64::from_le_bytes(count_bytes);

    file.read_exact(&mut count_bytes)?;
    let kv_count = u64::from_le_bytes(count_bytes);

    // Compute fast 64MB block hash
    file.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 64 * 1024 * 1024];
    let n = file.read(&mut buffer)?;
    hasher.update(&buffer[..n]);
    let fast_checksum = hex::encode(&hasher.finalize()[..8]);

    info!(
        path = %path_ref.display(),
        version = version,
        tensors = tensor_count,
        kv_pairs = kv_count,
        size_gb = (file_size_bytes as f64) / 1024.0 / 1024.0 / 1024.0,
        checksum = %fast_checksum,
        "✅ GGUF Model Header Verified"
    );

    Ok(GgufMetadata {
        file_path: path_ref.to_string_lossy().to_string(),
        version,
        tensor_count,
        kv_count,
        file_size_bytes,
        fast_checksum,
    })
}
