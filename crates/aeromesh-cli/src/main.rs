use std::net::SocketAddr;
use std::path::PathBuf;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use aeromesh_core::tailscale::TailscaleInspector;
use aeromesh_engine::{inspect_gguf_file, EngineSupervisor};

#[derive(Parser, Debug)]
#[command(name = "aeromesh")]
#[command(about = "AeroMesh: Distributed Fault-Tolerant LLM Cluster Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Verify GGUF model headers, layer count, and cross-node hash integrity
    ModelCheck {
        /// Path to the .gguf model file
        #[arg(value_name = "FILE")]
        path: PathBuf,
    },

    /// Probe Tailscale link quality (Direct WireGuard vs DERP, RTT latency)
    Probe {
        /// Target Tailscale address (e.g. 100.122.125.95:50052)
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Start a supervised CUDA RPC backend worker node
    Worker {
        /// Host IP to bind
        #[arg(long, default_value = "0.0.0.0")]
        host: String,

        /// Port to bind
        #[arg(long, default_value_t = 50052)]
        port: u16,
    },

    /// Run the multi-node cluster coordinator
    Coordinator {
        /// Path to GGUF model file
        #[arg(long)]
        model: PathBuf,

        /// Comma-separated list of remote RPC peer addresses (e.g. 100.122.125.95:50052)
        #[arg(long)]
        peers: Option<String>,

        /// Number of layers to offload to GPU (-1 for all)
        #[arg(long, default_value_t = -1, allow_hyphen_values = true)]
        ngl: i32,

        /// Prompt text to execute
        #[arg(long, default_value = "Write a short sentence about GPU clusters.")]
        prompt: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer())
        .with(tracing_subscriber::EnvFilter::from_default_env().add_directive("info".parse()?))
        .init();

    let cli = Cli::parse();
    let current_dir = std::env::current_dir()?;

    match cli.command {
        Commands::ModelCheck { path } => {
            info!("🔍 Inspecting GGUF model integrity at {:?}", path);
            let meta = inspect_gguf_file(&path)?;
            println!("\n========================================================");
            println!("   AEROMESH GGUF MODEL INTEGRITY REPORT");
            println!("========================================================");
            println!("  File Path:      {}", meta.file_path);
            println!("  GGUF Version:   v{}", meta.version);
            println!("  Tensor Count:   {}", meta.tensor_count);
            println!("  KV Metadata:    {}", meta.kv_count);
            println!("  File Size:      {:.2} GB", (meta.file_size_bytes as f64) / 1024.0 / 1024.0 / 1024.0);
            println!("  Fast Checksum:  {}", meta.fast_checksum);
            println!("  Status:         ✅ VALID (Ready for Zero-Copy mmap)");
            println!("========================================================\n");
        }

        Commands::Probe { target } => {
            info!("🛰️ Probing Tailscale target: {}", target);
            let addr: SocketAddr = target.parse().context("Invalid target socket address")?;
            let report = TailscaleInspector::probe_socket_link(addr).await?;
            println!("\n========================================================");
            println!("   AEROMESH TAILSCALE LINK QUALITY REPORT");
            println!("========================================================");
            println!("  Target Host:    {}", report.host_name);
            println!("  IP Address:     {}", report.ip);
            println!("  Direct WireGuard: {}", if report.is_direct_wireguard { "✅ YES" } else { "❌ NO (DERP Relay)" });
            if let Some(relay) = report.relay_region {
                println!("  Relay Region:   {}", relay);
            }
            println!("  TCP RTT Ping:   {:.2} ms", report.tcp_rtt_ms);
            println!("  Cluster Status: {}", if report.is_acceptable { "✅ ELIGIBLE (Direct / Low Latency)" } else { "❌ REJECTED (DERP Relay / RTT > 150ms)" });
            println!("========================================================\n");
        }

        Commands::Worker { host, port } => {
            info!("🛡️ Starting AeroMesh Worker Daemon on {}:{}", host, port);
            let mut supervisor = EngineSupervisor::new(&current_dir)?;
            supervisor.spawn_rpc_worker(&host, port)?;

            println!("\n========================================================");
            println!("   AEROMESH CUDA RPC WORKER RUNNING");
            println!("========================================================");
            println!("  Bound Address:  {}:{}", host, port);
            println!("  Protection:     Windows Job Object Active (Leak-Proof VRAM)");
            println!("  Press Ctrl+C to terminate worker safely.");
            println!("========================================================\n");

            tokio::signal::ctrl_c().await?;
            info!("Received shutdown signal. Reclaiming all resources...");
            supervisor.shutdown_all();
        }

        Commands::Coordinator {
            model,
            peers,
            ngl,
            prompt,
        } => {
            println!("\n========================================================");
            println!("   AEROMESH DISTRIBUTED COORDINATOR INITIALIZING");
            println!("========================================================");

            // Step 1: Check model
            info!("Step 1/3: Verifying local model file...");
            let meta = inspect_gguf_file(&model)?;
            info!(
                model = %model.display(),
                tensors = meta.tensor_count,
                checksum = %meta.fast_checksum,
                "Model integrity confirmed"
            );

            // Step 2: Probe Tailscale peers
            let peer_list: Vec<String> = peers
                .as_ref()
                .map(|p| p.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
                .unwrap_or_default();

            let mut valid_peers = Vec::new();
            if !peer_list.is_empty() {
                info!("Step 2/3: Probing Tailscale peer quality for {} nodes...", peer_list.len());
                for peer in &peer_list {
                    if let Ok(addr) = peer.parse::<SocketAddr>() {
                        match TailscaleInspector::probe_socket_link(addr).await {
                            Ok(quality) => {
                                if quality.is_acceptable {
                                    valid_peers.push(peer.clone());
                                    info!(peer = %peer, rtt_ms = quality.tcp_rtt_ms, "Peer approved for cluster");
                                } else {
                                    warn!(
                                        peer = %peer,
                                        rtt_ms = quality.tcp_rtt_ms,
                                        is_direct = quality.is_direct_wireguard,
                                        "Peer rejected: violates RTT <= 150ms or DERP rule"
                                    );
                                }
                            }
                            Err(e) => {
                                warn!(peer = %peer, error = %e, "Peer unreachable; skipping");
                            }
                        }
                    } else {
                        warn!(peer = %peer, "Invalid socket address format");
                    }
                }
            } else {
                info!("No remote peers configured. Running on local node.");
            }

            // Step 3: Launch inference execution
            info!("Step 3/3: Dispatching prompt across active cluster pipeline...");
            let mut supervisor = EngineSupervisor::new(&current_dir)?;
            let result = supervisor.run_completion(&model, &valid_peers, ngl, &prompt)?;

            println!("\n========================================================");
            println!("   AEROMESH INFERENCE GENERATION OUTPUT");
            println!("========================================================");
            println!("{}", result.trim());
            println!("========================================================\n");
            info!("🎉 Multi-node token generation completed successfully!");
        }
    }

    Ok(())
}
