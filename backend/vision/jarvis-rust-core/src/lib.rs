//! JARVIS Rust Core - High-performance vision and ML operations
//! 
//! This crate provides optimized implementations of:
//! - Quantized ML inference (INT4/INT8/FP16)
//! - Memory-efficient buffer management
//! - Vision processing with hardware acceleration
//! - Zero-copy Python interop

#![feature(portable_simd)]  // Enable portable SIMD

pub mod quantized_ml;
pub mod memory;
pub mod vision;
pub mod bridge;

use std::sync::Once;

// Global initialization
static INIT: Once = Once::new();

/// Initialize the JARVIS Rust core
pub fn initialize() {
    INIT.call_once(|| {
        // Initialize logging
        env_logger::init();
        
        // Initialize thread pool for Rayon
        rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus::get())
            .thread_name(|index| format!("jarvis-worker-{}", index))
            .build_global()
            .expect("Failed to initialize thread pool");
        
        log::info!("JARVIS Rust Core initialized");
        log::info!("CPU cores: {}", num_cpus::get());
        log::info!("SIMD support: {}", cfg!(feature = "simd"));
    });
}

/// Main error type for JARVIS operations
#[derive(thiserror::Error, Debug)]
pub enum JarvisError {
    #[error("Memory allocation failed: {0}")]
    MemoryError(String),
    
    #[error("ML inference error: {0}")]
    InferenceError(String),
    
    #[error("Vision processing error: {0}")]
    VisionError(String),
    
    #[error("Python bridge error: {0}")]
    BridgeError(String),
    
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, JarvisError>;

/// Performance metrics
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub inference_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub operations_per_second: f64,
}

/// Python module initialization
#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn jarvis_rust_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    initialize();
    
    // Add submodules
    bridge::register_python_module(m)?;
    
    Ok(())
}