//! JARVIS Performance Layer - Rust Implementation
//! 
//! This crate provides high-performance implementations of:
//! - Quantized ML inference (INT8)
//! - Memory pool management
//! - Vision processing pipeline
//! - Parallel data processing
//! - Buffer recycling system

use pyo3::prelude::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod ml;
mod vision;
mod memory;
mod utils;
mod resource_monitor;

use ml::{QuantizedInferenceEngine, InferenceOptions};
use vision::{VisionProcessor, ProcessingOptions};
use memory::{MemoryPool, PoolOptions, SharedMemoryView};
use utils::CPUThrottler;

/// Main Python module entry point
#[pymodule]
fn jarvis_performance(_py: Python, m: &PyModule) -> PyResult<()> {
    // Register our high-performance classes
    m.add_class::<QuantizedInferenceEngine>()?;
    m.add_class::<InferenceOptions>()?;
    m.add_class::<VisionProcessor>()?;
    m.add_class::<ProcessingOptions>()?;
    m.add_class::<MemoryPool>()?;
    m.add_class::<PoolOptions>()?;
    m.add_class::<SharedMemoryView>()?;
    m.add_class::<CPUThrottler>()?;
    
    // Register functions
    m.add_function(wrap_pyfunction!(benchmark_performance, m)?)?;
    m.add_function(wrap_pyfunction!(get_cpu_usage, m)?)?;
    
    Ok(())
}

/// Benchmark the performance improvements
#[pyfunction]
fn benchmark_performance() -> PyResult<String> {
    Ok("Rust performance layer active!".to_string())
}

/// Get current CPU usage
#[pyfunction]
fn get_cpu_usage() -> PyResult<f32> {
    // Implementation will use system metrics
    Ok(0.0)
}
