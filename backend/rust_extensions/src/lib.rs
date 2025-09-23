use pyo3::prelude::*;

mod memory_monitor;
mod model_loader;
mod quantization;

use memory_monitor::RustMemoryMonitor;
use model_loader::RustModelLoader;

/// High-performance Rust extensions for JARVIS ML memory management
#[pymodule]
fn jarvis_rust_extensions(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<RustMemoryMonitor>()?;
    m.add_class::<RustModelLoader>()?;
    m.add_function(wrap_pyfunction!(get_system_memory_info, m)?)?;
    m.add_function(wrap_pyfunction!(compress_data_lz4, m)?)?;
    m.add_function(wrap_pyfunction!(decompress_data_lz4, m)?)?;
    Ok(())
}

/// Fast system memory information
#[pyfunction]
fn get_system_memory_info() -> PyResult<(u64, u64, f32)> {
    use sysinfo::{System, SystemExt};
    
    let mut sys = System::new_all();
    sys.refresh_memory();
    
    let total = sys.total_memory();
    let available = sys.available_memory();
    let percent = ((total - available) as f32 / total as f32) * 100.0;
    
    Ok((total, available, percent))
}

/// LZ4 compression for fast model compression
#[pyfunction]
fn compress_data_lz4(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let compressed = lz4::block::compress(data, Some(lz4::block::CompressionMode::HIGHCOMPRESSION(12)), false)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Compression failed: {}", e)))?;
    
    Ok(pyo3::types::PyBytes::new(py, &compressed).into())
}

/// LZ4 decompression
#[pyfunction]
fn decompress_data_lz4(py: Python<'_>, data: &[u8], uncompressed_size: Option<usize>) -> PyResult<PyObject> {
    let decompressed = lz4::block::decompress(data, uncompressed_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Decompression failed: {}", e)))?;
    
    Ok(pyo3::types::PyBytes::new(py, &decompressed).into())
}