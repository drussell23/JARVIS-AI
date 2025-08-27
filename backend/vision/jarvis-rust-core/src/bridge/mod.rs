//! Python-Rust bridge using PyO3

pub mod pyo3_bindings;
pub mod serialization;

use crate::Result;

#[cfg(feature = "python-bindings")] 
use pyo3::prelude::*; 

/// Register Python module
#[cfg(feature = "python-bindings")]
pub fn register_python_module(m: &PyModule) -> PyResult<()> {
    // Add classes
    m.add_class::<pyo3_bindings::RustImageProcessor>()?; 
    m.add_class::<pyo3_bindings::RustQuantizedModel>()?; 
    m.add_class::<pyo3_bindings::RustMemoryPool>()?;
    
    // Add functions
    m.add_function(wrap_pyfunction!(pyo3_bindings::process_image_batch, m)?)?;
    m.add_function(wrap_pyfunction!(pyo3_bindings::quantize_model_weights, m)?)?;
    
    // Add submodule for zero-copy operations
    let zero_copy = PyModule::new(m.py(), "zero_copy")?;
    zero_copy.add_class::<pyo3_bindings::ZeroCopyArray>()?;
    m.add_submodule(zero_copy)?;
    
    Ok(())
}