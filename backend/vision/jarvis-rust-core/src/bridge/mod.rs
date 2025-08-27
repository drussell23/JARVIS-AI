//! Python-Rust bridge using PyO3

pub mod pyo3_bindings;
pub mod serialization;

use crate::Result;

#[cfg(feature = "python-bindings")] 
use pyo3::prelude::*; 

/// Register Python module with all bindings
#[cfg(feature = "python-bindings")]
pub fn register_python_module(m: &PyModule) -> PyResult<()> {
    // Use the comprehensive registration from pyo3_bindings
    pyo3_bindings::register_python_module(m)
}