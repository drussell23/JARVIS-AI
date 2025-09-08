pub mod pattern_recognition;
pub mod state_detection;
pub mod visual_features;
pub mod memory_pool;
pub mod python_bridge;

use pyo3::prelude::*;

/// Initialize the vision intelligence module
#[pymodule]
fn vision_intelligence(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Register pattern recognition components
    m.add_class::<pattern_recognition::PatternMatcher>()?;
    m.add_class::<pattern_recognition::VisualPattern>()?;
    
    // Register state detection components
    m.add_class::<state_detection::StateDetector>()?;
    m.add_class::<state_detection::StateSignature>()?;
    
    // Register visual feature extractors
    m.add_class::<visual_features::FeatureExtractor>()?;
    m.add_class::<visual_features::ColorHistogram>()?;
    m.add_class::<visual_features::StructuralFeatures>()?;
    
    // Register memory pool for efficient processing
    m.add_class::<memory_pool::VisionMemoryPool>()?;
    
    // Add initialization function
    m.add_function(wrap_pyfunction!(initialize_vision_intelligence, m)?)?;
    
    Ok(())
}

#[pyfunction]
fn initialize_vision_intelligence() -> PyResult<String> {
    // Initialize global resources
    memory_pool::initialize_global_pool();
    
    Ok("Vision Intelligence initialized successfully".to_string())
}