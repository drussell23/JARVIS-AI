//! PyO3 bindings for Python interop

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::buffer::PyBuffer;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};

use crate::{Result as RustResult, JarvisError};
use crate::vision::{ImageProcessor, ImageData, ImageFormat};
use crate::quantized_ml::{QuantizedInferenceEngine, QuantizedLayer, QuantizedTensor, QuantizationType};
use crate::memory::{MemoryManager, ZeroCopyBuffer};
use std::sync::{Arc, Mutex};

/// Python-accessible image processor
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustImageProcessor {
    processor: Arc<ImageProcessor>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustImageProcessor {
    #[new]
    fn new() -> Self {
        Self {
            processor: Arc::new(ImageProcessor::new()),
        }
    }
    
    /// Process numpy array image
    fn process_numpy_image(&self, py: Python, image: &PyArray3<u8>) -> PyResult<Py<PyArray3<u8>>> {
        // Get dimensions
        let shape = image.shape();
        let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2] as u8);
        
        // Zero-copy access to numpy data
        let image_slice = unsafe { image.as_slice()? };
        
        // Create ImageData
        let img_data = ImageData::from_raw(
            width, height, 
            image_slice.to_vec(),
            if channels == 3 { ImageFormat::Rgb8 } else { ImageFormat::Rgba8 }
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Process (example: resize)
        let processed = self.processor.resize(&img_data, width / 2, height / 2)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Return as numpy array
        let output_shape = [processed.height as usize, processed.width as usize, processed.channels as usize];
        let array = PyArray3::from_vec(py, processed.data, output_shape);
        Ok(array.to_owned())
    }
    
    /// Batch process images with zero-copy
    fn process_batch_zero_copy(&self, py: Python, images: Vec<&PyArray3<u8>>) -> PyResult<Vec<Py<PyArray3<u8>>>> {
        let mut results = Vec::new();
        
        for image in images {
            let result = self.process_numpy_image(py, image)?;
            results.push(result);
        }
        
        Ok(results)
    }
}

/// Python-accessible quantized model
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustQuantizedModel {
    engine: Arc<Mutex<QuantizedInferenceEngine>>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustQuantizedModel {
    #[new]
    fn new(use_simd: bool, thread_count: usize) -> Self {
        Self {
            engine: Arc::new(Mutex::new(QuantizedInferenceEngine::new(use_simd, thread_count))),
        }
    }
    
    /// Run inference on numpy array
    fn infer(&self, py: Python, input: &PyArray4<f32>) -> PyResult<Py<PyArray2<f32>>> {
        // Convert numpy to quantized tensor
        let shape = input.shape();
        let data = unsafe { input.as_slice()? };
        
        // Quantize input (simplified - in production would be more sophisticated)
        let quantized_data: Vec<i8> = data.iter()
            .map(|&v| (v * 127.0).round().max(-128.0).min(127.0) as i8)
            .collect();
        
        let quantized_input = QuantizedTensor::new(
            quantized_data,
            shape.to_vec(),
            1.0 / 127.0,
            0,
            QuantizationType::Int8,
        );
        
        // Run inference
        let result = self.engine.lock().unwrap()
            .infer(&quantized_input)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Return as numpy array
        let output_shape = [result.shape[0], result.shape[1]];
        let array = PyArray2::from_vec(py, result.outputs, output_shape);
        Ok(array.to_owned())
    }
    
    /// Add quantized layer from numpy weights
    fn add_linear_layer(&mut self, weights: &PyArray2<f32>, bias: Option<&PyArray1<f32>>) -> PyResult<()> {
        let weight_shape = weights.shape();
        let weight_data = unsafe { weights.as_slice()? };
        
        // Quantize weights to INT8
        let scale = weight_data.iter().fold(0.0f32, |max, &v| max.max(v.abs())) / 127.0;
        let quantized_weights: Vec<i8> = weight_data.iter()
            .map(|&v| (v / scale).round().max(-128.0).min(127.0) as i8)
            .collect();
        
        let weight_tensor = QuantizedTensor::new(
            quantized_weights,
            vec![weight_shape[0], weight_shape[1]],
            scale,
            0,
            QuantizationType::Int8,
        );
        
        // Handle bias
        let bias_vec = if let Some(b) = bias {
            Some(unsafe { b.as_slice()?.to_vec() })
        } else {
            None
        };
        
        // Add layer
        self.engine.lock().unwrap().add_layer(QuantizedLayer::Linear {
            weights: weight_tensor,
            bias: bias_vec,
        });
        
        Ok(())
    }
}

/// Python-accessible memory pool
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustMemoryPool {
    manager: Arc<MemoryManager>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustMemoryPool {
    #[new]
    fn new() -> Self {
        Self {
            manager: MemoryManager::global(),
        }
    }
    
    /// Allocate buffer
    fn allocate(&self, size: usize) -> PyResult<ZeroCopyArray> {
        let buffer = self.manager.allocate(size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(ZeroCopyArray {
            buffer: Arc::new(Mutex::new(ZeroCopyBuffer::from_rust(buffer))),
        })
    }
    
    /// Get memory statistics
    fn stats(&self) -> PyResult<Vec<(String, usize)>> {
        let stats = self.manager.stats();
        Ok(vec![
            ("total_allocated_bytes".to_string(), stats.total_allocated_bytes),
            ("active_allocations".to_string(), stats.active_allocations),
            ("pool_hits".to_string(), stats.pool_hits),
            ("pool_misses".to_string(), stats.pool_misses),
        ])
    }
}

/// Zero-copy array for Python
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct ZeroCopyArray {
    buffer: Arc<Mutex<ZeroCopyBuffer>>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl ZeroCopyArray {
    /// Get as numpy array (zero-copy view)
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<u8>> {
        let buffer = self.buffer.lock().unwrap();
        unsafe {
            let array = PyArray1::from_slice(py, buffer.as_slice());
            Ok(array)
        }
    }
    
    /// Get buffer size
    fn size(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }
    
    /// Create from numpy array (zero-copy)
    #[staticmethod]
    fn from_numpy(array: &PyArray1<u8>) -> PyResult<Self> {
        let py = array.py();
        
        unsafe {
            // Get pointer to numpy data
            let ptr = array.as_ptr() as *mut u8;
            let size = array.len();
            
            // Create zero-copy buffer referencing Python memory
            let buffer = ZeroCopyBuffer::from_python(
                array.as_ptr() as *mut pyo3::ffi::PyObject,
                ptr,
                size
            );
            
            Ok(Self {
                buffer: Arc::new(Mutex::new(buffer)),
            })
        }
    }
}

/// Process image batch function
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn process_image_batch(py: Python, images: Vec<&PyArray3<u8>>) -> PyResult<Vec<Py<PyArray3<u8>>>> {
    let processor = RustImageProcessor::new();
    processor.process_batch_zero_copy(py, images)
}

/// Quantize model weights function
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn quantize_model_weights(weights: &PyArray2<f32>) -> PyResult<Vec<i8>> {
    let data = unsafe { weights.as_slice()? };
    let scale = data.iter().fold(0.0f32, |max, &v| max.max(v.abs())) / 127.0;
    
    let quantized: Vec<i8> = data.iter()
        .map(|&v| (v / scale).round().max(-128.0).min(127.0) as i8)
        .collect();
    
    Ok(quantized)
}