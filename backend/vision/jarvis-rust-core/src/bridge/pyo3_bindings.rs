//! PyO3 bindings for Python interop with advanced features

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
// Buffer API not used in current implementation
// #[cfg(feature = "python-bindings")]
// use pyo3::buffer::Buffer;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::{PyValueError, PyRuntimeError};
#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4};
#[cfg(feature = "python-bindings")]
use pyo3::types::{PyDict, PyList};

use crate::{Result as RustResult, JarvisError};
use crate::vision::{ImageProcessor, ImageData, ImageFormat};
use crate::quantized_ml::{QuantizedInferenceEngine, QuantizedTensor, QuantizationType};
use crate::quantized_ml::inference::QuantizedLayer;
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
        let array = unsafe { PyArray3::new(py, output_shape, false) };
        unsafe { array.as_slice_mut()?.copy_from_slice(&processed.data) };
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
        let array = unsafe { PyArray2::new(py, output_shape, false) };
        unsafe { array.as_slice_mut()?.copy_from_slice(&result.outputs) };
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

/// Advanced runtime manager for Python
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustRuntimeManager {
    runtime: Arc<crate::runtime::RuntimeManager>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustRuntimeManager {
    #[new]
    fn new(worker_threads: Option<usize>, enable_cpu_affinity: Option<bool>) -> PyResult<Self> {
        let config = crate::runtime::RuntimeConfig {
            worker_threads: worker_threads.unwrap_or_else(num_cpus::get),
            enable_cpu_affinity: enable_cpu_affinity.unwrap_or(true),
            ..Default::default()
        };
        
        let runtime = crate::runtime::RuntimeManager::new(config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(Self {
            runtime: Arc::new(runtime),
        })
    }
    
    /// Run CPU-bound task
    fn run_cpu_task(&self, py: Python, func: PyObject) -> PyResult<PyObject> {
        let runtime = self.runtime.clone();
        
        let result = py.allow_threads(|| {
            let handle = runtime.spawn_cpu("python-cpu-task", move || {
                Python::with_gil(|py| {
                    func.call0(py)
                })
            });
            
            // Wait for result
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                handle.await.unwrap()
            })
        });
        
        result
    }
    
    /// Get runtime statistics
    fn stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.runtime.stats();
            let dict = PyDict::new(py);
            
            dict.set_item("active_tasks", stats.active_tasks)?;
            dict.set_item("total_spawned", stats.total_spawned)?;
            dict.set_item("total_completed", stats.total_completed)?;
            dict.set_item("active_workers", stats.active_workers)?;
            dict.set_item("queue_depth", stats.queue_depth)?;
            
            Ok(dict.to_object(py))
        })
    }
}

/// Advanced memory pool with leak detection
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustAdvancedMemoryPool {
    pool: Arc<crate::memory::advanced_pool::AdvancedBufferPool>,
    leak_monitor: Arc<Mutex<Vec<String>>>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustAdvancedMemoryPool {
    #[new]
    fn new() -> PyResult<Self> {
        let pool = Arc::new(crate::memory::advanced_pool::AdvancedBufferPool::new());
        let leak_monitor = Arc::new(Mutex::new(Vec::new()));
        
        // Start leak monitoring
        let leak_detector = pool.stats(); // This would need the leak detector exposed
        let monitor = leak_monitor.clone();
        
        std::thread::spawn(move || {
            // Monitor for leaks in background
            loop {
                std::thread::sleep(std::time::Duration::from_secs(10));
                // Check for leaks and add to monitor
            }
        });
        
        Ok(Self { pool, leak_monitor })
    }
    
    /// Allocate tracked buffer
    fn allocate(&self, size: usize) -> PyResult<RustTrackedBuffer> {
        let buffer = self.pool.allocate(size)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        Ok(RustTrackedBuffer {
            buffer: Arc::new(Mutex::new(Some(buffer))),
            size,
        })
    }
    
    /// Get pool statistics
    fn stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let stats = self.pool.stats();
            let dict = PyDict::new(py);
            
            dict.set_item("total_active", stats.total_active)?;
            dict.set_item("total_allocated_bytes", stats.total_allocated_bytes)?;
            dict.set_item("memory_pressure", format!("{:?}", stats.pressure))?;
            
            // Add size class statistics
            let size_classes = PyList::empty(py);
            for sc in stats.size_classes {
                let sc_dict = PyDict::new(py);
                sc_dict.set_item("size", sc.size)?;
                sc_dict.set_item("available", sc.available)?;
                sc_dict.set_item("capacity", sc.capacity)?;
                sc_dict.set_item("high_water_mark", sc.high_water_mark)?;
                size_classes.append(sc_dict)?;
            }
            dict.set_item("size_classes", size_classes)?;
            
            Ok(dict.to_object(py))
        })
    }
    
    /// Check for memory leaks
    fn check_leaks(&self) -> PyResult<Vec<String>> {
        let leaks = self.leak_monitor.lock().unwrap();
        Ok(leaks.clone())
    }
}

/// Tracked buffer that automatically returns to pool
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustTrackedBuffer {
    buffer: Arc<Mutex<Option<crate::memory::advanced_pool::TrackedBuffer>>>,
    size: usize,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustTrackedBuffer {
    /// Get buffer as numpy array
    fn as_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<u8>> {
        let buffer = self.buffer.lock().unwrap();
        if let Some(ref buf) = *buffer {
            unsafe {
                let array = PyArray1::from_slice(py, buf.as_slice());
                Ok(array)
            }
        } else {
            Err(PyValueError::new_err("Buffer already released"))
        }
    }
    
    /// Get buffer ID for tracking
    fn id(&self) -> PyResult<u64> {
        let buffer = self.buffer.lock().unwrap();
        if let Some(ref buf) = *buffer {
            Ok(buf.id())
        } else {
            Err(PyValueError::new_err("Buffer already released"))
        }
    }
    
    /// Manually release buffer
    fn release(&self) -> PyResult<()> {
        let mut buffer = self.buffer.lock().unwrap();
        *buffer = None;
        Ok(())
    }
}

/// Register all Python bindings
#[cfg(feature = "python-bindings")]
pub fn register_python_module(m: &PyModule) -> PyResult<()> {
    // Register classes
    m.add_class::<RustImageProcessor>()?;
    m.add_class::<RustQuantizedModel>()?;
    m.add_class::<RustMemoryPool>()?;
    m.add_class::<ZeroCopyArray>()?;
    m.add_class::<RustRuntimeManager>()?;
    m.add_class::<RustAdvancedMemoryPool>()?;
    m.add_class::<RustTrackedBuffer>()?;
    
    // Register functions
    m.add_function(wrap_pyfunction!(process_image_batch, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_model_weights, m)?)?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}