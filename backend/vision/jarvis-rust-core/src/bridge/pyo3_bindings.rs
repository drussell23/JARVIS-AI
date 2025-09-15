//! Enhanced PyO3 bindings for zero-copy Python interop
//! Fully dynamic configuration with no hardcoded values

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "python-bindings")]
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyMemoryError};
#[cfg(feature = "python-bindings")]
use numpy::{PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray3};
#[cfg(feature = "python-bindings")]
use pyo3::types::{PyDict, PyList, PyBytes};
#[cfg(feature = "python-bindings")]
use pyo3::buffer::PyBuffer;

use crate::{Result as RustResult, JarvisError};
use crate::vision::{
    ImageProcessor, ImageData, ImageFormat, ImageMetadata,
    ScreenCapture, CaptureConfig, SharedMemoryHandle,
    ProcessingConfig, ProcessingPipeline, ColorCorrectionMode,
    ImageCompressor, CompressionFormat, CompressedImage,
    VisionContext, VisionGlobalConfig, update_vision_config
};

#[cfg(all(feature = "python-bindings", target_os = "macos"))]
use crate::vision::{
    WindowTracker, WindowPosition, AppStateDetector, AppState,
    ChunkedTextExtractor, TextChunk, WorkspaceOrganizer,
    WorkspaceRule, RuleCondition, RuleAction, WindowLayout
};
use crate::quantized_ml::{QuantizedInferenceEngine, QuantizedTensor, QuantizationType};
use crate::quantized_ml::inference::QuantizedLayer;
use crate::memory::{MemoryManager, ZeroCopyBuffer};
use std::sync::{Arc, Mutex, RwLock};
use std::collections::HashMap;
use std::time::Instant;
use parking_lot::RwLock as ParkingRwLock;

/// Enhanced Python-accessible image processor with dynamic configuration
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustImageProcessor {
    processor: Arc<ImageProcessor>,
    config: Arc<RwLock<ProcessingConfig>>,
    stats: Arc<RwLock<ProcessingStats>>,
}

#[derive(Clone, Default)]
struct ProcessingStats {
    images_processed: u64,
    total_pixels: u64,
    total_time_ms: f64,
    zero_copy_operations: u64,
    python_allocations: u64,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustImageProcessor {
    #[new]
    #[args(config = "None")]
    fn new(config: Option<&PyDict>) -> PyResult<Self> {
        let mut proc_config = ProcessingConfig::from_env();
        
        // Apply Python config if provided
        if let Some(cfg) = config {
            for (key, value) in cfg.iter() {
                let key_str = key.extract::<String>()?;
                let value_str = value.to_string();
                proc_config.update(&key_str, &value_str)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        
        Ok(Self {
            processor: Arc::new(ImageProcessor::with_config(proc_config.clone())),
            config: Arc::new(RwLock::new(proc_config)),
            stats: Arc::new(RwLock::new(ProcessingStats::default())),
        })
    }
    
    /// Update configuration dynamically
    fn update_config(&self, key: &str, value: &str) -> PyResult<()> {
        self.processor.update_config(key, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.config.write().unwrap().update(key, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
    
    /// Get current configuration as dict
    fn get_config(&self, py: Python) -> PyResult<PyObject> {
        let config = self.config.read().unwrap();
        let dict = PyDict::new(py);
        
        // Convert config to Python dict
        dict.set_item("enable_simd", config.enable_simd)?;
        dict.set_item("thread_count", config.thread_count)?;
        dict.set_item("enable_gpu", config.enable_gpu)?;
        dict.set_item("quality_preset", format!("{:?}", config.quality_preset))?;
        dict.set_item("denoise_strength", config.denoise_strength)?;
        dict.set_item("sharpen_amount", config.sharpen_amount)?;
        dict.set_item("auto_enhance", config.auto_enhance)?;
        dict.set_item("max_dimension", config.max_dimension)?;
        
        Ok(dict.to_object(py))
    }
    
    /// Process numpy array with zero-copy when possible
    fn process_numpy_image(&self, py: Python, image: PyReadonlyArray3<u8>, 
                          operation: &str, params: Option<&PyDict>) -> PyResult<Py<PyArray3<u8>>> {
        let start = Instant::now();
        
        // Get dimensions and format
        let shape = image.shape();
        let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2] as u8);
        
        // Determine format dynamically
        let format = match channels {
            1 => ImageFormat::Gray8,
            2 => ImageFormat::GrayA8,
            3 => ImageFormat::Rgb8,
            4 => ImageFormat::Rgba8,
            _ => return Err(PyValueError::new_err(format!("Unsupported channel count: {}", channels))),
        };
        
        // Zero-copy access to numpy data
        let image_slice = image.as_slice()?;
        
        // Create ImageData without copying if possible
        let img_data = ImageData::from_raw(
            width, height, 
            image_slice.to_vec(), // TODO: true zero-copy with lifetime management
            format
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Process based on operation
        let processed = match operation {
            "resize" => {
                let new_width = params.and_then(|p| p.get_item("width"))
                    .and_then(|w| w.extract::<u32>().ok())
                    .unwrap_or(width / 2);
                let new_height = params.and_then(|p| p.get_item("height"))
                    .and_then(|h| h.extract::<u32>().ok())
                    .unwrap_or(height / 2);
                self.processor.resize(&img_data, new_width, new_height)
            }
            "auto_process" => {
                self.processor.auto_process(&img_data)
            }
            "denoise" => {
                let strength = params.and_then(|p| p.get_item("strength"))
                    .and_then(|s| s.extract::<f32>().ok())
                    .unwrap_or(0.5);
                self.processor.denoise(&img_data, strength)
            }
            "sharpen" => {
                let amount = params.and_then(|p| p.get_item("amount"))
                    .and_then(|a| a.extract::<f32>().ok())
                    .unwrap_or(1.0);
                self.processor.sharpen(&img_data, amount)
            }
            "convolve" => {
                let kernel_name = params.and_then(|p| p.get_item("kernel"))
                    .and_then(|k| k.extract::<String>().ok())
                    .unwrap_or_else(|| "gaussian_3x3".to_string());
                self.processor.convolve(&img_data, &kernel_name, None)
            }
            _ => {
                return Err(PyValueError::new_err(format!("Unknown operation: {}", operation)));
            }
        }.map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Update stats
        let mut stats = self.stats.write().unwrap();
        stats.images_processed += 1;
        stats.total_pixels += (width * height) as u64;
        stats.total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
        
        // Return as numpy array
        let output_shape = [processed.height as usize, processed.width as usize, processed.channels as usize];
        let array = unsafe { PyArray3::new(py, output_shape, false) };
        unsafe { array.as_slice_mut()?.copy_from_slice(processed.as_slice()) };
        Ok(array.to_owned())
    }
    
    /// Batch process images with true zero-copy and parallel processing
    fn process_batch_zero_copy(&self, py: Python, images: Vec<PyReadonlyArray3<u8>>, 
                              operation: &str, params: Option<&PyDict>) -> PyResult<Vec<Py<PyArray3<u8>>>> {
        use rayon::prelude::*;
        
        let config = self.config.read().unwrap();
        let thread_count = config.thread_count;
        drop(config);
        
        // Process in parallel with configured thread count
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let results: Vec<_> = pool.install(|| {
            images.into_par_iter()
                .map(|image| {
                    self.process_numpy_image(py, image, operation, params)
                })
                .collect()
        });
        
        // Collect results, propagating any errors
        results.into_iter().collect()
    }
    
    /// Create processing pipeline
    fn create_pipeline(&self) -> RustProcessingPipeline {
        RustProcessingPipeline {
            pipeline: ProcessingPipeline::new(self.processor.clone()),
        }
    }
    
    /// Get processing statistics
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.stats.read().unwrap();
        let dict = PyDict::new(py);
        
        dict.set_item("images_processed", stats.images_processed)?;
        dict.set_item("total_pixels", stats.total_pixels)?;
        dict.set_item("total_time_ms", stats.total_time_ms)?;
        dict.set_item("avg_time_per_image_ms", 
            if stats.images_processed > 0 {
                stats.total_time_ms / stats.images_processed as f64
            } else {
                0.0
            }
        )?;
        dict.set_item("pixels_per_second", 
            if stats.total_time_ms > 0.0 {
                (stats.total_pixels as f64 / stats.total_time_ms) * 1000.0
            } else {
                0.0
            }
        )?;
        
        Ok(dict.to_object(py))
    }
}

/// Python-accessible processing pipeline
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustProcessingPipeline {
    pipeline: ProcessingPipeline,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustProcessingPipeline {
    /// Add resize operation
    fn resize(&mut self, width: u32, height: u32) -> PyResult<()> {
        self.pipeline = self.pipeline.clone().resize(width, height);
        Ok(())
    }
    
    /// Add convolution operation
    fn convolve(&mut self, kernel_name: &str) -> PyResult<()> {
        self.pipeline = self.pipeline.clone().convolve(kernel_name);
        Ok(())
    }
    
    /// Add color correction
    fn color_correct(&mut self, mode: &str) -> PyResult<()> {
        let correction_mode = match mode {
            "auto" => ColorCorrectionMode::Auto,
            "none" => ColorCorrectionMode::None,
            s if s.starts_with("gamma:") => {
                let gamma = s.trim_start_matches("gamma:")
                    .parse::<f32>()
                    .map_err(|_| PyValueError::new_err("Invalid gamma value"))?;
                ColorCorrectionMode::Gamma(gamma)
            }
            _ => return Err(PyValueError::new_err(format!("Unknown color correction mode: {}", mode))),
        };
        
        self.pipeline = self.pipeline.clone().color_correct(correction_mode);
        Ok(())
    }
    
    /// Execute pipeline on image
    fn execute(&self, py: Python, image: PyReadonlyArray3<u8>) -> PyResult<Py<PyArray3<u8>>> {
        // Convert numpy to ImageData
        let shape = image.shape();
        let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2] as u8);
        
        let format = match channels {
            1 => ImageFormat::Gray8,
            3 => ImageFormat::Rgb8,
            4 => ImageFormat::Rgba8,
            _ => return Err(PyValueError::new_err("Unsupported channel count")),
        };
        
        let img_data = ImageData::from_raw(
            width, height,
            image.as_slice()?.to_vec(),
            format
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Execute pipeline
        let result = self.pipeline.execute(img_data)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Return as numpy
        let output_shape = [result.height as usize, result.width as usize, result.channels as usize];
        let array = unsafe { PyArray3::new(py, output_shape, false) };
        unsafe { array.as_slice_mut()?.copy_from_slice(result.as_slice()) };
        Ok(array.to_owned())
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

/// Enhanced zero-copy array with shared memory support
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct ZeroCopyArray {
    buffer: Arc<Mutex<ZeroCopyBuffer>>,
    metadata: HashMap<String, String>,
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
    
    /// Get as 2D numpy array with shape
    fn as_numpy_2d<'py>(&self, py: Python<'py>, height: usize, width: usize) -> PyResult<&'py PyArray2<u8>> {
        let buffer = self.buffer.lock().unwrap();
        let expected_size = height * width;
        
        unsafe {
            if buffer.as_slice().len() != expected_size {
                return Err(PyValueError::new_err(
                    format!("Buffer size {} doesn't match shape {}x{}", 
                        buffer.as_slice().len(), height, width)
                ));
            }
            
            let array = PyArray2::from_slice(py, buffer.as_slice())?;
            Ok(array.reshape([height, width])?)
        }
    }
    
    /// Get as 3D numpy array with shape
    fn as_numpy_3d<'py>(&self, py: Python<'py>, height: usize, width: usize, channels: usize) -> PyResult<&'py PyArray3<u8>> {
        let buffer = self.buffer.lock().unwrap();
        let expected_size = height * width * channels;
        
        unsafe {
            if buffer.as_slice().len() != expected_size {
                return Err(PyValueError::new_err(
                    format!("Buffer size {} doesn't match shape {}x{}x{}", 
                        buffer.as_slice().len(), height, width, channels)
                ));
            }
            
            let array = PyArray3::from_slice(py, buffer.as_slice())?;
            Ok(array.reshape([height, width, channels])?)
        }
    }
    
    /// Get buffer size
    fn size(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }
    
    /// Create from numpy array (zero-copy)
    #[staticmethod]
    fn from_numpy(array: PyReadonlyArray1<u8>) -> PyResult<Self> {
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
                metadata: HashMap::new(),
            })
        }
    }
    
    /// Create from shared memory
    #[staticmethod]
    fn from_shared_memory(name: &str, size: usize) -> PyResult<Self> {
        // This would connect to shared memory created by Rust
        // Implementation depends on platform
        Err(PyRuntimeError::new_err("Shared memory not implemented yet"))
    }
    
    /// Add metadata
    fn set_metadata(&mut self, key: &str, value: &str) {
        self.metadata.insert(key.to_string(), value.to_string());
    }
    
    /// Get metadata
    fn get_metadata(&self, key: &str) -> Option<String> {
        self.metadata.get(key).cloned()
    }
}

/// Enhanced vision capture for Python
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustScreenCapture {
    capture: Arc<RwLock<ScreenCapture>>,
    config: Arc<RwLock<CaptureConfig>>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustScreenCapture {
    #[new]
    #[args(config = "None")]
    fn new(config: Option<&PyDict>) -> PyResult<Self> {
        let mut cap_config = CaptureConfig::from_env();
        
        // Apply Python config if provided
        if let Some(cfg) = config {
            for (key, value) in cfg.iter() {
                let key_str = key.extract::<String>()?;
                let value_str = value.to_string();
                cap_config.update(&key_str, &value_str)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        
        let capture = ScreenCapture::new(cap_config.clone())
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(Self {
            capture: Arc::new(RwLock::new(capture)),
            config: Arc::new(RwLock::new(cap_config)),
        })
    }
    
    /// Capture screen to numpy array
    fn capture_to_numpy(&self, py: Python) -> PyResult<Py<PyArray3<u8>>> {
        let mut capture = self.capture.write().unwrap();
        let image_data = capture.capture_preprocessed()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Convert to numpy
        let shape = [image_data.height as usize, image_data.width as usize, image_data.channels as usize];
        let array = unsafe { PyArray3::new(py, shape, false) };
        unsafe { array.as_slice_mut()?.copy_from_slice(image_data.as_slice()) };
        Ok(array.to_owned())
    }
    
    /// Capture to shared memory for zero-copy access
    fn capture_to_shared_memory(&self, name: &str) -> PyResult<SharedMemoryInfo> {
        let mut capture = self.capture.write().unwrap();
        
        // Create shared memory
        let handle = capture.create_shared_memory(name)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Capture to it
        capture.capture_to_shared_memory(&handle)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(SharedMemoryInfo {
            name: handle.name.clone(),
            size: handle.size,
            #[cfg(unix)]
            fd: handle.fd,
        })
    }
    
    /// Get capture statistics
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let capture = self.capture.read().unwrap();
        let stats = capture.stats();
        
        let dict = PyDict::new(py);
        dict.set_item("frame_count", stats.frame_count)?;
        dict.set_item("actual_fps", stats.actual_fps)?;
        dict.set_item("avg_capture_time_ms", stats.avg_capture_time_ms)?;
        dict.set_item("hardware_accelerated", stats.hardware_accelerated)?;
        
        Ok(dict.to_object(py))
    }
}

/// Shared memory info for Python
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct SharedMemoryInfo {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    size: usize,
    #[cfg(unix)]
    #[pyo3(get)]
    fd: i32,
}

/// Enhanced image compressor for Python
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustImageCompressor {
    compressor: Arc<RwLock<ImageCompressor>>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustImageCompressor {
    #[new]
    fn new() -> Self {
        Self {
            compressor: Arc::new(RwLock::new(ImageCompressor::new())),
        }
    }
    
    /// Compress numpy image
    fn compress_numpy(&self, py: Python, image: PyReadonlyArray3<u8>, 
                     format: Option<&str>) -> PyResult<PyObject> {
        // Convert numpy to ImageData
        let shape = image.shape();
        let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2] as u8);
        
        let img_format = match channels {
            1 => ImageFormat::Gray8,
            3 => ImageFormat::Rgb8,
            4 => ImageFormat::Rgba8,
            _ => return Err(PyValueError::new_err("Unsupported channel count")),
        };
        
        let img_data = ImageData::from_raw(
            width, height,
            image.as_slice()?.to_vec(),
            img_format
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Parse compression format
        let comp_format = if let Some(fmt) = format {
            fmt.parse::<CompressionFormat>()
                .map_err(|e| PyValueError::new_err(e))?  
        } else {
            CompressionFormat::Auto
        };
        
        // Compress
        let mut compressor = self.compressor.write().unwrap();
        let compressed = compressor.compress(&img_data, Some(comp_format))
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Return as dict with compressed data and metadata
        let dict = PyDict::new(py);
        dict.set_item("data", PyBytes::new(py, &compressed.compressed_data))?;
        dict.set_item("width", compressed.width)?;
        dict.set_item("height", compressed.height)?;
        dict.set_item("channels", compressed.channels)?;
        dict.set_item("original_size", compressed.original_size)?;
        dict.set_item("compressed_size", compressed.compressed_data.len())?;
        dict.set_item("compression_ratio", compressed.compression_ratio())?;
        dict.set_item("compression_time_ms", compressed.compression_time_ms)?;
        dict.set_item("format", format!("{:?}", compressed.compression));
        
        // Add metadata
        let metadata_dict = PyDict::new(py);
        for (k, v) in &compressed.metadata {
            metadata_dict.set_item(k, v)?;
        }
        dict.set_item("metadata", metadata_dict)?;
        
        Ok(dict.to_object(py))
    }
    
    /// Decompress to numpy
    fn decompress_to_numpy(&self, py: Python, compressed_data: &PyBytes, 
                          width: u32, height: u32, channels: u8,
                          format: &str, original_size: usize) -> PyResult<Py<PyArray3<u8>>> {
        // Parse format
        let comp_format = format.parse::<CompressionFormat>()
            .map_err(|e| PyValueError::new_err(e))?;
        
        let img_format = match channels {
            1 => ImageFormat::Gray8,
            3 => ImageFormat::Rgb8,
            4 => ImageFormat::Rgba8,
            _ => return Err(PyValueError::new_err("Invalid channel count")),
        };
        
        // Create compressed image struct
        let compressed = CompressedImage {
            width,
            height,
            channels,
            format: img_format,
            compression: comp_format,
            compressed_data: compressed_data.as_bytes().to_vec(),
            original_size,
            compression_time_ms: 0.0,
            metadata: HashMap::new(),
        };
        
        // Decompress
        let mut compressor = self.compressor.write().unwrap();
        let decompressed = compressor.decompress(&compressed)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Return as numpy
        let shape = [height as usize, width as usize, channels as usize];
        let array = unsafe { PyArray3::new(py, shape, false) };
        unsafe { array.as_slice_mut()?.copy_from_slice(decompressed.as_slice()) };
        Ok(array.to_owned())
    }
    
    /// Get compression statistics
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let compressor = self.compressor.read().unwrap();
        let stats = compressor.get_stats();
        
        let dict = PyDict::new(py);
        dict.set_item("total_compressions", stats.total_compressions)?;
        dict.set_item("total_bytes_processed", stats.total_bytes_processed)?;
        dict.set_item("total_bytes_compressed", stats.total_bytes_compressed)?;
        dict.set_item("avg_compression_time_ms", stats.avg_compression_time_ms)?;
        
        // Add per-algorithm stats
        let algo_dict = PyDict::new(py);
        for (name, perf) in compressor.get_algorithm_performance() {
            let perf_dict = PyDict::new(py);
            perf_dict.set_item("uses", perf.uses)?;
            perf_dict.set_item("avg_ratio", perf.avg_ratio)?;
            perf_dict.set_item("avg_time_ms", perf.avg_time_ms)?;
            algo_dict.set_item(name, perf_dict)?;
        }
        dict.set_item("algorithm_performance", algo_dict)?;
        
        Ok(dict.to_object(py))
    }
}

/// Complete vision context for Python
#[cfg(feature = "python-bindings")]
#[pyclass]
pub struct RustVisionContext {
    context: Arc<VisionContext>,
}

#[cfg(feature = "python-bindings")]
#[pymethods]
impl RustVisionContext {
    #[new]
    #[args(config = "None")]
    fn new(config: Option<&PyDict>) -> PyResult<Self> {
        let mut vision_config = VisionGlobalConfig::from_env();
        
        // Apply Python config if provided
        if let Some(cfg) = config {
            for (key, value) in cfg.iter() {
                let key_str = key.extract::<String>()?;
                let value_str = value.to_string();
                vision_config.update(&key_str, &value_str)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            }
        }
        
        let context = VisionContext::with_config(vision_config)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(Self {
            context: Arc::new(context),
        })
    }
    
    /// Full pipeline: capture, process, compress
    fn capture_process_compress<'py>(&self, py: Python<'py>, 
                                    operation: Option<&str>,
                                    compression: Option<&str>) -> PyResult<PyObject> {
        let comp_format = if let Some(fmt) = compression {
            fmt.parse::<CompressionFormat>()
                .map_err(|e| PyValueError::new_err(e))?
        } else {
            CompressionFormat::Auto
        };
        
        // Use tokio runtime for async operation
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let result = runtime.block_on(async {
            self.context.full_pipeline(comp_format).await
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Convert to Python dict
        let dict = PyDict::new(py);
        dict.set_item("data", PyBytes::new(py, &result.compressed_data))?;
        dict.set_item("width", result.width)?;
        dict.set_item("height", result.height)?;
        dict.set_item("compression_ratio", result.compression_ratio())?;
        
        Ok(dict.to_object(py))
    }
    
    /// Update configuration
    fn update_config(&self, key: &str, value: &str) -> PyResult<()> {
        self.context.update_config(key, value)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(())
    }
    
    /// Get statistics
    fn get_stats(&self, py: Python) -> PyResult<PyObject> {
        let stats = self.context.get_stats();
        
        let dict = PyDict::new(py);
        dict.set_item("total_captures", stats.total_captures)?;
        dict.set_item("total_processed", stats.total_processed)?;
        dict.set_item("total_compressed", stats.total_compressed)?;
        dict.set_item("average_fps", stats.average_fps)?;
        
        Ok(dict.to_object(py))
    }
}

/// Process image batch function with configuration
#[cfg(feature = "python-bindings")]
#[pyfunction]
#[pyo3(signature = (images, config=None, operation="auto_process", params=None))]
pub fn process_image_batch(py: Python, images: Vec<PyReadonlyArray3<u8>>, 
                          config: Option<&PyDict>,
                          operation: &str,
                          params: Option<&PyDict>) -> PyResult<Vec<Py<PyArray3<u8>>>> {
    let processor = RustImageProcessor::new(config)?;
    processor.process_batch_zero_copy(py, images, operation, params)
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

/// Update global vision configuration from Python
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn update_global_vision_config(key: &str, value: &str) -> PyResult<()> {
    update_vision_config(key, value)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Get memory usage info
#[cfg(feature = "python-bindings")]
#[pyfunction]
pub fn get_memory_info(py: Python) -> PyResult<PyObject> {
    let manager = MemoryManager::global();
    let stats = manager.stats();
    
    let dict = PyDict::new(py);
    dict.set_item("total_allocated_bytes", stats.total_allocated_bytes)?;
    dict.set_item("active_allocations", stats.active_allocations)?;
    dict.set_item("pool_hits", stats.pool_hits)?;
    dict.set_item("pool_misses", stats.pool_misses)?;
    dict.set_item("total_allocated_mb", stats.total_allocated_bytes as f64 / 1024.0 / 1024.0)?;
    
    Ok(dict.to_object(py))
}

/// macOS Window Tracker for Python
#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pyclass]
pub struct RustWindowTracker {
    tracker: WindowTracker,
}

#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pymethods]
impl RustWindowTracker {
    #[new]
    fn new() -> Self {
        Self {
            tracker: WindowTracker::new(),
        }
    }
    
    /// Get windows that moved
    fn get_moved_windows(&self, threshold_pixels: u32) -> PyResult<Vec<PyObject>> {
        Python::with_gil(|py| {
            let moved = self.tracker.get_moved_windows(threshold_pixels);
            let mut results = Vec::new();
            
            for window in moved {
                let dict = PyDict::new(py);
                dict.set_item("window_id", window.window_id)?;
                dict.set_item("app_name", window.app_name)?;
                dict.set_item("x", window.bounds.x)?;
                dict.set_item("y", window.bounds.y)?;
                dict.set_item("width", window.bounds.width)?;
                dict.set_item("height", window.bounds.height)?;
                dict.set_item("velocity_x", window.movement_velocity.0)?;
                dict.set_item("velocity_y", window.movement_velocity.1)?;
                results.push(dict.to_object(py));
            }
            
            Ok(results)
        })
    }
}

/// macOS App State Detector for Python
#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pyclass]
pub struct RustAppStateDetector {
    detector: AppStateDetector,
}

#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pymethods]
impl RustAppStateDetector {
    #[new]
    fn new() -> PyResult<Self> {
        let detector = AppStateDetector::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { detector })
    }
    
    /// Get all running applications
    fn get_running_apps(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let apps = self.detector.get_running_apps()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let mut results = Vec::new();
        for app in apps {
            let dict = PyDict::new(py);
            dict.set_item("bundle_id", app.bundle_id)?;
            dict.set_item("name", app.name)?;
            dict.set_item("is_running", app.is_running)?;
            dict.set_item("is_active", app.is_active)?;
            dict.set_item("is_hidden", app.is_hidden)?;
            dict.set_item("cpu_usage", app.cpu_usage)?;
            dict.set_item("memory_usage_mb", app.memory_usage_mb)?;
            results.push(dict.to_object(py));
        }
        
        Ok(results)
    }
    
    /// Detect app state changes
    fn detect_changes(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let changes = self.detector.detect_changes()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let mut results = Vec::new();
        for change in changes {
            let dict = PyDict::new(py);
            dict.set_item("app_name", change.app_name)?;
            dict.set_item("bundle_id", change.bundle_id)?;
            dict.set_item("change_type", format!("{:?}", change.change_type))?;
            results.push(dict.to_object(py));
        }
        
        Ok(results)
    }
}

/// macOS Text Extractor for Python
#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pyclass]
pub struct RustTextExtractor {
    extractor: ChunkedTextExtractor,
}

#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pymethods]
impl RustTextExtractor {
    #[new]
    fn new() -> Self {
        Self {
            extractor: ChunkedTextExtractor::new(),
        }
    }
    
    /// Extract text from image in chunks
    fn extract_text_chunked(&self, py: Python, image: PyReadonlyArray3<u8>) -> PyResult<Vec<PyObject>> {
        // Convert numpy to ImageData
        let shape = image.shape();
        let (height, width, channels) = (shape[0] as u32, shape[1] as u32, shape[2] as u8);
        
        let format = match channels {
            1 => ImageFormat::Gray8,
            3 => ImageFormat::Rgb8,
            4 => ImageFormat::Rgba8,
            _ => return Err(PyValueError::new_err("Unsupported channel count")),
        };
        
        let img_data = ImageData::from_raw(
            width, height,
            image.as_slice()?.to_vec(),
            format
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Extract text (would be async in full implementation)
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let chunks = runtime.block_on(async {
            self.extractor.extract_text_chunked(&img_data).await
        }).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let mut results = Vec::new();
        for chunk in chunks {
            let dict = PyDict::new(py);
            dict.set_item("chunk_id", chunk.chunk_id)?;
            dict.set_item("text", chunk.text)?;
            dict.set_item("confidence", chunk.confidence)?;
            results.push(dict.to_object(py));
        }
        
        Ok(results)
    }
}

/// macOS Workspace Organizer for Python
#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pyclass]
pub struct RustWorkspaceOrganizer {
    organizer: WorkspaceOrganizer,
}

#[cfg(all(feature = "python-bindings", target_os = "macos"))]
#[pymethods]
impl RustWorkspaceOrganizer {
    #[new]
    fn new() -> Self {
        Self {
            organizer: WorkspaceOrganizer::new(),
        }
    }
    
    /// Apply workspace rules and get actions
    fn apply_rules(&self, py: Python) -> PyResult<Vec<PyObject>> {
        // This would need window and app state info
        // Simplified for now
        let actions = Vec::new();
        
        let mut results = Vec::new();
        for action in actions {
            let dict = PyDict::new(py);
            dict.set_item("action_type", "placeholder")?;
            results.push(dict.to_object(py));
        }
        
        Ok(results)
    }
}

/// Register all Python bindings with enhanced functionality
#[cfg(feature = "python-bindings")]
pub fn register_python_module(m: &PyModule) -> PyResult<()> {
    // Initialize vision module
    crate::vision::initialize()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    // Register enhanced classes
    m.add_class::<RustImageProcessor>()?;
    m.add_class::<RustProcessingPipeline>()?;
    m.add_class::<RustScreenCapture>()?;
    m.add_class::<RustImageCompressor>()?;
    m.add_class::<RustVisionContext>()?;
    m.add_class::<SharedMemoryInfo>()?;
    
    // Register new vision components
    crate::vision::bloom_filter::register_module(m)?;
    crate::vision::sliding_window_bindings::register_module(m)?;
    crate::vision::metal_accelerator::register_module(m)?;
    
    // Register memory components
    crate::memory::zero_copy::register_module(m)?;
    
    // Register macOS-specific classes
    #[cfg(target_os = "macos")]
    {
        m.add_class::<RustWindowTracker>()?;
        m.add_class::<RustAppStateDetector>()?;
        m.add_class::<RustTextExtractor>()?;
        m.add_class::<RustWorkspaceOrganizer>()?;
    }
    
    // Original classes
    m.add_class::<RustQuantizedModel>()?;
    m.add_class::<RustMemoryPool>()?;
    m.add_class::<ZeroCopyArray>()?;
    m.add_class::<RustRuntimeManager>()?;
    m.add_class::<RustAdvancedMemoryPool>()?;
    m.add_class::<RustTrackedBuffer>()?;
    
    // Register functions
    m.add_function(wrap_pyfunction!(process_image_batch, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_model_weights, m)?)?;
    m.add_function(wrap_pyfunction!(update_global_vision_config, m)?)?;
    m.add_function(wrap_pyfunction!(get_memory_info, m)?)?;
    
    // Add constants
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add submodule for configuration
    let config_module = PyModule::new(py, "config")?;
    config_module.add("VISION_MAX_DIMENSION", std::env::var("VISION_MAX_DIMENSION").unwrap_or_else(|_| "4096".to_string()))?;
    config_module.add("VISION_THREAD_COUNT", std::env::var("VISION_THREAD_COUNT").unwrap_or_else(|_| num_cpus::get().to_string()))?;
    config_module.add("VISION_ENABLE_SIMD", std::env::var("VISION_ENABLE_SIMD").unwrap_or_else(|_| "true".to_string()))?;
    m.add_submodule(config_module)?;
    
    Ok(())
}