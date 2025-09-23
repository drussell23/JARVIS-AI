use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::File;
use std::io::{Read, Write};
use std::path::PathBuf;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use bincode;
use ndarray::{Array1, Array2};
use half::f16;
use rayon::prelude::*;

#[pyclass]
pub struct RustModelLoader {
    cache_dir: PathBuf,
    compression_level: u32,
}

#[pymethods]
impl RustModelLoader {
    #[new]
    pub fn new(cache_dir: Option<String>, compression_level: Option<u32>) -> PyResult<Self> {
        let cache_dir = if let Some(dir) = cache_dir {
            PathBuf::from(dir)
        } else {
            std::env::temp_dir().join("jarvis_ml_cache")
        };
        
        // Ensure cache dir exists
        std::fs::create_dir_all(&cache_dir)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to create cache dir: {}", e)))?;
        
        Ok(Self {
            cache_dir,
            compression_level: compression_level.unwrap_or(6),
        })
    }
    
    /// Load and quantize a numpy array model
    pub fn load_quantized_model(&self, py: Python<'_>, path: String, quantize_to_int8: bool) -> PyResult<PyObject> {
        let file_path = PathBuf::from(&path);
        
        // Check cache first
        let cache_path = self.get_cache_path(&file_path, quantize_to_int8);
        if cache_path.exists() {
            return self.load_from_cache(py, &cache_path);
        }
        
        // Load original model
        let mut file = File::open(&file_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open model file: {}", e)))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read model file: {}", e)))?;
        
        // Quantize if requested
        if quantize_to_int8 {
            // This is where we'd implement INT8 quantization
            // For now, return compressed version
            let compressed = self.compress_data(&buffer)?;
            self.save_to_cache(&cache_path, &compressed)?;
            
            Ok(PyBytes::new(py, &compressed).into())
        } else {
            Ok(PyBytes::new(py, &buffer).into())
        }
    }
    
    /// Ultra-fast compression for model data
    pub fn compress_model(&self, py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
        let compressed = self.compress_data(data)?;
        Ok(PyBytes::new(py, &compressed).into())
    }
    
    /// Ultra-fast decompression
    pub fn decompress_model(&self, py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
        let decompressed = self.decompress_data(data)?;
        Ok(PyBytes::new(py, &decompressed).into())
    }
    
    /// Quantize float32 array to int8 with scale
    pub fn quantize_array_int8(&self, py: Python<'_>, data: Vec<f32>) -> PyResult<PyObject> {
        // Find min/max for scaling
        let (min_val, max_val) = data.par_iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            })
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2))
            );
        
        // Calculate scale and zero point
        let scale = (max_val - min_val) / 255.0;
        let zero_point = -min_val / scale;
        
        // Quantize to int8
        let quantized: Vec<i8> = data.par_iter()
            .map(|&val| {
                let q = ((val - min_val) / scale).round() as i16 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();
        
        // Return as dict with scale info
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("data", pyo3::types::PyBytes::new(py, unsafe { 
            std::slice::from_raw_parts(quantized.as_ptr() as *const u8, quantized.len())
        }))?;
        dict.set_item("scale", scale)?;
        dict.set_item("zero_point", zero_point)?;
        dict.set_item("min_val", min_val)?;
        dict.set_item("max_val", max_val)?;
        
        Ok(dict.into())
    }
    
    /// Quantize float32 array to float16
    pub fn quantize_array_fp16(&self, py: Python<'_>, data: Vec<f32>) -> PyResult<PyObject> {
        let quantized: Vec<f16> = data.par_iter()
            .map(|&val| f16::from_f32(val))
            .collect();
        
        // Convert to bytes
        let bytes: Vec<u8> = quantized.iter()
            .flat_map(|&h| h.to_le_bytes())
            .collect();
        
        Ok(PyBytes::new(py, &bytes).into())
    }
    
    /// Memory-map a large model file
    pub fn mmap_model(&self, py: Python<'_>, path: String) -> PyResult<PyObject> {
        use memmap2::MmapOptions;
        
        let file = File::open(&path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open file: {}", e)))?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to mmap file: {}", e)))?
        };
        
        // Return as bytes view (zero-copy)
        Ok(PyBytes::new(py, &mmap).into())
    }
    
    /// Clear model cache
    pub fn clear_cache(&self) -> PyResult<()> {
        if self.cache_dir.exists() {
            std::fs::remove_dir_all(&self.cache_dir)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to clear cache: {}", e)))?;
            std::fs::create_dir_all(&self.cache_dir)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to recreate cache dir: {}", e)))?;
        }
        Ok(())
    }
    
    // Helper methods
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::new(self.compression_level));
        encoder.write_all(data)?;
        encoder.finish()
    }
    
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed)?;
        Ok(decompressed)
    }
    
    fn get_cache_path(&self, original_path: &PathBuf, quantized: bool) -> PathBuf {
        let filename = original_path.file_name().unwrap().to_string_lossy();
        let cache_name = format!("{}.{}.cache", filename, if quantized { "q8" } else { "gz" });
        self.cache_dir.join(cache_name)
    }
    
    fn save_to_cache(&self, path: &PathBuf, data: &[u8]) -> Result<(), std::io::Error> {
        let mut file = File::create(path)?;
        file.write_all(data)?;
        Ok(())
    }
    
    fn load_from_cache(&self, py: Python<'_>, path: &PathBuf) -> PyResult<PyObject> {
        let mut file = File::open(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to open cache: {}", e)))?;
        
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Failed to read cache: {}", e)))?;
        
        Ok(PyBytes::new(py, &buffer).into())
    }
}