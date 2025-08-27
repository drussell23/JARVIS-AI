//! Zero-copy memory pool for efficient tensor management

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// Memory pool options
#[pyclass]
#[derive(Clone)]
pub struct PoolOptions {
    /// Initial pool size in MB
    pub initial_size_mb: usize,
    /// Maximum pool size in MB
    pub max_size_mb: usize,
    /// Chunk size for allocations
    pub chunk_size: usize,
    /// Enable memory reuse
    pub enable_reuse: bool,
}

#[pymethods]
impl PoolOptions {
    #[new]
    fn new() -> Self {
        PoolOptions {
            initial_size_mb: 100,  // 100MB initial
            max_size_mb: 2000,     // 2GB max
            chunk_size: 4096,      // 4KB chunks
            enable_reuse: true,
        }
    }
}

/// Memory chunk
struct MemoryChunk {
    data: Vec<u8>,
    in_use: bool,
    last_used: std::time::Instant,
}

/// Thread-safe memory pool
#[pyclass]
pub struct MemoryPool {
    chunks: Arc<Mutex<VecDeque<MemoryChunk>>>,
    options: PoolOptions,
    allocated_bytes: Arc<Mutex<usize>>,
}

#[pymethods]
impl MemoryPool {
    #[new]
    fn new(options: Option<PoolOptions>) -> Self {
        let opts = options.unwrap_or_else(PoolOptions::new);
        let mut pool = MemoryPool {
            chunks: Arc::new(Mutex::new(VecDeque::new())),
            options: opts.clone(),
            allocated_bytes: Arc::new(Mutex::new(0)),
        };
        
        // Pre-allocate initial chunks
        pool.preallocate(opts.initial_size_mb * 1024 * 1024);
        
        pool
    }
    
    /// Allocate memory from pool
    fn allocate(&self, size: usize) -> PyResult<Py<PyBytes>> {
        Python::with_gil(|py| {
            let mut chunks = self.chunks.lock().unwrap();
            let aligned_size = (size + self.options.chunk_size - 1) / self.options.chunk_size * self.options.chunk_size;
            
            // Try to reuse existing chunk
            if self.options.enable_reuse {
                for chunk in chunks.iter_mut() {
                    if !chunk.in_use && chunk.data.len() >= aligned_size {
                        chunk.in_use = true;
                        chunk.last_used = std::time::Instant::now();
                        
                        // Create Python bytes object without copying
                        let bytes = unsafe {
                            PyBytes::from_ptr(py, chunk.data.as_ptr(), size)
                        };
                        return Ok(bytes.into());
                    }
                }
            }
            
            // Check if we can allocate more
            let mut allocated = self.allocated_bytes.lock().unwrap();
            if *allocated + aligned_size > self.options.max_size_mb * 1024 * 1024 {
                // Try to free old chunks
                self.cleanup_old_chunks(&mut chunks, &mut allocated);
                
                if *allocated + aligned_size > self.options.max_size_mb * 1024 * 1024 {
                    return Err(pyo3::exceptions::PyMemoryError::new_err(
                        "Memory pool exhausted"
                    ));
                }
            }
            
            // Allocate new chunk
            let mut new_chunk = MemoryChunk {
                data: vec![0u8; aligned_size],
                in_use: true,
                last_used: std::time::Instant::now(),
            };
            
            *allocated += aligned_size;
            
            // Create Python bytes object
            let bytes = unsafe {
                PyBytes::from_ptr(py, new_chunk.data.as_ptr(), size)
            };
            
            chunks.push_back(new_chunk);
            Ok(bytes.into())
        })
    }
    
    /// Get zero-copy tensor buffer
    fn get_tensor_buffer(&self, shape: Vec<usize>, dtype_size: usize) -> PyResult<Py<PyBytes>> {
        let total_size = shape.iter().product::<usize>() * dtype_size;
        self.allocate(total_size)
    }
    
    /// Release memory back to pool
    fn release(&self, ptr: usize) -> PyResult<()> {
        let mut chunks = self.chunks.lock().unwrap();
        
        for chunk in chunks.iter_mut() {
            if chunk.data.as_ptr() as usize == ptr {
                chunk.in_use = false;
                chunk.last_used = std::time::Instant::now();
                return Ok(());
            }
        }
        
        Ok(()) // Ignore if not found
    }
    
    /// Get pool statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let chunks = self.chunks.lock().unwrap();
            let allocated = self.allocated_bytes.lock().unwrap();
            
            let in_use = chunks.iter().filter(|c| c.in_use).count();
            let free = chunks.iter().filter(|c| !c.in_use).count();
            
            let stats = pyo3::types::PyDict::new(py);
            stats.set_item("total_chunks", chunks.len())?;
            stats.set_item("in_use_chunks", in_use)?;
            stats.set_item("free_chunks", free)?;
            stats.set_item("allocated_mb", *allocated as f64 / 1024.0 / 1024.0)?;
            stats.set_item("max_mb", self.options.max_size_mb)?;
            
            Ok(stats.into())
        })
    }
    
    /// Clear unused memory
    fn clear_unused(&self) -> PyResult<usize> {
        let mut chunks = self.chunks.lock().unwrap();
        let mut allocated = self.allocated_bytes.lock().unwrap();
        
        let initial_count = chunks.len();
        let mut freed_bytes = 0;
        
        // Remove unused chunks
        chunks.retain(|chunk| {
            if !chunk.in_use {
                freed_bytes += chunk.data.len();
                false
            } else {
                true
            }
        });
        
        *allocated -= freed_bytes;
        let freed_count = initial_count - chunks.len();
        
        Ok(freed_count)
    }
    
    /// Create shared memory view for zero-copy operations
    fn create_shared_view(&self, size: usize) -> PyResult<SharedMemoryView> {
        let buffer = self.allocate(size)?;
        
        Python::with_gil(|py| {
            let ptr = buffer.as_ref(py).as_bytes().as_ptr() as usize;
            Ok(SharedMemoryView {
                pool: self.clone_ref(),
                ptr,
                size,
            })
        })
    }
}

impl MemoryPool {
    /// Pre-allocate memory
    fn preallocate(&mut self, bytes: usize) {
        let mut chunks = self.chunks.lock().unwrap();
        let mut allocated = self.allocated_bytes.lock().unwrap();
        
        let num_chunks = bytes / self.options.chunk_size;
        
        for _ in 0..num_chunks {
            chunks.push_back(MemoryChunk {
                data: vec![0u8; self.options.chunk_size],
                in_use: false,
                last_used: std::time::Instant::now(),
            });
            *allocated += self.options.chunk_size;
        }
    }
    
    /// Clean up old unused chunks
    fn cleanup_old_chunks(&self, chunks: &mut VecDeque<MemoryChunk>, 
                         allocated: &mut usize) {
        let threshold = std::time::Duration::from_secs(60); // 1 minute
        let now = std::time::Instant::now();
        
        chunks.retain(|chunk| {
            if !chunk.in_use && now.duration_since(chunk.last_used) > threshold {
                *allocated -= chunk.data.len();
                false
            } else {
                true
            }
        });
    }
    
    /// Clone reference for Python
    fn clone_ref(&self) -> Self {
        MemoryPool {
            chunks: Arc::clone(&self.chunks),
            options: self.options.clone(),
            allocated_bytes: Arc::clone(&self.allocated_bytes),
        }
    }
}

/// Shared memory view for zero-copy operations
#[pyclass]
pub struct SharedMemoryView {
    pool: MemoryPool,
    ptr: usize,
    size: usize,
}

#[pymethods]
impl SharedMemoryView {
    /// Get pointer address
    fn get_ptr(&self) -> usize {
        self.ptr
    }
    
    /// Get size
    fn get_size(&self) -> usize {
        self.size
    }
    
    /// Release view
    fn release(&self) -> PyResult<()> {
        self.pool.release(self.ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        pyo3::prepare_freethreaded_python();
        
        Python::with_gil(|_py| {
            let pool = MemoryPool::new(None);
            
            // Test allocation
            let result = pool.allocate(1024);
            assert!(result.is_ok());
            
            // Test stats
            let stats = pool.get_stats();
            assert!(stats.is_ok());
        });
    }
}