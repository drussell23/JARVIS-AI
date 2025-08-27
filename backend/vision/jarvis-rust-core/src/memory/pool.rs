//! Buffer pool implementation for efficient memory reuse

use std::sync::Arc;
use parking_lot::Mutex;
use crossbeam::queue::ArrayQueue;
use crate::{Result, JarvisError};

/// Buffer pool for fixed-size allocations
pub struct BufferPool {
    buffer_size: usize,
    capacity: usize,
    pool: Arc<ArrayQueue<Vec<u8>>>,
    allocated: Arc<Mutex<usize>>,
}

impl BufferPool {
    /// Create new buffer pool
    pub fn new(buffer_size: usize, capacity: usize) -> Self {
        let pool = Arc::new(ArrayQueue::new(capacity));
        
        // Pre-allocate some buffers
        let preallocate = capacity / 2;
        for _ in 0..preallocate {
            let buffer = vec![0u8; buffer_size];
            let _ = pool.push(buffer);
        }
        
        Self {
            buffer_size,
            capacity,
            pool,
            allocated: Arc::new(Mutex::new(preallocate)),
        }
    }
    
    /// Get buffer size for this pool
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }
    
    /// Acquire buffer from pool
    pub fn acquire(&self) -> Result<PooledBuffer> {
        // Try to get from pool
        if let Some(mut buffer) = self.pool.pop() {
            // Clear buffer for security
            buffer.fill(0);
            return Ok(PooledBuffer {
                buffer,
                pool: Some(self.pool.clone()),
                size: self.buffer_size,
            });
        }
        
        // Pool empty, allocate new buffer if under capacity
        let mut allocated = self.allocated.lock();
        if *allocated < self.capacity {
            *allocated += 1;
            drop(allocated);
            
            let buffer = vec![0u8; self.buffer_size];
            Ok(PooledBuffer {
                buffer,
                pool: Some(self.pool.clone()),
                size: self.buffer_size,
            })
        } else {
            Err(JarvisError::MemoryError("Buffer pool exhausted".to_string()))
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            buffer_size: self.buffer_size,
            capacity: self.capacity,
            available: self.pool.len(),
            allocated: *self.allocated.lock(),
        }
    }
}

/// Statistics for buffer pool
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub buffer_size: usize,
    pub capacity: usize,
    pub available: usize,
    pub allocated: usize,
}

/// Buffer that returns to pool when dropped
pub struct PooledBuffer {
    buffer: Vec<u8>,
    pool: Option<Arc<ArrayQueue<Vec<u8>>>>,
    size: usize,
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("size", &self.size)
            .field("has_pool", &self.pool.is_some())
            .finish()
    }
}

impl PooledBuffer {
    /// Create new buffer without pool (direct allocation)
    pub fn new_direct(size: usize) -> Self {
        Self {
            buffer: vec![0u8; size],
            pool: None,
            size,
        }
    }
    
    /// Get buffer length
    pub fn len(&self) -> usize {
        self.size
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    
    /// Get as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer[..self.size]
    }
    
    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer[..self.size]
    }
    
    /// Get pointer
    pub fn as_ptr(&self) -> *const u8 {
        self.buffer.as_ptr()
    }
    
    /// Get mutable pointer
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer.as_mut_ptr()
    }
    
    /// Take ownership of inner buffer
    pub fn into_vec(mut self) -> Vec<u8> {
        self.pool = None;  // Prevent returning to pool
        std::mem::take(&mut self.buffer)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool if it came from one
        if let Some(pool) = &self.pool {
            if !self.buffer.is_empty() {
                let buffer = std::mem::take(&mut self.buffer);
                let _ = pool.push(buffer);  // Ignore if pool is full
            }
        }
    }
}

impl AsRef<[u8]> for PooledBuffer {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsMut<[u8]> for PooledBuffer {
    fn as_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_pool() {
        let pool = BufferPool::new(1024, 10);
        
        // Acquire buffers
        let mut buffers = Vec::new();
        for _ in 0..5 {
            let buffer = pool.acquire().unwrap();
            assert_eq!(buffer.len(), 1024);
            buffers.push(buffer);
        }
        
        // Check stats
        let stats = pool.stats();
        assert_eq!(stats.buffer_size, 1024);
        assert!(stats.allocated >= 5);
        
        // Drop buffers to return to pool
        drop(buffers);
        
        // Should be able to acquire again
        let _buffer = pool.acquire().unwrap();
    }
}