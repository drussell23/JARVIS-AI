//! Buffer recycling system for zero-allocation operations

use std::sync::Arc;
use parking_lot::Mutex;
use std::collections::VecDeque;
use crate::Result;

/// Buffer recycler for reusing allocations
pub struct BufferRecycler {
    bins: Vec<Mutex<RecycleBin>>,
}

struct RecycleBin {
    size_class: usize,
    buffers: VecDeque<Vec<u8>>,
    max_buffers: usize,
}

impl BufferRecycler {
    pub fn new() -> Self {
        // Create bins for different size classes (powers of 2)
        let size_classes = vec![
            256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072
        ];
        
        let bins = size_classes.into_iter()
            .map(|size| Mutex::new(RecycleBin {
                size_class: size,
                buffers: VecDeque::new(),
                max_buffers: 10,
            }))
            .collect();
        
        Self { bins }
    }
    
    /// Get recycled buffer or allocate new one
    pub fn acquire(&self, size: usize) -> RecyclableBuffer {
        // Find appropriate bin
        let bin_idx = self.find_bin_index(size);
        
        if let Some(bin_mutex) = self.bins.get(bin_idx) {
            let mut bin = bin_mutex.lock();
            
            // Try to get recycled buffer
            if let Some(mut buffer) = bin.buffers.pop_front() {
                buffer.clear();
                buffer.resize(size, 0);
                return RecyclableBuffer {
                    buffer,
                    recycler: Some(Arc::new(self as *const BufferRecycler)),
                    original_capacity: bin.size_class,
                };
            }
        }
        
        // No recycled buffer available, allocate new
        let capacity = self.round_up_size(size);
        RecyclableBuffer {
            buffer: vec![0u8; size],
            recycler: Some(Arc::new(self as *const BufferRecycler)),
            original_capacity: capacity,
        }
    }
    
    /// Return buffer for recycling
    fn recycle(&self, mut buffer: Vec<u8>, original_capacity: usize) {
        let bin_idx = self.find_bin_index(original_capacity);
        
        if let Some(bin_mutex) = self.bins.get(bin_idx) {
            let mut bin = bin_mutex.lock();
            
            // Only keep if under limit and buffer is reasonably sized
            if bin.buffers.len() < bin.max_buffers && buffer.capacity() <= bin.size_class * 2 {
                buffer.clear();
                bin.buffers.push_back(buffer);
            }
        }
    }
    
    /// Find bin index for size
    fn find_bin_index(&self, size: usize) -> usize {
        self.bins.iter()
            .position(|bin| bin.lock().size_class >= size)
            .unwrap_or(self.bins.len() - 1)
    }
    
    /// Round up to next size class
    fn round_up_size(&self, size: usize) -> usize {
        for bin in &self.bins {
            let size_class = bin.lock().size_class;
            if size_class >= size {
                return size_class;
            }
        }
        size
    }
}

/// Buffer that can be recycled
pub struct RecyclableBuffer {
    buffer: Vec<u8>,
    recycler: Option<Arc<*const BufferRecycler>>,
    original_capacity: usize,
}

impl RecyclableBuffer {
    /// Create without recycler
    pub fn new(size: usize) -> Self {
        Self {
            buffer: vec![0u8; size],
            recycler: None,
            original_capacity: size,
        }
    }
    
    /// Get as slice
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
    
    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buffer
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    /// Resize buffer
    pub fn resize(&mut self, new_size: usize, value: u8) {
        self.buffer.resize(new_size, value);
    }
    
    /// Take ownership of buffer
    pub fn into_vec(mut self) -> Vec<u8> {
        self.recycler = None;  // Prevent recycling
        std::mem::take(&mut self.buffer)
    }
}

impl Drop for RecyclableBuffer {
    fn drop(&mut self) {
        if let Some(recycler_ptr) = &self.recycler {
            if !self.buffer.is_empty() {
                let buffer = std::mem::take(&mut self.buffer);
                unsafe {
                    let recycler = &**recycler_ptr;
                    (*recycler).recycle(buffer, self.original_capacity);
                }
            }
        }
    }
}

unsafe impl Send for RecyclableBuffer {}
unsafe impl Sync for RecyclableBuffer {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_buffer_recycling() {
        let recycler = BufferRecycler::new();
        
        // Acquire and release buffers
        {
            let buf1 = recycler.acquire(500);
            assert!(buf1.len() >= 500);
            
            let buf2 = recycler.acquire(1000);
            assert!(buf2.len() >= 1000);
        } // Buffers returned to recycler
        
        // Acquire again - should get recycled buffers
        let buf3 = recycler.acquire(400);
        assert!(buf3.len() >= 400);
    }
}