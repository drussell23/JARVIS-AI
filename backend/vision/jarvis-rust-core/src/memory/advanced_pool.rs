//! Advanced memory pool with leak detection and automatic cleanup

use std::sync::{Arc, Weak};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use dashmap::DashMap;
use flume::{Sender, Receiver};
use bytes::{Bytes, BytesMut};
use crate::{Result, JarvisError};

/// Memory allocation metadata
#[derive(Debug, Clone)]
pub struct AllocationMetadata {
    pub id: u64,
    pub size: usize,
    pub allocated_at: Instant,
    pub last_accessed: Instant,
    pub access_count: u64,
    pub stack_trace: Option<String>,
}

/// Advanced buffer pool with automatic memory management
pub struct AdvancedBufferPool {
    /// Pools organized by size class
    size_class_pools: Vec<SizeClassPool>,
    /// Active allocations tracking
    active_allocations: Arc<DashMap<u64, AllocationMetadata>>,
    /// Leak detector
    leak_detector: Arc<LeakDetector>,
    /// Memory pressure monitor
    pressure_monitor: Arc<MemoryPressureMonitor>,
    /// Allocation ID counter
    next_id: Arc<Mutex<u64>>,
}

/// Pool for specific size class
struct SizeClassPool {
    size: usize,
    /// Available buffers
    available: Arc<RwLock<Vec<BytesMut>>>,
    /// Maximum capacity
    capacity: usize,
    /// High water mark
    high_water_mark: Arc<Mutex<usize>>,
}

/// Detects and reports memory leaks
pub struct LeakDetector {
    /// Weak references to all allocations
    allocations: Arc<DashMap<u64, Weak<AllocationMetadata>>>,
    /// Leak detection channel
    leak_channel: (Sender<LeakReport>, Receiver<LeakReport>),
    /// Detection threshold
    leak_threshold: Duration,
}

#[derive(Debug, Clone)]
pub struct LeakReport {
    pub allocation_id: u64,
    pub size: usize,
    pub age: Duration,
    pub stack_trace: Option<String>,
}

/// Monitors system memory pressure
pub struct MemoryPressureMonitor {
    /// Pressure callbacks
    callbacks: Arc<RwLock<Vec<Box<dyn Fn(MemoryPressure) + Send + Sync>>>>,
    /// Current pressure level
    current_pressure: Arc<RwLock<MemoryPressure>>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

impl AdvancedBufferPool {
    pub fn new() -> Self {
        // Create size classes: 1KB, 4KB, 16KB, 64KB, 256KB, 1MB, 4MB, 16MB
        let size_classes = vec![
            1024,           // 1KB
            4096,           // 4KB
            16384,          // 16KB
            65536,          // 64KB
            262144,         // 256KB
            1048576,        // 1MB
            4194304,        // 4MB
            16777216,       // 16MB
        ];
        
        let size_class_pools: Vec<SizeClassPool> = size_classes
            .into_iter()
            .map(|size| SizeClassPool {
                size,
                available: Arc::new(RwLock::new(Vec::with_capacity(32))),
                capacity: 32,
                high_water_mark: Arc::new(Mutex::new(0)),
            })
            .collect();
        
        let leak_detector = Arc::new(LeakDetector::new(Duration::from_secs(300))); // 5 minute threshold
        let pressure_monitor = Arc::new(MemoryPressureMonitor::new());
        
        // Start background tasks
        let pool = Self {
            size_class_pools,
            active_allocations: Arc::new(DashMap::new()),
            leak_detector: leak_detector.clone(),
            pressure_monitor: pressure_monitor.clone(),
            next_id: Arc::new(Mutex::new(0)),
        };
        
        // Don't start background tasks here - they require a runtime
        // These should be started manually when a runtime is available
        // pool.start_leak_detection();
        
        // Start memory pressure monitoring
        // pool.start_pressure_monitoring();
        
        pool
    }
    
    /// Allocate buffer with tracking
    pub fn allocate(&self, size: usize) -> Result<TrackedBuffer> {
        // Find appropriate size class
        let pool_index = self.size_class_pools
            .iter()
            .position(|p| p.size >= size)
            .ok_or_else(|| JarvisError::MemoryError(
                format!("Requested size {} exceeds maximum pool size", size)
            ))?;
        
        let pool = &self.size_class_pools[pool_index];
        
        // Try to get from pool
        let buffer = {
            let mut available = pool.available.write();
            available.pop()
        };
        
        let buffer = match buffer {
            Some(mut buf) => {
                buf.clear();
                buf.resize(size, 0);
                buf
            }
            None => {
                // Check memory pressure before allocating
                self.check_memory_pressure()?;
                
                // Update high water mark
                let mut hwm = pool.high_water_mark.lock();
                *hwm = (*hwm).max(pool.capacity - pool.available.read().len() + 1);
                
                BytesMut::zeroed(pool.size)
            }
        };
        
        // Create tracked buffer
        let id = {
            let mut next_id = self.next_id.lock();
            let id = *next_id;
            *next_id += 1;
            id
        };
        
        let metadata = AllocationMetadata {
            id,
            size: pool.size,
            allocated_at: Instant::now(),
            last_accessed: Instant::now(),
            access_count: 1,
            stack_trace: Self::capture_stack_trace(),
        };
        
        self.active_allocations.insert(id, metadata.clone());
        self.leak_detector.track_allocation(id, Arc::new(metadata));
        
        Ok(TrackedBuffer {
            id,
            buffer: buffer.freeze(),
            pool: Some(Arc::downgrade(&self.size_class_pools[pool_index].available)),
            allocations: self.active_allocations.clone(),
            original_size: pool.size,
        })
    }
    
    /// Check memory pressure before allocation
    fn check_memory_pressure(&self) -> Result<()> {
        let pressure = *self.pressure_monitor.current_pressure.read();
        match pressure {
            MemoryPressure::Critical => {
                Err(JarvisError::MemoryError(
                    "Memory pressure critical - allocation denied".to_string()
                ))
            }
            MemoryPressure::High => {
                // Try to free some memory
                self.trim_pools(0.5);
                Ok(())
            }
            _ => Ok(())
        }
    }
    
    /// Trim pool capacity
    pub fn trim_pools(&self, fraction: f32) {
        for pool in &self.size_class_pools {
            let mut available = pool.available.write();
            let current_len = available.len();
            let trim_count = (current_len as f32 * fraction) as usize;
            available.truncate(current_len.saturating_sub(trim_count));
        }
    }
    
    /// Start leak detection background task
    fn start_leak_detection(&self) {
        let detector = self.leak_detector.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                detector.check_for_leaks();
            }
        });
    }
    
    /// Start memory pressure monitoring
    fn start_pressure_monitoring(&self) {
        let monitor = self.pressure_monitor.clone();
        let active_allocations = self.active_allocations.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            let mut sys = sysinfo::System::new();
            
            loop {
                interval.tick().await;
                
                sys.refresh_memory();
                let used_memory = sys.used_memory();
                let total_memory = sys.total_memory();
                let usage_percent = (used_memory as f64 / total_memory as f64) * 100.0;
                
                let pressure = match usage_percent {
                    x if x < 50.0 => MemoryPressure::Low,
                    x if x < 75.0 => MemoryPressure::Medium,
                    x if x < 90.0 => MemoryPressure::High,
                    _ => MemoryPressure::Critical,
                };
                
                monitor.update_pressure(pressure);
                
                // Log metrics
                let active_count = active_allocations.len();
                let total_allocated: usize = active_allocations
                    .iter()
                    .map(|entry| entry.value().size)
                    .sum();
                
                tracing::debug!(
                    "Memory pressure: {:?}, Usage: {:.1}%, Active allocations: {}, Total size: {} MB",
                    pressure, usage_percent, active_count, total_allocated / 1024 / 1024
                );
            }
        });
    }
    
    /// Capture stack trace for debugging
    fn capture_stack_trace() -> Option<String> {
        #[cfg(debug_assertions)]
        {
            Some(format!("{:?}", std::backtrace::Backtrace::capture()))
        }
        #[cfg(not(debug_assertions))]
        {
            None
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStatistics {
        let mut stats = PoolStatistics {
            size_classes: Vec::new(),
            total_active: self.active_allocations.len(),
            total_allocated_bytes: 0,
            pressure: *self.pressure_monitor.current_pressure.read(),
        };
        
        for pool in &self.size_class_pools {
            let available = pool.available.read().len();
            let high_water_mark = *pool.high_water_mark.lock();
            
            stats.size_classes.push(SizeClassStats {
                size: pool.size,
                available,
                capacity: pool.capacity,
                high_water_mark,
            });
        }
        
        stats.total_allocated_bytes = self.active_allocations
            .iter()
            .map(|entry| entry.value().size)
            .sum();
        
        stats
    }
}

/// Tracked buffer that returns to pool on drop
pub struct TrackedBuffer {
    id: u64,
    buffer: Bytes,
    pool: Option<Weak<RwLock<Vec<BytesMut>>>>,
    allocations: Arc<DashMap<u64, AllocationMetadata>>,
    original_size: usize,
}

impl TrackedBuffer {
    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }
    
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
    
    pub fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for TrackedBuffer {
    fn drop(&mut self) {
        // Remove from active allocations
        self.allocations.remove(&self.id);
        
        // Return to pool if possible
        if let Some(pool_weak) = &self.pool {
            if let Some(pool) = pool_weak.upgrade() {
                let mut available = pool.write();
                if available.len() < available.capacity() {
                    let mut buffer = BytesMut::with_capacity(self.original_size);
                    buffer.extend_from_slice(&self.buffer);
                    available.push(buffer);
                }
            }
        }
    }
}

impl LeakDetector {
    fn new(threshold: Duration) -> Self {
        Self {
            allocations: Arc::new(DashMap::new()),
            leak_channel: flume::unbounded(),
            leak_threshold: threshold,
        }
    }
    
    fn track_allocation(&self, id: u64, metadata: Arc<AllocationMetadata>) {
        self.allocations.insert(id, Arc::downgrade(&metadata));
    }
    
    fn check_for_leaks(&self) {
        let now = Instant::now();
        let mut to_remove = Vec::new();
        
        for entry in self.allocations.iter() {
            let (id, weak_ref) = entry.pair();
            
            if let Some(metadata) = weak_ref.upgrade() {
                let age = now.duration_since(metadata.allocated_at);
                if age > self.leak_threshold {
                    let report = LeakReport {
                        allocation_id: *id,
                        size: metadata.size,
                        age,
                        stack_trace: metadata.stack_trace.clone(),
                    };
                    
                    let _ = self.leak_channel.0.try_send(report);
                }
            } else {
                // Weak reference is dead, remove it
                to_remove.push(*id);
            }
        }
        
        for id in to_remove {
            self.allocations.remove(&id);
        }
    }
    
    pub fn leak_receiver(&self) -> Receiver<LeakReport> {
        self.leak_channel.1.clone()
    }
}

impl MemoryPressureMonitor {
    fn new() -> Self {
        Self {
            callbacks: Arc::new(RwLock::new(Vec::new())),
            current_pressure: Arc::new(RwLock::new(MemoryPressure::Low)),
        }
    }
    
    fn update_pressure(&self, pressure: MemoryPressure) {
        let old_pressure = *self.current_pressure.read();
        if old_pressure != pressure {
            *self.current_pressure.write() = pressure;
            
            // Notify callbacks
            let callbacks = self.callbacks.read();
            for callback in callbacks.iter() {
                callback(pressure);
            }
        }
    }
    
    pub fn register_callback<F>(&self, callback: F)
    where
        F: Fn(MemoryPressure) + Send + Sync + 'static,
    {
        self.callbacks.write().push(Box::new(callback));
    }
}

#[derive(Debug, Clone)]
pub struct PoolStatistics {
    pub size_classes: Vec<SizeClassStats>,
    pub total_active: usize,
    pub total_allocated_bytes: usize,
    pub pressure: MemoryPressure,
}

#[derive(Debug, Clone)]
pub struct SizeClassStats {
    pub size: usize,
    pub available: usize,
    pub capacity: usize,
    pub high_water_mark: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_advanced_pool() {
        let pool = AdvancedBufferPool::new();
        
        // Test allocation
        let buf1 = pool.allocate(1024).unwrap();
        assert_eq!(buf1.len(), 1024);
        
        let buf2 = pool.allocate(8192).unwrap();
        assert!(buf2.len() >= 8192);
        
        // Test that buffers are tracked
        assert_eq!(pool.active_allocations.len(), 2);
        
        // Drop buffers
        drop(buf1);
        drop(buf2);
        
        // Should be removed from active
        assert_eq!(pool.active_allocations.len(), 0);
    }
}