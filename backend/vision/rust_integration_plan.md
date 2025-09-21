# Rust Integration Plan for JARVIS Proactive Monitoring System

## Overview
This plan outlines the integration of high-performance Rust components into JARVIS's proactive monitoring system for real-time video capture on macOS with 16GB RAM.

## Current State Analysis

### Existing Rust Components
1. **rust_bridge.py** - Python-Rust bridge with basic bindings
   - RustImageProcessor for batch processing
   - RustAdvancedMemoryPool for memory management
   - RustRuntimeManager for CPU task scheduling
   - Visual feature extraction functions

2. **rust_integration.py** - Advanced Rust integration
   - Zero-copy memory management with leak detection
   - Async runtime with CPU affinity
   - Quantized ML inference
   - Hardware-accelerated image processing

3. **bloom_filter_network.py** - Python implementation with Rust hooks
   - Has provisions for rust_bloom_hash acceleration
   - Currently falls back to Python MurmurHash3

4. **jarvis-rust-core/** - Rust library with extensive dependencies
   - Metal support for macOS GPU acceleration
   - SIMD operations for CPU optimization
   - Zero-copy serialization
   - High-performance image processing

### Current Issues
- Rust components exist but are not actively used
- RUST_AVAILABLE is False in most modules
- Rust library needs to be built with `maturin develop`
- No integration with real-time monitoring loops

## Integration Architecture

### Phase 1: Build and Activate Rust Core (Week 1)

#### 1.1 Build Rust Library
```bash
cd backend/vision/jarvis-rust-core
cargo build --release --features python-bindings,simd
maturin develop --release
```

#### 1.2 Verify Rust Availability
- Update rust_bridge.py to properly detect built library
- Add build verification script
- Create health check for Rust components

#### 1.3 Initialize Global Rust Accelerator
```python
# In main.py startup
from vision.rust_integration import initialize_rust_acceleration
rust_accelerator = initialize_rust_acceleration(
    enable_memory_pool=True,
    enable_runtime_manager=True,
    worker_threads=8,  # For 16GB RAM system
    enable_cpu_affinity=True
)
```

### Phase 2: Integrate Bloom Filter Acceleration (Week 1)

#### 2.1 Implement Rust Bloom Filter
```rust
// In jarvis-rust-core/src/bloom_filter.rs
pub struct RustBloomFilter {
    bit_array: Vec<u64>,  // 64-bit blocks for SIMD
    num_hashes: u32,
    size_bits: usize,
}

impl RustBloomFilter {
    pub fn new(size_mb: f32, num_hashes: u32) -> Self {
        let size_bits = (size_mb * 1024.0 * 1024.0 * 8.0) as usize;
        let num_blocks = (size_bits + 63) / 64;
        Self {
            bit_array: vec![0u64; num_blocks],
            num_hashes,
            size_bits,
        }
    }
    
    #[inline]
    pub fn add(&mut self, hash: u64) {
        for i in 0..self.num_hashes {
            let bit_pos = self.hash_position(hash, i);
            let block = bit_pos / 64;
            let bit = bit_pos % 64;
            self.bit_array[block] |= 1u64 << bit;
        }
    }
    
    #[inline]
    pub fn contains(&self, hash: u64) -> bool {
        for i in 0..self.num_hashes {
            let bit_pos = self.hash_position(hash, i);
            let block = bit_pos / 64;
            let bit = bit_pos % 64;
            if self.bit_array[block] & (1u64 << bit) == 0 {
                return false;
            }
        }
        true
    }
}
```

#### 2.2 Update bloom_filter_network.py
```python
# Replace Python bloom filter with Rust version
if RUST_AVAILABLE:
    self.rust_filter = jarvis_rust_core.RustBloomFilter(
        size_mb=self.size_mb,
        num_hashes=self.num_hashes
    )
```

### Phase 3: Implement Sliding Window Buffer (Week 2)

#### 3.1 Rust Ring Buffer for Frame History
```rust
// Circular buffer for zero-copy frame storage
pub struct FrameRingBuffer {
    buffer: Arc<Mutex<Vec<u8>>>,
    frame_offsets: Vec<(usize, usize, u64)>,  // (start, len, timestamp)
    capacity: usize,
    write_pos: usize,
}

impl FrameRingBuffer {
    pub fn new(capacity_mb: usize) -> Self {
        let capacity = capacity_mb * 1024 * 1024;
        Self {
            buffer: Arc::new(Mutex::new(vec![0u8; capacity])),
            frame_offsets: Vec::with_capacity(1000),
            capacity,
            write_pos: 0,
        }
    }
    
    pub fn add_frame(&mut self, frame_data: &[u8], timestamp: u64) -> Result<()> {
        // Zero-copy write with wraparound
        let mut buffer = self.buffer.lock().unwrap();
        let frame_len = frame_data.len();
        
        if self.write_pos + frame_len > self.capacity {
            self.write_pos = 0;  // Wrap around
        }
        
        buffer[self.write_pos..self.write_pos + frame_len]
            .copy_from_slice(frame_data);
        
        self.frame_offsets.push((self.write_pos, frame_len, timestamp));
        self.write_pos += frame_len;
        
        Ok(())
    }
}
```

#### 3.2 Integration with real_time_interaction_handler.py
```python
# Use Rust frame buffer for temporal analysis
self.frame_buffer = jarvis_rust_core.FrameRingBuffer(
    capacity_mb=500  # 500MB for ~30 seconds at 1080p
)

# In monitoring loop
async def _add_frame_to_buffer(self, screenshot: np.ndarray):
    # Convert to bytes efficiently
    frame_bytes = screenshot.tobytes()
    timestamp = int(time.time() * 1000)
    await self.frame_buffer.add_frame_async(frame_bytes, timestamp)
```

### Phase 4: Metal Acceleration for macOS (Week 2)

#### 4.1 Metal Compute Shaders
```rust
// Metal shader for real-time image processing
use metal::{Device, CommandQueue, ComputePipelineState};

pub struct MetalAccelerator {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
}

impl MetalAccelerator {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;
        let command_queue = device.new_command_queue();
        
        // Load compute shader
        let library = device.new_library_with_source(
            include_str!("shaders/image_process.metal"),
            &metal::CompileOptions::new(),
        )?;
        
        let kernel = library.get_function("processImage", None)?;
        let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel)?;
        
        Ok(Self {
            device,
            command_queue,
            pipeline_state,
        })
    }
    
    pub fn process_frame_batch(&self, frames: &[&[u8]]) -> Vec<Vec<u8>> {
        // Batch process frames on GPU
        // Implementation details...
    }
}
```

#### 4.2 Metal Shader for Difference Detection
```metal
// image_process.metal
kernel void detectFrameDifference(
    texture2d<float, access::read> current [[texture(0)]],
    texture2d<float, access::read> previous [[texture(1)]],
    texture2d<float, access::write> difference [[texture(2)]],
    constant float& threshold [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float4 currentPixel = current.read(gid);
    float4 previousPixel = previous.read(gid);
    
    float4 diff = abs(currentPixel - previousPixel);
    float magnitude = length(diff.rgb);
    
    if (magnitude > threshold) {
        difference.write(float4(1.0), gid);
    } else {
        difference.write(float4(0.0), gid);
    }
}
```

### Phase 5: Zero-Copy Memory Pipeline (Week 3)

#### 5.1 Shared Memory Architecture
```rust
// Zero-copy shared memory for IPC
use memmap2::{MmapMut, MmapOptions};
use std::sync::atomic::{AtomicU64, Ordering};

pub struct SharedVisionMemory {
    mmap: MmapMut,
    size: usize,
    write_offset: Arc<AtomicU64>,
    read_offset: Arc<AtomicU64>,
}

impl SharedVisionMemory {
    pub fn new(size_mb: usize) -> Result<Self> {
        let size = size_mb * 1024 * 1024;
        let file = tempfile::tempfile()?;
        file.set_len(size as u64)?;
        
        let mmap = unsafe {
            MmapOptions::new()
                .len(size)
                .map_mut(&file)?
        };
        
        Ok(Self {
            mmap,
            size,
            write_offset: Arc::new(AtomicU64::new(0)),
            read_offset: Arc::new(AtomicU64::new(0)),
        })
    }
    
    pub fn write_frame(&mut self, data: &[u8]) -> Result<()> {
        let offset = self.write_offset.load(Ordering::SeqCst) as usize;
        let new_offset = (offset + data.len()) % self.size;
        
        // Zero-copy write
        self.mmap[offset..offset + data.len()].copy_from_slice(data);
        self.write_offset.store(new_offset as u64, Ordering::SeqCst);
        
        Ok(())
    }
}
```

#### 5.2 Python Integration
```python
# In claude_vision_analyzer_main.py
class ZeroCopyVisionAnalyzer:
    def __init__(self):
        self.shared_memory = jarvis_rust_core.SharedVisionMemory(
            size_mb=1000  # 1GB shared buffer
        )
        self.metal_accelerator = jarvis_rust_core.MetalAccelerator()
    
    async def analyze_frame_batch(self, frames: List[np.ndarray]):
        # Write frames to shared memory
        for frame in frames:
            await self.shared_memory.write_frame_async(frame.tobytes())
        
        # Process on GPU with zero-copy
        results = await self.metal_accelerator.process_batch_async(
            self.shared_memory.get_frame_pointers()
        )
        
        return results
```

### Phase 6: Performance Optimization (Week 3-4)

#### 6.1 SIMD Optimization for Feature Extraction
```rust
use wide::f32x8;

#[inline]
pub fn extract_features_simd(image: &[u8]) -> Vec<f32> {
    let mut features = Vec::with_capacity(256);
    
    // Process 8 pixels at a time with SIMD
    for chunk in image.chunks_exact(8) {
        let pixels = f32x8::from([
            chunk[0] as f32, chunk[1] as f32,
            chunk[2] as f32, chunk[3] as f32,
            chunk[4] as f32, chunk[5] as f32,
            chunk[6] as f32, chunk[7] as f32,
        ]);
        
        // Compute features in parallel
        let mean = pixels.reduce_add() / 8.0;
        let variance = (pixels - f32x8::splat(mean))
            .mul_add(pixels - f32x8::splat(mean), f32x8::splat(0.0))
            .reduce_add() / 8.0;
        
        features.push(mean);
        features.push(variance.sqrt());
    }
    
    features
}
```

#### 6.2 Parallel Processing Pipeline
```rust
use rayon::prelude::*;
use crossbeam::channel;

pub struct ParallelVisionPipeline {
    thread_pool: rayon::ThreadPool,
    frame_channel: (channel::Sender<Frame>, channel::Receiver<Frame>),
}

impl ParallelVisionPipeline {
    pub fn new(num_threads: usize) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        
        let frame_channel = channel::bounded(100);
        
        Self {
            thread_pool,
            frame_channel,
        }
    }
    
    pub fn process_frames_parallel(&self, frames: Vec<Frame>) -> Vec<ProcessedFrame> {
        self.thread_pool.install(|| {
            frames.par_iter()
                .map(|frame| {
                    // Process each frame in parallel
                    let features = extract_features_simd(&frame.data);
                    let objects = detect_objects_metal(&frame.data);
                    ProcessedFrame {
                        features,
                        objects,
                        timestamp: frame.timestamp,
                    }
                })
                .collect()
        })
    }
}
```

### Phase 7: Integration with Proactive Monitoring (Week 4)

#### 7.1 Update real_time_interaction_handler.py
```python
class RustAcceleratedInteractionHandler:
    def __init__(self):
        # Initialize Rust components
        self.rust_accelerator = get_rust_accelerator()
        self.frame_buffer = jarvis_rust_core.FrameRingBuffer(500)
        self.bloom_network = jarvis_rust_core.RustBloomNetwork()
        self.vision_pipeline = jarvis_rust_core.ParallelVisionPipeline(8)
        
    async def _monitoring_loop(self):
        """Main monitoring loop with Rust acceleration"""
        batch_size = 5
        frame_batch = []
        
        while self._is_monitoring:
            try:
                # Capture screenshot
                screenshot = await self._capture_screenshot_async()
                
                # Check for duplicate with Rust bloom filter
                frame_hash = await self.rust_accelerator.hash_image_fast(
                    screenshot.tobytes()
                )
                
                if not self.bloom_network.contains(frame_hash):
                    self.bloom_network.add(frame_hash)
                    frame_batch.append(screenshot)
                    
                    # Process batch when ready
                    if len(frame_batch) >= batch_size:
                        results = await self._process_frame_batch(frame_batch)
                        frame_batch = []
                        
                        # Handle results
                        await self._handle_detection_results(results)
                
                await asyncio.sleep(0.1)  # 10 FPS
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _process_frame_batch(self, frames: List[np.ndarray]):
        """Process frame batch with full Rust acceleration"""
        # Convert to Rust format
        rust_frames = [f.tobytes() for f in frames]
        
        # Process in parallel with Metal GPU
        results = await self.vision_pipeline.process_frames_async(rust_frames)
        
        return results
```

#### 7.2 Update claude_vision_analyzer_main.py
```python
class RustAcceleratedVisionAnalyzer:
    def __init__(self):
        super().__init__()
        self.rust_components = {
            'image_processor': RustImageProcessor(),
            'memory_pool': RustAdvancedMemoryPool(),
            'metal_accelerator': MetalAccelerator() if sys.platform == 'darwin' else None,
            'zero_copy_pipeline': ZeroCopyVisionPipeline()
        }
    
    async def analyze_screenshot(self, screenshot, prompt="", **kwargs):
        """Analyze with Rust acceleration"""
        # Use zero-copy pipeline for performance
        if self.rust_components['zero_copy_pipeline']:
            result = await self.rust_components['zero_copy_pipeline'].process_image(
                screenshot,
                model_name='claude-vision'
            )
            
            if result['success']:
                # Continue with Claude API call using processed features
                enhanced_prompt = self._enhance_prompt_with_features(
                    prompt, result['features']
                )
                return await super().analyze_screenshot(
                    screenshot, enhanced_prompt, **kwargs
                )
        
        # Fallback to standard processing
        return await super().analyze_screenshot(screenshot, prompt, **kwargs)
```

### Phase 8: Memory Management Strategy (Ongoing)

#### 8.1 Dynamic Memory Allocation
```python
# Updated memory configuration for 16GB system
MEMORY_CONFIG = {
    'total_ram_gb': 16,
    'jarvis_allocation_percent': 40,  # 6.4GB for JARVIS
    'component_budgets': {
        'rust_memory_pool': 2048,  # 2GB for Rust pool
        'frame_buffer': 1024,      # 1GB for frame history
        'bloom_filters': 100,      # 100MB for deduplication
        'claude_vision': 2048,     # 2GB for vision processing
        'metal_buffers': 1024,     # 1GB for GPU buffers
        'overhead': 256            # 256MB overhead
    }
}

# In integration_orchestrator.py
def allocate_rust_memory():
    """Allocate memory with Rust components"""
    total_mb = MEMORY_CONFIG['total_ram_gb'] * 1024
    jarvis_mb = int(total_mb * MEMORY_CONFIG['jarvis_allocation_percent'] / 100)
    
    # Initialize Rust memory pool
    rust_pool = jarvis_rust_core.RustAdvancedMemoryPool()
    rust_pool.configure(
        initial_size_mb=MEMORY_CONFIG['component_budgets']['rust_memory_pool'],
        max_size_mb=jarvis_mb,
        enable_leak_detection=True
    )
    
    return rust_pool
```

#### 8.2 Memory Pressure Handling
```rust
// Adaptive memory management
pub enum MemoryPressure {
    Low,
    Medium,
    High,
    Critical,
}

impl RustAdvancedMemoryPool {
    pub fn get_memory_pressure(&self) -> MemoryPressure {
        let used_percent = self.used_bytes as f64 / self.total_bytes as f64;
        
        match used_percent {
            p if p < 0.5 => MemoryPressure::Low,
            p if p < 0.7 => MemoryPressure::Medium,
            p if p < 0.9 => MemoryPressure::High,
            _ => MemoryPressure::Critical,
        }
    }
    
    pub fn adapt_to_pressure(&mut self) {
        match self.get_memory_pressure() {
            MemoryPressure::Critical => {
                // Emergency cleanup
                self.force_gc();
                self.compact_memory();
            },
            MemoryPressure::High => {
                // Reduce cache sizes
                self.reduce_cache_size(0.5);
            },
            _ => {}
        }
    }
}
```

## Performance Targets

### Baseline (Current Python Implementation)
- Frame processing: ~100ms per frame
- Memory usage: 3-4GB
- CPU usage: 40-60%
- Duplicate detection: ~5ms per check

### Target with Rust Integration
- Frame processing: <20ms per frame (5x improvement)
- Memory usage: 2-3GB (25% reduction)
- CPU usage: 20-30% (50% reduction)
- Duplicate detection: <0.5ms per check (10x improvement)
- GPU utilization: 30-50% (Metal acceleration)

## Implementation Timeline

### Week 1: Foundation
- Build Rust library
- Verify integration
- Implement Rust bloom filters
- Basic performance testing

### Week 2: Core Features
- Sliding window buffer
- Metal acceleration setup
- Initial GPU processing
- Integration testing

### Week 3: Optimization
- Zero-copy pipeline
- SIMD optimizations
- Parallel processing
- Memory pressure handling

### Week 4: Full Integration
- Update all monitoring loops
- Performance benchmarking
- Stability testing
- Documentation

## Testing Strategy

### Unit Tests
```python
# test_rust_integration.py
def test_bloom_filter_performance():
    rust_filter = jarvis_rust_core.RustBloomFilter(10.0, 7)
    python_filter = AdaptiveBloomFilter(10.0, 10000)
    
    # Benchmark insertions
    start = time.time()
    for i in range(100000):
        rust_filter.add(f"item_{i}")
    rust_time = time.time() - start
    
    start = time.time()
    for i in range(100000):
        python_filter.add(f"item_{i}")
    python_time = time.time() - start
    
    assert rust_time < python_time * 0.2  # 5x faster
```

### Integration Tests
```python
def test_zero_copy_pipeline():
    pipeline = ZeroCopyVisionPipeline()
    
    # Create test frames
    frames = [np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8) 
              for _ in range(10)]
    
    # Process batch
    start = time.time()
    results = asyncio.run(pipeline.process_frame_batch(frames))
    elapsed = time.time() - start
    
    assert elapsed < 0.2  # 200ms for 10 frames
    assert len(results) == 10
```

### Performance Benchmarks
```bash
# Run comprehensive benchmarks
python -m pytest tests/test_rust_performance.py -v --benchmark
```

## Monitoring and Metrics

### Performance Dashboard
```python
class RustPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'frame_processing_ms': [],
            'memory_usage_mb': [],
            'gpu_utilization': [],
            'bloom_filter_ops_per_sec': 0,
            'zero_copy_transfers': 0
        }
    
    def log_frame_processing(self, duration_ms: float):
        self.metrics['frame_processing_ms'].append(duration_ms)
        if len(self.metrics['frame_processing_ms']) > 1000:
            self.metrics['frame_processing_ms'].pop(0)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'avg_frame_time': np.mean(self.metrics['frame_processing_ms']),
            'p95_frame_time': np.percentile(self.metrics['frame_processing_ms'], 95),
            'memory_usage': self.get_current_memory_usage(),
            'gpu_utilization': self.get_gpu_utilization()
        }
```

## Deployment Steps

1. **Build Rust Components**
   ```bash
   cd backend/vision/jarvis-rust-core
   cargo build --release --features python-bindings,simd
   maturin develop --release
   ```

2. **Verify Installation**
   ```python
   python -c "import jarvis_rust_core; print(jarvis_rust_core.__version__)"
   ```

3. **Run Performance Tests**
   ```bash
   python backend/vision/test_rust_performance.py
   ```

4. **Update Configuration**
   ```python
   # In settings.json
   {
     "vision": {
       "enable_rust_acceleration": true,
       "rust_memory_pool_mb": 2048,
       "enable_metal_gpu": true,
       "parallel_workers": 8
     }
   }
   ```

5. **Monitor Performance**
   - Check logs for Rust initialization
   - Monitor memory usage with Activity Monitor
   - Verify GPU usage with Metal Performance HUD

## Success Criteria

1. **Performance**
   - 5x faster frame processing
   - 10x faster duplicate detection
   - 50% reduction in CPU usage

2. **Memory**
   - Stay within 6.4GB allocation (40% of 16GB)
   - Zero memory leaks detected
   - Efficient garbage collection

3. **Reliability**
   - No crashes during 24-hour test
   - Graceful fallback to Python if Rust fails
   - Proper error handling and logging

4. **User Experience**
   - Smoother real-time monitoring
   - Faster response to screen changes
   - Reduced system impact

## Next Steps

1. Start with Phase 1: Build and activate Rust core
2. Implement bloom filter acceleration
3. Add Metal GPU processing
4. Integrate zero-copy pipeline
5. Full system testing and optimization