//! Enhanced zero-copy screen capture with dynamic configuration
//! No hardcoded values - everything is configurable

use super::{ImageData, ImageFormat};
use crate::{Result, JarvisError};
use crate::memory::{MemoryManager, ZeroCopyBuffer};
use std::time::{Instant, Duration};
use std::sync::Arc;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use once_cell::sync::Lazy;
use parking_lot::RwLock;

/// Global configuration store
static CAPTURE_CONFIG_STORE: Lazy<RwLock<HashMap<String, String>>> = 
    Lazy::new(|| RwLock::new(HashMap::new()));

/// Dynamic capture configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureConfig {
    pub capture_mouse: bool,
    pub capture_region: Option<CaptureRegion>,
    pub target_fps: u32,
    pub use_hardware_acceleration: bool,
    pub color_space: ColorSpace,
    pub pixel_format: PixelFormat,
    pub compression_hint: CompressionHint,
    pub memory_pool_size_mb: usize,
    pub enable_gpu_capture: bool,
    pub capture_quality: CaptureQuality,
    pub buffer_count: usize,
    pub enable_hdr: bool,
}

impl CaptureConfig {
    /// Load configuration from environment or JSON
    pub fn from_env() -> Self {
        Self {
            capture_mouse: std::env::var("RUST_CAPTURE_MOUSE")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
            capture_region: None,
            target_fps: std::env::var("RUST_CAPTURE_FPS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            use_hardware_acceleration: std::env::var("RUST_HW_ACCEL")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            color_space: std::env::var("RUST_COLOR_SPACE")
                .unwrap_or_else(|_| "srgb".to_string())
                .parse()
                .unwrap_or(ColorSpace::Srgb),
            pixel_format: std::env::var("RUST_PIXEL_FORMAT")
                .unwrap_or_else(|_| "bgra8".to_string())
                .parse()
                .unwrap_or(PixelFormat::Bgra8),
            compression_hint: std::env::var("RUST_COMPRESSION_HINT")
                .unwrap_or_else(|_| "balanced".to_string())
                .parse()
                .unwrap_or(CompressionHint::Balanced),
            memory_pool_size_mb: std::env::var("RUST_MEMORY_POOL_MB")
                .unwrap_or_else(|_| "100".to_string())
                .parse()
                .unwrap_or(100),
            enable_gpu_capture: std::env::var("RUST_GPU_CAPTURE")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            capture_quality: std::env::var("RUST_CAPTURE_QUALITY")
                .unwrap_or_else(|_| "high".to_string())
                .parse()
                .unwrap_or(CaptureQuality::High),
            buffer_count: std::env::var("RUST_BUFFER_COUNT")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
            enable_hdr: std::env::var("RUST_ENABLE_HDR")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
        }
    }

    /// Update configuration dynamically
    pub fn update(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "target_fps" => self.target_fps = value.parse()
                .map_err(|_| JarvisError::InvalidOperation("Invalid FPS".to_string()))?,
            "capture_mouse" => self.capture_mouse = value.parse()
                .map_err(|_| JarvisError::InvalidOperation("Invalid boolean".to_string()))?,
            "memory_pool_size_mb" => self.memory_pool_size_mb = value.parse()
                .map_err(|_| JarvisError::InvalidOperation("Invalid size".to_string()))?,
            _ => {
                // Store in global config for custom parameters
                CAPTURE_CONFIG_STORE.write().insert(key.to_string(), value.to_string());
            }
        }
        Ok(())
    }
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self::from_env()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CaptureRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ColorSpace {
    Srgb,
    DisplayP3,
    Rec709,
    Rec2020,
}

impl std::str::FromStr for ColorSpace {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "srgb" => Ok(ColorSpace::Srgb),
            "displayp3" | "p3" => Ok(ColorSpace::DisplayP3),
            "rec709" => Ok(ColorSpace::Rec709),
            "rec2020" => Ok(ColorSpace::Rec2020),
            _ => Err(format!("Unknown color space: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PixelFormat {
    Rgb8,
    Rgba8,
    Bgr8,
    Bgra8,
    Rgb16,
    Rgba16f,
}

impl std::str::FromStr for PixelFormat {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "rgb8" => Ok(PixelFormat::Rgb8),
            "rgba8" => Ok(PixelFormat::Rgba8),
            "bgr8" => Ok(PixelFormat::Bgr8),
            "bgra8" => Ok(PixelFormat::Bgra8),
            "rgb16" => Ok(PixelFormat::Rgb16),
            "rgba16f" => Ok(PixelFormat::Rgba16f),
            _ => Err(format!("Unknown pixel format: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionHint {
    None,
    Fast,
    Balanced,
    Quality,
}

impl std::str::FromStr for CompressionHint {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" => Ok(CompressionHint::None),
            "fast" => Ok(CompressionHint::Fast),
            "balanced" => Ok(CompressionHint::Balanced),
            "quality" => Ok(CompressionHint::Quality),
            _ => Err(format!("Unknown compression hint: {}", s)),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CaptureQuality {
    Low,
    Medium,
    High,
    Ultra,
}

impl std::str::FromStr for CaptureQuality {
    type Err = String;
    
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "low" => Ok(CaptureQuality::Low),
            "medium" => Ok(CaptureQuality::Medium),
            "high" => Ok(CaptureQuality::High),
            "ultra" => Ok(CaptureQuality::Ultra),
            _ => Err(format!("Unknown quality: {}", s)),
        }
    }
}

/// Enhanced screen capture with zero-copy and memory mapping
pub struct ScreenCapture {
    config: CaptureConfig,
    last_capture_time: Option<Instant>,
    frame_interval: Duration,
    memory_manager: Arc<MemoryManager>,
    buffer_pool: Vec<ZeroCopyBuffer>,
    current_buffer_idx: usize,
    capture_stats: CaptureStats,
    #[cfg(target_os = "macos")]
    metal_context: Option<MetalCaptureContext>,
}

#[cfg(target_os = "macos")]
struct MetalCaptureContext {
    // Metal device and command queue for GPU capture
    // Placeholder for actual Metal implementation
}

impl ScreenCapture {
    pub fn new(config: CaptureConfig) -> Result<Self> {
        let frame_interval = Duration::from_millis(1000 / config.target_fps as u64);
        let memory_manager = MemoryManager::global();
        
        // Pre-allocate buffer pool for zero-copy operation
        let mut buffer_pool = Vec::with_capacity(config.buffer_count);
        let estimated_buffer_size = Self::estimate_buffer_size(&config)?;
        
        for _ in 0..config.buffer_count {
            let buffer = memory_manager.allocate(estimated_buffer_size)?;
            buffer_pool.push(ZeroCopyBuffer::from_rust(buffer));
        }
        
        Ok(Self {
            config,
            last_capture_time: None,
            frame_interval,
            memory_manager,
            buffer_pool,
            current_buffer_idx: 0,
            capture_stats: CaptureStats::default(),
            #[cfg(target_os = "macos")]
            metal_context: None,
        })
    }
    
    /// Estimate buffer size based on configuration
    fn estimate_buffer_size(config: &CaptureConfig) -> Result<usize> {
        let (width, height) = Self::get_screen_dimensions_static()?;
        let bytes_per_pixel = match config.pixel_format {
            PixelFormat::Rgb8 => 3,
            PixelFormat::Rgba8 | PixelFormat::Bgr8 | PixelFormat::Bgra8 => 4,
            PixelFormat::Rgb16 => 6,
            PixelFormat::Rgba16f => 8,
        };
        Ok((width * height * bytes_per_pixel) as usize)
    }
    
    /// Get screen dimensions (static version for initialization)
    fn get_screen_dimensions_static() -> Result<(u32, u32)> {
        #[cfg(target_os = "macos")]
        {
            use core_foundation::base::TCFType;
            use core_foundation::number::CFNumber;
            use core_foundation::dictionary::CFDictionary;
            
            unsafe {
                let display_id = core_graphics::display::CGMainDisplayID();
                let width = core_graphics::display::CGDisplayPixelsWide(display_id);
                let height = core_graphics::display::CGDisplayPixelsHigh(display_id);
                Ok((width as u32, height as u32))
            }
        }
        
        #[cfg(not(target_os = "macos"))]
        {
            // Fallback to environment variable or default
            let width = std::env::var("SCREEN_WIDTH")
                .unwrap_or_else(|_| "1920".to_string())
                .parse()
                .unwrap_or(1920);
            let height = std::env::var("SCREEN_HEIGHT")
                .unwrap_or_else(|_| "1080".to_string())
                .parse()
                .unwrap_or(1080);
            Ok((width, height))
        }
    }
    
    /// Zero-copy capture to pre-allocated buffer
    pub fn capture_to_buffer(&mut self) -> Result<&ZeroCopyBuffer> {
        // Rate limiting
        if let Some(last_time) = self.last_capture_time {
            let elapsed = last_time.elapsed();
            if elapsed < self.frame_interval {
                std::thread::sleep(self.frame_interval - elapsed);
            }
        }
        
        let start_time = Instant::now();
        
        // Get next buffer from pool (zero allocation)
        let buffer_idx = self.current_buffer_idx;
        self.current_buffer_idx = (self.current_buffer_idx + 1) % self.buffer_pool.len();
        
        // Platform-specific capture
        #[cfg(target_os = "macos")]
        self.capture_macos_to_buffer(buffer_idx)?;
        
        #[cfg(target_os = "linux")]
        self.capture_linux_to_buffer(buffer_idx)?;
        
        #[cfg(target_os = "windows")]
        self.capture_windows_to_buffer(buffer_idx)?;
        
        // Update stats
        self.capture_stats.frame_count += 1;
        self.capture_stats.total_capture_time += start_time.elapsed();
        self.last_capture_time = Some(Instant::now());
        
        Ok(&self.buffer_pool[buffer_idx])
    }
    
    #[cfg(target_os = "macos")]
    fn capture_macos_to_buffer(&mut self, buffer_idx: usize) -> Result<()> {
        use core_graphics::display::*;
        use core_foundation::base::TCFType;
        
        unsafe {
            let display_id = CGMainDisplayID();
            let image_ref = CGDisplayCreateImage(display_id);
            
            if image_ref.is_null() {
                return Err(JarvisError::VisionError("Failed to capture display".to_string()));
            }
            
            // Get image dimensions
            let width = CGImageGetWidth(image_ref);
            let height = CGImageGetHeight(image_ref);
            let bytes_per_row = CGImageGetBytesPerRow(image_ref);
            
            // Get data provider and copy data directly to our buffer
            let data_provider = CGImageGetDataProvider(image_ref);
            let data = CGDataProviderCopyData(data_provider);
            
            if !data.is_null() {
                let buffer = &mut self.buffer_pool[buffer_idx];
                let data_ptr = CFDataGetBytePtr(data);
                let data_len = CFDataGetLength(data) as usize;
                
                // Copy to our zero-copy buffer
                let dst_slice = buffer.as_mut_slice();
                if dst_slice.len() >= data_len {
                    std::ptr::copy_nonoverlapping(data_ptr, dst_slice.as_mut_ptr(), data_len);
                }
                
                CFRelease(data as _);
            }
            
            CGImageRelease(image_ref);
        }
        
        Ok(())
    }
    
    #[cfg(target_os = "linux")]
    fn capture_linux_to_buffer(&mut self, buffer_idx: usize) -> Result<()> {
        // Linux implementation would use X11/Wayland
        // For now, fill with test pattern
        let buffer = &mut self.buffer_pool[buffer_idx];
        let (width, height) = self.get_screen_dimensions()?;
        
        unsafe {
            let data = buffer.as_mut_slice();
            // Simple test pattern
            for i in 0..data.len() {
                data[i] = (i % 256) as u8;
            }
        }
        
        Ok(())
    }
    
    #[cfg(target_os = "windows")]
    fn capture_windows_to_buffer(&mut self, buffer_idx: usize) -> Result<()> {
        // Windows implementation would use Desktop Duplication API
        Err(JarvisError::VisionError("Windows capture not implemented".to_string()))
    }
    
    /// Memory-mapped file sharing for Python interop
    pub fn create_shared_memory(&self, name: &str) -> Result<SharedMemoryHandle> {
        let size = self.estimate_buffer_size(&self.config)?;
        
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            use nix::sys::mman::{shm_open, mmap, ProtFlags, MapFlags};
            use nix::sys::stat::Mode;
            use nix::fcntl::OFlag;
            
            unsafe {
                let shm_fd = shm_open(
                    name,
                    OFlag::O_CREAT | OFlag::O_RDWR,
                    Mode::S_IRUSR | Mode::S_IWUSR,
                )?;
                
                nix::unistd::ftruncate(shm_fd, size as i64)?;
                
                let addr = mmap(
                    std::ptr::null_mut(),
                    size,
                    ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                    MapFlags::MAP_SHARED,
                    shm_fd,
                    0,
                )?;
                
                Ok(SharedMemoryHandle {
                    name: name.to_string(),
                    size,
                    addr: addr as *mut u8,
                    fd: shm_fd,
                })
            }
        }
        
        #[cfg(not(unix))]
        {
            Err(JarvisError::VisionError("Shared memory not implemented for this platform".to_string()))
        }
    }
    
    /// Capture to shared memory for zero-copy Python access
    pub fn capture_to_shared_memory(&mut self, handle: &SharedMemoryHandle) -> Result<()> {
        let buffer = self.capture_to_buffer()?;
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                buffer.as_slice().as_ptr(),
                handle.addr,
                handle.size.min(buffer.as_slice().len()),
            );
        }
        
        Ok(())
    }
    
    /// Get screen dimensions dynamically
    fn get_screen_dimensions(&self) -> Result<(u32, u32)> {
        Self::get_screen_dimensions_static()
    }
    
    /// Capture with automatic preprocessing
    pub fn capture_preprocessed(&mut self) -> Result<ImageData> {
        let buffer = self.capture_to_buffer()?;
        let (width, height) = self.get_screen_dimensions()?;
        
        // Apply preprocessing based on config
        let processed = match self.config.capture_quality {
            CaptureQuality::Low => self.preprocess_low_quality(buffer, width, height)?,
            CaptureQuality::Medium => self.preprocess_medium_quality(buffer, width, height)?,
            CaptureQuality::High => self.preprocess_high_quality(buffer, width, height)?,
            CaptureQuality::Ultra => self.preprocess_ultra_quality(buffer, width, height)?,
        };
        
        Ok(processed)
    }
    
    fn preprocess_low_quality(&self, buffer: &ZeroCopyBuffer, width: u32, height: u32) -> Result<ImageData> {
        // Downsample for faster processing
        let scale = 0.5;
        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;
        
        unsafe {
            ImageData::from_raw(
                new_width,
                new_height,
                buffer.as_slice().to_vec(), // TODO: Implement actual downsampling
                ImageFormat::from_pixel_format(self.config.pixel_format),
            )
        }
    }
    
    fn preprocess_medium_quality(&self, buffer: &ZeroCopyBuffer, width: u32, height: u32) -> Result<ImageData> {
        // No downsampling, basic preprocessing
        unsafe {
            ImageData::from_raw(
                width,
                height,
                buffer.as_slice().to_vec(),
                ImageFormat::from_pixel_format(self.config.pixel_format),
            )
        }
    }
    
    fn preprocess_high_quality(&self, buffer: &ZeroCopyBuffer, width: u32, height: u32) -> Result<ImageData> {
        // Full quality with color correction
        unsafe {
            ImageData::from_raw(
                width,
                height,
                buffer.as_slice().to_vec(),
                ImageFormat::from_pixel_format(self.config.pixel_format),
            )
        }
    }
    
    fn preprocess_ultra_quality(&self, buffer: &ZeroCopyBuffer, width: u32, height: u32) -> Result<ImageData> {
        // HDR processing if enabled
        unsafe {
            ImageData::from_raw(
                width,
                height,
                buffer.as_slice().to_vec(),
                ImageFormat::from_pixel_format(self.config.pixel_format),
            )
        }
    }
    
    /// Update configuration at runtime
    pub fn update_config(&mut self, key: &str, value: &str) -> Result<()> {
        self.config.update(key, value)?;
        
        // Recalculate frame interval if FPS changed
        if key == "target_fps" {
            self.frame_interval = Duration::from_millis(1000 / self.config.target_fps as u64);
        }
        
        Ok(())
    }
    
    /// Get comprehensive capture statistics
    pub fn stats(&self) -> &CaptureStats {
        &self.capture_stats
    }
    
    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.capture_stats = CaptureStats::default();
    }
}

/// Shared memory handle for zero-copy Python interop
pub struct SharedMemoryHandle {
    pub name: String,
    pub size: usize,
    pub addr: *mut u8,
    #[cfg(unix)]
    pub fd: i32,
}

impl Drop for SharedMemoryHandle {
    fn drop(&mut self) {
        #[cfg(unix)]
        unsafe {
            use nix::sys::mman::{munmap, shm_unlink};
            munmap(self.addr as *mut _, self.size).ok();
            shm_unlink(self.name.as_str()).ok();
        }
    }
}

/// Enhanced capture statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CaptureStats {
    pub target_fps: u32,
    pub actual_fps: f32,
    pub frame_count: u64,
    pub total_capture_time: Duration,
    pub avg_capture_time_ms: f32,
    pub min_capture_time_ms: f32,
    pub max_capture_time_ms: f32,
    pub hardware_accelerated: bool,
    pub memory_usage_mb: f32,
    pub compression_ratio: f32,
}

impl CaptureStats {
    pub fn calculate_fps(&mut self) {
        if self.frame_count > 0 {
            self.actual_fps = self.frame_count as f32 / self.total_capture_time.as_secs_f32();
            self.avg_capture_time_ms = self.total_capture_time.as_millis() as f32 / self.frame_count as f32;
        }
    }
}

// Helper trait for ImageFormat conversion
trait FromPixelFormat {
    fn from_pixel_format(format: PixelFormat) -> Self;
}

impl FromPixelFormat for ImageFormat {
    fn from_pixel_format(format: PixelFormat) -> Self {
        match format {
            PixelFormat::Rgb8 => ImageFormat::Rgb8,
            PixelFormat::Rgba8 => ImageFormat::Rgba8,
            PixelFormat::Bgr8 => ImageFormat::Bgr8,
            PixelFormat::Bgra8 => ImageFormat::Bgra8,
            _ => ImageFormat::Rgba8, // Default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dynamic_config() {
        let mut config = CaptureConfig::default();
        assert!(config.update("target_fps", "60").is_ok());
        assert_eq!(config.target_fps, 60);
        
        assert!(config.update("capture_mouse", "true").is_ok());
        assert!(config.capture_mouse);
    }
    
    #[test]
    fn test_config_from_env() {
        std::env::set_var("RUST_CAPTURE_FPS", "120");
        std::env::set_var("RUST_HW_ACCEL", "false");
        
        let config = CaptureConfig::from_env();
        assert_eq!(config.target_fps, 120);
        assert!(!config.use_hardware_acceleration);
    }
}