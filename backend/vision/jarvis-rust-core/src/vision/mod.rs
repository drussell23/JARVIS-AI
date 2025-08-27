//! Vision processing with hardware acceleration

pub mod capture;
pub mod compression;
pub mod processing;

use crate::{Result, JarvisError};
use image::{DynamicImage, ImageBuffer, Rgb};
use std::sync::Arc;

pub use capture::{ScreenCapture, CaptureConfig};
pub use processing::{ImageProcessor, ProcessingPipeline};
pub use compression::{ImageCompressor, CompressionFormat};

/// Image data container
#[derive(Clone)]
pub struct ImageData {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub data: Vec<u8>,
    pub format: ImageFormat,
}

/// Supported image formats
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    Rgb8,
    Rgba8,
    Gray8,
    Bgr8,
    Bgra8,
}

impl ImageData {
    /// Create new image data
    pub fn new(width: u32, height: u32, channels: u8, format: ImageFormat) -> Self {
        let data_size = (width * height * channels as u32) as usize;
        Self {
            width,
            height,
            channels,
            data: vec![0; data_size],
            format,
        }
    }
    
    /// Create from raw data
    pub fn from_raw(width: u32, height: u32, data: Vec<u8>, format: ImageFormat) -> Result<Self> {
        let expected_size = width * height * format.channels() as u32;
        if data.len() != expected_size as usize {
            return Err(JarvisError::VisionError(
                format!("Invalid data size: expected {}, got {}", expected_size, data.len())
            ));
        }
        
        Ok(Self {
            width,
            height,
            channels: format.channels(),
            data,
            format,
        })
    }
    
    /// Convert to DynamicImage
    pub fn to_dynamic_image(&self) -> Result<DynamicImage> {
        match self.format {
            ImageFormat::Rgb8 => {
                let img = ImageBuffer::<Rgb<u8>, _>::from_raw(
                    self.width, self.height, self.data.clone()
                ).ok_or_else(|| JarvisError::VisionError("Failed to create image".to_string()))?;
                Ok(DynamicImage::ImageRgb8(img))
            }
            _ => Err(JarvisError::VisionError(
                format!("Unsupported format for conversion: {:?}", self.format)
            ))
        }
    }
    
    /// Get pixel at coordinates
    pub fn get_pixel(&self, x: u32, y: u32) -> Result<&[u8]> {
        if x >= self.width || y >= self.height {
            return Err(JarvisError::VisionError("Pixel coordinates out of bounds".to_string()));
        }
        
        let offset = ((y * self.width + x) * self.channels as u32) as usize;
        Ok(&self.data[offset..offset + self.channels as usize])
    }
    
    /// Set pixel at coordinates
    pub fn set_pixel(&mut self, x: u32, y: u32, pixel: &[u8]) -> Result<()> {
        if x >= self.width || y >= self.height {
            return Err(JarvisError::VisionError("Pixel coordinates out of bounds".to_string()));
        }
        
        if pixel.len() != self.channels as usize {
            return Err(JarvisError::VisionError("Invalid pixel data size".to_string()));
        }
        
        let offset = ((y * self.width + x) * self.channels as u32) as usize;
        self.data[offset..offset + self.channels as usize].copy_from_slice(pixel);
        Ok(())
    }
}

impl ImageFormat {
    pub fn channels(&self) -> u8 {
        match self {
            ImageFormat::Gray8 => 1,
            ImageFormat::Rgb8 | ImageFormat::Bgr8 => 3,
            ImageFormat::Rgba8 | ImageFormat::Bgra8 => 4,
        }
    }
}

/// Vision processing context
pub struct VisionContext {
    pub capture: Arc<ScreenCapture>,
    pub processor: Arc<ImageProcessor>,
    pub compressor: Arc<ImageCompressor>,
}

impl VisionContext {
    pub fn new() -> Result<Self> {
        Ok(Self {
            capture: Arc::new(ScreenCapture::new(CaptureConfig::default())?),
            processor: Arc::new(ImageProcessor::new()),
            compressor: Arc::new(ImageCompressor::new()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_image_data() {
        let mut img = ImageData::new(100, 100, 3, ImageFormat::Rgb8);
        assert_eq!(img.data.len(), 30000);
        
        // Test pixel operations
        img.set_pixel(50, 50, &[255, 128, 0]).unwrap();
        let pixel = img.get_pixel(50, 50).unwrap();
        assert_eq!(pixel, &[255, 128, 0]);
    }
}