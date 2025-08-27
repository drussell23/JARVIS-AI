//! Image processing operations with SIMD acceleration

use super::{ImageData, ImageFormat};
use crate::{Result, JarvisError};
use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

/// Image processor with hardware acceleration
pub struct ImageProcessor {
    use_simd: bool,
    thread_count: usize,
}

impl ImageProcessor {
    pub fn new() -> Self {
        Self {
            use_simd: cfg!(any(target_arch = "x86_64", target_arch = "aarch64")),
            thread_count: num_cpus::get(),
        }
    }
    
    /// Resize image using fast bilinear interpolation
    pub fn resize(&self, image: &ImageData, new_width: u32, new_height: u32) -> Result<ImageData> {
        let mut output = ImageData::new(new_width, new_height, image.channels, image.format);
        
        let x_ratio = image.width as f32 / new_width as f32;
        let y_ratio = image.height as f32 / new_height as f32;
        
        // Parallel processing
        output.data.par_chunks_mut((new_width * image.channels as u32) as usize)
            .enumerate()
            .for_each(|(y, row)| {
                let src_y = (y as f32 * y_ratio) as u32;
                
                for x in 0..new_width {
                    let src_x = (x as f32 * x_ratio) as u32;
                    let src_offset = ((src_y * image.width + src_x) * image.channels as u32) as usize;
                    let dst_offset = (x * image.channels as u32) as usize;
                    
                    // Simple nearest neighbor for now
                    row[dst_offset..dst_offset + image.channels as usize]
                        .copy_from_slice(&image.data[src_offset..src_offset + image.channels as usize]);
                }
            });
        
        Ok(output)
    }
    
    /// Convert between color formats
    pub fn convert_format(&self, image: &ImageData, target_format: ImageFormat) -> Result<ImageData> {
        if image.format == target_format {
            return Ok(image.clone());
        }
        
        match (image.format, target_format) {
            (ImageFormat::Rgb8, ImageFormat::Bgr8) => self.rgb_to_bgr(image),
            (ImageFormat::Bgr8, ImageFormat::Rgb8) => self.rgb_to_bgr(image), // Same operation
            (ImageFormat::Rgba8, ImageFormat::Rgb8) => self.rgba_to_rgb(image),
            (ImageFormat::Rgb8, ImageFormat::Gray8) => self.rgb_to_gray(image),
            _ => Err(JarvisError::VisionError(
                format!("Conversion from {:?} to {:?} not implemented", image.format, target_format)
            ))
        }
    }
    
    /// RGB to BGR conversion (also handles BGR to RGB)
    fn rgb_to_bgr(&self, image: &ImageData) -> Result<ImageData> {
        let mut output = image.clone();
        
        if self.use_simd {
            self.rgb_to_bgr_simd(&mut output.data)?;
        } else {
            output.data.par_chunks_mut(3).for_each(|pixel| {
                pixel.swap(0, 2);
            });
        }
        
        output.format = if image.format == ImageFormat::Rgb8 { 
            ImageFormat::Bgr8 
        } else { 
            ImageFormat::Rgb8 
        };
        
        Ok(output)
    }
    
    /// RGBA to RGB conversion
    fn rgba_to_rgb(&self, image: &ImageData) -> Result<ImageData> {
        let mut output = ImageData::new(image.width, image.height, 3, ImageFormat::Rgb8);
        
        output.data.par_chunks_mut(3)
            .zip(image.data.par_chunks(4))
            .for_each(|(dst, src)| {
                dst.copy_from_slice(&src[..3]);
            });
        
        Ok(output)
    }
    
    /// RGB to grayscale conversion
    fn rgb_to_gray(&self, image: &ImageData) -> Result<ImageData> {
        let mut output = ImageData::new(image.width, image.height, 1, ImageFormat::Gray8);
        
        output.data.par_iter_mut()
            .zip(image.data.par_chunks(3))
            .for_each(|(gray, rgb)| {
                // Use standard luminance weights
                *gray = (0.299 * rgb[0] as f32 + 
                        0.587 * rgb[1] as f32 + 
                        0.114 * rgb[2] as f32) as u8;
            });
        
        Ok(output)
    }
    
    /// SIMD-accelerated RGB to BGR
    #[cfg(target_arch = "aarch64")]
    fn rgb_to_bgr_simd(&self, data: &mut [u8]) -> Result<()> {
        use std::arch::aarch64::*;
        
        unsafe {
            let chunks = data.chunks_exact_mut(48); // Process 16 pixels at a time
            
            for chunk in chunks {
                // Load 48 bytes (16 RGB pixels)
                let v0 = vld3q_u8(chunk.as_ptr());
                
                // Swap R and B channels
                let swapped = uint8x16x3_t {
                    0: v0.2,  // B becomes R
                    1: v0.1,  // G stays G
                    2: v0.0,  // R becomes B
                };
                
                // Store back
                vst3q_u8(chunk.as_mut_ptr(), swapped);
            }
            
            // Handle remaining pixels
            let remainder = data.len() % 48;
            if remainder > 0 {
                let offset = data.len() - remainder;
                for i in (offset..data.len()).step_by(3) {
                    data.swap(i, i + 2);
                }
            }
        }
        
        Ok(())
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    fn rgb_to_bgr_simd(&self, _data: &mut [u8]) -> Result<()> {
        Err(JarvisError::InvalidOperation("SIMD not available".to_string()))
    }
    
    /// Apply convolution filter
    pub fn convolve(&self, image: &ImageData, kernel: &[f32]) -> Result<ImageData> {
        let kernel_size = (kernel.len() as f32).sqrt() as usize;
        if kernel_size * kernel_size != kernel.len() {
            return Err(JarvisError::VisionError("Invalid kernel size".to_string()));
        }
        
        let mut output = image.clone();
        let half_kernel = kernel_size / 2;
        
        // Process each channel separately
        for c in 0..image.channels {
            // Extract channel data
            let channel_data: Vec<f32> = image.data.iter()
                .skip(c as usize)
                .step_by(image.channels as usize)
                .map(|&v| v as f32)
                .collect();
            
            let channel_2d = Array2::from_shape_vec(
                (image.height as usize, image.width as usize), 
                channel_data
            )?;
            
            // Apply convolution
            let mut result = Array2::zeros((image.height as usize, image.width as usize));
            
            for y in half_kernel..(image.height as usize - half_kernel) {
                for x in half_kernel..(image.width as usize - half_kernel) {
                    let mut sum = 0.0;
                    
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let pixel_y = y + ky - half_kernel;
                            let pixel_x = x + kx - half_kernel;
                            sum += channel_2d[[pixel_y, pixel_x]] * kernel[ky * kernel_size + kx];
                        }
                    }
                    
                    result[[y, x]] = sum.max(0.0).min(255.0);
                }
            }
            
            // Write back to output
            for (i, &val) in result.iter().enumerate() {
                output.data[i * image.channels as usize + c as usize] = val as u8;
            }
        }
        
        Ok(output)
    }
}

/// Processing pipeline for chaining operations
pub struct ProcessingPipeline {
    operations: Vec<Box<dyn ProcessingOp>>,
}

trait ProcessingOp: Send + Sync {
    fn apply(&self, image: &ImageData) -> Result<ImageData>;
}

impl ProcessingPipeline {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
        }
    }
    
    /// Execute pipeline on image
    pub fn execute(&self, mut image: ImageData) -> Result<ImageData> {
        for op in &self.operations {
            image = op.apply(&image)?;
        }
        Ok(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_format_conversion() {
        let processor = ImageProcessor::new();
        
        // Create test RGB image
        let mut image = ImageData::new(2, 2, 3, ImageFormat::Rgb8);
        image.data = vec![
            255, 0, 0,    // Red
            0, 255, 0,    // Green
            0, 0, 255,    // Blue
            255, 255, 0,  // Yellow
        ];
        
        // Convert to grayscale
        let gray = processor.convert_format(&image, ImageFormat::Gray8).unwrap();
        assert_eq!(gray.channels, 1);
        assert_eq!(gray.data.len(), 4);
    }
}