//! Fast image compression for efficient data transfer

use super::{ImageData, ImageFormat};
use crate::{Result, JarvisError};
use lz4;
use zstd;

/// Compression format
#[derive(Debug, Clone, Copy)]
pub enum CompressionFormat {
    None,
    Lz4,
    Zstd(i32), // Compression level
}

/// Image compressor
pub struct ImageCompressor {
    // Reusable compression buffers
    work_buffer: Vec<u8>,
}

impl ImageCompressor {
    pub fn new() -> Self {
        Self {
            work_buffer: Vec::with_capacity(1024 * 1024), // 1MB initial capacity
        }
    }
    
    /// Compress image data
    pub fn compress(&mut self, image: &ImageData, format: CompressionFormat) -> Result<CompressedImage> {
        let compressed_data = match format {
            CompressionFormat::None => image.data.clone(),
            CompressionFormat::Lz4 => self.compress_lz4(&image.data)?,
            CompressionFormat::Zstd(level) => self.compress_zstd(&image.data, level)?,
        };
        
        Ok(CompressedImage {
            width: image.width,
            height: image.height,
            channels: image.channels,
            format: image.format,
            compression: format,
            compressed_data,
            original_size: image.data.len(),
        })
    }
    
    /// Decompress image data
    pub fn decompress(&mut self, compressed: &CompressedImage) -> Result<ImageData> {
        let decompressed_data = match compressed.compression {
            CompressionFormat::None => compressed.compressed_data.clone(),
            CompressionFormat::Lz4 => self.decompress_lz4(
                &compressed.compressed_data, 
                compressed.original_size
            )?,
            CompressionFormat::Zstd(_) => self.decompress_zstd(&compressed.compressed_data)?,
        };
        
        ImageData::from_raw(
            compressed.width,
            compressed.height,
            decompressed_data,
            compressed.format,
        )
    }
    
    /// LZ4 compression (fast)
    fn compress_lz4(&mut self, data: &[u8]) -> Result<Vec<u8>> {
        let max_size = lz4::block::compress_bound(data.len())?;
        self.work_buffer.resize(max_size, 0);
        
        let compressed_size = lz4::block::compress_to_buffer(
            data,
            None,  // No dictionary
            false, // No prepending size
            &mut self.work_buffer,
        )?;
        
        Ok(self.work_buffer[..compressed_size].to_vec())
    }
    
    /// LZ4 decompression
    fn decompress_lz4(&mut self, compressed: &[u8], original_size: usize) -> Result<Vec<u8>> {
        self.work_buffer.resize(original_size, 0);
        
        let decompressed_size = lz4::block::decompress_to_buffer(
            compressed,
            None,  // No prepending size
            &mut self.work_buffer,
        )?;
        
        if decompressed_size != original_size {
            return Err(JarvisError::VisionError(
                "Decompressed size mismatch".to_string()
            ));
        }
        
        Ok(self.work_buffer[..decompressed_size].to_vec())
    }
    
    /// ZSTD compression (higher compression ratio)
    fn compress_zstd(&mut self, data: &[u8], level: i32) -> Result<Vec<u8>> {
        zstd::encode_all(std::io::Cursor::new(data), level)
            .map_err(|e| JarvisError::VisionError(format!("ZSTD compression failed: {}", e)))
    }
    
    /// ZSTD decompression
    fn decompress_zstd(&mut self, compressed: &[u8]) -> Result<Vec<u8>> {
        zstd::decode_all(std::io::Cursor::new(compressed))
            .map_err(|e| JarvisError::VisionError(format!("ZSTD decompression failed: {}", e)))
    }
    
    /// Choose optimal compression based on image characteristics
    pub fn auto_compress(&mut self, image: &ImageData) -> Result<CompressedImage> {
        // Simple heuristic: use LZ4 for real-time, ZSTD for storage
        let format = if image.data.len() > 1024 * 1024 {
            CompressionFormat::Zstd(3) // Balanced compression
        } else {
            CompressionFormat::Lz4 // Fast compression
        };
        
        self.compress(image, format)
    }
}

/// Compressed image data
#[derive(Clone)]
pub struct CompressedImage {
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub format: ImageFormat,
    pub compression: CompressionFormat,
    pub compressed_data: Vec<u8>,
    pub original_size: usize,
}

impl CompressedImage {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        self.original_size as f32 / self.compressed_data.len() as f32
    }
    
    /// Get size reduction percentage
    pub fn size_reduction(&self) -> f32 {
        (1.0 - self.compressed_data.len() as f32 / self.original_size as f32) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lz4_compression() {
        let mut compressor = ImageCompressor::new();
        
        // Create test image with repetitive data (compresses well)
        let mut image = ImageData::new(100, 100, 3, ImageFormat::Rgb8);
        image.data.fill(128); // Uniform gray
        
        let compressed = compressor.compress(&image, CompressionFormat::Lz4).unwrap();
        assert!(compressed.compressed_data.len() < image.data.len());
        
        // Test decompression
        let decompressed = compressor.decompress(&compressed).unwrap();
        assert_eq!(decompressed.data, image.data);
    }
    
    #[test]
    fn test_compression_ratio() {
        let mut compressor = ImageCompressor::new();
        
        // Create test image
        let mut image = ImageData::new(100, 100, 3, ImageFormat::Rgb8);
        for i in 0..image.data.len() {
            image.data[i] = (i % 256) as u8;
        }
        
        let compressed = compressor.auto_compress(&image).unwrap();
        println!("Compression ratio: {:.2}x", compressed.compression_ratio());
        println!("Size reduction: {:.1}%", compressed.size_reduction());
    }
}