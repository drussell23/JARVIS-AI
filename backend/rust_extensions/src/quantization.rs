use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::sync::Arc;

/// Quantization utilities for ML models
pub struct Quantizer;

impl Quantizer {
    /// Quantize a 2D float array to INT8 with symmetric quantization
    pub fn quantize_symmetric_int8(array: &ArrayView2<f32>) -> (Array2<i8>, f32) {
        // Find the maximum absolute value
        let max_abs = array.iter()
            .map(|&x| x.abs())
            .fold(0.0f32, f32::max);
        
        // Calculate scale factor
        let scale = max_abs / 127.0;
        
        // Quantize
        let quantized = array.mapv(|x| {
            let q = (x / scale).round();
            q.clamp(-128.0, 127.0) as i8
        });
        
        (quantized, scale)
    }
    
    /// Quantize with per-channel scales (for weight matrices)
    pub fn quantize_per_channel_int8(array: &ArrayView2<f32>, axis: usize) -> (Array2<i8>, Vec<f32>) {
        let scales: Vec<f32> = if axis == 0 {
            // Per-row quantization
            (0..array.nrows())
                .map(|i| {
                    let row = array.row(i);
                    row.iter().map(|&x| x.abs()).fold(0.0f32, f32::max) / 127.0
                })
                .collect()
        } else {
            // Per-column quantization
            (0..array.ncols())
                .map(|i| {
                    let col = array.column(i);
                    col.iter().map(|&x| x.abs()).fold(0.0f32, f32::max) / 127.0
                })
                .collect()
        };
        
        // Quantize with per-channel scales
        let mut quantized = Array2::<i8>::zeros((array.nrows(), array.ncols()));
        
        for i in 0..array.nrows() {
            for j in 0..array.ncols() {
                let scale = if axis == 0 { scales[i] } else { scales[j] };
                let q = (array[[i, j]] / scale).round();
                quantized[[i, j]] = q.clamp(-128.0, 127.0) as i8;
            }
        }
        
        (quantized, scales)
    }
    
    /// Dynamic quantization for activations
    pub fn dynamic_quantize_int8(values: &[f32]) -> (Vec<i8>, f32, i8) {
        let (min_val, max_val) = values.par_iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            })
            .reduce(
                || (f32::INFINITY, f32::NEG_INFINITY),
                |(min1, max1), (min2, max2)| (min1.min(min2), max1.max(max2))
            );
        
        // Calculate scale and zero point for asymmetric quantization
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as i8;
        
        // Quantize
        let quantized: Vec<i8> = values.par_iter()
            .map(|&val| {
                let q = ((val - min_val) / scale).round() as i16 - 128;
                q.clamp(-128, 127) as i8
            })
            .collect();
        
        (quantized, scale, zero_point)
    }
    
    /// Dequantize INT8 back to float32
    pub fn dequantize_symmetric(quantized: &[i8], scale: f32) -> Vec<f32> {
        quantized.par_iter()
            .map(|&q| q as f32 * scale)
            .collect()
    }
    
    /// Dequantize with zero point
    pub fn dequantize_asymmetric(quantized: &[i8], scale: f32, zero_point: i8) -> Vec<f32> {
        quantized.par_iter()
            .map(|&q| (q as f32 + zero_point as f32) * scale)
            .collect()
    }
    
    /// Pack INT8 values into INT4 (nibbles)
    pub fn pack_int4(values: &[i8]) -> Vec<u8> {
        values.chunks(2)
            .map(|chunk| {
                let low = (chunk[0] & 0x0F) as u8;
                let high = if chunk.len() > 1 {
                    ((chunk[1] & 0x0F) << 4) as u8
                } else {
                    0u8
                };
                low | high
            })
            .collect()
    }
    
    /// Unpack INT4 to INT8
    pub fn unpack_int4(packed: &[u8]) -> Vec<i8> {
        packed.iter()
            .flat_map(|&byte| {
                let low = (byte & 0x0F) as i8;
                let high = ((byte >> 4) & 0x0F) as i8;
                vec![
                    if low > 7 { low - 16 } else { low },
                    if high > 7 { high - 16 } else { high }
                ]
            })
            .collect()
    }
}