//! Quantization-aware training utilities

use super::{QuantizedTensor, QuantizationType};
use crate::Result;
use ndarray::{Array2, ArrayView2};

/// Fake quantization for training
pub struct FakeQuantizer {
    num_bits: u8,
    symmetric: bool,
}

impl FakeQuantizer {
    pub fn new(num_bits: u8, symmetric: bool) -> Self {
        Self { num_bits, symmetric }
    }
    
    /// Apply fake quantization (straight-through estimator)
    pub fn fake_quantize(&self, tensor: &Array2<f32>) -> Array2<f32> {
        let (scale, zero_point) = self.compute_params(&tensor.view());
        
        // Quantize
        let quantized = tensor.mapv(|val| {
            let q = if self.symmetric {
                (val / scale).round()
            } else {
                ((val / scale) + zero_point as f32).round()
            };
            
            // Clamp to valid range
            let max_val = (1 << (self.num_bits - 1)) - 1;
            let min_val = if self.symmetric { -max_val - 1 } else { 0 };
            q.max(min_val as f32).min(max_val as f32)
        });
        
        // Dequantize
        quantized.mapv(|q| {
            if self.symmetric {
                q * scale
            } else {
                (q - zero_point as f32) * scale
            }
        })
    }
    
    fn compute_params(&self, tensor: &ArrayView2<f32>) -> (f32, i32) {
        let min_val = tensor.fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = tensor.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if self.symmetric {
            let scale = max_val.max(-min_val) / ((1 << (self.num_bits - 1)) - 1) as f32;
            (scale, 0)
        } else {
            let scale = (max_val - min_val) / ((1 << self.num_bits) - 1) as f32;
            let zero_point = ((-min_val) / scale).round() as i32;
            (scale, zero_point)
        }
    }
}