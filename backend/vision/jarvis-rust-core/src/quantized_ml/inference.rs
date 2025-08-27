//! High-performance quantized inference engine

use crate::{Result, JarvisError, Metrics};
use super::{QuantizedTensor, QuantizationType, Dequantize};
use ndarray::{Array2, Array4, Axis};
use std::time::Instant;
use parking_lot::RwLock;
use std::sync::Arc;

/// Inference result
#[derive(Debug, Clone)]
pub struct InferenceResult {
    pub outputs: Vec<f32>,
    pub shape: Vec<usize>,
    pub metrics: Metrics,
}

/// Quantized layer types
#[derive(Debug, Clone)]
pub enum QuantizedLayer {
    Linear {
        weights: QuantizedTensor<i8>,
        bias: Option<Vec<f32>>,
    },
    Conv2d {
        weights: QuantizedTensor<i8>,
        bias: Option<Vec<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    },
}

/// Quantized inference engine
pub struct QuantizedInferenceEngine {
    layers: Vec<QuantizedLayer>,
    use_simd: bool,
    thread_count: usize,
    metrics: Arc<RwLock<Metrics>>,
}

impl QuantizedInferenceEngine {
    pub fn new(use_simd: bool, thread_count: usize) -> Self {
        Self {
            layers: Vec::new(),
            use_simd,
            thread_count,
            metrics: Arc::new(RwLock::new(Metrics::default())),
        }
    }
    
    /// Add a quantized layer
    pub fn add_layer(&mut self, layer: QuantizedLayer) {
        self.layers.push(layer);
    }
    
    /// Run inference on quantized input
    pub fn infer(&self, input: &QuantizedTensor<i8>) -> Result<InferenceResult> {
        let start_time = Instant::now();
        let mut current = input.dequantize();
        let mut current_shape = input.shape.clone();
        
        // Process through layers
        for (idx, layer) in self.layers.iter().enumerate() {
            match layer {
                QuantizedLayer::Linear { weights, bias } => {
                    current = self.linear_forward(&current, &current_shape, weights, bias)?;
                    // Update shape for linear layer
                    let batch_size = current_shape[0];
                    let output_features = weights.shape[0];
                    current_shape = vec![batch_size, output_features];
                }
                QuantizedLayer::Conv2d { weights, bias, stride, padding } => {
                    current = self.conv2d_forward(
                        &current, &current_shape, weights, bias, *stride, *padding
                    )?;
                    // Update shape for conv2d
                    current_shape = self.compute_conv_output_shape(
                        &current_shape, &weights.shape, *stride, *padding
                    );
                }
            }
            
            log::debug!("Layer {} output shape: {:?}", idx, current_shape);
        }
        
        // Update metrics
        let elapsed = start_time.elapsed();
        let mut metrics = self.metrics.write();
        metrics.inference_time_ms = elapsed.as_secs_f64() * 1000.0;
        metrics.operations_per_second = 1.0 / elapsed.as_secs_f64();
        
        Ok(InferenceResult {
            outputs: current,
            shape: current_shape,
            metrics: metrics.clone(),
        })
    }
    
    /// Linear layer forward pass
    fn linear_forward(
        &self,
        input: &[f32],
        input_shape: &[usize],
        weights: &QuantizedTensor<i8>,
        bias: &Option<Vec<f32>>,
    ) -> Result<Vec<f32>> {
        let batch_size = input_shape[0];
        let input_features = input_shape[1];
        let output_features = weights.shape[0];
        
        if input_features != weights.shape[1] {
            return Err(JarvisError::InferenceError(
                format!("Dimension mismatch: input {} vs weights {}", 
                       input_features, weights.shape[1])
            ));
        }
        
        let mut output = vec![0.0f32; batch_size * output_features];
        
        // Dequantize weights once
        let w_deq = weights.dequantize();
        
        // Matrix multiplication with optional SIMD
        if self.use_simd && cfg!(target_arch = "aarch64") {
            self.gemm_simd(&input, &w_deq, &mut output, 
                          batch_size, output_features, input_features)?;
        } else {
            // Standard matrix multiplication
            for b in 0..batch_size {
                for o in 0..output_features {
                    let mut sum = 0.0f32;
                    for i in 0..input_features {
                        sum += input[b * input_features + i] 
                             * w_deq[o * input_features + i];
                    }
                    output[b * output_features + o] = sum;
                }
            }
        }
        
        // Add bias if present
        if let Some(b) = bias {
            for batch in 0..batch_size {
                for o in 0..output_features {
                    output[batch * output_features + o] += b[o];
                }
            }
        }
        
        Ok(output)
    }
    
    /// Conv2D forward pass
    fn conv2d_forward(
        &self,
        input: &[f32],
        input_shape: &[usize],  // [batch, channels, height, width]
        weights: &QuantizedTensor<i8>,  // [out_channels, in_channels, kernel_h, kernel_w]
        bias: &Option<Vec<f32>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Vec<f32>> {
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        
        let out_channels = weights.shape[0];
        let kernel_height = weights.shape[2];
        let kernel_width = weights.shape[3];
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
        
        let mut output = vec![0.0f32; batch_size * out_channels * out_height * out_width];
        
        // Dequantize weights
        let w_deq = weights.dequantize();
        
        // Convolution operation
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let mut sum = 0.0f32;
                        
                        // Apply kernel
                        for ic in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let ih = oh * stride.0 + kh;
                                    let iw = ow * stride.1 + kw;
                                    
                                    if ih >= padding.0 && ih < in_height + padding.0 &&
                                       iw >= padding.1 && iw < in_width + padding.1 {
                                        let ih_actual = ih - padding.0;
                                        let iw_actual = iw - padding.1;
                                        
                                        let input_idx = b * in_channels * in_height * in_width +
                                                       ic * in_height * in_width +
                                                       ih_actual * in_width + iw_actual;
                                        
                                        let weight_idx = oc * in_channels * kernel_height * kernel_width +
                                                        ic * kernel_height * kernel_width +
                                                        kh * kernel_width + kw;
                                        
                                        sum += input[input_idx] * w_deq[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        // Add bias if present
                        if let Some(b) = bias {
                            sum += b[oc];
                        }
                        
                        let output_idx = b * out_channels * out_height * out_width +
                                        oc * out_height * out_width +
                                        oh * out_width + ow;
                        output[output_idx] = sum;
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Compute convolution output shape
    fn compute_conv_output_shape(
        &self,
        input_shape: &[usize],
        kernel_shape: &[usize],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Vec<usize> {
        let batch_size = input_shape[0];
        let out_channels = kernel_shape[0];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        let kernel_height = kernel_shape[2];
        let kernel_width = kernel_shape[3];
        
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
        
        vec![batch_size, out_channels, out_height, out_width]
    }
    
    /// SIMD-accelerated matrix multiplication
    #[cfg(target_arch = "aarch64")]
    fn gemm_simd(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        use std::arch::aarch64::*;
        
        unsafe {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = vdupq_n_f32(0.0);
                    
                    // Process 4 elements at a time
                    for l in (0..k).step_by(4) {
                        if l + 4 <= k {
                            let a_vec = vld1q_f32(&a[i * k + l] as *const f32);
                            let b_vec = vld1q_f32(&b[l * n + j] as *const f32);
                            sum = vmlaq_f32(sum, a_vec, b_vec);
                        }
                    }
                    
                    // Sum all lanes and handle remaining elements
                    let mut total = vaddvq_f32(sum);
                    for l in (k / 4 * 4)..k {
                        total += a[i * k + l] * b[l * n + j];
                    }
                    
                    c[i * n + j] = total;
                }
            }
        }
        
        Ok(())
    }
    
    #[cfg(not(target_arch = "aarch64"))]
    fn gemm_simd(
        &self,
        _a: &[f32],
        _b: &[f32],
        _c: &mut [f32],
        _m: usize,
        _n: usize,
        _k: usize,
    ) -> Result<()> {
        Err(JarvisError::InvalidOperation(
            "SIMD operations not available on this platform".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_linear_inference() {
        let mut engine = QuantizedInferenceEngine::new(false, 1);
        
        // Create test weights
        let weights = QuantizedTensor::new(
            vec![1i8, 2, 3, 4],
            vec![2, 2],
            0.1,
            0,
            QuantizationType::Int8,
        );
        
        engine.add_layer(QuantizedLayer::Linear {
            weights,
            bias: Some(vec![0.1, 0.2]),
        });
        
        // Create test input
        let input = QuantizedTensor::new(
            vec![10i8, 20],
            vec![1, 2],
            0.1,
            0,
            QuantizationType::Int8,
        );
        
        let result = engine.infer(&input).unwrap();
        assert_eq!(result.shape, vec![1, 2]);
    }
}