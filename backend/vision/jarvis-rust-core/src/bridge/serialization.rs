//! Data serialization for Python-Rust communication

use serde::{Serialize, Deserialize};
use crate::{Result, JarvisError};

/// Request types from Python
#[derive(Debug, Serialize, Deserialize)]
pub enum PythonRequest {
    ProcessImage {
        image_data: Vec<u8>,
        width: u32,
        height: u32,
        operation: ImageOperation,
    },
    RunInference {
        model_id: String,
        input_data: Vec<f32>,
        input_shape: Vec<usize>,
    },
    AllocateMemory {
        size: usize,
        alignment: Option<usize>,
    },
}

/// Image operations
#[derive(Debug, Serialize, Deserialize)]
pub enum ImageOperation {
    Resize { width: u32, height: u32 },
    ConvertFormat { target: String },
    Compress { format: String },
    Convolve { kernel: Vec<f32> },
}

/// Response types to Python
#[derive(Debug, Serialize, Deserialize)]
pub enum RustResponse {
    ImageProcessed {
        data: Vec<u8>,
        width: u32,
        height: u32,
        format: String,
    },
    InferenceResult {
        outputs: Vec<f32>,
        shape: Vec<usize>,
        inference_time_ms: f64,
    },
    MemoryAllocated {
        buffer_id: u64,
        size: usize,
        address: usize,
    },
    Error {
        message: String,
    },
}

/// Async communication protocol
#[derive(Debug)]
pub struct AsyncProtocol {
    pending_requests: std::collections::HashMap<u64, PythonRequest>,
    next_id: u64,
}

impl AsyncProtocol {
    pub fn new() -> Self {
        Self {
            pending_requests: std::collections::HashMap::new(),
            next_id: 1,
        }
    }
    
    /// Submit request
    pub fn submit_request(&mut self, request: PythonRequest) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.pending_requests.insert(id, request);
        id
    }
    
    /// Get pending request
    pub fn get_request(&mut self, id: u64) -> Option<PythonRequest> {
        self.pending_requests.remove(&id)
    }
    
    /// Process request and return response
    pub async fn process_request(&mut self, id: u64) -> Result<RustResponse> {
        let request = self.get_request(id)
            .ok_or_else(|| JarvisError::InvalidOperation("Request not found".to_string()))?;
        
        match request {
            PythonRequest::ProcessImage { image_data, width, height, operation } => {
                // Process image based on operation
                // This is a placeholder - actual implementation would use vision module
                Ok(RustResponse::ImageProcessed {
                    data: image_data,
                    width,
                    height,
                    format: "rgb8".to_string(),
                })
            }
            PythonRequest::RunInference { model_id, input_data, input_shape } => {
                // Run inference
                // This is a placeholder - actual implementation would use ML module
                Ok(RustResponse::InferenceResult {
                    outputs: vec![0.0; 10], // Dummy output
                    shape: vec![1, 10],
                    inference_time_ms: 5.0,
                })
            }
            PythonRequest::AllocateMemory { size, alignment } => {
                // Allocate memory
                // This is a placeholder - actual implementation would use memory module
                Ok(RustResponse::MemoryAllocated {
                    buffer_id: id,
                    size,
                    address: 0x1000000, // Dummy address
                })
            }
        }
    }
}

/// Shared memory metadata for zero-copy transfer
#[derive(Debug, Serialize, Deserialize)]
pub struct SharedMemoryInfo {
    pub buffer_id: u64,
    pub address: usize,
    pub size: usize,
    pub format: String,
    pub dimensions: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_serialization() {
        let request = PythonRequest::ProcessImage {
            image_data: vec![0, 1, 2, 3],
            width: 100,
            height: 100,
            operation: ImageOperation::Resize { width: 50, height: 50 },
        };
        
        // Serialize
        let json = serde_json::to_string(&request).unwrap();
        
        // Deserialize
        let deserialized: PythonRequest = serde_json::from_str(&json).unwrap();
        
        match deserialized {
            PythonRequest::ProcessImage { width, height, .. } => {
                assert_eq!(width, 100);
                assert_eq!(height, 100);
            }
            _ => panic!("Wrong request type"),
        }
    }
}