//! Python-Rust bridge using PyO3

pub mod pyo3_bindings;  // Thread-safe implementation
pub mod serialization;
pub mod objc_bridge;
pub mod supervisor;
pub mod screencapture_stream;
pub mod notification_monitor;

use crate::Result;

pub use objc_bridge::{ObjCBridge, ObjCCommand, ObjCResponse, CaptureQuality, CaptureRegion};
pub use supervisor::{Supervisor, RestartStrategy, RestartConfig};
pub use screencapture_stream::{ScreenCaptureStream, DynamicCaptureConfig, PerformanceMode};
pub use notification_monitor::{NotificationMonitor, NotificationFilter, NotificationEvent};

#[cfg(feature = "python-bindings")] 
use pyo3::prelude::*; 

/// Register Python module with all bindings
#[cfg(feature = "python-bindings")]
pub fn register_python_module(m: &PyModule) -> PyResult<()> {
    // Use the thread-safe bindings
    pyo3_bindings::register_python_module(m)
}