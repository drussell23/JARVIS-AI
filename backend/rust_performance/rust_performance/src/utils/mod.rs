//! Utility functions

use std::time::{Duration, Instant};
use pyo3::prelude::*;

/// CPU throttling utility
#[pyclass]
pub struct CPUThrottler {
    target_percent: f32,
    last_check: Instant,
    check_interval: Duration,
}

#[pymethods]
impl CPUThrottler {
    #[new]
    fn new(target_percent: f32) -> Self {
        CPUThrottler {
            target_percent,
            last_check: Instant::now(),
            check_interval: Duration::from_millis(100),
        }
    }
    
    /// Apply throttling based on current load
    fn throttle(&mut self) {
        let now = Instant::now();
        if now.duration_since(self.last_check) < self.check_interval {
            return;
        }
        
        self.last_check = now;
        
        // Simple throttling - sleep to reduce CPU usage
        if self.target_percent < 100.0 {
            let sleep_ms = ((100.0 - self.target_percent) / 10.0) as u64;
            std::thread::sleep(Duration::from_millis(sleep_ms));
        }
    }
}