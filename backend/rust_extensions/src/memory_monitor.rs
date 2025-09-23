use pyo3::prelude::*;
use sysinfo::{System, SystemExt, ProcessExt, PidExt};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::HashMap;
use parking_lot::RwLock;

#[pyclass]
pub struct RustMemoryMonitor {
    system: Arc<RwLock<System>>,
    cache: Arc<Mutex<MemoryCache>>,
    #[pyo3(get)]
    update_interval: f64,
}

struct MemoryCache {
    last_update: Instant,
    stats: MemoryStats,
}

#[derive(Clone)]
struct MemoryStats {
    total_mb: f64,
    available_mb: f64,
    used_mb: f64,
    percent: f64,
    process_mb: f64,
    ml_models_mb: f64,
    ml_models_count: u32,
}

#[pymethods]
impl RustMemoryMonitor {
    #[new]
    pub fn new(update_interval: Option<f64>) -> Self {
        // Use mimalloc for better memory performance
        #[cfg(feature = "mimalloc")]
        let _ = mimalloc::MiMalloc;
        
        let mut system = System::new_all();
        system.refresh_all();
        
        let initial_stats = MemoryStats {
            total_mb: system.total_memory() as f64 / 1024.0 / 1024.0,
            available_mb: system.available_memory() as f64 / 1024.0 / 1024.0,
            used_mb: system.used_memory() as f64 / 1024.0 / 1024.0,
            percent: (system.used_memory() as f64 / system.total_memory() as f64) * 100.0,
            process_mb: 0.0,
            ml_models_mb: 0.0,
            ml_models_count: 0,
        };
        
        Self {
            system: Arc::new(RwLock::new(system)),
            cache: Arc::new(Mutex::new(MemoryCache {
                last_update: Instant::now(),
                stats: initial_stats,
            })),
            update_interval: update_interval.unwrap_or(0.1), // 100ms default
        }
    }
    
    /// Get memory statistics with caching for performance
    pub fn get_memory_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut cache = self.cache.lock().unwrap();
        
        // Check if cache is still valid
        if cache.last_update.elapsed().as_secs_f64() > self.update_interval {
            // Update system info
            let mut system = self.system.write();
            system.refresh_memory();
            
            // Get current process info
            let pid = std::process::id();
            let process_mb = if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
                process.memory() as f64 / 1024.0 / 1024.0
            } else {
                0.0
            };
            
            // Update stats
            cache.stats = MemoryStats {
                total_mb: system.total_memory() as f64 / 1024.0 / 1024.0,
                available_mb: system.available_memory() as f64 / 1024.0 / 1024.0,
                used_mb: system.used_memory() as f64 / 1024.0 / 1024.0,
                percent: (system.used_memory() as f64 / system.total_memory() as f64) * 100.0,
                process_mb,
                ml_models_mb: cache.stats.ml_models_mb, // Keep existing ML stats
                ml_models_count: cache.stats.ml_models_count,
            };
            cache.last_update = Instant::now();
        }
        
        // Convert to Python dict
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_mb", cache.stats.total_mb)?;
        dict.set_item("available_mb", cache.stats.available_mb)?;
        dict.set_item("used_mb", cache.stats.used_mb)?;
        dict.set_item("system_percent", cache.stats.percent)?;
        dict.set_item("process_mb", cache.stats.process_mb)?;
        dict.set_item("ml_models_mb", cache.stats.ml_models_mb)?;
        dict.set_item("ml_models_count", cache.stats.ml_models_count)?;
        dict.set_item("ml_percent", (cache.stats.ml_models_mb / cache.stats.total_mb) * 100.0)?;
        
        Ok(dict.into())
    }
    
    /// Update ML model memory usage (called from Python)
    pub fn update_ml_stats(&self, models_mb: f64, models_count: u32) -> PyResult<()> {
        let mut cache = self.cache.lock().unwrap();
        cache.stats.ml_models_mb = models_mb;
        cache.stats.ml_models_count = models_count;
        Ok(())
    }
    
    /// Get detailed process memory map
    pub fn get_memory_map(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut system = self.system.write();
        system.refresh_processes();
        
        let pid = std::process::id();
        let memory_regions = pyo3::types::PyList::empty(py);
        
        if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
            // Get virtual memory info
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("virtual_memory", process.virtual_memory())?;
            dict.set_item("memory", process.memory())?;
            dict.set_item("cpu_usage", process.cpu_usage())?;
            dict.set_item("name", process.name())?;
            
            return Ok(dict.into());
        }
        
        Ok(memory_regions.into())
    }
    
    /// Check if memory is below threshold (fast check for hot path)
    pub fn is_memory_available(&self, required_mb: f64, threshold_percent: f64) -> bool {
        let cache = self.cache.lock().unwrap();
        
        // Quick check without system refresh
        let would_use_mb = cache.stats.used_mb + required_mb;
        let would_use_percent = (would_use_mb / cache.stats.total_mb) * 100.0;
        
        would_use_percent < threshold_percent && cache.stats.available_mb > required_mb
    }
    
    /// Force memory cleanup suggestions
    pub fn suggest_cleanup(&self, py: Python<'_>, target_mb: f64) -> PyResult<PyObject> {
        let mut system = self.system.write();
        system.refresh_processes();
        
        let suggestions = pyo3::types::PyList::empty(py);
        let mut potential_freed = 0.0;
        
        // Find processes that could be cleaned up
        let mut processes: Vec<_> = system.processes()
            .iter()
            .map(|(_, p)| (p.memory() as f64 / 1024.0 / 1024.0, p.name().to_string()))
            .filter(|(mem, name)| {
                *mem > 50.0 && !name.contains("jarvis") && !name.contains("system")
            })
            .collect();
        
        processes.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        
        for (mem, name) in processes.iter().take(5) {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("process", name)?;
            dict.set_item("memory_mb", mem)?;
            suggestions.append(dict)?;
            
            potential_freed += mem;
            if potential_freed >= target_mb {
                break;
            }
        }
        
        let result = pyo3::types::PyDict::new(py);
        result.set_item("suggestions", suggestions)?;
        result.set_item("potential_freed_mb", potential_freed)?;
        
        Ok(result.into())
    }
}