//! High-performance resource monitoring for JARVIS
//! Provides CPU and memory monitoring with minimal overhead

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use sysinfo::{System, SystemExt, ProcessExt, CpuExt};

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceStats {
    pub cpu_percent: f64,
    pub memory_used_gb: f64,
    pub memory_available_gb: f64,
    pub memory_percent: f64,
    pub process_cpu_percent: f64,
    pub process_memory_mb: f64,
}

/// High-performance resource monitor
pub struct ResourceMonitor {
    system: Arc<std::sync::Mutex<System>>,
    running: Arc<AtomicBool>,
    interval: Duration,
    max_cpu: f64,
    max_memory_gb: f64,
    stats: Arc<std::sync::Mutex<ResourceStats>>,
}

impl ResourceMonitor {
    /// Create a new resource monitor
    pub fn new(max_cpu: f64, max_memory_gb: f64, interval_seconds: f64) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        
        Self {
            system: Arc::new(std::sync::Mutex::new(system)),
            running: Arc::new(AtomicBool::new(false)),
            interval: Duration::from_secs_f64(interval_seconds),
            max_cpu,
            max_memory_gb,
            stats: Arc::new(std::sync::Mutex::new(ResourceStats {
                cpu_percent: 0.0,
                memory_used_gb: 0.0,
                memory_available_gb: 0.0,
                memory_percent: 0.0,
                process_cpu_percent: 0.0,
                process_memory_mb: 0.0,
            })),
        }
    }
    
    /// Start monitoring in a background thread
    pub fn start(&self) -> Arc<AtomicBool> {
        self.running.store(true, Ordering::SeqCst);
        let running = Arc::clone(&self.running);
        let system = Arc::clone(&self.system);
        let stats = Arc::clone(&self.stats);
        let interval = self.interval;
        let max_cpu = self.max_cpu;
        let max_memory_gb = self.max_memory_gb;
        
        thread::spawn(move || {
            let mut last_check = Instant::now();
            
            while running.load(Ordering::SeqCst) {
                let now = Instant::now();
                
                // Only update if interval has passed
                if now.duration_since(last_check) >= interval {
                    // Update system info
                    let mut sys = system.lock().unwrap();
                    sys.refresh_cpu();
                    sys.refresh_memory();
                    sys.refresh_processes();
                    
                    // Calculate stats
                    let cpu_percent = sys.global_cpu_info().cpu_usage() as f64;
                    let total_memory = sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0; // GB
                    let used_memory = sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0; // GB
                    let available_memory = sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0; // GB
                    let memory_percent = (used_memory / total_memory) * 100.0;
                    
                    // Get current process stats
                    let pid = sysinfo::get_current_pid().unwrap();
                    let (process_cpu, process_memory_mb) = if let Some(process) = sys.process(pid) {
                        (
                            process.cpu_usage() as f64,
                            process.memory() as f64 / 1024.0 / 1024.0 // MB
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    
                    // Update stats
                    let mut current_stats = stats.lock().unwrap();
                    current_stats.cpu_percent = cpu_percent;
                    current_stats.memory_used_gb = used_memory;
                    current_stats.memory_available_gb = available_memory;
                    current_stats.memory_percent = memory_percent;
                    current_stats.process_cpu_percent = process_cpu;
                    current_stats.process_memory_mb = process_memory_mb;
                    
                    // Check thresholds
                    if cpu_percent > max_cpu {
                        eprintln!("⚠️ High CPU usage: {:.1}%", cpu_percent);
                    }
                    if used_memory > max_memory_gb {
                        eprintln!("⚠️ High memory usage: {:.1}GB", used_memory);
                    }
                    
                    last_check = now;
                }
                
                // Sleep to prevent CPU spinning
                thread::sleep(Duration::from_millis(100));
            }
        });
        
        Arc::clone(&self.running)
    }
    
    /// Stop monitoring
    pub fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
    }
    
    /// Get current stats
    pub fn get_stats(&self) -> ResourceStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        let stats = self.stats.lock().unwrap();
        stats.cpu_percent < self.max_cpu && 
        stats.memory_used_gb < self.max_memory_gb
    }
}

/// macOS-specific optimizations
#[cfg(target_os = "macos")]
pub mod macos {
    use mach2::mach_time::{mach_absolute_time, mach_timebase_info};
    use mach2::vm_statistics64::*;
    use mach2::mach_port::mach_port_t;
    use mach2::kern_return::KERN_SUCCESS;
    use std::mem;
    
    /// Get precise CPU usage using Mach APIs
    pub fn get_cpu_usage_precise() -> f64 {
        // Implementation using Mach host_processor_info
        // This is more accurate than generic sysinfo
        0.0 // Placeholder
    }
    
    /// Get memory pressure using Mach VM statistics
    pub fn get_memory_pressure() -> (f64, f64) {
        unsafe {
            let mut vm_stat: vm_statistics64 = mem::zeroed();
            let mut count = mem::size_of::<vm_statistics64>() as u32 / mem::size_of::<u32>() as u32;
            
            let host_port = mach2::mach_init::mach_host_self();
            let result = mach2::vm::host_statistics64(
                host_port,
                HOST_VM_INFO64,
                &mut vm_stat as *mut _ as *mut _,
                &mut count
            );
            
            if result == KERN_SUCCESS {
                let page_size = 4096u64; // macOS page size
                let total_pages = vm_stat.free_count + vm_stat.active_count + 
                                 vm_stat.inactive_count + vm_stat.wire_count;
                let used_pages = vm_stat.active_count + vm_stat.wire_count;
                
                let total_gb = (total_pages as f64 * page_size as f64) / 1024.0 / 1024.0 / 1024.0;
                let used_gb = (used_pages as f64 * page_size as f64) / 1024.0 / 1024.0 / 1024.0;
                
                (used_gb, total_gb)
            } else {
                (0.0, 0.0)
            }
        }
    }
}

/// C API for Python integration
#[no_mangle]
pub extern "C" fn create_resource_monitor(
    max_cpu: f64,
    max_memory_gb: f64,
    interval_seconds: f64
) -> *mut ResourceMonitor {
    Box::into_raw(Box::new(ResourceMonitor::new(max_cpu, max_memory_gb, interval_seconds)))
}

#[no_mangle]
pub extern "C" fn start_resource_monitor(monitor: *mut ResourceMonitor) {
    unsafe {
        if !monitor.is_null() {
            (*monitor).start();
        }
    }
}

#[no_mangle]
pub extern "C" fn stop_resource_monitor(monitor: *mut ResourceMonitor) {
    unsafe {
        if !monitor.is_null() {
            (*monitor).stop();
        }
    }
}

#[no_mangle]
pub extern "C" fn get_resource_stats(monitor: *mut ResourceMonitor) -> ResourceStats {
    unsafe {
        if !monitor.is_null() {
            (*monitor).get_stats()
        } else {
            ResourceStats {
                cpu_percent: 0.0,
                memory_used_gb: 0.0,
                memory_available_gb: 0.0,
                memory_percent: 0.0,
                process_cpu_percent: 0.0,
                process_memory_mb: 0.0,
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn free_resource_monitor(monitor: *mut ResourceMonitor) {
    unsafe {
        if !monitor.is_null() {
            (*monitor).stop();
            drop(Box::from_raw(monitor));
        }
    }
}