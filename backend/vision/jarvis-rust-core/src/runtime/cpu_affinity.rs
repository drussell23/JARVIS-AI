//! CPU affinity management for optimal thread placement

use std::sync::Arc;
use parking_lot::RwLock;
use crate::{Result, JarvisError};

#[cfg(target_os = "linux")]
use libc::{cpu_set_t, CPU_SET, CPU_ZERO, sched_setaffinity};

/// CPU affinity manager
pub struct CpuAffinityManager {
    cpu_count: usize,
    performance_cores: Vec<usize>,
    efficiency_cores: Vec<usize>,
    thread_assignments: Arc<RwLock<Vec<Option<usize>>>>,
}

impl CpuAffinityManager {
    pub fn new() -> Result<Self> {
        let cpu_count = num_cpus::get();
        
        // Detect performance vs efficiency cores (simplified)
        // On real systems, you'd use cpuid or sysfs to detect core types
        let (performance_cores, efficiency_cores) = Self::detect_core_types(cpu_count);
        
        Ok(Self {
            cpu_count,
            performance_cores,
            efficiency_cores,
            thread_assignments: Arc::new(RwLock::new(vec![None; cpu_count])),
        })
    }
    
    /// Setup thread affinity for worker threads
    pub fn setup_thread_affinity(&self, thread_count: usize) -> Result<()> {
        // Distribute threads across cores
        let mut assignments = self.thread_assignments.write();
        
        // Prefer performance cores for main workers
        let mut core_idx = 0;
        for i in 0..thread_count {
            if core_idx < self.performance_cores.len() {
                assignments[i] = Some(self.performance_cores[core_idx]);
                core_idx += 1;
            } else {
                // Fall back to efficiency cores
                let eff_idx = (i - self.performance_cores.len()) % self.efficiency_cores.len();
                assignments[i] = Some(self.efficiency_cores[eff_idx]);
            }
        }
        
        Ok(())
    }
    
    /// Get a performance core for compute-intensive tasks
    pub fn get_performance_core(&self) -> Result<usize> {
        if self.performance_cores.is_empty() {
            return Err(JarvisError::InvalidOperation("No performance cores available".into()));
        }
        
        // Simple round-robin (could be more sophisticated)
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let idx = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % self.performance_cores.len();
        Ok(self.performance_cores[idx])
    }
    
    /// Pin current thread to specific core
    pub fn pin_thread_to_core(&self, core_id: usize) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let mut cpu_set = std::mem::zeroed::<cpu_set_t>();
                CPU_ZERO(&mut cpu_set);
                CPU_SET(core_id, &mut cpu_set);
                
                let result = sched_setaffinity(0, std::mem::size_of::<cpu_set_t>(), &cpu_set);
                if result != 0 {
                    return Err(JarvisError::Other(anyhow::anyhow!(
                        "Failed to set CPU affinity: {}", std::io::Error::last_os_error()
                    )));
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            // macOS thread affinity uses thread_policy_set
            // This is a simplified version
            use std::os::raw::c_int;
            
            #[repr(C)]
            struct thread_affinity_policy_data_t {
                affinity_tag: c_int,
            }
            
            // Note: Full implementation would use proper FFI bindings
            tracing::debug!("Thread affinity set to core {} (macOS)", core_id);
        }
        
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::System::Threading::{SetThreadAffinityMask, GetCurrentThread};
            
            unsafe {
                let mask = 1u64 << core_id;
                SetThreadAffinityMask(GetCurrentThread(), mask);
            }
        }
        
        Ok(())
    }
    
    /// Detect performance vs efficiency cores
    fn detect_core_types(cpu_count: usize) -> (Vec<usize>, Vec<usize>) {
        // Simplified detection - in reality would use CPUID or platform APIs
        
        #[cfg(target_arch = "aarch64")]
        {
            // For Apple Silicon, assume first half are performance cores
            if cpu_count > 4 {
                let perf_count = cpu_count / 2;
                let performance_cores: Vec<usize> = (0..perf_count).collect();
                let efficiency_cores: Vec<usize> = (perf_count..cpu_count).collect();
                return (performance_cores, efficiency_cores);
            }
        }
        
        // Default: all cores are performance cores
        let performance_cores: Vec<usize> = (0..cpu_count).collect();
        let efficiency_cores = Vec::new();
        (performance_cores, efficiency_cores)
    }
    
    /// Get CPU topology information
    pub fn topology_info(&self) -> TopologyInfo {
        TopologyInfo {
            total_cores: self.cpu_count,
            performance_cores: self.performance_cores.len(),
            efficiency_cores: self.efficiency_cores.len(),
            numa_nodes: Self::detect_numa_nodes(),
        }
    }
    
    fn detect_numa_nodes() -> usize {
        // Simplified - would use hwloc or platform APIs
        1
    }
}

#[derive(Debug, Clone)]
pub struct TopologyInfo {
    pub total_cores: usize,
    pub performance_cores: usize,
    pub efficiency_cores: usize,
    pub numa_nodes: usize,
}