//! Work stealing scheduler for load balancing

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use crossbeam::deque::{Injector, Stealer, Worker};
use parking_lot::Mutex;
use dashmap::DashMap;

/// Work stealing scheduler
pub struct WorkStealingScheduler {
    /// Global task queue
    injector: Arc<Injector<TaskItem>>,
    /// Per-worker queues (wrapped for thread safety)
    workers: Arc<Mutex<Vec<Worker<TaskItem>>>>,
    /// Stealers for each worker
    stealers: Arc<Vec<Stealer<TaskItem>>>,
    /// Task registry
    tasks: Arc<DashMap<u64, TaskMetadata>>,
    /// Load balancing stats
    steal_count: Arc<AtomicU64>,
}

#[derive(Clone)]
struct TaskItem {
    id: u64,
    priority: u8,
    work: Arc<dyn Fn() + Send + Sync>,
}

#[derive(Debug, Clone)]
struct TaskMetadata {
    worker_id: Option<usize>,
    steal_count: u32,
    created_at: std::time::Instant,
}

impl WorkStealingScheduler {
    pub fn new(worker_count: usize) -> Self {
        let injector = Arc::new(Injector::new());
        let mut workers = Vec::with_capacity(worker_count);
        let mut stealers = Vec::with_capacity(worker_count);
        
        for _ in 0..worker_count {
            let worker = Worker::new_fifo();
            stealers.push(worker.stealer());
            workers.push(worker);
        }
        
        Self {
            injector,
            workers: Arc::new(Mutex::new(workers)),
            stealers: Arc::new(stealers),
            tasks: Arc::new(DashMap::new()),
            steal_count: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Register a new task
    pub fn register_task(&self, task_id: u64) {
        self.tasks.insert(task_id, TaskMetadata {
            worker_id: None,
            steal_count: 0,
            created_at: std::time::Instant::now(),
        });
    }
    
    /// Complete a task
    pub fn complete_task(&self, task_id: u64) {
        self.tasks.remove(&task_id);
    }
    
    /// Push task to global queue
    pub fn push_global(&self, task: TaskItem) {
        self.injector.push(task);
    }
    
    /// Push task to specific worker
    pub fn push_worker(&self, worker_id: usize, task: TaskItem) {
        let workers = self.workers.lock();
        if worker_id < workers.len() {
            if let Some(mut metadata) = self.tasks.get_mut(&task.id) {
                metadata.worker_id = Some(worker_id);
            }
            workers[worker_id].push(task);
        }
    }
    
    /// Try to find work for a worker
    pub fn find_work(&self, worker_id: usize) -> Option<TaskItem> {
        // First, try local queue
        {
            let workers = self.workers.lock();
            if let Some(task) = workers.get(worker_id).and_then(|w| w.pop()) {
                return Some(task);
            }
        }
        
        // Then try to steal from global queue
        if let Some(task) = self.injector.steal().success() {
            return Some(task);
        }
        
        // Finally, try to steal from other workers
        let stealers = &self.stealers;
        let worker_count = stealers.len();
        
        // Start from a random position to avoid always stealing from the same worker
        let start = rand::random::<usize>() % worker_count;
        
        for i in 0..worker_count {
            let stealer_id = (start + i) % worker_count;
            if stealer_id == worker_id {
                continue; // Don't steal from self
            }
            
            if let Some(task) = stealers[stealer_id].steal().success() {
                // Update steal statistics
                self.steal_count.fetch_add(1, Ordering::Relaxed);
                if let Some(mut metadata) = self.tasks.get_mut(&task.id) {
                    metadata.steal_count += 1;
                    metadata.worker_id = Some(worker_id);
                }
                
                return Some(task);
            }
        }
        
        None
    }
    
    /// Get load statistics for each worker
    pub fn load_stats(&self) -> Vec<WorkerLoadStats> {
        let workers = self.workers.lock();
        workers.iter().enumerate().map(|(id, worker)| {
            WorkerLoadStats {
                worker_id: id,
                queue_length: worker.len(),
                tasks_assigned: self.tasks.iter()
                    .filter(|entry| entry.value().worker_id == Some(id))
                    .count(),
            }
        }).collect()
    }
    
    /// Get global statistics
    pub fn global_stats(&self) -> GlobalStats {
        GlobalStats {
            total_tasks: self.tasks.len(),
            global_queue_length: self.injector.len(),
            total_steals: self.steal_count.load(Ordering::Relaxed),
            avg_steal_count: if self.tasks.is_empty() { 
                0.0 
            } else {
                self.tasks.iter()
                    .map(|entry| entry.value().steal_count as f64)
                    .sum::<f64>() / self.tasks.len() as f64
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorkerLoadStats {
    pub worker_id: usize,
    pub queue_length: usize,
    pub tasks_assigned: usize,
}

#[derive(Debug, Clone)]
pub struct GlobalStats {
    pub total_tasks: usize,
    pub global_queue_length: usize,
    pub total_steals: u64,
    pub avg_steal_count: f64,
}