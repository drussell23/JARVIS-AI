//! Task pool for dedicated thread management

use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crossbeam::channel::{bounded, unbounded, Sender, Receiver};
use parking_lot::Mutex;
use crate::Result;

/// Task pool for managing dedicated threads
pub struct TaskPool {
    name: String,
    threads: Vec<JoinHandle<()>>,
    sender: Sender<Task>,
    shutdown: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
}

type Task = Box<dyn FnOnce() + Send + 'static>;

impl TaskPool {
    pub fn new(name: &str, thread_count: usize) -> Self {
        let (sender, receiver) = unbounded();
        let receiver = Arc::new(Mutex::new(receiver));
        let shutdown = Arc::new(AtomicBool::new(false));
        let active_tasks = Arc::new(AtomicUsize::new(0));
        
        let mut threads = Vec::with_capacity(thread_count);
        
        for i in 0..thread_count {
            let thread_name = format!("{}-{}", name, i);
            let receiver = receiver.clone();
            let shutdown = shutdown.clone();
            let active_tasks = active_tasks.clone();
            
            let handle = thread::Builder::new()
                .name(thread_name)
                .stack_size(2 * 1024 * 1024) // 2MB stack
                .spawn(move || {
                    worker_thread(receiver, shutdown, active_tasks);
                })
                .expect("Failed to spawn worker thread");
            
            threads.push(handle);
        }
        
        Self {
            name: name.to_string(),
            threads,
            sender,
            shutdown,
            active_tasks,
        }
    }
    
    /// Submit task to pool
    pub fn submit<F>(&self, task: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(crate::JarvisError::InvalidOperation("Task pool is shut down".into()));
        }
        
        self.sender.send(Box::new(task))
            .map_err(|_| crate::JarvisError::InvalidOperation("Failed to submit task".into()))?;
        
        Ok(())
    }
    
    /// Get number of active tasks
    pub fn active_tasks(&self) -> usize {
        self.active_tasks.load(Ordering::Relaxed)
    }
    
    /// Shutdown the pool
    pub async fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        
        // Send shutdown signal to all threads
        for _ in &self.threads {
            let _ = self.sender.send(Box::new(|| {}));
        }
        
        // Note: Can't join threads from async context
        // In production, you'd handle this differently
    }
}

fn worker_thread(
    receiver: Arc<Mutex<Receiver<Task>>>,
    shutdown: Arc<AtomicBool>,
    active_tasks: Arc<AtomicUsize>,
) {
    while !shutdown.load(Ordering::Relaxed) {
        // Try to receive with timeout
        let task = {
            let receiver = receiver.lock();
            match receiver.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(task) => task,
                Err(_) => continue,
            }
        };
        
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        
        // Execute task
        active_tasks.fetch_add(1, Ordering::Relaxed);
        task();
        active_tasks.fetch_sub(1, Ordering::Relaxed);
    }
}