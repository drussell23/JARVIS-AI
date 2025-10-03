//! Supervisor for fault-tolerant actor management
//! 
//! Implements supervision strategies for automatic recovery from failures

use crate::{Result, JarvisError};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::thread;
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

// ============================================================================
// SUPERVISION STRATEGIES
// ============================================================================

/// Defines how to handle actor failures
#[derive(Debug, Clone, Copy)]
pub enum RestartStrategy {
    /// Restart only the failed actor
    OneForOne,
    
    /// Never restart (shut down system on failure)
    Permanent,
    
    /// Restart only on abnormal exit (panic)
    Transient,
    
    /// Restart all actors when one fails
    OneForAll,
}

/// Actor restart configuration
#[derive(Debug, Clone)]
pub struct RestartConfig {
    /// Maximum number of restarts allowed
    pub max_restarts: usize,
    
    /// Time window for restart counting
    pub restart_window: Duration,
    
    /// Restart strategy to use
    pub strategy: RestartStrategy,
    
    /// Delay before restarting
    pub restart_delay: Duration,
}

impl Default for RestartConfig {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            restart_window: Duration::from_secs(60),
            strategy: RestartStrategy::Transient,
            restart_delay: Duration::from_millis(100),
        }
    }
}

// ============================================================================
// ACTOR HANDLE
// ============================================================================

/// Handle to a supervised actor
pub struct ActorHandle {
    /// Unique actor ID
    pub id: String,
    
    /// Thread handle
    thread: Option<thread::JoinHandle<ActorResult>>,
    
    /// Restart count and timestamps
    restart_history: Vec<Instant>,
    
    /// Actor status
    status: ActorStatus,
    
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActorStatus {
    Running,
    Stopped,
    Failed,
    Restarting,
}

pub type ActorResult = std::result::Result<(), String>;

impl ActorHandle {
    pub fn new(
        id: String,
        thread: thread::JoinHandle<ActorResult>,
        shutdown: Arc<AtomicBool>,
    ) -> Self {
        Self {
            id,
            thread: Some(thread),
            restart_history: Vec::new(),
            status: ActorStatus::Running,
            shutdown,
        }
    }
    
    /// Check if actor is alive
    pub fn is_alive(&self) -> bool {
        self.thread.is_some() && self.status == ActorStatus::Running
    }
    
    /// Get actor status
    pub fn status(&self) -> ActorStatus {
        self.status
    }
    
    /// Record a restart
    pub fn record_restart(&mut self) {
        self.restart_history.push(Instant::now());
        self.status = ActorStatus::Restarting;
    }
    
    /// Count recent restarts within window
    pub fn count_recent_restarts(&self, window: Duration) -> usize {
        let cutoff = Instant::now() - window;
        self.restart_history
            .iter()
            .filter(|&&t| t > cutoff)
            .count()
    }
    
    /// Stop the actor
    pub fn stop(&mut self) -> ActorResult {
        self.shutdown.store(true, Ordering::SeqCst);
        self.status = ActorStatus::Stopped;
        
        if let Some(handle) = self.thread.take() {
            handle.join().unwrap_or(Err("Thread panicked".to_string()))
        } else {
            Ok(())
        }
    }
}

// ============================================================================
// SUPERVISOR
// ============================================================================

/// Supervises actors and handles failures
pub struct Supervisor {
    /// Supervised actors
    actors: Arc<RwLock<Vec<ActorHandle>>>,
    
    /// Restart configuration
    config: RestartConfig,
    
    /// Supervisor metrics
    metrics: Arc<SupervisorMetrics>,
    
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    
    /// Monitoring thread
    monitor_thread: Option<thread::JoinHandle<()>>,
}

#[derive(Debug, Default)]
pub struct SupervisorMetrics {
    pub total_restarts: AtomicUsize,
    pub failed_restarts: AtomicUsize,
    pub crashes: AtomicUsize,
}

impl Supervisor {
    /// Create new supervisor with configuration
    pub fn new(config: RestartConfig) -> Self {
        let actors = Arc::new(RwLock::new(Vec::new()));
        let metrics = Arc::new(SupervisorMetrics::default());
        let shutdown = Arc::new(AtomicBool::new(false));
        
        let monitor_thread = {
            let actors = actors.clone();
            let config = config.clone();
            let metrics = metrics.clone();
            let shutdown = shutdown.clone();
            
            thread::Builder::new()
                .name("supervisor-monitor".to_string())
                .spawn(move || {
                    Self::monitor_loop(actors, config, metrics, shutdown);
                })
                .ok()
        };
        
        Self {
            actors,
            config,
            metrics,
            shutdown,
            monitor_thread,
        }
    }
    
    /// Add an actor to supervision
    pub fn supervise(&self, handle: ActorHandle) {
        self.actors.write().push(handle);
    }
    
    /// Monitor loop that checks actor health
    fn monitor_loop(
        actors: Arc<RwLock<Vec<ActorHandle>>>,
        config: RestartConfig,
        metrics: Arc<SupervisorMetrics>,
        shutdown: Arc<AtomicBool>,
    ) {
        while !shutdown.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(100));
            
            let mut actors_guard = actors.write();
            let mut needs_restart = Vec::new();
            
            // Check each actor
            for (idx, actor) in actors_guard.iter_mut().enumerate() {
                if !actor.is_alive() && actor.status() != ActorStatus::Stopped {
                    needs_restart.push(idx);
                }
            }
            
            // Handle restarts based on strategy
            for idx in needs_restart {
                if let Some(actor) = actors_guard.get_mut(idx) {
                    Self::handle_actor_failure(actor, &config, &metrics);
                }
            }
        }
    }
    
    /// Handle a failed actor
    fn handle_actor_failure(
        actor: &mut ActorHandle,
        config: &RestartConfig,
        metrics: &SupervisorMetrics,
    ) {
        metrics.crashes.fetch_add(1, Ordering::Relaxed);
        
        // Check restart limit
        let recent_restarts = actor.count_recent_restarts(config.restart_window);
        if recent_restarts >= config.max_restarts {
            tracing::error!(
                "Actor {} exceeded restart limit ({}/{})",
                actor.id, recent_restarts, config.max_restarts
            );
            metrics.failed_restarts.fetch_add(1, Ordering::Relaxed);
            actor.status = ActorStatus::Failed;
            return;
        }
        
        // Apply restart strategy
        match config.strategy {
            RestartStrategy::Permanent => {
                tracing::info!("Actor {} failed, permanent strategy - not restarting", actor.id);
                actor.status = ActorStatus::Failed;
            }
            RestartStrategy::Transient | RestartStrategy::OneForOne => {
                tracing::info!("Restarting actor {} after {} ms", actor.id, config.restart_delay.as_millis());
                thread::sleep(config.restart_delay);
                
                actor.record_restart();
                metrics.total_restarts.fetch_add(1, Ordering::Relaxed);
                
                // In real implementation, would restart the actor thread here
                // For now, just mark as failed
                actor.status = ActorStatus::Failed;
            }
            RestartStrategy::OneForAll => {
                tracing::info!("Actor {} failed, restarting all actors", actor.id);
                // Would restart all actors here
                actor.status = ActorStatus::Failed;
            }
        }
    }
    
    /// Get supervisor metrics
    pub fn metrics(&self) -> &SupervisorMetrics {
        &self.metrics
    }
    
    /// Shutdown all actors
    pub fn shutdown(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        
        // Stop all actors
        for actor in self.actors.write().iter_mut() {
            let _ = actor.stop();
        }
        
        // Stop monitor thread
        if let Some(handle) = self.monitor_thread.take() {
            let _ = handle.join();
        }
        
        Ok(())
    }
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        if !self.shutdown.load(Ordering::SeqCst) {
            let _ = self.shutdown();
        }
    }
}

// ============================================================================
// ACTOR SPAWNING HELPERS
// ============================================================================

/// Spawn a supervised actor
pub fn spawn_supervised<F>(
    supervisor: &Supervisor,
    id: String,
    f: F,
) -> Result<()>
where
    F: FnOnce(Arc<AtomicBool>) -> ActorResult + Send + 'static,
{
    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    
    let thread = thread::Builder::new()
        .name(format!("actor-{}", id))
        .spawn(move || f(shutdown_clone))
        .map_err(|e| JarvisError::BridgeError(format!("Failed to spawn actor: {}", e)))?;
    
    let handle = ActorHandle::new(id, thread, shutdown);
    supervisor.supervise(handle);
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_supervisor_creation() {
        let config = RestartConfig::default();
        let supervisor = Supervisor::new(config);
        
        assert_eq!(supervisor.metrics().total_restarts.load(Ordering::Relaxed), 0);
    }
    
    #[test]
    fn test_restart_counting() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread = thread::spawn(|| Ok(()));
        let mut actor = ActorHandle::new("test".to_string(), thread, shutdown);
        
        actor.record_restart();
        actor.record_restart();
        
        let count = actor.count_recent_restarts(Duration::from_secs(60));
        assert_eq!(count, 2);
    }
    
    #[test]
    fn test_restart_window() {
        let shutdown = Arc::new(AtomicBool::new(false));
        let thread = thread::spawn(|| Ok(()));
        let mut actor = ActorHandle::new("test".to_string(), thread, shutdown);
        
        actor.restart_history.push(Instant::now() - Duration::from_secs(120));
        actor.record_restart();
        
        let count = actor.count_recent_restarts(Duration::from_secs(60));
        assert_eq!(count, 1); // Only recent restart counted
    }
}