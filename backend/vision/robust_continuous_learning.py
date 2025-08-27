#!/usr/bin/env python3
"""
Robust Advanced Continuous Learning System for Vision System v2.0
Handles asyncio properly, manages CPU usage, and provides graceful degradation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import asyncio
from collections import defaultdict, deque
import threading
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import signal
from contextlib import contextmanager

from .experience_replay_system import get_experience_replay_system, Experience, ReplayBatch
from .meta_learning_framework import get_meta_learning_framework, MetaLearningFramework
from .continuous_learning_pipeline import OnlineLearner

# Import data structures early
from .advanced_continuous_learning import (
    LearningTask,
    AdaptiveLearningRateAdjuster,
    PerformanceMonitor
)

logger = logging.getLogger(__name__)


@dataclass
class SystemResources:
    """Track system resource usage"""
    cpu_percent: float
    memory_percent: float
    available_memory_mb: float
    io_wait: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class LearningConfig:
    """Configuration for learning system"""
    # Resource limits
    max_cpu_percent: float = 50.0  # Maximum CPU usage allowed
    max_memory_percent: float = 30.0  # Maximum memory usage allowed
    min_free_memory_mb: float = 1000.0  # Minimum free memory required
    
    # Timing configuration
    retraining_interval_hours: float = 6.0
    mini_batch_interval_minutes: float = 15.0
    task_timeout_seconds: int = 300  # 5 minutes max per task
    
    # Learning parameters
    batch_size: int = 128
    num_epochs: int = 5
    learning_rate: float = 0.001
    
    # Adaptive scheduling
    enable_adaptive_scheduling: bool = True
    load_factor_threshold: float = 0.8  # Reduce activity when system load > 80%
    critical_load_threshold: float = 0.95  # Stop all learning when load > 95%
    
    # Health monitoring
    health_check_interval_seconds: int = 30
    unhealthy_threshold_count: int = 3  # Mark unhealthy after 3 failed checks


class ResourceMonitor:
    """Monitor system resources and enforce limits"""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.resource_history = deque(maxlen=100)
        self.process = psutil.Process()
        
    def get_current_resources(self) -> SystemResources:
        """Get current system resource usage"""
        try:
            # Get system-wide stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Get IO wait if available
            io_wait = 0.0
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            if hasattr(cpu_times, 'iowait'):
                io_wait = cpu_times.iowait
            
            resources = SystemResources(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                available_memory_mb=memory.available / (1024 * 1024),
                io_wait=io_wait
            )
            
            self.resource_history.append(resources)
            return resources
            
        except Exception as e:
            logger.error(f"Failed to get system resources: {e}")
            return SystemResources(
                cpu_percent=0.0,
                memory_percent=0.0,
                available_memory_mb=float('inf')
            )
    
    def get_average_resources(self, window_seconds: int = 60) -> SystemResources:
        """Get average resource usage over time window"""
        if not self.resource_history:
            return self.get_current_resources()
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_resources = [r for r in self.resource_history if r.timestamp > cutoff_time]
        
        if not recent_resources:
            return self.resource_history[-1]
        
        return SystemResources(
            cpu_percent=np.mean([r.cpu_percent for r in recent_resources]),
            memory_percent=np.mean([r.memory_percent for r in recent_resources]),
            available_memory_mb=np.mean([r.available_memory_mb for r in recent_resources]),
            io_wait=np.mean([r.io_wait for r in recent_resources])
        )
    
    def is_resource_available(self) -> Tuple[bool, str]:
        """Check if resources are available for learning"""
        resources = self.get_current_resources()
        
        # Check CPU
        if resources.cpu_percent > self.config.max_cpu_percent:
            return False, f"CPU usage too high: {resources.cpu_percent:.1f}%"
        
        # Check memory
        if resources.memory_percent > self.config.max_memory_percent:
            return False, f"Memory usage too high: {resources.memory_percent:.1f}%"
        
        # Check free memory
        if resources.available_memory_mb < self.config.min_free_memory_mb:
            return False, f"Insufficient free memory: {resources.available_memory_mb:.1f}MB"
        
        # Check IO wait
        if resources.io_wait > 20.0:  # High IO wait
            return False, f"High IO wait: {resources.io_wait:.1f}%"
        
        return True, "Resources available"
    
    def get_load_factor(self) -> float:
        """Get system load factor (0.0 = idle, 1.0 = fully loaded)"""
        resources = self.get_average_resources()
        
        # Weighted combination of metrics
        cpu_factor = resources.cpu_percent / 100.0
        memory_factor = resources.memory_percent / 100.0
        io_factor = min(resources.io_wait / 50.0, 1.0)  # Cap at 50% IO wait
        
        # Weighted average
        load_factor = (
            0.5 * cpu_factor +
            0.3 * memory_factor +
            0.2 * io_factor
        )
        
        return min(load_factor, 1.0)


class AdaptiveScheduler:
    """Adaptive task scheduling based on system load"""
    
    def __init__(self, config: LearningConfig, resource_monitor: ResourceMonitor):
        self.config = config
        self.resource_monitor = resource_monitor
        self.schedule_adjustments = deque(maxlen=50)
        
    def get_adjusted_interval(self, base_interval: timedelta) -> timedelta:
        """Get adjusted interval based on system load"""
        if not self.config.enable_adaptive_scheduling:
            return base_interval
        
        load_factor = self.resource_monitor.get_load_factor()
        
        # Exponential backoff based on load
        if load_factor > self.config.critical_load_threshold:
            # System critically loaded - pause learning
            return timedelta(hours=24)  # Effectively disable
        elif load_factor > self.config.load_factor_threshold:
            # System heavily loaded - slow down
            multiplier = 1.0 + (load_factor - self.config.load_factor_threshold) * 10
            return base_interval * multiplier
        else:
            # System has capacity - normal scheduling
            return base_interval
    
    def should_execute_task(self, task_priority: float) -> bool:
        """Determine if task should execute based on priority and load"""
        load_factor = self.resource_monitor.get_load_factor()
        
        # Critical load - only highest priority tasks
        if load_factor > self.config.critical_load_threshold:
            return task_priority > 0.9
        
        # High load - high priority tasks only
        if load_factor > self.config.load_factor_threshold:
            return task_priority > 0.7
        
        # Normal load - all tasks
        return True
    
    def get_batch_size(self, base_batch_size: int) -> int:
        """Adjust batch size based on available resources"""
        resources = self.resource_monitor.get_current_resources()
        
        # Reduce batch size if memory is constrained
        memory_factor = 1.0 - (resources.memory_percent / 100.0)
        memory_factor = max(memory_factor, 0.1)  # At least 10%
        
        # Reduce batch size if CPU is busy
        cpu_factor = 1.0 - (resources.cpu_percent / 100.0)
        cpu_factor = max(cpu_factor, 0.2)  # At least 20%
        
        # Combined factor
        adjustment_factor = min(memory_factor, cpu_factor)
        
        adjusted_size = int(base_batch_size * adjustment_factor)
        return max(adjusted_size, 16)  # Minimum batch size of 16


class RobustAdvancedContinuousLearning:
    """
    Robust continuous learning system with proper resource management
    """
    
    def __init__(self, model: nn.Module, config: Optional[LearningConfig] = None):
        # Core components
        self.model = model
        self.config = config or LearningConfig()
        
        # Initialize subsystems
        self.experience_replay = get_experience_replay_system()
        self.meta_framework = get_meta_learning_framework(model)
        self.resource_monitor = ResourceMonitor(self.config)
        self.adaptive_scheduler = AdaptiveScheduler(self.config, self.resource_monitor)
        
        # Task management
        self.task_queue: deque = deque()
        self.active_tasks: Dict[str, LearningTask] = {}
        self.task_history: deque = deque(maxlen=1000)
        
        # Timing tracking
        self.last_retraining = datetime.now()
        self.last_mini_batch = datetime.now()
        self.last_health_check = datetime.now()
        
        # Health monitoring
        self.health_status = {
            'healthy': True,
            'consecutive_failures': 0,
            'last_error': None,
            'last_success': datetime.now()
        }
        
        # Threading and control
        self.running = True
        self.paused = False
        self.executor = ThreadPoolExecutor(max_workers=2)  # Limited workers
        self.learning_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        
        # Performance tracking
        self.performance_monitor = PerformanceMonitor()
        self.lr_adjuster = AdaptiveLearningRateAdjuster()
        
        # Initialize based on config
        self._start_learning_system()
        
        logger.info("Robust Advanced Continuous Learning System initialized")
    
    def _start_learning_system(self):
        """Start the learning system with proper async handling"""
        # Check if we should start based on environment
        if os.getenv('DISABLE_CONTINUOUS_LEARNING', '').lower() == 'true':
            self.running = False
            logger.warning("Continuous Learning DISABLED via environment variable")
            return
        
        # Start in a separate thread with its own event loop
        def run_learning_loop():
            """Run learning loop in dedicated thread"""
            # Set thread name for debugging
            threading.current_thread().name = "ContinuousLearning"
            
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the main learning loop
                loop.run_until_complete(self._learning_main_loop())
            except Exception as e:
                logger.error(f"Learning loop crashed: {e}")
            finally:
                # Clean up
                loop.close()
        
        # Start the thread
        learning_thread = threading.Thread(target=run_learning_loop, daemon=True)
        learning_thread.start()
        
        logger.info("Learning system started successfully")
    
    async def _learning_main_loop(self):
        """Main learning loop with proper error handling and resource management"""
        logger.info("Starting continuous learning main loop")
        
        while self.running:
            try:
                # Check system health
                await self._check_health()
                
                # Skip if paused or unhealthy
                if self.paused or not self.health_status['healthy']:
                    await asyncio.sleep(30)
                    continue
                
                # Check resource availability
                available, reason = self.resource_monitor.is_resource_available()
                if not available:
                    logger.debug(f"Skipping learning cycle: {reason}")
                    await asyncio.sleep(60)
                    continue
                
                # Run learning cycle with timeout
                try:
                    await asyncio.wait_for(
                        self._learning_cycle(),
                        timeout=self.config.task_timeout_seconds
                    )
                    
                    # Update health on success
                    self.health_status['consecutive_failures'] = 0
                    self.health_status['last_success'] = datetime.now()
                    
                except asyncio.TimeoutError:
                    logger.warning("Learning cycle timed out")
                    self.health_status['consecutive_failures'] += 1
                
                # Adaptive sleep based on system load
                base_sleep = 60  # 1 minute
                load_factor = self.resource_monitor.get_load_factor()
                adjusted_sleep = base_sleep * (1.0 + load_factor * 2)  # Up to 3x slower
                
                await asyncio.sleep(adjusted_sleep)
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
                self.health_status['consecutive_failures'] += 1
                self.health_status['last_error'] = str(e)
                
                # Exponential backoff on errors
                backoff = min(300, 60 * (2 ** self.health_status['consecutive_failures']))
                await asyncio.sleep(backoff)
    
    async def _check_health(self):
        """Check system health and update status"""
        now = datetime.now()
        
        # Only check periodically
        if (now - self.last_health_check).seconds < self.config.health_check_interval_seconds:
            return
        
        self.last_health_check = now
        
        # Check consecutive failures
        if self.health_status['consecutive_failures'] >= self.config.unhealthy_threshold_count:
            self.health_status['healthy'] = False
            logger.warning("Learning system marked unhealthy due to repeated failures")
        else:
            self.health_status['healthy'] = True
        
        # Log resource status
        resources = self.resource_monitor.get_current_resources()
        logger.debug(f"Resources - CPU: {resources.cpu_percent:.1f}%, "
                    f"Memory: {resources.memory_percent:.1f}%, "
                    f"Free: {resources.available_memory_mb:.1f}MB")
    
    async def _learning_cycle(self):
        """Single learning cycle with adaptive behavior"""
        # Get adjusted intervals based on load
        retraining_interval = self.adaptive_scheduler.get_adjusted_interval(
            timedelta(hours=self.config.retraining_interval_hours)
        )
        mini_batch_interval = self.adaptive_scheduler.get_adjusted_interval(
            timedelta(minutes=self.config.mini_batch_interval_minutes)
        )
        
        # Check if retraining is needed
        if await self._should_retrain(retraining_interval):
            await self._schedule_retraining()
        
        # Check if mini-batch training is needed
        elif await self._should_mini_batch(mini_batch_interval):
            await self._schedule_mini_batch_training()
        
        # Process task queue
        if self.task_queue:
            await self._process_task_queue()
        
        # Periodic maintenance
        await self._perform_maintenance()
    
    async def _should_retrain(self, interval: timedelta) -> bool:
        """Determine if full retraining is needed"""
        # Time-based check
        if (datetime.now() - self.last_retraining) > interval:
            return True
        
        # Performance-based check
        if self.performance_monitor.significant_degradation():
            return True
        
        # Buffer utilization check (async-safe)
        try:
            loop = asyncio.get_event_loop()
            stats_future = asyncio.ensure_future(
                self.experience_replay.get_statistics()
            )
            stats = await asyncio.wait_for(stats_future, timeout=5.0)
            
            if stats['buffer_stats']['utilization'] > 0.9:
                return True
        except Exception as e:
            logger.debug(f"Could not check buffer stats: {e}")
        
        return False
    
    async def _should_mini_batch(self, interval: timedelta) -> bool:
        """Determine if mini-batch training is needed"""
        # Time-based check
        if (datetime.now() - self.last_mini_batch) > interval:
            return True
        
        # Experience count check
        try:
            stats = await asyncio.wait_for(
                self.experience_replay.get_statistics(),
                timeout=5.0
            )
            if stats['buffer_stats']['current_size'] > 500:
                return True
        except Exception:
            pass
        
        return False
    
    async def _schedule_retraining(self):
        """Schedule a retraining task with resource awareness"""
        task = LearningTask(
            task_id=f"retrain_{datetime.now().timestamp()}",
            task_type="retraining",
            data_source="experience_replay",
            priority=0.9,
            created_at=datetime.now(),
            metadata={
                'reason': 'scheduled',
                'load_factor': self.resource_monitor.get_load_factor()
            }
        )
        
        self.task_queue.append(task)
        logger.info(f"Scheduled retraining task: {task.task_id}")
    
    async def _schedule_mini_batch_training(self):
        """Schedule mini-batch training with adaptive parameters"""
        # Get adaptive batch size
        batch_size = self.adaptive_scheduler.get_batch_size(self.config.batch_size)
        
        task = LearningTask(
            task_id=f"mini_batch_{datetime.now().timestamp()}",
            task_type="fine_tuning", 
            data_source="experience_replay",
            priority=0.5,
            created_at=datetime.now(),
            metadata={
                'batch_size': batch_size,
                'sampling_method': 'prioritized',
                'adaptive': True
            }
        )
        
        self.task_queue.append(task)
        logger.debug(f"Scheduled mini-batch training: {task.task_id} (batch_size={batch_size})")
    
    async def _process_task_queue(self):
        """Process tasks with resource management"""
        if not self.task_queue:
            return
        
        # Sort by priority
        sorted_tasks = sorted(self.task_queue, key=lambda t: t.priority, reverse=True)
        
        # Find highest priority task that can execute
        task_to_execute = None
        for task in sorted_tasks:
            if self.adaptive_scheduler.should_execute_task(task.priority):
                task_to_execute = task
                break
        
        if not task_to_execute:
            logger.debug("No tasks eligible for execution due to system load")
            return
        
        # Remove from queue and add to active
        self.task_queue.remove(task_to_execute)
        self.active_tasks[task_to_execute.task_id] = task_to_execute
        
        try:
            # Execute with resource monitoring
            await self._execute_task_with_monitoring(task_to_execute)
            
            # Record success
            self.task_history.append({
                'task': task_to_execute,
                'status': 'completed',
                'timestamp': datetime.now()
            })
            
        except Exception as e:
            logger.error(f"Task {task_to_execute.task_id} failed: {e}")
            
            # Record failure
            self.task_history.append({
                'task': task_to_execute,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now()
            })
        
        finally:
            # Remove from active tasks
            self.active_tasks.pop(task_to_execute.task_id, None)
    
    async def _execute_task_with_monitoring(self, task: LearningTask):
        """Execute task with resource monitoring and throttling"""
        logger.info(f"Executing task {task.task_id} (type={task.task_type})")
        
        # Create a context manager for resource-limited execution
        @contextmanager
        def resource_limited_execution():
            # Set process nice value to lower priority
            original_nice = os.nice(0)
            try:
                os.nice(10)  # Lower priority
                yield
            finally:
                # Restore original priority
                os.nice(original_nice - os.nice(0))
        
        # Execute task with resource limits
        with resource_limited_execution():
            if task.task_type == "retraining":
                await self._perform_retraining_safe(task)
            elif task.task_type == "fine_tuning":
                await self._perform_fine_tuning_safe(task)
            elif task.task_type == "adaptation":
                await self._perform_adaptation_safe(task)
    
    async def _perform_retraining_safe(self, task: LearningTask):
        """Perform retraining with safety checks"""
        logger.info(f"Starting safe retraining: {task.task_id}")
        
        # Create checkpoint
        checkpoint_id = self.meta_framework.create_checkpoint()
        
        try:
            # Adaptive parameters
            num_epochs = min(self.config.num_epochs, 3)  # Limit epochs under load
            batch_size = self.adaptive_scheduler.get_batch_size(self.config.batch_size)
            
            for epoch in range(num_epochs):
                # Check if we should continue
                if not self.running or self.paused:
                    logger.info("Retraining interrupted")
                    break
                
                # Check resources before each epoch
                available, reason = self.resource_monitor.is_resource_available()
                if not available:
                    logger.warning(f"Stopping retraining: {reason}")
                    break
                
                logger.info(f"Retraining epoch {epoch + 1}/{num_epochs}")
                
                # Train for limited batches per epoch
                max_batches = 50 if self.resource_monitor.get_load_factor() > 0.5 else 100
                
                for batch_idx in range(max_batches):
                    # Throttle based on resources
                    await self._throttle_if_needed()
                    
                    # Sample batch
                    replay_batch = await self.experience_replay.sample_batch(
                        batch_size,
                        method='prioritized'
                    )
                    
                    if not replay_batch.experiences:
                        break
                    
                    # Train step
                    training_batch = self._prepare_training_batch(replay_batch)
                    metrics = self.meta_framework.train_step(
                        training_batch,
                        task_id='retraining'
                    )
                    
                    # Update priorities
                    td_errors = [metrics['loss']] * len(replay_batch.experiences)
                    experience_ids = [exp.experience_id for exp in replay_batch.experiences]
                    self.experience_replay.update_priorities(experience_ids, td_errors)
                    
                    # Adaptive learning rate
                    self.lr_adjuster.update(metrics)
                    self._update_learning_rate()
                
                # Evaluate
                eval_metrics = await self._evaluate_model()
                self.performance_monitor.record(eval_metrics)
                
                # Early stopping
                if self.performance_monitor.should_early_stop():
                    logger.info("Early stopping triggered")
                    break
            
            # Update last retraining time
            self.last_retraining = datetime.now()
            logger.info(f"Completed safe retraining: {task.task_id}")
            
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            # Restore from checkpoint on failure
            self.meta_framework.restore_checkpoint(checkpoint_id)
            raise
    
    async def _perform_fine_tuning_safe(self, task: LearningTask):
        """Perform fine-tuning with safety checks"""
        logger.debug(f"Starting safe fine-tuning: {task.task_id}")
        
        batch_size = task.metadata.get('batch_size', 128)
        num_batches = min(task.metadata.get('num_batches', 10), 20)  # Limit batches
        
        for i in range(num_batches):
            # Throttle if needed
            await self._throttle_if_needed()
            
            # Sample batch
            replay_batch = await self.experience_replay.sample_batch(
                batch_size,
                method=task.metadata.get('sampling_method', 'prioritized')
            )
            
            if not replay_batch.experiences:
                continue
            
            # Train
            training_batch = self._prepare_training_batch(replay_batch)
            metrics = self.meta_framework.train_step(
                training_batch,
                task_id='fine_tuning'
            )
            
            self.performance_monitor.record(metrics)
        
        self.last_mini_batch = datetime.now()
        logger.debug(f"Completed safe fine-tuning: {task.task_id}")
    
    async def _perform_adaptation_safe(self, task: LearningTask):
        """Perform adaptation with safety checks"""
        logger.info(f"Starting safe adaptation: {task.task_id}")
        
        context = task.metadata.get('context', {})
        self.meta_framework.adapt(context)
        
        logger.info(f"Completed safe adaptation: {task.task_id}")
    
    async def _throttle_if_needed(self):
        """Throttle execution based on system resources"""
        load_factor = self.resource_monitor.get_load_factor()
        
        # Progressive throttling
        if load_factor > 0.9:
            await asyncio.sleep(1.0)  # Heavy throttle
        elif load_factor > 0.7:
            await asyncio.sleep(0.5)  # Medium throttle
        elif load_factor > 0.5:
            await asyncio.sleep(0.1)  # Light throttle
    
    def _prepare_training_batch(self, replay_batch: ReplayBatch) -> Dict[str, torch.Tensor]:
        """Convert replay batch to training batch"""
        inputs = []
        targets = []
        
        for exp in replay_batch.experiences:
            inputs.append(exp.command_embedding)
            target = hash(exp.intent) % 100  # Simple mapping
            targets.append(target)
        
        return {
            'input': torch.tensor(np.array(inputs), dtype=torch.float32),
            'target': torch.tensor(targets, dtype=torch.long)
        }
    
    async def _evaluate_model(self) -> Dict[str, float]:
        """Evaluate model performance"""
        eval_batch = await self.experience_replay.sample_batch(256, method='recent')
        
        if not eval_batch.experiences:
            return {'accuracy': 0.0, 'loss': float('inf')}
        
        batch = self._prepare_training_batch(eval_batch)
        
        self.model.eval()
        with torch.no_grad():
            output = self.model(batch['input'])
            loss = nn.functional.cross_entropy(output, batch['target'])
            predictions = output.argmax(dim=1)
            accuracy = (predictions == batch['target']).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'loss': loss.item(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_learning_rate(self):
        """Update optimizer learning rate"""
        new_lr = self.lr_adjuster.get_adjusted_lr()
        for param_group in self.meta_framework.optimizer.param_groups:
            param_group['lr'] = new_lr
    
    async def _perform_maintenance(self):
        """Perform periodic maintenance tasks"""
        # Clean up old task history
        if len(self.task_history) > 900:
            # Keep most recent 800 entries
            self.task_history = deque(list(self.task_history)[-800:], maxlen=1000)
        
        # Trigger experience compression if needed
        stats = await self.experience_replay.get_statistics()
        if stats['buffer_stats']['utilization'] > 0.8:
            logger.debug("Triggering experience compression")
            # This would trigger compression in experience replay
    
    def pause(self):
        """Pause learning temporarily"""
        self.paused = True
        logger.info("Continuous learning paused")
    
    def resume(self):
        """Resume learning"""
        self.paused = False
        self.health_status['consecutive_failures'] = 0
        logger.info("Continuous learning resumed")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        resources = self.resource_monitor.get_current_resources()
        load_factor = self.resource_monitor.get_load_factor()
        
        # Get task stats
        completed_tasks = len([t for t in self.task_history if t.get('status') == 'completed'])
        failed_tasks = len([t for t in self.task_history if t.get('status') == 'failed'])
        
        return {
            'status': 'healthy' if self.health_status['healthy'] else 'unhealthy',
            'running': self.running,
            'paused': self.paused,
            'health': self.health_status,
            'timing': {
                'last_retraining': self.last_retraining.isoformat(),
                'last_mini_batch': self.last_mini_batch.isoformat(),
                'last_health_check': self.last_health_check.isoformat()
            },
            'tasks': {
                'queued': len(self.task_queue),
                'active': len(self.active_tasks),
                'completed': completed_tasks,
                'failed': failed_tasks
            },
            'resources': {
                'cpu_percent': resources.cpu_percent,
                'memory_percent': resources.memory_percent,
                'available_memory_mb': resources.available_memory_mb,
                'load_factor': load_factor,
                'throttled': load_factor > self.config.load_factor_threshold
            },
            'configuration': {
                'max_cpu_percent': self.config.max_cpu_percent,
                'max_memory_percent': self.config.max_memory_percent,
                'adaptive_scheduling': self.config.enable_adaptive_scheduling
            },
            'performance': self.performance_monitor.get_summary()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the learning system"""
        logger.info("Shutting down Robust Advanced Continuous Learning")
        
        self.running = False
        self._shutdown_event.set()
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            logger.info(f"Cancelling active task: {task_id}")
        
        # Save state
        state_path = Path("backend/data/robust_learning_state.pt")
        state_path.parent.mkdir(exist_ok=True)
        
        try:
            torch.save({
                'model_state': self.model.state_dict(),
                'meta_state': self.meta_framework.get_state(),
                'task_history': list(self.task_history)[-100:],  # Last 100 tasks
                'health_status': self.health_status,
                'config': self.config
            }, state_path)
            logger.info(f"Saved learning state to {state_path}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
        
        # Shutdown components
        await self.experience_replay.shutdown()
        self.executor.shutdown(wait=True)
        
        logger.info("Robust Advanced Continuous Learning shutdown complete")


# Helper classes already imported at the top


# Factory function
def get_robust_continuous_learning(model: nn.Module, config: Optional[LearningConfig] = None) -> RobustAdvancedContinuousLearning:
    """Get robust continuous learning instance"""
    return RobustAdvancedContinuousLearning(model, config)