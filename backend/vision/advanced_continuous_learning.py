#!/usr/bin/env python3
"""
Advanced Continuous Learning System for Vision System v2.0
This now uses the robust implementation if available
"""

# Try to use robust implementation first
_ROBUST_AVAILABLE = False
# Disabled until integrate_robust_learning.py is created
# try:
#     # Import only the essentials to avoid circular import
#     from .integrate_robust_learning import (
#         get_advanced_continuous_learning as _get_robust,
#         LearningTask as _RobustLearningTask,
#         FederatedUpdate as _RobustFederatedUpdate,
#     )
#     _ROBUST_AVAILABLE = True
# except ImportError:
#     # Continue with original implementation below
#     pass

# If robust is available, use those imports
if _ROBUST_AVAILABLE:
    get_advanced_continuous_learning = _get_robust
    LearningTask = _RobustLearningTask
    FederatedUpdate = _RobustFederatedUpdate
    
    # We'll define the other classes below to avoid circular imports
    # Set __all__ and skip some definitions
    __all__ = [
        'get_advanced_continuous_learning',
        'AdvancedContinuousLearning',
        'LearningTask',
        'FederatedUpdate', 
        'AdaptiveLearningRateAdjuster',
        'PerformanceMonitor'
    ]
else:
    # Continue with original implementation
    pass

# Only define original implementation if robust is not available
if not _ROBUST_AVAILABLE:
    import torch
    import torch.nn as nn
    import numpy as np
    from typing import Dict, List, Optional, Any, Tuple
    from dataclasses import dataclass, field
    from datetime import datetime, timedelta
    import logging
    from pathlib import Path
    import json
    import asyncio
    from collections import defaultdict, deque
    import threading
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    import time

    from .experience_replay_system import get_experience_replay_system, Experience, ReplayBatch
    from .meta_learning_framework import get_meta_learning_framework, MetaLearningFramework
    from .continuous_learning_pipeline import OnlineLearner

    logger = logging.getLogger(__name__)

    @dataclass
    class LearningTask:
        """Represents a learning task"""
        task_id: str
        task_type: str  # retraining, fine_tuning, adaptation
        data_source: str  # experience_replay, current_batch, federated
        priority: float
        created_at: datetime
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class FederatedUpdate:
        """Update from federated learning"""
        update_id: str
        source_id: str  # Anonymous user/device ID
        model_updates: Dict[str, torch.Tensor]
        performance_metrics: Dict[str, float]
        data_statistics: Dict[str, Any]
        timestamp: datetime

    class AdvancedContinuousLearning:
        """
        Advanced continuous learning system with:
        - Experience replay integration
        - Meta-learning framework
        - Periodic retraining
        - Distributed/federated learning support
        - Privacy preservation
        - Auto learning rate adjustment
        """
        
        def __init__(self, model: nn.Module):
            # Core components
            self.model = model
            self.experience_replay = get_experience_replay_system()
            self.meta_framework = get_meta_learning_framework(model)
            
            # Learning configuration
            self.retraining_interval = timedelta(hours=6)
            self.last_retraining = datetime.now()
            self.mini_batch_interval = timedelta(minutes=15)
            self.last_mini_batch = datetime.now()
            
            # Task queue
            self.task_queue: deque = deque()
            self.active_tasks: Dict[str, LearningTask] = {}
            
            # Federated learning
            self.federated_enabled = False
            self.federated_updates: deque = deque(maxlen=1000)
            self.privacy_budget = 1.0  # Differential privacy budget
            
            # Learning rate adjustment
            self.lr_adjuster = AdaptiveLearningRateAdjuster()
            
            # Performance monitoring
            self.performance_monitor = PerformanceMonitor()
            
            # Threading
            self.executor = ThreadPoolExecutor(max_workers=4)
            self.learning_thread: Optional[threading.Thread] = None
            self.running = True
            
            # Start continuous learning (unless disabled)
            import os
            if os.getenv('DISABLE_CONTINUOUS_LEARNING', '').lower() != 'true':
                self._start_continuous_learning()
                logger.info("Advanced Continuous Learning System initialized")
            else:
                self.running = False
                logger.warning("Continuous Learning DISABLED via environment variable")
        
        def _start_continuous_learning(self):
            """Start background learning threads"""
            def learning_loop():
                # Create a new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                while self.running:
                    try:
                        # Run the learning cycle in the thread's event loop
                        loop.run_until_complete(self._learning_cycle())
                        time.sleep(60)  # Use time.sleep, not asyncio.sleep
                    except Exception as e:
                        logger.error(f"Learning cycle error: {e}")
                        time.sleep(300)  # Wait 5 minutes on error
                
                loop.close()
            
            self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
            self.learning_thread.start()
        
        async def _learning_cycle(self):
            """Main learning cycle"""
            # Check if retraining is needed
            if self._should_retrain():
                await self._schedule_retraining()
            
            # Check if mini-batch training is needed
            elif self._should_mini_batch():
                await self._schedule_mini_batch_training()
            
            # Process federated updates if available
            if self.federated_enabled and self.federated_updates:
                await self._process_federated_updates()
            
            # Process task queue
            await self._process_task_queue()
            
            # Adapt if needed
            if self.meta_framework.should_adapt():
                await self._adapt_learning()
        
        def _should_retrain(self) -> bool:
            """Determine if full retraining is needed"""
            # Time-based retraining
            if (datetime.now() - self.last_retraining) > self.retraining_interval:
                return True
            
            # Performance-based retraining
            if self.performance_monitor.significant_degradation():
                return True
            
            # Experience buffer is full
            stats = asyncio.run(self.experience_replay.get_statistics())
            if stats['buffer_stats']['utilization'] > 0.9:
                return True
            
            return False
        
        def _should_mini_batch(self) -> bool:
            """Determine if mini-batch training is needed"""
            # Time-based
            if (datetime.now() - self.last_mini_batch) > self.mini_batch_interval:
                return True
            
            # Experience count based
            stats = asyncio.run(self.experience_replay.get_statistics())
            if stats['buffer_stats']['current_size'] > 500:
                return True
            
            return False
        
        async def _schedule_retraining(self):
            """Schedule a full retraining task"""
            task = LearningTask(
                task_id=f"retrain_{datetime.now().timestamp()}",
                task_type="retraining",
                data_source="experience_replay",
                priority=0.9,
                created_at=datetime.now(),
                metadata={
                    'reason': 'scheduled',
                    'buffer_size': len(self.experience_replay.replay_buffer.buffer)
                }
            )
            
            self.task_queue.append(task)
            logger.info(f"Scheduled retraining task: {task.task_id}")
        
        async def _schedule_mini_batch_training(self):
            """Schedule mini-batch training"""
            task = LearningTask(
                task_id=f"mini_batch_{datetime.now().timestamp()}",
                task_type="fine_tuning",
                data_source="experience_replay",
                priority=0.5,
                created_at=datetime.now(),
                metadata={
                    'batch_size': 256,
                    'sampling_method': 'prioritized'
                }
            )
            
            self.task_queue.append(task)
            logger.debug(f"Scheduled mini-batch training: {task.task_id}")
        
        async def _process_task_queue(self):
            """Process learning tasks from queue"""
            if not self.task_queue:
                return
            
            # Sort by priority
            sorted_queue = sorted(self.task_queue, key=lambda t: t.priority, reverse=True)
            
            # Process highest priority task
            task = sorted_queue[0]
            self.task_queue.remove(task)
            
            # Add to active tasks
            self.active_tasks[task.task_id] = task
            
            try:
                if task.task_type == "retraining":
                    await self._perform_retraining(task)
                elif task.task_type == "fine_tuning":
                    await self._perform_fine_tuning(task)
                elif task.task_type == "adaptation":
                    await self._perform_adaptation(task)
                
                # Remove from active tasks
                del self.active_tasks[task.task_id]
                
            except Exception as e:
                logger.error(f"Task {task.task_id} failed: {e}")
                del self.active_tasks[task.task_id]
        
        async def _perform_retraining(self, task: LearningTask):
            """Perform full model retraining"""
            logger.info(f"Starting retraining: {task.task_id}")
            
            # Create checkpoint before retraining
            checkpoint_id = self.meta_framework.create_checkpoint()
            
            # Select learning strategy
            strategy = self.meta_framework.select_learning_strategy({
                'task_type': 'retraining',
                'data_size': 'large'
            })
            
            # Prepare data
            num_epochs = 5
            batch_size = strategy.hyperparameters['batch_size']
            
            for epoch in range(num_epochs):
                logger.info(f"Retraining epoch {epoch + 1}/{num_epochs}")
                
                # Sample diverse batches
                for _ in range(100):  # 100 batches per epoch
                    # Get batch from experience replay
                    replay_batch = await self.experience_replay.sample_batch(
                        batch_size,
                        method='prioritized'
                    )
                    
                    if not replay_batch.experiences:
                        continue
                    
                    # Convert to training batch
                    training_batch = self._prepare_training_batch(replay_batch)
                    
                    # Train step
                    metrics = self.meta_framework.train_step(training_batch, task_id='retraining')
                    
                    # Update priorities based on loss
                    td_errors = [metrics['loss']] * len(replay_batch.experiences)
                    experience_ids = [exp.experience_id for exp in replay_batch.experiences]
                    self.experience_replay.update_priorities(experience_ids, td_errors)
                    
                    # Adjust learning rate
                    self.lr_adjuster.update(metrics)
                    new_lr = self.lr_adjuster.get_adjusted_lr()
                    for param_group in self.meta_framework.optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                # Evaluate after each epoch
                eval_metrics = await self._evaluate_model()
                self.performance_monitor.record(eval_metrics)
                
                # Early stopping
                if self.performance_monitor.should_early_stop():
                    logger.info("Early stopping triggered")
                    break
            
            # Update Fisher Information for EWC
            data_loader = self._create_data_loader(batch_size)
            self.meta_framework.forgetting_prevention.compute_fisher_information(
                data_loader,
                task_id='retraining',
                num_samples=500
            )
            
            self.last_retraining = datetime.now()
            logger.info(f"Completed retraining: {task.task_id}")
        
        async def _perform_fine_tuning(self, task: LearningTask):
            """Perform mini-batch fine-tuning"""
            logger.debug(f"Starting fine-tuning: {task.task_id}")
            
            batch_size = task.metadata.get('batch_size', 128)
            num_batches = task.metadata.get('num_batches', 10)
            
            for i in range(num_batches):
                # Sample batch
                replay_batch = await self.experience_replay.sample_batch(
                    batch_size,
                    method=task.metadata.get('sampling_method', 'prioritized')
                )
                
                if not replay_batch.experiences:
                    continue
                
                # Convert to training batch
                training_batch = self._prepare_training_batch(replay_batch)
                
                # Train step
                metrics = self.meta_framework.train_step(
                    training_batch,
                    task_id='fine_tuning'
                )
                
                # Update performance
                self.performance_monitor.record(metrics)
            
            self.last_mini_batch = datetime.now()
            logger.debug(f"Completed fine-tuning: {task.task_id}")
        
        async def _perform_adaptation(self, task: LearningTask):
            """Perform rapid adaptation for new patterns"""
            logger.info(f"Starting adaptation: {task.task_id}")
            
            # This would implement MAML-style adaptation
            # For now, delegate to meta-framework
            context = task.metadata.get('context', {})
            self.meta_framework.adapt(context)
            
            logger.info(f"Completed adaptation: {task.task_id}")
        
        def _prepare_training_batch(self, replay_batch: ReplayBatch) -> Dict[str, torch.Tensor]:
            """Convert replay batch to training batch"""
            inputs = []
            targets = []
            
            for exp in replay_batch.experiences:
                # Use command embedding as input
                inputs.append(exp.command_embedding)
                
                # Create target (intent as class index)
                # In real implementation, this would map intents to indices
                target = hash(exp.intent) % 100  # Simple mapping
                targets.append(target)
            
            return {
                'input': torch.tensor(np.array(inputs), dtype=torch.float32),
                'target': torch.tensor(targets, dtype=torch.long)
            }
        
        async def _evaluate_model(self) -> Dict[str, float]:
            """Evaluate model performance"""
            # Sample evaluation batch
            eval_batch = await self.experience_replay.sample_batch(
                256,
                method='recent'
            )
            
            if not eval_batch.experiences:
                return {'accuracy': 0.0, 'loss': float('inf')}
            
            # Prepare batch
            batch = self._prepare_training_batch(eval_batch)
            
            # Evaluate
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
        
        def _create_data_loader(self, batch_size: int):
            """Create data loader for Fisher Information computation"""
            # Simplified data loader that samples from experience replay
            class ReplayDataset:
                def __init__(self, replay_system):
                    self.replay_system = replay_system
                
                def __len__(self):
                    return len(self.replay_system.replay_buffer.buffer)
                
                def __getitem__(self, idx):
                    exp = self.replay_system.replay_buffer.buffer[idx]
                    return {
                        'input': torch.tensor(exp.command_embedding, dtype=torch.float32),
                        'target': torch.tensor(hash(exp.intent) % 100, dtype=torch.long)
                    }
            
            dataset = ReplayDataset(self.experience_replay)
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True
            )
        
        async def _adapt_learning(self):
            """Trigger learning adaptation"""
            # Analyze current patterns
            patterns = self.experience_replay.patterns
            
            context = {
                'num_patterns': len(patterns),
                'pattern_types': defaultdict(int)
            }
            
            for pattern in patterns.values():
                context['pattern_types'][pattern.pattern_type] += 1
            
            # Adapt
            self.meta_framework.adapt(context)
        
        async def add_federated_update(self, update: FederatedUpdate):
            """Add federated learning update"""
            if not self.federated_enabled:
                return
            
            # Verify privacy budget
            if self.privacy_budget <= 0:
                logger.warning("Privacy budget exhausted, ignoring federated update")
                return
            
            # Add noise for differential privacy
            noisy_update = self._add_differential_privacy_noise(update)
            
            self.federated_updates.append(noisy_update)
            
            # Update privacy budget
            self.privacy_budget -= 0.01  # Simple linear budget
            
            logger.debug(f"Added federated update: {update.update_id}")
        
        def _add_differential_privacy_noise(self, update: FederatedUpdate) -> FederatedUpdate:
            """Add differential privacy noise to federated update"""
            epsilon = 1.0  # Privacy parameter
            sensitivity = 0.1  # Estimated sensitivity
            
            # Add Laplacian noise to model updates
            noisy_updates = {}
            for param_name, param_value in update.model_updates.items():
                noise_scale = sensitivity / epsilon
                noise = torch.tensor(
                    np.random.laplace(0, noise_scale, param_value.shape),
                    dtype=param_value.dtype
                )
                noisy_updates[param_name] = param_value + noise
            
            # Create noisy update
            noisy_update = FederatedUpdate(
                update_id=update.update_id,
                source_id=hashlib.sha256(update.source_id.encode()).hexdigest()[:8],  # Anonymize
                model_updates=noisy_updates,
                performance_metrics=update.performance_metrics,
                data_statistics={},  # Remove detailed statistics
                timestamp=update.timestamp
            )
            
            return noisy_update
        
        async def _process_federated_updates(self):
            """Process accumulated federated updates"""
            if len(self.federated_updates) < 10:  # Need minimum updates
                return
            
            logger.info(f"Processing {len(self.federated_updates)} federated updates")
            
            # Aggregate updates using federated averaging
            aggregated_updates = {}
            
            # Get recent updates
            recent_updates = list(self.federated_updates)[-50:]
            
            # Average model updates
            for param_name in self.model.state_dict().keys():
                param_updates = []
                for update in recent_updates:
                    if param_name in update.model_updates:
                        param_updates.append(update.model_updates[param_name])
                
                if param_updates:
                    aggregated_updates[param_name] = torch.stack(param_updates).mean(dim=0)
            
            # Apply aggregated updates
            if aggregated_updates:
                # Weighted combination with current model
                alpha = 0.1  # Federation weight
                current_state = self.model.state_dict()
                
                for param_name, aggregated_value in aggregated_updates.items():
                    if param_name in current_state:
                        current_state[param_name] = (
                            (1 - alpha) * current_state[param_name] +
                            alpha * aggregated_value
                        )
                
                self.model.load_state_dict(current_state)
                logger.info("Applied federated updates to model")
            
            # Clear processed updates
            self.federated_updates.clear()
        
        async def record_interaction(
            self,
            command: str,
            command_embedding: np.ndarray,
            intent: str,
            confidence: float,
            handler: str,
            response: str,
            success: bool,
            latency_ms: float,
            user_id: Optional[str] = None,
            context: Optional[Dict[str, Any]] = None,
            feedback: Optional[Dict[str, Any]] = None
        ):
            """Record interaction in experience replay"""
            experience_id = await self.experience_replay.add_experience(
                command=command,
                command_embedding=command_embedding,
                intent=intent,
                confidence=confidence,
                handler=handler,
                response=response,
                success=success,
                latency_ms=latency_ms,
                user_id=user_id,
                context=context,
                feedback=feedback
            )
            
            logger.debug(f"Recorded experience: {experience_id}")
        
        def get_status(self) -> Dict[str, Any]:
            """Get comprehensive status"""
            replay_stats = asyncio.run(self.experience_replay.get_statistics())
            meta_metrics = self.meta_framework.get_metrics()
            
            return {
                'continuous_learning': {
                    'last_retraining': self.last_retraining.isoformat(),
                    'last_mini_batch': self.last_mini_batch.isoformat(),
                    'active_tasks': len(self.active_tasks),
                    'queued_tasks': len(self.task_queue)
                },
                'experience_replay': replay_stats,
                'meta_learning': meta_metrics,
                'federated_learning': {
                    'enabled': self.federated_enabled,
                    'updates_pending': len(self.federated_updates),
                    'privacy_budget': self.privacy_budget
                },
                'performance': self.performance_monitor.get_summary(),
                'learning_rate': self.lr_adjuster.current_lr
            }
        
        async def shutdown(self):
            """Shutdown the continuous learning system"""
            logger.info("Shutting down Advanced Continuous Learning")
            
            self.running = False
            
            # Wait for threads
            if self.learning_thread:
                self.learning_thread.join(timeout=5)
            
            # Save state
            state_path = Path("backend/data/advanced_learning_state.pt")
            self.meta_framework.save_state(state_path)
            
            # Shutdown components
            await self.experience_replay.shutdown()
            
            self.executor.shutdown(wait=True)
            
            logger.info("Advanced Continuous Learning shutdown complete")

    class AdaptiveLearningRateAdjuster:
        """Automatically adjusts learning rate based on performance"""
        
        def __init__(self):
            self.current_lr = 0.001
            self.loss_history = deque(maxlen=100)
            self.patience = 10
            self.cooldown = 5
            self.factor = 0.5
            self.min_lr = 1e-6
            
            self.best_loss = float('inf')
            self.patience_counter = 0
            self.cooldown_counter = 0
        
        def update(self, metrics: Dict[str, float]):
            """Update based on training metrics"""
            loss = metrics.get('loss', 0.0)
            self.loss_history.append(loss)
            
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return
            
            # Check if loss is improving
            if loss < self.best_loss * 0.99:  # 1% improvement threshold
                self.best_loss = loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Reduce learning rate if patience exceeded
            if self.patience_counter >= self.patience:
                self.current_lr = max(self.min_lr, self.current_lr * self.factor)
                self.patience_counter = 0
                self.cooldown_counter = self.cooldown
                logger.info(f"Reduced learning rate to {self.current_lr}")
        
        def get_adjusted_lr(self) -> float:
            """Get current adjusted learning rate"""
            # Additional adjustments based on loss variance
            if len(self.loss_history) >= 20:
                recent_variance = np.var(list(self.loss_history)[-20:])
                if recent_variance > 0.1:  # High variance
                    return self.current_lr * 0.9  # Slightly reduce
            
            return self.current_lr

    class PerformanceMonitor:
        """Monitor learning performance for decision making"""
        
        def __init__(self):
            self.metrics_history = deque(maxlen=1000)
            self.baseline_performance = None
            self.degradation_threshold = 0.1  # 10% degradation
            self.early_stop_patience = 5
            self.no_improvement_counter = 0
            
        def record(self, metrics: Dict[str, float]):
            """Record performance metrics"""
            self.metrics_history.append({
                'timestamp': datetime.now(),
                'metrics': metrics
            })
            
            # Update baseline
            if self.baseline_performance is None:
                self.baseline_performance = metrics.get('accuracy', 0.0)
            elif metrics.get('accuracy', 0.0) > self.baseline_performance:
                self.baseline_performance = metrics.get('accuracy', 0.0)
                self.no_improvement_counter = 0
            else:
                self.no_improvement_counter += 1
        
        def significant_degradation(self) -> bool:
            """Check if there's significant performance degradation"""
            if not self.metrics_history or self.baseline_performance is None:
                return False
            
            recent_accuracy = np.mean([
                h['metrics'].get('accuracy', 0.0)
                for h in list(self.metrics_history)[-10:]
            ])
            
            degradation = (self.baseline_performance - recent_accuracy) / self.baseline_performance
            
            return degradation > self.degradation_threshold
        
        def should_early_stop(self) -> bool:
            """Determine if training should stop early"""
            return self.no_improvement_counter >= self.early_stop_patience
        
        def get_summary(self) -> Dict[str, Any]:
            """Get performance summary"""
            if not self.metrics_history:
                return {}
            
            recent_metrics = [h['metrics'] for h in list(self.metrics_history)[-50:]]
            
            return {
                'baseline_accuracy': self.baseline_performance,
                'recent_accuracy': np.mean([m.get('accuracy', 0) for m in recent_metrics]),
                'recent_loss': np.mean([m.get('loss', 0) for m in recent_metrics]),
                'no_improvement_steps': self.no_improvement_counter,
                'total_recordings': len(self.metrics_history)
            }

    # Singleton instance
    _advanced_learning: Optional[AdvancedContinuousLearning] = None

    def get_advanced_continuous_learning(model: nn.Module) -> AdvancedContinuousLearning:
        """Get singleton instance of advanced continuous learning"""
        global _advanced_learning
        if _advanced_learning is None:
            _advanced_learning = AdvancedContinuousLearning(model)
        return _advanced_learning
    
    # Set __all__ for original implementation
    __all__ = [
        'get_advanced_continuous_learning',
        'AdvancedContinuousLearning',
        'LearningTask',
        'FederatedUpdate', 
        'AdaptiveLearningRateAdjuster',
        'PerformanceMonitor'
    ]