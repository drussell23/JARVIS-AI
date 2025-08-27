#!/usr/bin/env python3
"""
Continuous Learning Pipeline for Vision System v2.0
Implements online learning, model updates, and performance monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import json
import pickle
from collections import defaultdict, deque
import threading
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class LearningEvent:
    """Single learning event in the pipeline"""
    event_id: str
    timestamp: datetime
    event_type: str  # command, feedback, performance, error
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    
@dataclass
class ModelCheckpoint:
    """Model checkpoint for versioning and rollback"""
    version: str
    timestamp: datetime
    metrics: Dict[str, float]
    model_state: Dict[str, Any]
    metadata: Dict[str, Any]
    

@dataclass 
class PerformanceWindow:
    """Time window for performance analysis"""
    start_time: datetime
    end_time: datetime
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)


class OnlineLearner(nn.Module):
    """Neural network that supports online learning"""
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        # For online learning
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.loss_history = deque(maxlen=1000)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.output_layer(features)
        return output, features
    
    def online_update(self, input_batch, target_batch, learning_rate=None):
        """Perform online update with a batch of examples"""
        if learning_rate:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        
        self.train()
        self.optimizer.zero_grad()
        
        output, _ = self.forward(input_batch)
        loss = nn.functional.mse_loss(output, target_batch)
        
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        return loss.item()


class ContinuousLearningPipeline:
    """
    Production-grade continuous learning pipeline
    Features:
    - Real-time model updates
    - A/B testing for model versions
    - Automatic rollback on performance degradation
    - Distributed learning support
    """
    
    def __init__(self):
        # Learning components
        self.online_learner = OnlineLearner()
        self.learning_buffer = deque(maxlen=10000)
        self.feedback_buffer = deque(maxlen=5000)
        
        # Model versioning
        self.current_version = "v2.0.0"
        self.model_checkpoints: Dict[str, ModelCheckpoint] = {}
        self.active_models: Dict[str, OnlineLearner] = {
            'production': self.online_learner,
            'candidate': None
        }
        
        # Performance monitoring
        self.performance_windows: deque = deque(maxlen=100)
        self.current_window: Optional[PerformanceWindow] = None
        self.window_duration = timedelta(minutes=15)
        
        # A/B testing
        self.ab_test_active = False
        self.ab_test_split = 0.1  # 10% to candidate model
        self.ab_test_results = defaultdict(lambda: {
            'requests': 0,
            'successes': 0,
            'latencies': []
        })
        
        # Learning configuration
        self.learning_config = {
            'batch_size': 32,
            'update_frequency': 60,  # seconds
            'min_examples_for_update': 50,
            'performance_threshold': 0.95,  # Rollback if performance drops below this
            'learning_rate_schedule': {
                'initial': 0.001,
                'decay': 0.95,
                'min_lr': 0.0001
            }
        }
        
        # Threading for background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.learning_thread: Optional[threading.Thread] = None
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        # Initialize pipeline
        self._initialize_pipeline()
        
        logger.info("Continuous Learning Pipeline initialized")
    
    def _initialize_pipeline(self):
        """Initialize all pipeline components"""
        # Load existing checkpoints
        self._load_checkpoints()
        
        # Start monitoring window
        self._start_new_performance_window()
        
        # Start background threads
        self.running = True
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background learning and monitoring tasks"""
        # Learning thread
        def learning_loop():
            while self.running:
                try:
                    self._process_learning_cycle()
                    time.sleep(self.learning_config['update_frequency'])
                except Exception as e:
                    logger.error(f"Learning cycle error: {e}")
                    time.sleep(300)  # Wait 5 minutes on error
        
        # Monitoring thread
        def monitoring_loop():
            while self.running:
                try:
                    self._update_performance_monitoring()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
        
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        
        self.learning_thread.start()
        self.monitoring_thread.start()
    
    async def record_learning_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ):
        """Record a learning event in the pipeline"""
        event = LearningEvent(
            event_id=f"{event_type}_{datetime.now().timestamp()}",
            timestamp=datetime.now(),
            event_type=event_type,
            data=data,
            user_id=user_id,
            session_id=session_id
        )
        
        # Route to appropriate buffer
        if event_type == 'feedback':
            self.feedback_buffer.append(event)
        else:
            self.learning_buffer.append(event)
        
        # Update performance metrics
        if event_type == 'command':
            self._update_command_metrics(data)
    
    def _update_command_metrics(self, data: Dict[str, Any]):
        """Update performance metrics from command data"""
        if self.current_window:
            self.current_window.total_requests += 1
            
            if data.get('success', False):
                self.current_window.successful_requests += 1
            else:
                self.current_window.failed_requests += 1
                
                # Track error type
                error_type = data.get('error_type', 'unknown')
                self.current_window.error_types[error_type] = \
                    self.current_window.error_types.get(error_type, 0) + 1
            
            # Update latency
            if 'latency_ms' in data:
                # Simple running average
                prev_avg = self.current_window.avg_latency_ms
                n = self.current_window.total_requests
                self.current_window.avg_latency_ms = \
                    (prev_avg * (n - 1) + data['latency_ms']) / n
    
    def _process_learning_cycle(self):
        """Process one learning cycle"""
        # Check if we have enough examples
        if len(self.learning_buffer) < self.learning_config['min_examples_for_update']:
            return
        
        logger.info(f"Processing learning cycle with {len(self.learning_buffer)} examples")
        
        # Prepare training batch
        batch = self._prepare_training_batch()
        
        if not batch:
            return
        
        # Perform online learning
        self._perform_online_learning(batch)
        
        # Evaluate performance
        performance = self._evaluate_current_performance()
        
        # Decide on model update
        if performance['success_rate'] > self.learning_config['performance_threshold']:
            # Performance is good, consider updating production
            self._consider_model_update()
        else:
            # Performance degraded, consider rollback
            self._consider_rollback(performance)
    
    def _prepare_training_batch(self) -> Optional[Dict[str, torch.Tensor]]:
        """Prepare a training batch from learning buffer"""
        batch_size = min(
            self.learning_config['batch_size'],
            len(self.learning_buffer)
        )
        
        if batch_size == 0:
            return None
        
        # Sample from buffer
        batch_events = [self.learning_buffer.popleft() for _ in range(batch_size)]
        
        # Extract features and targets
        inputs = []
        targets = []
        
        for event in batch_events:
            if event.event_type == 'command':
                # Extract command embedding and routing result
                input_features = event.data.get('embedding', np.zeros(768))
                target_features = event.data.get('route_embedding', np.zeros(128))
                
                inputs.append(input_features)
                targets.append(target_features)
        
        if not inputs:
            return None
        
        return {
            'inputs': torch.tensor(np.array(inputs), dtype=torch.float32),
            'targets': torch.tensor(np.array(targets), dtype=torch.float32),
            'metadata': [e.data for e in batch_events]
        }
    
    def _perform_online_learning(self, batch: Dict[str, torch.Tensor]):
        """Perform online learning update"""
        # Calculate adaptive learning rate
        current_lr = self._calculate_learning_rate()
        
        # Update model
        loss = self.online_learner.online_update(
            batch['inputs'],
            batch['targets'],
            learning_rate=current_lr
        )
        
        logger.info(f"Online learning update completed. Loss: {loss:.4f}, LR: {current_lr:.6f}")
        
        # Track learning progress
        self._track_learning_progress(loss, batch['metadata'])
    
    def _calculate_learning_rate(self) -> float:
        """Calculate adaptive learning rate based on performance"""
        base_lr = self.learning_config['learning_rate_schedule']['initial']
        decay = self.learning_config['learning_rate_schedule']['decay']
        min_lr = self.learning_config['learning_rate_schedule']['min_lr']
        
        # Decay based on number of updates
        n_updates = len(self.online_learner.loss_history)
        lr = base_lr * (decay ** (n_updates / 1000))
        
        # Adjust based on recent performance
        if self.current_window and self.current_window.total_requests > 100:
            success_rate = self.current_window.successful_requests / self.current_window.total_requests
            
            # Reduce learning rate if performance is good
            if success_rate > 0.95:
                lr *= 0.5
            # Increase learning rate if performance is poor
            elif success_rate < 0.8:
                lr *= 2.0
        
        return max(min_lr, min(base_lr, lr))
    
    def _evaluate_current_performance(self) -> Dict[str, float]:
        """Evaluate current model performance"""
        if not self.current_window or self.current_window.total_requests == 0:
            return {'success_rate': 1.0, 'avg_latency': 0.0}
        
        success_rate = self.current_window.successful_requests / self.current_window.total_requests
        
        return {
            'success_rate': success_rate,
            'avg_latency': self.current_window.avg_latency_ms,
            'total_requests': self.current_window.total_requests,
            'error_rate': self.current_window.failed_requests / self.current_window.total_requests
        }
    
    def _consider_model_update(self):
        """Consider updating production model"""
        # Create candidate model with current state
        candidate = OnlineLearner()
        candidate.load_state_dict(self.online_learner.state_dict())
        
        # Start A/B test if not already running
        if not self.ab_test_active:
            self._start_ab_test(candidate)
        else:
            # Check A/B test results
            self._evaluate_ab_test()
    
    def _start_ab_test(self, candidate_model: OnlineLearner):
        """Start A/B test with candidate model"""
        logger.info("Starting A/B test for model update")
        
        self.active_models['candidate'] = candidate_model
        self.ab_test_active = True
        self.ab_test_results.clear()
        
        # Create checkpoint
        checkpoint = self._create_checkpoint(candidate_model, "candidate")
        self.model_checkpoints[checkpoint.version] = checkpoint
    
    def _evaluate_ab_test(self):
        """Evaluate A/B test results and decide on promotion"""
        if not self.ab_test_active:
            return
        
        prod_results = self.ab_test_results['production']
        cand_results = self.ab_test_results['candidate']
        
        # Need minimum samples
        if prod_results['requests'] < 100 or cand_results['requests'] < 20:
            return
        
        # Calculate success rates
        prod_success_rate = prod_results['successes'] / max(1, prod_results['requests'])
        cand_success_rate = cand_results['successes'] / max(1, cand_results['requests'])
        
        # Calculate average latencies
        prod_avg_latency = np.mean(prod_results['latencies']) if prod_results['latencies'] else 0
        cand_avg_latency = np.mean(cand_results['latencies']) if cand_results['latencies'] else 0
        
        logger.info(f"A/B Test Results - Production: {prod_success_rate:.2%} success, {prod_avg_latency:.1f}ms")
        logger.info(f"A/B Test Results - Candidate: {cand_success_rate:.2%} success, {cand_avg_latency:.1f}ms")
        
        # Promote if candidate is better
        if cand_success_rate > prod_success_rate * 0.95 and cand_avg_latency < prod_avg_latency * 1.1:
            self._promote_candidate_model()
        else:
            self._reject_candidate_model()
    
    def _promote_candidate_model(self):
        """Promote candidate model to production"""
        logger.info("Promoting candidate model to production")
        
        # Swap models
        self.active_models['production'] = self.active_models['candidate']
        self.online_learner = self.active_models['production']
        
        # Update version
        old_version = self.current_version
        version_parts = old_version.split('.')
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        self.current_version = '.'.join(version_parts)
        
        # Create production checkpoint
        checkpoint = self._create_checkpoint(self.online_learner, "production")
        self.model_checkpoints[self.current_version] = checkpoint
        
        # End A/B test
        self.ab_test_active = False
        self.active_models['candidate'] = None
        
        logger.info(f"Model updated: {old_version} -> {self.current_version}")
    
    def _reject_candidate_model(self):
        """Reject candidate model and continue with production"""
        logger.info("Rejecting candidate model, keeping production")
        
        self.ab_test_active = False
        self.active_models['candidate'] = None
        self.ab_test_results.clear()
    
    def _consider_rollback(self, performance: Dict[str, float]):
        """Consider rolling back to a previous model version"""
        logger.warning(f"Performance degraded: {performance}")
        
        # Find best previous checkpoint
        best_checkpoint = None
        best_success_rate = performance['success_rate']
        
        for version, checkpoint in self.model_checkpoints.items():
            if checkpoint.metrics.get('success_rate', 0) > best_success_rate:
                best_checkpoint = checkpoint
                best_success_rate = checkpoint.metrics['success_rate']
        
        if best_checkpoint:
            logger.info(f"Rolling back to version {best_checkpoint.version}")
            self._rollback_to_checkpoint(best_checkpoint)
    
    def _rollback_to_checkpoint(self, checkpoint: ModelCheckpoint):
        """Rollback to a specific checkpoint"""
        # Load model state
        self.online_learner.load_state_dict(checkpoint.model_state)
        self.active_models['production'] = self.online_learner
        
        # Update version
        self.current_version = checkpoint.version
        
        # Clear A/B test if running
        self.ab_test_active = False
        self.active_models['candidate'] = None
        
        logger.info(f"Rolled back to version {checkpoint.version}")
    
    def _create_checkpoint(self, model: OnlineLearner, checkpoint_type: str) -> ModelCheckpoint:
        """Create a model checkpoint"""
        performance = self._evaluate_current_performance()
        
        checkpoint = ModelCheckpoint(
            version=f"{self.current_version}-{checkpoint_type}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            timestamp=datetime.now(),
            metrics={
                'success_rate': performance['success_rate'],
                'avg_latency': performance['avg_latency'],
                'total_requests': performance['total_requests']
            },
            model_state=model.state_dict(),
            metadata={
                'learning_rate': self._calculate_learning_rate(),
                'loss_history': list(model.loss_history)[-100:],
                'checkpoint_type': checkpoint_type
            }
        )
        
        return checkpoint
    
    def _update_performance_monitoring(self):
        """Update performance monitoring windows"""
        now = datetime.now()
        
        # Check if current window should be closed
        if self.current_window and (now - self.current_window.start_time) > self.window_duration:
            # Calculate percentiles for latency
            if hasattr(self, '_latency_buffer'):
                latencies = [l for l in self._latency_buffer if l > 0]
                if latencies:
                    self.current_window.p95_latency_ms = np.percentile(latencies, 95)
                    self.current_window.p99_latency_ms = np.percentile(latencies, 99)
            
            # Save window
            self.performance_windows.append(self.current_window)
            
            # Start new window
            self._start_new_performance_window()
    
    def _start_new_performance_window(self):
        """Start a new performance monitoring window"""
        self.current_window = PerformanceWindow(
            start_time=datetime.now(),
            end_time=datetime.now() + self.window_duration
        )
        
        # Reset latency buffer
        self._latency_buffer = []
    
    def _track_learning_progress(self, loss: float, metadata: List[Dict]):
        """Track learning progress for visualization"""
        # This would integrate with monitoring systems
        progress_data = {
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            'learning_rate': self._calculate_learning_rate(),
            'model_version': self.current_version,
            'batch_size': len(metadata),
            'performance': self._evaluate_current_performance()
        }
        
        # Log or send to monitoring system
        logger.debug(f"Learning progress: {progress_data}")
    
    def select_model_for_request(self) -> Tuple[str, OnlineLearner]:
        """Select model for incoming request (A/B testing)"""
        if self.ab_test_active and self.active_models['candidate'] is not None:
            # A/B test split
            if np.random.random() < self.ab_test_split:
                return 'candidate', self.active_models['candidate']
        
        return 'production', self.active_models['production']
    
    def record_request_result(
        self,
        model_version: str,
        success: bool,
        latency_ms: float
    ):
        """Record result for model performance tracking"""
        if self.ab_test_active:
            self.ab_test_results[model_version]['requests'] += 1
            if success:
                self.ab_test_results[model_version]['successes'] += 1
            self.ab_test_results[model_version]['latencies'].append(latency_ms)
        
        # Also track in latency buffer
        if hasattr(self, '_latency_buffer'):
            self._latency_buffer.append(latency_ms)
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning pipeline status"""
        performance = self._evaluate_current_performance()
        
        status = {
            'pipeline_version': '3.0',
            'model_version': self.current_version,
            'learning_buffer_size': len(self.learning_buffer),
            'feedback_buffer_size': len(self.feedback_buffer),
            'current_performance': performance,
            'ab_test_active': self.ab_test_active,
            'ab_test_results': dict(self.ab_test_results) if self.ab_test_active else None,
            'checkpoints_available': len(self.model_checkpoints),
            'learning_config': self.learning_config,
            'performance_windows': len(self.performance_windows),
            'recent_losses': list(self.online_learner.loss_history)[-10:] if self.online_learner.loss_history else []
        }
        
        return status
    
    def _save_checkpoints(self):
        """Save checkpoints to disk"""
        checkpoint_dir = Path("backend/data/learning_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'current_version': self.current_version,
            'checkpoint_list': list(self.model_checkpoints.keys())
        }
        
        with open(checkpoint_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        # Save individual checkpoints
        for version, checkpoint in self.model_checkpoints.items():
            checkpoint_file = checkpoint_dir / f"{version}.pt"
            torch.save({
                'model_state': checkpoint.model_state,
                'metrics': checkpoint.metrics,
                'metadata': checkpoint.metadata,
                'timestamp': checkpoint.timestamp.isoformat()
            }, checkpoint_file)
    
    def _load_checkpoints(self):
        """Load checkpoints from disk"""
        checkpoint_dir = Path("backend/data/learning_checkpoints")
        
        if not checkpoint_dir.exists():
            return
        
        # Load metadata
        metadata_file = checkpoint_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                self.current_version = metadata.get('current_version', self.current_version)
        
        # Load checkpoints
        for checkpoint_file in checkpoint_dir.glob("*.pt"):
            if checkpoint_file.stem != "metadata":
                try:
                    data = torch.load(checkpoint_file)
                    checkpoint = ModelCheckpoint(
                        version=checkpoint_file.stem,
                        timestamp=datetime.fromisoformat(data['timestamp']),
                        metrics=data['metrics'],
                        model_state=data['model_state'],
                        metadata=data['metadata']
                    )
                    self.model_checkpoints[checkpoint.version] = checkpoint
                except Exception as e:
                    logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
    
    async def shutdown(self):
        """Gracefully shutdown the pipeline"""
        logger.info("Shutting down Continuous Learning Pipeline")
        
        self.running = False
        
        # Wait for threads
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        # Save checkpoints
        self._save_checkpoints()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Continuous Learning Pipeline shutdown complete")


# Singleton instance
_learning_pipeline: Optional[ContinuousLearningPipeline] = None


def get_learning_pipeline() -> ContinuousLearningPipeline:
    """Get singleton instance of learning pipeline"""
    global _learning_pipeline
    if _learning_pipeline is None:
        _learning_pipeline = ContinuousLearningPipeline()
    return _learning_pipeline