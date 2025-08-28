#!/usr/bin/env python3
"""
Meta-Learning Framework for Vision System v2.0
Implements learning strategy selection, performance adaptation, and catastrophic forgetting prevention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
from collections import defaultdict, deque
import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class LearningStrategy:
    """Configuration for a specific learning strategy"""
    strategy_id: str
    name: str
    description: str
    hyperparameters: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    usage_count: int = 0
    last_used: Optional[datetime] = None
    effectiveness_score: float = 0.5

@dataclass
class ModelSnapshot:
    """Snapshot of model state for forgetting prevention"""
    snapshot_id: str
    timestamp: datetime
    model_state: Dict[str, torch.Tensor]
    performance_metrics: Dict[str, float]
    task_performance: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskPerformance:
    """Track performance on specific tasks"""
    task_id: str
    task_type: str  # intent, domain, skill
    performance_history: List[float]
    current_performance: float
    stability_score: float  # How stable the performance is
    last_evaluated: datetime

class LearningStrategySelector:
    """
    Selects optimal learning strategies based on current performance
    and learning context
    """
    
    def __init__(self):
        self.strategies = self._initialize_strategies()
        self.strategy_performance = defaultdict(list)
        self.current_strategy: Optional[LearningStrategy] = None
        self.selection_history = deque(maxlen=100)
        
    def _initialize_strategies(self) -> Dict[str, LearningStrategy]:
        """Initialize available learning strategies"""
        strategies = {
            'sgd_conservative': LearningStrategy(
                strategy_id='sgd_conservative',
                name='Conservative SGD',
                description='Slow, stable learning with SGD',
                hyperparameters={
                    'optimizer': 'sgd',
                    'lr': 0.001,
                    'momentum': 0.9,
                    'weight_decay': 0.0001,
                    'batch_size': 32
                }
            ),
            'adam_balanced': LearningStrategy(
                strategy_id='adam_balanced',
                name='Balanced Adam',
                description='Balanced learning with Adam optimizer',
                hyperparameters={
                    'optimizer': 'adam',
                    'lr': 0.0001,
                    'betas': (0.9, 0.999),
                    'weight_decay': 0.0001,
                    'batch_size': 64
                }
            ),
            'adam_aggressive': LearningStrategy(
                strategy_id='adam_aggressive',
                name='Aggressive Adam',
                description='Fast learning for rapid adaptation',
                hyperparameters={
                    'optimizer': 'adam',
                    'lr': 0.001,
                    'betas': (0.9, 0.999),
                    'weight_decay': 0,
                    'batch_size': 128
                }
            ),
            'cyclic_learning': LearningStrategy(
                strategy_id='cyclic_learning',
                name='Cyclic Learning Rate',
                description='Cyclic LR for escaping local minima',
                hyperparameters={
                    'optimizer': 'sgd',
                    'lr_min': 0.0001,
                    'lr_max': 0.01,
                    'cycle_length': 1000,
                    'batch_size': 64
                }
            ),
            'meta_sgd': LearningStrategy(
                strategy_id='meta_sgd',
                name='Meta-SGD',
                description='Meta-learned learning rates per parameter',
                hyperparameters={
                    'optimizer': 'meta_sgd',
                    'meta_lr': 0.01,
                    'inner_lr': 0.001,
                    'batch_size': 32
                }
            )
        }
        
        return strategies
    
    def select_strategy(
        self,
        current_performance: Dict[str, float],
        learning_context: Dict[str, Any]
    ) -> LearningStrategy:
        """Select optimal learning strategy based on context"""
        # Analyze current situation
        performance_trend = self._analyze_performance_trend(current_performance)
        stability = learning_context.get('stability', 1.0)
        urgency = learning_context.get('urgency', 0.5)
        task_diversity = learning_context.get('task_diversity', 0.5)
        
        # Score each strategy
        strategy_scores = {}
        
        for strategy_id, strategy in self.strategies.items():
            score = self._score_strategy(
                strategy,
                performance_trend,
                stability,
                urgency,
                task_diversity
            )
            strategy_scores[strategy_id] = score
        
        # Select best strategy
        best_strategy_id = max(strategy_scores.items(), key=lambda x: x[1])[0]
        selected_strategy = self.strategies[best_strategy_id]
        
        # Update tracking
        selected_strategy.usage_count += 1
        selected_strategy.last_used = datetime.now()
        self.current_strategy = selected_strategy
        
        self.selection_history.append({
            'timestamp': datetime.now(),
            'strategy_id': best_strategy_id,
            'context': learning_context,
            'scores': strategy_scores
        })
        
        logger.info(f"Selected learning strategy: {selected_strategy.name}")
        
        return selected_strategy
    
    def _analyze_performance_trend(self, performance: Dict[str, float]) -> str:
        """Analyze performance trend: improving, stable, or declining"""
        if 'history' in performance and len(performance['history']) > 3:
            recent = performance['history'][-3:]
            if all(recent[i] <= recent[i+1] for i in range(len(recent)-1)):
                return 'improving'
            elif all(recent[i] >= recent[i+1] for i in range(len(recent)-1)):
                return 'declining'
        return 'stable'
    
    def _score_strategy(
        self,
        strategy: LearningStrategy,
        performance_trend: str,
        stability: float,
        urgency: float,
        task_diversity: float
    ) -> float:
        """Score a strategy based on current context"""
        score = 0.5  # Base score
        
        # Historical performance
        if strategy.performance_history:
            score += 0.2 * np.mean(strategy.performance_history[-10:])
        
        # Match strategy to context
        if strategy.strategy_id == 'sgd_conservative':
            # Good for stable, low-urgency situations
            score += 0.3 * stability * (1 - urgency)
        
        elif strategy.strategy_id == 'adam_aggressive':
            # Good for urgent adaptation needs
            score += 0.3 * urgency * (1 - stability)
            
        elif strategy.strategy_id == 'cyclic_learning':
            # Good when performance is stuck
            if performance_trend == 'stable':
                score += 0.3
                
        elif strategy.strategy_id == 'meta_sgd':
            # Good for diverse tasks
            score += 0.3 * task_diversity
        
        # Penalize overused strategies (exploration bonus)
        if strategy.usage_count > 0:
            recency = (datetime.now() - strategy.last_used).total_seconds() / 3600
            score += 0.1 * min(1.0, recency / 24)  # Bonus for strategies not used in last 24h
        
        return min(1.0, max(0.0, score))
    
    def update_strategy_performance(self, strategy_id: str, performance: float):
        """Update strategy performance history"""
        if strategy_id in self.strategies:
            strategy = self.strategies[strategy_id]
            strategy.performance_history.append(performance)
            
            # Update effectiveness score (moving average)
            if len(strategy.performance_history) > 0:
                recent_performance = strategy.performance_history[-20:]
                strategy.effectiveness_score = np.mean(recent_performance)

class CatastrophicForgettingPrevention:
    """
    Prevents catastrophic forgetting using elastic weight consolidation
    and intelligent rehearsal
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fisher_information: Dict[str, torch.Tensor] = {}
        self.optimal_params: Dict[str, torch.Tensor] = {}
        self.task_performances: Dict[str, TaskPerformance] = {}
        self.model_snapshots: List[ModelSnapshot] = []
        self.ewc_lambda = 0.1  # Elastic weight consolidation strength
        
    def compute_fisher_information(
        self,
        data_loader,
        task_id: str,
        num_samples: int = 200
    ):
        """Compute Fisher Information Matrix for current task"""
        logger.info(f"Computing Fisher Information for task: {task_id}")
        
        # Set model to eval mode
        self.model.eval()
        
        # Initialize Fisher Information
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)
        
        # Sample data for Fisher computation
        samples_processed = 0
        
        for batch in data_loader:
            if samples_processed >= num_samples:
                break
                
            # Forward pass
            self.model.zero_grad()
            output = self.model(batch['input'])
            loss = F.cross_entropy(output, batch['target'])
            
            # Backward pass
            loss.backward()
            
            # Accumulate gradients
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data ** 2
            
            samples_processed += batch['input'].size(0)
        
        # Normalize
        for name in fisher:
            fisher[name] /= samples_processed
            
        # Store Fisher Information and optimal parameters
        self.fisher_information[task_id] = fisher
        self.optimal_params[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }
        
        logger.info(f"Fisher Information computed for {samples_processed} samples")
    
    def elastic_weight_consolidation_loss(self, current_task: Optional[str] = None) -> torch.Tensor:
        """Calculate EWC loss to prevent forgetting"""
        ewc_loss = 0
        
        # Sum over all previous tasks
        for task_id, fisher in self.fisher_information.items():
            if task_id == current_task:
                continue  # Don't regularize current task
                
            for name, param in self.model.named_parameters():
                if name in fisher:
                    # EWC penalty
                    optimal_param = self.optimal_params[task_id][name]
                    ewc_loss += (fisher[name] * (param - optimal_param) ** 2).sum()
        
        return self.ewc_lambda * ewc_loss
    
    def create_snapshot(self, performance_metrics: Dict[str, float]) -> str:
        """Create a snapshot of current model state"""
        snapshot_id = f"snapshot_{datetime.now().timestamp()}"
        
        # Calculate task performances
        task_perf = {}
        for task_id, perf in self.task_performances.items():
            task_perf[task_id] = perf.current_performance
        
        snapshot = ModelSnapshot(
            snapshot_id=snapshot_id,
            timestamp=datetime.now(),
            model_state={
                name: param.data.clone().cpu()
                for name, param in self.model.named_parameters()
            },
            performance_metrics=performance_metrics.copy(),
            task_performance=task_perf,
            metadata={
                'model_architecture': str(self.model),
                'total_parameters': sum(p.numel() for p in self.model.parameters())
            }
        )
        
        self.model_snapshots.append(snapshot)
        
        # Keep only recent snapshots
        if len(self.model_snapshots) > 10:
            self.model_snapshots.pop(0)
        
        logger.info(f"Created model snapshot: {snapshot_id}")
        
        return snapshot_id
    
    def should_restore_snapshot(self, current_performance: Dict[str, float]) -> bool:
        """Determine if we should restore a previous snapshot"""
        if not self.model_snapshots:
            return False
        
        # Check for significant performance degradation
        for task_id, perf in self.task_performances.items():
            if perf.current_performance < perf.stability_score * 0.8:  # 20% drop
                logger.warning(f"Significant performance drop on task {task_id}")
                return True
        
        # Check overall performance
        if 'accuracy' in current_performance:
            recent_snapshot = self.model_snapshots[-1]
            if current_performance['accuracy'] < recent_snapshot.performance_metrics.get('accuracy', 1.0) * 0.9:
                return True
        
        return False
    
    def restore_best_snapshot(self) -> Optional[str]:
        """Restore the best performing snapshot"""
        if not self.model_snapshots:
            return None
        
        # Find best snapshot based on overall performance
        best_snapshot = max(
            self.model_snapshots,
            key=lambda s: s.performance_metrics.get('accuracy', 0)
        )
        
        # Restore model state
        for name, param in self.model.named_parameters():
            if name in best_snapshot.model_state:
                param.data = best_snapshot.model_state[name].to(param.device)
        
        logger.info(f"Restored snapshot: {best_snapshot.snapshot_id}")
        
        return best_snapshot.snapshot_id
    
    def update_task_performance(self, task_id: str, performance: float):
        """Update performance tracking for a specific task"""
        if task_id not in self.task_performances:
            self.task_performances[task_id] = TaskPerformance(
                task_id=task_id,
                task_type='general',
                performance_history=[],
                current_performance=performance,
                stability_score=performance,
                last_evaluated=datetime.now()
            )
        else:
            task_perf = self.task_performances[task_id]
            task_perf.performance_history.append(performance)
            task_perf.current_performance = performance
            task_perf.last_evaluated = datetime.now()
            
            # Update stability score (exponential moving average)
            task_perf.stability_score = 0.9 * task_perf.stability_score + 0.1 * performance
            
            # Keep history bounded
            if len(task_perf.performance_history) > 100:
                task_perf.performance_history.pop(0)

class MetaLearningFramework:
    """
    Complete meta-learning framework with strategy selection,
    performance adaptation, and forgetting prevention
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.strategy_selector = LearningStrategySelector()
        self.forgetting_prevention = CatastrophicForgettingPrevention(base_model)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.current_performance = {
            'accuracy': 0.0,
            'loss': float('inf'),
            'task_performances': {}
        }
        
        # Learning configuration
        self.current_strategy: Optional[LearningStrategy] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        
        # Meta-learning components
        self.meta_optimizer = torch.optim.Adam(
            self.base_model.parameters(),
            lr=0.001
        )
        
        # Adaptation tracking
        self.adaptation_history = []
        self.last_adaptation = datetime.now()
        
        logger.info("Meta-Learning Framework initialized")
    
    def select_learning_strategy(self, context: Optional[Dict[str, Any]] = None) -> LearningStrategy:
        """Select optimal learning strategy based on current state"""
        # Prepare context
        learning_context = context or {}
        learning_context.update({
            'stability': self._calculate_stability(),
            'task_diversity': self._calculate_task_diversity(),
            'urgency': self._calculate_urgency()
        })
        
        # Select strategy
        strategy = self.strategy_selector.select_strategy(
            self.current_performance,
            learning_context
        )
        
        self.current_strategy = strategy
        
        # Configure optimizer based on strategy
        self._configure_optimizer(strategy)
        
        return strategy
    
    def _configure_optimizer(self, strategy: LearningStrategy):
        """Configure optimizer based on selected strategy"""
        params = strategy.hyperparameters
        
        if params['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.base_model.parameters(),
                lr=params.get('lr', 0.001),
                momentum=params.get('momentum', 0.9),
                weight_decay=params.get('weight_decay', 0.0001)
            )
            
        elif params['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.base_model.parameters(),
                lr=params.get('lr', 0.0001),
                betas=params.get('betas', (0.9, 0.999)),
                weight_decay=params.get('weight_decay', 0.0001)
            )
            
        elif params['optimizer'] == 'meta_sgd':
            # Simplified meta-SGD
            self.optimizer = torch.optim.SGD(
                self.base_model.parameters(),
                lr=params.get('inner_lr', 0.001)
            )
        
        # Configure scheduler if needed
        if strategy.strategy_id == 'cyclic_learning':
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optimizer,
                base_lr=params['lr_min'],
                max_lr=params['lr_max'],
                step_size_up=params['cycle_length'] // 2
            )
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        task_id: str = 'general'
    ) -> Dict[str, float]:
        """Single training step with forgetting prevention"""
        self.base_model.train()
        
        # Forward pass
        output = self.base_model(batch['input'])
        
        # Task loss
        task_loss = F.cross_entropy(output, batch['target'])
        
        # EWC loss for forgetting prevention
        ewc_loss = self.forgetting_prevention.elastic_weight_consolidation_loss(task_id)
        
        # Total loss
        total_loss = task_loss + ewc_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), 1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Scheduler step
        if self.scheduler:
            self.scheduler.step()
        
        # Calculate accuracy
        predictions = output.argmax(dim=1)
        accuracy = (predictions == batch['target']).float().mean().item()
        
        # Update performance tracking
        step_metrics = {
            'loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'ewc_loss': ewc_loss.item(),
            'accuracy': accuracy,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        self._update_performance(step_metrics, task_id)
        
        return step_metrics
    
    def _update_performance(self, metrics: Dict[str, float], task_id: str):
        """Update performance tracking"""
        # Update current performance
        self.current_performance['accuracy'] = metrics['accuracy']
        self.current_performance['loss'] = metrics['loss']
        
        # Update task performance
        self.forgetting_prevention.update_task_performance(task_id, metrics['accuracy'])
        
        # Add to history
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': metrics,
            'task_id': task_id,
            'strategy': self.current_strategy.strategy_id if self.current_strategy else None
        })
    
    def should_adapt(self) -> bool:
        """Determine if adaptation is needed"""
        # Don't adapt too frequently
        if (datetime.now() - self.last_adaptation) < timedelta(minutes=5):
            return False
        
        # Check if restoration is needed
        if self.forgetting_prevention.should_restore_snapshot(self.current_performance):
            return True
        
        # Check if strategy change is beneficial
        if self._performance_declining():
            return True
        
        return False
    
    def adapt(self, context: Optional[Dict[str, Any]] = None):
        """Adapt learning based on current performance"""
        logger.info("Adapting learning strategy...")
        
        # Check if we should restore a snapshot
        if self.forgetting_prevention.should_restore_snapshot(self.current_performance):
            snapshot_id = self.forgetting_prevention.restore_best_snapshot()
            logger.info(f"Restored model from snapshot: {snapshot_id}")
        
        # Select new learning strategy
        new_strategy = self.select_learning_strategy(context)
        
        # Update strategy performance
        if self.current_strategy:
            avg_performance = np.mean([
                h['metrics']['accuracy']
                for h in self.performance_history
                if h['strategy'] == self.current_strategy.strategy_id
            ][-50:])  # Last 50 steps
            
            self.strategy_selector.update_strategy_performance(
                self.current_strategy.strategy_id,
                avg_performance
            )
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': datetime.now(),
            'previous_strategy': self.current_strategy.strategy_id if self.current_strategy else None,
            'new_strategy': new_strategy.strategy_id,
            'reason': 'performance_decline' if self._performance_declining() else 'scheduled',
            'current_performance': self.current_performance.copy()
        })
        
        self.last_adaptation = datetime.now()
        
        logger.info(f"Adapted to strategy: {new_strategy.name}")
    
    def _calculate_stability(self) -> float:
        """Calculate current learning stability"""
        if len(self.performance_history) < 10:
            return 0.5
        
        recent_performance = [h['metrics']['accuracy'] for h in self.performance_history[-10:]]
        stability = 1.0 - np.std(recent_performance)
        
        return max(0.0, min(1.0, stability))
    
    def _calculate_task_diversity(self) -> float:
        """Calculate diversity of recent tasks"""
        if len(self.performance_history) < 20:
            return 0.5
        
        recent_tasks = [h['task_id'] for h in self.performance_history[-20:]]
        unique_tasks = len(set(recent_tasks))
        diversity = unique_tasks / len(recent_tasks)
        
        return diversity
    
    def _calculate_urgency(self) -> float:
        """Calculate learning urgency based on performance"""
        if self._performance_declining():
            return 0.8
        
        # Check accuracy
        if self.current_performance['accuracy'] < 0.6:
            return 0.9
        elif self.current_performance['accuracy'] < 0.8:
            return 0.6
        
        return 0.3
    
    def _performance_declining(self) -> bool:
        """Check if performance is declining"""
        if len(self.performance_history) < 20:
            return False
        
        recent = [h['metrics']['accuracy'] for h in self.performance_history[-20:]]
        older = [h['metrics']['accuracy'] for h in self.performance_history[-40:-20]]
        
        if older and np.mean(recent) < np.mean(older) * 0.95:  # 5% decline
            return True
        
        return False
    
    def create_checkpoint(self) -> str:
        """Create a checkpoint of current state"""
        return self.forgetting_prevention.create_snapshot(self.current_performance)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics"""
        metrics = {
            'current_performance': self.current_performance,
            'current_strategy': self.current_strategy.name if self.current_strategy else None,
            'stability': self._calculate_stability(),
            'task_diversity': self._calculate_task_diversity(),
            'adaptations_count': len(self.adaptation_history),
            'task_performances': {
                task_id: perf.current_performance
                for task_id, perf in self.forgetting_prevention.task_performances.items()
            },
            'strategy_effectiveness': {
                strategy_id: strategy.effectiveness_score
                for strategy_id, strategy in self.strategy_selector.strategies.items()
            }
        }
        
        return metrics
    
    def save_state(self, path: Path):
        """Save meta-learning state"""
        state = {
            'model_state': self.base_model.state_dict(),
            'performance_history': list(self.performance_history),
            'adaptation_history': self.adaptation_history,
            'strategy_performances': {
                s_id: s.performance_history
                for s_id, s in self.strategy_selector.strategies.items()
            },
            'fisher_information': self.forgetting_prevention.fisher_information,
            'optimal_params': self.forgetting_prevention.optimal_params
        }
        
        torch.save(state, path)
        logger.info(f"Saved meta-learning state to {path}")
    
    def load_state(self, path: Path):
        """Load meta-learning state"""
        state = torch.load(path)
        
        self.base_model.load_state_dict(state['model_state'])
        self.performance_history = deque(state['performance_history'], maxlen=1000)
        self.adaptation_history = state['adaptation_history']
        
        # Restore strategy performances
        for s_id, history in state['strategy_performances'].items():
            if s_id in self.strategy_selector.strategies:
                self.strategy_selector.strategies[s_id].performance_history = history
        
        # Restore Fisher Information
        self.forgetting_prevention.fisher_information = state.get('fisher_information', {})
        self.forgetting_prevention.optimal_params = state.get('optimal_params', {})
        
        logger.info(f"Loaded meta-learning state from {path}")

# Singleton instance
_meta_framework: Optional[MetaLearningFramework] = None

def get_meta_learning_framework(model: nn.Module) -> MetaLearningFramework:
    """Get singleton instance of meta-learning framework"""
    global _meta_framework
    if _meta_framework is None:
        _meta_framework = MetaLearningFramework(model)
    return _meta_framework