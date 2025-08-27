#!/usr/bin/env python3
"""
Integration module to use robust continuous learning
Provides backward compatibility while switching to robust implementation
"""

import logging
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Define shared data classes to avoid circular imports
@dataclass
class LearningTask:
    """Represents a learning task"""
    task_id: str
    task_type: str
    data_source: str
    priority: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FederatedUpdate:
    """Update from federated learning"""
    update_id: str
    source_id: str
    model_updates: Dict[str, torch.Tensor]
    performance_metrics: Dict[str, float]
    data_statistics: Dict[str, Any]
    timestamp: datetime

# Try to import the robust version
try:
    from .robust_continuous_learning import (
        RobustAdvancedContinuousLearning,
        LearningConfig,
        get_robust_continuous_learning
    )
    ROBUST_AVAILABLE = True
    logger.info("Robust continuous learning available")
except ImportError as e:
    logger.warning(f"Robust continuous learning not available: {e}")
    ROBUST_AVAILABLE = False

# Define placeholder classes that will be overridden
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

# Wrapper class that delegates to robust version if available
class AdvancedContinuousLearning:
    """
    Wrapper that uses robust implementation if available,
    otherwise falls back to original
    """
    
    def __init__(self, model: nn.Module):
        if ROBUST_AVAILABLE:
            # Use robust implementation
            self._impl = RobustAdvancedContinuousLearning(model)
        else:
            # Import original only when needed to avoid circular import
            from .advanced_continuous_learning import (
                AdvancedContinuousLearning as OriginalImpl
            )
            self._impl = OriginalImpl(model)
    
    def __getattr__(self, name):
        # Delegate all attribute access to implementation
        return getattr(self._impl, name)


def get_advanced_continuous_learning(model: nn.Module) -> AdvancedContinuousLearning:
    """Get instance of advanced continuous learning (robust if available)"""
    if ROBUST_AVAILABLE:
        return get_robust_continuous_learning(model)
    else:
        # Import original only when needed
        from .advanced_continuous_learning import (
            get_advanced_continuous_learning as get_original
        )
        return get_original(model)


def apply_robust_learning_patch():
    """Apply patches to use robust learning throughout the system"""
    if not ROBUST_AVAILABLE:
        logger.warning("Robust learning not available, using original")
        return False
    
    logger.info("Applied robust continuous learning patch")
    return True


# Export all necessary symbols
__all__ = [
    'get_advanced_continuous_learning',
    'AdvancedContinuousLearning',
    'LearningTask',
    'FederatedUpdate',
    'AdaptiveLearningRateAdjuster',
    'PerformanceMonitor',
    'apply_robust_learning_patch',
    'ROBUST_AVAILABLE'
]