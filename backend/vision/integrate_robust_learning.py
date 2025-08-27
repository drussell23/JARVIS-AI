#!/usr/bin/env python3
"""
Integration module to use robust continuous learning
Provides backward compatibility while switching to robust implementation
"""

import logging
from typing import Optional
import torch.nn as nn

logger = logging.getLogger(__name__)

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

# Import original for compatibility
try:
    from .advanced_continuous_learning import (
        AdvancedContinuousLearning as OriginalAdvancedContinuousLearning,
        LearningTask,
        FederatedUpdate,
        AdaptiveLearningRateAdjuster,
        PerformanceMonitor
    )
except ImportError as e:
    logger.warning(f"Could not import from advanced_continuous_learning: {e}")
    # Define minimal versions for compatibility
    from dataclasses import dataclass
    from datetime import datetime
    from typing import Dict, Any
    
    @dataclass
    class LearningTask:
        task_id: str
        task_type: str
        data_source: str
        priority: float
        created_at: datetime
        metadata: Dict[str, Any]
    
    @dataclass 
    class FederatedUpdate:
        update_id: str
        source_id: str
        model_updates: dict
        performance_metrics: dict
        data_statistics: dict
        timestamp: datetime
    
    class AdaptiveLearningRateAdjuster:
        def __init__(self):
            self.current_lr = 0.001
    
    class PerformanceMonitor:
        def __init__(self):
            pass
    
    class OriginalAdvancedContinuousLearning:
        def __init__(self, model):
            self.model = model
            self.running = False


class AdvancedContinuousLearning:
    """
    Wrapper class that uses robust implementation if available,
    falls back to original otherwise
    """
    
    def __init__(self, model: nn.Module):
        if ROBUST_AVAILABLE:
            # Create config with sensible defaults
            config = LearningConfig(
                max_cpu_percent=40.0,  # Conservative CPU limit
                max_memory_percent=25.0,  # Conservative memory limit
                min_free_memory_mb=1500.0,  # Ensure 1.5GB free
                retraining_interval_hours=6.0,
                mini_batch_interval_minutes=15.0,
                task_timeout_seconds=300,
                enable_adaptive_scheduling=True,
                load_factor_threshold=0.7,  # Start throttling at 70% load
                critical_load_threshold=0.9  # Stop at 90% load
            )
            
            logger.info("Using robust continuous learning implementation")
            self._impl = get_robust_continuous_learning(model, config)
            self._is_robust = True
        else:
            logger.warning("Falling back to original continuous learning")
            self._impl = OriginalAdvancedContinuousLearning(model)
            self._is_robust = False
    
    # Delegate all methods to implementation
    def __getattr__(self, name):
        """Delegate attribute access to implementation"""
        return getattr(self._impl, name)
    
    async def record_interaction(
        self,
        command: str,
        command_embedding,
        intent: str,
        confidence: float,
        handler: str,
        response: str,
        success: bool,
        latency_ms: float,
        user_id: Optional[str] = None,
        context: Optional[dict] = None,
        feedback: Optional[dict] = None
    ):
        """Record interaction - compatible with both implementations"""
        return await self._impl.record_interaction(
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
    
    def get_status(self):
        """Get status - compatible interface"""
        return self._impl.get_status()
    
    async def shutdown(self):
        """Shutdown - compatible interface"""
        return await self._impl.shutdown()
    
    def pause(self):
        """Pause learning"""
        if self._is_robust:
            self._impl.pause()
        else:
            # Original doesn't have pause, just set running to false
            self._impl.running = False
    
    def resume(self):
        """Resume learning"""
        if self._is_robust:
            self._impl.resume()
        else:
            # Original doesn't have resume, just set running to true
            self._impl.running = True
    
    async def add_federated_update(self, update: FederatedUpdate):
        """Add federated update"""
        return await self._impl.add_federated_update(update)


# Singleton management
_advanced_learning: Optional[AdvancedContinuousLearning] = None


def get_advanced_continuous_learning(model: nn.Module) -> AdvancedContinuousLearning:
    """Get singleton instance of advanced continuous learning"""
    global _advanced_learning
    if _advanced_learning is None:
        _advanced_learning = AdvancedContinuousLearning(model)
    return _advanced_learning


# Monkey patch the original module to use robust version
def apply_robust_learning_patch():
    """Apply patch to use robust learning everywhere"""
    try:
        import vision.advanced_continuous_learning as original_module
        
        # Replace the factory function
        original_module.get_advanced_continuous_learning = get_advanced_continuous_learning
        
        # Replace the class (for direct instantiation)
        original_module.AdvancedContinuousLearning = AdvancedContinuousLearning
        
        logger.info("Applied robust continuous learning patch")
        return True
        
    except Exception as e:
        logger.error(f"Failed to apply robust learning patch: {e}")
        return False