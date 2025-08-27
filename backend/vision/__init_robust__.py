#!/usr/bin/env python3
"""
Initialize robust continuous learning on import
This file should be imported early to ensure robust implementation is used
"""

import logging

logger = logging.getLogger(__name__)

# Apply the robust learning patch on import
try:
    from .integrate_robust_learning import apply_robust_learning_patch
    
    if apply_robust_learning_patch():
        logger.info("Robust continuous learning initialized successfully")
    else:
        logger.warning("Could not initialize robust continuous learning")
        
except Exception as e:
    logger.error(f"Error initializing robust continuous learning: {e}")


# Export the necessary components
from .integrate_robust_learning import (
    get_advanced_continuous_learning,
    LearningTask,
    FederatedUpdate,
    AdaptiveLearningRateAdjuster,
    PerformanceMonitor
)

__all__ = [
    'get_advanced_continuous_learning',
    'LearningTask',
    'FederatedUpdate',
    'AdaptiveLearningRateAdjuster',
    'PerformanceMonitor'
]