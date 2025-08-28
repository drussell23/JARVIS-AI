#!/usr/bin/env python3
"""
Vision Module - Lazy Loading Wrapper
All models are loaded on-demand or through parallel loading
"""

import logging

logger = logging.getLogger(__name__)

# Import core components with lazy loading
try:
    from .lazy_vision_engine import get_lazy_vision_engine, initialize_vision_engine_models
    from .dynamic_vision_engine import DynamicVisionEngine
    
    # Create default instance
    vision_engine = get_lazy_vision_engine()
    
    logger.info("Vision module initialized with lazy loading")
except ImportError as e:
    logger.warning(f"Could not import lazy vision components: {e}")
    vision_engine = None

# Export main interfaces
__all__ = [
    'vision_engine',
    'get_lazy_vision_engine',
    'initialize_vision_engine_models',
    'DynamicVisionEngine'
]
