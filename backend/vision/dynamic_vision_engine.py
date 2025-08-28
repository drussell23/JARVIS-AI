#!/usr/bin/env python3
"""
Dynamic Vision Engine - Wrapper for Lazy Loading
This module wraps the lazy vision engine for backward compatibility
"""

import logging
from .lazy_vision_engine import (
    get_lazy_vision_engine,
    initialize_vision_engine_models,
    VisionCapability,
    VisionIntent
)

logger = logging.getLogger(__name__)

# Create a wrapper class that delegates to lazy engine
class DynamicVisionEngine:
    """Wrapper for lazy vision engine - maintains compatibility"""
    
    def __init__(self):
        self._engine = get_lazy_vision_engine()
        logger.info("Dynamic Vision Engine initialized with lazy loading")
        
    def __getattr__(self, name):
        """Delegate all attributes to lazy engine"""
        return getattr(self._engine, name)
        
    async def initialize(self):
        """Initialize models - for compatibility"""
        if not self._engine._models_loaded:
            await initialize_vision_engine_models()
            
    def get_model_info(self):
        """Get model loading status"""
        return self._engine.get_model_info()

# For backward compatibility
def get_vision_engine():
    """Get the global vision engine instance"""
    return DynamicVisionEngine()

def get_dynamic_vision_engine():
    """Get the dynamic vision engine - alias for compatibility"""
    return get_vision_engine()
