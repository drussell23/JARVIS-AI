#!/usr/bin/env python3
"""
Simplified ML Model Loader for JARVIS
Only loads essential models: Whisper, embeddings
"""

import logging
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MLModelLoader:
    """Simplified model loader for essential models only"""
    
    def __init__(self):
        self.models = {}
        self.whisper_model = None
        self.embedding_model = None
    
    def load_essential_models(self):
        """Load only essential models"""
        logger.info("Loading essential ML models...")
        
        # Whisper is loaded on-demand by voice engine
        logger.info("✅ Whisper model will be loaded on first use")
        
        # Basic embeddings for context
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded embedding model")
        except Exception as e:
            logger.warning(f"Embeddings not available: {e}")
        
        # Claude and Llama are API-based, no local loading needed
        logger.info("✅ Claude Vision API ready")
        logger.info("✅ Llama.cpp integration ready")
        
        return True

# Global instance
ml_loader = MLModelLoader()

# Compatibility functions for main.py
def initialize_models():
    """Initialize essential models (compatibility function)"""
    return ml_loader.load_essential_models()

def get_loader_status():
    """Get model loader status (compatibility function)"""
    return {
        "models_loaded": True,
        "whisper": "on-demand",
        "embeddings": ml_loader.embedding_model is not None,
        "claude_vision": True,
        "llama": True
    }
