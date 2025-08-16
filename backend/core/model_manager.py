"""
Model Manager - Brain Switcher for JARVIS
Implements tiered model loading with automatic switching based on complexity and memory
"""

import os
import logging
import psutil
import asyncio
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tiers for different complexity levels"""
    TINY = "tiny"      # TinyLlama - 1GB - Always loaded
    STANDARD = "std"   # Phi-2 - 2GB - On-demand
    ADVANCED = "adv"   # Mistral-7B - 4GB - Complex tasks
    

class ModelInfo:
    """Information about a model"""
    def __init__(self, name: str, path: str, size_gb: float, tier: ModelTier, 
                 capabilities: Dict[str, float], context_size: int = 2048):
        self.name = name
        self.path = path
        self.size_gb = size_gb
        self.tier = tier
        self.capabilities = capabilities  # Scores for different task types
        self.context_size = context_size
        self.last_used: Optional[datetime] = None
        self.load_count = 0
        self.avg_response_time = 0.0
        

class ModelManager:
    """Manages tiered model loading and switching"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Model registry
        self.models: Dict[ModelTier, ModelInfo] = {
            ModelTier.TINY: ModelInfo(
                name="TinyLlama-1.1B",
                path="tinyllama-1.1b.gguf",
                size_gb=1.0,
                tier=ModelTier.TINY,
                capabilities={
                    "chat": 0.7,
                    "code": 0.5,
                    "analysis": 0.4,
                    "creative": 0.6
                },
                context_size=2048
            ),
            ModelTier.STANDARD: ModelInfo(
                name="Phi-2",
                path="phi-2.gguf",
                size_gb=2.0,
                tier=ModelTier.STANDARD,
                capabilities={
                    "chat": 0.85,
                    "code": 0.8,
                    "analysis": 0.75,
                    "creative": 0.8
                },
                context_size=2048
            ),
            ModelTier.ADVANCED: ModelInfo(
                name="Mistral-7B",
                path="mistral-7b-instruct.gguf",
                size_gb=4.0,
                tier=ModelTier.ADVANCED,
                capabilities={
                    "chat": 0.95,
                    "code": 0.9,
                    "analysis": 0.95,
                    "creative": 0.9
                },
                context_size=4096
            )
        }
        
        # Loaded models cache
        self.loaded_models: Dict[ModelTier, Any] = {}
        
        # Performance tracking
        self.performance_stats = {}
        
        # Memory thresholds
        self.memory_thresholds = {
            "critical": 85,  # Unload all but tiny
            "high": 75,      # Unload advanced
            "moderate": 60,  # Normal operation
            "low": 40        # Can load any model
        }
        
        # Initialize with tiny model
        asyncio.create_task(self._ensure_tiny_loaded())
        
    async def _ensure_tiny_loaded(self):
        """Ensure tiny model is always loaded"""
        try:
            await self.load_model(ModelTier.TINY)
            logger.info("Tiny model loaded for instant responses")
        except Exception as e:
            logger.error(f"Failed to load tiny model: {e}")
            
    async def get_model_for_task(self, task_type: str, complexity: float, 
                                 context_length: int) -> Tuple[Any, ModelTier]:
        """Get the best model for a given task"""
        # Check memory pressure
        memory_state = self._get_memory_state()
        
        # Determine required tier based on complexity
        required_tier = self._determine_required_tier(task_type, complexity, context_length)
        
        # Adjust based on memory
        selected_tier = self._adjust_tier_for_memory(required_tier, memory_state)
        
        # Load model if needed
        model = await self._get_or_load_model(selected_tier)
        
        return model, selected_tier
        
    def _determine_required_tier(self, task_type: str, complexity: float, 
                                context_length: int) -> ModelTier:
        """Determine the required model tier based on task characteristics"""
        # Context length check
        if context_length > 3000:
            return ModelTier.ADVANCED  # Only Mistral has 4k context
            
        # Complexity-based selection
        if complexity > 0.8:
            return ModelTier.ADVANCED
        elif complexity > 0.5:
            return ModelTier.STANDARD
        else:
            return ModelTier.TINY
            
    def _get_memory_state(self) -> str:
        """Get current memory state"""
        mem = psutil.virtual_memory()
        percent = mem.percent
        
        if percent >= self.memory_thresholds["critical"]:
            return "critical"
        elif percent >= self.memory_thresholds["high"]:
            return "high"
        elif percent >= self.memory_thresholds["moderate"]:
            return "moderate"
        else:
            return "low"
            
    def _adjust_tier_for_memory(self, requested_tier: ModelTier, 
                                memory_state: str) -> ModelTier:
        """Adjust tier based on memory pressure"""
        if memory_state == "critical":
            # Only use tiny model
            return ModelTier.TINY
        elif memory_state == "high":
            # Downgrade from advanced if requested
            if requested_tier == ModelTier.ADVANCED:
                return ModelTier.STANDARD
        # Otherwise use requested tier
        return requested_tier
        
    async def _get_or_load_model(self, tier: ModelTier) -> Any:
        """Get model, loading if necessary"""
        if tier in self.loaded_models:
            model = self.loaded_models[tier]
            self.models[tier].last_used = datetime.now()
            return model
            
        # Need to load model
        return await self.load_model(tier)
        
    async def load_model(self, tier: ModelTier) -> Any:
        """Load a model into memory"""
        model_info = self.models[tier]
        model_path = self.models_dir / model_info.path
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
            
        # Check if we need to unload other models first
        await self._manage_memory_for_load(model_info.size_gb)
        
        logger.info(f"Loading {tier.value} model: {model_info.name}")
        
        try:
            # Import here to avoid circular dependencies
            try:
                from langchain_community.llms import LlamaCpp
                
                # Create model with optimized settings
                model = LlamaCpp(
                    model_path=str(model_path),
                    n_gpu_layers=1,  # Use Metal GPU
                    n_ctx=model_info.context_size,
                    n_batch=256 if tier == ModelTier.TINY else 512,
                    temperature=0.7,
                    max_tokens=256,
                    n_threads=4 if tier == ModelTier.TINY else 6,
                    use_mlock=True,
                    verbose=False,
                    f16_kv=True,
                    streaming=False
                )
            except ImportError:
                logger.warning("LlamaCpp not available, using mock model for testing")
                # Create a mock model for testing
                class MockLLM:
                    def __init__(self, **kwargs):
                        self.model_path = kwargs.get('model_path', '')
                        self.n_ctx = kwargs.get('n_ctx', 2048)
                    
                    def __call__(self, prompt: str, **kwargs) -> str:
                        return f"Mock response from {model_info.name} for: {prompt[:50]}..."
                    
                    def invoke(self, prompt: str, **kwargs) -> str:
                        return self.__call__(prompt, **kwargs)
                
                model = MockLLM(
                    model_path=str(model_path),
                    n_ctx=model_info.context_size
                )
            
            # Cache the model
            self.loaded_models[tier] = model
            model_info.load_count += 1
            model_info.last_used = datetime.now()
            
            logger.info(f"Successfully loaded {model_info.name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load {model_info.name}: {e}")
            raise
            
    async def _manage_memory_for_load(self, required_gb: float):
        """Manage memory before loading a new model"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        if available_gb < required_gb + 1:  # Keep 1GB buffer
            # Need to unload models
            await self._unload_least_used_models(required_gb + 1 - available_gb)
            
    async def _unload_least_used_models(self, needed_gb: float):
        """Unload least recently used models to free memory"""
        freed_gb = 0
        
        # Sort loaded models by last used time (exclude tiny)
        loaded_tiers = [
            (tier, info) for tier, info in self.models.items()
            if tier in self.loaded_models and tier != ModelTier.TINY
        ]
        loaded_tiers.sort(key=lambda x: x[1].last_used or datetime.min)
        
        for tier, info in loaded_tiers:
            if freed_gb >= needed_gb:
                break
                
            logger.info(f"Unloading {info.name} to free memory")
            await self.unload_model(tier)
            freed_gb += info.size_gb
            
    async def unload_model(self, tier: ModelTier):
        """Unload a model from memory"""
        if tier in self.loaded_models:
            # Explicitly delete to free memory
            del self.loaded_models[tier]
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info(f"Unloaded {self.models[tier].name}")
            
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about model usage"""
        stats = {
            "loaded_models": [
                {
                    "tier": tier.value,
                    "name": self.models[tier].name,
                    "size_gb": self.models[tier].size_gb,
                    "last_used": self.models[tier].last_used.isoformat() 
                               if self.models[tier].last_used else None,
                    "load_count": self.models[tier].load_count
                }
                for tier in self.loaded_models
            ],
            "memory_state": self._get_memory_state(),
            "total_models": len(self.models),
            "loaded_count": len(self.loaded_models)
        }
        return stats
        
    async def optimize_for_workload(self, recent_tasks: list):
        """Optimize model loading based on recent workload"""
        # Analyze recent tasks to predict future needs
        task_types = {}
        for task in recent_tasks:
            task_type = task.get("type", "chat")
            task_types[task_type] = task_types.get(task_type, 0) + 1
            
        # Preload models based on patterns
        if task_types.get("code", 0) > 5:
            # Lots of coding tasks, ensure standard model is loaded
            await self._get_or_load_model(ModelTier.STANDARD)
        elif task_types.get("analysis", 0) > 3:
            # Complex analysis, consider loading advanced
            if self._get_memory_state() in ["low", "moderate"]:
                await self._get_or_load_model(ModelTier.ADVANCED)