"""
Memory-Efficient Model Manager with Swapping
Handles loading, unloading, and swapping of ML models
"""

import os
import gc
import time
import pickle
import gzip
import psutil
import logging
import weakref
import threading
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .optimization_config import (
    OptimizationConfig, ModelPriority, OPTIMIZATION_CONFIG
)

logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Information about a loaded model"""
    name: str
    model: Any
    priority: ModelPriority
    size_mb: float
    last_used: datetime
    load_time_seconds: float
    access_count: int = 0
    is_locked: bool = False  # Prevent unloading

class ModelManager:
    """
    Manages ML models with automatic swapping and memory limits
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OPTIMIZATION_CONFIG
        
        # Model storage
        self._models: OrderedDict[str, ModelInfo] = OrderedDict()
        self._model_factories: Dict[str, Callable] = {}
        self._swap_dir = self.config.model_swap.swap_directory
        os.makedirs(self._swap_dir, exist_ok=True)
        
        # Memory tracking
        self._current_memory_mb = 0.0
        self._memory_limit_mb = self.config.get_memory_limit_bytes() / (1024**2)
        
        # Threading
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._gc_thread = None
        self._running = True
        
        # Monitoring
        self._swap_count = 0
        self._load_count = 0
        self._last_gc = time.time()
        
        # Start background tasks
        self._start_background_tasks()
        
        logger.info(f"Model Manager initialized with {self._memory_limit_mb:.1f}MB limit")
    
    def register_model_factory(self, name: str, factory: Callable, 
                             priority: ModelPriority = None):
        """Register a factory function to create a model"""
        self._model_factories[name] = factory
        
        # Override priority if specified
        if priority:
            self.config.model_swap.model_priorities[name] = priority
    
    def get_model(self, name: str, timeout: float = 30.0) -> Any:
        """
        Get a model, loading it if necessary
        Blocks until model is available or timeout
        """
        start_time = time.time()
        
        with self._lock:
            # Check if already loaded
            if name in self._models:
                model_info = self._models[name]
                model_info.last_used = datetime.now()
                model_info.access_count += 1
                
                # Move to end (LRU)
                self._models.move_to_end(name)
                
                return model_info.model
            
            # Check if we need to make room
            if not self._ensure_memory_available(name):
                raise MemoryError(f"Cannot load model {name}: insufficient memory")
            
            # Load the model
            model = self._load_model(name)
            
            if model is None:
                raise ValueError(f"Failed to load model {name}")
            
            load_time = time.time() - start_time
            logger.info(f"Model {name} loaded in {load_time:.2f}s")
            
            return model
    
    async def get_model_async(self, name: str) -> Any:
        """Async version of get_model"""
        return await asyncio.to_thread(self.get_model, name)
    
    def _load_model(self, name: str) -> Optional[Any]:
        """Load a model from disk or create new"""
        try:
            # First try to load from swap
            swap_path = os.path.join(self._swap_dir, f"{name}.pkl.gz")
            if os.path.exists(swap_path):
                logger.info(f"Loading {name} from swap...")
                with gzip.open(swap_path, 'rb') as f:
                    model = pickle.load(f)
                os.remove(swap_path)  # Remove swap file
                self._swap_count += 1
            
            # Otherwise create new
            elif name in self._model_factories:
                logger.info(f"Creating new {name} model...")
                model = self._model_factories[name]()
            else:
                logger.error(f"No factory registered for model {name}")
                return None
            
            # Estimate size
            size_mb = self._estimate_model_size(model)
            
            # Store model info
            priority = self.config.model_swap.model_priorities.get(
                name, ModelPriority.MEDIUM
            )
            
            model_info = ModelInfo(
                name=name,
                model=model,
                priority=priority,
                size_mb=size_mb,
                last_used=datetime.now(),
                load_time_seconds=0.0
            )
            
            self._models[name] = model_info
            self._current_memory_mb += size_mb
            self._load_count += 1
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {name}: {e}")
            return None
    
    def _ensure_memory_available(self, new_model_name: str) -> bool:
        """Ensure enough memory is available for a new model"""
        # Check system memory
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            logger.warning(f"System memory critical: {memory_percent}%")
            self._emergency_cleanup()
        
        # Check our memory limit
        memory_usage_percent = (self._current_memory_mb / self._memory_limit_mb) * 100
        
        if memory_usage_percent < self.config.memory.memory_warning_threshold:
            return True
        
        # Need to free memory
        logger.info(f"Memory usage at {memory_usage_percent:.1f}%, freeing space...")
        
        # Get priority of new model
        new_priority = self.config.model_swap.model_priorities.get(
            new_model_name, ModelPriority.MEDIUM
        )
        
        # Unload models with lower priority
        models_to_unload = []
        
        for name, info in self._models.items():
            # Don't unload locked or essential models
            if info.is_locked or info.priority == ModelPriority.ESSENTIAL:
                continue
            
            # Unload if lower priority or not used recently
            if (info.priority.value > new_priority.value or
                info.last_used < datetime.now() - timedelta(
                    seconds=self.config.memory.model_cache_timeout_seconds)):
                models_to_unload.append(name)
        
        # Unload in LRU order
        for name in models_to_unload:
            self._unload_model(name, swap=True)
            
            # Check if we have enough space
            if (self._current_memory_mb / self._memory_limit_mb) * 100 < 70:
                break
        
        return True
    
    def _unload_model(self, name: str, swap: bool = True):
        """Unload a model from memory"""
        if name not in self._models:
            return
        
        model_info = self._models[name]
        
        # Skip if locked
        if model_info.is_locked:
            return
        
        logger.info(f"Unloading model {name} (size: {model_info.size_mb:.1f}MB)")
        
        # Swap to disk if requested
        if swap and self.config.model_swap.swap_to_disk:
            swap_path = os.path.join(self._swap_dir, f"{name}.pkl.gz")
            try:
                with gzip.open(swap_path, 'wb', compresslevel=1) as f:
                    pickle.dump(model_info.model, f)
                logger.info(f"Swapped {name} to disk")
            except Exception as e:
                logger.error(f"Failed to swap {name}: {e}")
        
        # Clear from memory
        del self._models[name]
        self._current_memory_mb -= model_info.size_mb
        
        # Force garbage collection
        if self.config.memory.force_gc_on_model_unload:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB"""
        try:
            # For PyTorch models
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() * p.element_size() 
                                 for p in model.parameters())
                return total_params / (1024**2)
            
            # For sklearn models
            elif hasattr(model, '__sizeof__'):
                return model.__sizeof__() / (1024**2)
            
            # For numpy arrays
            elif isinstance(model, np.ndarray):
                return model.nbytes / (1024**2)
            
            # Default estimate
            else:
                # Use pickle to estimate
                import io
                buffer = io.BytesIO()
                pickle.dump(model, buffer)
                return buffer.tell() / (1024**2)
                
        except Exception:
            return 10.0  # Default 10MB estimate
    
    def _emergency_cleanup(self):
        """Emergency cleanup when system memory is critical"""
        logger.warning("Emergency cleanup triggered!")
        
        # Unload all non-essential models
        for name, info in list(self._models.items()):
            if info.priority != ModelPriority.ESSENTIAL and not info.is_locked:
                self._unload_model(name, swap=False)  # Don't swap in emergency
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def lock_model(self, name: str):
        """Lock a model to prevent unloading"""
        if name in self._models:
            self._models[name].is_locked = True
    
    def unlock_model(self, name: str):
        """Unlock a model to allow unloading"""
        if name in self._models:
            self._models[name].is_locked = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory and performance statistics"""
        with self._lock:
            loaded_models = {
                name: {
                    "size_mb": info.size_mb,
                    "last_used": info.last_used.isoformat(),
                    "access_count": info.access_count,
                    "priority": info.priority.name
                }
                for name, info in self._models.items()
            }
            
            return {
                "loaded_models": loaded_models,
                "num_loaded": len(self._models),
                "memory_usage_mb": self._current_memory_mb,
                "memory_limit_mb": self._memory_limit_mb,
                "memory_usage_percent": (self._current_memory_mb / self._memory_limit_mb) * 100,
                "swap_count": self._swap_count,
                "load_count": self._load_count,
                "system_memory_percent": psutil.virtual_memory().percent
            }
    
    def _background_gc(self):
        """Background garbage collection and monitoring"""
        while self._running:
            try:
                time.sleep(self.config.memory.gc_interval_seconds)
                
                with self._lock:
                    # Check for models to unload
                    current_time = datetime.now()
                    timeout = timedelta(seconds=self.config.memory.model_cache_timeout_seconds)
                    
                    for name, info in list(self._models.items()):
                        if (info.priority == ModelPriority.LOW and 
                            info.last_used < current_time - timeout and
                            not info.is_locked):
                            self._unload_model(name, swap=True)
                
                # Log stats periodically
                if self.config.log_performance_metrics:
                    stats = self.get_stats()
                    logger.info(f"Model Manager: {stats['num_loaded']} models, "
                              f"{stats['memory_usage_mb']:.1f}MB used "
                              f"({stats['memory_usage_percent']:.1f}%)")
                
            except Exception as e:
                logger.error(f"Error in background GC: {e}")
    
    def _start_background_tasks(self):
        """Start background monitoring and GC"""
        self._gc_thread = threading.Thread(target=self._background_gc, daemon=True)
        self._gc_thread.start()
    
    def shutdown(self):
        """Shutdown the model manager"""
        logger.info("Shutting down Model Manager...")
        self._running = False
        
        # Save all models to swap
        with self._lock:
            for name in list(self._models.keys()):
                self._unload_model(name, swap=True)
        
        self._executor.shutdown(wait=True)
        logger.info("Model Manager shutdown complete")

# Global model manager instance
_model_manager: Optional[ModelManager] = None

def get_model_manager() -> ModelManager:
    """Get or create the global model manager"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager