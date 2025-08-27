"""
Phase 2: Smart Caching System
Inference result caching, gradient accumulation, feature map reuse
Temporal coherence exploitation for video/continuous monitoring
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import OrderedDict, deque
import time
import hashlib
import pickle
import logging
from functools import lru_cache
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry with metadata"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    size_bytes: int = 0
    computation_time: float = 0.0
    accuracy_score: float = 1.0  # For adaptive caching


class SmartCache:
    """
    Advanced caching system with:
    - LRU + frequency-based eviction
    - Temporal coherence detection
    - Gradient accumulation
    - Feature map reuse
    """
    
    def __init__(self, 
                 max_size_mb: float = 500,
                 ttl_seconds: float = 60.0,
                 enable_compression: bool = True):
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        
        # Multiple cache levels
        self.inference_cache = OrderedDict()  # Results cache
        self.feature_cache = OrderedDict()    # Feature map cache
        self.gradient_cache = OrderedDict()   # Gradient accumulation
        self.temporal_cache = deque(maxlen=30)  # Temporal coherence
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size': 0,
            'computation_saved': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info(f"✅ Smart Cache initialized")
        logger.info(f"   Max size: {max_size_mb}MB")
        logger.info(f"   TTL: {ttl_seconds}s")
        logger.info(f"   Compression: {enable_compression}")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Create hashable representation
        key_data = []
        
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # For tensors, use shape, dtype, and sample of data
                key_data.append((
                    arg.shape,
                    str(arg.dtype),
                    arg.flatten()[:10].tolist() if arg.numel() > 0 else []
                ))
            elif isinstance(arg, np.ndarray):
                key_data.append((
                    arg.shape,
                    str(arg.dtype),
                    arg.flatten()[:10].tolist() if arg.size > 0 else []
                ))
            else:
                key_data.append(str(arg))
        
        # Add kwargs
        for k, v in sorted(kwargs.items()):
            key_data.append((k, str(v)))
        
        # Generate hash
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        if isinstance(obj, torch.Tensor):
            return obj.numel() * obj.element_size()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, dict):
            return sum(self._estimate_size(v) for v in obj.values()) + 1000
        elif isinstance(obj, (list, tuple)):
            return sum(self._estimate_size(v) for v in obj) + 100
        else:
            # Fallback to pickle size estimate
            try:
                return len(pickle.dumps(obj))
            except:
                return 1000  # Default estimate
    
    def _evict_if_needed(self, required_size: int):
        """Evict entries if cache is full"""
        with self._lock:
            while self.stats['current_size'] + required_size > self.max_size_bytes:
                if not self.inference_cache:
                    break
                
                # Evict least recently used with lowest score
                min_score = float('inf')
                evict_key = None
                
                for key, entry in self.inference_cache.items():
                    # Score based on recency, frequency, and accuracy
                    age = time.time() - entry.timestamp
                    score = (entry.access_count / (age + 1)) * entry.accuracy_score
                    
                    if score < min_score:
                        min_score = score
                        evict_key = key
                
                if evict_key:
                    evicted = self.inference_cache.pop(evict_key)
                    self.stats['current_size'] -= evicted.size_bytes
                    self.stats['evictions'] += 1
                else:
                    break
    
    @contextmanager
    def cached_computation(self, cache_key: Optional[str] = None):
        """Context manager for cached computations"""
        start_time = time.time()
        yield_value = None
        
        def cache_result(value):
            nonlocal yield_value
            yield_value = value
            return value
        
        yield cache_result
        
        computation_time = time.time() - start_time
        
        # Store in cache if key provided
        if cache_key and yield_value is not None:
            self.put(cache_key, yield_value, computation_time=computation_time)
    
    def get(self, key: str, default=None) -> Any:
        """Get from cache with TTL check"""
        with self._lock:
            if key in self.inference_cache:
                entry = self.inference_cache[key]
                
                # Check TTL
                if time.time() - entry.timestamp > self.ttl_seconds:
                    self.inference_cache.pop(key)
                    self.stats['current_size'] -= entry.size_bytes
                    return default
                
                # Update access info
                entry.access_count += 1
                entry.timestamp = time.time()
                
                # Move to end (most recently used)
                self.inference_cache.move_to_end(key)
                
                self.stats['hits'] += 1
                self.stats['computation_saved'] += entry.computation_time
                
                return entry.value
            
            self.stats['misses'] += 1
            return default
    
    def put(self, key: str, value: Any, 
            computation_time: float = 0.0,
            accuracy_score: float = 1.0):
        """Store in cache"""
        size = self._estimate_size(value)
        
        with self._lock:
            # Evict if needed
            self._evict_if_needed(size)
            
            # Create entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size,
                computation_time=computation_time,
                accuracy_score=accuracy_score
            )
            
            # Store
            self.inference_cache[key] = entry
            self.stats['current_size'] += size
    
    def cache_features(self, layer_name: str, features: torch.Tensor) -> str:
        """Cache intermediate feature maps"""
        key = f"features_{layer_name}_{self._generate_key(features)}"
        
        with self._lock:
            if key not in self.feature_cache:
                # Compress if enabled
                if self.enable_compression and features.dtype == torch.float32:
                    # Simple compression: convert to half precision
                    compressed = features.half()
                    self.feature_cache[key] = compressed
                else:
                    self.feature_cache[key] = features.clone()
                
                # Limit feature cache size
                if len(self.feature_cache) > 100:
                    self.feature_cache.popitem(last=False)
        
        return key
    
    def get_features(self, feature_key: str) -> Optional[torch.Tensor]:
        """Retrieve cached features"""
        with self._lock:
            if feature_key in self.feature_cache:
                features = self.feature_cache[feature_key]
                
                # Decompress if needed
                if features.dtype == torch.float16:
                    return features.float()
                return features
            
            return None
    
    def accumulate_gradients(self, param_name: str, gradient: torch.Tensor):
        """Accumulate gradients for efficient updates"""
        with self._lock:
            if param_name not in self.gradient_cache:
                self.gradient_cache[param_name] = []
            
            # Add gradient
            self.gradient_cache[param_name].append(gradient.clone())
            
            # Limit accumulation
            if len(self.gradient_cache[param_name]) > 10:
                self.gradient_cache[param_name].pop(0)
    
    def get_accumulated_gradients(self, param_name: str) -> Optional[torch.Tensor]:
        """Get accumulated gradients"""
        with self._lock:
            if param_name in self.gradient_cache:
                gradients = self.gradient_cache[param_name]
                if gradients:
                    # Return mean of accumulated gradients
                    return torch.stack(gradients).mean(dim=0)
            
            return None
    
    def exploit_temporal_coherence(self, current_frame: torch.Tensor) -> Optional[Dict[str, Any]]:
        """
        Exploit temporal coherence for video/continuous monitoring
        Returns cached result if frame is similar to recent ones
        """
        with self._lock:
            # Add to temporal cache
            self.temporal_cache.append({
                'frame': current_frame,
                'timestamp': time.time(),
                'result': None
            })
            
            # Check similarity with recent frames
            if len(self.temporal_cache) > 1:
                prev_entry = self.temporal_cache[-2]
                prev_frame = prev_entry['frame']
                
                # Calculate similarity (simple MSE)
                if current_frame.shape == prev_frame.shape:
                    diff = (current_frame - prev_frame).abs().mean().item()
                    
                    # If very similar, reuse previous result
                    if diff < 0.01 and prev_entry['result'] is not None:
                        logger.debug(f"Temporal coherence hit: diff={diff:.4f}")
                        self.stats['hits'] += 1
                        return prev_entry['result']
            
            return None
    
    def update_temporal_result(self, result: Dict[str, Any]):
        """Update result for latest temporal frame"""
        with self._lock:
            if self.temporal_cache:
                self.temporal_cache[-1]['result'] = result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(1, total_requests)
            
            return {
                'hit_rate': hit_rate,
                'total_hits': self.stats['hits'],
                'total_misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'current_size_mb': self.stats['current_size'] / 1024 / 1024,
                'computation_saved_s': self.stats['computation_saved'],
                'cache_sizes': {
                    'inference': len(self.inference_cache),
                    'features': len(self.feature_cache),
                    'gradients': len(self.gradient_cache),
                    'temporal': len(self.temporal_cache)
                }
            }
    
    def clear(self):
        """Clear all caches"""
        with self._lock:
            self.inference_cache.clear()
            self.feature_cache.clear()
            self.gradient_cache.clear()
            self.temporal_cache.clear()
            self.stats['current_size'] = 0
            logger.info("All caches cleared")


class CachedModel(nn.Module):
    """Wrapper to add caching to any model"""
    
    def __init__(self, model: nn.Module, cache: SmartCache):
        super().__init__()
        self.model = model
        self.cache = cache
        
        # Register hooks for feature caching
        self._register_feature_hooks()
    
    def _register_feature_hooks(self):
        """Register forward hooks to cache features"""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    # Cache feature maps for important layers
                    if any(key in name for key in ['conv', 'linear', 'attention']):
                        self.cache.cache_features(name, output)
            return hook
        
        for name, module in self.model.named_modules():
            module.register_forward_hook(make_hook(name))
    
    def forward(self, x: torch.Tensor, use_cache: bool = True) -> torch.Tensor:
        """Forward with caching"""
        if use_cache:
            # Generate cache key
            cache_key = self.cache._generate_key(x)
            
            # Check cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Check temporal coherence
            temporal_result = self.cache.exploit_temporal_coherence(x)
            if temporal_result is not None:
                return temporal_result['output']
            
            # Compute with timing
            start_time = time.time()
            with self.cache.cached_computation(cache_key) as cache_result:
                output = self.model(x)
                computation_time = time.time() - start_time
                
                # Cache result
                result = {
                    'output': output,
                    'computation_time': computation_time
                }
                cache_result(output)
                
                # Update temporal cache
                self.cache.update_temporal_result(result)
                
                return output
        else:
            return self.model(x)


# Decorator for cached functions
def smart_cache(cache_instance: SmartCache):
    """Decorator to add caching to any function"""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_instance._generate_key(*args, **kwargs)
            
            # Check cache
            cached = cache_instance.get(cache_key)
            if cached is not None:
                return cached
            
            # Compute
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Store in cache
            cache_instance.put(cache_key, result, computation_time)
            
            return result
        
        return wrapper
    return decorator


# Test function
def test_smart_cache():
    """Test smart caching system"""
    cache = SmartCache(max_size_mb=100, ttl_seconds=60)
    
    # Test basic caching
    @smart_cache(cache)
    def expensive_computation(x):
        time.sleep(0.1)  # Simulate expensive computation
        return x ** 2
    
    # First call - miss
    result1 = expensive_computation(5)
    print(f"Result 1: {result1}")
    
    # Second call - hit
    result2 = expensive_computation(5)
    print(f"Result 2: {result2}")
    
    # Test temporal coherence
    frames = [torch.randn(3, 224, 224) for _ in range(5)]
    frames[2] = frames[1] + 0.001  # Very similar to previous
    
    for i, frame in enumerate(frames):
        result = cache.exploit_temporal_coherence(frame)
        if result:
            print(f"Frame {i}: Used temporal coherence")
        else:
            print(f"Frame {i}: Computed new result")
            cache.update_temporal_result({'output': f"result_{i}"})
    
    # Print stats
    print(f"\nCache stats: {cache.get_stats()}")


if __name__ == "__main__":
    test_smart_cache()