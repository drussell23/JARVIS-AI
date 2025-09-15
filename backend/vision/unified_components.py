"""
Unified component interfaces that automatically use the best available implementation.
These wrappers provide consistent APIs regardless of whether Rust or Python is used.
"""

import logging
from typing import Optional, Any, Dict, Union
import numpy as np
import asyncio
from abc import ABC, abstractmethod

from .dynamic_component_loader import (
    get_component_loader,
    ComponentType,
    ImplementationType
)

logger = logging.getLogger(__name__)

class UnifiedBloomFilter:
    """
    Unified bloom filter that automatically uses Rust when available.
    Provides consistent interface for both implementations.
    """
    
    def __init__(self, size_mb: float = 10.0, num_hashes: int = 7):
        """Initialize bloom filter with the best available implementation."""
        self.size_mb = size_mb
        self.num_hashes = num_hashes
        self._impl = None
        self._impl_type = None
        self._loader = get_component_loader()
        
        # Register for component changes
        self._loader.register_change_callback(
            ComponentType.BLOOM_FILTER,
            self._on_implementation_change
        )
        
        # Get initial implementation
        self._update_implementation()
        
    def _update_implementation(self):
        """Update to the current best implementation."""
        self._impl = self._loader.get_component(ComponentType.BLOOM_FILTER)
        self._impl_type = self._loader.get_active_implementation(ComponentType.BLOOM_FILTER)
        
        if self._impl is None:
            # Create fallback if no implementation available
            logger.warning("No bloom filter implementation available, creating basic fallback")
            from .bloom_filter import PythonBloomFilter
            self._impl = PythonBloomFilter(self.size_mb, self.num_hashes)
            self._impl_type = ImplementationType.PYTHON
            
        logger.info(f"Bloom filter using {self._impl_type.value if self._impl_type else 'fallback'} implementation")
        
    def _on_implementation_change(self, comp_type: ComponentType, new_impl):
        """Handle implementation changes."""
        logger.info(f"Bloom filter implementation changed to {new_impl.implementation.value}")
        
        # Get current state if possible
        old_impl = self._impl
        
        # Update implementation
        self._update_implementation()
        
        # Try to migrate state if both support it
        if hasattr(old_impl, 'export_state') and hasattr(self._impl, 'import_state'):
            try:
                state = old_impl.export_state()
                self._impl.import_state(state)
                logger.info("Successfully migrated bloom filter state")
            except Exception as e:
                logger.warning(f"Could not migrate bloom filter state: {e}")
                
    def add(self, item: Union[bytes, str]):
        """Add an item to the bloom filter."""
        if isinstance(item, str):
            item = item.encode('utf-8')
            
        if hasattr(self._impl, 'add'):
            self._impl.add(item)
        else:
            # Handle different method names
            if hasattr(self._impl, 'insert'):
                self._impl.insert(item)
            else:
                raise AttributeError(f"Bloom filter implementation missing add method")
                
    def contains(self, item: Union[bytes, str]) -> bool:
        """Check if an item might be in the set."""
        if isinstance(item, str):
            item = item.encode('utf-8')
            
        if hasattr(self._impl, 'contains'):
            return self._impl.contains(item)
        elif hasattr(self._impl, 'might_contain'):
            return self._impl.might_contain(item)
        elif hasattr(self._impl, 'check'):
            return self._impl.check(item)
        else:
            raise AttributeError(f"Bloom filter implementation missing contains method")
            
    def clear(self):
        """Clear the bloom filter."""
        if hasattr(self._impl, 'clear'):
            self._impl.clear()
        elif hasattr(self._impl, 'reset'):
            self._impl.reset()
        else:
            # Recreate if no clear method
            self._update_implementation()
            
    def get_saturation(self) -> float:
        """Get the saturation level (0.0 to 1.0)."""
        if hasattr(self._impl, 'get_saturation'):
            return self._impl.get_saturation()
        elif hasattr(self._impl, 'saturation'):
            return self._impl.saturation()
        else:
            # Estimate if not available
            return 0.0
            
    @property
    def implementation_type(self) -> Optional[ImplementationType]:
        """Get the current implementation type."""
        return self._impl_type
        
    def __repr__(self):
        impl_name = self._impl_type.value if self._impl_type else "unknown"
        return f"UnifiedBloomFilter({self.size_mb}MB, impl={impl_name})"


class UnifiedSlidingWindow:
    """
    Unified sliding window that automatically uses Rust when available.
    """
    
    def __init__(self, window_size: int = 30, overlap_threshold: float = 0.9):
        """Initialize sliding window with the best available implementation."""
        self.window_size = window_size
        self.overlap_threshold = overlap_threshold
        self._impl = None
        self._impl_type = None
        self._loader = get_component_loader()
        
        # Register for component changes
        self._loader.register_change_callback(
            ComponentType.SLIDING_WINDOW,
            self._on_implementation_change
        )
        
        # Get initial implementation
        self._update_implementation()
        
    def _update_implementation(self):
        """Update to the current best implementation."""
        self._impl = self._loader.get_component(ComponentType.SLIDING_WINDOW)
        self._impl_type = self._loader.get_active_implementation(ComponentType.SLIDING_WINDOW)
        
        if self._impl is None:
            # Create fallback
            logger.warning("No sliding window implementation available, creating basic fallback")
            from .sliding_window import SlidingWindow
            self._impl = SlidingWindow(self.window_size)
            self._impl_type = ImplementationType.PYTHON
            
    def _on_implementation_change(self, comp_type: ComponentType, new_impl):
        """Handle implementation changes."""
        logger.info(f"Sliding window implementation changed to {new_impl.implementation.value}")
        self._update_implementation()
        
    def add_frame(self, frame_data: Union[Dict[str, Any], bytes], timestamp: float) -> Dict[str, Any]:
        """Add a frame and check for duplicates."""
        if hasattr(self._impl, 'add_frame'):
            return self._impl.add_frame(frame_data, timestamp)
        elif hasattr(self._impl, 'process_frame'):
            # Handle Rust interface
            if isinstance(frame_data, dict):
                # Extract raw bytes from dict
                data = frame_data.get('data', b'')
                if isinstance(data, str):
                    import base64
                    data = base64.b64decode(data)
            else:
                data = frame_data
                
            return self._impl.process_frame(data, timestamp)
        else:
            raise AttributeError("Sliding window implementation missing frame processing method")
            
    def get_window_frames(self) -> list:
        """Get current frames in the window."""
        if hasattr(self._impl, 'get_window_frames'):
            return self._impl.get_window_frames()
        elif hasattr(self._impl, 'get_frames'):
            return self._impl.get_frames()
        else:
            return []
            
    @property
    def implementation_type(self) -> Optional[ImplementationType]:
        """Get the current implementation type."""
        return self._impl_type


class UnifiedMemoryPool:
    """
    Unified memory pool that automatically uses Rust when available.
    """
    
    def __init__(self):
        """Initialize memory pool with the best available implementation."""
        self._impl = None
        self._impl_type = None
        self._loader = get_component_loader()
        
        # Register for component changes
        self._loader.register_change_callback(
            ComponentType.MEMORY_POOL,
            self._on_implementation_change
        )
        
        # Get initial implementation
        self._update_implementation()
        
    def _update_implementation(self):
        """Update to the current best implementation."""
        self._impl = self._loader.get_component(ComponentType.MEMORY_POOL)
        self._impl_type = self._loader.get_active_implementation(ComponentType.MEMORY_POOL)
        
        if self._impl is None:
            # Create fallback
            logger.warning("No memory pool implementation available, creating Python fallback")
            from .memory.python_memory_pool import PythonMemoryPool
            self._impl = PythonMemoryPool()
            self._impl_type = ImplementationType.PYTHON
            
    def _on_implementation_change(self, comp_type: ComponentType, new_impl):
        """Handle implementation changes."""
        logger.info(f"Memory pool implementation changed to {new_impl.implementation.value}")
        
        # Clean up old pool
        if self._impl and hasattr(self._impl, 'clear_pool'):
            self._impl.clear_pool()
            
        self._update_implementation()
        
    def allocate(self, size: int) -> Any:
        """Allocate a buffer of the specified size."""
        if hasattr(self._impl, 'allocate'):
            return self._impl.allocate(size)
        elif hasattr(self._impl, 'get_buffer'):
            return self._impl.get_buffer(size)
        else:
            # Basic fallback
            return np.zeros(size, dtype=np.uint8)
            
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if hasattr(self._impl, 'stats'):
            stats_data = self._impl.stats()
            # Normalize stats format
            if isinstance(stats_data, dict):
                stats_data['implementation'] = self._impl_type.value if self._impl_type else 'unknown'
                return stats_data
            else:
                return {'implementation': self._impl_type.value if self._impl_type else 'unknown'}
        else:
            return {'implementation': self._impl_type.value if self._impl_type else 'unknown'}
            
    @property
    def implementation_type(self) -> Optional[ImplementationType]:
        """Get the current implementation type."""
        return self._impl_type


class UnifiedZeroCopyPool:
    """
    Unified zero-copy pool that automatically uses Rust when available.
    """
    
    def __init__(self):
        """Initialize zero-copy pool with the best available implementation."""
        self._impl = None
        self._impl_type = None
        self._loader = get_component_loader()
        
        # Register for component changes
        self._loader.register_change_callback(
            ComponentType.ZERO_COPY_POOL,
            self._on_implementation_change
        )
        
        # Get initial implementation
        self._update_implementation()
        
    def _update_implementation(self):
        """Update to the current best implementation."""
        self._impl = self._loader.get_component(ComponentType.ZERO_COPY_POOL)
        self._impl_type = self._loader.get_active_implementation(ComponentType.ZERO_COPY_POOL)
        
        if self._impl is None:
            # Create fallback
            logger.warning("No zero-copy pool implementation available, creating Python fallback")
            from .memory.zero_copy_fallback import PythonZeroCopyPool
            self._impl = PythonZeroCopyPool()
            self._impl_type = ImplementationType.PYTHON
            
    def _on_implementation_change(self, comp_type: ComponentType, new_impl):
        """Handle implementation changes."""
        logger.info(f"Zero-copy pool implementation changed to {new_impl.implementation.value}")
        
        # Clean up old pool
        if self._impl and hasattr(self._impl, 'clear'):
            self._impl.clear()
            
        self._update_implementation()
        
    def allocate(self, size: int) -> Any:
        """Allocate a zero-copy buffer."""
        return self._impl.allocate(size)
        
    def stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if hasattr(self._impl, 'stats'):
            stats_data = self._impl.stats()
            if isinstance(stats_data, dict):
                stats_data['implementation'] = self._impl_type.value if self._impl_type else 'unknown'
                return stats_data
        return {'implementation': self._impl_type.value if self._impl_type else 'unknown'}
        
    @property
    def implementation_type(self) -> Optional[ImplementationType]:
        """Get the current implementation type."""
        return self._impl_type


# Convenience functions for creating unified components
def create_bloom_filter(size_mb: float = 10.0, num_hashes: int = 7) -> UnifiedBloomFilter:
    """Create a unified bloom filter that will use Rust when available."""
    return UnifiedBloomFilter(size_mb, num_hashes)

def create_sliding_window(window_size: int = 30, overlap_threshold: float = 0.9) -> UnifiedSlidingWindow:
    """Create a unified sliding window that will use Rust when available."""
    return UnifiedSlidingWindow(window_size, overlap_threshold)

def create_memory_pool() -> UnifiedMemoryPool:
    """Create a unified memory pool that will use Rust when available."""
    return UnifiedMemoryPool()

def create_zero_copy_pool() -> UnifiedZeroCopyPool:
    """Create a unified zero-copy pool that will use Rust when available."""
    return UnifiedZeroCopyPool()