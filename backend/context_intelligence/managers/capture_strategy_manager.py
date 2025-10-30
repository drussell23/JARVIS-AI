"""
Capture Strategy Manager for JARVIS
====================================

Implements intelligent capture fallback strategies:

1. Primary: Capture specific window
2. Fallback 1: Capture entire space
3. Fallback 2: Use cached screenshot (if <60s old)
4. Fallback 3: Return user-friendly error

Uses Error Handling Matrix for graceful degradation.

This module provides a comprehensive capture strategy system that handles
various failure scenarios gracefully, maintains a cache of recent captures,
and integrates with the Error Handling Matrix for sophisticated fallback
behavior.

Author: Derek Russell
Date: 2025-10-19
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import json

logger = logging.getLogger(__name__)

# Import Error Handling Matrix
try:
    from .error_handling_matrix import (
        get_error_handling_matrix,
        initialize_error_handling_matrix,
        FallbackChain,
        ExecutionReport,
        ErrorMessageGenerator,
        ResultQuality
    )
    ERROR_MATRIX_AVAILABLE = True
except ImportError:
    ERROR_MATRIX_AVAILABLE = False
    logger.warning("Error Handling Matrix not available")


# ============================================================================
# CAPTURE CACHE
# ============================================================================

@dataclass
class CachedCapture:
    """Cached screenshot with metadata.
    
    Represents a cached screenshot capture with associated metadata including
    timing information, capture method, and window/space identifiers.
    
    Attributes:
        image: Image data (PIL Image, numpy array, etc.)
        window_id: Optional window identifier for window-specific captures
        space_id: Space identifier where the capture was taken
        timestamp: When the capture was created
        method: Method used for capture (e.g., 'window_capture', 'space_capture')
        metadata: Additional metadata dictionary
    """
    image: Any  # Image data (PIL Image, numpy array, etc.)
    window_id: Optional[int]
    space_id: int
    timestamp: datetime
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self, max_age_seconds: float = 60.0) -> bool:
        """Check if cache is still valid based on age.
        
        Args:
            max_age_seconds: Maximum age in seconds before cache is considered stale
            
        Returns:
            True if cache is still valid, False otherwise
        """
        age = (datetime.now() - self.timestamp).total_seconds()
        return age < max_age_seconds

    def age_seconds(self) -> float:
        """Get age of cached capture in seconds.
        
        Returns:
            Age in seconds since capture was created
        """
        return (datetime.now() - self.timestamp).total_seconds()


class CaptureCache:
    """
    Manages cached screenshots with TTL.

    Features:
    - Time-based expiration
    - Space-based caching
    - Window-based caching
    - Automatic cleanup
    
    The cache maintains separate indexes for space-based and window-based
    captures to enable efficient lookups for different capture scenarios.
    """

    def __init__(self, default_ttl: float = 60.0, max_entries: int = 100):
        """
        Initialize capture cache.

        Args:
            default_ttl: Default time-to-live in seconds for cached entries
            max_entries: Maximum number of cache entries before cleanup
        """
        self.default_ttl = default_ttl
        self.max_entries = max_entries

        # Cache by space_id
        self._space_cache: Dict[int, CachedCapture] = {}

        # Cache by window_id
        self._window_cache: Dict[int, CachedCapture] = {}

        logger.info(f"[CAPTURE-CACHE] Initialized (ttl={default_ttl}s, max_entries={max_entries})")

    def get_by_space(self, space_id: int, max_age: Optional[float] = None) -> Optional[CachedCapture]:
        """
        Get cached capture by space ID.

        Args:
            space_id: Space ID to look up
            max_age: Maximum age in seconds (uses default if not provided)

        Returns:
            CachedCapture if valid entry exists, None otherwise
        """
        max_age = max_age or self.default_ttl

        cached = self._space_cache.get(space_id)
        if cached and cached.is_valid(max_age):
            logger.info(f"[CAPTURE-CACHE] ✅ Cache hit for space {space_id} (age={cached.age_seconds():.1f}s)")
            return cached

        # Remove stale cache
        if cached:
            logger.debug(f"[CAPTURE-CACHE] Cache expired for space {space_id}")
            self._space_cache.pop(space_id, None)

        return None

    def get_by_window(self, window_id: int, max_age: Optional[float] = None) -> Optional[CachedCapture]:
        """Get cached capture by window ID.
        
        Args:
            window_id: Window ID to look up
            max_age: Maximum age in seconds (uses default if not provided)
            
        Returns:
            CachedCapture if valid entry exists, None otherwise
        """
        max_age = max_age or self.default_ttl

        cached = self._window_cache.get(window_id)
        if cached and cached.is_valid(max_age):
            logger.info(f"[CAPTURE-CACHE] ✅ Cache hit for window {window_id} (age={cached.age_seconds():.1f}s)")
            return cached

        # Remove stale cache
        if cached:
            logger.debug(f"[CAPTURE-CACHE] Cache expired for window {window_id}")
            self._window_cache.pop(window_id, None)

        return None

    def store(self, capture: CachedCapture) -> None:
        """Store capture in cache.
        
        Args:
            capture: CachedCapture instance to store
        """
        # Store by space
        self._space_cache[capture.space_id] = capture

        # Store by window if available
        if capture.window_id is not None:
            self._window_cache[capture.window_id] = capture

        # Cleanup if too many entries
        self._cleanup_old_entries()

        logger.debug(f"[CAPTURE-CACHE] Stored capture for space={capture.space_id}, window={capture.window_id}")

    def _cleanup_old_entries(self) -> None:
        """Remove old entries if cache is too large."""
        total_entries = len(self._space_cache) + len(self._window_cache)

        if total_entries > self.max_entries:
            # Remove oldest entries
            space_entries = [(sid, c.timestamp) for sid, c in self._space_cache.items()]
            space_entries.sort(key=lambda x: x[1])

            # Remove oldest 10%
            remove_count = max(1, int(len(space_entries) * 0.1))
            for sid, _ in space_entries[:remove_count]:
                self._space_cache.pop(sid, None)

            logger.info(f"[CAPTURE-CACHE] Cleaned up {remove_count} old entries")

    def clear(self) -> None:
        """Clear all cache entries."""
        self._space_cache.clear()
        self._window_cache.clear()
        logger.info("[CAPTURE-CACHE] Cleared all cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics including entry counts and configuration
        """
        return {
            "space_entries": len(self._space_cache),
            "window_entries": len(self._window_cache),
            "total_entries": len(self._space_cache) + len(self._window_cache),
            "max_entries": self.max_entries,
            "default_ttl": self.default_ttl
        }


# ============================================================================
# CAPTURE STRATEGY MANAGER
# ============================================================================

class CaptureStrategyManager:
    """
    Manages intelligent capture strategies with fallbacks.

    Implements the capture fallback chain:
    1. Primary: Capture specific window
    2. Fallback 1: Capture entire space
    3. Fallback 2: Use cached screenshot (if <60s old)
    4. Fallback 3: Return user-friendly error

    Uses Error Handling Matrix for graceful degradation when available,
    otherwise falls back to sequential execution with basic error handling.
    
    Attributes:
        cache: CaptureCache instance for managing cached screenshots
        error_matrix: Optional Error Handling Matrix for sophisticated fallback behavior
    """

    def __init__(
        self,
        cache_ttl: float = 60.0,
        max_cache_entries: int = 100,
        enable_error_matrix: bool = True
    ):
        """
        Initialize capture strategy manager.

        Args:
            cache_ttl: Cache time-to-live in seconds
            max_cache_entries: Maximum number of cache entries
            enable_error_matrix: Whether to use Error Handling Matrix if available
        """
        self.cache = CaptureCache(default_ttl=cache_ttl, max_entries=max_cache_entries)

        # Initialize Error Handling Matrix
        self.error_matrix = None
        if enable_error_matrix and ERROR_MATRIX_AVAILABLE:
            try:
                self.error_matrix = get_error_handling_matrix()
                if not self.error_matrix:
                    self.error_matrix = initialize_error_handling_matrix(
                        default_timeout=30.0,
                        aggregation_strategy="first_success",
                        recovery_strategy="continue"
                    )
                logger.info("✅ Error Handling Matrix available for capture strategies")
            except Exception as e:
                logger.warning(f"Failed to initialize Error Handling Matrix: {e}")

        logger.info(f"[CAPTURE-STRATEGY] Initialized (cache_ttl={cache_ttl}s, matrix_enabled={self.error_matrix is not None})")

    async def capture_with_fallbacks(
        self,
        space_id: int,
        window_id: Optional[int] = None,
        window_capture_func: Optional[Callable] = None,
        space_capture_func: Optional[Callable] = None,
        cache_max_age: Optional[float] = None
    ) -> Tuple[bool, Any, str]:
        """
        Capture with intelligent fallbacks.

        Attempts to capture a screenshot using multiple strategies in order of preference:
        1. Window-specific capture (if window_id provided)
        2. Space-wide capture
        3. Cached screenshot (if recent enough)
        4. User-friendly error message

        Args:
            space_id: Space ID to capture
            window_id: Optional window ID for specific window capture
            window_capture_func: Async function to capture window (func(window_id, space_id))
            space_capture_func: Async function to capture space (func(space_id))
            cache_max_age: Maximum cache age in seconds

        Returns:
            Tuple of (success: bool, image_data: Any, message: str)
            
        Example:
            >>> manager = CaptureStrategyManager()
            >>> success, image, msg = await manager.capture_with_fallbacks(
            ...     space_id=1,
            ...     window_id=123,
            ...     window_capture_func=my_window_capture,
            ...     space_capture_func=my_space_capture
            ... )
            >>> if success:
            ...     print(f"Captured: {msg}")
        """
        logger.info(f"[CAPTURE-STRATEGY] Starting capture for space={space_id}, window={window_id}")

        cache_max_age = cache_max_age or self.cache.default_ttl

        # Use Error Handling Matrix if available
        if self.error_matrix:
            return await self._capture_with_matrix(
                space_id,
                window_id,
                window_capture_func,
                space_capture_func,
                cache_max_age
            )

        # Fallback to simple sequential capture
        return await self._capture_sequential(
            space_id,
            window_id,
            window_capture_func,
            space_capture_func,
            cache_max_age
        )

    async def _capture_with_matrix(
        self,
        space_id: int,
        window_id: Optional[int],
        window_capture_func: Optional[Callable],
        space_capture_func: Optional[Callable],
        cache_max_age: float
    ) -> Tuple[bool, Any, str]:
        """Capture using Error Handling Matrix.
        
        Args:
            space_id: Space ID to capture
            window_id: Optional window ID
            window_capture_func: Window capture function
            space_capture_func: Space capture function
            cache_max_age: Maximum cache age in seconds
            
        Returns:
            Tuple of (success, image_data, message)
        """
        logger.info(f"[CAPTURE-STRATEGY] Using Error Handling Matrix")

        # Build fallback chain
        chain = FallbackChain(f"capture_space_{space_id}")

        # 1. Primary: Capture specific window (if window_id provided)
        if window_id is not None and window_capture_func:
            async def capture_window():
                logger.info(f"[CAPTURE-STRATEGY] Attempting window capture: {window_id}")
                result = await window_capture_func(window_id, space_id)

                # Cache the result
                if result:
                    cached = CachedCapture(
                        image=result,
                        window_id=window_id,
                        space_id=space_id,
                        timestamp=datetime.now(),
                        method="window_capture"
                    )
                    self.cache.store(cached)

                return result

            chain.add_primary(capture_window, name=f"window_{window_id}", timeout=10.0)

        # 2. Fallback 1: Capture entire space
        if space_capture_func:
            async def capture_space():
                logger.info(f"[CAPTURE-STRATEGY] Attempting space capture: {space_id}")
                result = await space_capture_func(space_id)

                # Cache the result
                if result:
                    cached = CachedCapture(
                        image=result,
                        window_id=None,
                        space_id=space_id,
                        timestamp=datetime.now(),
                        method="space_capture"
                    )
                    self.cache.store(cached)

                return result

            if window_id is not None and window_capture_func:
                chain.add_fallback(capture_space, name=f"space_{space_id}", timeout=15.0)
            else:
                # If no window capture, make space capture primary
                chain.add_primary(capture_space, name=f"space_{space_id}", timeout=15.0)

        # 3. Fallback 2: Use cached screenshot
        async def use_cache():
            logger.info(f"[CAPTURE-STRATEGY] Attempting cache lookup: space={space_id}")

            # Try window cache first if window_id provided
            if window_id is not None:
                cached = self.cache.get_by_window(window_id, max_age=cache_max_age)
                if cached:
                    logger.info(f"[CAPTURE-STRATEGY] ✅ Using cached window capture (age={cached.age_seconds():.1f}s)")
                    return cached.image

            # Try space cache
            cached = self.cache.get_by_space(space_id, max_age=cache_max_age)
            if cached:
                logger.info(f"[CAPTURE-STRATEGY] ✅ Using cached space capture (age={cached.age_seconds():.1f}s)")
                return cached.image

            # No valid cache
            raise Exception(f"No valid cache for space {space_id} (max_age={cache_max_age}s)")

        chain.add_secondary(use_cache, name="cache", timeout=1.0)

        # 4. Fallback 3: Error message (will be generated by ErrorMessageGenerator)

        # Execute chain
        report = await self.error_matrix.execute_chain(chain, stop_on_success=True)

        # Process result
        if report.success and report.final_result:
            # Determine message based on quality
            if report.final_status == ResultQuality.FULL:
                message = f"Captured space {space_id}"
                if window_id:
                    message = f"Captured window {window_id} in space {space_id}"
            elif report.final_status == ResultQuality.DEGRADED:
                message = f"Captured space {space_id} using fallback method"
            elif report.final_status == ResultQuality.PARTIAL:
                message = f"Using cached capture for space {space_id}"
            else:
                message = f"Captured space {space_id} with minimal quality"

            # Add warnings if any
            if report.warnings:
                message += f" ({len(report.warnings)} warning(s))"

            logger.info(f"[CAPTURE-STRATEGY] ✅ {message}")

            return True, report.final_result, message

        else:
            # Generate user-friendly error
            error_msg = ErrorMessageGenerator.generate_message(
                report,
                include_technical=False,
                include_suggestions=True
            )

            logger.error(f"[CAPTURE-STRATEGY] ❌ Capture failed:\n{error_msg}")

            return False, None, f"Unable to capture Space {space_id}"

    async def _capture_sequential(
        self,
        space_id: int,
        window_id: Optional[int],
        window_capture_func: Optional[Callable],
        space_capture_func: Optional[Callable],
        cache_max_age: float
    ) -> Tuple[bool, Any, str]:
        """Fallback sequential capture (without Error Handling Matrix).
        
        Args:
            space_id: Space ID to capture
            window_id: Optional window ID
            window_capture_func: Window capture function
            space_capture_func: Space capture function
            cache_max_age: Maximum cache age in seconds
            
        Returns:
            Tuple of (success, image_data, message)
        """
        logger.warning("[CAPTURE-STRATEGY] Error Handling Matrix not available, using sequential fallback")

        # 1. Try window capture
        if window_id is not None and window_capture_func:
            try:
                logger.info(f"[CAPTURE-STRATEGY] Attempting window capture: {window_id}")
                result = await asyncio.wait_for(window_capture_func(window_id, space_id), timeout=10.0)

                if result:
                    # Cache and return
                    cached = CachedCapture(
                        image=result,
                        window_id=window_id,
                        space_id=space_id,
                        timestamp=datetime.now(),
                        method="window_capture"
                    )
                    self.cache.store(cached)

                    return True, result, f"Captured window {window_id} in space {space_id}"

            except Exception as e:
                logger.warning(f"[CAPTURE-STRATEGY] Window capture failed: {e}")

        # 2. Try space capture
        if space_capture_func:
            try:
                logger.info(f"[CAPTURE-STRATEGY] Attempting space capture: {space_id}")
                result = await asyncio.wait_for(space_capture_func(space_id), timeout=15.0)

                if result:
                    # Cache and return
                    cached = CachedCapture(
                        image=result,
                        window_id=None,
                        space_id=space_id,
                        timestamp=datetime.now(),
                        method="space_capture"
                    )
                    self.cache.store(cached)

                    return True, result, f"Captured space {space_id}"

            except Exception as e:
                logger.warning(f"[CAPTURE-STRATEGY] Space capture failed: {e}")

        # 3. Try cache
        try:
            logger.info(f"[CAPTURE-STRATEGY] Attempting cache lookup")

            # Try window cache first
            if window_id is not None:
                cached = self.cache.get_by_window(window_id, max_age=cache_max_age)
                if cached:
                    return True, cached.image, f"Using cached capture (age={cached.age_seconds():.1f}s)"

            # Try space cache
            cached = self.cache.get_by_space(space_id, max_age=cache_max_age)
            if cached:
                return True, cached.image, f"Using cached capture (age={cached.age_seconds():.1f}s)"

        except Exception as e:
            logger.warning(f"[CAPTURE-STRATEGY] Cache lookup failed: {e}")

        # 4. All methods failed
        logger.error(f"[CAPTURE-STRATEGY] All capture methods failed for space {space_id}")
        return False, None, f"Unable to capture Space {space_id}"

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary containing cache statistics and configuration
        """
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear all cached captures."""
        self.cache.clear()


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

_global_manager: Optional[CaptureStrategyManager] = None


def get_capture_strategy_manager() -> Optional[CaptureStrategyManager]:
    """Get the global capture strategy manager instance.
    
    Returns:
        The global CaptureStrategyManager instance if initialized, None otherwise
    """
    return _global_manager


def initialize_capture_strategy_manager(
    cache_ttl: float = 60.0,
    max_cache_entries: int = 100,
    enable_error_matrix: bool = True
) -> CaptureStrategyManager:
    """Initialize the global capture strategy manager.
    
    Args:
        cache_ttl: Cache time-to-live in seconds
        max_cache_entries: Maximum number of cache entries
        enable_error_matrix: Whether to enable Error Handling Matrix integration
        
    Returns:
        The initialized CaptureStrategyManager instance
        
    Example:
        >>> manager = initialize_capture_strategy_manager(
        ...     cache_ttl=120.0,
        ...     max_cache_entries=200,
        ...     enable_error_matrix=True
        ... )
        >>> print("Capture strategy manager initialized")
    """
    global _global_manager
    _global_manager = CaptureStrategyManager(
        cache_ttl=cache_ttl,
        max_cache_entries=max_cache_entries,
        enable_error_matrix=enable_error_matrix
    )
    logger.info("[CAPTURE-STRATEGY] Global instance initialized")
    return _global_manager