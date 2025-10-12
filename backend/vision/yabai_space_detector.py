#!/usr/bin/env python3
"""
Yabai Space Detector - Advanced async integration with Yabai window manager
Provides robust, dynamic multi-space intelligence with comprehensive error handling
"""

import asyncio
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class YabaiStatus(Enum):
    """Yabai availability status"""
    AVAILABLE = "available"
    NOT_INSTALLED = "not_installed"
    NO_PERMISSIONS = "no_permissions"
    ERROR = "error"


@dataclass
class SpaceInfo:
    """Represents a Mission Control space"""
    index: int
    label: str
    display: int
    window_ids: List[int] = field(default_factory=list)
    is_visible: bool = False
    is_focused: bool = False
    space_type: str = "desktop"

    @property
    def window_count(self) -> int:
        """Number of windows in this space"""
        return len(self.window_ids)

    @property
    def display_name(self) -> str:
        """Human-readable space name"""
        return self.label if self.label else f"Desktop {self.index}"


@dataclass
class WindowInfo:
    """Represents a window"""
    window_id: int
    app_name: str
    title: str
    space_index: int
    display: int
    frame: Dict[str, float]
    is_minimized: bool = False
    is_visible: bool = True
    is_focused: bool = False

    @property
    def area(self) -> float:
        """Window area in pixels"""
        return self.frame.get('w', 0) * self.frame.get('h', 0)

    @property
    def is_substantial(self) -> bool:
        """Whether window is large enough to be meaningful"""
        return self.area > 10000  # > 100x100 pixels


@dataclass
class CachedData:
    """Cached Yabai query results"""
    spaces: List[SpaceInfo]
    windows: List[WindowInfo]
    timestamp: datetime

    def is_expired(self, ttl_seconds: int = 5) -> bool:
        """Check if cache is expired"""
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds


class YabaiSpaceDetector:
    """
    Advanced async Yabai integration for multi-space intelligence

    Features:
    - Async subprocess calls for non-blocking queries
    - Intelligent caching with configurable TTL
    - Comprehensive error handling with fallbacks
    - Automatic Yabai availability detection
    - Multi-display support
    - Performance monitoring
    """

    def __init__(
        self,
        cache_ttl: int = 5,
        query_timeout: int = 5,
        enable_cache: bool = True
    ):
        """
        Initialize Yabai space detector

        Args:
            cache_ttl: Cache time-to-live in seconds
            query_timeout: Max time for Yabai queries in seconds
            enable_cache: Whether to enable query caching
        """
        self.cache_ttl = cache_ttl
        self.query_timeout = query_timeout
        self.enable_cache = enable_cache

        self._cache: Optional[CachedData] = None
        self._yabai_path: Optional[str] = None
        self._yabai_status: Optional[YabaiStatus] = None
        self._lock = asyncio.Lock()

        # Performance metrics
        self._query_count = 0
        self._cache_hits = 0
        self._total_query_time = 0.0

    async def check_availability(self) -> YabaiStatus:
        """
        Check if Yabai is installed and accessible

        Returns:
            YabaiStatus indicating availability
        """
        if self._yabai_status is not None:
            return self._yabai_status

        try:
            # Check if yabai command exists
            yabai_path = shutil.which('yabai')
            if not yabai_path:
                logger.warning("[YABAI] Yabai not found in PATH")
                self._yabai_status = YabaiStatus.NOT_INSTALLED
                return self._yabai_status

            self._yabai_path = yabai_path

            # Try a simple query to verify it works
            process = await asyncio.create_subprocess_exec(
                yabai_path, '-m', 'query', '--spaces',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.query_timeout
                )

                if process.returncode == 0:
                    logger.info("[YABAI] ✅ Yabai is available and functional")
                    self._yabai_status = YabaiStatus.AVAILABLE
                else:
                    error_msg = stderr.decode('utf-8', errors='ignore')
                    if 'accessibility' in error_msg.lower():
                        logger.warning("[YABAI] ⚠️ Yabai requires Accessibility permissions")
                        self._yabai_status = YabaiStatus.NO_PERMISSIONS
                    else:
                        logger.error(f"[YABAI] ❌ Yabai query failed: {error_msg}")
                        self._yabai_status = YabaiStatus.ERROR

            except asyncio.TimeoutError:
                logger.error(f"[YABAI] ❌ Yabai query timed out after {self.query_timeout}s")
                self._yabai_status = YabaiStatus.ERROR

        except Exception as e:
            logger.error(f"[YABAI] ❌ Error checking Yabai availability: {e}")
            self._yabai_status = YabaiStatus.ERROR

        return self._yabai_status

    async def get_spaces(self, force_refresh: bool = False) -> List[SpaceInfo]:
        """
        Get all Mission Control spaces

        Args:
            force_refresh: Bypass cache and query directly

        Returns:
            List of SpaceInfo objects
        """
        # Check cache first
        if (self.enable_cache and not force_refresh and
            self._cache and not self._cache.is_expired(self.cache_ttl)):
            self._cache_hits += 1
            logger.debug("[YABAI] 📦 Cache hit for spaces query")
            return self._cache.spaces

        # Check if Yabai is available
        status = await self.check_availability()
        if status != YabaiStatus.AVAILABLE:
            logger.warning(f"[YABAI] Yabai not available (status: {status.value}), using fallback")
            return await self._fallback_space_detection()

        # Query Yabai
        start_time = datetime.now()

        try:
            process = await asyncio.create_subprocess_exec(
                self._yabai_path, '-m', 'query', '--spaces',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.query_timeout
            )

            if process.returncode != 0:
                error_msg = stderr.decode('utf-8', errors='ignore')
                logger.error(f"[YABAI] Query failed: {error_msg}")
                return await self._fallback_space_detection()

            # Parse JSON response
            spaces_data = json.loads(stdout.decode('utf-8'))
            spaces = [
                SpaceInfo(
                    index=s['index'],
                    label=s.get('label', ''),
                    display=s['display'],
                    window_ids=s.get('windows', []),
                    is_visible=bool(s.get('visible', 0)),
                    is_focused=bool(s.get('focused', 0)),
                    space_type=s.get('type', 'desktop')
                )
                for s in spaces_data
            ]

            # Update metrics
            query_time = (datetime.now() - start_time).total_seconds()
            self._query_count += 1
            self._total_query_time += query_time

            logger.info(f"[YABAI] ✅ Retrieved {len(spaces)} spaces in {query_time:.3f}s")

            # Cache windows alongside spaces
            windows = await self._query_windows_internal()
            async with self._lock:
                self._cache = CachedData(
                    spaces=spaces,
                    windows=windows,
                    timestamp=datetime.now()
                )

            return spaces

        except asyncio.TimeoutError:
            logger.error(f"[YABAI] ❌ Spaces query timed out after {self.query_timeout}s")
            return await self._fallback_space_detection()
        except json.JSONDecodeError as e:
            logger.error(f"[YABAI] ❌ Failed to parse spaces JSON: {e}")
            return await self._fallback_space_detection()
        except Exception as e:
            logger.error(f"[YABAI] ❌ Unexpected error querying spaces: {e}")
            return await self._fallback_space_detection()

    async def get_windows(self, force_refresh: bool = False) -> List[WindowInfo]:
        """
        Get all windows across all spaces

        Args:
            force_refresh: Bypass cache and query directly

        Returns:
            List of WindowInfo objects
        """
        # Check cache first
        if (self.enable_cache and not force_refresh and
            self._cache and not self._cache.is_expired(self.cache_ttl)):
            self._cache_hits += 1
            logger.debug("[YABAI] 📦 Cache hit for windows query")
            return self._cache.windows

        # If cache miss, get_spaces will refresh both
        await self.get_spaces(force_refresh=force_refresh)

        return self._cache.windows if self._cache else []

    async def _query_windows_internal(self) -> List[WindowInfo]:
        """Internal method to query windows"""
        try:
            process = await asyncio.create_subprocess_exec(
                self._yabai_path, '-m', 'query', '--windows',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.query_timeout
            )

            if process.returncode != 0:
                logger.error(f"[YABAI] Windows query failed")
                return []

            # Parse JSON response
            windows_data = json.loads(stdout.decode('utf-8'))
            windows = [
                WindowInfo(
                    window_id=w['id'],
                    app_name=w.get('app', 'Unknown'),
                    title=w.get('title', ''),
                    space_index=w.get('space', 0),
                    display=w.get('display', 1),
                    frame=w.get('frame', {}),
                    is_minimized=bool(w.get('minimized', 0)),
                    is_visible=bool(w.get('visible', 1)),
                    is_focused=bool(w.get('focused', 0))
                )
                for w in windows_data
                if w.get('app', 'Unknown') not in ['Window Server', 'Dock', 'SystemUIServer']
            ]

            logger.info(f"[YABAI] ✅ Retrieved {len(windows)} windows")
            return windows

        except Exception as e:
            logger.error(f"[YABAI] ❌ Error querying windows: {e}")
            return []

    async def get_workspace_data(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Get complete workspace data (spaces + windows)

        Args:
            force_refresh: Bypass cache and query directly

        Returns:
            Dict with 'spaces' and 'windows' keys
        """
        spaces = await self.get_spaces(force_refresh=force_refresh)
        windows = await self.get_windows(force_refresh=force_refresh)

        return {
            'spaces': spaces,
            'windows': windows,
            'timestamp': datetime.now().isoformat(),
            'yabai_status': self._yabai_status.value if self._yabai_status else 'unknown'
        }

    async def _fallback_space_detection(self) -> List[SpaceInfo]:
        """
        Fallback space detection when Yabai unavailable
        Uses basic Core Graphics window detection
        """
        logger.info("[YABAI] 📦 Using fallback space detection")

        # Import the old space detection bridge if available
        try:
            # Try to use the Obj-C bridge we created earlier
            import ctypes
            bridge_path = Path(__file__).parent / 'space_detection_bridge.dylib'

            if bridge_path.exists():
                lib = ctypes.CDLL(str(bridge_path))

                # Get space count
                lib.get_space_count.restype = ctypes.c_int
                space_count = lib.get_space_count()

                # Create basic space info
                spaces = [
                    SpaceInfo(
                        index=i,
                        label=f"Desktop {i}",
                        display=1,
                        window_ids=[],
                        is_visible=(i == 1),
                        is_focused=(i == 1)
                    )
                    for i in range(1, space_count + 1)
                ]

                logger.info(f"[YABAI] 📦 Fallback detected {len(spaces)} spaces")
                return spaces
        except Exception as e:
            logger.debug(f"[YABAI] Could not use Obj-C bridge fallback: {e}")

        # Ultra-basic fallback - just report current space
        return [
            SpaceInfo(
                index=1,
                label="Current Desktop",
                display=1,
                window_ids=[],
                is_visible=True,
                is_focused=True
            )
        ]

    def get_installation_instructions(self) -> str:
        """Get Yabai installation instructions"""
        return """
Sir, for enhanced multi-space intelligence, I recommend installing Yabai.

**Installation via Homebrew:**
```
brew install koekeishiya/formulae/yabai
```

**Grant Accessibility Permissions:**
1. Open System Preferences → Security & Privacy → Privacy
2. Click "Accessibility" in the left sidebar
3. Click the lock icon and authenticate
4. Add Yabai to the list and check the box

After installation, I'll be able to provide detailed workspace analysis.
""".strip()

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance metrics"""
        avg_query_time = (
            self._total_query_time / self._query_count
            if self._query_count > 0 else 0
        )

        cache_hit_rate = (
            self._cache_hits / (self._query_count + self._cache_hits)
            if (self._query_count + self._cache_hits) > 0 else 0
        )

        return {
            'total_queries': self._query_count,
            'cache_hits': self._cache_hits,
            'cache_hit_rate': f"{cache_hit_rate:.1%}",
            'avg_query_time_ms': f"{avg_query_time * 1000:.1f}",
            'yabai_status': self._yabai_status.value if self._yabai_status else 'unknown',
            'cache_enabled': self.enable_cache
        }
