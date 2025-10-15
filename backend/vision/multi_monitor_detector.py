"""
Multi-Monitor Support for JARVIS Vision Intelligence

This module provides comprehensive multi-monitor detection and management
capabilities for macOS systems, enabling JARVIS to understand and analyze
content across multiple displays.

Key Features:
- Real-time display detection using Core Graphics APIs
- Space-to-display mapping via Yabai integration
- Per-monitor screenshot capture for vision analysis
- Display-aware context understanding and query routing

Author: Derek Russell
Date: 2025-01-14
Branch: multi-monitor-support
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path

try:
    import Quartz
    import AppKit
    from Foundation import NSRect, NSPoint, NSSize
    MACOS_AVAILABLE = True
except ImportError:
    MACOS_AVAILABLE = False
    logging.warning("macOS frameworks not available - multi-monitor support disabled")

logger = logging.getLogger(__name__)


@dataclass
class DisplayInfo:
    """Information about a connected display"""
    display_id: int
    resolution: Tuple[int, int]
    position: Tuple[int, int]
    is_primary: bool
    refresh_rate: float = 60.0
    color_depth: int = 32
    name: str = ""
    spaces: List[int] = field(default_factory=list)
    active_space: int = 1
    last_updated: float = field(default_factory=time.time)


@dataclass
class SpaceDisplayMapping:
    """Mapping between Yabai spaces and displays"""
    space_id: int
    display_id: int
    space_name: str = ""
    is_active: bool = False
    last_seen: float = field(default_factory=time.time)


@dataclass
class MonitorCaptureResult:
    """Result of multi-monitor capture operation"""
    success: bool
    displays_captured: Dict[int, np.ndarray]
    failed_displays: List[int]
    capture_time: float
    total_displays: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultiMonitorDetector:
    """
    Core detector for multi-monitor support on macOS
    
    Provides comprehensive display detection, space mapping, and screenshot
    capture capabilities across multiple monitors.
    """
    
    def __init__(self, yabai_path: str = "yabai"):
        """
        Initialize the multi-monitor detector
        
        Args:
            yabai_path: Path to yabai executable
        """
        self.yabai_path = yabai_path
        self.displays: Dict[int, DisplayInfo] = {}
        self.space_mappings: Dict[int, SpaceDisplayMapping] = {}
        self.last_detection_time = 0.0
        self.detection_cache_duration = 5.0  # Cache for 5 seconds
        
        # Performance tracking
        self.capture_stats = {
            "total_captures": 0,
            "successful_captures": 0,
            "failed_captures": 0,
            "average_capture_time": 0.0
        }
        
        logger.info("MultiMonitorDetector initialized")
    
    async def detect_displays(self, force_refresh: bool = False) -> List[DisplayInfo]:
        """
        Detect and return all connected displays
        
        Args:
            force_refresh: Force refresh even if cache is valid
            
        Returns:
            List of DisplayInfo objects for all connected displays
        """
        current_time = time.time()
        
        # Use cache if recent and not forcing refresh
        if (not force_refresh and 
            current_time - self.last_detection_time < self.detection_cache_duration and
            self.displays):
            logger.debug("Using cached display information")
            return list(self.displays.values())
        
        if not MACOS_AVAILABLE:
            logger.error("macOS frameworks not available")
            return []
        
        try:
            displays = []
            
            # Get display list from Core Graphics
            display_count = Quartz.CGGetActiveDisplayList(0, None, None)
            if display_count == 0:
                logger.warning("No displays detected")
                return []
            
            # Allocate array for display IDs
            display_ids = (Quartz.CGDirectDisplayID * display_count)()
            actual_count = Quartz.CGGetActiveDisplayList(
                display_count, display_ids, None
            )
            
            logger.info(f"Detected {actual_count} displays")
            
            for i in range(actual_count):
                display_id = display_ids[i]
                
                # Get display bounds
                bounds = Quartz.CGDisplayBounds(display_id)
                width = int(bounds.size.width)
                height = int(bounds.size.height)
                x = int(bounds.origin.x)
                y = int(bounds.origin.y)
                
                # Check if primary display
                is_primary = Quartz.CGDisplayIsMain(display_id)
                
                # Get display name (if available)
                display_name = f"Display {display_id}"
                try:
                    # Try to get display name from system
                    display_name = self._get_display_name(display_id)
                except Exception as e:
                    logger.debug(f"Could not get display name for {display_id}: {e}")
                
                display_info = DisplayInfo(
                    display_id=display_id,
                    resolution=(width, height),
                    position=(x, y),
                    is_primary=is_primary,
                    name=display_name
                )
                
                displays.append(display_info)
                self.displays[display_id] = display_info
                
                logger.info(f"Display {display_id}: {width}x{height} at ({x}, {y}) - {'Primary' if is_primary else 'Secondary'}")
            
            self.last_detection_time = current_time
            return displays
            
        except Exception as e:
            logger.error(f"Error detecting displays: {e}")
            return []
    
    def _get_display_name(self, display_id: int) -> str:
        """Get human-readable display name"""
        try:
            # Try to get display name from Core Graphics
            # This is a simplified approach - could be enhanced
            if display_id == Quartz.CGMainDisplayID():
                return "Primary Display"
            else:
                return f"Display {display_id}"
        except Exception:
            return f"Display {display_id}"
    
    async def get_space_display_mapping(self, force_refresh: bool = False) -> Dict[int, int]:
        """
        Map each Yabai space to its corresponding display
        
        Args:
            force_refresh: Force refresh of mappings
            
        Returns:
            Dictionary mapping space_id -> display_id
        """
        try:
            # First ensure we have current display info
            await self.detect_displays(force_refresh)
            
            # Get space information from Yabai
            space_mappings = await self._get_yabai_space_mappings()
            
            # Update our internal mappings
            self.space_mappings = space_mappings
            
            # Return simple space_id -> display_id mapping
            result = {}
            for space_id, mapping in space_mappings.items():
                result[space_id] = mapping.display_id
            
            logger.info(f"Mapped {len(result)} spaces to displays")
            return result
            
        except Exception as e:
            logger.error(f"Error getting space-display mapping: {e}")
            return {}
    
    async def _get_yabai_space_mappings(self) -> Dict[int, SpaceDisplayMapping]:
        """Get space mappings from Yabai CLI"""
        try:
            # Query Yabai for spaces
            result = await asyncio.create_subprocess_exec(
                self.yabai_path, "-m", "query", "--spaces",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.error(f"Yabai spaces query failed: {stderr.decode()}")
                return {}
            
            # Parse Yabai output (simplified - would need proper JSON parsing)
            spaces_data = stdout.decode().strip()
            logger.debug(f"Yabai spaces data: {spaces_data}")
            
            # For now, create basic mappings
            # In a full implementation, this would parse the actual Yabai JSON output
            mappings = {}
            for display_id in self.displays.keys():
                # Create a basic mapping (this is simplified)
                space_id = len(mappings) + 1
                mapping = SpaceDisplayMapping(
                    space_id=space_id,
                    display_id=display_id,
                    space_name=f"Space {space_id}",
                    is_active=True
                )
                mappings[space_id] = mapping
            
            return mappings
            
        except Exception as e:
            logger.error(f"Error querying Yabai for space mappings: {e}")
            return {}
    
    async def capture_all_displays(self, include_inactive: bool = False) -> MonitorCaptureResult:
        """
        Capture screenshots for all displays and their active spaces
        
        Args:
            include_inactive: Whether to capture inactive spaces
            
        Returns:
            MonitorCaptureResult with screenshots and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure we have current display information
            displays = await self.detect_displays()
            if not displays:
                return MonitorCaptureResult(
                    success=False,
                    displays_captured={},
                    failed_displays=[],
                    capture_time=time.time() - start_time,
                    total_displays=0,
                    error="No displays detected"
                )
            
            displays_captured = {}
            failed_displays = []
            
            # Capture each display
            for display_info in displays:
                try:
                    screenshot = await self._capture_display(display_info)
                    if screenshot is not None:
                        displays_captured[display_info.display_id] = screenshot
                        logger.debug(f"Captured display {display_info.display_id}: {screenshot.shape}")
                    else:
                        failed_displays.append(display_info.display_id)
                        logger.warning(f"Failed to capture display {display_info.display_id}")
                        
                except Exception as e:
                    failed_displays.append(display_info.display_id)
                    logger.error(f"Error capturing display {display_info.display_id}: {e}")
            
            capture_time = time.time() - start_time
            
            # Update statistics
            self.capture_stats["total_captures"] += 1
            if displays_captured:
                self.capture_stats["successful_captures"] += 1
            else:
                self.capture_stats["failed_captures"] += 1
            
            # Update average capture time
            total_captures = self.capture_stats["total_captures"]
            current_avg = self.capture_stats["average_capture_time"]
            self.capture_stats["average_capture_time"] = (
                (current_avg * (total_captures - 1) + capture_time) / total_captures
            )
            
            success = len(displays_captured) > 0
            
            result = MonitorCaptureResult(
                success=success,
                displays_captured=displays_captured,
                failed_displays=failed_displays,
                capture_time=capture_time,
                total_displays=len(displays),
                metadata={
                    "capture_method": "core_graphics",
                    "displays_info": {d.display_id: {
                        "resolution": d.resolution,
                        "position": d.position,
                        "is_primary": d.is_primary
                    } for d in displays}
                }
            )
            
            logger.info(f"Multi-monitor capture completed: {len(displays_captured)}/{len(displays)} displays captured in {capture_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error in multi-monitor capture: {e}")
            return MonitorCaptureResult(
                success=False,
                displays_captured={},
                failed_displays=list(self.displays.keys()),
                capture_time=time.time() - start_time,
                total_displays=len(self.displays),
                error=str(e)
            )
    
    async def _capture_display(self, display_info: DisplayInfo) -> Optional[np.ndarray]:
        """
        Capture screenshot of a specific display
        
        Args:
            display_info: Display information
            
        Returns:
            Screenshot as numpy array, or None if failed
        """
        if not MACOS_AVAILABLE:
            return None
        
        try:
            # Create display bounds
            bounds = Quartz.CGRect(
                Quartz.CGPoint(display_info.position[0], display_info.position[1]),
                Quartz.CGSize(display_info.resolution[0], display_info.resolution[1])
            )
            
            # Capture the display
            image_ref = Quartz.CGWindowListCreateImage(
                bounds,
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID,
                Quartz.kCGWindowImageDefault
            )
            
            if image_ref is None:
                logger.warning(f"Failed to create image for display {display_info.display_id}")
                return None
            
            # Convert to numpy array
            width = Quartz.CGImageGetWidth(image_ref)
            height = Quartz.CGImageGetHeight(image_ref)
            
            # Create data provider
            data_provider = Quartz.CGImageGetDataProvider(image_ref)
            data = Quartz.CGDataProviderCopyData(data_provider)
            
            # Convert to numpy array
            # Note: This is a simplified conversion - proper implementation would handle
            # different pixel formats and color spaces
            image_data = np.frombuffer(data, dtype=np.uint8)
            
            # Reshape to image dimensions (assuming RGBA format)
            if len(image_data) == width * height * 4:
                image_array = image_data.reshape((height, width, 4))
                # Convert RGBA to RGB (remove alpha channel)
                image_array = image_array[:, :, :3]
            else:
                logger.warning(f"Unexpected image data size for display {display_info.display_id}")
                return None
            
            logger.debug(f"Captured display {display_info.display_id}: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error capturing display {display_info.display_id}: {e}")
            return None
    
    async def get_display_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all displays and their current state
        
        Returns:
            Dictionary with display summary information
        """
        try:
            displays = await self.detect_displays()
            space_mappings = await self.get_space_display_mapping()
            
            summary = {
                "total_displays": len(displays),
                "displays": [],
                "space_mappings": space_mappings,
                "detection_time": self.last_detection_time,
                "capture_stats": self.capture_stats.copy()
            }
            
            for display_info in displays:
                display_summary = {
                    "id": display_info.display_id,
                    "name": display_info.name,
                    "resolution": display_info.resolution,
                    "position": display_info.position,
                    "is_primary": display_info.is_primary,
                    "spaces": [space_id for space_id, mapping in space_mappings.items() 
                              if mapping.display_id == display_info.display_id]
                }
                summary["displays"].append(display_summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting display summary: {e}")
            return {"error": str(e)}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the detector"""
        return {
            "capture_stats": self.capture_stats.copy(),
            "displays_cached": len(self.displays),
            "space_mappings_cached": len(self.space_mappings),
            "last_detection_time": self.last_detection_time,
            "cache_age": time.time() - self.last_detection_time
        }


# Convenience functions for easy integration
async def detect_all_monitors() -> List[DisplayInfo]:
    """Convenience function to detect all monitors"""
    detector = MultiMonitorDetector()
    return await detector.detect_displays()


async def capture_multi_monitor_screenshots() -> MonitorCaptureResult:
    """Convenience function to capture all monitor screenshots"""
    detector = MultiMonitorDetector()
    return await detector.capture_all_displays()


async def get_monitor_summary() -> Dict[str, Any]:
    """Convenience function to get monitor summary"""
    detector = MultiMonitorDetector()
    return await detector.get_display_summary()


if __name__ == "__main__":
    # Test the multi-monitor detector
    async def test_detector():
        detector = MultiMonitorDetector()
        
        print("🔍 Detecting displays...")
        displays = await detector.detect_displays()
        print(f"Found {len(displays)} displays:")
        
        for display in displays:
            print(f"  - {display.name}: {display.resolution[0]}x{display.resolution[1]} at ({display.position[0]}, {display.position[1]}) {'[Primary]' if display.is_primary else ''}")
        
        print("\n🗺️ Getting space mappings...")
        mappings = await detector.get_space_display_mapping()
        print(f"Space mappings: {mappings}")
        
        print("\n📸 Capturing screenshots...")
        result = await detector.capture_all_displays()
        print(f"Capture result: {result.success}, {len(result.displays_captured)} displays captured")
        
        print("\n📊 Performance stats:")
        stats = detector.get_performance_stats()
        print(f"  - Total captures: {stats['capture_stats']['total_captures']}")
        print(f"  - Average capture time: {stats['capture_stats']['average_capture_time']:.2f}s")
    
    # Run the test
    asyncio.run(test_detector())
