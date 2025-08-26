#!/usr/bin/env python3
"""
Multi-Window Capture System for JARVIS
Captures multiple windows efficiently for workspace analysis
"""

import os
import cv2
import numpy as np
import Quartz
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from .window_detector import WindowDetector, WindowInfo
from .simple_capture import capture_screen_simple, capture_window_area

logger = logging.getLogger(__name__)


@dataclass
class WindowCapture:
    """Captured window data"""
    window_info: WindowInfo
    image: np.ndarray
    resolution_scale: float
    capture_time: float


class MultiWindowCapture:
    """Captures multiple windows with intelligent optimization"""
    
    def __init__(self):
        self.window_detector = WindowDetector()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Capture settings
        self.max_windows = 5
        self.focused_resolution = 1.0  # Full resolution
        self.context_resolution = 0.5  # Half resolution for background windows
        self.thumbnail_resolution = 0.25  # Quarter resolution for thumbnails
        
    def capture_single_window(self, window_info: WindowInfo, 
                            resolution_scale: float = 1.0) -> Optional[np.ndarray]:
        """Capture a single window"""
        try:
            # Use different approach - capture entire screen then crop
            # This is more reliable on modern macOS
            bounds = window_info.bounds
            
            # Get main display
            main_display = Quartz.CGMainDisplayID()
            
            # Capture entire screen
            image_ref = Quartz.CGDisplayCreateImage(main_display)
            
            if not image_ref:
                logger.warning(f"Failed to capture window: {window_info.app_name}")
                return None
            
            # Convert to numpy array
            width = Quartz.CGImageGetWidth(image_ref)
            height = Quartz.CGImageGetHeight(image_ref)
            
            # Create bitmap context
            colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
            bitmapInfo = Quartz.kCGImageAlphaPremultipliedLast
            
            # Allocate memory for image data
            bytes_per_row = 4 * width
            data = np.zeros((height, width, 4), dtype=np.uint8)
            
            # Create context and draw image
            # Use None for data pointer to let Core Graphics manage memory
            context = Quartz.CGBitmapContextCreate(
                None,  # Let CG manage memory
                width,
                height,
                8,
                bytes_per_row,
                colorspace,
                bitmapInfo
            )
            
            if context:
                Quartz.CGContextDrawImage(
                    context,
                    Quartz.CGRectMake(0, 0, width, height),
                    image_ref
                )
                
                # Fallback: try direct data access
                data_provider = Quartz.CGImageGetDataProvider(image_ref)
                raw_data = Quartz.CGDataProviderCopyData(data_provider)
                
                if raw_data:
                    # Calculate expected size
                    expected_size = height * width * 4
                    actual_size = len(raw_data)
                    
                    if actual_size >= expected_size:
                        image_data = np.frombuffer(
                            raw_data,
                            dtype=np.uint8
                        )[:expected_size].reshape((height, width, 4))
                        
                        image = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)
                    else:
                        logger.error(f"Data size mismatch: expected {expected_size}, got {actual_size}")
                        return None
                else:
                    logger.error("Could not get image data")
                    return None
                
                # Crop to window bounds if we captured full screen
                if bounds['x'] > 0 or bounds['y'] > 0:
                    x, y = int(bounds['x']), int(bounds['y'])
                    w, h = int(bounds['width']), int(bounds['height'])
                    
                    # Ensure crop bounds are within image
                    x = max(0, min(x, image.shape[1] - 1))
                    y = max(0, min(y, image.shape[0] - 1))
                    w = min(w, image.shape[1] - x)
                    h = min(h, image.shape[0] - y)
                    
                    image = image[y:y+h, x:x+w]
                
                # Apply resolution scaling if needed
                if resolution_scale < 1.0:
                    new_width = int(width * resolution_scale)
                    new_height = int(height * resolution_scale)
                    image = cv2.resize(image, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
                
                return image
            
        except Exception as e:
            logger.error(f"Error with Quartz capture for {window_info.app_name}: {e}")
            # Fallback to simple capture
            logger.info("Trying simple capture method...")
            return self._simple_capture_fallback(window_info, resolution_scale)
    
    def capture_workspace_screenshot(self) -> Optional[np.ndarray]:
        """Capture entire workspace (all displays)"""
        try:
            # Get main display ID
            main_display = Quartz.CGMainDisplayID()
            
            # Capture entire screen
            image_ref = Quartz.CGDisplayCreateImage(main_display)
            
            if not image_ref:
                return None
            
            # Convert to numpy array (same process as single window)
            width = Quartz.CGImageGetWidth(image_ref)
            height = Quartz.CGImageGetHeight(image_ref)
            
            colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
            bitmapInfo = Quartz.kCGImageAlphaPremultipliedLast
            
            bytes_per_row = 4 * width
            data = np.zeros((height, width, 4), dtype=np.uint8)
            
            context = Quartz.CGBitmapContextCreate(
                data.ctypes.data,
                width, height, 8, bytes_per_row,
                colorspace, bitmapInfo
            )
            
            if context:
                Quartz.CGContextDrawImage(
                    context,
                    Quartz.CGRectMake(0, 0, width, height),
                    image_ref
                )
                
                return cv2.cvtColor(data, cv2.COLOR_RGBA2RGB)
                
        except Exception as e:
            logger.error(f"Error capturing workspace: {e}")
            return None
    
    async def capture_multiple_windows(self, 
                                     query_type: Optional[str] = None) -> List[WindowCapture]:
        """Capture multiple windows based on query context"""
        
        # Get all windows
        windows = self.window_detector.get_all_windows()
        
        if not windows:
            logger.warning("No windows found to capture")
            return []
        
        # Determine which windows to capture based on query
        windows_to_capture = self._select_windows_for_query(windows, query_type)
        
        # Capture windows asynchronously
        captures = []
        capture_tasks = []
        
        for window in windows_to_capture[:self.max_windows]:
            # Determine resolution based on focus
            if window.is_focused:
                resolution = self.focused_resolution
            elif len(captures) < 2:  # First two background windows get medium res
                resolution = self.context_resolution
            else:
                resolution = self.thumbnail_resolution
            
            # Create async task for capture
            task = asyncio.create_task(
                self._async_capture_window(window, resolution)
            )
            capture_tasks.append(task)
        
        # Wait for all captures to complete
        capture_results = await asyncio.gather(*capture_tasks, return_exceptions=True)
        
        # Filter successful captures
        import time
        for i, result in enumerate(capture_results):
            if isinstance(result, WindowCapture):
                captures.append(result)
            else:
                logger.warning(f"Failed to capture window: {result}")
        
        return captures
    
    async def _async_capture_window(self, window: WindowInfo, 
                                  resolution: float) -> WindowCapture:
        """Async wrapper for window capture"""
        loop = asyncio.get_event_loop()
        
        # Run capture in thread pool
        image = await loop.run_in_executor(
            self.executor,
            self.capture_single_window,
            window,
            resolution
        )
        
        if image is None:
            raise Exception(f"Failed to capture {window.app_name}")
        
        import time
        return WindowCapture(
            window_info=window,
            image=image,
            resolution_scale=resolution,
            capture_time=time.time()
        )
    
    def _select_windows_for_query(self, windows: List[WindowInfo], 
                                query_type: Optional[str]) -> List[WindowInfo]:
        """Select relevant windows using dynamic ML-based analysis"""
        
        # Use dynamic multi-window engine for intelligent selection
        try:
            from .dynamic_multi_window_engine import get_dynamic_multi_window_engine
            
            # Create a query based on the query type
            if query_type:
                query = f"Show me windows related to {query_type}"
            else:
                query = "Show me all relevant windows"
            
            # Get dynamic analysis
            engine = get_dynamic_multi_window_engine()
            analysis = engine.analyze_windows_for_query(query, windows)
            
            # Combine primary and context windows
            selected_windows = analysis.primary_windows + analysis.context_windows
            
            # If no windows selected by ML, fall back to basic selection
            if not selected_windows:
                # Always include focused window first
                focused = [w for w in windows if w.is_focused]
                others = [w for w in windows if not w.is_focused]
                
                # Sort others by size (larger windows likely more important)
                others.sort(key=lambda w: w.bounds['width'] * w.bounds['height'], 
                           reverse=True)
                
                selected_windows = focused + others
            
            return selected_windows
            
        except Exception as e:
            logger.warning(f"Dynamic window selection failed: {e}")
            # Fallback: focused window + largest windows
            focused = [w for w in windows if w.is_focused]
            others = [w for w in windows if not w.is_focused]
            others.sort(key=lambda w: w.bounds['width'] * w.bounds['height'], 
                       reverse=True)
            return focused + others
    
    def create_workspace_composite(self, captures: List[WindowCapture], 
                                 max_width: int = 2000) -> np.ndarray:
        """Create a composite image of all captured windows"""
        if not captures:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Calculate layout
        focused_capture = next((c for c in captures if c.window_info.is_focused), None)
        
        if focused_capture:
            # Focused window takes up 60% of width
            main_width = int(max_width * 0.6)
            main_image = focused_capture.image
            
            # Resize if needed
            if main_image.shape[1] > main_width:
                scale = main_width / main_image.shape[1]
                new_height = int(main_image.shape[0] * scale)
                main_image = cv2.resize(main_image, (main_width, new_height))
            
            # Create sidebar for other windows
            sidebar_width = max_width - main_width - 20  # 20px gap
            sidebar_images = []
            
            for capture in captures:
                if capture != focused_capture:
                    img = capture.image
                    # Resize to fit sidebar
                    if img.shape[1] > sidebar_width:
                        scale = sidebar_width / img.shape[1]
                        new_height = int(img.shape[0] * scale)
                        img = cv2.resize(img, (sidebar_width, new_height))
                    
                    # Add window label
                    label = capture.window_info.app_name
                    cv2.putText(img, label, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    sidebar_images.append(img)
            
            # Stack sidebar images
            if sidebar_images:
                # Ensure all sidebar images have same width
                max_sidebar_width = max(img.shape[1] for img in sidebar_images)
                padded_sidebar_images = []
                
                for img in sidebar_images[:4]:  # Max 4
                    if img.shape[1] < max_sidebar_width:
                        pad_width = max_sidebar_width - img.shape[1]
                        img = np.pad(img, ((0, 0), (0, pad_width), (0, 0)), 
                                   mode='constant', constant_values=128)
                    padded_sidebar_images.append(img)
                
                sidebar = np.vstack(padded_sidebar_images)
                
                # Combine main and sidebar
                max_height = max(main_image.shape[0], sidebar.shape[0])
                
                # Pad images to same height
                if main_image.shape[0] < max_height:
                    pad = max_height - main_image.shape[0]
                    main_image = np.pad(main_image, ((0, pad), (0, 0), (0, 0)))
                
                if sidebar.shape[0] < max_height:
                    pad = max_height - sidebar.shape[0]
                    sidebar = np.pad(sidebar, ((0, pad), (0, 0), (0, 0)))
                
                # Add gap
                gap = np.ones((max_height, 20, 3), dtype=np.uint8) * 128
                
                composite = np.hstack([main_image, gap, sidebar])
            else:
                composite = main_image
        else:
            # No focused window, create grid
            images = [c.image for c in captures[:4]]
            
            # Resize all to same width
            target_width = max_width // 2
            resized = []
            
            for img in images:
                if img.shape[1] != target_width:
                    scale = target_width / img.shape[1]
                    new_height = int(img.shape[0] * scale)
                    img = cv2.resize(img, (target_width, new_height))
                resized.append(img)
            
            # Create 2x2 grid
            if len(resized) >= 2:
                row1 = np.hstack(resized[:2])
                if len(resized) >= 4:
                    row2 = np.hstack(resized[2:4])
                    composite = np.vstack([row1, row2])
                else:
                    composite = row1
            else:
                composite = resized[0] if resized else np.zeros((100, 100, 3))
        
        return composite.astype(np.uint8)
    
    def _simple_capture_fallback(self, window_info: WindowInfo, 
                               resolution_scale: float = 1.0) -> Optional[np.ndarray]:
        """Fallback capture method using screencapture command"""
        try:
            # Use simple capture for window area
            bounds = window_info.bounds
            image = capture_window_area(
                bounds['x'], 
                bounds['y'], 
                bounds['width'], 
                bounds['height']
            )
            
            if image is not None and resolution_scale < 1.0:
                # Apply resolution scaling
                new_width = int(image.shape[1] * resolution_scale)
                new_height = int(image.shape[0] * resolution_scale)
                image = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception as e:
            logger.error(f"Simple capture also failed: {e}")
            return None


async def test_multi_window_capture():
    """Test multi-window capture system"""
    print("📸 Testing Multi-Window Capture System")
    print("=" * 50)
    
    capture_system = MultiWindowCapture()
    
    # Test 1: Capture current workspace
    print("\n1️⃣ Capturing current workspace...")
    captures = await capture_system.capture_multiple_windows()
    
    print(f"\n✅ Captured {len(captures)} windows:")
    for i, capture in enumerate(captures):
        window = capture.window_info
        img_shape = capture.image.shape
        print(f"   {i+1}. {window.app_name}: {img_shape[1]}x{img_shape[0]} "
              f"@ {capture.resolution_scale:.0%} scale")
    
    # Test 2: Create composite image
    print("\n2️⃣ Creating workspace composite...")
    composite = capture_system.create_workspace_composite(captures)
    print(f"   Composite size: {composite.shape[1]}x{composite.shape[0]}")
    
    # Save composite for inspection
    cv2.imwrite("workspace_composite.png", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
    print("   Saved to: workspace_composite.png")
    
    # Test 3: Query-specific capture
    print("\n3️⃣ Testing query-specific captures...")
    
    for query_type in ["messages", "errors", "documentation"]:
        print(f"\n   Query: '{query_type}'")
        captures = await capture_system.capture_multiple_windows(query_type)
        print(f"   Captured: {[c.window_info.app_name for c in captures[:3]]}")
    
    print("\n✅ Multi-window capture test complete!")


if __name__ == "__main__":
    asyncio.run(test_multi_window_capture())