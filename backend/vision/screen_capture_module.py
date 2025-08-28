#!/usr/bin/env python3
"""
Screen Capture Module for JARVIS Vision System
Continuously captures and analyzes desktop screenshots
"""

import asyncio
import base64
import io
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# Platform-specific imports
import sys
if sys.platform == 'darwin':
    # macOS implementation
    import Quartz
    import Quartz.CoreGraphics as CG
    from AppKit import NSScreen
else:
    # Fallback for other platforms
    try:
        import pyautogui
    except ImportError:
        pyautogui = None

logger = logging.getLogger(__name__)

@dataclass
class ScreenCapture:
    """Container for screen capture data"""
    timestamp: datetime
    image: Image.Image
    region: Optional[Tuple[int, int, int, int]] = None  # x, y, width, height
    screen_index: int = 0
    
    @property
    def base64_image(self) -> str:
        """Get base64 encoded image"""
        buffer = io.BytesIO()
        self.image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    def get_compressed_image(self, quality: int = 85) -> Image.Image:
        """Get compressed version of the image"""
        buffer = io.BytesIO()
        self.image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class ScreenCaptureModule:
    """Handles continuous screen capture for JARVIS vision"""
    
    def __init__(self, capture_interval: float = 2.0):
        self.capture_interval = capture_interval
        self.is_capturing = False
        self.capture_task = None
        self.last_capture = None
        self.capture_count = 0
        self.capture_callbacks = []
        
        # Performance settings
        self.compression_quality = 85
        self.max_resolution = (1920, 1080)  # Downscale if larger
        self.differential_mode = True  # Only send changes
        self.last_hash = None
        
        # Screen information
        self.screens = self._get_screen_info()
        
    def _get_screen_info(self) -> List[Dict[str, Any]]:
        """Get information about available screens"""
        screens = []
        
        if sys.platform == 'darwin':
            for i, screen in enumerate(NSScreen.screens()):
                frame = screen.frame()
                screens.append({
                    'index': i,
                    'x': int(frame.origin.x),
                    'y': int(frame.origin.y),
                    'width': int(frame.size.width),
                    'height': int(frame.size.height),
                    'is_main': i == 0
                })
        else:
            # Fallback implementation
            try:
                import tkinter as tk
                root = tk.Tk()
                screens.append({
                    'index': 0,
                    'x': 0,
                    'y': 0,
                    'width': root.winfo_screenwidth(),
                    'height': root.winfo_screenheight(),
                    'is_main': True
                })
                root.destroy()
            except:
                # Default to common resolution
                screens.append({
                    'index': 0,
                    'x': 0,
                    'y': 0,
                    'width': 1920,
                    'height': 1080,
                    'is_main': True
                })
                
        return screens
    
    def capture_screen(self, screen_index: int = 0, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[ScreenCapture]:
        """Capture a screenshot of the specified screen or region"""
        try:
            if sys.platform == 'darwin':
                return self._capture_screen_macos(screen_index, region)
            elif pyautogui:
                return self._capture_screen_pyautogui(region)
            else:
                logger.error("No screen capture method available")
                return None
                
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def _capture_screen_macos(self, screen_index: int = 0, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[ScreenCapture]:
        """Capture screen on macOS using Quartz"""
        try:
            # Create screen capture
            if region:
                x, y, width, height = region
                rect = CG.CGRectMake(x, y, width, height)
                image_ref = CG.CGWindowListCreateImage(
                    rect,
                    CG.kCGWindowListOptionOnScreenOnly,
                    CG.kCGNullWindowID,
                    CG.kCGWindowImageDefault
                )
            else:
                # Capture entire screen
                screens = self.screens
                if screen_index >= len(screens):
                    screen_index = 0
                    
                screen = screens[screen_index]
                rect = CG.CGRectMake(
                    screen['x'], screen['y'],
                    screen['width'], screen['height']
                )
                image_ref = CG.CGWindowListCreateImage(
                    rect,
                    CG.kCGWindowListOptionOnScreenOnly,
                    CG.kCGNullWindowID,
                    CG.kCGWindowImageDefault
                )
            
            # Convert to PIL Image
            width = CG.CGImageGetWidth(image_ref)
            height = CG.CGImageGetHeight(image_ref)
            bytes_per_row = CG.CGImageGetBytesPerRow(image_ref)
            pixel_data = CG.CGDataProviderCopyData(CG.CGImageGetDataProvider(image_ref))
            
            image = Image.frombuffer(
                "RGBA", (width, height), pixel_data, "raw", "BGRA", bytes_per_row, 1
            ).convert('RGB')
            
            # Optimize image size if needed
            image = self._optimize_image(image)
            
            return ScreenCapture(
                timestamp=datetime.now(),
                image=image,
                region=region,
                screen_index=screen_index
            )
            
        except Exception as e:
            logger.error(f"macOS screen capture error: {e}")
            return None
    
    def _capture_screen_pyautogui(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[ScreenCapture]:
        """Capture screen using pyautogui (cross-platform fallback)"""
        try:
            if region:
                screenshot = pyautogui.screenshot(region=region)
            else:
                screenshot = pyautogui.screenshot()
                
            # Convert to RGB if needed
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')
                
            # Optimize image
            screenshot = self._optimize_image(screenshot)
            
            return ScreenCapture(
                timestamp=datetime.now(),
                image=screenshot,
                region=region,
                screen_index=0
            )
            
        except Exception as e:
            logger.error(f"PyAutoGUI screen capture error: {e}")
            return None
    
    def _optimize_image(self, image: Image.Image) -> Image.Image:
        """Optimize image for processing and transmission"""
        # Downscale if too large
        if image.width > self.max_resolution[0] or image.height > self.max_resolution[1]:
            image.thumbnail(self.max_resolution, Image.Resampling.LANCZOS)
            
        return image
    
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Apply sharpening
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        
        # Increase size for better OCR
        width, height = sharpened.size
        if width < 1000:
            scale = 1000 / width
            new_size = (int(width * scale), int(height * scale))
            sharpened = sharpened.resize(new_size, Image.Resampling.LANCZOS)
            
        return sharpened
    
    def get_image_hash(self, image: Image.Image) -> str:
        """Get perceptual hash of image for change detection"""
        # Resize to 8x8
        small = image.resize((8, 8), Image.Resampling.LANCZOS).convert('L')
        
        # Get average pixel value
        pixels = np.array(small)
        avg = pixels.mean()
        
        # Create hash
        diff = pixels > avg
        return ''.join(['1' if d else '0' for d in diff.flatten()])
    
    def has_significant_change(self, image: Image.Image, threshold: float = 0.1) -> bool:
        """Check if image has significant changes from last capture"""
        if not self.differential_mode or not self.last_hash:
            return True
            
        current_hash = self.get_image_hash(image)
        
        # Calculate Hamming distance
        distance = sum(c1 != c2 for c1, c2 in zip(self.last_hash, current_hash))
        change_ratio = distance / len(current_hash)
        
        return change_ratio > threshold
    
    async def start_continuous_capture(self):
        """Start continuous screen capture loop"""
        if self.is_capturing:
            return
            
        self.is_capturing = True
        self.capture_task = asyncio.create_task(self._capture_loop())
        logger.info(f"Started continuous capture (interval: {self.capture_interval}s)")
    
    async def stop_continuous_capture(self):
        """Stop continuous screen capture"""
        self.is_capturing = False
        if self.capture_task:
            self.capture_task.cancel()
            try:
                await self.capture_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped continuous capture")
    
    async def _capture_loop(self):
        """Main capture loop"""
        while self.is_capturing:
            try:
                start_time = time.time()
                
                # Capture screen
                capture = self.capture_screen()
                
                if capture:
                    # Check for significant changes
                    if self.has_significant_change(capture.image):
                        self.last_capture = capture
                        self.capture_count += 1
                        
                        # Update hash
                        self.last_hash = self.get_image_hash(capture.image)
                        
                        # Notify callbacks
                        await self._notify_callbacks(capture)
                        
                        logger.debug(f"Capture #{self.capture_count} completed")
                    else:
                        logger.debug("No significant changes detected, skipping")
                
                # Wait for next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.capture_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                await asyncio.sleep(self.capture_interval)
    
    async def _notify_callbacks(self, capture: ScreenCapture):
        """Notify all registered callbacks"""
        for callback in self.capture_callbacks:
            try:
                await callback(capture)
            except Exception as e:
                logger.error(f"Callback error: {e}")
    
    def add_capture_callback(self, callback):
        """Add callback for new captures"""
        self.capture_callbacks.append(callback)
    
    def remove_capture_callback(self, callback):
        """Remove capture callback"""
        if callback in self.capture_callbacks:
            self.capture_callbacks.remove(callback)
    
    def get_capture_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            'is_capturing': self.is_capturing,
            'capture_count': self.capture_count,
            'capture_interval': self.capture_interval,
            'last_capture_time': self.last_capture.timestamp.isoformat() if self.last_capture else None,
            'screens': self.screens,
            'differential_mode': self.differential_mode,
            'compression_quality': self.compression_quality
        }

async def test_screen_capture():
    """Test screen capture functionality"""
    print("🖥️ Testing Screen Capture Module")
    print("=" * 50)
    
    # Create capture module
    capture_module = ScreenCaptureModule(capture_interval=2.0)
    
    # Test single capture
    print("\n📸 Testing single capture...")
    capture = capture_module.capture_screen()
    
    if capture:
        print(f"✅ Captured screen: {capture.image.size}")
        print(f"   Timestamp: {capture.timestamp}")
        print(f"   Base64 size: {len(capture.base64_image)} chars")
        
        # Test OCR preprocessing
        ocr_image = capture_module.preprocess_for_ocr(capture.image)
        print(f"   OCR preprocessed size: {ocr_image.size}")
    else:
        print("❌ Failed to capture screen")
    
    # Test continuous capture
    print("\n🔄 Testing continuous capture...")
    
    capture_count = 0
    async def capture_callback(capture):
        nonlocal capture_count
        capture_count += 1
        print(f"   Capture #{capture_count} at {capture.timestamp.strftime('%H:%M:%S')}")
    
    capture_module.add_capture_callback(capture_callback)
    
    # Start capture
    await capture_module.start_continuous_capture()
    
    # Run for 6 seconds
    await asyncio.sleep(6)
    
    # Stop capture
    await capture_module.stop_continuous_capture()
    
    # Show stats
    stats = capture_module.get_capture_stats()
    print(f"\n📊 Capture Statistics:")
    print(f"   Total captures: {stats['capture_count']}")
    print(f"   Screens detected: {len(stats['screens'])}")
    
    print("\n✅ Screen capture test complete!")

if __name__ == "__main__":
    asyncio.run(test_screen_capture())