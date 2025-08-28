"""
Asynchronous Screen Capture with Performance Optimizations
Provides non-blocking screen capture for faster vision responses
"""

import asyncio
import os
import tempfile
import time
import logging
from typing import Optional, Dict, Any, Tuple
import platform
from PIL import Image
import numpy as np
from functools import lru_cache
import hashlib

logger = logging.getLogger(__name__)

class AsyncScreenCapture:
    """High-performance async screen capture"""
    
    def __init__(self):
        self.platform = platform.system().lower()
        self._last_capture_time = 0
        self._last_capture_hash = None
        self._capture_cache = {}
        self._capture_lock = asyncio.Lock()
        
    async def capture_screen_async(self, 
                                  use_cache: bool = True,
                                  cache_duration: float = 0.5) -> Optional[Image.Image]:
        """
        Capture screen asynchronously with intelligent caching
        
        Args:
            use_cache: Whether to use cached screenshot if recent
            cache_duration: How long to cache screenshots (seconds)
        """
        async with self._capture_lock:
            # Check cache first
            current_time = time.time()
            if use_cache and (current_time - self._last_capture_time) < cache_duration:
                if self._last_capture_hash in self._capture_cache:
                    logger.debug(f"Using cached screenshot (age: {current_time - self._last_capture_time:.2f}s)")
                    return self._capture_cache[self._last_capture_hash]
            
            # Capture based on platform
            if self.platform == "darwin":
                image = await self._capture_macos_async()
            elif self.platform == "linux":
                image = await self._capture_linux_async()
            elif self.platform == "win32":
                image = await self._capture_windows_async()
            else:
                logger.error(f"Unsupported platform: {self.platform}")
                return None
            
            if image:
                # Update cache
                self._last_capture_time = current_time
                image_hash = self._compute_image_hash(image)
                self._last_capture_hash = image_hash
                
                # Clean old cache entries
                if len(self._capture_cache) > 5:
                    self._capture_cache.clear()
                
                self._capture_cache[image_hash] = image
                
            return image
    
    async def _capture_macos_async(self) -> Optional[Image.Image]:
        """Async screen capture for macOS using subprocess"""
        temp_file = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                temp_file = tmp.name
            
            # Run screencapture asynchronously
            proc = await asyncio.create_subprocess_exec(
                "screencapture", "-x", "-C", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=2.0  # 2 second timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                logger.error("Screenshot capture timed out")
                return None
            
            if proc.returncode == 0 and os.path.exists(temp_file):
                # Load image asynchronously
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, Image.open, temp_file)
                return image
            else:
                logger.error(f"Screenshot failed with code {proc.returncode}")
                return None
                
        except Exception as e:
            logger.error(f"macOS async capture failed: {e}")
            return None
        finally:
            # Clean up temp file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    async def _capture_linux_async(self) -> Optional[Image.Image]:
        """Async screen capture for Linux"""
        try:
            # Try scrot first
            temp_file = tempfile.mktemp(suffix=".png")
            proc = await asyncio.create_subprocess_exec(
                "scrot", "-z", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await proc.communicate()
            
            if proc.returncode == 0 and os.path.exists(temp_file):
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, Image.open, temp_file)
                os.unlink(temp_file)
                return image
                
        except FileNotFoundError:
            logger.warning("scrot not found, trying import")
            
        # Fallback to import
        try:
            temp_file = tempfile.mktemp(suffix=".png")
            proc = await asyncio.create_subprocess_exec(
                "import", "-window", "root", temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            await proc.communicate()
            
            if proc.returncode == 0 and os.path.exists(temp_file):
                loop = asyncio.get_event_loop()
                image = await loop.run_in_executor(None, Image.open, temp_file)
                os.unlink(temp_file)
                return image
                
        except Exception as e:
            logger.error(f"Linux async capture failed: {e}")
            
        return None
    
    async def _capture_windows_async(self) -> Optional[Image.Image]:
        """Async screen capture for Windows"""
        try:
            # Use PIL ImageGrab in executor
            from PIL import ImageGrab
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(None, ImageGrab.grab)
            return image
        except Exception as e:
            logger.error(f"Windows async capture failed: {e}")
            return None
    
    def _compute_image_hash(self, image: Image.Image) -> str:
        """Compute a fast hash of the image for caching"""
        # Resize to small size for fast hashing
        small = image.resize((64, 64), Image.Resampling.NEAREST)
        # Convert to bytes
        pixels = small.tobytes()
        # Return hash
        return hashlib.md5(pixels).hexdigest()
    
    async def capture_with_compression(self, 
                                     max_dimension: int = 1920,
                                     quality: int = 85) -> Optional[Dict[str, Any]]:
        """
        Capture and optionally compress screenshot for faster processing
        
        Args:
            max_dimension: Maximum width or height
            quality: JPEG quality if converting (1-100)
        """
        image = await self.capture_screen_async()
        
        if not image:
            return None
        
        result = {
            'original_size': image.size,
            'compressed': False
        }
        
        # Check if compression needed
        width, height = image.size
        if max(width, height) > max_dimension:
            # Calculate new dimensions
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                lambda: image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            )
            result['compressed'] = True
            result['new_size'] = image.size
        
        result['image'] = image
        return result
    
    async def capture_region_async(self, 
                                  x: int, y: int, 
                                  width: int, height: int) -> Optional[Image.Image]:
        """Capture a specific region of the screen"""
        full_screen = await self.capture_screen_async()
        
        if not full_screen:
            return None
        
        # Crop to region
        try:
            region = full_screen.crop((x, y, x + width, y + height))
            return region
        except Exception as e:
            logger.error(f"Failed to crop region: {e}")
            return None

# Global instance
_async_capture = None

def get_async_capture() -> AsyncScreenCapture:
    """Get singleton async capture instance"""
    global _async_capture
    if _async_capture is None:
        _async_capture = AsyncScreenCapture()
    return _async_capture

async def capture_screen_optimized() -> Optional[Image.Image]:
    """Convenience function for optimized screen capture"""
    capture = get_async_capture()
    return await capture.capture_screen_async()

async def capture_for_analysis(max_size: int = 1920) -> Optional[Dict[str, Any]]:
    """Capture and prepare screenshot for AI analysis"""
    capture = get_async_capture()
    return await capture.capture_with_compression(max_dimension=max_size)