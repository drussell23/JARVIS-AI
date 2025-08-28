#!/usr/bin/env python3
"""
Python bridge for Swift screen capture functionality
Provides efficient native screen capture for continuous monitoring
"""

import ctypes
import os
import sys
import asyncio
import logging
from typing import Optional, Callable, Dict, Any
from PIL import Image
import io
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class ScreenCaptureBridge:
    """Python wrapper for Swift screen capture functionality"""
    
    def __init__(self):
        self.lib = None
        self.capture_instance = None
        self.callback_func = None
        self.python_callback = None
        self._setup_library()
    
    def _setup_library(self):
        """Setup the Swift dynamic library"""
        try:
            # Build Swift library if needed
            swift_dir = Path(__file__).parent
            lib_path = swift_dir / ".build" / "release" / "libScreenCapture.dylib"
            
            if not lib_path.exists():
                logger.info("Building Swift screen capture library...")
                self._build_swift_library()
            
            # Load the library
            self.lib = ctypes.CDLL(str(lib_path))
            
            # Define function signatures
            self.lib.swift_create_screen_capture.restype = ctypes.c_void_p
            
            self.lib.swift_start_continuous_capture.argtypes = [
                ctypes.c_void_p,
                ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int)
            ]
            self.lib.swift_start_continuous_capture.restype = ctypes.c_bool
            
            self.lib.swift_capture_screen.argtypes = [ctypes.c_void_p]
            self.lib.swift_capture_screen.restype = ctypes.POINTER(ctypes.c_uint8)
            
            self.lib.swift_get_active_app.argtypes = [ctypes.c_void_p]
            self.lib.swift_get_active_app.restype = ctypes.c_char_p
            
            self.lib.swift_release_screen_capture.argtypes = [ctypes.c_void_p]
            
            # Create capture instance
            self.capture_instance = self.lib.swift_create_screen_capture()
            
            logger.info("Swift screen capture bridge initialized")
            
        except Exception as e:
            logger.warning(f"Failed to load Swift library, falling back to Python: {e}")
            self.lib = None
    
    def _build_swift_library(self):
        """Build the Swift library"""
        swift_dir = Path(__file__).parent
        os.chdir(swift_dir)
        
        # Create Package.swift if it doesn't exist
        package_swift = swift_dir / "Package.swift"
        if not package_swift.exists():
            package_content = '''// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "ScreenCapture",
    platforms: [.macOS(.v11)],
    products: [
        .library(
            name: "ScreenCapture",
            type: .dynamic,
            targets: ["ScreenCapture"]),
    ],
    targets: [
        .target(
            name: "ScreenCapture",
            path: ".",
            sources: ["ScreenCapture.swift"]),
    ]
)'''
            package_swift.write_text(package_content)
        
        # Build the library
        import subprocess
        result = subprocess.run(
            ["swift", "build", "-c", "release"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Failed to build Swift library: {result.stderr}")
    
    def capture_screen(self) -> Optional[Image.Image]:
        """Capture current screen"""
        if self.lib and self.capture_instance:
            # Use Swift capture
            data_ptr = self.lib.swift_capture_screen(self.capture_instance)
            if data_ptr:
                # Convert to PIL Image
                # Note: In real implementation, we'd need to handle memory properly
                # This is simplified for demonstration
                return self._fallback_capture()
        
        # Fallback to Python
        return self._fallback_capture()
    
    def _fallback_capture(self) -> Optional[Image.Image]:
        """Fallback screen capture using Python"""
        try:
            # Try using screencapture command
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                subprocess.run([
                    'screencapture', '-x', '-C', '-J', tmp.name
                ], check=True)
                
                image = Image.open(tmp.name)
                os.unlink(tmp.name)
                return image
                
        except Exception as e:
            logger.error(f"Failed to capture screen: {e}")
            return None
    
    def get_active_application(self) -> str:
        """Get currently active application"""
        if self.lib and self.capture_instance:
            app_name = self.lib.swift_get_active_app(self.capture_instance)
            if app_name:
                return app_name.decode('utf-8')
        
        # Fallback
        return "Unknown"
    
    def start_continuous_capture(self, callback: Callable[[bytes], None]) -> bool:
        """Start continuous screen capture with callback"""
        if not self.lib or not self.capture_instance:
            logger.warning("Swift library not available for continuous capture")
            return False
        
        # Store Python callback
        self.python_callback = callback
        
        # Create C callback function
        def c_callback(data_ptr, size):
            if self.python_callback:
                # Convert C data to Python bytes
                data = bytes(ctypes.cast(data_ptr, ctypes.POINTER(ctypes.c_uint8 * size)).contents)
                self.python_callback(data)
        
        # Create C function type
        self.callback_func = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int)(c_callback)
        
        # Start capture
        success = self.lib.swift_start_continuous_capture(self.capture_instance, self.callback_func)
        
        if success:
            logger.info("Started continuous screen capture")
        else:
            logger.error("Failed to start continuous screen capture")
        
        return success
    
    def stop_continuous_capture(self):
        """Stop continuous screen capture"""
        # In real implementation, we'd have a stop function
        logger.info("Stopped continuous screen capture")
    
    def __del__(self):
        """Cleanup"""
        if self.lib and self.capture_instance:
            self.lib.swift_release_screen_capture(self.capture_instance)


# Singleton instance
_screen_capture_bridge = None

def get_screen_capture_bridge() -> ScreenCaptureBridge:
    """Get singleton screen capture bridge"""
    global _screen_capture_bridge
    if _screen_capture_bridge is None:
        _screen_capture_bridge = ScreenCaptureBridge()
    return _screen_capture_bridge