#!/usr/bin/env python3
"""
Simple window capture using screencapture command
More reliable fallback for macOS
"""

import os
import cv2
import numpy as np
import subprocess
import tempfile
from typing import Optional
from dataclasses import dataclass

@dataclass
class SimpleCapture:
    """Simple capture result"""
    image: np.ndarray
    success: bool

def capture_screen_simple() -> Optional[np.ndarray]:
    """Capture screen using macOS screencapture command"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp_path = tmp.name
        
        # Use screencapture command (most reliable on macOS)
        cmd = ['screencapture', '-x', '-C', tmp_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(tmp_path):
            # Read the image
            image = cv2.imread(tmp_path)
            
            # Clean up
            os.unlink(tmp_path)
            
            # Convert BGR to RGB
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return None
        
    except Exception as e:
        print(f"Simple capture error: {e}")
        return None

def capture_window_area(x: int, y: int, width: int, height: int) -> Optional[np.ndarray]:
    """Capture a specific area of the screen"""
    try:
        # First capture entire screen
        full_screen = capture_screen_simple()
        
        if full_screen is not None:
            # Crop to window area
            x = max(0, x)
            y = max(0, y)
            
            # Ensure we don't exceed image bounds
            max_y = min(y + height, full_screen.shape[0])
            max_x = min(x + width, full_screen.shape[1])
            
            cropped = full_screen[y:max_y, x:max_x]
            
            return cropped if cropped.size > 0 else None
        
        return None
        
    except Exception as e:
        print(f"Window area capture error: {e}")
        return None

def test_simple_capture():
    """Test the simple capture method"""
    print("Testing simple capture...")
    
    # Test full screen
    image = capture_screen_simple()
    if image is not None:
        print(f"✅ Screen captured: {image.shape}")
        cv2.imwrite("simple_capture_test.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("Saved to: simple_capture_test.png")
    else:
        print("❌ Screen capture failed")
    
    # Test window area
    if image is not None:
        # Capture center quarter of screen
        h, w = image.shape[:2]
        window_img = capture_window_area(w//4, h//4, w//2, h//2)
        
        if window_img is not None:
            print(f"✅ Window area captured: {window_img.shape}")
        else:
            print("❌ Window area capture failed")

if __name__ == "__main__":
    test_simple_capture()