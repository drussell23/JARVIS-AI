#!/usr/bin/env python3
"""
Direct Swift capture execution for purple indicator
This bypasses all the complex bridging and runs Swift directly
"""

import subprocess
import os
import logging
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)

class DirectSwiftCapture:
    """Execute Swift capture directly for purple indicator"""
    
    def __init__(self):
        self.swift_script = Path(__file__).parent / "simple_swift_capture.swift"
        self.capture_process = None
        self.is_capturing = False
        
    async def start_capture(self) -> bool:
        """Start Swift capture - shows purple indicator"""
        if self.is_capturing:
            logger.warning("Capture already running")
            return True
            
        logger.info("[DIRECT] Starting Swift capture for purple indicator...")
        
        try:
            # Run the Swift script directly
            self.capture_process = subprocess.Popen(
                ["swift", str(self.swift_script), "--capture"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.is_capturing = True
            
            # Don't monitor output asynchronously - it blocks the Swift execution
            # The Swift script will run for 30 seconds with purple indicator
            
            logger.info("[DIRECT] ✅ Swift capture started - purple indicator should be visible!")
            return True
            
        except Exception as e:
            logger.error(f"[DIRECT] Failed to start Swift capture: {e}")
            return False
    
    async def _monitor_output(self):
        """Monitor Swift process output"""
        if not self.capture_process:
            return
            
        try:
            # Read output in real-time
            for line in iter(self.capture_process.stdout.readline, ''):
                if line:
                    logger.info(f"[SWIFT OUTPUT] {line.strip()}")
                    
            # Process ended
            self.is_capturing = False
            return_code = self.capture_process.wait()
            
            if return_code == 0:
                logger.info("[DIRECT] Swift capture completed successfully")
            else:
                stderr = self.capture_process.stderr.read()
                logger.error(f"[DIRECT] Swift capture failed with code {return_code}: {stderr}")
                
        except Exception as e:
            logger.error(f"[DIRECT] Error monitoring Swift output: {e}")
            self.is_capturing = False
    
    def stop_capture(self):
        """Stop capture if running"""
        if self.capture_process and self.is_capturing:
            logger.info("[DIRECT] Stopping Swift capture...")
            self.capture_process.terminate()
            self.capture_process = None
            self.is_capturing = False
            logger.info("[DIRECT] Swift capture stopped")

# Global instance
_direct_capture = None

def get_direct_capture():
    """Get or create direct capture instance"""
    global _direct_capture
    if _direct_capture is None:
        _direct_capture = DirectSwiftCapture()
    return _direct_capture

async def start_direct_swift_capture() -> bool:
    """Start direct Swift capture with purple indicator"""
    capture = get_direct_capture()
    return await capture.start_capture()

def stop_direct_swift_capture():
    """Stop direct Swift capture"""
    capture = get_direct_capture()
    capture.stop_capture()

def is_direct_capturing() -> bool:
    """Check if capturing"""
    capture = get_direct_capture()
    return capture.is_capturing

# Test function
async def test_direct_capture():
    """Test the direct capture"""
    print("\n🟣 Testing Direct Swift Capture")
    print("=" * 60)
    
    success = await start_direct_swift_capture()
    
    if success:
        print("✅ Capture started - check for purple indicator!")
        print("⏳ Waiting 10 seconds...")
        await asyncio.sleep(10)
        
        print("Stopping capture...")
        stop_direct_swift_capture()
        print("✅ Test complete!")
    else:
        print("❌ Failed to start capture")

if __name__ == "__main__":
    asyncio.run(test_direct_capture())