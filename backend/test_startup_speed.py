#!/usr/bin/env python3
"""Test backend startup speed with lazy loading"""

import time
import subprocess
import os
import signal
import sys

def test_startup():
    """Test backend startup and measure time"""
    print("🚀 Testing backend startup with lazy loading...")
    
    start_time = time.time()
    
    # Start backend
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Monitor output
    ml_loading_started = False
    server_started = False
    vision_loaded = False
    
    try:
        for line in process.stdout:
            print(line.rstrip())
            
            # Check for key milestones
            if "Lazy Dynamic Vision Engine initialized" in line:
                vision_loaded = True
                print(f"✅ Vision engine loaded (lazy): {time.time() - start_time:.1f}s")
                
            if "Starting parallel ML model initialization" in line:
                ml_loading_started = True
                print(f"🧠 ML loading started: {time.time() - start_time:.1f}s")
                
            if "Waiting for application startup" in line:
                server_started = True
                print(f"✅ Server ready: {time.time() - start_time:.1f}s")
                
            if "Application startup complete" in line:
                print(f"✅ Fully started: {time.time() - start_time:.1f}s")
                break
                
            # Stop after 30 seconds
            if time.time() - start_time > 30:
                print(f"⏱️ Timeout after 30 seconds")
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        # Kill the process
        os.kill(process.pid, signal.SIGTERM)
        process.wait()
        
    print(f"\n📊 Summary:")
    print(f"  Total startup time: {time.time() - start_time:.1f}s")
    print(f"  Vision loaded: {'Yes (lazy)' if vision_loaded else 'No'}")
    print(f"  ML loading: {'Started' if ml_loading_started else 'Not started'}")
    print(f"  Server ready: {'Yes' if server_started else 'No'}")

if __name__ == "__main__":
    test_startup()