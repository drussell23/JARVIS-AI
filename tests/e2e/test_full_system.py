#!/usr/bin/env python3
"""
Test full system startup
"""

import subprocess
import time
import requests
import os
import signal
import sys

def test_full_system():
    """Test full system startup"""
    print("=== JARVIS System Test ===\n")
    
    # Kill any existing processes
    print("1. Cleaning up existing processes...")
    subprocess.run("killall -9 python3 python 2>/dev/null || true", shell=True)
    time.sleep(2)
    
    # Start backend
    print("\n2. Starting backend on port 8000...")
    backend_process = subprocess.Popen(
        [sys.executable, "start_system.py", "--backend-only", "--no-browser"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    print(f"   Backend PID: {backend_process.pid}")
    
    # Wait for backend to start
    print("\n3. Waiting for backend to initialize...")
    start_time = time.time()
    backend_ready = False
    
    while time.time() - start_time < 60:  # Wait up to 60 seconds
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                backend_ready = True
                print("   ✓ Backend is ready!")
                break
        except:
            pass
        
        # Check if process is still running
        if backend_process.poll() is not None:
            print(f"   ✗ Backend exited with code: {backend_process.returncode}")
            stdout, stderr = backend_process.communicate()
            print("   STDOUT (last 500 chars):", stdout[-500:] if stdout else "None")
            print("   STDERR (last 500 chars):", stderr[-500:] if stderr else "None")
            return False
        
        print(f"   Waiting... ({int(time.time() - start_time)}s)")
        time.sleep(3)
    
    if not backend_ready:
        print("   ✗ Backend failed to start within 60 seconds")
        backend_process.terminate()
        return False
    
    # Test endpoints
    print("\n4. Testing API endpoints...")
    endpoints = [
        ("http://localhost:8000/docs", "API Documentation"),
        ("http://localhost:8000/voice/jarvis/status", "Voice Status"),
        ("http://localhost:8000/vision/status", "Vision Status"),
    ]
    
    all_ok = True
    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"   ✓ {name}: OK")
            else:
                print(f"   ✗ {name}: {response.status_code}")
                all_ok = False
        except Exception as e:
            print(f"   ✗ {name}: {str(e)}")
            all_ok = False
    
    # Cleanup
    print("\n5. Cleaning up...")
    backend_process.terminate()
    try:
        backend_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        backend_process.kill()
        backend_process.wait()
    
    print("\n=== Test Complete ===")
    return backend_ready and all_ok

if __name__ == "__main__":
    success = test_full_system()
    sys.exit(0 if success else 1)