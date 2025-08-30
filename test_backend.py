#!/usr/bin/env python3
"""
Test script to verify backend starts correctly
"""

import requests
import time
import subprocess
import os
import signal
import sys

def test_backend():
    """Test that backend starts and responds correctly"""
    print("Starting backend test...")
    
    # Start the backend
    backend_path = os.path.join(os.path.dirname(__file__), "backend")
    process = subprocess.Popen(
        [sys.executable, "main.py", "--port", "8000"],
        cwd=backend_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    print(f"Backend started with PID: {process.pid}")
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(10)
    
    # Test endpoints
    endpoints = [
        ("http://localhost:8000/health", "Health Check"),
        ("http://localhost:8000/docs", "API Docs"),
        ("http://localhost:8000/voice/jarvis/status", "Voice Status"),
    ]
    
    for url, name in endpoints:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {name}: OK")
            else:
                print(f"✗ {name}: {response.status_code}")
        except Exception as e:
            print(f"✗ {name}: {str(e)}")
    
    # Check process status
    if process.poll() is None:
        print("\n✓ Backend process is running")
    else:
        print(f"\n✗ Backend process exited with code: {process.returncode}")
        stdout, stderr = process.communicate()
        print("STDOUT:", stdout[-500:] if stdout else "None")
        print("STDERR:", stderr[-500:] if stderr else "None")
    
    # Clean up
    print("\nShutting down backend...")
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    
    print("Test complete")

if __name__ == "__main__":
    test_backend()