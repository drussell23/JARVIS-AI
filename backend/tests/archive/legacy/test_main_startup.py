#!/usr/bin/env python3
"""
Test script to verify main.py starts correctly
"""
import subprocess
import time
import requests
import os
import sys

def test_main_startup():
    print("🧪 Testing main.py startup...")
    
    # Set environment variables
    env = os.environ.copy()
    env["OPTIMIZE_STARTUP"] = "true"
    env["BACKEND_PARALLEL_IMPORTS"] = "true"
    env["BACKEND_LAZY_LOAD_MODELS"] = "true"
    env["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Start main.py
    print("Starting main.py...")
    process = subprocess.Popen(
        [sys.executable, "main.py", "--port", "8011"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Monitor output
    start_time = time.time()
    health_checked = False
    
    try:
        while True:
            line = process.stdout.readline()
            if not line:
                break
                
            print(f"  {line.strip()}")
            
            # Check for startup completion
            if "Application startup complete" in line:
                print("\n✅ Backend started successfully!")
                
                # Test health endpoint
                time.sleep(2)
                try:
                    response = requests.get("http://localhost:8011/health", timeout=5)
                    if response.status_code == 200:
                        print("✅ Health check passed!")
                        health_data = response.json()
                        print(f"  Mode: {health_data.get('mode')}")
                        print(f"  Components: {health_data.get('components')}")
                        health_checked = True
                        break
                except Exception as e:
                    print(f"❌ Health check failed: {e}")
                    
            # Check for errors
            if "ERROR" in line or "Failed" in line:
                print(f"\n❌ Error detected: {line.strip()}")
                
            # Timeout check
            if time.time() - start_time > 60:
                print("\n❌ Timeout: Backend did not start within 60 seconds")
                break
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        # Cleanup
        process.terminate()
        time.sleep(1)
        if process.poll() is None:
            process.kill()
            
    return health_checked

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    success = test_main_startup()
    
    if success:
        print("\n✅ main.py is working correctly!")
        print("The backend should start properly when using start_system.py")
    else:
        print("\n❌ main.py has issues - check the output above")
        print("The system may fall back to main_minimal.py")
        
    sys.exit(0 if success else 1)