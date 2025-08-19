#!/usr/bin/env python3
"""
Debug backend startup issues
"""

import os
import sys
import subprocess
import time

print("🔍 JARVIS Backend Debug Tool")
print("=" * 50)

# Check environment
print("\n1️⃣ Checking environment...")
api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    print(f"✅ ANTHROPIC_API_KEY found: {api_key[:10]}...")
else:
    print("❌ ANTHROPIC_API_KEY not set!")

# Check ports
print("\n2️⃣ Checking ports...")
def check_port(port):
    result = subprocess.run(
        f"lsof -i:{port}", 
        shell=True, 
        capture_output=True, 
        text=True
    )
    return result.returncode != 0

for port in [8000, 8001]:
    if check_port(port):
        print(f"✅ Port {port} is available")
    else:
        print(f"❌ Port {port} is in use!")
        # Try to kill process
        subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True)
        time.sleep(1)
        if check_port(port):
            print(f"   ✅ Killed process on port {port}")

# Check dependencies
print("\n3️⃣ Checking critical dependencies...")
try:
    import fastapi
    print("✅ FastAPI installed")
except ImportError:
    print("❌ FastAPI not installed!")

try:
    import anthropic
    print("✅ Anthropic SDK installed")
except ImportError:
    print("❌ Anthropic SDK not installed!")

# Try minimal startup
print("\n4️⃣ Starting backend with debug output...")
print("Running: python main.py --port 8000")
print("-" * 50)

# Set environment
env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["USE_CLAUDE"] = "1"

# Start backend with full output
try:
    process = subprocess.Popen(
        [sys.executable, "main.py", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True
    )
    
    # Read output for 10 seconds
    start_time = time.time()
    while time.time() - start_time < 10:
        line = process.stdout.readline()
        if line:
            print(line.strip())
        if process.poll() is not None:
            print(f"\n❌ Process exited with code: {process.returncode}")
            break
    
    if process.poll() is None:
        print("\n✅ Backend appears to be running!")
        print("Try accessing: http://localhost:8000/docs")
        process.terminate()
        
except Exception as e:
    print(f"\n❌ Error starting backend: {e}")

print("\n" + "=" * 50)
print("💡 If backend hangs during startup:")
print("1. Check if all vision dependencies are installed")
print("2. Try disabling vision features temporarily")
print("3. Check memory usage (Activity Monitor)")
print("4. Run: pip install -r requirements.txt")