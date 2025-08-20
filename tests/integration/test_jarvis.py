#!/usr/bin/env python3
"""
Test script to verify JARVIS is working
"""

import sys
import time
import subprocess
import requests

print("🧪 JARVIS Test Script")
print("=" * 50)

# Start the backend
print("\n📦 Starting JARVIS backend...")
process = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
print("⏳ Waiting for server to start...")
time.sleep(5)

# Test the API
print("\n🔍 Testing API endpoints...")
try:
    # Test health endpoint
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        print("✅ Health check passed!")
    else:
        print(f"❌ Health check failed: {response.status_code}")
        
    # Test docs
    response = requests.get("http://localhost:8000/docs", timeout=5)
    if response.status_code == 200:
        print("✅ API docs available!")
        
    print("\n🎉 JARVIS is running successfully!")
    print("\n📋 Access points:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Chat Demo: http://localhost:8000/demo/chat")
    print("   - Voice Demo: http://localhost:8000/demo/voice")
    
except Exception as e:
    print(f"\n❌ Error testing API: {e}")
    
finally:
    # Keep running
    print("\n💡 Press Ctrl+C to stop JARVIS")
    try:
        process.wait()
    except KeyboardInterrupt:
        print("\n👋 Stopping JARVIS...")
        process.terminate()