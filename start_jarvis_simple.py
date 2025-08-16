#!/usr/bin/env python3
"""
Simple JARVIS startup script
"""

import os
import sys
import subprocess
import time
import webbrowser

print("""
╔══════════════════════════════════════════════════╗
║             🤖 JARVIS AI System                  ║
║          Starting in simplified mode...          ║
╚══════════════════════════════════════════════════╝
""")

# Start the backend
print("\n🚀 Starting JARVIS backend...")

# Set environment variables
os.environ["USE_QUANTIZED_MODELS"] = "true"
os.environ["PREFER_LANGCHAIN"] = "0"  # Disable LangChain initially

# Change to backend directory and start server
cmd = [
    sys.executable, "-m", "uvicorn", 
    "backend.main:app", 
    "--reload", 
    "--host", "0.0.0.0", 
    "--port", "8000"
]

print(f"Running: {' '.join(cmd)}")
process = subprocess.Popen(cmd)

# Wait for server to start
print("\n⏳ Waiting for server to start...")
time.sleep(5)

# Open browser
print("\n🌐 Opening JARVIS in browser...")
webbrowser.open("http://localhost:8000/docs")

print("\n✅ JARVIS is running!")
print("\n📋 Access points:")
print("   - API Docs: http://localhost:8000/docs")
print("   - Chat Demo: http://localhost:8000/demo/chat")
print("   - Voice Demo: http://localhost:8000/demo/voice")
print("\n💡 Press Ctrl+C to stop JARVIS")

try:
    process.wait()
except KeyboardInterrupt:
    print("\n\n👋 Stopping JARVIS...")
    process.terminate()
    print("✅ JARVIS stopped successfully")