#!/usr/bin/env python3
"""
Simple test to debug backend startup issues
"""

import os
import sys
import subprocess

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), "backend")
sys.path.insert(0, backend_path)

# Set environment
os.environ["JARVIS_MEMORY_LEVEL"] = "critical"
os.environ["JARVIS_MODEL_PRECISION"] = "8bit"
os.environ["DYLD_LIBRARY_PATH"] = os.path.join(backend_path, "swift_bridge/.build/release")

print("🔍 Testing backend startup directly...")
print("=" * 50)

# Try running uvicorn directly to see errors
cmd = [
    sys.executable, "-m", "uvicorn", 
    "main:app",
    "--host", "0.0.0.0",
    "--port", "8010",
    "--workers", "1"
]

print(f"Running: {' '.join(cmd)}")
print(f"Working dir: {backend_path}")
print(f"Python: {sys.executable}")
print(f"Memory level: {os.environ.get('JARVIS_MEMORY_LEVEL')}")
print()

try:
    # Run with direct output to see errors
    subprocess.run(cmd, cwd=backend_path, check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Backend failed with exit code: {e.returncode}")
except KeyboardInterrupt:
    print("\n✅ Backend stopped by user")
except Exception as e:
    print(f"❌ Error: {e}")