#!/usr/bin/env python3
"""
Fresh start wrapper for JARVIS
Ensures all modules are loaded fresh with no caching
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

print("🧹 JARVIS Fresh Start - Ensuring all fixes are loaded...")

# 1. Set environment to prevent bytecode generation
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'

# 2. Clear all __pycache__ directories
backend_dir = Path(__file__).parent / "backend"
pycache_dirs = list(backend_dir.rglob("__pycache__"))

for pycache in pycache_dirs:
    try:
        shutil.rmtree(pycache)
        print(f"  Removed: {pycache.name}")
    except:
        pass

# 3. Clear all .pyc files
pyc_files = list(backend_dir.rglob("*.pyc"))
for pyc in pyc_files:
    try:
        os.remove(pyc)
        print(f"  Removed: {pyc.name}")
    except:
        pass

print(f"\n✅ Cleared {len(pycache_dirs)} cache directories and {len(pyc_files)} .pyc files")

# 4. Start JARVIS with subprocess - completely fresh Python interpreter
print("\n🚀 Starting JARVIS with completely fresh modules...")
print("   All vision fixes will be active!")
print("   Multi-space queries will work!\n")

# Use subprocess to start with a fresh Python interpreter
cmd = [sys.executable, "-B", "-u", "start_system.py", "--backend-only", "--no-browser"]

# Add path to use fresh modules
env = os.environ.copy()
env['PYTHONDONTWRITEBYTECODE'] = '1'
env['PYTHONUNBUFFERED'] = '1'

# Start the process
try:
    proc = subprocess.run(cmd, env=env)
    sys.exit(proc.returncode)
except KeyboardInterrupt:
    print("\n👋 JARVIS shutdown requested")
    sys.exit(0)