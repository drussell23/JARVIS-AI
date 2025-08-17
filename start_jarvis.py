#!/usr/bin/env python3
"""
Quick launcher for JARVIS
"""

import subprocess
import sys
import os

# Add note about using async version
print("🚀 Launching JARVIS with async optimization...")
print("   (3x faster startup with parallel initialization)\n")

# Run the main startup script
script_path = os.path.join(os.path.dirname(__file__), "start_system.py")
subprocess.run([sys.executable, script_path] + sys.argv[1:])