#!/usr/bin/env python3
"""
Simple backend startup script that handles import issues
"""

import os
import sys
import subprocess

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['USE_TORCH'] = '1'
os.environ['USE_TF'] = '0'

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

print("🚀 Starting JARVIS Backend...")
print("=" * 50)

# Check for API key
if not os.getenv("ANTHROPIC_API_KEY"):
    print("⚠️  Warning: ANTHROPIC_API_KEY not set")
    print("   JARVIS will have limited functionality")
else:
    print("✅ Anthropic API key found")

# Check optional dependencies
try:
    import librosa
    print("✅ Audio processing (librosa) available")
except ImportError:
    print("⚠️  Audio ML features limited (librosa not installed)")

try:
    import torch
    print("✅ PyTorch available for deep learning")
except ImportError:
    print("⚠️  Deep learning features limited (PyTorch not installed)")

# Run the backend
print("\n🎯 Starting FastAPI server on port 8000...")
print("=" * 50 + "\n")

try:
    subprocess.run([sys.executable, "main.py", "--port", "8000"])
except KeyboardInterrupt:
    print("\n\n✋ Backend stopped by user")
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)