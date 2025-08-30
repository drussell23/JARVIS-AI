#!/usr/bin/env python3
"""Debug script to see exact backend error"""

import os
import sys
import subprocess

# Setup environment
backend_path = os.path.join(os.path.dirname(__file__), "backend")
os.environ["JARVIS_MEMORY_LEVEL"] = "critical"
os.environ["JARVIS_MODEL_PRECISION"] = "8bit"
os.environ["DYLD_LIBRARY_PATH"] = os.path.join(backend_path, "swift_bridge/.build/release")

print("🔍 Running backend with full error output...")
print("=" * 50)

# Run with stderr and stdout visible
cmd = [
    sys.executable, "-c",
    """
import sys
sys.path.insert(0, '{backend_path}')

print("Importing main...")
try:
    from main import app
    print("✅ Main imported successfully")
    
    # Try to start uvicorn
    import uvicorn
    print("Starting uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8010, workers=1)
    
except Exception as e:
    print(f"❌ Error: {{e}}")
    import traceback
    traceback.print_exc()
""".format(backend_path=backend_path)
]

subprocess.run(cmd, cwd=backend_path)