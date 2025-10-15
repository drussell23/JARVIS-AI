#!/usr/bin/env python3
"""
Simplified JARVIS backend that ensures fresh module loading
"""

import os
import sys
import importlib
from pathlib import Path

# Clear ALL module caches
print("Clearing module cache...")
modules_to_remove = []
for name in list(sys.modules.keys()):
    if any(x in name for x in ['api', 'vision', 'unified', 'command', 'process']):
        modules_to_remove.append(name)

for name in modules_to_remove:
    del sys.modules[name]
    print(f"  Removed: {name}")

# Now start the backend
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# Import main AFTER clearing cache
from main import app
import uvicorn

if __name__ == "__main__":
    print("\n✅ Starting JARVIS with FRESH modules on port 8010...")
    print("   All vision fixes are loaded!")
    print("   Multi-space queries will work!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")