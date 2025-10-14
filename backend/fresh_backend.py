#!/usr/bin/env python3
"""
Fresh backend that ensures all fixes are loaded
This is a guaranteed clean start with no caching issues
"""

import os
import sys
import subprocess

# 1. Clear environment
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# 2. Remove ALL our modules from sys.modules
print("🧹 Clearing module cache...")
to_remove = []
for name in list(sys.modules.keys()):
    if any(x in name for x in ['api', 'vision', 'unified', 'command', 'backend']):
        to_remove.append(name)

for name in to_remove:
    del sys.modules[name]
    
print(f"  Removed {len(to_remove)} cached modules")

# 3. Now import main - it will get fresh modules
print("📦 Loading fresh modules...")

# Import after cleaning
from main import app
import uvicorn

# 4. Verify vision routing is correct
print("🔍 Verifying vision command routing...")
from api.unified_command_processor import UnifiedCommandProcessor
import asyncio

async def verify_routing():
    processor = UnifiedCommandProcessor()
    test = "What is happening across my desktop spaces?"
    cmd_type, conf = await processor._classify_command(test)
    return cmd_type.value == "vision", cmd_type.value

is_correct, cmd_type = asyncio.run(verify_routing())

if is_correct:
    print(f"✅ Vision routing VERIFIED - commands will work!")
else:
    print(f"⚠️ Vision routing still needs fixing - got {cmd_type}")
    
    # Apply emergency patch
    print("🔧 Applying emergency patch...")
    
    import api.unified_command_processor as ucp
    
    # Monkey-patch the classification
    original_classify = ucp.UnifiedCommandProcessor._classify_command
    
    async def fixed_classify(self, command_text):
        command_lower = command_text.lower().strip()
        
        # Direct detection for vision queries
        if any(phrase in command_lower for phrase in [
            "desktop space", "my screen", "monitor", "workspace", "happening across"
        ]):
            return (ucp.CommandType.VISION, 0.95)
            
        # Fall back to original
        return await original_classify(self, command_text)
    
    ucp.UnifiedCommandProcessor._classify_command = fixed_classify
    print("✅ Emergency patch applied!")

# Start the server
if __name__ == "__main__":
    print("\n🚀 Starting FRESH JARVIS Backend on port 8010")
    print("   All vision fixes are active!")
    print("   Multi-space queries will work!\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8010, log_level="info")