#!/usr/bin/env python3
"""
Simple test for weather command
"""

import asyncio
import os
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

async def test_weather():
    """Test weather command"""
    
    # Set API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY")
        return
    
    # Import after path setup
    from voice.intelligent_command_handler import IntelligentCommandHandler
    
    handler = IntelligentCommandHandler()
    
    print("Testing weather command...")
    
    try:
        # Set a timeout for the test
        response, handler_type = await asyncio.wait_for(
            handler.handle_command("what's the weather today"),
            timeout=30.0
        )
        print(f"✅ Success!")
        print(f"Handler: {handler_type}")
        print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")
    except asyncio.TimeoutError:
        print("❌ Command timed out after 30 seconds")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_weather())