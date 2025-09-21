#!/usr/bin/env python3
"""Test Vision Command Handler Fix"""

import asyncio
import os
from dotenv import load_dotenv

async def test_fix():
    """Test the vision command handler fix"""
    print("🔧 Testing Vision Command Handler Fix")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # First, set up the vision analyzer
    print("\n1️⃣ Setting up vision analyzer...")
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from api.vision_websocket import set_vision_analyzer
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    set_vision_analyzer(vision)
    print("✅ Vision analyzer set in vision_websocket module")
    
    # Now test the vision command handler
    print("\n2️⃣ Testing vision command handler...")
    from api.vision_command_handler import vision_command_handler
    
    # Test monitoring command
    result = await vision_command_handler.handle_command("start monitoring my screen")
    print(f"Result: {result}")
    
    if result.get('handled'):
        if result.get('error'):
            print(f"❌ Error: {result.get('response')}")
        else:
            print(f"✅ Success: {result.get('response')}")
    else:
        print("❌ Command not handled")
    
    # Clean up - stop monitoring if started
    if result.get('handled') and not result.get('error'):
        print("\n3️⃣ Stopping monitoring...")
        stop_result = await vision_command_handler.handle_command("stop monitoring")
        print(f"Stop result: {stop_result}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_fix())