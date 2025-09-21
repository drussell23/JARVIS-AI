#!/usr/bin/env python3
"""Simple test to find the actual multi-space error"""

import asyncio
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

async def test_direct():
    """Test the vision command handler directly"""
    print("🔍 Testing Vision Command Handler Directly")
    print("=" * 80)
    
    try:
        # Import and test vision command handler
        from api.vision_command_handler import vision_command_handler
        
        # Initialize with a mock API key
        await vision_command_handler.initialize_intelligence("test-key")
        
        command = "can you see the Cursor IDE in the other desktop space?"
        print(f"\n📝 Testing command: '{command}'")
        
        # Check if multi-space is detected
        if vision_command_handler.intelligence:
            should_use = vision_command_handler.intelligence._should_use_multi_space(command)
            print(f"✅ Multi-space detected: {should_use}")
            print(f"   multi_space_enabled: {vision_command_handler.intelligence.multi_space_enabled}")
        
        # Try to handle the command
        result = await vision_command_handler.handle_command(command)
        
        print(f"\n📊 Result:")
        print(f"   Handled: {result.get('handled')}")
        print(f"   Response: {result.get('response', '')[:100]}...")
        print(f"   Error: {result.get('error', 'None')}")
        
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_direct())