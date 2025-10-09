#!/usr/bin/env python3
"""
Test Direct Unlock Via API
==========================

Tests the unlock command directly through the API without context handling
"""

import asyncio
import sys
sys.path.insert(0, '.')

from api.unified_command_processor import get_unified_processor
from api.jarvis_voice_api import JARVISCommand

async def test_direct_unlock():
    """Test unlock command directly"""
    print("\n🔧 Testing Direct Unlock Command")
    print("="*60)
    
    try:
        # Get unified processor
        processor = get_unified_processor(None)  # No API key needed for unlock
        
        # Create command
        command = JARVISCommand(text="unlock my screen")
        
        print(f"\n📝 Processing command: '{command.text}'")
        
        # Process command
        result = await processor.process_command(command.text, websocket=None)
        
        print(f"\n📊 Result:")
        print(f"   Success: {result.get('success')}")
        print(f"   Response: {result.get('response')}")
        print(f"   Command Type: {result.get('command_type')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔐 Testing Direct Unlock Command Processing")
    asyncio.run(test_direct_unlock())