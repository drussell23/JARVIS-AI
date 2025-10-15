#!/usr/bin/env python3
"""
Quick test of voice demo with all services running
"""

import asyncio
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor

class TestWebSocket:
    """Test websocket that prints messages"""
    async def send_json(self, data):
        msg_type = data.get('type')
        if msg_type == 'response':
            text = data.get('text', data.get('message'))
            speak = data.get('speak', False)
            print(f"\n📨 Response:")
            print(f"   Text: {text}")
            print(f"   Speak: {speak}")
            
            if speak and text:
                print(f"   🔊 JARVIS should say: '{text}'")

async def quick_test():
    """Quick test without locking screen"""
    print("🧪 Quick Voice Demo Test")
    print("="*50)
    
    processor = UnifiedCommandProcessor()
    handler = wrap_with_enhanced_context(processor)
    websocket = TestWebSocket()
    
    # Test a command that requires screen
    command = "open Safari"
    print(f"\n📋 Test command: '{command}'")
    print("   (Screen is currently UNLOCKED)")
    
    result = await handler.process_with_context(command, websocket)
    
    print(f"\n✅ Result: {result.get('response', 'No response')[:100]}...")

if __name__ == "__main__":
    asyncio.run(quick_test())