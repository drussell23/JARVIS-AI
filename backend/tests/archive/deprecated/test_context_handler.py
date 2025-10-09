#!/usr/bin/env python3
"""
Test Context Handler
====================

Tests the enhanced context handler directly
"""

import asyncio
import sys
sys.path.insert(0, '.')

from api.unified_command_processor import get_unified_processor
from api.simple_context_handler_enhanced import wrap_with_enhanced_context

async def test_context_handler():
    """Test context handler directly"""
    print("\n🔧 Testing Context Handler")
    print("="*60)
    
    try:
        # Get unified processor
        processor = get_unified_processor(None)
        
        # Wrap with context handler
        context_handler = wrap_with_enhanced_context(processor)
        
        print("\n✅ Context handler created successfully")
        
        # Test unlock command
        command = "unlock my screen"
        print(f"\n📝 Processing command: '{command}'")
        
        # Process through context handler
        result = await context_handler.process_command(command, websocket=None)
        
        print(f"\n📊 Result:")
        print(f"   Success: {result.get('success')}")
        print(f"   Response: {result.get('response')}")
        print(f"   Context Handled: {result.get('context_handled')}")
        print(f"   Screen Unlocked: {result.get('screen_unlocked')}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🧠 Testing Enhanced Context Handler")
    asyncio.run(test_context_handler())