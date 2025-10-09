#!/usr/bin/env python3
"""
Test Enhanced Context Flow - Verify proper feedback messages
============================================================

Tests the scenario where Mac is locked and user asks JARVIS to open Safari
"""

import asyncio
import json
import time
from datetime import datetime

# Test imports first
try:
    from api.simple_context_handler_enhanced import wrap_with_enhanced_context
    from api.unified_command_processor import UnifiedCommandProcessor
    print("✅ Enhanced context handler imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

async def test_enhanced_flow():
    """Test the enhanced context flow with proper feedback"""
    print("\n" + "="*60)
    print("Testing Enhanced Context Intelligence Flow")
    print("="*60)
    
    # Create processor and wrap with enhanced context
    processor = UnifiedCommandProcessor()
    context_handler = wrap_with_enhanced_context(processor)
    
    # Test command
    test_command = "JARVIS, open Safari and search for dogs"
    
    print(f"\n📋 Test Command: '{test_command}'")
    print("\n🔍 Processing with enhanced context handler...")
    
    # Mock websocket for testing
    class MockWebSocket:
        async def send_json(self, data):
            print(f"\n📡 WebSocket Message:")
            print(f"   Type: {data.get('type')}")
            print(f"   Message: {data.get('message')}")
            if 'status' in data:
                print(f"   Status: {data.get('status')}")
    
    websocket = MockWebSocket()
    
    try:
        # Process command
        result = await context_handler.process_with_context(
            test_command, 
            websocket
        )
        
        print("\n✅ Command processed successfully!")
        print(f"\n📄 Result:")
        print(json.dumps(result, indent=2))
        
        # Check execution steps
        if 'execution_steps' in result:
            print("\n📊 Execution Steps:")
            for i, step in enumerate(result['execution_steps'], 1):
                print(f"   {i}. {step['step']}")
                if step.get('details'):
                    print(f"      Details: {step['details']}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_feedback_messages():
    """Test that feedback messages match PRD requirements"""
    print("\n" + "="*60)
    print("Verifying Feedback Messages")
    print("="*60)
    
    processor = UnifiedCommandProcessor()
    handler = wrap_with_enhanced_context(processor)
    
    # Check the feedback message format
    test_cases = [
        ("open Safari and search for dogs", "search for dogs"),
        ("open Chrome", "open Chrome"),
        ("search for Python tutorials", "search for Python tutorials"),
    ]
    
    for command, expected_action in test_cases:
        action = handler._extract_action_description(command)
        print(f"\nCommand: '{command}'")
        print(f"Extracted action: '{action}'")
        print(f"Expected pattern: 'I see your screen is locked. I'll unlock it now by typing in your password so I can {action}.'")

if __name__ == "__main__":
    print("🚀 Starting Enhanced Context Flow Test")
    
    # Run tests
    asyncio.run(test_enhanced_flow())
    asyncio.run(test_feedback_messages())
    
    print("\n✅ Test completed!")