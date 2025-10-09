#!/usr/bin/env python3
"""
Test Enhanced Context Feedback Messages
======================================

Verifies the enhanced context handler provides correct feedback
"""

import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch

# Test imports
from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor

class MessageCapture:
    """Captures all messages sent via websocket"""
    def __init__(self):
        self.messages = []
        
    async def send_json(self, data):
        self.messages.append(data)

async def test_locked_screen_feedback():
    """Test feedback when screen is locked"""
    print("\n" + "="*60)
    print("Test: Locked Screen Feedback")
    print("="*60)
    
    # Mock the screen lock check to return True
    with patch('api.direct_unlock_handler_fixed.check_screen_locked_direct', new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True
        
        # Mock the unlock to succeed
        with patch('api.direct_unlock_handler_fixed.unlock_screen_direct', new_callable=AsyncMock) as mock_unlock:
            mock_unlock.return_value = True
            
            # Create components
            processor = UnifiedCommandProcessor()
            handler = wrap_with_enhanced_context(processor)
            websocket = MessageCapture()
            
            # Test command
            command = "open Safari and search for dogs"
            print(f"\n📋 Command: '{command}'")
            print("📊 Screen Status: LOCKED (mocked)")
            
            # Process
            result = await handler.process_with_context(command, websocket)
            
            print("\n📡 WebSocket Messages:")
            for i, msg in enumerate(websocket.messages, 1):
                print(f"\n{i}. Type: {msg.get('type')}")
                if msg.get('message'):
                    print(f"   Message: {msg.get('message')}")
                if msg.get('status'):
                    print(f"   Status: {msg.get('status')}")
            
            # Verify the feedback
            print("\n✅ Verification:")
            
            # Check for context update with unlock message
            context_updates = [m for m in websocket.messages if m.get('type') == 'context_update']
            if context_updates:
                unlock_msg = context_updates[0].get('message', '')
                expected_parts = ["I see your screen is locked", "I'll unlock it now", "search for dogs"]
                
                all_found = all(part in unlock_msg for part in expected_parts)
                if all_found:
                    print(f"   ✅ Correct feedback: '{unlock_msg}'")
                else:
                    print(f"   ❌ Incorrect feedback: '{unlock_msg}'")
                    for part in expected_parts:
                        if part not in unlock_msg:
                            print(f"      Missing: '{part}'")
            else:
                print("   ❌ No context update messages sent")
            
            # Check execution steps
            if result.get('execution_steps'):
                print("\n📊 Execution Steps:")
                for step in result['execution_steps']:
                    print(f"   - {step['step']}")

async def test_unlocked_screen_feedback():
    """Test feedback when screen is already unlocked"""
    print("\n" + "="*60)
    print("Test: Unlocked Screen (No Special Handling)")
    print("="*60)
    
    # Mock screen as unlocked
    with patch('api.direct_unlock_handler_fixed.check_screen_locked_direct', new_callable=AsyncMock) as mock_check:
        mock_check.return_value = False
        
        processor = UnifiedCommandProcessor()
        handler = wrap_with_enhanced_context(processor)
        websocket = MessageCapture()
        
        command = "open Safari and search for cats"
        print(f"\n📋 Command: '{command}'")
        print("📊 Screen Status: UNLOCKED (mocked)")
        
        result = await handler.process_with_context(command, websocket)
        
        print("\n✅ Verification:")
        
        # Should have no context updates about unlocking
        context_updates = [m for m in websocket.messages if m.get('type') == 'context_update']
        if not context_updates:
            print("   ✅ No unlock messages sent (correct)")
        else:
            print("   ❌ Unexpected context updates sent:")
            for msg in context_updates:
                print(f"      - {msg.get('message')}")

async def test_various_commands():
    """Test feedback for various command types"""
    print("\n" + "="*60)
    print("Test: Various Command Patterns")
    print("="*60)
    
    test_cases = [
        ("open Safari and search for Python tutorials", "search for Python tutorials"),
        ("open Chrome", "open Chrome"),
        ("show me the weather", "show you the weather"),
        ("create a new document", "create a new document"),
        ("find my recent files", "find my recent files")
    ]
    
    with patch('api.direct_unlock_handler_fixed.check_screen_locked_direct', new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True
        
        with patch('api.direct_unlock_handler_fixed.unlock_screen_direct', new_callable=AsyncMock) as mock_unlock:
            mock_unlock.return_value = True
            
            processor = UnifiedCommandProcessor()
            handler = wrap_with_enhanced_context(processor)
            
            for command, expected_action in test_cases:
                print(f"\n🔹 Testing: '{command}'")
                
                websocket = MessageCapture()
                await handler.process_with_context(command, websocket)
                
                context_msgs = [m for m in websocket.messages if m.get('type') == 'context_update']
                if context_msgs:
                    msg = context_msgs[0].get('message', '')
                    if expected_action.lower() in msg.lower():
                        print(f"   ✅ Contains expected action: '{expected_action}'")
                    else:
                        print(f"   ❌ Missing action. Got: '{msg}'")

if __name__ == "__main__":
    print("🚀 Testing Enhanced Context Feedback Messages")
    
    # Run tests
    asyncio.run(test_locked_screen_feedback())
    asyncio.run(test_unlocked_screen_feedback())
    asyncio.run(test_various_commands())
    
    print("\n✅ All tests completed!")