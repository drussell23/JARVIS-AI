#!/usr/bin/env python3
"""
Debug Enhanced Context Handler
==============================
"""

import asyncio
import logging
from unittest.mock import AsyncMock, patch

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test imports
from api.simple_context_handler_enhanced import EnhancedSimpleContextHandler
from api.unified_command_processor import UnifiedCommandProcessor

async def debug_screen_detection():
    """Debug why screen requirement detection might not be working"""
    
    # Create handler directly
    processor = UnifiedCommandProcessor()
    handler = EnhancedSimpleContextHandler(processor)
    
    # Test commands
    test_commands = [
        "open Safari and search for dogs",
        "open Chrome", 
        "search for Python tutorials",
        "lock screen",
        "what time is it"
    ]
    
    print("\n" + "="*60)
    print("Debug: Screen Requirement Detection")
    print("="*60)
    
    # Check patterns
    print("\n📋 Screen Required Patterns:")
    for pattern in handler.screen_required_patterns[:10]:  # Show first 10
        print(f"   - '{pattern}'")
    print(f"   ... and {len(handler.screen_required_patterns) - 10} more")
    
    print("\n🔍 Testing Commands:")
    for cmd in test_commands:
        requires = handler._requires_screen(cmd)
        print(f"\n   Command: '{cmd}'")
        print(f"   Requires Screen: {requires}")
        
        # Check which pattern matched
        cmd_lower = cmd.lower()
        matched = []
        for pattern in handler.screen_required_patterns:
            if pattern in cmd_lower:
                matched.append(pattern)
        
        if matched:
            print(f"   Matched patterns: {matched}")

async def test_full_flow_debug():
    """Test full flow with debug output"""
    print("\n" + "="*60)
    print("Debug: Full Context Flow")
    print("="*60)
    
    # Mock screen locked
    with patch('api.direct_unlock_handler_fixed.check_screen_locked_direct', new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True
        
        with patch('api.direct_unlock_handler_fixed.unlock_screen_direct', new_callable=AsyncMock) as mock_unlock:
            mock_unlock.return_value = True
            
            # Create handler
            processor = UnifiedCommandProcessor()
            handler = EnhancedSimpleContextHandler(processor)
            
            # Mock websocket
            class DebugWebSocket:
                async def send_json(self, data):
                    print(f"\n🔔 WebSocket Message:")
                    print(f"   Type: {data.get('type')}")
                    if data.get('message'):
                        print(f"   Message: {data.get('message')}")
            
            ws = DebugWebSocket()
            
            # Test command
            command = "open Safari and search for dogs"
            print(f"\n📋 Testing command: '{command}'")
            
            # Check if requires screen
            requires = handler._requires_screen(command)
            print(f"📊 Requires screen: {requires}")
            
            if requires:
                # Extract action
                action = handler._extract_action_description(command)
                print(f"📝 Extracted action: '{action}'")
                
                # Expected message
                expected = f"I see your screen is locked. I'll unlock it now by typing in your password so I can {action}."
                print(f"💬 Expected message: '{expected}'")
            
            # Process command
            print("\n🚀 Processing command...")
            try:
                result = await handler.process_with_context(command, ws)
                print(f"\n✅ Success: {result.get('success')}")
                print(f"📄 Response: {result.get('response')}")
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_screen_detection())
    asyncio.run(test_full_flow_debug())