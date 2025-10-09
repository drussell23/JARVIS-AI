#!/usr/bin/env python3
"""
Test that messages aren't duplicated
====================================
"""

import asyncio
from unittest.mock import AsyncMock, patch

from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor

class MessageTracker:
    """Track all messages sent"""
    def __init__(self):
        self.messages = []
        
    async def send_json(self, data):
        if data.get('type') == 'response':
            text = data.get('text', '')
            if text:
                self.messages.append({
                    'text': text,
                    'speak': data.get('speak', False),
                    'intermediate': data.get('intermediate', False)
                })
                print(f"\n📨 Message {len(self.messages)}:")
                print(f"   Text: '{text}'")
                print(f"   Speak: {data.get('speak')}")
                print(f"   Intermediate: {data.get('intermediate')}")

async def test_no_duplicates():
    """Test that lock message isn't duplicated"""
    print("🧪 Testing for duplicate messages")
    print("="*50)
    
    # Mock locked screen
    with patch('api.direct_unlock_handler_fixed.check_screen_locked_direct', 
               new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True
        
        with patch('api.direct_unlock_handler_fixed.unlock_screen_direct',
                   new_callable=AsyncMock) as mock_unlock:
            mock_unlock.return_value = True
            
            # Create components
            processor = UnifiedCommandProcessor()
            handler = wrap_with_enhanced_context(processor)
            tracker = MessageTracker()
            
            # Process command
            command = "open Safari and search for dogs"
            print(f"\n📋 Command: '{command}'")
            print("\n🔍 Processing with locked screen...")
            
            result = await handler.process_with_context(command, tracker)
            
            # Analyze messages
            print("\n" + "="*50)
            print("📊 ANALYSIS")
            print("="*50)
            
            print(f"\nTotal messages sent: {len(tracker.messages)}")
            
            # Check for duplicates
            lock_messages = []
            for msg in tracker.messages:
                if "screen is locked" in msg['text'].lower():
                    lock_messages.append(msg['text'])
            
            print(f"\nLock detection messages: {len(lock_messages)}")
            if lock_messages:
                for i, msg in enumerate(lock_messages, 1):
                    print(f"  {i}. '{msg}'")
            
            # Check final response
            print(f"\nFinal response in result: '{result.get('response', '')}'")
            
            # Verify no duplication
            if len(lock_messages) > 1:
                print("\n❌ DUPLICATE DETECTED! Lock message sent multiple times")
            elif len(lock_messages) == 1:
                print("\n✅ SUCCESS! Lock message sent only once")
                
                # Check if it's in the final response
                final_response = result.get('response', '')
                if "screen is locked" in final_response.lower():
                    print("   ⚠️  But it's also in the final response text")
                else:
                    print("   ✅ And it's not duplicated in the final response")
            else:
                print("\n❌ No lock message sent at all")

if __name__ == "__main__":
    asyncio.run(test_no_duplicates())