#!/usr/bin/env python3
"""
Standalone Voice Demo - Works Without Full JARVIS
=================================================

Shows the voice feedback messages that would be spoken
"""

import asyncio
import subprocess
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor

class MockVoiceWebSocket:
    """Mock websocket that simulates voice feedback"""
    def __init__(self):
        self.messages = []
        self.voice_messages = []
        
    async def send_json(self, data):
        self.messages.append(data)
        
        if data.get('type') == 'response' and data.get('speak'):
            text = data.get('text', '')
            if text:
                self.voice_messages.append(text)
                print(f"\n🔊 JARVIS would say: \"{text}\"")
                
                # Simulate speaking time
                speak_time = len(text) * 0.05  # ~0.05 seconds per character
                print(f"   [Speaking for {speak_time:.1f} seconds...]")
                await asyncio.sleep(min(speak_time, 3.0))

async def demo_voice_flow():
    """Demonstrate the voice flow"""
    
    print("\n" + "="*70)
    print("🎭 VOICE FEEDBACK DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows what JARVIS would say when your screen is locked")
    
    # Simulate locked screen scenario
    print("\n📍 Scenario Setup:")
    print("   • Your Mac screen is LOCKED 🔒")
    print("   • You say: \"JARVIS, open Safari and search for dogs\"")
    print("   • Watch what JARVIS says...")
    
    print("\n" + "-"*50)
    
    # Create components
    processor = UnifiedCommandProcessor()
    handler = wrap_with_enhanced_context(processor)
    websocket = MockVoiceWebSocket()
    
    # Mock the screen as locked
    from unittest.mock import patch, AsyncMock
    
    with patch('api.direct_unlock_handler_fixed.check_screen_locked_direct', 
               new_callable=AsyncMock) as mock_check:
        mock_check.return_value = True  # Screen is locked
        
        with patch('api.direct_unlock_handler_fixed.unlock_screen_direct',
                   new_callable=AsyncMock) as mock_unlock:
            mock_unlock.return_value = True  # Unlock succeeds
            
            # Process command
            command = "open Safari and search for dogs"
            print(f"\n🎤 Processing: '{command}'")
            print("\n" + "-"*50)
            
            result = await handler.process_with_context(command, websocket)
    
    # Summary
    print("\n" + "="*50)
    print("📊 VOICE FEEDBACK SUMMARY")
    print("="*50)
    
    if websocket.voice_messages:
        print(f"\n✅ JARVIS would speak {len(websocket.voice_messages)} message(s):")
        for i, msg in enumerate(websocket.voice_messages, 1):
            print(f"\n{i}. \"{msg}\"")
        
        # Check for key phrases
        print("\n🔍 Key feedback elements:")
        
        has_lock_detection = any("screen is locked" in msg.lower() 
                               for msg in websocket.voice_messages)
        print(f"   • Lock detection announced: {'✅ Yes' if has_lock_detection else '❌ No'}")
        
        has_unlock_intent = any("unlock" in msg.lower() and "now" in msg.lower()
                               for msg in websocket.voice_messages)
        print(f"   • Unlock intent stated: {'✅ Yes' if has_unlock_intent else '❌ No'}")
        
        has_action_description = any("search for dogs" in msg.lower()
                                   for msg in websocket.voice_messages)
        print(f"   • Action described: {'✅ Yes' if has_action_description else '❌ No'}")
    else:
        print("\n❌ No voice messages would be spoken")
    
    # Show the ideal flow
    print("\n" + "-"*50)
    print("\n💡 IDEAL VOICE FLOW:")
    print("1. Before unlock: \"I see your screen is locked. I'll unlock it now by")
    print("                  typing in your password so I can search for dogs.\"")
    print("2. After unlock:  \"Screen unlocked. Now executing your command...\"")
    print("3. Final result:  \"I unlocked your screen and searched for dogs in Safari.\"")

async def test_various_commands():
    """Test voice feedback for different commands"""
    print("\n\n" + "="*70)
    print("🧪 TESTING VARIOUS COMMANDS")
    print("="*70)
    
    test_commands = [
        "open Chrome",
        "create a new document",
        "show me the weather",
        "take a screenshot"
    ]
    
    handler = wrap_with_enhanced_context(None)
    
    print("\n🔍 How different commands would be announced:")
    
    for cmd in test_commands:
        # Extract action using the handler's method
        action = handler._extract_action_description(cmd)
        message = f"I see your screen is locked. I'll unlock it now by typing in your password so I can {action}."
        
        print(f"\n📌 Command: \"{cmd}\"")
        print(f"🔊 JARVIS: \"{message}\"")

if __name__ == "__main__":
    print("🚀 JARVIS Voice Feedback Demo (Standalone)")
    print("\nThis demo works without JARVIS running")
    
    # Run demos
    asyncio.run(demo_voice_flow())
    asyncio.run(test_various_commands())
    
    print("\n\n✅ Demo complete!")
    print("\n📝 To hear actual voice in production:")
    print("   1. Make sure JARVIS is running (python main.py)")
    print("   2. Lock your screen")
    print("   3. Say a command that needs the screen")
    print("   4. Listen for JARVIS to speak the feedback")