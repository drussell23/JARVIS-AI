#!/usr/bin/env python3
"""
Test JARVIS monitoring command through the voice API
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.jarvis_voice_api import JARVISVoiceAPI, JARVISCommand

async def test_jarvis_monitoring():
    """Test that monitoring commands work through JARVIS voice API"""
    print("\n🧪 TESTING JARVIS MONITORING COMMAND")
    print("=" * 60)
    
    # Initialize JARVIS Voice API
    jarvis = JARVISVoiceAPI()
    
    # Test commands
    test_commands = [
        "start monitoring my screen",
        "stop monitoring"
    ]
    
    for cmd_text in test_commands:
        print(f"\n📝 Testing: '{cmd_text}'")
        
        # Create command
        command = JARVISCommand(
            text=cmd_text,
            source="test",
            user_id="test_user"
        )
        
        # Process command
        try:
            result = await jarvis.process_command(command)
            
            print(f"\n📊 Result:")
            print(f"   Status: {result.get('status', 'unknown')}")
            print(f"   Command Type: {result.get('command_type', 'unknown')}")
            print(f"   Response: {result.get('response', 'No response')}")
            print(f"   Monitoring Active: {result.get('monitoring_active', 'unknown')}")
            
            # Check response length
            response_len = len(result.get('response', ''))
            print(f"\n   Response Length: {response_len} characters")
            
            if response_len < 200:
                print("   ✅ Response is concise!")
            else:
                print("   ⚠️ Response is too long")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        # Wait between commands
        await asyncio.sleep(2)
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    # Set up environment
    os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'test-key')
    
    asyncio.run(test_jarvis_monitoring())