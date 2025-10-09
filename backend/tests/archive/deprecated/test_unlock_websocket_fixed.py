#!/usr/bin/env python3
"""
Test Unlock Command with Correct WebSocket Format
=================================================

Tests that "unlock my screen" works with the correct message format
"""

import asyncio
import json
from datetime import datetime
import websockets


async def test_unlock_command():
    """Test unlock my screen command via WebSocket with correct format"""
    print("\n🧪 Testing Manual Unlock Command - Fixed Format")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%I:%M %p')}")
    
    try:
        print("\n📡 Connecting to JARVIS WebSocket...")
        
        # Connect to JARVIS
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            print("✅ Connected to JARVIS")
            
            # Wait for welcome message
            welcome = await ws.recv()
            welcome_data = json.loads(welcome)
            print(f"\n🤖 JARVIS: {welcome_data.get('message', 'Connected')}")
            
            # Send the unlock command in the correct format
            command_data = {
                "type": "command",  # Correct type field
                "text": "unlock my screen"  # Text command
            }
            
            print(f"\n🗣️  Sending: 'unlock my screen' (correct format)")
            await ws.send(json.dumps(command_data))
            
            # Collect responses
            responses = []
            print("\n📨 Responses from JARVIS:")
            print("-"*40)
            
            # Wait for responses
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    # Skip debug logs
                    if data.get('type') == 'debug_log':
                        print(f"   [DEBUG] {data.get('message')}")
                        continue
                    
                    if data.get('text'):
                        print(f"   JARVIS: {data['text']}")
                        if data.get('speak'):
                            print(f"           (spoken aloud)")
                    
                    # Check command type
                    if data.get('command_type'):
                        print(f"   [Command type: {data['command_type']}]")
                    
                    # Check if this is the final response
                    if data.get('type') == 'response' and data.get('text'):
                        # Look for final response patterns
                        text_lower = data['text'].lower()
                        if any(phrase in text_lower for phrase in [
                            'successfully', 'unlocked', "couldn't unlock", 
                            "don't have a handler", "failed"
                        ]):
                            break
                        
            except asyncio.TimeoutError:
                # Normal - no more messages
                pass
            
            print("-"*40)
            
            # Analyze results
            print("\n📊 Analysis:")
            
            # Check if we got a proper response
            success = False
            for resp in responses:
                if resp.get('type') != 'response':
                    continue
                    
                text = resp.get('text', '').lower()
                if 'unlock' in text and ('successfully' in text or "i'll unlock" in text):
                    success = True
                    print("   ✅ Unlock command processed successfully!")
                    print(f"   Response: {resp.get('text')}")
                    break
                elif "don't have a handler" in text:
                    print("   ❌ Still getting handler error")
                    print(f"   Response: {resp.get('text')}")
                    break
                elif "couldn't unlock" in text or "failed" in text:
                    print("   ⚠️  Unlock failed (check if daemon is running)")
                    print(f"   Response: {resp.get('text')}")
                    # Check if password is set
                    print("\n   💡 Tips:")
                    print("      1. Run: ./enable_screen_unlock.sh")
                    print("      2. Check daemon: ps aux | grep 8765")
                    print("      3. Restart JARVIS if needed")
                    success = True  # Command was processed correctly
                    break
            
            if not success and responses:
                print("   ❓ Unexpected response:")
                for resp in responses:
                    if resp.get('text') and resp.get('type') == 'response':
                        print(f"      {resp.get('text')}")
                
    except (ConnectionRefusedError, OSError) as e:
        print("\n❌ Error: Cannot connect to JARVIS")
        print("   Make sure JARVIS is running on port 8000")
        print("   Run: cd backend && python start_jarvis_correct_port.sh")
        return
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return
    
    print("\n🎉 Test complete!")


if __name__ == "__main__":
    print("🔧 Testing Manual Unlock Command with Correct Format")
    print("This verifies that 'unlock my screen' is properly handled")
    print("\nNote: Make sure JARVIS is running on port 8000")
    asyncio.run(test_unlock_command())