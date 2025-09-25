#!/usr/bin/env python3
"""
Test Voice Feedback for Lock/Unlock
===================================

Tests that JARVIS speaks the lock detection feedback
"""

import asyncio
import json
import websockets
import subprocess
import time

async def test_voice_feedback():
    """Test that JARVIS speaks the feedback messages"""
    
    print("\n" + "="*60)
    print("🎤 Testing JARVIS Voice Feedback")
    print("="*60)
    
    # Connect to JARVIS WebSocket
    uri = "ws://localhost:8888/ws/jarvis"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to JARVIS WebSocket")
            
            # Wait for connection message
            msg = await websocket.recv()
            data = json.loads(msg)
            print(f"📨 Received: {data.get('type')}")
            
            # Lock the screen first
            print("\n🔒 Locking screen in 3 seconds...")
            for i in range(3, 0, -1):
                print(f"   {i}...")
                time.sleep(1)
                
            lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
            subprocess.run(lock_cmd, shell=True)
            
            print("   Waiting for lock...")
            await asyncio.sleep(3)
            
            # Send command
            command = "open Safari and search for dogs"
            print(f"\n🎤 Sending command: '{command}'")
            
            await websocket.send(json.dumps({
                "type": "command",
                "command": command
            }))
            
            # Listen for responses
            print("\n📡 Listening for responses...")
            response_count = 0
            spoken_messages = []
            
            while response_count < 5:  # Listen for up to 5 messages
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(msg)
                    
                    msg_type = data.get('type')
                    print(f"\n📨 Message {response_count + 1}:")
                    print(f"   Type: {msg_type}")
                    
                    if msg_type == 'response':
                        text = data.get('text', data.get('message', ''))
                        speak = data.get('speak', False)
                        intermediate = data.get('intermediate', False)
                        
                        print(f"   Text: {text}")
                        print(f"   Speak: {speak}")
                        print(f"   Intermediate: {intermediate}")
                        
                        if speak and text:
                            spoken_messages.append(text)
                            print(f"   🔊 SHOULD BE SPOKEN: '{text}'")
                    
                    response_count += 1
                    
                except asyncio.TimeoutError:
                    print("\n⏱️  Timeout waiting for more messages")
                    break
            
            # Verify spoken messages
            print("\n" + "="*60)
            print("📊 Summary of Spoken Messages:")
            print("="*60)
            
            if spoken_messages:
                for i, msg in enumerate(spoken_messages, 1):
                    print(f"{i}. '{msg}'")
                    
                # Check for expected feedback
                lock_feedback_found = any("screen is locked" in msg.lower() for msg in spoken_messages)
                unlock_feedback_found = any("unlock" in msg.lower() for msg in spoken_messages)
                
                print(f"\n✅ Lock detection feedback: {'Found' if lock_feedback_found else 'NOT FOUND'}")
                print(f"✅ Unlock intent feedback: {'Found' if unlock_feedback_found else 'NOT FOUND'}")
            else:
                print("❌ No spoken messages found!")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure JARVIS is running (python main.py)")

if __name__ == "__main__":
    print("🚀 JARVIS Voice Feedback Test")
    print("This tests if JARVIS speaks the lock detection messages")
    
    asyncio.run(test_voice_feedback())