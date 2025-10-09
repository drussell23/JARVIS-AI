#!/usr/bin/env python3
"""
Test Complete Voice Flow with JARVIS Running
============================================

Tests the voice feedback when screen is locked
"""

import asyncio
import json
import websockets
import subprocess
import time
import aiohttp

async def test_jarvis_connection():
    """Test that JARVIS is running and accessible"""
    print("1️⃣ Testing JARVIS connection...")
    
    try:
        async with aiohttp.ClientSession() as session:
            # Check status
            async with session.get('http://localhost:8000/voice/jarvis/status') as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"   ✅ JARVIS status: {data.get('status', 'Unknown')}")
                    return True
                else:
                    print(f"   ❌ JARVIS returned status: {response.status}")
                    return False
    except Exception as e:
        print(f"   ❌ Could not connect to JARVIS: {e}")
        return False

async def test_websocket_connection():
    """Test WebSocket connection to JARVIS"""
    print("\n2️⃣ Testing WebSocket connection...")
    
    try:
        uri = "ws://localhost:8000/ws/jarvis"
        async with websockets.connect(uri) as websocket:
            # Wait for connection message
            msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
            data = json.loads(msg)
            print(f"   ✅ Connected: {data.get('type', 'Unknown')} message received")
            return websocket
    except Exception as e:
        print(f"   ❌ WebSocket error: {e}")
        return None

async def test_voice_feedback():
    """Test the complete voice feedback flow"""
    print("\n3️⃣ Testing voice feedback flow...")
    
    # Connect to JARVIS WebSocket
    uri = "ws://localhost:8000/ws/jarvis"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("   ✅ Connected to JARVIS WebSocket")
            
            # Receive connection message
            connect_msg = await websocket.recv()
            print(f"   📨 Connection: {json.loads(connect_msg).get('type')}")
            
            # Lock the screen
            print("\n4️⃣ Locking screen...")
            lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
            subprocess.run(lock_cmd, shell=True)
            
            print("   ⏳ Waiting for lock to complete...")
            await asyncio.sleep(3)
            
            # Send command
            command = "open Safari and search for dogs"
            print(f"\n5️⃣ Sending command: '{command}'")
            
            await websocket.send(json.dumps({
                "type": "command",
                "command": command
            }))
            
            # Listen for responses
            print("\n6️⃣ Listening for JARVIS responses...")
            
            responses = []
            voice_messages = []
            start_time = time.time()
            
            while time.time() - start_time < 30:  # Listen for up to 30 seconds
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(msg)
                    
                    msg_type = data.get('type')
                    text = data.get('text', data.get('message', ''))
                    
                    if msg_type == 'response':
                        responses.append(data)
                        
                        if data.get('speak') and text:
                            voice_messages.append(text)
                            print(f"\n   🔊 JARVIS says: '{text}'")
                            
                            # Check if it's the lock detection message
                            if "screen is locked" in text.lower() and "unlock" in text.lower():
                                print("   ✅ Lock detection feedback spoken!")
                        
                        # Check for intermediate responses
                        if data.get('intermediate'):
                            print(f"   📍 Intermediate: {text}")
                    
                    elif msg_type == 'debug_log':
                        print(f"   🐛 Debug: {text}")
                    
                    elif msg_type == 'processing':
                        print(f"   ⏳ Processing: {data.get('message', 'Working...')}")
                        
                except asyncio.TimeoutError:
                    # Check if we're done
                    if voice_messages:
                        break
                    continue
            
            # Summary
            print("\n" + "="*60)
            print("📊 RESULTS SUMMARY")
            print("="*60)
            
            print(f"\n✅ Total responses: {len(responses)}")
            print(f"🔊 Voice messages: {len(voice_messages)}")
            
            if voice_messages:
                print("\n📝 Voice messages spoken:")
                for i, msg in enumerate(voice_messages, 1):
                    print(f"   {i}. '{msg}'")
                
                # Check for expected flow
                lock_detected = any("screen is locked" in m.lower() for m in voice_messages)
                unlock_intent = any("unlock" in m.lower() and "now" in m.lower() for m in voice_messages)
                action_mentioned = any("search for dogs" in m.lower() for m in voice_messages)
                
                print("\n✅ Flow verification:")
                print(f"   • Lock detected: {'✅' if lock_detected else '❌'}")
                print(f"   • Unlock intent: {'✅' if unlock_intent else '❌'}")
                print(f"   • Action mentioned: {'✅' if action_mentioned else '❌'}")
                
                if lock_detected and unlock_intent:
                    print("\n🎉 SUCCESS! Voice feedback is working correctly!")
                else:
                    print("\n⚠️  Voice feedback may not be complete")
            else:
                print("\n❌ No voice messages were spoken")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_tts_directly():
    """Test TTS endpoint directly"""
    print("\n7️⃣ Testing Text-to-Speech directly...")
    
    test_message = "Testing JARVIS voice feedback system"
    
    try:
        async with aiohttp.ClientSession() as session:
            url = "http://localhost:8000/api/jarvis/speak"
            payload = {"text": test_message}
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    print(f"   ✅ TTS working - should hear: '{test_message}'")
                    await asyncio.sleep(2)  # Let it speak
                else:
                    print(f"   ❌ TTS error: {response.status}")
                    
    except Exception as e:
        print(f"   ❌ TTS test failed: {e}")

async def main():
    """Run all tests"""
    print("🚀 JARVIS Voice Feedback Test Suite")
    print("="*60)
    
    # Test connections
    jarvis_ok = await test_jarvis_connection()
    
    if not jarvis_ok:
        print("\n❌ JARVIS is not running!")
        print("   Start it with: python main.py")
        return
    
    # Test WebSocket
    ws = await test_websocket_connection()
    if ws:
        await ws.close()
    
    # Test TTS
    await test_tts_directly()
    
    # Main test
    print("\n" + "="*60)
    print("🎯 MAIN TEST: Lock Screen Voice Feedback")
    print("="*60)
    print("\nThis will:")
    print("  1. Lock your screen")
    print("  2. Send 'open Safari and search for dogs'")
    print("  3. Listen for JARVIS to speak the feedback")
    
    print("\n⏳ Starting in 3 seconds...")
    for i in range(3, 0, -1):
        print(f"   {i}...")
        await asyncio.sleep(1)
    
    await test_voice_feedback()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())