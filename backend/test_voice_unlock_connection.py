#!/usr/bin/env python3
"""
Test Voice Unlock Connection
============================

Debug why Voice Unlock isn't connecting.
"""

import asyncio
import logging
import websockets

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_direct_websocket():
    """Test direct WebSocket connection"""
    print("\n1️⃣ Testing Direct WebSocket Connection...")
    print("-"*50)
    
    try:
        uri = "ws://localhost:8765/voice-unlock"
        print(f"Connecting to: {uri}")
        
        async with websockets.connect(uri) as ws:
            print("✅ Connected!")
            
            # Send handshake
            import json
            handshake = json.dumps({
                "type": "command",
                "command": "handshake",
                "parameters": {"client": "test"}
            })
            
            await ws.send(handshake)
            response = await ws.recv()
            print(f"Response: {response}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")


async def test_voice_unlock_connector():
    """Test Voice Unlock connector"""
    print("\n2️⃣ Testing Voice Unlock Connector...")
    print("-"*50)
    
    try:
        from api.voice_unlock_integration import VoiceUnlockDaemonConnector
        
        connector = VoiceUnlockDaemonConnector()
        print("Created connector")
        
        await connector.connect()
        print(f"Connected: {connector.connected}")
        
        if connector.connected:
            status = await connector.get_status()
            print(f"Status: {status}")
        else:
            print("❌ Failed to connect")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_voice_unlock_integration():
    """Test the integration function"""
    print("\n3️⃣ Testing Voice Unlock Integration...")
    print("-"*50)
    
    try:
        from api.voice_unlock_integration import initialize_voice_unlock, voice_unlock_connector
        
        print(f"Current connector: {voice_unlock_connector}")
        
        success = await initialize_voice_unlock()
        print(f"Initialize result: {success}")
        
        # Check global connector
        from api import voice_unlock_integration as vui
        print(f"Global connector after init: {vui.voice_unlock_connector}")
        print(f"Connected: {vui.voice_unlock_connector.connected if vui.voice_unlock_connector else 'None'}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def test_lock_command():
    """Test the actual lock command"""
    print("\n4️⃣ Testing Lock Command...")
    print("-"*50)
    
    try:
        from api.voice_unlock_integration import handle_voice_unlock_in_jarvis, voice_unlock_connector
        
        print(f"Connector before: {voice_unlock_connector}")
        
        result = await handle_voice_unlock_in_jarvis("lock my screen")
        
        print(f"\nResult:")
        print(f"Success: {result.get('success')}")
        print(f"Response: {result.get('response')}")
        print(f"Method: {result.get('method', 'not specified')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    print("🔍 DEBUGGING VOICE UNLOCK CONNECTION")
    print("="*50)
    
    await test_direct_websocket()
    await test_voice_unlock_connector()
    await test_voice_unlock_integration()
    await test_lock_command()
    
    print("\n" + "="*50)
    print("DIAGNOSIS COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())