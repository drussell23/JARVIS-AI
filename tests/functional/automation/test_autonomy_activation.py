#!/usr/bin/env python3
"""
Test script for JARVIS Autonomy Activation
Tests the full autonomy activation flow
"""

import asyncio
import websockets
import json
import time


async def test_autonomy_activation():
    """Test autonomy activation via WebSocket"""
    uri = "ws://localhost:8000/voice/jarvis/stream"
    
    print("🤖 JARVIS Autonomy Activation Test")
    print("=" * 50)
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection confirmation
            message = await websocket.recv()
            data = json.loads(message)
            print(f"✅ Connected: {data.get('message', 'Connected to JARVIS')}")
            
            # Test 1: Send autonomy activation command
            print("\n📡 Test 1: Sending 'activate full autonomy' command...")
            await websocket.send(json.dumps({
                "type": "command",
                "text": "activate full autonomy"
            }))
            
            # Receive responses
            response_count = 0
            while response_count < 3:  # Expect multiple responses
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    print(f"\n📨 Response type: {data['type']}")
                    
                    if data['type'] == 'processing':
                        print("  ⏳ Processing command...")
                    elif data['type'] == 'response':
                        print(f"  💬 JARVIS: {data['text']}")
                        if 'command_type' in data:
                            print(f"  🎯 Command type: {data['command_type']}")
                        if 'autonomy_result' in data:
                            result = data['autonomy_result']
                            print(f"  📊 Autonomy activation: {'✅ SUCCESS' if result.get('success') else '❌ FAILED'}")
                            if result.get('activation_steps'):
                                print("  📋 Activation steps:")
                                for step in result['activation_steps']:
                                    print(f"     ✓ {step}")
                    elif data['type'] == 'autonomy_status':
                        print(f"  🔄 Autonomy status: {'ENABLED' if data.get('enabled') else 'DISABLED'}")
                        if 'systems' in data:
                            print("  🖥️  System status:")
                            for system, status in data['systems'].items():
                                print(f"     • {system}: {'✅' if status else '❌'}")
                    elif data['type'] == 'error':
                        print(f"  ❌ Error: {data.get('message', 'Unknown error')}")
                    
                    response_count += 1
                except asyncio.TimeoutError:
                    print("  ⏱️  Timeout waiting for response")
                    break
            
            # Test 2: Send direct mode change
            print("\n📡 Test 2: Sending direct mode change to autonomous...")
            await websocket.send(json.dumps({
                "type": "set_mode",
                "mode": "autonomous"
            }))
            
            # Wait for mode change confirmation
            message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
            data = json.loads(message)
            if data['type'] == 'mode_changed':
                print(f"  ✅ Mode changed to: {data['mode']}")
            
            # Test 3: Check status
            print("\n📡 Test 3: Testing deactivation...")
            await websocket.send(json.dumps({
                "type": "command",
                "text": "disable autonomy"
            }))
            
            # Receive deactivation responses
            response_count = 0
            while response_count < 2:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    if data['type'] == 'response':
                        print(f"  💬 JARVIS: {data['text']}")
                    elif data['type'] == 'autonomy_status':
                        print(f"  🔄 Autonomy status: {'ENABLED' if data.get('enabled') else 'DISABLED'}")
                    response_count += 1
                except asyncio.TimeoutError:
                    break
            
            print("\n✅ All tests completed!")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nMake sure:")
        print("  1. Backend is running (python start_system.py)")
        print("  2. ANTHROPIC_API_KEY is set")
        print("  3. WebSocket endpoint is accessible")


async def test_voice_commands():
    """Test various voice commands for autonomy"""
    uri = "ws://localhost:8000/voice/jarvis/stream"
    
    print("\n\n🎤 Testing Voice Command Variations")
    print("=" * 50)
    
    test_commands = [
        "Hey JARVIS, activate full autonomy",
        "Enable autonomous mode",
        "Activate iron man mode",
        "Activate all systems",
        "Manual mode",
        "Stand down"
    ]
    
    try:
        async with websockets.connect(uri) as websocket:
            # Wait for connection
            await websocket.recv()
            
            for command in test_commands:
                print(f"\n📡 Testing: '{command}'")
                
                await websocket.send(json.dumps({
                    "type": "command",
                    "text": command
                }))
                
                # Get response
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                    data = json.loads(message)
                    
                    if data['type'] == 'response':
                        print(f"  ✅ Recognized as: {data.get('command_type', 'standard command')}")
                    
                    # Check for autonomy status
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    if data['type'] == 'autonomy_status':
                        print(f"  🔄 Autonomy: {'ON' if data.get('enabled') else 'OFF'}")
                        
                except asyncio.TimeoutError:
                    print("  ⏱️  No autonomy change detected")
                    
    except Exception as e:
        print(f"\n❌ Voice command test failed: {e}")


if __name__ == "__main__":
    print("🚀 Starting JARVIS Autonomy Tests\n")
    
    # Run tests
    asyncio.run(test_autonomy_activation())
    asyncio.run(test_voice_commands())
    
    print("\n\n📋 Summary:")
    print("If all tests passed, JARVIS autonomy activation is working correctly!")
    print("You can now say 'activate full autonomy' in the UI to enable all systems.")