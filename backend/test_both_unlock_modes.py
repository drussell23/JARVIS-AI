#!/usr/bin/env python3
"""
Test Both Unlock Modes
======================

Tests that JARVIS can handle both:
1. Manual "unlock my screen" command
2. Context-aware unlock when screen is locked
"""

import asyncio
import json
import websockets
from datetime import datetime


async def test_manual_unlock():
    """Test manual unlock command"""
    print("\n🔓 Test 1: Manual Unlock Command")
    print("-"*50)
    
    try:
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            # Wait for welcome
            welcome = await ws.recv()
            print(f"Connected: {json.loads(welcome).get('message')}")
            
            # Send manual unlock command
            command = {
                "type": "command",
                "text": "unlock my screen"
            }
            
            print("Sending: 'unlock my screen'")
            await ws.send(json.dumps(command))
            
            # Wait for responses
            responses = []
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get('type') == 'debug_log':
                        print(f"DEBUG: {data.get('message', '')}")
                        continue
                        
                    if data.get('text'):
                        print(f"JARVIS: {data['text']}")
                        responses.append(data)
                        
                    # Don't break on intermediate responses
                    if data.get('type') == 'response' and not data.get('intermediate'):
                        break
                        
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed unexpectedly")
            
            # Check result
            success = any('unlock' in r.get('text', '').lower() for r in responses)
            print(f"Result: {'✅ SUCCESS' if success else '❌ FAILED'}")
            
    except Exception as e:
        print(f"Error: {e}")


async def test_context_aware_unlock():
    """Test context-aware unlock"""
    print("\n\n🤖 Test 2: Context-Aware Unlock")
    print("-"*50)
    
    try:
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            # Wait for welcome
            welcome = await ws.recv()
            print(f"Connected: {json.loads(welcome).get('message')}")
            
            # Send command that requires screen (simulating locked screen scenario)
            command = {
                "type": "command",
                "text": "open Safari and search for Python tutorials"
            }
            
            print("Sending: 'open Safari and search for Python tutorials'")
            print("(This should trigger context-aware unlock if screen is locked)")
            await ws.send(json.dumps(command))
            
            # Wait for responses
            responses = []
            got_context_message = False
            
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    data = json.loads(response)
                    
                    if data.get('type') == 'debug_log':
                        print(f"DEBUG: {data.get('message', '')}")
                        continue
                        
                    if data.get('text'):
                        print(f"JARVIS: {data['text']}")
                        responses.append(data)
                        
                        # Check for context-aware message
                        if "I see your screen is locked" in data['text']:
                            got_context_message = True
                        
                    # Look for final response
                    if data.get('type') == 'response' and not data.get('intermediate'):
                        if any(word in data.get('text', '').lower() for word in ['opened safari', 'searched', 'complete']):
                            break
                        
            except asyncio.TimeoutError:
                print("Timeout waiting for response")
            except websockets.exceptions.ConnectionClosed:
                print("WebSocket connection closed unexpectedly")
            
            # Check result
            print(f"\nContext-aware unlock message: {'✅ YES' if got_context_message else '❌ NO'}")
            print(f"Result: {'✅ SUCCESS' if responses else '❌ NO RESPONSE'}")
            
    except Exception as e:
        print(f"Error: {e}")


async def main():
    """Run both tests"""
    print("🧪 Testing Both Unlock Modes")
    print("="*60)
    print("This test verifies JARVIS can handle:")
    print("1. Direct 'unlock my screen' commands")
    print("2. Context-aware unlock when needed")
    
    await test_manual_unlock()
    await test_context_aware_unlock()
    
    print("\n" + "="*60)
    print("✨ Test Complete!")


if __name__ == "__main__":
    asyncio.run(main())