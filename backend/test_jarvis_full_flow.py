#!/usr/bin/env python3
"""Test full JARVIS flow as frontend does"""

import asyncio
import websockets
import json
import httpx

async def test_full_jarvis_flow():
    """Test the complete flow that frontend uses"""
    
    print("1️⃣ Checking JARVIS status...")
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/voice/jarvis/status")
        status = response.json()
        print(f"   Status: {status['status']}")
        print(f"   Features: {len(status.get('features', []))} available")
    
    print("\n2️⃣ Connecting to WebSocket...")
    uri = "ws://localhost:8000/voice/jarvis/stream"
    
    async with websockets.connect(uri) as websocket:
        # Wait for connected message
        message = await websocket.recv()
        data = json.loads(message)
        print(f"   Connected: {data['type']} - {data.get('message', '')}")
        
        # Send monitoring command
        print("\n3️⃣ Sending monitoring command...")
        command = {
            "type": "command",
            "text": "start monitoring my screen"
        }
        await websocket.send(json.dumps(command))
        
        # Collect all responses
        responses = []
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            responses.append(data)
            print(f"   Response {len(responses)}: {data['type']}")
            
            if data.get('type') == 'response':
                response_text = data.get('text', '')
                print(f"\n📨 Final response: {response_text[:200]}...")
                
                if 'Failed to start' in response_text:
                    print("\n❌ Monitoring failed through WebSocket")
                    
                    # Let's test the REST API too
                    print("\n4️⃣ Testing REST API endpoint...")
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://localhost:8000/voice/jarvis/command",
                            json={"text": "start monitoring my screen"}
                        )
                        result = response.json()
                        print(f"   REST API response: {result['response'][:200]}...")
                        
                        if 'Failed to start' in result['response']:
                            print("❌ Also failed through REST API")
                        else:
                            print("✅ Works through REST API!")
                else:
                    print("\n✅ Monitoring successful through WebSocket!")
                break

if __name__ == "__main__":
    asyncio.run(test_full_jarvis_flow())