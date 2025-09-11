#!/usr/bin/env python3
"""
Verify Weather Fix - Test that weather queries no longer hang
"""

import asyncio
import aiohttp
import json
import time
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

async def test_api_endpoint():
    """Test the REST API endpoint with weather query"""
    print("🔍 Testing REST API Endpoint")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test weather command via API
    test_command = {
        "text": "what's the weather today"
    }
    
    print(f"\n1. Sending command: '{test_command['text']}'")
    start_time = time.time()
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/api/voice/command",
                json=test_command,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    print(f"✅ Response received in {elapsed:.2f}s")
                    print(f"   Response: {result.get('response', 'No response')}")
                    
                    # Check if it's a timeout response
                    if "trouble reading the weather" in result.get('response', ''):
                        print("   ⚠️  Got timeout fallback response (Weather app should open)")
                    elif "weather" in result.get('response', '').lower():
                        print("   ✅ Got actual weather data!")
                else:
                    print(f"❌ HTTP {response.status}: {await response.text()}")
                    
    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        print(f"❌ Client timeout after {elapsed:.2f}s - API might still be hanging")
    except Exception as e:
        print(f"❌ Error: {e}")


async def test_websocket_endpoint():
    """Test WebSocket endpoint with weather query"""
    print("\n\n🔍 Testing WebSocket Endpoint")
    print("=" * 60)
    
    ws_url = "ws://localhost:8000/ws"
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(ws_url) as ws:
                print("✅ Connected to WebSocket")
                
                # Send voice command
                message = {
                    "type": "voice_command",
                    "command": "jarvis what's the weather today"
                }
                
                print(f"\n1. Sending: {message}")
                start_time = time.time()
                
                await ws.send_json(message)
                
                # Wait for response with timeout
                try:
                    response = await asyncio.wait_for(
                        ws.receive_json(),
                        timeout=30.0
                    )
                    elapsed = time.time() - start_time
                    
                    print(f"✅ Response received in {elapsed:.2f}s")
                    print(f"   Type: {response.get('type')}")
                    print(f"   Success: {response.get('success')}")
                    print(f"   Response: {response.get('response', 'No response')}")
                    
                    # Check response content
                    if response.get('response'):
                        if "trouble reading the weather" in response['response']:
                            print("   ⚠️  Got timeout fallback response")
                        elif "weather" in response['response'].lower():
                            print("   ✅ Got actual weather data!")
                        
                except asyncio.TimeoutError:
                    elapsed = time.time() - start_time
                    print(f"❌ WebSocket timeout after {elapsed:.2f}s")
                    
    except Exception as e:
        print(f"❌ WebSocket error: {e}")


async def test_direct_jarvis():
    """Test JARVIS directly to isolate the issue"""
    print("\n\n🔍 Testing JARVIS Directly (No API)")
    print("=" * 60)
    
    try:
        # Import and initialize JARVIS
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
        from voice.jarvis_agent_voice import JARVISAgentVoice
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("❌ No API key found")
            return
            
        print("1. Initializing JARVIS...")
        vision_analyzer = ClaudeVisionAnalyzerMain(api_key)
        jarvis = JARVISAgentVoice(vision_analyzer=vision_analyzer)
        
        print("2. Testing weather command...")
        start_time = time.time()
        
        try:
            response = await asyncio.wait_for(
                jarvis.process_voice_input("what's the weather today"),
                timeout=30.0
            )
            elapsed = time.time() - start_time
            
            print(f"✅ Response in {elapsed:.2f}s: {response}")
            
        except asyncio.TimeoutError:
            elapsed = time.time() - start_time
            print(f"❌ Direct JARVIS call timed out after {elapsed:.2f}s")
            print("   This confirms the hang is in JARVIS weather processing")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all tests"""
    print("🌤️ Weather Fix Verification")
    print("This will test if weather queries still hang or timeout properly\n")
    
    # Test API endpoint
    await test_api_endpoint()
    
    # Test WebSocket
    await test_websocket_endpoint()
    
    # Test direct JARVIS
    await test_direct_jarvis()
    
    print("\n\n✅ Verification Complete!")
    print("\nExpected behavior with fix:")
    print("- API calls should respond within 25 seconds")
    print("- If weather vision fails, Weather app should open")
    print("- No more indefinite 'Processing...' hang")


if __name__ == "__main__":
    asyncio.run(main())