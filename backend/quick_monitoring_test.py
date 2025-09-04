#!/usr/bin/env python3
"""
Quick test to check monitoring command flow
"""

import asyncio
import httpx
import json

async def quick_test():
    """Quick test of monitoring command"""
    print("\n🧪 Quick Monitoring Command Test\n")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # 1. Check debug info
    print("\n1️⃣ Checking system state...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/debug/monitoring")
            debug_info = response.json()
            
            print(f"✅ Vision System:")
            print(f"   - Enhanced Vision: {debug_info['vision_system']['enhanced_vision_available']}")
            print(f"   - Video Streaming: {debug_info['vision_system']['video_streaming_available']}")
            print(f"   - macOS Capture: {debug_info['vision_system']['macos_capture_available']}")
            
            print(f"\n✅ JARVIS System:")
            print(f"   - JARVIS Available: {debug_info['jarvis_system']['jarvis_voice_available']}")
            print(f"   - Claude Chatbot: {debug_info['jarvis_system']['claude_chatbot_available']}")
            
            print(f"\n✅ Command Routing:")
            print(f"   - 'start monitoring my screen' is vision command: {debug_info['is_vision_command']}")
            print(f"   - Expected route: {debug_info['expected_route']}")
    except Exception as e:
        print(f"❌ Debug check failed: {e}")
    
    # 2. Test command routing
    print("\n2️⃣ Testing command routing...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/debug/test_command_route",
                json={"command": "start monitoring my screen"}
            )
            route_info = response.json()
            
            print(f"✅ Command: '{route_info['command']}'")
            print(f"   - Is vision command: {route_info['is_vision_command']}")
            print(f"   - Has monitoring keywords: {route_info['has_monitoring_keywords']}")
            print(f"   - Has screen keywords: {route_info['has_screen_keywords']}")
            print(f"   - Expected handler: {route_info['expected_handler']}")
    except Exception as e:
        print(f"❌ Route test failed: {e}")
    
    # 3. Send actual command
    print("\n3️⃣ Sending monitoring command...")
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/voice/jarvis/command",
                json={"text": "start monitoring my screen"},
                timeout=30.0
            )
            result = response.json()
            response_text = result.get("response", "")
            
            print(f"✅ Response received:")
            print(f"   {response_text[:200]}{'...' if len(response_text) > 200 else ''}")
            
            # Check response type
            if "Task completed successfully" in response_text:
                print("\n❌ Got GENERIC response - monitoring not properly activated")
            elif any(phrase in response_text.lower() for phrase in [
                "video capturing", 
                "purple recording indicator",
                "monitoring your screen",
                "swift",
                "macos"
            ]):
                print("\n✅ Got MONITORING response - system working correctly!")
            else:
                print("\n⚠️  Got unexpected response type")
                
    except Exception as e:
        print(f"❌ Command failed: {e}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(quick_test())