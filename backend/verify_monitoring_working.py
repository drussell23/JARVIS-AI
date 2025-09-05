#!/usr/bin/env python3
"""
Final verification that monitoring command is working properly
"""

import asyncio
import httpx
import json

async def verify_monitoring():
    """Verify that monitoring command returns correct response"""
    print("\n✅ MONITORING COMMAND VERIFICATION\n")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Test the monitoring command
    print("\n🔍 Testing 'start monitoring my screen' command...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/voice/jarvis/command",
                json={"text": "start monitoring my screen"},
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                
                print(f"\n📊 Response Analysis:")
                print(f"   - Status Code: {response.status_code}")
                print(f"   - Success: {result.get('success', False)}")
                print(f"   - Response Length: {len(response_text)} chars")
                
                # Check for key phrases
                checks = {
                    "video_capturing": "video capturing" in response_text.lower(),
                    "macos_native": "macos" in response_text.lower(),
                    "purple_indicator": "purple" in response_text.lower() or "recording indicator" in response_text.lower(),
                    "monitoring_active": "monitoring your screen" in response_text.lower(),
                    "fps_mentioned": "30 fps" in response_text.lower() or "30fps" in response_text.lower()
                }
                
                print(f"\n🔍 Response Content Checks:")
                for check, passed in checks.items():
                    status = "✅" if passed else "❌"
                    print(f"   {status} {check.replace('_', ' ').title()}: {passed}")
                
                print(f"\n📝 Full Response:")
                print(f"   {response_text}")
                
                # Overall verdict
                is_monitoring_response = any([
                    checks["video_capturing"],
                    checks["macos_native"],
                    checks["purple_indicator"],
                    checks["monitoring_active"]
                ])
                
                is_generic_response = "Task completed successfully" in response_text and "Yes sir, I can see your screen" in response_text
                
                print(f"\n🎯 Verdict:")
                if is_monitoring_response and not is_generic_response:
                    print("   ✅ SUCCESS: Monitoring command is working correctly!")
                    print("   The system is returning the proper video capture activation response.")
                else:
                    print("   ❌ FAILED: Response is not the expected monitoring activation message")
                    print(f"   - Is monitoring response: {is_monitoring_response}")
                    print(f"   - Is generic response: {is_generic_response}")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        print(f"   Type: {type(e).__name__}")
        print(f"   Traceback:")
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Verification complete!")

if __name__ == "__main__":
    asyncio.run(verify_monitoring())