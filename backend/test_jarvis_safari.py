#!/usr/bin/env python3
"""
Test JARVIS Safari Opening
Simple test to verify JARVIS can open Safari
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.jarvis_ai_core import get_jarvis_ai_core


async def test_safari():
    """Test opening Safari through JARVIS AI Core"""
    print("\n🧪 Testing JARVIS Safari Command")
    print("=" * 40)
    
    # Initialize JARVIS
    print("Initializing JARVIS AI Core...")
    jarvis = get_jarvis_ai_core()
    
    # Test the command
    print("\nSending command: 'open Safari'")
    result = await jarvis.process_speech_command("open Safari")
    
    print("\nResult:")
    print(f"Intent: {result.get('intent')}")
    print(f"Action: {result.get('action')}")
    print(f"Confidence: {result.get('confidence')}")
    print(f"Response: {result.get('response')}")
    
    if result.get('executed'):
        print(f"\n✅ Command executed!")
        print(f"Success: {result['execution_result']['success']}")
        print(f"Message: {result['execution_result']['message']}")
    else:
        print("\n⚠️  Command was not executed automatically")
        print("This might be because:")
        print("- Confidence was too low")
        print("- Intent was not recognized as app_control")
        print("- There was an error in execution")
    
    return result


if __name__ == "__main__":
    result = asyncio.run(test_safari())
    
    # Summary
    print("\n" + "=" * 40)
    if result.get('executed') and result.get('execution_result', {}).get('success'):
        print("✅ SUCCESS: Safari should now be open!")
    else:
        print("❌ FAILED: Safari was not opened")
        print("\nTroubleshooting:")
        print("1. Check if ANTHROPIC_API_KEY is set")
        print("2. Verify macOS permissions for automation")
        print("3. Check the logs above for errors")