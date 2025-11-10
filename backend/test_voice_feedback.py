#!/usr/bin/env python3
"""
Test Voice Unlock with JARVIS Voice Feedback
==============================================

Tests the new voice feedback feature where JARVIS audibly greets you
by name when unlocking the screen.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_jarvis_voice():
    """Test JARVIS voice feedback directly"""
    from voice.voice_unlock_integration import VoiceUnlockIntegration

    print("\n" + "=" * 80)
    print("🔊 TESTING JARVIS VOICE FEEDBACK")
    print("=" * 80)

    # Initialize voice unlock integration
    voice_unlock = VoiceUnlockIntegration()

    print("\n📋 Voice Configuration:")
    print(f"   Voice Name: {voice_unlock.jarvis_voice_name}")
    print(f"   Normal Rate: {voice_unlock.jarvis_voice_rate} WPM")
    print(f"   Urgent Rate: {voice_unlock.urgent_voice_rate} WPM")
    print(f"   Voice Feedback Enabled: {voice_unlock.enable_voice_feedback}")

    # Test 1: Normal welcome message
    print("\n🎤 Test 1: Welcome Message (Normal Rate)")
    print("   Testing: 'Welcome, Derek J. Russell!'")
    await voice_unlock._speak_jarvis("Welcome, Derek J. Russell!", urgent=False)
    await asyncio.sleep(3)  # Wait for speech to complete

    # Test 2: Urgent failure message
    print("\n🎤 Test 2: Failure Message (Urgent Rate)")
    print("   Testing: 'Voice not recognized'")
    await voice_unlock._speak_jarvis("Voice not recognized", urgent=True)
    await asyncio.sleep(2)

    # Test 3: Access denied
    print("\n🎤 Test 3: Access Denied (Urgent Rate)")
    print("   Testing: 'Access denied'")
    await voice_unlock._speak_jarvis("Access denied", urgent=True)
    await asyncio.sleep(2)

    print("\n" + "=" * 80)
    print("✅ VOICE FEEDBACK TEST COMPLETE")
    print("=" * 80)
    print("\n💡 If you heard JARVIS speak, voice feedback is working!")
    print("   - Daniel (British male voice) should be used")
    print("   - Welcome messages use 175 WPM")
    print("   - Failure/urgent messages use 200 WPM\n")


if __name__ == "__main__":
    asyncio.run(test_jarvis_voice())
