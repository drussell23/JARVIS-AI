#!/usr/bin/env python3
"""
Simple Test - JARVIS Voice Only
=================================

Tests just the JARVIS voice output without importing the full unlock system.
"""

import asyncio
import os
import subprocess


async def speak_jarvis(message: str, urgent: bool = False):
    """Speak with JARVIS voice (Daniel - British male voice)"""
    jarvis_voice_name = os.getenv('JARVIS_VOICE_NAME', 'Daniel')
    jarvis_voice_rate = 175 if not urgent else 200

    try:
        # Run say command asynchronously
        proc = await asyncio.create_subprocess_exec(
            'say',
            '-v', jarvis_voice_name,
            '-r', str(jarvis_voice_rate),
            message,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL
        )

        print(f"🔊 Speaking: '{message}' (voice={jarvis_voice_name}, rate={jarvis_voice_rate} WPM)")

        # Wait for speech to complete
        await proc.wait()

    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    print("\n" + "=" * 80)
    print("🔊 JARVIS VOICE FEEDBACK TEST")
    print("=" * 80)

    # Test 1: Welcome message (what you'll hear on successful unlock)
    print("\n🎤 Test 1: Successful Unlock")
    await speak_jarvis("Welcome, Derek J. Russell!", urgent=False)

    await asyncio.sleep(1)

    # Test 2: Failure message
    print("\n🎤 Test 2: Voice Not Recognized")
    await speak_jarvis("Voice not recognized", urgent=True)

    await asyncio.sleep(1)

    # Test 3: Access denied
    print("\n🎤 Test 3: Access Denied")
    await speak_jarvis("Access denied", urgent=True)

    print("\n" + "=" * 80)
    print("✅ TEST COMPLETE")
    print("=" * 80)
    print("\n💡 Summary:")
    print("   ✓ Used Daniel voice (British male)")
    print("   ✓ Welcome messages: 175 WPM (normal rate)")
    print("   ✓ Failure messages: 200 WPM (urgent rate)")
    print("\n   When you unlock with voice, you'll hear:")
    print("   'Welcome, Derek J. Russell!' in the same voice!\n")


if __name__ == "__main__":
    asyncio.run(main())
