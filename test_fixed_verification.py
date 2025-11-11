#!/usr/bin/env python3
"""Test voice verification with fixed audio handling."""

import asyncio
import sys
import os
import numpy as np
import pyaudio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_fixed_verification():
    """Test with fixed audio handling."""

    print("\n" + "="*80)
    print("TESTING FIXED VOICE VERIFICATION")
    print("="*80)

    from backend.voice.speaker_verification_service import SpeakerVerificationService

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n✅ Service ready with {len(service.speaker_profiles)} profiles")

    # Test 1: With random int16 audio (simulating JARVIS)
    print("\n1️⃣ Testing with random int16 audio (JARVIS format)...")
    random_int16 = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    random_bytes = random_int16.tobytes()

    result = await service.verify_speaker(random_bytes)
    print(f"   Random audio confidence: {result.get('confidence', 0):.2%}")

    # Test 2: Record actual voice
    print("\n2️⃣ Recording your voice...")
    print("   Say 'unlock my screen' in 3...")
    await asyncio.sleep(1)
    print("   2...")
    await asyncio.sleep(1)
    print("   1...")
    await asyncio.sleep(1)
    print("   🎤 RECORDING NOW - SPEAK!")

    # Record as int16 (JARVIS format)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    RECORD_SECONDS = 3

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,  # int16 format
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    frames = []
    for _ in range(int(SAMPLE_RATE / CHUNK_SIZE * RECORD_SECONDS)):
        data = stream.read(CHUNK_SIZE)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    print("   ✅ Recording complete!")

    # Convert to bytes (already int16)
    audio_bytes = b''.join(frames)

    # Test with the raw int16 bytes
    print("\n3️⃣ Testing with your voice (int16 PCM)...")
    result = await service.verify_speaker(audio_bytes)
    print(f"   Verified: {result.get('verified', False)}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")
    print(f"   Speaker: {result.get('speaker_name', 'None')}")

    # Check what the threshold is
    if "Derek J. Russell" in service.speaker_profiles:
        profile = service.speaker_profiles["Derek J. Russell"]
        threshold = profile.get('threshold', 0.45)
        print(f"\n   Threshold: {threshold*100:.0f}%")

        confidence = result.get('confidence', 0)
        if confidence < threshold:
            print(f"   Need {(threshold - confidence)*100:.1f}% more confidence")
        else:
            print(f"   ✅ Above threshold by {(confidence - threshold)*100:.1f}%")

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if result.get('confidence', 0) > 0.20:
        print("✅ Voice verification is working! Confidence is reasonable.")
        print("   If still below threshold, record fresh samples:")
        print("   python backend/quick_voice_enhancement.py")
    else:
        print("⚠️ Confidence is still low. This could mean:")
        print("   1. The stored embedding doesn't match your current voice")
        print("   2. Recording conditions are different")
        print("   3. Microphone or environment has changed")
        print("\n   Solution: Re-record voice samples in current environment")

if __name__ == "__main__":
    asyncio.run(test_fixed_verification())