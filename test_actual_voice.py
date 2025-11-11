#!/usr/bin/env python3
"""Test with actual voice recording."""

import asyncio
import sys
import os
import numpy as np
import pyaudio

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_actual_voice():
    """Record and test actual voice."""

    print("\n" + "="*80)
    print("TESTING WITH ACTUAL VOICE RECORDING")
    print("="*80)

    from backend.voice.speaker_verification_service import SpeakerVerificationService

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n✅ Service ready with {len(service.speaker_profiles)} profiles")

    # Record audio
    print("\n📢 Say 'unlock my screen' when ready...")
    print("   Recording for 3 seconds in 2...")
    await asyncio.sleep(1)
    print("   Recording for 3 seconds in 1...")
    await asyncio.sleep(1)
    print("   🎤 RECORDING NOW - SPEAK!")

    # Audio parameters (match JARVIS)
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 1024
    RECORD_SECONDS = 3

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
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

    # Convert to bytes
    audio_bytes = b''.join(frames)

    # Test 1: Raw bytes as JARVIS would send
    print("\n1️⃣ Testing with raw PCM bytes (as JARVIS sends)...")
    result = await service.verify_speaker(audio_bytes)
    print(f"   Result: {result.get('verified', False)}")
    print(f"   Confidence: {result.get('confidence', 0):.2%}")

    # Test 2: Convert to float32 array
    print("\n2️⃣ Testing with float32 conversion...")
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    audio_float32 = audio_int16.astype(np.float32) / 32768.0
    result2 = await service.verify_speaker(audio_float32.tobytes())
    print(f"   Result: {result2.get('verified', False)}")
    print(f"   Confidence: {result2.get('confidence', 0):.2%}")

    # Test 3: Check what the engine expects
    print("\n3️⃣ Checking audio format...")
    print(f"   Raw bytes length: {len(audio_bytes)}")
    print(f"   Expected samples: {SAMPLE_RATE * RECORD_SECONDS}")
    print(f"   Int16 array shape: {audio_int16.shape}")
    print(f"   Float32 array shape: {audio_float32.shape}")

    # Debug the embedding extraction
    print("\n4️⃣ Extracting embedding directly...")
    engine = service.speechbrain_engine
    embedding = await engine.extract_speaker_embedding(audio_bytes)
    if embedding is not None:
        if embedding.ndim == 2:
            embedding = embedding.squeeze(0)
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding norm: {np.linalg.norm(embedding):.4f}")

        # Compare with stored
        if "Derek J. Russell" in service.speaker_profiles:
            stored = service.speaker_profiles["Derek J. Russell"]["embedding"]
            similarity = np.dot(embedding, stored) / (np.linalg.norm(embedding) * np.linalg.norm(stored))
            print(f"   Similarity to stored: {similarity:.2%}")
    else:
        print("   ❌ Could not extract embedding")

    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(test_actual_voice())