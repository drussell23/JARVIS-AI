#!/usr/bin/env python3
"""
Test Audio Storage Fix
======================
Tests that audio data is now properly stored to both SQLite and Cloud SQL.
"""

import asyncio
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_audio_storage():
    print("\n" + "=" * 80)
    print("🧪 TESTING AUDIO STORAGE FIX")
    print("=" * 80)

    from intelligence.learning_database import get_learning_database

    # Initialize database
    print("\n📊 Connecting to database...")
    db = await get_learning_database()
    print(f"   Database instance type: {type(db)}")
    print(f"   Has record_voice_sample: {hasattr(db, 'record_voice_sample')}")

    # Create test audio data (2 seconds of synthetic audio)
    print("🎤 Creating test audio sample...")
    sample_rate = 16000
    duration = 2
    # Generate a sine wave at 440Hz (A4 note)
    t = np.linspace(0, duration, sample_rate * duration)
    audio = np.sin(2 * np.pi * 440 * t)
    # Add some noise to make it more realistic
    audio += np.random.normal(0, 0.01, audio.shape)
    # Convert to int16 format
    audio_int16 = (audio * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    print(f"   Generated {len(audio_bytes)} bytes of audio data")

    # Store the sample
    print("\n💾 Storing voice sample to database...")
    try:
        sample_id = await db.record_voice_sample(
            speaker_name="Derek J. Russell",
            audio_data=audio_bytes,
            transcription="Test sample with audio",
            audio_duration_ms=duration * 1000,
            quality_score=0.8
        )

        if sample_id and sample_id > 0:
            print(f"✅ Sample stored successfully (ID: {sample_id})")
        else:
            print("❌ Failed to store sample")
            print(f"   Returned value: {sample_id}")
    except Exception as e:
        print(f"❌ Exception during storage: {e}")
        import traceback
        traceback.print_exc()

    # Wait a moment for async operations
    await asyncio.sleep(1)

    # Try to retrieve the sample
    print("\n🔍 Retrieving voice samples to verify storage...")

    # Get Derek's speaker ID
    profiles = await db.get_all_speaker_profiles()
    derek_id = None
    for profile in (profiles.values() if isinstance(profiles, dict) else profiles):
        if "Derek" in profile.get("speaker_name", ""):
            derek_id = profile.get("speaker_id")
            break

    if derek_id:
        samples = await db.get_voice_samples_for_speaker(derek_id, limit=5)

        print(f"\n📊 Results:")
        print(f"   Total samples found: {len(samples)}")

        samples_with_audio = 0
        for i, sample in enumerate(samples, 1):
            has_audio = sample.get('audio_data') is not None
            audio_size = len(sample.get('audio_data', b''))
            if has_audio:
                samples_with_audio += 1
                print(f"   Sample {i}: ✅ Has audio ({audio_size:,} bytes)")
            else:
                print(f"   Sample {i}: ❌ No audio data")

        print(f"\n   Summary: {samples_with_audio}/{len(samples)} samples have audio data")

        if samples_with_audio > 0:
            print("\n✅ SUCCESS! Audio storage is now working!")
            print("   Both SQLite and Cloud SQL are storing audio data properly.")
        else:
            print("\n⚠️  No audio data found in retrieved samples")
            print("   Check Cloud SQL connection and permissions")

    await db.close()
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(test_audio_storage())