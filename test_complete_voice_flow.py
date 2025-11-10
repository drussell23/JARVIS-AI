#!/usr/bin/env python3
"""Test complete voice sample storage and retrieval flow."""

import asyncio
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend'))

from backend.intelligence.learning_database import JARVISLearningDatabase
from backend.intelligence.cloud_database_adapter import CloudDatabaseAdapter, DatabaseConfig
import numpy as np


async def test_complete_flow():
    """Test the complete voice sample storage and retrieval flow."""

    print("\n" + "="*60)
    print("TESTING COMPLETE VOICE SAMPLE FLOW")
    print("="*60)

    # Initialize database with Cloud SQL adapter
    config = DatabaseConfig()
    adapter = CloudDatabaseAdapter(config)
    await adapter.initialize()

    db = JARVISLearningDatabase()
    db.adapter = adapter  # Use Cloud SQL adapter
    await db.initialize()

    # Test with speaker ID 1 (Derek J. Russell) which has samples
    speaker_id_with_samples = 1

    print(f"\n✅ Testing retrieval for speaker ID {speaker_id_with_samples}:")
    samples = await db.get_voice_samples_for_speaker(speaker_id_with_samples, limit=5)

    if samples:
        print(f"   Found {len(samples)} samples!")
        for i, sample in enumerate(samples[:3], 1):
            audio_size = len(sample['audio_data']) if sample.get('audio_data') else 0
            print(f"   Sample {i}: ID={sample['sample_id']}, Audio={audio_size} bytes, Quality={sample.get('quality_score', 'N/A')}")
    else:
        print("   ❌ No samples found")

    # Now test creating a NEW speaker and adding a sample
    test_speaker = "Test Speaker"
    print(f"\n✅ Creating new speaker '{test_speaker}':")

    # Get or create speaker
    new_speaker_id = await db.get_or_create_speaker_profile(test_speaker)
    print(f"   Created/found speaker with ID: {new_speaker_id}")

    # Create test audio data
    test_audio = np.random.random(8000).astype(np.float32).tobytes()  # 0.5 second of audio

    # Store a voice sample
    print(f"\n✅ Storing new voice sample for speaker ID {new_speaker_id}:")
    sample_id = await db.record_voice_sample(
        speaker_id=new_speaker_id,
        audio_data=test_audio,
        sample_rate=16000,
        duration_ms=500,
        transcription="Test audio sample"
    )

    if sample_id:
        print(f"   Successfully stored sample with ID: {sample_id}")

        # Now retrieve it back
        print(f"\n✅ Retrieving samples for new speaker ID {new_speaker_id}:")
        new_samples = await db.get_voice_samples_for_speaker(new_speaker_id, limit=5)

        if new_samples:
            print(f"   Found {len(new_samples)} samples!")
            for sample in new_samples:
                audio_size = len(sample['audio_data']) if sample.get('audio_data') else 0
                print(f"   Sample ID={sample['sample_id']}, Audio={audio_size} bytes, Transcription='{sample.get('transcription', 'N/A')}'")
        else:
            print("   ❌ No samples found")
    else:
        print("   ❌ Failed to store sample")

    # Clean up database
    await db.cleanup()

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_complete_flow())