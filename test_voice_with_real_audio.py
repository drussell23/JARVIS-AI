#!/usr/bin/env python3
"""Test voice verification with simulated real audio (using stored samples)."""

import asyncio
import asyncpg
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_with_real_audio():
    """Test using actual stored voice samples."""

    print("\n" + "="*80)
    print("TESTING WITH ACTUAL VOICE SAMPLES")
    print("="*80)

    # Get database password
    from backend.core.secret_manager import get_db_password
    db_password = get_db_password()

    # Connect to database
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        # Get a stored voice sample for Derek J. Russell
        print("\n1️⃣  RETRIEVING STORED VOICE SAMPLE:")
        print("-" * 40)

        sample = await conn.fetchrow("""
            SELECT
                vs.sample_id,
                vs.audio_data,
                LENGTH(vs.audio_data) as audio_size,
                vs.sample_rate,
                vs.duration_seconds,
                vs.transcription,
                vs.confidence_score
            FROM voice_samples vs
            JOIN speaker_profiles sp ON vs.speaker_id = sp.speaker_id
            WHERE sp.is_primary_user = true
                AND vs.audio_data IS NOT NULL
                AND LENGTH(vs.audio_data) > 1000
            ORDER BY vs.created_at DESC
            LIMIT 1
        """)

        if sample:
            print(f"\n✅ Found voice sample:")
            print(f"   ID: {sample['sample_id']}")
            print(f"   Size: {sample['audio_size']} bytes")
            print(f"   Duration: {sample['duration_seconds']:.2f}s")
            print(f"   Sample Rate: {sample['sample_rate']}Hz")
            print(f"   Transcription: {sample['transcription']}")
            print(f"   Quality: {sample['confidence_score']:.2%}")

            # Use this audio for verification
            audio_data = sample['audio_data']

            print("\n2️⃣  TESTING VERIFICATION WITH ACTUAL VOICE:")
            print("-" * 40)

            # Initialize the speaker verification service
            from backend.voice.speaker_verification_service import SpeakerVerificationService

            service = SpeakerVerificationService()
            await service.initialize()

            print(f"\n✅ Service initialized:")
            print(f"   Model dimension: {service.current_model_dimension}")
            print(f"   Verification threshold: {service.verification_threshold}")
            print(f"   Loaded profiles: {len(service.speaker_profiles)}")

            # Test verification
            print("\n Testing verification with actual stored voice sample...")
            result = await service.verify_speaker(audio_data)

            print(f"\n✅ VERIFICATION RESULT:")
            print(f"   Verified: {result.get('verified', False)}")
            print(f"   Confidence: {result.get('confidence', 0):.2%}")
            print(f"   Speaker Name: {result.get('speaker_name', 'None')}")
            print(f"   Is Owner: {result.get('is_owner', False)}")
            print(f"   Adaptive Threshold: {result.get('adaptive_threshold', 0):.2%}")

            if result.get('verified'):
                print(f"\n🎉 SUCCESS! Voice verification passed!")
                print(f"   The system correctly recognized Derek J. Russell")
            else:
                print(f"\n⚠️  Verification failed but confidence is {result.get('confidence', 0):.2%}")
                print(f"   This is using the owner's actual voice sample")
                print(f"   The stored embedding may need updating")

        else:
            print("\n❌ No voice samples found in database!")
            print("   Need to enroll voice samples first")

            # Check if we have any samples at all
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM voice_samples
            """)
            print(f"\n   Total samples in database: {count}")

    finally:
        await conn.close()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_with_real_audio())