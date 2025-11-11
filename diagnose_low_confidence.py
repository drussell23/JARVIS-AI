#!/usr/bin/env python3
"""Diagnose why confidence is 7.67% after BEAST MODE."""

import asyncio
import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def diagnose_confidence():
    """Diagnose the low confidence issue."""

    print("\n" + "="*80)
    print("DIAGNOSING 7.67% CONFIDENCE ISSUE")
    print("="*80)

    from backend.voice.speaker_verification_service import SpeakerVerificationService

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n📊 SERVICE STATUS:")
    print("-" * 40)
    print(f"Model dimension: {service.current_model_dimension}")
    print(f"Verification threshold: {service.verification_threshold*100:.0f}%")
    print(f"Profiles loaded: {len(service.speaker_profiles)}")

    # Check profile details
    for name, profile in service.speaker_profiles.items():
        print(f"\n👤 Profile: {name}")

        # Check embedding
        if 'embedding' in profile:
            emb = profile['embedding']
            if isinstance(emb, np.ndarray):
                print(f"   Embedding shape: {emb.shape}")
                print(f"   Embedding dtype: {emb.dtype}")
                print(f"   Embedding norm: {np.linalg.norm(emb):.4f}")
                print(f"   First 5 values: {emb[:5]}")
            else:
                print(f"   ❌ Embedding is not numpy array: {type(emb)}")
        else:
            print("   ❌ No embedding in profile!")

        # Check acoustic features
        if 'acoustic_features' in profile:
            af = profile['acoustic_features']
            if af and isinstance(af, dict):
                print(f"   Acoustic features: {len(af)} parameters")
                if 'pitch_mean' in af:
                    print(f"   Pitch mean: {af['pitch_mean']:.1f} Hz")
                if 'formant1_mean' in af:
                    print(f"   Formant 1: {af['formant1_mean']:.1f} Hz")
            else:
                print(f"   ❌ No acoustic features")

        print(f"   Threshold: {profile.get('threshold', 'default')}")
        print(f"   Total samples: {profile.get('total_samples', 0)}")

    print("\n🔍 POSSIBLE ISSUES:")
    print("-" * 40)

    # Check if embedding was updated after BEAST MODE
    from backend.intelligence.learning_database import LearningDatabase
    db = LearningDatabase()
    await db.initialize()

    # Get the latest embedding from database
    import asyncpg
    from backend.core.secret_manager import get_db_password

    db_password = get_db_password()
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        # Check when embedding was last updated
        profile_data = await conn.fetchrow("""
            SELECT
                embedding_data,
                acoustic_features,
                total_samples,
                last_updated,
                enrollment_quality_score
            FROM speaker_profiles
            WHERE speaker_id = 1
        """)

        if profile_data:
            print(f"\n📅 DATABASE STATUS:")
            print(f"   Last updated: {profile_data['last_updated']}")
            print(f"   Total samples: {profile_data['total_samples']}")
            print(f"   Quality score: {profile_data['enrollment_quality_score']:.2%}")

            if profile_data['embedding_data']:
                db_emb = np.frombuffer(profile_data['embedding_data'], dtype=np.float32)
                print(f"   DB embedding shape: {db_emb.shape}")
                print(f"   DB embedding norm: {np.linalg.norm(db_emb):.4f}")

                # Compare with loaded embedding
                if 'Derek J. Russell' in service.speaker_profiles:
                    loaded_emb = service.speaker_profiles['Derek J. Russell']['embedding']
                    if isinstance(loaded_emb, np.ndarray):
                        # Check if they're the same
                        if np.array_equal(db_emb, loaded_emb):
                            print("   ✅ Loaded embedding matches database")
                        else:
                            print("   ❌ Loaded embedding DIFFERS from database!")
                            print(f"      Difference norm: {np.linalg.norm(db_emb - loaded_emb):.4f}")

            # Check recent voice samples
            recent_samples = await conn.fetch("""
                SELECT
                    sample_id,
                    confidence_score,
                    duration_seconds,
                    created_at,
                    LENGTH(audio_data) as audio_bytes
                FROM voice_samples
                WHERE speaker_id = 1
                ORDER BY created_at DESC
                LIMIT 5
            """)

            print(f"\n📼 RECENT SAMPLES:")
            for i, sample in enumerate(recent_samples, 1):
                print(f"   {i}. {sample['created_at']}: {sample['confidence_score']:.2%} confidence, {sample['audio_bytes'] or 0} bytes")

    finally:
        await conn.close()

    print("\n💡 DIAGNOSIS:")
    print("-" * 40)

    print("""
    The 7.67% confidence suggests:

    1. ✅ System IS detecting some similarity (not 0%)
    2. ❌ But embedding doesn't match current voice well

    This typically happens when:
    - Voice samples were recorded in different conditions
    - Background noise levels differ
    - Microphone position/distance changed
    - Voice characteristics changed (tired, different time of day)

    SOLUTION:
    Re-record samples in your CURRENT environment:
       python backend/quick_voice_enhancement.py

    This will create a fresh embedding matching your current voice.
    """)

    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(diagnose_confidence())