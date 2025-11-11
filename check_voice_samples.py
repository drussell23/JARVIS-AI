#!/usr/bin/env python3
"""Check voice samples in database after recording."""

import asyncio
import asyncpg
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def check_voice_samples():
    """Check voice samples in database."""

    print("\n" + "="*80)
    print("VOICE SAMPLES DATABASE CHECK")
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
        # 1. Check total samples
        print("\n1️⃣  SAMPLE COUNTS:")
        print("-" * 40)

        total = await conn.fetchval("""
            SELECT COUNT(*) FROM voice_samples WHERE speaker_id = 1
        """)
        print(f"Total samples for Derek J. Russell: {total}")

        with_audio = await conn.fetchval("""
            SELECT COUNT(*) FROM voice_samples
            WHERE speaker_id = 1 AND audio_data IS NOT NULL
        """)
        print(f"Samples WITH audio_data: {with_audio}")

        without_audio = await conn.fetchval("""
            SELECT COUNT(*) FROM voice_samples
            WHERE speaker_id = 1 AND audio_data IS NULL
        """)
        print(f"Samples WITHOUT audio_data: {without_audio}")

        # 2. Check recent samples
        print("\n2️⃣  RECENT SAMPLES:")
        print("-" * 40)

        recent = await conn.fetch("""
            SELECT
                sample_id,
                LENGTH(audio_data) as audio_bytes,
                duration_seconds,
                confidence_score,
                created_at
            FROM voice_samples
            WHERE speaker_id = 1
            ORDER BY created_at DESC
            LIMIT 10
        """)

        for i, sample in enumerate(recent, 1):
            audio_str = f"{sample['audio_bytes']} bytes" if sample['audio_bytes'] else "NO AUDIO"
            print(f"\n{i}. Sample ID {sample['sample_id']}:")
            print(f"   Audio: {audio_str}")
            print(f"   Duration: {sample['duration_seconds']:.2f}s" if sample['duration_seconds'] else "   Duration: N/A")
            print(f"   Quality: {sample['confidence_score']:.2%}" if sample['confidence_score'] else "   Quality: N/A")
            print(f"   Created: {sample['created_at']}")

        # 3. Check if profile has acoustic features
        print("\n3️⃣  PROFILE ACOUSTIC FEATURES:")
        print("-" * 40)

        profile = await conn.fetchrow("""
            SELECT
                acoustic_features,
                total_samples,
                enrollment_quality_score
            FROM speaker_profiles
            WHERE speaker_id = 1
        """)

        if profile['acoustic_features']:
            print("✅ Acoustic features are stored!")
            # Count features
            features = profile['acoustic_features']
            if isinstance(features, dict):
                print(f"   Feature count: {len(features)}")
                if 'pitch_mean' in features:
                    print(f"   Pitch mean: {features['pitch_mean']:.1f} Hz")
        else:
            print("❌ No acoustic features stored")

        print(f"\nProfile stats:")
        print(f"   Total samples in profile: {profile['total_samples']}")
        print(f"   Quality score: {profile['enrollment_quality_score']:.2%}")

    finally:
        await conn.close()

    print("\n" + "="*80)
    print("DATABASE CHECK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(check_voice_samples())