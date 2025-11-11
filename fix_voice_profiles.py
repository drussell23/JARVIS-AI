#!/usr/bin/env python3
"""Fix voice profile issues - consolidate and update embeddings."""

import asyncio
import asyncpg
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def fix_voice_profiles():
    """Fix the voice profile issues causing low confidence."""

    print("\n" + "="*80)
    print("VOICE PROFILE FIX - CONSOLIDATION & UPDATE")
    print("="*80)

    # Get database password
    try:
        from core.secret_manager import get_secret
        db_password = get_secret("jarvis-db-password")
    except Exception as e:
        print(f"❌ Failed to get password from Secret Manager: {e}")
        print("💡 Run: gcloud secrets versions access latest --secret='jarvis-db-password'")
        return

    # Connect to database
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        print("\n1️⃣  CURRENT STATE:")
        print("-" * 40)

        # Check current profiles
        profiles = await conn.fetch("""
            SELECT
                speaker_id,
                speaker_name,
                total_samples,
                LENGTH(voiceprint_embedding) as embedding_size,
                pitch_mean_hz,
                formant_f1_hz,
                energy_mean,
                is_primary_user
            FROM speaker_profiles
            ORDER BY speaker_id
        """)

        for p in profiles:
            print(f"\nSpeaker {p['speaker_id']}: {p['speaker_name']}")
            print(f"  Samples: {p['total_samples']}, Embedding: {p['embedding_size']} bytes")
            print(f"  Primary: {p['is_primary_user']}, Has features: {p['pitch_mean_hz'] is not None}")

        print("\n2️⃣  PROPOSED FIX:")
        print("-" * 40)
        print("\nWe need to:")
        print("1. Merge 'Derek' and 'Derek J. Russell' profiles")
        print("2. Use the acoustic features from 'Derek J. Russell' (ID: 1)")
        print("3. Keep the larger embedding from 'Derek' (ID: 2)")
        print("4. Consolidate all samples under one profile")
        print("5. Mark it as primary user")

        # Ask for confirmation
        response = input("\nProceed with fix? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted.")
            return

        print("\n3️⃣  APPLYING FIX:")
        print("-" * 40)

        # Step 1: Update Derek J. Russell profile with Derek's embedding
        print("\n✅ Copying embedding from Derek to Derek J. Russell...")
        await conn.execute("""
            UPDATE speaker_profiles
            SET voiceprint_embedding = (
                SELECT voiceprint_embedding FROM speaker_profiles WHERE speaker_id = 2
            ),
            embedding_dimension = 768,
            is_primary_user = true,
            security_level = 'high'
            WHERE speaker_id = 1
        """)

        # Step 2: Move all samples from Derek (ID: 2) to Derek J. Russell (ID: 1)
        print("✅ Moving samples from Derek to Derek J. Russell...")
        sample_count = await conn.fetchval("""
            UPDATE voice_samples
            SET speaker_id = 1
            WHERE speaker_id = 2
            RETURNING COUNT(*)
        """)
        print(f"   Moved {sample_count or 0} samples")

        # Step 3: Update sample counts
        print("✅ Updating sample counts...")
        await conn.execute("""
            UPDATE speaker_profiles
            SET total_samples = (
                SELECT COUNT(*) FROM voice_samples WHERE speaker_id = 1
            )
            WHERE speaker_id = 1
        """)

        # Step 4: Calculate and update acoustic features for all samples
        print("✅ Recalculating acoustic features from all samples...")
        features = await conn.fetchrow("""
            SELECT
                AVG(pitch_mean) as avg_pitch,
                STDDEV(pitch_mean) as std_pitch,
                AVG(energy_mean) as avg_energy,
                AVG(duration_ms) as avg_duration,
                AVG(quality_score) as avg_quality
            FROM voice_samples
            WHERE speaker_id = 1
                AND pitch_mean IS NOT NULL
        """)

        if features and features['avg_pitch']:
            await conn.execute("""
                UPDATE speaker_profiles
                SET
                    average_pitch_hz = $1,
                    pitch_std_hz = $2,
                    energy_mean = $3,
                    enrollment_quality_score = $4
                WHERE speaker_id = 1
            """, features['avg_pitch'], features['std_pitch'],
                features['avg_energy'], features['avg_quality'])

        # Step 5: Delete the duplicate Derek profile
        print("✅ Removing duplicate 'Derek' profile...")
        await conn.execute("DELETE FROM speaker_profiles WHERE speaker_id = 2")

        # Step 6: Verify the fix
        print("\n4️⃣  VERIFICATION:")
        print("-" * 40)

        final_profile = await conn.fetchrow("""
            SELECT
                speaker_id,
                speaker_name,
                total_samples,
                LENGTH(voiceprint_embedding) as embedding_size,
                pitch_mean_hz,
                energy_mean,
                is_primary_user,
                enrollment_quality_score
            FROM speaker_profiles
            WHERE speaker_id = 1
        """)

        print(f"\n✅ Final Profile: {final_profile['speaker_name']}")
        print(f"   Total Samples: {final_profile['total_samples']}")
        print(f"   Embedding Size: {final_profile['embedding_size']} bytes")
        print(f"   Has Acoustic Features: {final_profile['pitch_mean_hz'] is not None}")
        print(f"   Is Primary User: {final_profile['is_primary_user']}")
        print(f"   Quality Score: {final_profile['enrollment_quality_score']:.2f}")

        print("\n" + "="*80)
        print("✅ PROFILE FIX COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Restart JARVIS to reload profiles")
        print("2. Try voice unlock again")
        print("3. If still low confidence, run voice re-enrollment")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(fix_voice_profiles())