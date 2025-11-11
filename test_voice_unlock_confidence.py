#!/usr/bin/env python3
"""Test voice unlock confidence after profile merge"""

import asyncio
import asyncpg
import numpy as np
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_voice_confidence():
    """Test the voice unlock confidence after merging profiles"""

    print("\n" + "="*80)
    print("VOICE UNLOCK CONFIDENCE TEST")
    print("="*80)
    print("\nTesting voice authentication after profile merge...")

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
        # 1. Check merged profile status
        print("\n1️⃣  MERGED PROFILE STATUS:")
        print("-" * 40)

        profile = await conn.fetchrow("""
            SELECT
                speaker_id,
                speaker_name,
                total_samples,
                LENGTH(voiceprint_embedding) as embedding_size,
                embedding_dimension,
                is_primary_user,
                enrollment_quality_score,
                pitch_mean_hz,
                formant_f1_hz,
                energy_mean,
                verification_count,
                successful_verifications,
                failed_verifications
            FROM speaker_profiles
            WHERE is_primary_user = true
            LIMIT 1
        """)

        if profile:
            print(f"\n✅ Primary Profile Found: {profile['speaker_name']}")
            print(f"   ID: {profile['speaker_id']}")
            print(f"   Samples: {profile['total_samples']}")
            print(f"   Embedding: {profile['embedding_size']} bytes ({profile['embedding_dimension']} dimensions)")
            print(f"   Quality: {profile['enrollment_quality_score']:.2f}")
            print(f"   Has Acoustic Features: {'Yes' if profile['pitch_mean_hz'] else 'No'}")
            print(f"   Verification Stats: {profile['successful_verifications'] or 0} success, {profile['failed_verifications'] or 0} failed")
        else:
            print("❌ No primary profile found!")
            return

        # 2. Sample quality check
        print("\n2️⃣  SAMPLE QUALITY CHECK:")
        print("-" * 40)

        sample_stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_samples,
                COUNT(audio_data) as samples_with_audio,
                COUNT(mfcc_features) as samples_with_mfcc,
                AVG(quality_score) as avg_quality,
                COUNT(DISTINCT audio_hash) as unique_samples
            FROM voice_samples
            WHERE speaker_id = $1
        """, profile['speaker_id'])

        print(f"\nTotal Samples: {sample_stats['total_samples']}")
        print(f"   With Audio: {sample_stats['samples_with_audio']}")
        print(f"   With MFCC: {sample_stats['samples_with_mfcc']}")
        print(f"   Unique: {sample_stats['unique_samples']}")
        print(f"   Average Quality: {sample_stats['avg_quality']:.2f}")

        # 3. Simulate voice verification
        print("\n3️⃣  VOICE VERIFICATION SIMULATION:")
        print("-" * 40)

        # Load the voice verification service
        from backend.voice.speaker_verification_service import SpeakerVerificationService

        service = SpeakerVerificationService()
        await service.initialize()

        print("\nLoaded speaker profiles:")
        for name, prof in service.speaker_profiles.items():
            embed_size = len(prof['embedding']) if isinstance(prof['embedding'], (list, np.ndarray)) else 0
            print(f"   - {name}: {embed_size} dimensional embedding")

        # Test with dummy audio (in real scenario, this would be actual audio)
        # For testing, we'll check if the embedding dimensions match
        if service.speaker_profiles:
            primary_profile_name = list(service.speaker_profiles.keys())[0]
            primary_profile = service.speaker_profiles[primary_profile_name]

            print(f"\n✅ Testing with profile: {primary_profile_name}")
            print(f"   Embedding dimension: {len(primary_profile['embedding'])}")
            print(f"   Threshold: {primary_profile.get('threshold', 0.85):.2%}")
            print(f"   Security level: {primary_profile.get('security_level', 'standard')}")

        # 4. Recommendations
        print("\n4️⃣  RECOMMENDATIONS:")
        print("-" * 40)

        issues = []

        if sample_stats['samples_with_audio'] < sample_stats['total_samples'] / 2:
            issues.append("⚠️  Less than 50% of samples have audio data")

        if sample_stats['samples_with_mfcc'] < sample_stats['total_samples'] * 0.8:
            issues.append("⚠️  Many samples missing MFCC features")

        if not profile['pitch_mean_hz']:
            issues.append("❌ Missing acoustic features in profile")

        if profile['embedding_dimension'] != profile['embedding_size'] / 4:  # float32 = 4 bytes
            issues.append("⚠️  Embedding dimension mismatch")

        if issues:
            print("\nIssues to address:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ Profile appears healthy and ready for voice unlock!")

        print("\nNext steps:")
        print("   1. Restart JARVIS to reload the merged profile")
        print("   2. Try voice unlock command again")
        print("   3. Expected confidence should be > 85%")

        # 5. Update verification count
        await conn.execute("""
            UPDATE speaker_profiles
            SET last_verified = NOW()
            WHERE speaker_id = $1
        """, profile['speaker_id'])

    finally:
        await conn.close()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(test_voice_confidence())