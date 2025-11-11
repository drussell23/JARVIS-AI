#!/usr/bin/env python3
"""Diagnose why voice verification is failing for the owner."""

import asyncio
import asyncpg
import numpy as np
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def diagnose_verification():
    """Diagnose why voice verification is failing."""

    print("\n" + "="*80)
    print("VOICE VERIFICATION DIAGNOSTIC")
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
        # 1. Check what's in the database
        print("\n1️⃣  DATABASE CHECK:")
        print("-" * 40)

        profile = await conn.fetchrow("""
            SELECT
                speaker_id,
                speaker_name,
                is_primary_user,
                LENGTH(voiceprint_embedding) as embedding_bytes,
                embedding_dimension,
                total_samples,
                enrollment_quality_score,
                successful_verifications,
                failed_verifications,
                last_updated
            FROM speaker_profiles
            WHERE is_primary_user = true
            LIMIT 1
        """)

        if profile:
            print(f"\n✅ Primary Owner Profile Found:")
            print(f"   Name: {profile['speaker_name']}")
            print(f"   ID: {profile['speaker_id']}")
            print(f"   Embedding: {profile['embedding_bytes']} bytes")
            print(f"   Dimension: {profile['embedding_dimension']}")
            print(f"   Samples: {profile['total_samples']}")
            print(f"   Quality: {profile['enrollment_quality_score']:.2f}")
            print(f"   Last Updated: {profile['last_updated']}")

            # Check if embedding dimension matches bytes
            expected_dimension = profile['embedding_bytes'] // 4  # float32 = 4 bytes
            if expected_dimension != profile['embedding_dimension']:
                print(f"\n⚠️  DIMENSION MISMATCH:")
                print(f"   Database says: {profile['embedding_dimension']}D")
                print(f"   Actual data: {expected_dimension}D")
                print(f"   This will cause verification to fail!")

        else:
            print("❌ No primary owner profile found!")
            return

        # 2. Load and check the actual embedding
        print("\n2️⃣  EMBEDDING CHECK:")
        print("-" * 40)

        embedding_data = await conn.fetchval("""
            SELECT voiceprint_embedding
            FROM speaker_profiles
            WHERE speaker_id = $1
        """, profile['speaker_id'])

        if embedding_data:
            # Try to load as float32
            try:
                embedding = np.frombuffer(embedding_data, dtype=np.float32)
                print(f"\n✅ Loaded embedding as float32: {len(embedding)} dimensions")

                # Check values
                print(f"   Min value: {np.min(embedding):.4f}")
                print(f"   Max value: {np.max(embedding):.4f}")
                print(f"   Mean: {np.mean(embedding):.4f}")
                print(f"   Std: {np.std(embedding):.4f}")

                # Check if it's normalized
                norm = np.linalg.norm(embedding)
                print(f"   L2 Norm: {norm:.4f}")
                if abs(norm - 1.0) < 0.01:
                    print(f"   ✅ Embedding is normalized")
                else:
                    print(f"   ⚠️  Embedding is NOT normalized (norm should be ~1.0)")

                # Check for NaN or Inf
                if np.any(np.isnan(embedding)):
                    print(f"   ❌ Embedding contains NaN values!")
                if np.any(np.isinf(embedding)):
                    print(f"   ❌ Embedding contains Inf values!")

            except Exception as e:
                print(f"❌ Failed to load embedding: {e}")
        else:
            print("❌ No embedding data found!")

        # 3. Check what the speechbrain model expects
        print("\n3️⃣  MODEL EXPECTATIONS:")
        print("-" * 40)

        # Load the speaker verification service to check
        from backend.voice.speaker_verification_service import SpeakerVerificationService

        service = SpeakerVerificationService()
        await service.initialize()

        print(f"\n✅ Service initialized:")
        print(f"   Current model dimension: {service.current_model_dimension}")
        print(f"   Verification threshold: {service.verification_threshold}")
        print(f"   Loaded profiles: {len(service.speaker_profiles)}")

        if service.speaker_profiles:
            for name, prof in service.speaker_profiles.items():
                print(f"\n   Profile: {name}")
                if 'embedding' in prof:
                    emb = prof['embedding']
                    if isinstance(emb, np.ndarray):
                        print(f"     Embedding shape: {emb.shape}")
                        print(f"     Embedding dtype: {emb.dtype}")
                    else:
                        print(f"     Embedding type: {type(emb)}")
                print(f"     Is Primary: {prof.get('is_primary_user', False)}")
                print(f"     Threshold: {prof.get('threshold', 'default')}")

        # 4. Test verification with dummy audio
        print("\n4️⃣  VERIFICATION TEST:")
        print("-" * 40)

        # Create test audio (random for now)
        test_audio = np.random.random(16000).astype(np.float32).tobytes()

        print("\n Testing verification with random audio...")
        result = await service.verify_speaker(test_audio)

        print(f"\n Result:")
        print(f"   Verified: {result.get('verified', False)}")
        print(f"   Confidence: {result.get('confidence', 0):.4f}")
        print(f"   Speaker Name: {result.get('speaker_name', 'None')}")
        print(f"   Is Owner: {result.get('is_owner', False)}")
        print(f"   Primary User: {result.get('primary_user', 'None')}")

        # 5. Diagnosis
        print("\n5️⃣  DIAGNOSIS:")
        print("-" * 40)

        issues = []

        # Check dimension mismatch
        if profile['embedding_dimension'] != service.current_model_dimension:
            issues.append(f"❌ Dimension mismatch: DB has {profile['embedding_dimension']}D but model expects {service.current_model_dimension}D")

        # Check if profile was loaded
        if len(service.speaker_profiles) == 0:
            issues.append("❌ No profiles loaded in service")
        elif profile['speaker_name'] not in service.speaker_profiles:
            issues.append(f"❌ Profile '{profile['speaker_name']}' not loaded in service")

        # Check embedding quality
        if embedding_data and norm < 0.1:
            issues.append(f"❌ Embedding has very low norm ({norm:.4f}) - may be corrupted")

        # Check samples
        if profile['total_samples'] == 0:
            issues.append("❌ No voice samples - needs enrollment")

        if issues:
            print("\n🔴 ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ No obvious issues found")

        print("\n📋 RECOMMENDATIONS:")
        print("   1. Check if JARVIS is properly loading profiles on startup")
        print("   2. Verify the speechbrain model is initialized correctly")
        print("   3. Consider re-enrolling voice with proper 192D embeddings")
        print("   4. Check audio input quality and format")

    finally:
        await conn.close()

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(diagnose_verification())