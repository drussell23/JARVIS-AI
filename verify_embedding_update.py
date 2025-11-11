#!/usr/bin/env python3
"""Verify the embedding was updated and test it."""

import asyncio
import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def verify_update():
    """Verify embedding update and test it."""

    print("\n" + "="*80)
    print("VERIFYING EMBEDDING UPDATE")
    print("="*80)

    # Check database directly
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
        # Check the stored embedding
        print("\n1️⃣ Checking database embedding...")
        result = await conn.fetchrow("""
            SELECT
                voiceprint_embedding,
                enrollment_quality_score,
                last_updated
            FROM speaker_profiles
            WHERE speaker_id = 1
        """)

        if result and result['voiceprint_embedding']:
            emb = np.frombuffer(result['voiceprint_embedding'], dtype=np.float32)
            print(f"   ✅ Embedding in DB: shape {emb.shape}, norm {np.linalg.norm(emb):.4f}")
            print(f"   Quality score: {result['enrollment_quality_score']:.2%}")
            print(f"   Last updated: {result['last_updated']}")
        else:
            print("   ❌ No embedding in database!")

        # Now check what the service is loading
        print("\n2️⃣ Checking what service loads...")
        from backend.voice.speaker_verification_service import SpeakerVerificationService

        service = SpeakerVerificationService()
        await service.initialize()

        print(f"   Service initialized with {len(service.speaker_profiles)} profiles")

        if "Derek J. Russell" in service.speaker_profiles:
            profile = service.speaker_profiles["Derek J. Russell"]
            if 'embedding' in profile:
                loaded_emb = profile['embedding']
                if isinstance(loaded_emb, np.ndarray):
                    print(f"   ✅ Loaded embedding: shape {loaded_emb.shape}, norm {np.linalg.norm(loaded_emb):.4f}")

                    # Compare with DB
                    if result and result['voiceprint_embedding']:
                        db_emb = np.frombuffer(result['voiceprint_embedding'], dtype=np.float32)
                        if np.array_equal(db_emb, loaded_emb):
                            print("   ✅ Loaded embedding MATCHES database")
                        else:
                            print("   ❌ Loaded embedding DIFFERS from database!")
                            similarity = np.dot(db_emb, loaded_emb) / (np.linalg.norm(db_emb) * np.linalg.norm(loaded_emb))
                            print(f"      Similarity: {similarity:.2%}")
                else:
                    print(f"   ❌ Embedding is not numpy array: {type(loaded_emb)}")
            else:
                print("   ❌ No embedding in loaded profile!")
        else:
            print("   ❌ Profile not loaded!")

        # Test with random audio
        print("\n3️⃣ Testing with random audio...")
        test_audio = np.random.randn(16000).astype(np.float32)
        result = await service.verify_speaker(test_audio.tobytes())
        print(f"   Confidence with random: {result.get('confidence', 0):.2%}")

        # The issue might be the embedding field name
        print("\n4️⃣ Checking profile structure...")
        if "Derek J. Russell" in service.speaker_profiles:
            profile = service.speaker_profiles["Derek J. Russell"]
            print("   Profile keys:", list(profile.keys()))
            if 'voiceprint_embedding' in profile:
                print("   ⚠️ Profile has 'voiceprint_embedding' but may need 'embedding'")

    finally:
        await conn.close()

    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(verify_update())