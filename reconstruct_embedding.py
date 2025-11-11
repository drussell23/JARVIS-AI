#!/usr/bin/env python3
"""Reconstruct embedding from recent voice samples with audio data."""

import asyncio
import sys
import os
import numpy as np

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def reconstruct_embedding():
    """Reconstruct embedding from recent audio samples."""

    print("\n" + "="*80)
    print("RECONSTRUCTING EMBEDDING FROM RECENT SAMPLES")
    print("="*80)

    from backend.voice.speaker_verification_service import SpeakerVerificationService

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n📊 Current Status:")
    print(f"   Model dimension: {service.current_model_dimension}")
    print(f"   Profiles loaded: {len(service.speaker_profiles)}")

    # Get Derek's profile
    profile = service.speaker_profiles.get("Derek J. Russell")
    if not profile:
        print("❌ Profile not found!")
        return

    speaker_id = profile.get("speaker_id", 1)
    print(f"   Speaker ID: {speaker_id}")

    # Try to reconstruct embedding from audio samples
    print("\n🔄 Attempting to reconstruct embedding from audio samples...")

    # This uses the _reconstruct_embedding_from_samples method
    reconstructed = await service._reconstruct_embedding_from_samples("Derek J. Russell", speaker_id)

    if reconstructed is not None:
        print("\n✅ Successfully reconstructed embedding!")
        print(f"   Shape: {reconstructed.shape}")
        print(f"   Norm: {np.linalg.norm(reconstructed):.4f}")

        # Compare with existing
        existing = profile["embedding"]
        if isinstance(existing, np.ndarray):
            similarity = np.dot(existing, reconstructed) / (np.linalg.norm(existing) * np.linalg.norm(reconstructed))
            print(f"\n   Similarity to old embedding: {similarity:.2%}")

        # Update the profile
        print("\n💾 Updating profile with reconstructed embedding...")

        # Update in memory
        profile["embedding"] = reconstructed
        profile["voiceprint_embedding"] = reconstructed.tobytes()

        # Update in database
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
            await conn.execute("""
                UPDATE speaker_profiles
                SET embedding_data = $1,
                    last_updated = CURRENT_TIMESTAMP
                WHERE speaker_id = $2
            """, reconstructed.tobytes(), speaker_id)

            print("✅ Database updated with new embedding")

            # Verify it worked
            test_audio = np.random.randn(16000).astype(np.float32)
            test_result = await service.verify_speaker(test_audio.tobytes())
            print(f"\n📊 Test with random audio:")
            print(f"   Confidence: {test_result.get('confidence', 0):.2%}")

        finally:
            await conn.close()

        print("\n🎯 NEXT STEP:")
        print("-" * 40)
        print("Restart JARVIS to load the new embedding:")
        print("   python start_system.py --restart")
        print("\nThen test voice unlock:")
        print("   Say: 'unlock my screen'")

    else:
        print("\n❌ Could not reconstruct embedding")
        print("   This means audio samples may not have audio_data stored")
        print("\n   Solution: Record new samples with audio")
        print("   python backend/quick_voice_enhancement.py")

    print("\n" + "="*80)

if __name__ == "__main__":
    asyncio.run(reconstruct_embedding())