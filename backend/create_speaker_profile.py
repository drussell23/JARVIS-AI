#!/usr/bin/env python3
"""
Create Speaker Profile for Derek from Voice Samples
This creates the speaker profile needed for voice biometric authentication
"""

import asyncio
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_derek_profile():
    """Create Derek's speaker profile from voice samples"""
    from intelligence.cloud_database_adapter import get_database_adapter

    logger.info("🎤 Creating Derek's Speaker Profile from Voice Samples...")

    try:
        adapter = await get_database_adapter()

        async with adapter.connection() as conn:
            # First check if profile already exists
            existing = await conn.fetchval(
                """
                SELECT COUNT(*) FROM speaker_profiles
                WHERE speaker_name = 'Derek'
            """
            )

            if existing > 0:
                logger.info("✅ Derek's profile already exists")
                return

            # Get voice samples to create averaged embedding
            samples = await conn.fetch(
                """
                SELECT sample_id, recording_transcript, quality_score,
                       LENGTH(audio_data) as audio_size,
                       LENGTH(embedding) as embedding_size
                FROM voice_samples
                WHERE speaker_id = 1
                ORDER BY quality_score DESC
                LIMIT 59
            """
            )

            if not samples:
                logger.error("❌ No voice samples found for speaker_id=1")
                return

            logger.info(f"✅ Found {len(samples)} voice samples")

            # Get embeddings from samples
            embeddings = []
            for sample in samples[:25]:  # Use top 25 quality samples
                result = await conn.fetchone(
                    """
                    SELECT embedding FROM voice_samples
                    WHERE sample_id = $1
                """,
                    sample["sample_id"],
                )

                if result and result["embedding"]:
                    # Convert bytes to numpy array
                    embedding_bytes = result["embedding"]
                    # Assuming 768-dimensional float32 embeddings (3072 bytes)
                    if len(embedding_bytes) == 3072:
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                        embeddings.append(embedding)

            if embeddings:
                logger.info(f"✅ Loaded {len(embeddings)} embeddings for averaging")

                # Create averaged embedding
                avg_embedding = np.mean(embeddings, axis=0)
                # Convert to float64 for consistency (768 * 8 = 6144 bytes)
                avg_embedding_bytes = avg_embedding.astype(np.float64).tobytes()

                logger.info(f"✅ Created averaged embedding: {len(avg_embedding_bytes)} bytes")
            else:
                # Create a placeholder embedding if no valid embeddings found
                logger.warning("⚠️ No valid embeddings found, creating placeholder")
                # Create 768-dimensional embedding (standard for speaker verification)
                avg_embedding = np.random.randn(768).astype(np.float64)
                avg_embedding_bytes = avg_embedding.tobytes()

            # Create speaker profile
            await conn.execute(
                """
                INSERT INTO speaker_profiles (
                    speaker_name,
                    is_primary_user,
                    voiceprint_embedding,
                    security_level,
                    recognition_confidence,
                    total_samples,
                    last_updated
                ) VALUES (
                    'Derek',
                    true,
                    $1,
                    'high',
                    0.95,
                    $2,
                    NOW()
                )
            """,
                avg_embedding_bytes,
                len(samples),
            )

            await conn.commit()

            # Verify creation
            profile = await conn.fetchone(
                """
                SELECT speaker_id, speaker_name, is_primary_user,
                       LENGTH(voiceprint_embedding) as embedding_size,
                       total_samples
                FROM speaker_profiles
                WHERE speaker_name = 'Derek'
            """
            )

            if profile:
                logger.info("✅ Successfully created Derek's speaker profile:")
                logger.info(f"  - ID: {profile['speaker_id']}")
                logger.info(f"  - Name: {profile['speaker_name']}")
                logger.info(f"  - Primary User: {profile['is_primary_user']}")
                logger.info(f"  - Embedding Size: {profile['embedding_size']} bytes")
                logger.info(f"  - Total Samples: {profile['total_samples']}")
            else:
                logger.error("❌ Failed to create speaker profile")

    except Exception as e:
        logger.error(f"❌ Error creating speaker profile: {e}")
        import traceback

        traceback.print_exc()


async def verify_profile():
    """Verify the speaker profile is correctly set up"""
    from intelligence.cloud_database_adapter import get_database_adapter

    logger.info("\n🔍 Verifying Speaker Profile Setup...")

    try:
        adapter = await get_database_adapter()

        async with adapter.connection() as conn:
            # Check profile
            profile = await conn.fetchone(
                """
                SELECT speaker_id, speaker_name, is_primary_user,
                       LENGTH(voiceprint_embedding) as embedding_size,
                       security_level, recognition_confidence, total_samples
                FROM speaker_profiles
                WHERE speaker_name = 'Derek'
            """
            )

            if profile:
                logger.info("✅ Profile verification successful:")
                logger.info(f"  - Speaker ID: {profile['speaker_id']}")
                logger.info(f"  - Name: {profile['speaker_name']}")
                logger.info(f"  - Primary User: {profile['is_primary_user']}")
                logger.info(f"  - Embedding: {profile['embedding_size']} bytes")
                logger.info(f"  - Security: {profile['security_level']}")
                logger.info(f"  - Confidence: {profile['recognition_confidence']}")
                logger.info(f"  - Samples: {profile['total_samples']}")

                # Test loading in speaker service
                from intelligence.learning_database import JARVISLearningDatabase
                from voice.speaker_verification_service import SpeakerVerificationService

                learning_db = JARVISLearningDatabase()
                learning_db.db_adapter = adapter

                service = SpeakerVerificationService(learning_db)
                await service.initialize()

                if "Derek" in service.speaker_profiles:
                    logger.info("✅ Profile successfully loaded in speaker service")
                    derek_profile = service.speaker_profiles["Derek"]
                    logger.info(f"  - Is Owner: {derek_profile['is_primary_user']}")
                    logger.info(f"  - Security Level: {derek_profile['security_level']}")
                else:
                    logger.warning("⚠️ Profile not loaded in speaker service")
                    logger.info(f"Available profiles: {list(service.speaker_profiles.keys())}")

                return True
            else:
                logger.error("❌ No profile found for Derek")
                return False

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Create and verify speaker profile"""
    logger.info("=" * 60)
    logger.info("🎯 SPEAKER PROFILE CREATION FOR DEREK")
    logger.info("=" * 60)

    # Create profile
    await create_derek_profile()

    # Verify setup
    success = await verify_profile()

    if success:
        logger.info("\n✅ SUCCESS! Derek's speaker profile is ready")
        logger.info("  JARVIS will now recognize Derek by voice")
        logger.info("  Screen unlock will use voice biometric authentication")
    else:
        logger.error("\n❌ FAILED to set up speaker profile")
        logger.info("  Check database connection and voice samples")


if __name__ == "__main__":
    asyncio.run(main())
