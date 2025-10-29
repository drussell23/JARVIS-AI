#!/usr/bin/env python3
"""
Test script to verify speaker profiles in database
"""
import asyncio
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intelligence.learning_database import JARVISLearningDatabase

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_speaker_profiles():
    """Test database connection and speaker profiles"""

    logger.info("=" * 60)
    logger.info("SPEAKER PROFILE DATABASE TEST")
    logger.info("=" * 60)

    try:
        # Initialize database
        logger.info("🔗 Initializing database connection...")
        db = JARVISLearningDatabase()
        await db.initialize()
        logger.info("✅ Database connected successfully")

        # Get all speaker profiles
        logger.info("\n📋 Fetching speaker profiles...")
        profiles = await db.get_all_speaker_profiles()

        if not profiles:
            logger.warning("⚠️ No speaker profiles found in database!")
            logger.info("\n🔍 Checking if speaker_profiles table exists...")

            # Check if table exists
            async with db.pool.acquire() as conn:
                result = await conn.fetchval(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'speaker_profiles'
                    );
                """
                )

                if result:
                    logger.info("✅ Table 'speaker_profiles' exists")

                    # Count rows
                    count = await conn.fetchval("SELECT COUNT(*) FROM speaker_profiles;")
                    logger.info(f"📊 Total rows in speaker_profiles: {count}")

                    # Check for Derek specifically
                    derek_profile = await conn.fetchrow(
                        """
                        SELECT * FROM speaker_profiles
                        WHERE speaker_name = 'Derek J. Russell'
                        OR speaker_name = 'Derek'
                        OR speaker_name ILIKE '%derek%';
                    """
                    )

                    if derek_profile:
                        logger.info(f"✅ Found Derek's profile: {dict(derek_profile)}")
                    else:
                        logger.warning("❌ No profile found for Derek")
                else:
                    logger.error("❌ Table 'speaker_profiles' does not exist!")
        else:
            logger.info(f"✅ Found {len(profiles)} speaker profiles:")

            for profile in profiles:
                speaker_name = profile.get("speaker_name", "Unknown")
                speaker_id = profile.get("speaker_id", "Unknown")
                is_primary = profile.get("is_primary_user", False)
                has_embedding = bool(profile.get("voiceprint_embedding"))

                logger.info(f"\n  👤 Speaker: {speaker_name}")
                logger.info(f"     ID: {speaker_id}")
                logger.info(f"     Primary User: {is_primary}")
                logger.info(f"     Has Embedding: {has_embedding}")

                if has_embedding:
                    embedding_bytes = profile.get("voiceprint_embedding")
                    logger.info(f"     Embedding Size: {len(embedding_bytes)} bytes")

                # Check for recorded samples
                if "sample_count" in profile:
                    logger.info(f"     Sample Count: {profile['sample_count']}")

        # Check voice_samples table
        logger.info("\n🎤 Checking voice_samples table...")
        async with db.pool.acquire() as conn:
            sample_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM voice_samples
                WHERE speaker_name = 'Derek J. Russell'
                OR speaker_name = 'Derek'
                OR speaker_name ILIKE '%derek%';
            """
            )

            logger.info(f"📊 Found {sample_count} voice samples for Derek")

            if sample_count > 0:
                # Get sample details
                samples = await conn.fetch(
                    """
                    SELECT sample_id, speaker_name, created_at,
                           LENGTH(audio_data) as audio_size,
                           LENGTH(embedding_data) as embedding_size
                    FROM voice_samples
                    WHERE speaker_name ILIKE '%derek%'
                    ORDER BY created_at DESC
                    LIMIT 5;
                """
                )

                logger.info("\n📝 Recent voice samples (up to 5):")
                for sample in samples:
                    logger.info(f"  • Sample {sample['sample_id'][:8]}...")
                    logger.info(f"    Speaker: {sample['speaker_name']}")
                    logger.info(f"    Audio Size: {sample['audio_size']} bytes")
                    logger.info(f"    Embedding Size: {sample['embedding_size']} bytes")
                    logger.info(f"    Created: {sample['created_at']}")

        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Test failed: {e}", exc_info=True)
    finally:
        if "db" in locals():
            await db.cleanup()


if __name__ == "__main__":
    asyncio.run(test_speaker_profiles())
