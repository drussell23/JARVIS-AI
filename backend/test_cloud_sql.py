#!/usr/bin/env python3
"""
Test Cloud SQL connection directly
"""
import asyncio
import logging

import asyncpg

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_connection():
    """Test direct connection to Cloud SQL via proxy"""
    import os

    config = {
        "host": "127.0.0.1",
        "port": 5432,
        "database": "jarvis_learning",
        "user": "jarvis",
        "password": os.getenv("JARVIS_DB_PASSWORD", "YOUR_PASSWORD_HERE"),
    }

    logger.info(f"Connecting to Cloud SQL at {config['host']}:{config['port']}")
    logger.info(f"Database: {config['database']}, User: {config['user']}")

    try:
        # Test connection
        conn = await asyncpg.connect(**config)
        logger.info("✅ Connection successful!")

        # Test query
        result = await conn.fetchval("SELECT version();")
        logger.info(f"Database version: {result}")

        # Check for speaker_profiles table
        exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'speaker_profiles'
            );
        """
        )

        if exists:
            logger.info("✅ speaker_profiles table exists")

            # Count profiles
            count = await conn.fetchval("SELECT COUNT(*) FROM speaker_profiles;")
            logger.info(f"Total speaker profiles: {count}")

            # Look for Derek's profile
            derek = await conn.fetchrow(
                """
                SELECT speaker_id, speaker_name, is_primary_user,
                       LENGTH(voiceprint_embedding) as embedding_size
                FROM speaker_profiles
                WHERE speaker_name ILIKE '%derek%'
                LIMIT 1;
            """
            )

            if derek:
                logger.info(f"✅ Found Derek's profile:")
                logger.info(f"  - ID: {derek['speaker_id']}")
                logger.info(f"  - Name: {derek['speaker_name']}")
                logger.info(f"  - Primary User: {derek['is_primary_user']}")
                logger.info(f"  - Embedding Size: {derek['embedding_size']} bytes")
            else:
                logger.warning("❌ No profile found for Derek")
        else:
            logger.warning("❌ speaker_profiles table does not exist")

            # List all tables
            tables = await conn.fetch(
                """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """
            )

            if tables:
                logger.info(f"Available tables: {[t['table_name'] for t in tables]}")
            else:
                logger.info("No tables found in database")

        # Check for voice samples table structure
        logger.info("\n🎤 Checking voice samples...")

        # First check if table exists
        vs_exists = await conn.fetchval(
            """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = 'voice_samples'
            );
        """
        )

        if vs_exists:
            # Get table columns
            columns = await conn.fetch(
                """
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'voice_samples'
                ORDER BY ordinal_position;
            """
            )

            logger.info("voice_samples table columns:")
            for col in columns:
                logger.info(f"  - {col['column_name']}: {col['data_type']}")

            # Check total samples
            sample_count = await conn.fetchval("SELECT COUNT(*) FROM voice_samples;")
            logger.info(f"\nTotal voice samples: {sample_count}")

            # Check for samples linked to speaker profile
            linked_samples = await conn.fetchval(
                """
                SELECT COUNT(*) FROM voice_samples
                WHERE speaker_id = 1;
            """
            )

            if linked_samples:
                logger.info(f"✅ Found {linked_samples} voice samples linked to Derek's profile")
            else:
                logger.warning("❌ No voice samples linked to Derek's profile")
        else:
            logger.warning("❌ voice_samples table does not exist")

        await conn.close()

    except Exception as e:
        logger.error(f"❌ Connection failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_connection())
