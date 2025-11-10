#!/usr/bin/env python3
"""
Database Migration: Add audio_data column to voice_samples table

This migration adds raw audio storage to support robust embedding reconstruction
when model dimensions change. Works with both SQLite and CloudSQL (PostgreSQL).

Usage:
    python3 backend/migrate_voice_storage.py
"""
import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from intelligence.learning_database import get_learning_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def migrate_database():
    """Add audio_data column to voice_samples table if it doesn't exist"""
    logger.info("🔄 Starting voice storage migration...")

    try:
        # Get database instance (could be SQLite or CloudSQL)
        db = await get_learning_database()

        if not db or not db._initialized:
            logger.error("❌ Failed to initialize database")
            return False

        # Check if this is CloudSQL or SQLite
        is_cloud = hasattr(db.db, 'is_cloud') and db.db.is_cloud
        db_type = "CloudSQL (PostgreSQL)" if is_cloud else "SQLite"

        logger.info(f"📊 Connected to {db_type}")

        async with db.db.cursor() as cursor:
            # Check if audio_data column already exists
            try:
                if is_cloud:
                    # PostgreSQL: Check information_schema
                    await cursor.execute("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name='voice_samples'
                        AND column_name='audio_data'
                    """)
                else:
                    # SQLite: Check pragma
                    await cursor.execute("PRAGMA table_info(voice_samples)")

                result = await cursor.fetchall()

                # Check if column exists
                if is_cloud:
                    column_exists = len(result) > 0
                else:
                    # SQLite returns column info: (cid, name, type, notnull, dflt_value, pk)
                    column_exists = any(row[1] == 'audio_data' for row in result)

                if column_exists:
                    logger.info("✅ audio_data column already exists - migration not needed")
                    return True

                logger.info("➕ Adding audio_data column to voice_samples table...")

                # Add the column based on database type
                if is_cloud:
                    await cursor.execute("""
                        ALTER TABLE voice_samples
                        ADD COLUMN audio_data BYTEA
                    """)
                else:
                    await cursor.execute("""
                        ALTER TABLE voice_samples
                        ADD COLUMN audio_data BLOB
                    """)

                await db.db.commit()

                logger.info("✅ Successfully added audio_data column")
                logger.info("📝 Note: Existing voice samples will have NULL audio_data")
                logger.info("   → New samples will automatically store raw audio")
                logger.info("   → Old profiles will use fallback migration (padding/truncation)")

                return True

            except Exception as e:
                logger.error(f"❌ Migration failed: {e}", exc_info=True)
                return False

    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}", exc_info=True)
        return False


async def verify_migration():
    """Verify the migration was successful"""
    logger.info("\n🔍 Verifying migration...")

    try:
        db = await get_learning_database()

        async with db.db.cursor() as cursor:
            # Try to query the new column
            await cursor.execute("""
                SELECT COUNT(*)
                FROM voice_samples
                WHERE audio_data IS NOT NULL
            """)
            result = await cursor.fetchone()
            count = result[0] if result else 0

            logger.info(f"✅ Migration verified successfully")
            logger.info(f"   → {count} voice samples have audio_data stored")

            # Get total count
            await cursor.execute("SELECT COUNT(*) FROM voice_samples")
            total_result = await cursor.fetchone()
            total = total_result[0] if total_result else 0

            if total > 0:
                logger.info(f"   → {total - count} legacy samples (will use fallback migration)")

            return True

    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False


async def main():
    """Run the migration"""
    logger.info("=" * 60)
    logger.info("Voice Storage Migration Tool")
    logger.info("=" * 60)

    success = await migrate_database()

    if success:
        await verify_migration()
        logger.info("\n✅ Migration completed successfully!")
        logger.info("\n📋 Next steps:")
        logger.info("   1. Existing voice profiles will continue to work")
        logger.info("   2. New voice enrollments will store raw audio")
        logger.info("   3. Re-enroll speakers to enable advanced reconstruction")
        return 0
    else:
        logger.error("\n❌ Migration failed - please check logs")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
