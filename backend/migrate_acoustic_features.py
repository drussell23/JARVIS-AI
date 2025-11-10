#!/usr/bin/env python3
"""
🔬 ACOUSTIC FEATURES DATABASE MIGRATION
═══════════════════════════════════════════════════════════════════════════════

Migrates speaker_profiles table to include comprehensive biometric features:
- Pitch statistics (mean, std, range, min, max)
- Formant frequencies (F1-F4 with variance)
- Spectral features (centroid, rolloff, flux, entropy, flatness, bandwidth)
- Temporal characteristics (speaking rate, pauses, syllables, articulation)
- Energy features (mean, std, dynamic range)
- Voice quality (jitter, shimmer, HNR)
- Statistical modeling (covariance matrix, feature statistics)

Fully async, handles both CloudSQL (PostgreSQL) and local (SQLite).
Zero downtime, backward compatible.

Author: Claude Code + Derek J. Russell
Version: 1.0.0
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from intelligence.learning_database import get_learning_database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AcousticFeaturesMigration:
    """
    🔬 Comprehensive database migration for acoustic biometric features
    """

    def __init__(self, db):
        self.db = db
        self.is_postgres = hasattr(self.db.db, 'server_version')

    async def detect_database_type(self) -> str:
        """Detect if we're using SQLite or PostgreSQL"""
        try:
            async with self.db.db.cursor() as cursor:
                await cursor.execute("SELECT version()")
                result = await cursor.fetchone()
                version = result[0] if isinstance(result, tuple) else result['version']

                if 'PostgreSQL' in version:
                    return 'postgresql'
                elif 'SQLite' in version:
                    return 'sqlite'
                else:
                    return 'unknown'
        except Exception:
            # Assume SQLite if version() doesn't work
            return 'sqlite'

    async def check_column_exists(self, table: str, column: str) -> bool:
        """Check if a column exists in a table"""
        try:
            async with self.db.db.cursor() as cursor:
                if self.is_postgres:
                    await cursor.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = %s AND column_name = %s
                        """,
                        (table, column)
                    )
                else:
                    await cursor.execute(f"PRAGMA table_info({table})")
                    columns = await cursor.fetchall()
                    for col in columns:
                        col_name = col[1] if isinstance(col, tuple) else col['name']
                        if col_name == column:
                            return True
                    return False

                result = await cursor.fetchone()
                return result is not None

        except Exception as e:
            logger.warning(f"Error checking column {column}: {e}")
            return False

    async def add_column_safe(self, table: str, column: str, column_type: str, default: str = None):
        """Safely add a column if it doesn't exist"""
        try:
            exists = await self.check_column_exists(table, column)

            if exists:
                logger.debug(f"   ✓ Column {column} already exists")
                return True

            async with self.db.db.cursor() as cursor:
                default_clause = f" DEFAULT {default}" if default else ""

                if self.is_postgres:
                    sql = f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}{default_clause}"
                else:
                    sql = f"ALTER TABLE {table} ADD COLUMN {column} {column_type}{default_clause}"

                await cursor.execute(sql)
                await self.db.db.commit()
                logger.info(f"   ✅ Added column: {column}")
                return True

        except Exception as e:
            logger.error(f"   ❌ Failed to add column {column}: {e}")
            return False

    async def migrate(self) -> bool:
        """
        Run the complete migration

        Returns:
            bool: True if migration successful, False otherwise
        """
        try:
            logger.info("=" * 80)
            logger.info("🔬 ACOUSTIC FEATURES DATABASE MIGRATION")
            logger.info("=" * 80)

            # Detect database type
            db_type = await self.detect_database_type()
            logger.info(f"\n📊 Database Type: {db_type.upper()}")

            blob_type = "BYTEA" if db_type == 'postgresql' else "BLOB"

            logger.info("\n🎵 Adding Pitch Features...")
            await self.add_column_safe('speaker_profiles', 'pitch_mean_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'pitch_std_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'pitch_range_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'pitch_min_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'pitch_max_hz', 'REAL')

            logger.info("\n🎼 Adding Formant Features...")
            await self.add_column_safe('speaker_profiles', 'formant_f1_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f1_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f2_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f2_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f3_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f3_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f4_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'formant_f4_std', 'REAL')

            logger.info("\n📊 Adding Spectral Features...")
            await self.add_column_safe('speaker_profiles', 'spectral_centroid_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_centroid_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_rolloff_hz', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_rolloff_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_flux', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_flux_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_entropy', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_entropy_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_flatness', 'REAL')
            await self.add_column_safe('speaker_profiles', 'spectral_bandwidth_hz', 'REAL')

            logger.info("\n⏱️ Adding Temporal Features...")
            await self.add_column_safe('speaker_profiles', 'speaking_rate_wpm', 'REAL')
            await self.add_column_safe('speaker_profiles', 'speaking_rate_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'pause_ratio', 'REAL')
            await self.add_column_safe('speaker_profiles', 'pause_ratio_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'syllable_rate', 'REAL')
            await self.add_column_safe('speaker_profiles', 'articulation_rate', 'REAL')

            logger.info("\n🎚️ Adding Energy Features...")
            await self.add_column_safe('speaker_profiles', 'energy_mean', 'REAL')
            await self.add_column_safe('speaker_profiles', 'energy_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'energy_dynamic_range_db', 'REAL')

            logger.info("\n🔊 Adding Voice Quality Features...")
            await self.add_column_safe('speaker_profiles', 'jitter_percent', 'REAL')
            await self.add_column_safe('speaker_profiles', 'jitter_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'shimmer_percent', 'REAL')
            await self.add_column_safe('speaker_profiles', 'shimmer_std', 'REAL')
            await self.add_column_safe('speaker_profiles', 'harmonic_to_noise_ratio_db', 'REAL')
            await self.add_column_safe('speaker_profiles', 'hnr_std', 'REAL')

            logger.info("\n📈 Adding Statistical Features...")
            await self.add_column_safe('speaker_profiles', 'feature_covariance_matrix', blob_type)
            await self.add_column_safe('speaker_profiles', 'feature_statistics', 'JSON' if db_type == 'postgresql' else 'TEXT')

            logger.info("\n🎯 Adding Quality & Security Metrics...")
            await self.add_column_safe('speaker_profiles', 'enrollment_quality_score', 'REAL')
            await self.add_column_safe('speaker_profiles', 'feature_extraction_version', 'TEXT', "'v1.0'")
            await self.add_column_safe('speaker_profiles', 'embedding_dimension', 'INTEGER')
            await self.add_column_safe('speaker_profiles', 'verification_count', 'INTEGER', '0')
            await self.add_column_safe('speaker_profiles', 'successful_verifications', 'INTEGER', '0')
            await self.add_column_safe('speaker_profiles', 'failed_verifications', 'INTEGER', '0')
            await self.add_column_safe('speaker_profiles', 'last_verified', 'TIMESTAMP')

            logger.info("\n✅ Migration completed successfully!")
            logger.info("=" * 80)

            # Verify migration
            logger.info("\n🔍 Verifying migration...")
            async with self.db.db.cursor() as cursor:
                if db_type == 'postgresql':
                    await cursor.execute(
                        """
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_name = 'speaker_profiles'
                        ORDER BY ordinal_position
                        """
                    )
                    columns = await cursor.fetchall()
                    column_names = [col[0] if isinstance(col, tuple) else col['column_name'] for col in columns]
                else:
                    await cursor.execute("PRAGMA table_info(speaker_profiles)")
                    columns = await cursor.fetchall()
                    column_names = [col[1] if isinstance(col, tuple) else col['name'] for col in columns]

                logger.info(f"\n📋 Total columns in speaker_profiles: {len(column_names)}")

                # Check key acoustic features
                key_features = [
                    'pitch_mean_hz', 'formant_f1_hz', 'spectral_centroid_hz',
                    'jitter_percent', 'shimmer_percent', 'harmonic_to_noise_ratio_db',
                    'feature_covariance_matrix', 'enrollment_quality_score'
                ]

                missing = [f for f in key_features if f not in column_names]

                if missing:
                    logger.warning(f"⚠️ Missing columns: {missing}")
                    return False
                else:
                    logger.info("✅ All key acoustic features present!")
                    return True

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}", exc_info=True)
            return False


async def main():
    """Run the migration"""
    logger.info("\n🚀 Starting Acoustic Features Migration...\n")

    try:
        # Get database instance
        db = await get_learning_database()

        # Run migration
        migration = AcousticFeaturesMigration(db)
        success = await migration.migrate()

        # Close database
        await db.close()

        if success:
            logger.info("\n" + "=" * 80)
            logger.info("✅ MIGRATION COMPLETE")
            logger.info("=" * 80)
            logger.info("\n💡 Next steps:")
            logger.info("   1. Re-run enrollment: python3 backend/quick_voice_enhancement.py")
            logger.info("   2. Or update existing profile: python3 backend/update_acoustic_features.py")
            logger.info("   3. Test verification: Say 'unlock my screen'\n")
            return 0
        else:
            logger.error("\n❌ Migration failed - check logs above")
            return 1

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
