#!/usr/bin/env python3
"""
Cleanup script for invalid speaker profiles in the database.

This script removes:
- Profiles with no voiceprint embedding (incomplete enrollments)
- Profiles with placeholder names like 'unknown', 'test', etc.

Usage:
    python cleanup_invalid_profiles.py [--dry-run] [--force]

Options:
    --dry-run   Show what would be deleted without actually deleting
    --force     Skip confirmation prompt
"""

import asyncio
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


PLACEHOLDER_NAMES = {'unknown', 'test', 'placeholder', '', 'none', 'null'}


async def get_invalid_profiles(db) -> list:
    """Get all invalid profiles that should be cleaned up."""
    invalid_profiles = []

    async with db.db.cursor() as cursor:
        # Find profiles with no embedding
        await cursor.execute("""
            SELECT speaker_id, speaker_name,
                   voiceprint_embedding IS NULL as no_embedding,
                   total_samples,
                   created_at
            FROM speaker_profiles
            WHERE voiceprint_embedding IS NULL
               OR speaker_name IS NULL
               OR LOWER(speaker_name) IN ('unknown', 'test', 'placeholder', '', 'none', 'null')
        """)
        rows = await cursor.fetchall()

        for row in rows:
            invalid_profiles.append({
                'speaker_id': row['speaker_id'],
                'speaker_name': row['speaker_name'] or '<NULL>',
                'no_embedding': row['no_embedding'],
                'total_samples': row['total_samples'] or 0,
                'created_at': row['created_at'],
                'reason': 'no_embedding' if row['no_embedding'] else 'placeholder_name'
            })

    return invalid_profiles


async def delete_profiles(db, profile_ids: list, dry_run: bool = False) -> int:
    """Delete profiles by their IDs."""
    if dry_run:
        logger.info(f"[DRY RUN] Would delete {len(profile_ids)} profiles")
        return 0

    if not profile_ids:
        return 0

    async with db.db.cursor() as cursor:
        # Delete related voice samples first
        placeholders = ','.join(['%s'] * len(profile_ids))
        await cursor.execute(
            f"DELETE FROM voice_samples WHERE speaker_id IN ({placeholders})",
            profile_ids
        )
        samples_deleted = cursor.rowcount
        logger.info(f"Deleted {samples_deleted} related voice samples")

        # Delete the profiles
        await cursor.execute(
            f"DELETE FROM speaker_profiles WHERE speaker_id IN ({placeholders})",
            profile_ids
        )
        profiles_deleted = cursor.rowcount

        await db.db.commit()
        logger.info(f"Deleted {profiles_deleted} invalid profiles")

        return profiles_deleted


async def main():
    parser = argparse.ArgumentParser(description='Clean up invalid speaker profiles')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt')
    args = parser.parse_args()

    try:
        from intelligence.learning_database import JARVISLearningDatabase
    except ImportError:
        logger.error("Could not import JARVISLearningDatabase. Make sure you're running from the backend directory.")
        sys.exit(1)

    db = JARVISLearningDatabase()
    await db.initialize()

    try:
        # Find invalid profiles
        logger.info("🔍 Searching for invalid profiles...")
        invalid_profiles = await get_invalid_profiles(db)

        if not invalid_profiles:
            logger.info("✅ No invalid profiles found! Database is clean.")
            return

        # Display what was found
        logger.info(f"\n📋 Found {len(invalid_profiles)} invalid profile(s):\n")
        print("-" * 80)
        print(f"{'ID':<8} {'Name':<25} {'Reason':<20} {'Samples':<10} {'Created'}")
        print("-" * 80)

        for profile in invalid_profiles:
            print(f"{profile['speaker_id']:<8} {profile['speaker_name']:<25} "
                  f"{profile['reason']:<20} {profile['total_samples']:<10} "
                  f"{profile['created_at']}")

        print("-" * 80)
        print()

        # Confirm deletion
        if not args.dry_run and not args.force:
            response = input("Do you want to delete these profiles? (yes/no): ")
            if response.lower() not in ('yes', 'y'):
                logger.info("❌ Cancelled by user")
                return

        # Delete profiles
        profile_ids = [p['speaker_id'] for p in invalid_profiles]
        deleted = await delete_profiles(db, profile_ids, dry_run=args.dry_run)

        if args.dry_run:
            logger.info(f"🔍 [DRY RUN] Would have deleted {len(profile_ids)} profiles")
        else:
            logger.info(f"✅ Successfully deleted {deleted} invalid profiles")

    finally:
        await db.close()


if __name__ == '__main__':
    asyncio.run(main())
