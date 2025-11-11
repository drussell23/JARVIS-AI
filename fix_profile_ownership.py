#!/usr/bin/env python3
"""Fix profile ownership and speaker recognition."""

import asyncio
import asyncpg
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def fix_profile_ownership():
    """Ensure Derek J. Russell is recognized as the device owner."""

    print("\n" + "="*80)
    print("FIXING PROFILE OWNERSHIP")
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
        # 1. Check current profile status
        print("\n1️⃣  CURRENT PROFILE STATUS:")
        print("-" * 40)

        profiles = await conn.fetch("""
            SELECT
                speaker_id,
                speaker_name,
                is_primary_user,
                security_level,
                LENGTH(voiceprint_embedding) as embedding_size,
                embedding_dimension,
                total_samples,
                verification_count,
                successful_verifications,
                failed_verifications
            FROM speaker_profiles
            ORDER BY speaker_id
        """)

        for profile in profiles:
            print(f"\nProfile ID {profile['speaker_id']}: {profile['speaker_name']}")
            print(f"  Primary User: {profile['is_primary_user']}")
            print(f"  Security Level: {profile['security_level']}")
            print(f"  Embedding: {profile['embedding_size']} bytes ({profile['embedding_dimension']} dim)")
            print(f"  Samples: {profile['total_samples']}")
            print(f"  Verifications: {profile['successful_verifications'] or 0} success / {profile['failed_verifications'] or 0} failed")

        # 2. Fix ownership
        print("\n2️⃣  FIXING OWNERSHIP:")
        print("-" * 40)

        # Ensure only Derek J. Russell is primary
        print("\n✅ Setting Derek J. Russell as primary owner...")

        # First, remove primary from all profiles
        await conn.execute("""
            UPDATE speaker_profiles
            SET is_primary_user = false
        """)

        # Then set Derek J. Russell as primary with high security
        result = await conn.execute("""
            UPDATE speaker_profiles
            SET
                is_primary_user = true,
                security_level = 'high',
                speaker_name = 'Derek J. Russell'
            WHERE speaker_id = 1
            RETURNING speaker_name
        """)

        # Also ensure the name is exactly "Derek J. Russell"
        await conn.execute("""
            UPDATE speaker_profiles
            SET speaker_name = 'Derek J. Russell'
            WHERE speaker_name ILIKE '%derek%'
        """)

        # 3. Verify the fix
        print("\n3️⃣  VERIFICATION:")
        print("-" * 40)

        fixed_profile = await conn.fetchrow("""
            SELECT
                speaker_id,
                speaker_name,
                is_primary_user,
                security_level,
                embedding_dimension,
                total_samples
            FROM speaker_profiles
            WHERE is_primary_user = true
        """)

        if fixed_profile:
            print(f"\n✅ Primary Owner Set:")
            print(f"   Name: {fixed_profile['speaker_name']}")
            print(f"   ID: {fixed_profile['speaker_id']}")
            print(f"   Security: {fixed_profile['security_level']}")
            print(f"   Embedding: {fixed_profile['embedding_dimension']}D")
            print(f"   Samples: {fixed_profile['total_samples']}")
        else:
            print("\n❌ No primary owner found!")

        # 4. Clear any cached profiles
        print("\n4️⃣  CLEARING CACHE:")
        print("-" * 40)

        # Touch a timestamp to force profile reload
        await conn.execute("""
            UPDATE speaker_profiles
            SET last_updated = NOW()
            WHERE speaker_id = 1
        """)

        print("✅ Cache invalidated - profiles will reload")

        print("\n" + "="*80)
        print("✅ OWNERSHIP FIXED!")
        print("="*80)
        print("\nDerek J. Russell is now set as the primary device owner.")
        print("Restart JARVIS to apply changes.")

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(fix_profile_ownership())