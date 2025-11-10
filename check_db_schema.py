#!/usr/bin/env python3
"""Check database schema."""

import asyncio
import asyncpg
import sys
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def check_schema():
    """Check the database schema."""

    # Get password from GCP Secret Manager
    try:
        from core.secret_manager import get_secret
        db_password = get_secret("jarvis-db-password")
        print("✅ Retrieved database password from Secret Manager")
    except Exception as e:
        print(f"❌ Failed to get password from Secret Manager: {e}")
        print("💡 Run: gcloud secrets versions access latest --secret='jarvis-db-password'")
        return

    # Connect to Cloud SQL
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        database="jarvis_learning",
        user="jarvis",
        password=db_password,
    )

    try:
        # List all tables
        print("\n=== TABLES IN DATABASE ===")
        tables = await conn.fetch("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)

        for table in tables:
            print(f"  - {table['table_name']}")

        # Check speaker_profiles columns
        print("\n=== SPEAKER_PROFILES SCHEMA ===")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'speaker_profiles'
            ORDER BY ordinal_position
        """)

        for col in columns:
            nullable = "" if col['is_nullable'] == 'NO' else " (nullable)"
            print(f"  {col['column_name']}: {col['data_type']}{nullable}")

        # Check voice_samples columns
        print("\n=== VOICE_SAMPLES SCHEMA ===")
        columns = await conn.fetch("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'voice_samples'
            ORDER BY ordinal_position
        """)

        for col in columns:
            nullable = "" if col['is_nullable'] == 'NO' else " (nullable)"
            print(f"  {col['column_name']}: {col['data_type']}{nullable}")

    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(check_schema())