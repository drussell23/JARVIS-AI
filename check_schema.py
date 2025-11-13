#!/usr/bin/env python3
import asyncio
import asyncpg
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))

async def check_schema():
    from core.secret_manager import get_db_password
    password = get_db_password()

    conn = await asyncpg.connect(
        host='127.0.0.1',
        port=5432,
        database='jarvis_learning',
        user='jarvis',
        password=password
    )

    # Get actual columns
    columns = await conn.fetch("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'speaker_profiles'
        ORDER BY ordinal_position
    """)

    print("CloudSQL speaker_profiles columns:")
    for col in columns:
        print(f"  • {col['column_name']}: {col['data_type']}")

    await conn.close()

asyncio.run(check_schema())
