#!/usr/bin/env python3
"""
Update speaker profile thresholds to 75% for proper security testing.
"""
import asyncio
import sys
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent')

async def update_thresholds():
    from backend.intelligence.learning_database import get_learning_database

    print("🔐 Connecting to Cloud SQL database...")
    db = await get_learning_database()

    # Get current profiles
    print("\n📊 Current speaker profiles:")
    profiles = await db.get_all_speaker_profiles()
    for p in profiles:
        threshold = p.get('recognition_confidence', 0.5)
        print(f"  • {p['speaker_name']}: {threshold:.0%} threshold (ID: {p['speaker_id']})")

    # Update thresholds via direct SQL
    print("\n🔧 Updating thresholds to 75%...")

    # Use the cloud_db_adapter to execute SQL directly
    if hasattr(db, 'cloud_db'):
        cloud_db = db.cloud_db
        if cloud_db and cloud_db.pool:
            async with cloud_db.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE speaker_profiles
                    SET recognition_confidence = 0.75
                    WHERE recognition_confidence != 0.75
                """)
                print(f"✅ Updated {result} profile(s)")
        else:
            print("❌ Cloud SQL not available")
            return False
    else:
        print("❌ Cloud database not initialized")
        return False

    # Verify updates
    print("\n✅ Verification:")
    profiles = await db.get_all_speaker_profiles()
    for p in profiles:
        threshold = p.get('recognition_confidence', 0.5)
        print(f"  • {p['speaker_name']}: {threshold:.0%} threshold")

    print("\n🎉 Thresholds updated successfully!")
    return True

if __name__ == "__main__":
    success = asyncio.run(update_thresholds())
    sys.exit(0 if success else 1)
