#!/usr/bin/env python3
"""
Test Voice Profile Cache System
================================

Tests the ultra-robust voice profile caching system:
1. Bootstrap from CloudSQL
2. SQLite cache verification
3. FAISS cache verification
4. Offline authentication capability
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))


async def test_voice_cache():
    """Test voice profile cache system"""
    print("\n" + "=" * 60)
    print("  Voice Profile Cache Test")
    print("=" * 60 + "\n")

    try:
        from intelligence.learning_database import JARVISLearningDatabase

        # Initialize database (will trigger bootstrap if needed)
        print("1️⃣  Initializing learning database...")
        learning_db = JARVISLearningDatabase()
        await learning_db.initialize()
        print("✅ Learning database initialized\n")

        # Check hybrid sync status
        if hasattr(learning_db, 'hybrid_sync') and learning_db.hybrid_sync:
            hs = learning_db.hybrid_sync
            print("2️⃣  Hybrid Sync Status:")
            print(f"   CloudSQL Available: {hs.metrics.cloudsql_available}")
            print(f"   Voice Profiles Cached: {hs.metrics.voice_profiles_cached}")
            print(f"   Cache Last Updated: {hs.metrics.voice_cache_last_updated}")
            print(f"   FAISS Cache Size: {hs.metrics.cache_size}")
            print(f"   Circuit State: {hs.metrics.circuit_state}\n")

            # Check FAISS cache
            if hs.faiss_cache:
                print("3️⃣  FAISS Cache Status:")
                print(f"   Size: {hs.faiss_cache.size()} embeddings")
                print(f"   Dimension: {hs.faiss_cache.dimension}D")
                print()

            # Check SQLite cache
            print("4️⃣  SQLite Cache Status:")
            async with hs.sqlite_conn.execute("""
                SELECT speaker_name, total_samples, LENGTH(voiceprint_embedding) as embedding_size
                FROM speaker_profiles
            """) as cursor:
                rows = await cursor.fetchall()

                if rows:
                    print(f"   Found {len(rows)} cached profiles:")
                    for row in rows:
                        name, samples, size = row
                        print(f"      • {name}: {samples} samples, {size} bytes embedding")
                else:
                    print("   ⚠️  No profiles in SQLite cache")
            print()

            # Test offline read
            print("5️⃣  Testing Offline Voice Profile Read...")
            test_profile = await hs.read_voice_profile("Derek J. Russell")
            if test_profile:
                print("✅ Profile read successful (offline capable!)")
                print(f"   Name: {test_profile['speaker_name']}")
                print(f"   Samples: {test_profile.get('total_samples', 0)}")
                print(f"   Embedding: {len(test_profile.get('embedding', []))}D")
            else:
                print("❌ Profile read failed")
            print()

        else:
            print("⚠️  Hybrid sync not available")

        # Shutdown
        if hasattr(learning_db, 'hybrid_sync') and learning_db.hybrid_sync:
            await learning_db.hybrid_sync.shutdown()

        print("=" * 60)
        print("✅ Voice cache test complete!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(test_voice_cache())
