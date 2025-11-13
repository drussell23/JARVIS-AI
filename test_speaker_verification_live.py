#!/usr/bin/env python3
"""
Live test of speaker verification system
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "backend"))


async def test_verification():
    print("=" * 60)
    print("  Testing Speaker Verification System")
    print("=" * 60)
    print()

    # Test 1: Import and initialize
    print("1️⃣  Importing speaker verification service...")
    try:
        from voice.speaker_verification_service import SpeakerVerificationService
        print("✅ Import successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # Test 2: Initialize service
    print("\n2️⃣  Initializing speaker verification service...")
    service = SpeakerVerificationService()
    try:
        await service.initialize()
        print("✅ Initialization successful")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test 3: Check profiles loaded
    print(f"\n3️⃣  Checking loaded profiles...")
    print(f"   Profiles loaded: {len(service.speaker_profiles)}")
    for name, profile in service.speaker_profiles.items():
        print(f"   - {name}: {profile.total_samples} samples, threshold: {profile.threshold:.0%}")

    # Test 4: Check encoder
    print(f"\n4️⃣  Checking speaker encoder...")
    encoder_ready = getattr(service, '_encoder_preloaded', False)
    print(f"   Encoder preloaded: {encoder_ready}")

    if service.speechbrain_engine:
        print(f"   SpeechBrain engine: Available")
        has_speaker_encoder = hasattr(service.speechbrain_engine, 'speaker_encoder')
        print(f"   Has speaker_encoder attr: {has_speaker_encoder}")
    else:
        print(f"   ❌ SpeechBrain engine: NOT AVAILABLE")

    # Test 5: Check database
    print(f"\n5️⃣  Checking voice database...")
    try:
        from intelligence.learning_database import JARVISLearningDatabase
        db = JARVISLearningDatabase()
        await db.initialize()

        if db.hybrid_sync:
            print(f"   ✅ Hybrid sync enabled")
            print(f"   SQLite: {db.hybrid_sync.sqlite_path}")
            print(f"   FAISS cache size: {db.hybrid_sync.faiss_cache.size() if db.hybrid_sync.faiss_cache else 0}")
        else:
            print(f"   ❌ Hybrid sync NOT available")

    except Exception as e:
        print(f"   ❌ Database check failed: {e}")

    print("\n" + "=" * 60)
    print("✅ Diagnostic complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_verification())
