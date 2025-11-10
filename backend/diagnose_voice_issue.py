#!/usr/bin/env python3
"""
Diagnose voice unlock issue - why confidence is only 30%
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def main():
    print("\n" + "=" * 80)
    print("🔍 VOICE UNLOCK DIAGNOSTIC")
    print("=" * 80)

    # 1. Check database connection
    print("\n1. Database Connection Check:")
    from intelligence.learning_database import get_learning_database

    try:
        db = await get_learning_database()
        print("   ✅ Connected to database")

        # Check if it's Cloud SQL or SQLite
        if hasattr(db.db, '_pool'):
            print("   ✅ Using Cloud SQL (PostgreSQL)")
        else:
            print("   ⚠️  Using SQLite (local only)")

    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return

    # 2. Check speaker profiles
    print("\n2. Speaker Profiles in Database:")
    async with db.db.cursor() as cursor:
        # Get all profiles
        await cursor.execute("""
            SELECT speaker_id, speaker_name, total_samples,
                   pitch_mean_hz, formant_f1_hz, spectral_centroid_hz,
                   embedding_dimension
            FROM speaker_profiles
            ORDER BY speaker_id
        """)
        profiles = await cursor.fetchall()

        if not profiles:
            print("   ❌ No speaker profiles found!")
        else:
            for profile in profiles:
                if isinstance(profile, dict):
                    sid = profile['speaker_id']
                    name = profile['speaker_name']
                    samples = profile['total_samples']
                    pitch = profile.get('pitch_mean_hz')
                    f1 = profile.get('formant_f1_hz')
                    spectral = profile.get('spectral_centroid_hz')
                    dim = profile.get('embedding_dimension', 0)
                else:
                    sid, name, samples, pitch, f1, spectral, dim = profile

                has_acoustic = bool(pitch or f1 or spectral)

                print(f"\n   Profile #{sid}: {name}")
                print(f"      Samples: {samples}")
                print(f"      Embedding Dimension: {dim}")
                print(f"      Pitch: {pitch:.1f} Hz" if pitch else "      Pitch: None")
                print(f"      Formant F1: {f1:.0f} Hz" if f1 else "      Formant F1: None")
                print(f"      Spectral Centroid: {spectral:.0f} Hz" if spectral else "      Spectral Centroid: None")
                print(f"      Acoustic Features: {'✅ YES - BEAST MODE' if has_acoustic else '❌ NO - Basic mode only'}")

    # 3. Check what the service would load
    print("\n3. Testing Speaker Verification Service Load:")
    from voice.speaker_verification_service import SpeakerVerificationService

    try:
        service = SpeakerVerificationService(learning_db=db)
        await service.initialize_fast()

        print(f"   ✅ Service initialized")
        print(f"   Profiles loaded: {len(service.speaker_profiles)}")

        for name, profile in service.speaker_profiles.items():
            print(f"\n   Loaded Profile: {name}")
            print(f"      Speaker ID: {profile['speaker_id']}")
            print(f"      Embedding shape: {profile['embedding'].shape if hasattr(profile['embedding'], 'shape') else 'unknown'}")
            print(f"      Total samples: {profile.get('total_samples', 0)}")

            # Check acoustic features
            af = profile.get('acoustic_features', {})
            has_af = any(v is not None for v in af.values())
            print(f"      Acoustic Features in memory: {'✅ YES' if has_af else '❌ NO'}")

            if has_af:
                print(f"         Pitch: {af.get('pitch_mean_hz')}")
                print(f"         F1: {af.get('formant_f1_hz')}")
                print(f"         Spectral: {af.get('spectral_centroid_hz')}")

    except Exception as e:
        print(f"   ❌ Service initialization error: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80 + "\n")

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())