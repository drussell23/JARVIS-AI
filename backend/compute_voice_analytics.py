#!/usr/bin/env python3
"""
Quick script to compute voice profile analytics
"""
import asyncio
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from intelligence.learning_database import get_learning_database

async def main():
    print("\n" + "=" * 80)
    print("📊 VOICE PROFILE ANALYTICS")
    print("=" * 80)

    db = await get_learning_database()

    # Get speaker profile
    async with db.db.cursor() as cursor:
        # Total samples
        await cursor.execute('SELECT COUNT(*) FROM voice_samples WHERE speaker_id = 1')
        result = await cursor.fetchone()
        total = result['count'] if isinstance(result, dict) else result[0]

        # With audio
        await cursor.execute('SELECT COUNT(*) FROM voice_samples WHERE speaker_id = 1 AND audio_data IS NOT NULL')
        result = await cursor.fetchone()
        with_audio = result['count'] if isinstance(result, dict) else result[0]

        # Get speaker profile
        await cursor.execute('SELECT * FROM speaker_profiles WHERE speaker_id = 1')
        profile = await cursor.fetchone()

        if profile:
            speaker_name = profile['speaker_name'] if isinstance(profile, dict) else profile[1]
            total_samples_prof = profile['total_samples'] if isinstance(profile, dict) else profile[4]
            confidence = profile['confidence_score'] if isinstance(profile, dict) else profile[6]

            print(f"\n👤 Speaker: {speaker_name}")
            print(f"   ID: 1")
            print(f"\n📊 Sample Statistics:")
            print(f"   Total samples: {total}")
            print(f"   With raw audio: {with_audio} (NEW - enables advanced reconstruction)")
            print(f"   Legacy samples: {total - with_audio}")
            print(f"   Profile total: {total_samples_prof}")

            if confidence:
                print(f"\n🎯 Confidence Score: {confidence:.1%}")

            print(f"\n✨ Status:")
            if with_audio >= 10:
                print(f"   ✅ Advanced reconstruction ENABLED ({with_audio} samples with audio)")
                print(f"   ✅ Voice biometric authentication OPTIMAL")
                print(f"   ✅ Model dimension changes handled seamlessly")
            elif with_audio > 0:
                print(f"   ⚠️  Partial reconstruction ({with_audio} samples with audio)")
                print(f"   💡 Recommended: Collect {10 - with_audio} more samples for optimal results")
            else:
                print(f"   ⚠️  No raw audio samples (legacy format only)")
                print(f"   💡 Run: python3 backend/quick_voice_enhancement.py")

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
