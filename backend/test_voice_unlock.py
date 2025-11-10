#!/usr/bin/env python3
"""
Test voice unlock to diagnose why confidence is low
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def main():
    print("\n" + "=" * 80)
    print("🔍 VOICE UNLOCK DIAGNOSTIC TEST")
    print("=" * 80)

    # Get the speaker service
    from voice.speaker_verification_service import _global_speaker_service

    if not _global_speaker_service:
        print("❌ Global speaker service not available. Starting backend...")
        from voice.speaker_verification_service import SpeakerVerificationService
        from intelligence.learning_database import get_learning_database

        db = await get_learning_database()
        service = SpeakerVerificationService(learning_db=db)
        await service.initialize_fast()
    else:
        service = _global_speaker_service
        print("✅ Using existing global speaker service")

    # Check loaded profiles
    print(f"\n📊 Loaded profiles: {len(service.speaker_profiles)}")
    for name, profile in service.speaker_profiles.items():
        acoustic_features = profile.get("acoustic_features", {})
        has_beast = any(v is not None for v in acoustic_features.values())
        print(f"   • {name}: {'🔬 BEAST MODE' if has_beast else '⚠️  BASIC MODE'}")

    # Simulate a voice unlock attempt
    print("\n🎤 Simulating voice unlock...")

    # Create test audio (silence for now)
    import numpy as np
    sample_rate = 16000
    duration = 2  # seconds
    audio_data = (np.random.randn(sample_rate * duration) * 0.01).astype(np.float32)
    audio_bytes = audio_data.tobytes()

    # Try verification
    result = await service.verify_speaker(audio_bytes, speaker_name="Derek J. Russell")

    print(f"\n📊 Verification Result:")
    print(f"   • Is owner: {result['is_owner']}")
    print(f"   • Confidence: {result['confidence']:.2%}")
    print(f"   • Speaker: {result['speaker_name']}")
    print(f"   • Verified: {result['verified']}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 80)

    await service.cleanup()
    if hasattr(db, 'close'):
        await db.close()

if __name__ == "__main__":
    asyncio.run(main())