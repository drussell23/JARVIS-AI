#!/usr/bin/env python3
"""Test voice verification with lower threshold to see actual confidence."""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

async def test_threshold():
    """Test with different thresholds."""

    print("\n" + "="*80)
    print("TESTING VOICE VERIFICATION THRESHOLDS")
    print("="*80)

    from backend.voice.speaker_verification_service import SpeakerVerificationService

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n Current settings:")
    print(f"   Model dimension: {service.current_model_dimension}")
    print(f"   Default threshold: {service.verification_threshold} ({service.verification_threshold*100:.0f}%)")
    print(f"   Profiles loaded: {len(service.speaker_profiles)}")

    # Test with different thresholds
    thresholds_to_test = [0.45, 0.30, 0.15, 0.10, 0.05]

    print(f"\n Threshold Analysis:")
    print(f" {'Threshold':<12} {'Status':<20}")
    print(f" {'-'*12} {'-'*20}")

    for threshold in thresholds_to_test:
        if 0.0767 >= threshold:  # Your last confidence was 7.67%
            status = "✅ Would unlock"
        else:
            status = "❌ Would reject"
        print(f" {threshold*100:>5.0f}%       {status}")

    print(f"\n Your last attempt: 7.67% confidence")
    print(f" Current threshold: {service.verification_threshold*100:.0f}%")
    print(f" Needed: {(service.verification_threshold - 0.0767)*100:.1f}% more confidence")

    print("\n" + "="*80)

    # Suggest next steps
    print("\n📊 RECOMMENDATIONS:")
    print("-" * 40)

    if 0.0767 < 0.10:
        print("\n1. Re-record voice samples in your current environment:")
        print("   python backend/quick_voice_enhancement.py")
        print("\n2. After recording, re-enable BEAST MODE:")
        print("   python backend/enable_beast_mode_now.py")
        print("\n3. Restart JARVIS and test again")
    else:
        print("\n1. Consider temporarily lowering threshold to 10%")
        print("2. System will learn and improve confidence over time")

    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(test_threshold())