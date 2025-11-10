#!/usr/bin/env python3
"""Quick script to check voice profile status"""
import asyncio
import sys
sys.path.insert(0, 'backend')

async def main():
    from voice.speaker_verification_service import SpeakerVerificationService

    print("=" * 60)
    print("VOICE PROFILE DIAGNOSTIC")
    print("=" * 60)

    service = SpeakerVerificationService()
    await service.initialize()

    print(f"\n📊 Found {len(service.speaker_profiles)} speaker profile(s):\n")

    for name, profile in service.speaker_profiles.items():
        print(f"✅ Speaker: {name}")
        print(f"   - ID: {profile['speaker_id']}")
        print(f"   - Primary User: {profile['is_primary_user']}")
        print(f"   - Security Level: {profile['security_level']}")
        print(f"   - Embedding Dimension: {profile['embedding'].shape[0]}D")
        print(f"   - Total Samples: {profile.get('total_samples', 0)}")
        print(f"   - Quality: {profile.get('quality', 'unknown')}")
        print(f"   - Threshold: {profile.get('threshold', 0):.2%}")
        print()

    if len(service.speaker_profiles) == 0:
        print("❌ NO PROFILES FOUND!")
        print("\nYou need to create a voice profile first.")
        print("The profile name must match exactly what the system expects.")
        print("\nCheck your database for existing profiles with:")
        print("  SELECT speaker_name, is_primary_user FROM speaker_profiles;")

    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
