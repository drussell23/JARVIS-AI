#!/usr/bin/env python3
"""
Test script to verify voice diversity - ensures all 36 voices are unique
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.audio.gcp_tts_service import GCPTTSService, VoiceProfileGenerator

async def main():
    print("🎭 Testing Voice Diversity")
    print("=" * 70)

    # Initialize service
    tts = GCPTTSService()
    generator = VoiceProfileGenerator(tts)

    # Generate all 36 profiles
    print("\n📋 Generating 36 attacker voice profiles...")
    profiles = await generator.generate_attacker_profiles(count=36)

    print(f"✅ Generated {len(profiles)} profiles\n")

    # Check for uniqueness
    voice_names = [p.name for p in profiles]
    unique_voices = set(voice_names)

    print(f"🔍 Uniqueness Check:")
    print(f"   Total profiles: {len(profiles)}")
    print(f"   Unique voices: {len(unique_voices)}")

    if len(unique_voices) == len(profiles):
        print(f"   ✅ ALL VOICES ARE UNIQUE!\n")
    else:
        print(f"   ⚠️  DUPLICATES DETECTED!\n")

        # Find duplicates
        from collections import Counter
        voice_counts = Counter(voice_names)
        duplicates = {name: count for name, count in voice_counts.items() if count > 1}

        print(f"   Duplicate voices:")
        for voice_name, count in duplicates.items():
            print(f"      - {voice_name}: used {count} times")
        print()

    # Show all voice profiles with details
    print(f"\n📊 All {len(profiles)} Voice Profiles:")
    print("-" * 70)

    # Group by language
    by_language = {}
    for i, profile in enumerate(profiles):
        lang = profile.language_code
        if lang not in by_language:
            by_language[lang] = []
        by_language[lang].append((i, profile))

    for lang_code in sorted(by_language.keys()):
        voices = by_language[lang_code]
        print(f"\n{lang_code} ({len(voices)} voices):")
        for idx, profile in voices:
            print(f"  {idx:2}. {profile.name:30} | Rate: {profile.speaking_rate:4.2f}x | Pitch: {profile.pitch:+5.1f}")

    print("\n" + "=" * 70)

    # Verify against expected distribution
    expected_distribution = {
        "en-US": 20,  # US English variations
        "en-GB": 4,   # British English
        "en-AU": 4,   # Australian English
        "en-IN": 4,   # Indian English
    }

    actual_distribution = {lang: len(voices) for lang, voices in by_language.items()}

    print("\n📊 Language Distribution:")
    for lang, expected_count in expected_distribution.items():
        actual_count = actual_distribution.get(lang, 0)
        status = "✅" if actual_count == expected_count else "⚠️ "
        print(f"   {status} {lang}: {actual_count}/{expected_count}")

    # Final verdict
    print("\n" + "=" * 70)
    if len(unique_voices) == 36 and all(
        actual_distribution.get(lang, 0) == count
        for lang, count in expected_distribution.items()
    ):
        print("✅ VOICE DIVERSITY TEST PASSED!")
        print("   All 36 voices are unique and properly distributed.")
    else:
        print("⚠️  VOICE DIVERSITY TEST FAILED!")
        print("   Some issues detected - see details above.")

    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
