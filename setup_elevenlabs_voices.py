#!/usr/bin/env python3
"""
ElevenLabs Voice Discovery and Configuration Tool

This script helps you:
1. Discover available ElevenLabs voices
2. Configure curated voices for security testing
3. Generate sample audio for voice verification
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.audio.elevenlabs_tts_service import (
    ElevenLabsTTSService,
    ElevenLabsVoiceConfig,
    VoiceAccent,
    VoiceGender
)


# Pre-configured curated voices for security testing
# These are example configurations - you'll need to replace voice_ids with actual IDs from your account
CURATED_VOICES = {
    # ===== AFRICAN AMERICAN ENGLISH =====
    "african_american_male_1": {
        "voice_id": "PLACEHOLDER_ID_1",  # Replace with actual voice ID
        "name": "AfricanAmericanMale1",
        "accent": "african_american",
        "gender": "male",
        "description": "African American male voice, deep and resonant",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "african_american_male_2": {
        "voice_id": "PLACEHOLDER_ID_2",
        "name": "AfricanAmericanMale2",
        "accent": "african_american",
        "gender": "male",
        "description": "African American male voice, conversational",
        "language_code": "en",
        "stability": 0.6,
        "similarity_boost": 0.7,
        "style": 0.2,
        "use_speaker_boost": True
    },
    "african_american_female_1": {
        "voice_id": "PLACEHOLDER_ID_3",
        "name": "AfricanAmericanFemale1",
        "accent": "african_american",
        "gender": "female",
        "description": "African American female voice, warm and clear",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "african_american_female_2": {
        "voice_id": "PLACEHOLDER_ID_4",
        "name": "AfricanAmericanFemale2",
        "accent": "african_american",
        "gender": "female",
        "description": "African American female voice, energetic",
        "language_code": "en",
        "stability": 0.4,
        "similarity_boost": 0.8,
        "style": 0.3,
        "use_speaker_boost": True
    },

    # ===== AFRICAN ACCENTS =====
    "nigerian_male": {
        "voice_id": "PLACEHOLDER_ID_5",
        "name": "NigerianMale",
        "accent": "african",
        "gender": "male",
        "description": "Nigerian English accent, male",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "nigerian_female": {
        "voice_id": "PLACEHOLDER_ID_6",
        "name": "NigerianFemale",
        "accent": "african",
        "gender": "female",
        "description": "Nigerian English accent, female",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "kenyan_male": {
        "voice_id": "PLACEHOLDER_ID_7",
        "name": "KenyanMale",
        "accent": "african",
        "gender": "male",
        "description": "Kenyan English accent, male",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "kenyan_female": {
        "voice_id": "PLACEHOLDER_ID_8",
        "name": "KenyanFemale",
        "accent": "african",
        "gender": "female",
        "description": "Kenyan English accent, female",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "south_african_male": {
        "voice_id": "PLACEHOLDER_ID_9",
        "name": "SouthAfricanMale",
        "accent": "african",
        "gender": "male",
        "description": "South African English accent, male",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "south_african_female": {
        "voice_id": "PLACEHOLDER_ID_10",
        "name": "SouthAfricanFemale",
        "accent": "african",
        "gender": "female",
        "description": "South African English accent, female",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },

    # ===== ASIAN ACCENTS (English spoken with Asian accent) =====
    "chinese_male": {
        "voice_id": "PLACEHOLDER_ID_11",
        "name": "ChineseMale",
        "accent": "asian",
        "gender": "male",
        "description": "Chinese-accented English, male",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "chinese_female": {
        "voice_id": "PLACEHOLDER_ID_12",
        "name": "ChineseFemale",
        "accent": "asian",
        "gender": "female",
        "description": "Chinese-accented English, female",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "japanese_male": {
        "voice_id": "PLACEHOLDER_ID_13",
        "name": "JapaneseMale",
        "accent": "asian",
        "gender": "male",
        "description": "Japanese-accented English, male",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "japanese_female": {
        "voice_id": "PLACEHOLDER_ID_14",
        "name": "JapaneseFemale",
        "accent": "asian",
        "gender": "female",
        "description": "Japanese-accented English, female",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "korean_male": {
        "voice_id": "PLACEHOLDER_ID_15",
        "name": "KoreanMale",
        "accent": "asian",
        "gender": "male",
        "description": "Korean-accented English, male",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
    "korean_female": {
        "voice_id": "PLACEHOLDER_ID_16",
        "name": "KoreanFemale",
        "accent": "asian",
        "gender": "female",
        "description": "Korean-accented English, female",
        "language_code": "en",
        "stability": 0.5,
        "similarity_boost": 0.75,
        "style": 0.0,
        "use_speaker_boost": True
    },
}


async def discover_voices():
    """Discover all available ElevenLabs voices"""
    print("=" * 70)
    print("🔍 Discovering ElevenLabs Voices")
    print("=" * 70)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("\n❌ Error: ELEVENLABS_API_KEY environment variable not set")
        print("\nTo set it:")
        print("   export ELEVENLABS_API_KEY='your-api-key-here'")
        print("\nGet your API key from: https://elevenlabs.io/settings")
        return

    service = ElevenLabsTTSService(api_key=api_key)

    try:
        print("\n📡 Fetching voices from ElevenLabs API...")
        voices = await service.discover_voices(force_refresh=True)

        cache_dir = Path.home() / ".jarvis" / "tts_cache" / "elevenlabs"
        all_voices_file = cache_dir / "all_voices.json"

        print(f"\n✅ Discovered {len(voices)} voices")
        print(f"\n💾 All voices saved to: {all_voices_file}")
        print("\n📋 Next steps:")
        print(f"   1. Review voices in: {all_voices_file}")
        print(f"   2. Find voice IDs for desired accents")
        print(f"   3. Update CURATED_VOICES in this script")
        print(f"   4. Run: python3 {__file__} --configure")

    finally:
        await service.close()


async def configure_voices():
    """Save curated voice configuration"""
    print("=" * 70)
    print("💾 Configuring Curated Voices")
    print("=" * 70)

    cache_dir = Path.home() / ".jarvis" / "tts_cache" / "elevenlabs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    curated_file = cache_dir / "curated_voices.json"

    # Check if voice IDs have been updated
    placeholder_count = sum(
        1 for v in CURATED_VOICES.values()
        if v['voice_id'].startswith('PLACEHOLDER')
    )

    if placeholder_count > 0:
        print(f"\n⚠️  WARNING: {placeholder_count} voices still have PLACEHOLDER_ID")
        print("\n❌ You must update voice_ids with actual IDs before configuring!")
        print("\nSteps:")
        print("   1. Run: python3 setup_elevenlabs_voices.py --discover")
        print("   2. Review all_voices.json in cache directory")
        print("   3. Find voices that match desired accents")
        print("   4. Update CURATED_VOICES in this script with real voice IDs")
        print("   5. Run this command again")
        return

    # Save configuration
    with open(curated_file, 'w') as f:
        json.dump(CURATED_VOICES, f, indent=2)

    print(f"\n✅ Saved {len(CURATED_VOICES)} curated voices to: {curated_file}")
    print("\n📊 Voice breakdown:")

    accent_counts = {}
    for voice in CURATED_VOICES.values():
        accent = voice['accent']
        accent_counts[accent] = accent_counts.get(accent, 0) + 1

    for accent, count in sorted(accent_counts.items()):
        print(f"   {accent}: {count} voices")

    print("\n✅ Configuration complete!")
    print("\n📋 Next steps:")
    print("   python3 test_elevenlabs_integration.py  # Test the integration")


async def test_voices():
    """Generate sample audio for configured voices"""
    print("=" * 70)
    print("🎤 Testing Configured Voices")
    print("=" * 70)

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("\n❌ Error: ELEVENLABS_API_KEY environment variable not set")
        return

    service = ElevenLabsTTSService(api_key=api_key)

    try:
        # Load curated voices
        service.load_curated_voices()

        if not service.available_voices:
            print("\n❌ No curated voices configured!")
            print("   Run: python3 setup_elevenlabs_voices.py --configure")
            return

        test_phrase = "unlock my screen"
        output_dir = Path("/tmp/elevenlabs_test_samples")
        output_dir.mkdir(exist_ok=True)

        print(f"\n📁 Saving test samples to: {output_dir}\n")

        # Test first 3 voices as samples
        voices_to_test = list(service.available_voices.values())[:3]

        for voice_config in voices_to_test:
            print(f"🔊 Generating: {voice_config.name:30} ({voice_config.accent.value})")

            try:
                audio_data = await service.synthesize_speech(
                    text=test_phrase,
                    voice_config=voice_config,
                    use_cache=True
                )

                output_file = output_dir / f"{voice_config.name}.mp3"
                output_file.write_bytes(audio_data)
                print(f"   ✅ Saved: {output_file}")

            except Exception as e:
                print(f"   ❌ Failed: {e}")

        print("\n" + "=" * 70)
        print("✅ COMPLETE!")
        print(f"\nTo listen to samples:")
        print(f"   afplay {output_dir}/*.mp3")
        print(f"\nOr open folder:")
        print(f"   open {output_dir}")
        print("=" * 70)

    finally:
        await service.close()


async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="ElevenLabs Voice Discovery and Configuration Tool"
    )
    parser.add_argument(
        '--discover',
        action='store_true',
        help='Discover all available ElevenLabs voices'
    )
    parser.add_argument(
        '--configure',
        action='store_true',
        help='Save curated voice configuration'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Generate test audio samples'
    )

    args = parser.parse_args()

    if args.discover:
        await discover_voices()
    elif args.configure:
        await configure_voices()
    elif args.test:
        await test_voices()
    else:
        # Show usage
        print("=" * 70)
        print("🎙️  ElevenLabs Voice Setup Tool")
        print("=" * 70)
        print("\nUsage:")
        print("   python3 setup_elevenlabs_voices.py --discover      # Discover available voices")
        print("   python3 setup_elevenlabs_voices.py --configure     # Save curated voices")
        print("   python3 setup_elevenlabs_voices.py --test          # Test with samples")
        print("\n📋 Setup Steps:")
        print("   1. Set ELEVENLABS_API_KEY environment variable")
        print("   2. Run --discover to find available voices")
        print("   3. Update CURATED_VOICES in this script with real voice IDs")
        print("   4. Run --configure to save configuration")
        print("   5. Run --test to verify voices work")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
