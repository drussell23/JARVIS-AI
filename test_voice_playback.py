#!/usr/bin/env python3
"""
Test script to generate and save sample voices for manual verification
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.audio.gcp_tts_service import GCPTTSService, VoiceProfileGenerator

async def main():
    print("=" * 70)
    print("🎤 Voice Sample Generator")
    print("=" * 70)

    tts = GCPTTSService()
    generator = VoiceProfileGenerator(tts)

    # Generate diverse profiles
    profiles = await generator.generate_attacker_profiles(count=12)

    test_phrase = "unlock my screen"
    output_dir = Path("/tmp/jarvis_voice_samples")
    output_dir.mkdir(exist_ok=True)

    print(f"\n📁 Saving voice samples to: {output_dir}\n")

    # Generate and save samples
    samples_to_test = [
        (0, "US-Male"),
        (1, "US-Female"),
        (7, "British-Male"),
        (8, "British-Female"),
        (9, "Australian-Male"),
        (10, "Australian-Female"),
        (11, "Indian-Male"),
        (12, "Indian-Female"),
        (3, "African-American-Male"),
        (5, "African-American-Female"),
        (13, "Asian-Accent-Male"),
        (15, "Hispanic-Male"),
    ]

    for idx, label in samples_to_test:
        if idx >= len(profiles):
            continue

        profile = profiles[idx]
        print(f"🔊 Generating: {label:25} - {profile.name}")

        audio_data = await tts.synthesize_speech(
            text=test_phrase,
            voice_config=profile,
            use_cache=True
        )

        output_file = output_dir / f"{label}.mp3"
        output_file.write_bytes(audio_data)
        print(f"   ✅ Saved: {output_file}")

    print("\n" + "=" * 70)
    print("✅ COMPLETE!")
    print(f"\nTo listen to samples, run:")
    print(f"   afplay /tmp/jarvis_voice_samples/US-Male.mp3")
    print(f"   afplay /tmp/jarvis_voice_samples/British-Male.mp3")
    print(f"   afplay /tmp/jarvis_voice_samples/Indian-Female.mp3")
    print("\nOr open the folder:")
    print(f"   open /tmp/jarvis_voice_samples")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
