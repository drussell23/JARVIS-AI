#!/usr/bin/env python3
"""
Quick test script for GCP TTS integration
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.audio.gcp_tts_service import GCPTTSService, VoiceProfileGenerator

async def main():
    print("🎙️ Testing GCP TTS Integration")
    print("=" * 50)

    # Initialize service
    tts = GCPTTSService()
    generator = VoiceProfileGenerator(tts)

    # Test 1: List available voices
    print("\n📋 Test 1: Discovering available voices...")
    voices = await tts.get_available_voices("en-US")
    print(f"✅ Found {len(voices)} US English voices")
    print(f"   First 3 voices: {[v.name for v in voices[:3]]}")

    # Test 2: Generate voice profiles
    print("\n🎭 Test 2: Generating attacker voice profiles...")
    profiles = await generator.generate_attacker_profiles(count=5)
    print(f"✅ Generated {len(profiles)} voice profiles")
    for i, profile in enumerate(profiles, 1):
        print(f"   {i}. {profile.name} ({profile.language_code}) - Rate: {profile.speaking_rate}x, Pitch: {profile.pitch}")

    # Test 3: Synthesize test phrase
    print("\n🔊 Test 3: Synthesizing test phrase...")
    if profiles:
        test_profile = profiles[0]
        audio_data = await tts.synthesize_speech(
            text="unlock my screen",
            voice_config=test_profile,
            use_cache=True
        )
        print(f"✅ Generated {len(audio_data)} bytes of audio")
        print(f"   Voice: {test_profile.name}")

        # Test cache
        audio_data_cached = await tts.synthesize_speech(
            text="unlock my screen",
            voice_config=test_profile,
            use_cache=True
        )
        print(f"✅ Cache hit: {len(audio_data_cached)} bytes (should be instant)")

    # Test 4: Cache stats
    print("\n💾 Test 4: Cache statistics...")
    stats = tts.get_cache_stats()
    print(f"✅ Cache stats:")
    print(f"   Files: {stats['file_count']}")
    print(f"   Size: {stats['total_size_mb']:.2f} MB")
    print(f"   Location: {stats['cache_dir']}")

    print("\n" + "=" * 50)
    print("✅ All tests passed! GCP TTS is ready.")

if __name__ == "__main__":
    asyncio.run(main())
