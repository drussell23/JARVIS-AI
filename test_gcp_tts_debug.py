#!/usr/bin/env python3
"""
Debug script to test GCP TTS integration in voice security tester
"""
import asyncio
import logging
import sys
from pathlib import Path

# Setup logging to see all messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

sys.path.insert(0, str(Path(__file__).parent))

from backend.voice_unlock.voice_security_tester import (
    VoiceSecurityTester,
    VoiceProfile,
    PlaybackConfig
)

async def main():
    print("=" * 70)
    print("🔍 GCP TTS Debug Test")
    print("=" * 70)

    # Create tester with minimal config
    config = {
        'authorized_user': 'Sir',
        'test_phrase': 'unlock my screen',
        'test_mode': 'quick'  # Just 3 profiles for quick test
    }

    playback_config = PlaybackConfig(enabled=False)  # Silent mode

    print("\n📋 Initializing Voice Security Tester...")
    tester = VoiceSecurityTester(
        config=config,
        playback_config=playback_config
    )

    print(f"\n🎯 Testing {len(tester.test_profiles)} profiles:")
    for i, profile in enumerate(tester.test_profiles):
        print(f"   {i+1}. {profile.value}")

    print("\n" + "=" * 70)
    print("🧪 Testing TTS Generation")
    print("=" * 70)

    # Test first profile
    test_profile = tester.test_profiles[0]
    test_phrase = "unlock my screen"

    print(f"\n🎤 Generating audio for profile: {test_profile.value}")
    print(f"   Phrase: '{test_phrase}'")
    print()

    # This will trigger all the debug logging
    audio_file = await tester._try_tts_engines(
        profile=test_profile,
        text=test_phrase
    )

    print("\n" + "=" * 70)
    print("📊 Results")
    print("=" * 70)

    if audio_file:
        print(f"✅ Generated audio file: {audio_file}")
        print(f"   Size: {audio_file.stat().st_size} bytes")
        print(f"   Extension: {audio_file.suffix}")

        if audio_file.suffix == '.mp3':
            print("   ✅ Using GCP TTS (MP3 format)")
        elif audio_file.suffix == '.wav':
            print("   ⚠️  Using macOS 'say' fallback (WAV format)")
    else:
        print("❌ Failed to generate audio")

    print("\n" + "=" * 70)
    print("📁 Cache Status")
    print("=" * 70)

    cache_stats = tester.gcp_tts.get_cache_stats()
    print(f"   Cache directory: {cache_stats['cache_dir']}")
    print(f"   Cached files: {cache_stats['file_count']}")
    print(f"   Total size: {cache_stats['total_size_mb']:.2f} MB")

if __name__ == "__main__":
    asyncio.run(main())
