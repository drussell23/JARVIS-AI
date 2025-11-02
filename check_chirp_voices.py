#!/usr/bin/env python3
"""
Check which Chirp-HD voices actually exist
"""
import asyncio
from google.cloud import texttospeech

async def main():
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices()

    chirp_voices = {}
    for voice in response.voices:
        if 'Chirp-HD' in voice.name:
            lang = voice.language_codes[0]
            if lang not in chirp_voices:
                chirp_voices[lang] = []
            chirp_voices[lang].append((voice.name, voice.ssml_gender))

    print("=" * 70)
    print("Available Chirp-HD Voices")
    print("=" * 70)

    for lang in sorted(chirp_voices.keys()):
        print(f"\n{lang}:")
        for name, gender in sorted(chirp_voices[lang]):
            gender_str = texttospeech.SsmlVoiceGender(gender).name
            print(f"   {name:30} - {gender_str}")

if __name__ == "__main__":
    asyncio.run(main())
