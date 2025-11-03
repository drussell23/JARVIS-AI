#!/usr/bin/env python3
"""
ElevenLabs TTS Service with Advanced Caching
Provides African American, African, and Asian accent voices
"""
import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

import aiohttp

logger = logging.getLogger(__name__)


class VoiceAccent(Enum):
    """Voice accent categories"""
    AFRICAN_AMERICAN = "african_american"
    AFRICAN = "african"
    ASIAN = "asian"
    HISPANIC = "hispanic"
    EUROPEAN = "european"


class VoiceGender(Enum):
    """Voice gender types"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass
class ElevenLabsVoiceConfig:
    """Configuration for an ElevenLabs voice"""
    voice_id: str
    name: str
    accent: VoiceAccent
    gender: VoiceGender
    description: str
    language_code: str = "en"
    stability: float = 0.5
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'voice_id': self.voice_id,
            'name': self.name,
            'accent': self.accent.value,
            'gender': self.gender.value,
            'description': self.description,
            'language_code': self.language_code,
            'stability': self.stability,
            'similarity_boost': self.similarity_boost,
            'style': self.style,
            'use_speaker_boost': self.use_speaker_boost
        }


class ElevenLabsTTSService:
    """
    Advanced ElevenLabs TTS service with:
    - Async API calls
    - Persistent caching (free tier optimization)
    - Dynamic voice discovery
    - Accent-based voice selection
    """

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        """
        Initialize ElevenLabs TTS service

        Args:
            api_key: ElevenLabs API key (defaults to SecretManager -> env var)
            cache_dir: Directory for caching audio files (defaults to ~/.jarvis/tts_cache/elevenlabs)
        """
        # Try to get API key from SecretManager first, then env var
        self.api_key = api_key
        if not self.api_key:
            try:
                from backend.core.secret_manager import SecretManager
                secret_mgr = SecretManager()
                self.api_key = secret_mgr.get_secret("elevenlabs-api-key")
                if self.api_key:
                    logger.info("✅ ElevenLabs API key loaded from GCP Secret Manager")
            except Exception as e:
                logger.debug(f"Could not load from Secret Manager: {e}")

        # Fallback to environment variable
        if not self.api_key:
            self.api_key = os.getenv("ELEVENLABS_API_KEY")
            if self.api_key:
                logger.info("✅ ElevenLabs API key loaded from environment variable")

        if not self.api_key:
            logger.warning("⚠️  No ElevenLabs API key found. Checked: GCP Secret Manager, ELEVENLABS_API_KEY env var")

        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path.home() / ".jarvis" / "tts_cache" / "elevenlabs"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Voice metadata cache
        self.voice_metadata_file = self.cache_dir / "voice_metadata.json"
        self.available_voices: Dict[str, ElevenLabsVoiceConfig] = {}

        # Session for async requests
        self.session: Optional[aiohttp.ClientSession] = None

        logger.info(f"📦 ElevenLabs TTS cache directory: {self.cache_dir}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def close(self):
        """Close aiohttp session"""
        if self.session and not self.session.closed:
            await self.session.close()

    def _get_cache_key(self, text: str, voice_config: ElevenLabsVoiceConfig) -> str:
        """Generate cache key for audio file"""
        config_str = json.dumps(voice_config.to_dict(), sort_keys=True)
        cache_input = f"{text}|{config_str}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.mp3"

    async def discover_voices(self, force_refresh: bool = False) -> Dict[str, ElevenLabsVoiceConfig]:
        """
        Discover available voices from ElevenLabs API

        Args:
            force_refresh: Force refresh from API (ignore cached metadata)

        Returns:
            Dictionary of voice_id -> ElevenLabsVoiceConfig
        """
        # Check cached metadata
        if not force_refresh and self.voice_metadata_file.exists():
            try:
                with open(self.voice_metadata_file, 'r') as f:
                    cached_data = json.load(f)

                self.available_voices = {}
                for voice_id, voice_data in cached_data.items():
                    self.available_voices[voice_id] = ElevenLabsVoiceConfig(
                        voice_id=voice_data['voice_id'],
                        name=voice_data['name'],
                        accent=VoiceAccent(voice_data['accent']),
                        gender=VoiceGender(voice_data['gender']),
                        description=voice_data['description'],
                        language_code=voice_data.get('language_code', 'en'),
                        stability=voice_data.get('stability', 0.5),
                        similarity_boost=voice_data.get('similarity_boost', 0.75),
                        style=voice_data.get('style', 0.0),
                        use_speaker_boost=voice_data.get('use_speaker_boost', True)
                    )

                logger.info(f"✅ Loaded {len(self.available_voices)} voices from cache")
                return self.available_voices
            except Exception as e:
                logger.warning(f"⚠️  Failed to load cached voice metadata: {e}")

        # Fetch from API
        if not self.api_key:
            logger.error("❌ Cannot discover voices without API key")
            return {}

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/voices"

            async with session.get(url) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ Failed to fetch voices: {response.status} - {error_text}")
                    return {}

                data = await response.json()
                voices = data.get('voices', [])

                logger.info(f"🔍 Discovered {len(voices)} voices from ElevenLabs API")

                # We'll use pre-made voices and categorize by accent
                # ElevenLabs doesn't provide accent tags, so we'll manually curate
                self.available_voices = {}

                # Save all voices for manual curation
                all_voices_file = self.cache_dir / "all_voices.json"
                with open(all_voices_file, 'w') as f:
                    json.dump(voices, f, indent=2)

                logger.info(f"💾 Saved all voices to {all_voices_file} for manual review")

                return self.available_voices

        except Exception as e:
            logger.error(f"❌ Error discovering voices: {e}")
            return {}

    async def get_voices_by_accent(self, accent: VoiceAccent) -> List[ElevenLabsVoiceConfig]:
        """Get voices filtered by accent"""
        if not self.available_voices:
            await self.discover_voices()

        return [
            voice for voice in self.available_voices.values()
            if voice.accent == accent
        ]

    async def synthesize_speech(
        self,
        text: str,
        voice_config: ElevenLabsVoiceConfig,
        use_cache: bool = True
    ) -> bytes:
        """
        Synthesize speech using ElevenLabs API with caching

        Args:
            text: Text to synthesize
            voice_config: Voice configuration
            use_cache: Use cached audio if available

        Returns:
            Audio data as bytes (MP3 format)
        """
        # Check cache first
        cache_key = self._get_cache_key(text, voice_config)
        cache_path = self._get_cache_path(cache_key)

        if use_cache and cache_path.exists():
            logger.debug(f"✅ Using cached audio: {cache_path.name}")
            return cache_path.read_bytes()

        # Generate new audio
        if not self.api_key:
            raise ValueError("ElevenLabs API key not configured")

        try:
            session = await self._get_session()
            url = f"{self.BASE_URL}/text-to-speech/{voice_config.voice_id}"

            payload = {
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {
                    "stability": voice_config.stability,
                    "similarity_boost": voice_config.similarity_boost,
                    "style": voice_config.style,
                    "use_speaker_boost": voice_config.use_speaker_boost
                }
            }

            logger.debug(f"🎤 Generating audio with voice: {voice_config.name} ({voice_config.accent.value})")

            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

                audio_data = await response.read()

                # Cache the audio
                cache_path.write_bytes(audio_data)
                logger.info(f"💾 Cached audio: {cache_path.name} ({len(audio_data)} bytes)")

                return audio_data

        except Exception as e:
            logger.error(f"❌ Failed to synthesize speech: {e}")
            raise

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.cache_dir.exists():
            return {
                'cache_dir': str(self.cache_dir),
                'exists': False,
                'file_count': 0,
                'total_size_mb': 0.0
            }

        files = list(self.cache_dir.glob("*.mp3"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            'cache_dir': str(self.cache_dir),
            'exists': True,
            'file_count': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'voices_configured': len(self.available_voices)
        }

    def clear_cache(self):
        """Clear all cached audio files"""
        if self.cache_dir.exists():
            for f in self.cache_dir.glob("*.mp3"):
                f.unlink()
            logger.info("🗑️  Cleared ElevenLabs TTS cache")

    def load_curated_voices(self) -> Dict[str, ElevenLabsVoiceConfig]:
        """
        Load manually curated voice profiles for security testing

        These are hand-picked voices that provide:
        - African American English accents
        - African accents (Nigerian, Kenyan, South African)
        - Asian accents (Chinese, Japanese, Korean influenced English)
        """
        # Since we can't dynamically determine accent from API,
        # we'll configure specific voices manually after discovery
        # Users should run discover_voices() first, review all_voices.json,
        # then configure specific voice IDs here

        curated_file = self.cache_dir / "curated_voices.json"

        if curated_file.exists():
            try:
                with open(curated_file, 'r') as f:
                    curated_data = json.load(f)

                self.available_voices = {}
                for voice_id, voice_data in curated_data.items():
                    self.available_voices[voice_id] = ElevenLabsVoiceConfig(
                        voice_id=voice_data['voice_id'],
                        name=voice_data['name'],
                        accent=VoiceAccent(voice_data['accent']),
                        gender=VoiceGender(voice_data['gender']),
                        description=voice_data['description'],
                        language_code=voice_data.get('language_code', 'en'),
                        stability=voice_data.get('stability', 0.5),
                        similarity_boost=voice_data.get('similarity_boost', 0.75),
                        style=voice_data.get('style', 0.0),
                        use_speaker_boost=voice_data.get('use_speaker_boost', True)
                    )

                logger.info(f"✅ Loaded {len(self.available_voices)} curated voices")
                return self.available_voices
            except Exception as e:
                logger.error(f"❌ Failed to load curated voices: {e}")

        # Return default configuration with example voices
        # NOTE: These voice IDs need to be replaced with actual IDs from your account
        logger.warning("⚠️  No curated voices found. Using example configuration.")
        logger.warning("⚠️  Run discover_voices() and configure curated_voices.json")

        return {}

    def save_curated_voices(self, voices: Dict[str, ElevenLabsVoiceConfig]):
        """Save curated voice configurations"""
        curated_file = self.cache_dir / "curated_voices.json"

        voice_dict = {
            voice_id: voice.to_dict()
            for voice_id, voice in voices.items()
        }

        with open(curated_file, 'w') as f:
            json.dump(voice_dict, f, indent=2)

        logger.info(f"💾 Saved {len(voices)} curated voices to {curated_file}")


async def main():
    """Test ElevenLabs TTS service"""
    print("=" * 70)
    print("🧪 ElevenLabs TTS Service Test")
    print("=" * 70)

    service = ElevenLabsTTSService()

    # Discover voices
    print("\n🔍 Discovering voices...")
    voices = await service.discover_voices(force_refresh=True)
    print(f"✅ Found {len(voices)} voices")

    # Show cache stats
    stats = service.get_cache_stats()
    print(f"\n📊 Cache Stats:")
    print(f"   Directory: {stats['cache_dir']}")
    print(f"   Files: {stats['file_count']}")
    print(f"   Size: {stats['total_size_mb']:.2f} MB")

    await service.close()


if __name__ == "__main__":
    asyncio.run(main())
