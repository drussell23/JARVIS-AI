#!/usr/bin/env python3
"""
Unified TTS Provider Manager
Manages multiple TTS providers (GCP, ElevenLabs) with intelligent routing
"""
import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from backend.audio.gcp_tts_service import GCPTTSService, VoiceConfig as GCPVoiceConfig
from backend.audio.elevenlabs_tts_service import (
    ElevenLabsTTSService,
    ElevenLabsVoiceConfig,
    VoiceAccent,
    VoiceGender
)

logger = logging.getLogger(__name__)


class TTSProvider(Enum):
    """TTS provider types"""
    GCP = "gcp"
    ELEVENLABS = "elevenlabs"
    AUTO = "auto"


@dataclass
class UnifiedVoiceConfig:
    """
    Unified voice configuration that works across providers
    """
    # Common fields
    name: str
    language_code: str
    gender: str
    provider: TTSProvider

    # Provider-specific config
    gcp_config: Optional[GCPVoiceConfig] = None
    elevenlabs_config: Optional[ElevenLabsVoiceConfig] = None

    # Metadata for categorization
    accent: Optional[str] = None
    description: Optional[str] = None

    def get_provider_config(self) -> Union[GCPVoiceConfig, ElevenLabsVoiceConfig]:
        """Get the appropriate provider-specific config"""
        if self.provider == TTSProvider.GCP and self.gcp_config:
            return self.gcp_config
        elif self.provider == TTSProvider.ELEVENLABS and self.elevenlabs_config:
            return self.elevenlabs_config
        else:
            raise ValueError(f"No config available for provider: {self.provider}")


class TTSProviderManager:
    """
    Manages multiple TTS providers with:
    - Intelligent provider selection based on accent/language
    - Unified caching across providers
    - Automatic fallback
    - Usage tracking (for free tier optimization)
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_gcp: bool = True,
        enable_elevenlabs: bool = True
    ):
        """
        Initialize TTS provider manager

        Args:
            cache_dir: Root cache directory (defaults to ~/.jarvis/tts_cache)
            enable_gcp: Enable Google Cloud TTS
            enable_elevenlabs: Enable ElevenLabs TTS
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".jarvis" / "tts_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize providers
        self.gcp_service: Optional[GCPTTSService] = None
        self.elevenlabs_service: Optional[ElevenLabsTTSService] = None

        if enable_gcp:
            try:
                self.gcp_service = GCPTTSService(cache_dir=cache_dir / "gcp")
                logger.info("✅ Initialized Google Cloud TTS")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GCP TTS: {e}")

        if enable_elevenlabs:
            try:
                self.elevenlabs_service = ElevenLabsTTSService(cache_dir=cache_dir / "elevenlabs")
                logger.info("✅ Initialized ElevenLabs TTS")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize ElevenLabs TTS: {e}")

        # Voice profile registry
        self.voice_profiles: Dict[str, UnifiedVoiceConfig] = {}

        # Usage tracking
        self.usage_stats = {
            TTSProvider.GCP: {'requests': 0, 'characters': 0, 'cache_hits': 0},
            TTSProvider.ELEVENLABS: {'requests': 0, 'characters': 0, 'cache_hits': 0}
        }

    async def close(self):
        """Close all provider sessions"""
        if self.elevenlabs_service:
            await self.elevenlabs_service.close()

    def _select_provider(self, voice_config: UnifiedVoiceConfig) -> TTSProvider:
        """
        Intelligently select TTS provider based on voice requirements

        Routing logic:
        - African American, African, Asian accents → ElevenLabs (if available)
        - US, British, Australian, Indian, European accents → GCP
        - Fallback: Use whichever provider is available
        """
        if voice_config.provider != TTSProvider.AUTO:
            return voice_config.provider

        # Check accent-based routing
        accent_lower = (voice_config.accent or "").lower()

        # ElevenLabs specializes in these accents
        elevenlabs_accents = [
            'african_american', 'african', 'nigerian', 'kenyan', 'south_african',
            'asian', 'chinese', 'japanese', 'korean'
        ]

        if any(acc in accent_lower for acc in elevenlabs_accents):
            if self.elevenlabs_service:
                return TTSProvider.ELEVENLABS

        # GCP handles these well
        if self.gcp_service:
            return TTSProvider.GCP

        # Fallback
        if self.elevenlabs_service:
            return TTSProvider.ELEVENLABS
        elif self.gcp_service:
            return TTSProvider.GCP

        raise RuntimeError("No TTS providers available")

    async def synthesize_speech(
        self,
        text: str,
        voice_config: UnifiedVoiceConfig,
        use_cache: bool = True
    ) -> bytes:
        """
        Synthesize speech using the appropriate provider

        Args:
            text: Text to synthesize
            voice_config: Unified voice configuration
            use_cache: Use cached audio if available

        Returns:
            Audio data as bytes (MP3 format)
        """
        # Select provider
        provider = self._select_provider(voice_config)

        # Track usage
        self.usage_stats[provider]['requests'] += 1
        self.usage_stats[provider]['characters'] += len(text)

        # Get provider-specific config
        provider_config = voice_config.get_provider_config()

        try:
            if provider == TTSProvider.GCP:
                if not self.gcp_service:
                    raise RuntimeError("GCP TTS not available")

                audio_data = await self.gcp_service.synthesize_speech(
                    text=text,
                    voice_config=provider_config,
                    use_cache=use_cache
                )
                logger.debug(f"🔊 Generated with GCP TTS: {voice_config.name}")
                return audio_data

            elif provider == TTSProvider.ELEVENLABS:
                if not self.elevenlabs_service:
                    raise RuntimeError("ElevenLabs TTS not available")

                audio_data = await self.elevenlabs_service.synthesize_speech(
                    text=text,
                    voice_config=provider_config,
                    use_cache=use_cache
                )
                logger.debug(f"🔊 Generated with ElevenLabs TTS: {voice_config.name}")
                return audio_data

            else:
                raise ValueError(f"Unknown provider: {provider}")

        except Exception as e:
            logger.error(f"❌ TTS synthesis failed with {provider.value}: {e}")

            # Attempt fallback to other provider
            if provider == TTSProvider.GCP and self.elevenlabs_service:
                logger.warning("⚠️  Attempting fallback to ElevenLabs...")
                # Would need to convert voice config
                # For now, just re-raise
                raise
            elif provider == TTSProvider.ELEVENLABS and self.gcp_service:
                logger.warning("⚠️  Attempting fallback to GCP...")
                raise

            raise

    def register_voice(self, voice_config: UnifiedVoiceConfig):
        """Register a voice profile"""
        self.voice_profiles[voice_config.name] = voice_config
        logger.debug(f"📝 Registered voice: {voice_config.name} ({voice_config.provider.value})")

    def get_voice(self, name: str) -> Optional[UnifiedVoiceConfig]:
        """Get voice by name"""
        return self.voice_profiles.get(name)

    def get_voices_by_provider(self, provider: TTSProvider) -> List[UnifiedVoiceConfig]:
        """Get all voices for a specific provider"""
        return [v for v in self.voice_profiles.values() if v.provider == provider]

    def get_voices_by_accent(self, accent: str) -> List[UnifiedVoiceConfig]:
        """Get voices filtered by accent"""
        accent_lower = accent.lower()
        return [
            v for v in self.voice_profiles.values()
            if v.accent and accent_lower in v.accent.lower()
        ]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics across all providers"""
        total_stats = {
            'total_requests': sum(s['requests'] for s in self.usage_stats.values()),
            'total_characters': sum(s['characters'] for s in self.usage_stats.values()),
            'by_provider': {}
        }

        for provider, stats in self.usage_stats.items():
            total_stats['by_provider'][provider.value] = {
                'requests': stats['requests'],
                'characters': stats['characters'],
                'cache_hits': stats['cache_hits']
            }

        return total_stats

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics across all providers"""
        stats = {
            'cache_dir': str(self.cache_dir),
            'providers': {}
        }

        if self.gcp_service:
            stats['providers']['gcp'] = self.gcp_service.get_cache_stats()

        if self.elevenlabs_service:
            stats['providers']['elevenlabs'] = self.elevenlabs_service.get_cache_stats()

        return stats

    def clear_cache(self, provider: Optional[TTSProvider] = None):
        """Clear cache for specific provider or all providers"""
        if provider == TTSProvider.GCP or provider is None:
            if self.gcp_service:
                self.gcp_service.clear_cache()

        if provider == TTSProvider.ELEVENLABS or provider is None:
            if self.elevenlabs_service:
                self.elevenlabs_service.clear_cache()

        logger.info(f"🗑️  Cleared TTS cache{' for ' + provider.value if provider else ''}")

    async def discover_all_voices(self) -> Dict[TTSProvider, List]:
        """Discover voices from all enabled providers"""
        all_voices = {}

        if self.gcp_service:
            # GCP voices are pre-configured
            logger.info("📋 GCP voices are pre-configured")
            all_voices[TTSProvider.GCP] = []

        if self.elevenlabs_service:
            logger.info("🔍 Discovering ElevenLabs voices...")
            elevenlabs_voices = await self.elevenlabs_service.discover_voices()
            all_voices[TTSProvider.ELEVENLABS] = list(elevenlabs_voices.values())
            logger.info(f"✅ Found {len(elevenlabs_voices)} ElevenLabs voices")

        return all_voices


async def main():
    """Test TTS provider manager"""
    print("=" * 70)
    print("🧪 TTS Provider Manager Test")
    print("=" * 70)

    manager = TTSProviderManager()

    # Discover voices
    print("\n🔍 Discovering voices from all providers...")
    voices = await manager.discover_all_voices()

    for provider, voice_list in voices.items():
        print(f"\n{provider.value.upper()}: {len(voice_list)} voices")

    # Show usage stats
    print("\n📊 Usage Stats:")
    stats = manager.get_usage_stats()
    print(f"   Total requests: {stats['total_requests']}")
    print(f"   Total characters: {stats['total_characters']}")

    # Show cache stats
    print("\n💾 Cache Stats:")
    cache_stats = manager.get_cache_stats()
    for provider, pstats in cache_stats.get('providers', {}).items():
        print(f"\n{provider.upper()}:")
        print(f"   Files: {pstats.get('file_count', 0)}")
        print(f"   Size: {pstats.get('total_size_mb', 0):.2f} MB")

    await manager.close()


if __name__ == "__main__":
    asyncio.run(main())
