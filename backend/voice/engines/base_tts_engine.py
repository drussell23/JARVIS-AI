"""
Base TTS Engine Interface
Defines common interface for all Text-to-Speech engines
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TTSEngine(Enum):
    """Supported TTS engines"""

    GTTS = "gtts"  # Google Text-to-Speech (free, online)
    COQUI = "coqui"  # Coqui TTS (local, neural)
    PYTTSX3 = "pyttsx3"  # System TTS (local, fast)
    ELEVENLABS = "elevenlabs"  # ElevenLabs (premium, online)
    MACOS = "macos"  # macOS native (local, fast)


@dataclass
class TTSConfig:
    """Configuration for TTS engine"""

    name: str
    engine: TTSEngine
    language: str = "en"
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    sample_rate: int = 22050
    model_path: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class TTSResult:
    """Result of TTS synthesis"""

    audio_data: bytes
    sample_rate: int
    duration_ms: float
    latency_ms: float
    engine: TTSEngine
    voice: str
    metadata: Dict


class BaseTTSEngine(ABC):
    """Base class for all TTS engines"""

    def __init__(self, config: TTSConfig):
        self.config = config
        self.initialized = False

    @abstractmethod
    async def initialize(self):
        """Initialize the TTS engine"""

    @abstractmethod
    async def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize

        Returns:
            TTSResult with audio data and metadata
        """

    @abstractmethod
    async def get_available_voices(self) -> List[str]:
        """Get list of available voices for this engine"""

    @abstractmethod
    async def cleanup(self):
        """Cleanup engine resources"""

    def is_initialized(self) -> bool:
        """Check if engine is initialized"""
        return self.initialized
