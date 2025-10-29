"""
Base STT Engine Interface
All engines must implement this interface
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from ..stt_config import ModelConfig, STTEngine

logger = logging.getLogger(__name__)


@dataclass
class STTResult:
    """Result from STT engine"""

    text: str
    confidence: float
    engine: STTEngine
    model_name: str
    latency_ms: float
    audio_duration_ms: float
    metadata: Dict = field(default_factory=dict)
    audio_hash: Optional[str] = None
    speaker_identified: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class BaseSTTEngine(ABC):
    """Base class for all STT engines"""

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self.model = None
        self.initialized = False
        logger.info(f"Creating {model_config.engine.value} engine: {model_config.name}")

    @abstractmethod
    async def initialize(self):
        """Initialize the engine (load model, etc.)"""

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> STTResult:
        """Transcribe audio data"""

    async def cleanup(self):
        """Cleanup resources"""
        self.model = None
        self.initialized = False
        logger.info(f"Cleaned up {self.model_config.name}")

    def __repr__(self):
        return f"<{self.__class__.__name__}(model={self.model_config.name}, initialized={self.initialized})>"


def create_stt_engine(model_config: ModelConfig) -> BaseSTTEngine:
    """Factory function to create STT engines based on configuration"""

    if model_config.engine == STTEngine.WAV2VEC2:
        from .wav2vec2_engine import Wav2Vec2Engine

        return Wav2Vec2Engine(model_config)
    elif model_config.engine == STTEngine.SPEECHBRAIN:
        from .speechbrain_engine import SpeechBrainEngine

        return SpeechBrainEngine(model_config)
    elif model_config.engine == STTEngine.WHISPER:
        from .whisper_engine import WhisperEngine

        return WhisperEngine(model_config)
    elif model_config.engine == STTEngine.VOSK:
        from .vosk_engine import VoskEngine

        return VoskEngine(model_config)
    elif model_config.engine == STTEngine.MLKIT:
        from .mlkit_engine import MLKitEngine

        return MLKitEngine(model_config)
    else:
        raise ValueError(f"Unknown STT engine: {model_config.engine}")
