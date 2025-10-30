"""
Base STT Engine Interface

This module provides the foundational interface and data structures for all Speech-to-Text (STT) engines.
It defines the abstract base class that all STT engines must implement, along with result data structures
and a factory function for creating engine instances.

The module supports multiple STT engines including Wav2Vec2, SpeechBrain, Whisper, Vosk, and MLKit,
providing a unified interface for speech recognition across different backends.
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
    """Result from STT engine containing transcription and metadata.
    
    This dataclass encapsulates all information returned by an STT engine after
    processing audio data, including the transcribed text, confidence scores,
    performance metrics, and optional metadata.
    
    Attributes:
        text: The transcribed text from the audio
        confidence: Confidence score between 0.0 and 1.0
        engine: The STT engine type that produced this result
        model_name: Name of the specific model used
        latency_ms: Processing time in milliseconds
        audio_duration_ms: Duration of the input audio in milliseconds
        metadata: Additional engine-specific metadata
        audio_hash: Optional hash of the input audio for caching/deduplication
        speaker_identified: Optional speaker identification result
        timestamp: When the transcription was completed
    """

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
    """Base class for all STT engines.
    
    This abstract base class defines the interface that all STT engines must implement.
    It provides common initialization patterns and ensures consistent behavior across
    different engine implementations.
    
    Attributes:
        model_config: Configuration object containing model settings
        model: The loaded model instance (engine-specific)
        initialized: Whether the engine has been successfully initialized
    """

    def __init__(self, model_config: ModelConfig) -> None:
        """Initialize the base STT engine.
        
        Args:
            model_config: Configuration object containing engine and model settings
        """
        self.model_config = model_config
        self.model = None
        self.initialized = False
        logger.info(f"Creating {model_config.engine.value} engine: {model_config.name}")

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the engine (load model, etc.).
        
        This method must be implemented by each engine to handle model loading,
        resource allocation, and any other initialization tasks required before
        the engine can process audio.
        
        Raises:
            NotImplementedError: If not implemented by subclass
            RuntimeError: If initialization fails
        """
        pass

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> STTResult:
        """Transcribe audio data to text.
        
        This method must be implemented by each engine to process raw audio data
        and return a structured result containing the transcription and metadata.
        
        Args:
            audio_data: Raw audio data in bytes format
            
        Returns:
            STTResult containing transcription text, confidence, and metadata
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If audio_data is invalid or empty
            RuntimeError: If transcription fails
        """
        pass

    async def cleanup(self) -> None:
        """Cleanup resources used by the engine.
        
        This method releases any resources held by the engine, including
        loaded models and allocated memory. Should be called when the
        engine is no longer needed.
        """
        self.model = None
        self.initialized = False
        logger.info(f"Cleaned up {self.model_config.name}")

    def __repr__(self) -> str:
        """Return string representation of the engine.
        
        Returns:
            String representation showing class name, model name, and initialization status
        """
        return f"<{self.__class__.__name__}(model={self.model_config.name}, initialized={self.initialized})>"


def create_stt_engine(model_config: ModelConfig) -> BaseSTTEngine:
    """Factory function to create STT engines based on configuration.
    
    This factory function creates and returns the appropriate STT engine instance
    based on the engine type specified in the model configuration. It handles
    the dynamic importing of engine-specific classes to avoid circular imports.
    
    Args:
        model_config: Configuration object specifying the engine type and settings
        
    Returns:
        An instance of the appropriate STT engine subclass
        
    Raises:
        ValueError: If the specified engine type is not supported
        ImportError: If the required engine module cannot be imported
        
    Example:
        >>> from backend.voice.stt_config import ModelConfig, STTEngine
        >>> config = ModelConfig(name="whisper-base", engine=STTEngine.WHISPER)
        >>> engine = create_stt_engine(config)
        >>> isinstance(engine, BaseSTTEngine)
        True
    """
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