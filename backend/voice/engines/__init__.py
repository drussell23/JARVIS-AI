"""
STT Engine Implementations
Each engine is self-contained and implements the base interface
"""

from .base_engine import BaseSTTEngine, STTResult
from .vosk_engine import VoskEngine
from .wav2vec_engine import Wav2VecEngine
from .whisper_gcp_engine import WhisperGCPEngine
from .whisper_local_engine import WhisperLocalEngine

__all__ = [
    "BaseSTTEngine",
    "STTResult",
    "VoskEngine",
    "Wav2VecEngine",
    "WhisperLocalEngine",
    "WhisperGCPEngine",
]
