#!/usr/bin/env python3
"""
VAD Base Interface
Abstract base class for Voice Activity Detection implementations
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class VADConfig:
    """Configuration for VAD processing"""
    sample_rate: int = 16000
    frame_duration_ms: int = 30  # Frame size in milliseconds (10, 20, or 30)
    aggressiveness: int = 2  # VAD aggressiveness level (0-3)
    speech_threshold: float = 0.5  # Minimum probability to consider speech
    min_speech_duration_ms: int = 300  # Minimum continuous speech duration
    max_silence_duration_ms: int = 300  # Maximum silence before cutting speech
    padding_duration_ms: int = 200  # Padding before/after speech segments


@dataclass
class SpeechSegment:
    """Single speech segment detected by VAD"""
    start_sample: int
    end_sample: int
    audio_data: np.ndarray
    confidence: float
    duration_ms: float

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds"""
        return self.duration_ms / 1000.0


class VADBase(ABC):
    """
    Abstract base class for Voice Activity Detection

    All VAD implementations must inherit from this class and implement
    the required methods for frame-level and batch-level processing.
    """

    def __init__(self, config: VADConfig):
        """
        Initialize VAD with configuration

        Args:
            config: VAD configuration parameters
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.frame_duration_ms = config.frame_duration_ms

        # Calculate frame size in samples
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)

    @abstractmethod
    def is_speech(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Determine if a single audio frame contains speech

        Args:
            frame: Audio frame (numpy array, float32, normalized to [-1, 1])
                  Length must match frame_size

        Returns:
            Tuple of (is_speech: bool, confidence: float)
            - is_speech: True if frame contains speech
            - confidence: Probability/confidence score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def process_audio(self, audio: np.ndarray) -> Iterator[SpeechSegment]:
        """
        Process entire audio buffer and yield speech segments

        Args:
            audio: Audio data (numpy array, float32, normalized to [-1, 1])
                  Sample rate must match config.sample_rate

        Yields:
            SpeechSegment: Detected speech segments with timing and confidence
        """
        pass

    def filter_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove silence from audio, keeping only speech segments

        Args:
            audio: Audio data (numpy array, float32)

        Returns:
            Filtered audio with silence removed
        """
        speech_segments = list(self.process_audio(audio))

        if not speech_segments:
            # No speech detected - return empty array
            return np.array([], dtype=np.float32)

        # Concatenate all speech segments
        filtered = np.concatenate([seg.audio_data for seg in speech_segments])
        return filtered

    def get_speech_ratio(self, audio: np.ndarray) -> float:
        """
        Calculate ratio of speech to total audio duration

        Args:
            audio: Audio data (numpy array, float32)

        Returns:
            Speech ratio (0.0 to 1.0)
        """
        total_duration = len(audio) / self.sample_rate

        if total_duration == 0:
            return 0.0

        speech_segments = list(self.process_audio(audio))
        speech_duration = sum(seg.duration_seconds for seg in speech_segments)

        return min(speech_duration / total_duration, 1.0)

    @staticmethod
    def float_to_int16(audio: np.ndarray) -> np.ndarray:
        """
        Convert float32 audio [-1, 1] to int16 PCM

        Args:
            audio: Float32 audio normalized to [-1, 1]

        Returns:
            Int16 PCM audio
        """
        return (audio * 32767).astype(np.int16)

    @staticmethod
    def int16_to_float(audio: np.ndarray) -> np.ndarray:
        """
        Convert int16 PCM audio to float32 [-1, 1]

        Args:
            audio: Int16 PCM audio

        Returns:
            Float32 audio normalized to [-1, 1]
        """
        return audio.astype(np.float32) / 32768.0
