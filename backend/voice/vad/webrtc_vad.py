#!/usr/bin/env python3
"""
WebRTC VAD Implementation
Lightweight frame-level speech/non-speech classifier using webrtcvad
"""

import logging
from typing import Iterator
import numpy as np

try:
    import webrtcvad
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logging.warning("webrtcvad not available - install with: pip install webrtcvad")

from .base import VADBase, VADConfig, SpeechSegment

logger = logging.getLogger(__name__)


class WebRTCVAD(VADBase):
    """
    WebRTC-based Voice Activity Detection

    Fast, lightweight frame-level VAD using Google's WebRTC library.
    Processes audio in small frames (10-30ms) and classifies each as speech/non-speech.

    Features:
    - Very low latency (10-30ms frame processing)
    - Low CPU usage
    - Configurable aggressiveness (0-3)
    - Supports 8kHz, 16kHz, 32kHz, 48kHz sample rates
    - Frame durations: 10ms, 20ms, 30ms
    """

    def __init__(self, config: VADConfig):
        """
        Initialize WebRTC VAD

        Args:
            config: VAD configuration

        Raises:
            ImportError: If webrtcvad library not available
            ValueError: If configuration parameters invalid
        """
        if not WEBRTC_AVAILABLE:
            raise ImportError("webrtcvad library required. Install with: pip install webrtcvad")

        super().__init__(config)

        # Validate WebRTC-specific requirements
        self._validate_config()

        # Create WebRTC VAD instance
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(config.aggressiveness)

        logger.info(f"🎤 WebRTC VAD initialized:")
        logger.info(f"   Sample rate: {self.sample_rate}Hz")
        logger.info(f"   Frame duration: {self.frame_duration_ms}ms")
        logger.info(f"   Frame size: {self.frame_size} samples")
        logger.info(f"   Aggressiveness: {config.aggressiveness}")

    def _validate_config(self):
        """Validate configuration for WebRTC VAD requirements"""
        # WebRTC supports only specific sample rates
        valid_sample_rates = [8000, 16000, 32000, 48000]
        if self.sample_rate not in valid_sample_rates:
            raise ValueError(
                f"WebRTC VAD requires sample rate in {valid_sample_rates}, got {self.sample_rate}"
            )

        # WebRTC supports only specific frame durations
        valid_frame_durations = [10, 20, 30]
        if self.frame_duration_ms not in valid_frame_durations:
            raise ValueError(
                f"WebRTC VAD requires frame duration in {valid_frame_durations}ms, got {self.frame_duration_ms}ms"
            )

        # Aggressiveness must be 0-3
        if not 0 <= self.config.aggressiveness <= 3:
            raise ValueError(
                f"WebRTC VAD aggressiveness must be 0-3, got {self.config.aggressiveness}"
            )

    def is_speech(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Determine if a single audio frame contains speech

        Args:
            frame: Audio frame (float32, normalized to [-1, 1])
                  Length must be exactly frame_size

        Returns:
            Tuple of (is_speech: bool, confidence: float)
            - WebRTC returns binary decision, so confidence is 0.0 or 1.0
        """
        # Validate frame size
        if len(frame) != self.frame_size:
            logger.warning(f"Frame size mismatch: expected {self.frame_size}, got {len(frame)}")
            return False, 0.0

        # Convert float32 to int16 PCM
        pcm_frame = self.float_to_int16(frame)

        # WebRTC VAD requires bytes
        frame_bytes = pcm_frame.tobytes()

        # Check if speech
        try:
            is_speech = self.vad.is_speech(frame_bytes, self.sample_rate)
            # WebRTC is binary, so confidence is either 0.0 or 1.0
            confidence = 1.0 if is_speech else 0.0
            return is_speech, confidence
        except Exception as e:
            logger.error(f"WebRTC VAD error: {e}")
            return False, 0.0

    def process_audio(self, audio: np.ndarray) -> Iterator[SpeechSegment]:
        """
        Process entire audio buffer and yield speech segments

        Uses a state machine to track speech/silence transitions:
        1. Silent state: Accumulate frames until speech detected
        2. Speech state: Accumulate frames until silence detected
        3. Emit speech segment when silence threshold reached

        Args:
            audio: Audio data (float32, normalized to [-1, 1])

        Yields:
            SpeechSegment: Detected speech segments
        """
        if len(audio) == 0:
            return

        # State tracking
        in_speech = False
        speech_frames = []
        speech_start_sample = 0
        silence_frames = 0

        # Calculate thresholds in frames
        min_speech_frames = int(self.config.min_speech_duration_ms / self.frame_duration_ms)
        max_silence_frames = int(self.config.max_silence_duration_ms / self.frame_duration_ms)
        padding_frames = int(self.config.padding_duration_ms / self.frame_duration_ms)

        # Process audio frame by frame
        num_frames = len(audio) // self.frame_size

        for i in range(num_frames):
            start_idx = i * self.frame_size
            end_idx = start_idx + self.frame_size
            frame = audio[start_idx:end_idx]

            # Check if frame contains speech
            is_speech_frame, confidence = self.is_speech(frame)

            if not in_speech:
                # Currently in silence - check if speech started
                if is_speech_frame:
                    # Start new speech segment
                    in_speech = True
                    speech_start_sample = max(0, start_idx - padding_frames * self.frame_size)
                    speech_frames = []
                    silence_frames = 0
            else:
                # Currently in speech
                if is_speech_frame:
                    # Continue speech
                    silence_frames = 0
                else:
                    # Silence during speech
                    silence_frames += 1

                    # Check if silence too long - end speech segment
                    if silence_frames >= max_silence_frames:
                        # End speech segment
                        speech_end_sample = min(
                            len(audio),
                            end_idx + padding_frames * self.frame_size
                        )

                        # Extract speech segment
                        segment_audio = audio[speech_start_sample:speech_end_sample]

                        # Only yield if meets minimum duration
                        num_speech_frames = len(speech_frames)
                        if num_speech_frames >= min_speech_frames:
                            duration_ms = (len(segment_audio) / self.sample_rate) * 1000

                            yield SpeechSegment(
                                start_sample=speech_start_sample,
                                end_sample=speech_end_sample,
                                audio_data=segment_audio,
                                confidence=1.0,  # WebRTC is binary
                                duration_ms=duration_ms
                            )

                        # Reset state
                        in_speech = False
                        speech_frames = []
                        silence_frames = 0

            # Track speech frames
            if in_speech:
                speech_frames.append(i)

        # Handle case where speech continues to end of audio
        if in_speech and len(speech_frames) >= min_speech_frames:
            speech_end_sample = len(audio)
            segment_audio = audio[speech_start_sample:speech_end_sample]
            duration_ms = (len(segment_audio) / self.sample_rate) * 1000

            yield SpeechSegment(
                start_sample=speech_start_sample,
                end_sample=speech_end_sample,
                audio_data=segment_audio,
                confidence=1.0,
                duration_ms=duration_ms
            )
