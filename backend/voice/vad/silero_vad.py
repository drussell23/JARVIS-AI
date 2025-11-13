#!/usr/bin/env python3
"""
Silero VAD Implementation
Neural-network based VAD for accurate speech detection
"""

import logging
from typing import Iterator, Optional
import numpy as np
import asyncio
import torch

try:
    # Silero VAD is loaded via torch.hub
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False
    logging.warning("PyTorch not available - Silero VAD will not work")

from .base import VADBase, VADConfig, SpeechSegment

logger = logging.getLogger(__name__)


class SileroVAD(VADBase):
    """
    Silero VAD - Neural network-based Voice Activity Detection

    More accurate than WebRTC VAD, especially for:
    - Noisy environments
    - Different accents and languages
    - Music/speech separation
    - Low-quality audio

    Features:
    - Neural network inference (ONNX or PyTorch)
    - Confidence scores (not just binary)
    - Supports 8kHz and 16kHz
    - Better noise robustness
    """

    def __init__(self, config: VADConfig):
        """
        Initialize Silero VAD

        Args:
            config: VAD configuration

        Raises:
            ImportError: If PyTorch not available
        """
        if not SILERO_AVAILABLE:
            raise ImportError("PyTorch required for Silero VAD. Install with: pip install torch")

        super().__init__(config)

        self.model = None
        self.device = torch.device('cpu')  # Use CPU for consistency
        self._load_model()

        logger.info(f"🧠 Silero VAD initialized:")
        logger.info(f"   Sample rate: {self.sample_rate}Hz")
        logger.info(f"   Speech threshold: {config.speech_threshold}")
        logger.info(f"   Device: {self.device}")

    def _load_model(self):
        """Load Silero VAD model from torch hub"""
        try:
            logger.info("Loading Silero VAD model from torch.hub...")

            # Load model from Silero's repository
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False  # Use PyTorch model, not ONNX
            )

            self.model = model.to(self.device)
            self.model.eval()

            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = utils

            logger.info("✅ Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD model: {e}")
            raise

    def is_speech(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Determine if a single audio frame contains speech using Silero

        Args:
            frame: Audio frame (float32, normalized to [-1, 1])

        Returns:
            Tuple of (is_speech: bool, confidence: float)
            - Silero returns probability score 0.0-1.0
        """
        if self.model is None:
            logger.warning("Silero model not loaded")
            return False, 0.0

        # Validate frame
        if len(frame) == 0:
            return False, 0.0

        try:
            # Silero expects exactly 512 samples (16kHz) or 256 samples (8kHz)
            chunk_size = 512 if self.sample_rate == 16000 else 256

            # Pad or truncate frame to expected size
            if len(frame) < chunk_size:
                frame = np.pad(frame, (0, chunk_size - len(frame)), mode='constant')
            elif len(frame) > chunk_size:
                frame = frame[:chunk_size]

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(frame).float()

            # Silero expects batch dimension: (batch, samples)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Run inference
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()

            # Apply threshold
            is_speech = speech_prob >= self.config.speech_threshold

            return is_speech, speech_prob

        except Exception as e:
            logger.error(f"Silero inference error: {e}")
            return False, 0.0

    def process_audio(self, audio: np.ndarray) -> Iterator[SpeechSegment]:
        """
        Process entire audio buffer and yield speech segments using Silero

        Silero provides get_speech_timestamps() which returns speech segments
        with start/end timestamps and confidence scores.

        Args:
            audio: Audio data (float32, normalized to [-1, 1])

        Yields:
            SpeechSegment: Detected speech segments
        """
        if len(audio) == 0:
            return

        if self.model is None:
            logger.warning("Silero model not loaded")
            return

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()

            # Get speech timestamps from Silero
            # Returns: [{'start': sample_idx, 'end': sample_idx}, ...]
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.config.speech_threshold,
                min_speech_duration_ms=self.config.min_speech_duration_ms,
                max_speech_duration_s=30,  # Max 30s per segment
                min_silence_duration_ms=self.config.max_silence_duration_ms,
                window_size_samples=512,  # Default for 16kHz
                speech_pad_ms=self.config.padding_duration_ms
            )

            # Convert timestamps to SpeechSegment objects
            for timestamp in speech_timestamps:
                start_sample = timestamp['start']
                end_sample = timestamp['end']

                # Extract audio segment
                segment_audio = audio[start_sample:end_sample]

                # Calculate duration
                duration_ms = (len(segment_audio) / self.sample_rate) * 1000

                # Get average confidence for this segment
                # Silero expects exactly 512 samples (16kHz) or 256 samples (8kHz)
                # We need to chunk the segment and average the confidence scores
                chunk_size = 512 if self.sample_rate == 16000 else 256
                confidences = []

                for i in range(0, len(segment_audio), chunk_size):
                    chunk = segment_audio[i:i + chunk_size]

                    # Skip if chunk is too small
                    if len(chunk) < chunk_size:
                        # Pad the last chunk to chunk_size
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), mode='constant')

                    chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                    with torch.no_grad():
                        chunk_confidence = self.model(chunk_tensor, self.sample_rate).item()
                        confidences.append(chunk_confidence)

                # Average confidence across all chunks
                confidence = np.mean(confidences) if confidences else self.config.speech_threshold

                yield SpeechSegment(
                    start_sample=start_sample,
                    end_sample=end_sample,
                    audio_data=segment_audio,
                    confidence=confidence,
                    duration_ms=duration_ms
                )

        except Exception as e:
            logger.error(f"Silero process_audio error: {e}")
            return


class AsyncSileroVAD(SileroVAD):
    """
    Async wrapper for Silero VAD to prevent blocking

    Runs Silero inference in thread pool executor to keep
    the async event loop responsive.
    """

    async def is_speech_async(self, frame: np.ndarray) -> tuple[bool, float]:
        """
        Async version of is_speech - runs in thread pool

        Args:
            frame: Audio frame (float32)

        Returns:
            Tuple of (is_speech: bool, confidence: float)
        """
        return await asyncio.to_thread(self.is_speech, frame)

    async def process_audio_async(self, audio: np.ndarray) -> list[SpeechSegment]:
        """
        Async version of process_audio - runs in thread pool

        Args:
            audio: Audio data (float32)

        Returns:
            List of detected speech segments
        """
        def _process_sync():
            return list(self.process_audio(audio))

        return await asyncio.to_thread(_process_sync)

    async def filter_silence_async(self, audio: np.ndarray) -> np.ndarray:
        """
        Async version of filter_silence - runs in thread pool

        Args:
            audio: Audio data (float32)

        Returns:
            Filtered audio with silence removed
        """
        return await asyncio.to_thread(self.filter_silence, audio)
