"""
Whisper Local STT Engine
OpenAI Whisper running locally with CoreML optimization (macOS)
Balanced accuracy and performance for medium-RAM scenarios
"""

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from ..stt_config import STTEngine
from .base_engine import BaseSTTEngine, STTResult

logger = logging.getLogger(__name__)


class WhisperLocalEngine(BaseSTTEngine):
    """
    OpenAI Whisper running locally.

    Features:
    - High accuracy (76-95% depending on model size)
    - Multiple model sizes (tiny, base, small, medium, large)
    - CoreML optimization on macOS (M1/M2/M3)
    - CPU and GPU support
    - Multilingual support
    - 1-5GB RAM depending on model
    """

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model_size = model_config.model_path or "base"  # tiny, base, small, medium, large
        self.device = None
        self.models_dir = Path.home() / ".jarvis" / "models" / "stt" / "whisper"

    async def initialize(self):
        """Initialize Whisper model with optimal settings"""
        if self.initialized:
            return

        logger.info(f"🔧 Initializing Whisper Local: {self.model_config.name}")
        logger.info(f"   Model size: {self.model_size}")

        try:
            # Import whisper (lazy import)
            import whisper

            # Ensure models directory exists
            self.models_dir.mkdir(parents=True, exist_ok=True)

            # Determine optimal device
            self.device = self._get_optimal_device()
            logger.info(f"   Using device: {self.device}")

            # Load model
            logger.info(f"   Loading Whisper {self.model_size} model...")
            self.model = await asyncio.to_thread(
                whisper.load_model,
                self.model_size,
                device=self.device,
                download_root=str(self.models_dir),
            )

            self.initialized = True
            logger.info(f"✅ Whisper Local initialized: {self.model_config.name}")

        except Exception as e:
            logger.error(f"Failed to initialize Whisper Local: {e}")
            raise

    def _get_optimal_device(self) -> str:
        """Determine optimal device for Whisper"""
        import torch

        # Whisper uses PyTorch, check for available devices
        if torch.cuda.is_available():
            logger.info("   🎮 CUDA GPU available")
            return "cuda"

        # Whisper doesn't natively support MPS yet, use CPU on macOS
        # CoreML optimization is available through separate package
        logger.info("   💻 Using CPU")
        return "cpu"

    async def transcribe(self, audio_data: bytes) -> STTResult:
        """
        Transcribe audio using Whisper.

        Args:
            audio_data: Raw audio bytes (any format)

        Returns:
            STTResult with transcription and confidence
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Convert audio bytes to numpy array
            audio_array = await self._bytes_to_audio_array(audio_data)
            audio_duration_ms = len(audio_array) / 16000 * 1000

            # Transcribe with Whisper
            result = await asyncio.to_thread(
                self.model.transcribe,
                audio_array,
                language="en",  # Derek speaks English
                fp16=False,  # Use FP32 for CPU (more stable)
                verbose=False,
            )

            # Extract transcription
            transcription_text = result.get("text", "").strip()

            # Calculate confidence from segment data
            confidence = self._calculate_confidence(result)

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"🎤 Whisper transcribed: '{transcription_text[:50]}...' "
                f"(confidence={confidence:.2f}, latency={latency_ms:.0f}ms)"
            )

            return STTResult(
                text=transcription_text,
                confidence=confidence,
                engine=STTEngine.WHISPER_LOCAL,
                model_name=self.model_config.name,
                latency_ms=latency_ms,
                audio_duration_ms=audio_duration_ms,
                metadata={
                    "device": self.device,
                    "model_size": self.model_size,
                    "language": result.get("language", "en"),
                    "segments": result.get("segments", []),
                },
            )

        except Exception as e:
            logger.error(f"Whisper Local transcription failed: {e}")
            raise

    async def _bytes_to_audio_array(self, audio_data: bytes) -> np.ndarray:
        """
        Convert audio bytes to numpy array suitable for Whisper.

        Whisper expects 16kHz mono float32.
        """
        try:
            # Try with librosa
            import io

            import librosa

            # Load and resample to 16kHz mono
            audio_array, sr = await asyncio.to_thread(
                librosa.load,
                io.BytesIO(audio_data),
                sr=16000,
                mono=True,
            )

            return audio_array

        except ImportError:
            logger.warning("librosa not available, using scipy")
            # Fallback to scipy
            import io

            import scipy.io.wavfile as wavfile
            import scipy.signal as signal

            sr, audio_array = await asyncio.to_thread(wavfile.read, io.BytesIO(audio_data))

            # Convert to mono
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)

            # Convert to float32 normalized to [-1, 1]
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            elif audio_array.dtype == np.int32:
                audio_array = audio_array.astype(np.float32) / 2147483648.0

            # Resample to 16kHz if needed
            if sr != 16000:
                num_samples = int(len(audio_array) * 16000 / sr)
                audio_array = await asyncio.to_thread(signal.resample, audio_array, num_samples)

            return audio_array

    def _calculate_confidence(self, result: dict) -> float:
        """
        Calculate confidence from Whisper result.

        Whisper provides no_speech_prob and avg_logprob per segment.
        """
        try:
            segments = result.get("segments", [])

            if not segments:
                return 0.5  # No segments, moderate confidence

            # Calculate confidence from segment probabilities
            confidences = []
            for segment in segments:
                # Whisper provides avg_logprob (negative value, closer to 0 is better)
                avg_logprob = segment.get("avg_logprob", -1.0)

                # Whisper also provides no_speech_prob (probability of silence)
                no_speech_prob = segment.get("no_speech_prob", 0.0)

                # Convert logprob to confidence
                # avg_logprob typically ranges from -1.0 (good) to -3.0 (bad)
                # Map to [0, 1] range
                confidence = max(0.0, min(1.0, (avg_logprob + 3.0) / 2.0))

                # Penalize if likely silence
                confidence *= 1.0 - no_speech_prob

                confidences.append(confidence)

            # Weight by segment duration
            total_duration = sum(s.get("end", 0) - s.get("start", 0) for s in segments)
            if total_duration == 0:
                return 0.5

            weighted_confidence = (
                sum(
                    conf * (seg.get("end", 0) - seg.get("start", 0))
                    for conf, seg in zip(confidences, segments)
                )
                / total_duration
            )

            return max(0.0, min(1.0, weighted_confidence))

        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5

    async def cleanup(self):
        """Cleanup Whisper resources"""
        if self.model is not None:
            del self.model
            self.model = None

            # Clear CUDA cache if used
            if self.device == "cuda":
                import torch

                torch.cuda.empty_cache()

        await super().cleanup()
        logger.info(f"🧹 Whisper Local engine cleaned up: {self.model_config.name}")
