"""
Vosk STT Engine
Ultra-fast, lightweight, CPU-only engine for low-latency transcription
Perfect for resource-constrained environments and fast fallback
"""

import asyncio
import json
import logging
import time
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

from ..stt_config import STTEngine
from .base_engine import BaseSTTEngine, STTResult

logger = logging.getLogger(__name__)


class VoskEngine(BaseSTTEngine):
    """
    Vosk engine - ultra-fast, lightweight, CPU-only STT.

    Features:
    - No GPU required (pure CPU)
    - Very small models (40MB - 1.8GB)
    - Low latency (50-100ms)
    - Streaming support
    - Offline (no internet needed)
    - Good accuracy (88-92%)
    - Perfect for fallback when RAM is tight
    """

    def __init__(self, model_config):
        super().__init__(model_config)
        self.recognizer = None
        self.sample_rate = 16000  # Vosk expects 16kHz
        self.models_dir = Path.home() / ".jarvis" / "models" / "stt" / "vosk"
        self.model_path_resolved = None

    async def initialize(self):
        """Initialize Vosk model"""
        if self.initialized:
            return

        logger.info(f"🔧 Initializing Vosk: {self.model_config.name}")

        try:
            # Import vosk (lazy import)
            from vosk import KaldiRecognizer, Model

            # Ensure model is downloaded
            await self._ensure_model_downloaded()

            # Load model
            logger.info(f"   Loading model from: {self.model_path_resolved}")
            self.model = await asyncio.to_thread(Model, str(self.model_path_resolved))

            # Create recognizer
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.recognizer.SetWords(True)  # Enable word-level timestamps

            self.initialized = True
            logger.info(f"✅ Vosk initialized: {self.model_config.name}")

        except Exception as e:
            logger.error(f"Failed to initialize Vosk: {e}")
            raise

    async def _ensure_model_downloaded(self):
        """Download Vosk model if not present"""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Determine model directory name
        model_name = self.model_config.model_path  # e.g., "vosk-model-small-en-us-0.15"
        model_dir = self.models_dir / model_name

        if model_dir.exists():
            logger.info(f"   Model already downloaded: {model_dir}")
            self.model_path_resolved = model_dir
            return

        # Model not found - download it
        download_url = self.model_config.download_url
        if not download_url:
            raise ValueError(f"Model {model_name} not found and no download URL configured")

        logger.info(f"   Downloading Vosk model from: {download_url}")
        logger.info(f"   This may take a few minutes...")

        zip_path = self.models_dir / f"{model_name}.zip"

        try:
            # Download model
            await asyncio.to_thread(
                urlretrieve,
                download_url,
                zip_path,
            )

            logger.info(f"   Download complete, extracting...")

            # Extract model
            await asyncio.to_thread(self._extract_zip, zip_path, self.models_dir)

            # Remove zip file
            zip_path.unlink()

            logger.info(f"   Model ready: {model_dir}")
            self.model_path_resolved = model_dir

        except Exception as e:
            logger.error(f"Failed to download/extract model: {e}")
            # Cleanup partial downloads
            if zip_path.exists():
                zip_path.unlink()
            raise

    def _extract_zip(self, zip_path: Path, extract_to: Path):
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    async def transcribe(self, audio_data: bytes) -> STTResult:
        """
        Transcribe audio using Vosk.

        Args:
            audio_data: Raw audio bytes (will be converted to 16kHz mono PCM)

        Returns:
            STTResult with transcription and confidence
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Convert audio to 16kHz mono PCM int16
            audio_array = await self._bytes_to_pcm16(audio_data)
            audio_duration_ms = len(audio_array) / (self.sample_rate * 2) * 1000

            # Reset recognizer state
            await asyncio.to_thread(self.recognizer.Reset)

            # Feed audio data to recognizer
            await asyncio.to_thread(self.recognizer.AcceptWaveform, audio_array.tobytes())

            # Get final result
            result_json = await asyncio.to_thread(self.recognizer.FinalResult)
            result = json.loads(result_json)

            # Extract text and confidence
            transcription_text = result.get("text", "")

            # Vosk doesn't provide word-level confidence in simple API
            # We'll estimate based on result structure
            confidence = self._estimate_confidence(result)

            latency_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"🎤 Vosk transcribed: '{transcription_text[:50]}...' "
                f"(confidence={confidence:.2f}, latency={latency_ms:.0f}ms)"
            )

            return STTResult(
                text=transcription_text,
                confidence=confidence,
                engine=STTEngine.VOSK,
                model_name=self.model_config.name,
                latency_ms=latency_ms,
                audio_duration_ms=audio_duration_ms,
                metadata={
                    "sample_rate": self.sample_rate,
                    "model_path": str(self.model_path_resolved),
                    "words": result.get("result", []),  # Word-level timestamps
                },
            )

        except Exception as e:
            logger.error(f"Vosk transcription failed: {e}")
            raise

    async def _bytes_to_pcm16(self, audio_data: bytes) -> np.ndarray:
        """
        Convert audio bytes to 16kHz mono PCM int16 format.

        Vosk requires this specific format.
        """
        try:
            # Try with librosa (best quality)
            import io

            import librosa

            # Load audio (librosa auto-detects format)
            audio_array, sr = await asyncio.to_thread(
                librosa.load,
                io.BytesIO(audio_data),
                sr=self.sample_rate,
                mono=True,
            )

            # Convert float32 [-1, 1] to int16
            audio_int16 = (audio_array * 32767).astype(np.int16)

            return audio_int16

        except ImportError:
            logger.warning("librosa not available, using scipy")
            # Fallback to scipy
            import io

            import scipy.io.wavfile as wavfile

            sr, audio_array = await asyncio.to_thread(wavfile.read, io.BytesIO(audio_data))

            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1).astype(audio_array.dtype)

            # Convert to int16 if needed
            if audio_array.dtype != np.int16:
                if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                    # Float to int16
                    audio_array = (audio_array * 32767).astype(np.int16)
                elif audio_array.dtype == np.int32:
                    # Int32 to int16
                    audio_array = (audio_array / 65536).astype(np.int16)

            # Resample if needed
            if sr != self.sample_rate:
                import scipy.signal as signal

                num_samples = int(len(audio_array) * self.sample_rate / sr)
                audio_array = await asyncio.to_thread(signal.resample, audio_array, num_samples)
                audio_array = audio_array.astype(np.int16)

            return audio_array

    def _estimate_confidence(self, result: dict) -> float:
        """
        Estimate confidence from Vosk result.

        Vosk doesn't provide overall confidence, but word-level results
        have confidence scores.
        """
        try:
            # Get word-level results
            words = result.get("result", [])

            if not words:
                # No words detected - low confidence
                return 0.5 if result.get("text", "") else 0.3

            # Average word-level confidence
            confidences = [word.get("conf", 0.5) for word in words]
            avg_confidence = sum(confidences) / len(confidences)

            return max(0.0, min(1.0, avg_confidence))

        except Exception as e:
            logger.warning(f"Failed to estimate confidence: {e}")
            return 0.5  # Default moderate confidence

    async def transcribe_stream(self, audio_stream):
        """
        Streaming transcription (for real-time use).

        Args:
            audio_stream: AsyncIterator yielding audio chunks (bytes)

        Yields:
            Partial transcription results
        """
        if not self.initialized:
            await self.initialize()

        await asyncio.to_thread(self.recognizer.Reset)

        async for chunk in audio_stream:
            # Convert chunk to PCM16
            audio_pcm16 = await self._bytes_to_pcm16(chunk)

            # Feed to recognizer
            is_final = await asyncio.to_thread(
                self.recognizer.AcceptWaveform, audio_pcm16.tobytes()
            )

            if is_final:
                # Final result for this chunk
                result_json = await asyncio.to_thread(self.recognizer.Result)
                result = json.loads(result_json)
                yield result.get("text", "")
            else:
                # Partial result
                partial_json = await asyncio.to_thread(self.recognizer.PartialResult)
                partial = json.loads(partial_json)
                yield partial.get("partial", "")

        # Get final result
        final_json = await asyncio.to_thread(self.recognizer.FinalResult)
        final = json.loads(final_json)
        yield final.get("text", "")

    async def cleanup(self):
        """Cleanup Vosk resources"""
        if self.recognizer is not None:
            del self.recognizer
            self.recognizer = None

        if self.model is not None:
            del self.model
            self.model = None

        await super().cleanup()
        logger.info(f"🧹 Vosk engine cleaned up: {self.model_config.name}")
