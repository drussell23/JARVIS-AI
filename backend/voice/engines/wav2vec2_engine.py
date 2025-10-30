"""
Wav2Vec2 STT Engine

This module implements a speech-to-text engine using Facebook's Wav2Vec2 model
through Hugging Face transformers. Wav2Vec2 is a self-supervised learning approach
for speech recognition that learns powerful representations from raw audio.

The engine supports GPU acceleration (CUDA/MPS) and provides high-quality
transcription for various audio inputs.
"""

import logging
import time
from typing import Optional

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from ..stt_config import ModelConfig
from .base_engine import BaseSTTEngine, STTResult

logger = logging.getLogger(__name__)


class Wav2Vec2Engine(BaseSTTEngine):
    """Wav2Vec2 speech recognition engine using Hugging Face transformers.
    
    This engine leverages Facebook's Wav2Vec2 model for speech-to-text conversion.
    It supports automatic device selection (MPS, CUDA, CPU) and provides efficient
    transcription with normalized audio processing.
    
    Attributes:
        processor: Wav2Vec2 processor for audio preprocessing and text decoding.
        model: Wav2Vec2ForCTC model for speech recognition.
        device: PyTorch device (CPU, CUDA, or MPS) used for inference.
    """

    def __init__(self, model_config: ModelConfig):
        """Initialize Wav2Vec2Engine with model configuration.
        
        Args:
            model_config: Configuration object containing model path and settings.
        """
        super().__init__(model_config)
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2ForCTC] = None
        self.device = None

    async def initialize(self) -> None:
        """Initialize Wav2Vec2 model and processor.
        
        Loads the Wav2Vec2 processor and model from the specified path or uses
        the default Facebook model. Automatically selects the best available
        device (MPS > CUDA > CPU) and moves the model to that device.
        
        Raises:
            Exception: If model loading or device setup fails.
            
        Example:
            >>> engine = Wav2Vec2Engine(model_config)
            >>> await engine.initialize()
        """
        if self.initialized:
            return

        start = time.time()
        logger.info(f"Initializing Wav2Vec2: {self.model_config.model_path}")

        try:
            # Load processor and model
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.model_config.model_path or "facebook/wav2vec2-base"
            )
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_config.model_path or "facebook/wav2vec2-base"
            )

            # Set device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")

            self.model.to(self.device)
            self.model.eval()

            self.initialized = True
            elapsed = time.time() - start
            logger.info(f"Wav2Vec2 {self.model_config.name} ready ({elapsed:.2f}s)")
            logger.info(f"   Using device: {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize Wav2Vec2: {e}")
            raise

    async def transcribe(self, audio_data: bytes) -> STTResult:
        """Transcribe audio data using Wav2Vec2 model.
        
        Processes raw audio bytes (16kHz, 16-bit PCM) through the Wav2Vec2 pipeline:
        1. Converts bytes to normalized float32 array
        2. Preprocesses audio with the processor
        3. Runs inference on the selected device
        4. Decodes predictions to text
        
        Args:
            audio_data: Raw audio data as bytes (16kHz, 16-bit PCM format).
            
        Returns:
            STTResult containing transcription text, confidence score, timing
            information, and metadata about the inference process.
            
        Raises:
            Exception: If transcription fails due to model errors or invalid audio.
            
        Example:
            >>> with open("audio.wav", "rb") as f:
            ...     audio_bytes = f.read()
            >>> result = await engine.transcribe(audio_bytes)
            >>> print(result.text)
            "Hello world"
        """
        if not self.initialized:
            await self.initialize()

        start = time.time()

        try:
            # Convert bytes to numpy array (assuming 16kHz, 16-bit PCM)
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)

            # Normalize
            audio_array = audio_array / 32768.0

            # Process audio
            input_values = self.processor(
                audio_array, sampling_rate=16000, return_tensors="pt"
            ).input_values

            # Move to device
            input_values = input_values.to(self.device)

            # Get predictions
            with torch.no_grad():
                logits = self.model(input_values).logits

            # Decode
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = self.processor.batch_decode(predicted_ids)[0]

            # Calculate latency
            latency_ms = (time.time() - start) * 1000
            audio_duration_ms = len(audio_array) / 16  # 16 samples per ms at 16kHz

            return STTResult(
                text=transcription,
                confidence=0.95,  # Wav2Vec2 doesn't provide confidence scores directly
                engine=self.model_config.engine,
                model_name=self.model_config.name,
                latency_ms=latency_ms,
                audio_duration_ms=audio_duration_ms,
                metadata={"device": str(self.device), "model_path": self.model_config.model_path},
            )

        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            raise

    async def cleanup(self) -> None:
        """Clean up Wav2Vec2 model resources and free memory.
        
        Properly disposes of the model and processor objects, and clears
        GPU memory cache if CUDA is available. This helps prevent memory
        leaks when switching between models or shutting down.
        
        Example:
            >>> await engine.cleanup()
        """
        if self.model:
            del self.model
        if self.processor:
            del self.processor

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        await super().cleanup()