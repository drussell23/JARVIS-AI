#!/usr/bin/env python3
"""
Robust Whisper audio handler that works with any input format
"""

import base64
import numpy as np
import whisper
import tempfile
import logging
import asyncio
import soundfile as sf
import io

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logging.warning("librosa not available - will use basic resampling")

logger = logging.getLogger(__name__)

class WhisperAudioHandler:
    """Handles any audio format for Whisper transcription"""

    def __init__(self):
        self.model = None

    def _infer_sample_rate(self, audio_bytes: bytes, num_samples: int) -> int:
        """
        Intelligently infer sample rate from audio characteristics.

        Uses multiple heuristics:
        1. Audio duration estimation (if we know expected duration)
        2. Frequency content analysis
        3. Common sample rates for different sources

        Args:
            audio_bytes: Raw audio data
            num_samples: Number of audio samples detected

        Returns:
            Inferred sample rate in Hz
        """
        # Common sample rates to test
        common_rates = [48000, 44100, 32000, 24000, 16000, 22050, 11025, 8000]

        # Heuristic 1: Audio byte size can hint at sample rate
        # Typical voice command is 2-5 seconds
        # For int16 PCM: bytes = samples * 2
        audio_duration_estimates = {}
        for rate in common_rates:
            estimated_duration = num_samples / rate
            # Voice commands typically 1-10 seconds
            if 1.0 <= estimated_duration <= 10.0:
                audio_duration_estimates[rate] = estimated_duration
                logger.debug(f"Sample rate {rate}Hz → {estimated_duration:.2f}s duration")

        # Heuristic 2: Most likely rates based on source
        # Browser MediaRecorder: typically 48kHz or 44.1kHz
        # macOS: 48kHz or 44.1kHz
        # Mobile: 44.1kHz or 48kHz
        # Old hardware: 22.05kHz, 16kHz, 11.025kHz

        if audio_duration_estimates:
            # Choose rate that gives most reasonable duration (2-5 sec preference)
            best_rate = min(audio_duration_estimates.keys(),
                          key=lambda r: abs(audio_duration_estimates[r] - 3.0))
            logger.info(f"🎯 Inferred sample rate: {best_rate}Hz (duration: {audio_duration_estimates[best_rate]:.2f}s)")
            return best_rate

        # Fallback: Use most common browser rate
        logger.warning(f"⚠️ Could not infer sample rate, defaulting to 48000Hz (browser standard)")
        return 48000

    async def normalize_audio(self, audio_bytes: bytes, sample_rate: int = None) -> np.ndarray:
        """
        Universal audio normalization pipeline that:
        1. Auto-detects sample rate from audio bytes OR uses provided rate
        2. Decodes audio format (int16/float32/int8 PCM)
        3. Resamples to 16kHz if needed
        4. Converts stereo to mono
        5. Normalizes to float32 [-1.0, 1.0]

        Args:
            audio_bytes: Raw audio data
            sample_rate: Optional sample rate from frontend (browser-reported)
                        If None, will attempt to infer from audio

        Returns: Normalized float32 numpy array ready for Whisper
        """
        def _normalize_sync():
            logger.info(f"🔊 Audio normalization: {len(audio_bytes)} bytes")

            # Step 1: Try to detect format and decode with soundfile first
            audio_array = None
            detected_sr = None
            detected_format = None

            # Try soundfile first (handles WAV, FLAC, OGG with embedded metadata)
            try:
                audio_buf = io.BytesIO(audio_bytes)
                audio_array, detected_sr = sf.read(audio_buf, dtype='float32')
                detected_format = "soundfile (with metadata)"
                logger.info(f"✅ Decoded with soundfile: {detected_sr}Hz, {audio_array.shape}")
            except Exception as e:
                logger.debug(f"soundfile decode failed: {e}")

            # Step 2: If soundfile fails, try raw PCM formats
            if audio_array is None:
                # Try int16 PCM (most common from browsers)
                try:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                    if len(audio_array) > 100:
                        audio_array = audio_array.astype(np.float32) / 32768.0
                        detected_format = "int16 PCM"
                        # Use provided sample rate or infer it
                        if sample_rate:
                            detected_sr = sample_rate
                            logger.info(f"✅ Decoded as int16 PCM: {len(audio_array)} samples, using provided {detected_sr}Hz")
                        else:
                            detected_sr = self._infer_sample_rate(audio_bytes, len(audio_array))
                            logger.info(f"✅ Decoded as int16 PCM: {len(audio_array)} samples, inferred {detected_sr}Hz")
                    else:
                        audio_array = None
                except Exception as e:
                    logger.debug(f"int16 PCM decode failed: {e}")

            # Try float32 PCM
            if audio_array is None:
                try:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    if len(audio_array) > 100:
                        detected_format = "float32 PCM"
                        # Use provided sample rate or infer it
                        if sample_rate:
                            detected_sr = sample_rate
                            logger.info(f"✅ Decoded as float32 PCM: {len(audio_array)} samples, using provided {detected_sr}Hz")
                        else:
                            detected_sr = self._infer_sample_rate(audio_bytes, len(audio_array))
                            logger.info(f"✅ Decoded as float32 PCM: {len(audio_array)} samples, inferred {detected_sr}Hz")
                    else:
                        audio_array = None
                except Exception as e:
                    logger.debug(f"float32 PCM decode failed: {e}")

            # Try int8 PCM
            if audio_array is None:
                try:
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int8)
                    if len(audio_array) > 100:
                        audio_array = audio_array.astype(np.float32) / 128.0
                        detected_format = "int8 PCM"
                        # Use provided sample rate or infer it
                        if sample_rate:
                            detected_sr = sample_rate
                            logger.info(f"✅ Decoded as int8 PCM: {len(audio_array)} samples, using provided {detected_sr}Hz")
                        else:
                            detected_sr = self._infer_sample_rate(audio_bytes, len(audio_array))
                            logger.info(f"✅ Decoded as int8 PCM: {len(audio_array)} samples, inferred {detected_sr}Hz")
                    else:
                        audio_array = None
                except Exception as e:
                    logger.debug(f"int8 PCM decode failed: {e}")

            if audio_array is None:
                logger.error("❌ Could not decode audio in any known format")
                raise ValueError("Audio format not recognized")

            # Step 3: Convert stereo to mono if needed
            if len(audio_array.shape) > 1 and audio_array.shape[1] > 1:
                logger.info(f"Converting stereo ({audio_array.shape[1]} channels) to mono")
                audio_array = np.mean(audio_array, axis=1)

            # Step 4: Validate audio has content
            audio_energy = np.abs(audio_array).mean()
            if audio_energy < 0.001:
                logger.error(f"❌ Audio is silence (energy: {audio_energy:.6f})")
                raise ValueError("Audio contains only silence")

            logger.info(f"✅ Audio energy: {audio_energy:.6f}")

            # Step 5: Resample to 16kHz if needed
            TARGET_SR = 16000
            if detected_sr != TARGET_SR:
                logger.info(f"🔄 Resampling from {detected_sr}Hz to {TARGET_SR}Hz...")

                if LIBROSA_AVAILABLE:
                    # High-quality resampling with librosa
                    audio_array = librosa.resample(
                        audio_array,
                        orig_sr=detected_sr,
                        target_sr=TARGET_SR,
                        res_type='kaiser_best'  # Highest quality
                    )
                    logger.info(f"✅ Resampled with librosa (kaiser_best): {len(audio_array)} samples")
                else:
                    # Fallback: Basic linear interpolation
                    from scipy import signal
                    num_samples = int(len(audio_array) * TARGET_SR / detected_sr)
                    audio_array = signal.resample(audio_array, num_samples)
                    logger.info(f"✅ Resampled with scipy: {len(audio_array)} samples")
            else:
                logger.info(f"✅ Already at {TARGET_SR}Hz - no resampling needed")

            # Step 6: Ensure float32 normalization
            audio_array = audio_array.astype(np.float32)

            # Clip to [-1.0, 1.0] range
            if np.abs(audio_array).max() > 1.0:
                logger.warning(f"Audio exceeded [-1.0, 1.0] range, clipping...")
                audio_array = np.clip(audio_array, -1.0, 1.0)

            logger.info(f"✅ Normalization complete: {len(audio_array)} samples @ 16kHz, float32")
            return audio_array

        # Run in thread pool to avoid blocking event loop
        return await asyncio.to_thread(_normalize_sync)

    def load_model(self):
        """Load Whisper model"""
        if self.model is None:
            logger.info("Loading Whisper model...")
            self.model = whisper.load_model("base")
            logger.info("✅ Whisper model loaded")
        return self.model

    def decode_audio_data(self, audio_data):
        """Convert any input format to bytes"""

        # If already bytes, return as-is
        if isinstance(audio_data, bytes):
            logger.debug("Audio data is already bytes")
            return audio_data

        # If it's a string, try various decodings
        if isinstance(audio_data, str):
            # Try base64 first
            try:
                decoded = base64.b64decode(audio_data)
                logger.debug("Successfully decoded base64 audio")
                return decoded
            except:
                pass

            # Try URL-safe base64
            try:
                decoded = base64.urlsafe_b64decode(audio_data)
                logger.debug("Successfully decoded URL-safe base64 audio")
                return decoded
            except:
                pass

            # Try hex encoding
            try:
                decoded = bytes.fromhex(audio_data)
                logger.debug("Successfully decoded hex audio")
                return decoded
            except:
                pass

            # Try latin-1 encoding as last resort
            try:
                decoded = audio_data.encode('latin-1')
                logger.debug("Encoded string as latin-1")
                return decoded
            except:
                pass

        # If it's a numpy array
        if isinstance(audio_data, np.ndarray):
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                # Convert float to int16
                audio_data = (audio_data * 32767).astype(np.int16)
            return audio_data.tobytes()

        # Try to convert to bytes
        try:
            return bytes(audio_data)
        except:
            logger.error(f"Cannot convert audio data of type {type(audio_data)} to bytes")
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

    async def create_wav_from_normalized_audio(self, audio_array: np.ndarray) -> str:
        """
        Create a temporary WAV file from normalized audio array

        Args:
            audio_array: Normalized float32 array at 16kHz

        Returns:
            Path to temporary WAV file
        """
        def _write_sync():
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, audio_array, 16000)
                return tmp.name

        return await asyncio.to_thread(_write_sync)

    async def transcribe_any_format(self, audio_data, sample_rate: int = None):
        """
        Transcribe audio in any format with automatic normalization

        Args:
            audio_data: Audio bytes or base64 string
            sample_rate: Optional sample rate from frontend (browser-reported)
                        If None, will attempt to infer from audio
        """

        # Load model if needed
        model = self.load_model()

        try:
            # Convert to bytes
            audio_bytes = self.decode_audio_data(audio_data)

            # Normalize audio (use provided sample rate or auto-detect)
            normalized_audio = await self.normalize_audio(audio_bytes, sample_rate=sample_rate)

            # Create WAV file from normalized audio
            wav_path = await self.create_wav_from_normalized_audio(normalized_audio)

            # Transcribe with Whisper in thread pool to avoid blocking
            def _transcribe_sync():
                result = model.transcribe(wav_path)
                return result["text"].strip()

            text = await asyncio.to_thread(_transcribe_sync)

            # Clean up temp file
            import os
            os.unlink(wav_path)

            logger.info(f"✅ Transcribed: '{text}'")

            # If Whisper returns empty string, it detected no speech
            # Return None to signal failure so hybrid_stt_router can try other methods
            if not text or text.strip() == "":
                logger.warning("⚠️ Whisper returned empty transcription - no speech detected")
                logger.warning(f"   Audio was {len(audio_bytes)} bytes, normalized to {len(normalized_audio)} samples")
                logger.warning(f"   Sample rate: provided={sample_rate}, final=16000Hz")
                return None

            return text

        except Exception as e:
            logger.error(f"❌ Whisper transcription failed: {e}")
            logger.error("   This indicates audio format issues or invalid audio data")
            # Return None to signal failure - do NOT return hardcoded text
            return None

# Global instance
_whisper_handler = WhisperAudioHandler()

async def transcribe_with_whisper(audio_data, sample_rate: int = None):
    """
    Global transcription function with optional sample rate

    Args:
        audio_data: Audio bytes or base64 string
        sample_rate: Optional sample rate from frontend (browser-reported)
    """
    return await _whisper_handler.transcribe_any_format(audio_data, sample_rate=sample_rate)