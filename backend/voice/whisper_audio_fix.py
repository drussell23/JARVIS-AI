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

logger = logging.getLogger(__name__)

class WhisperAudioHandler:
    """Handles any audio format for Whisper transcription"""

    def __init__(self):
        self.model = None

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

    def create_wav_from_bytes(self, audio_bytes, sample_rate=16000):
        """Create a WAV file from raw audio bytes with robust format detection"""

        logger.info(f"🔍 Audio preprocessing: {len(audio_bytes)} bytes")

        # Try to interpret as different formats
        audio_array = None
        detected_format = None

        # Try as raw PCM int16 (most common format from browsers)
        try:
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            # Validate it's reasonable audio data
            if len(audio_array) > 100:  # At least 100 samples
                audio_array = audio_array.astype(np.float32) / 32768.0
                detected_format = "int16 PCM"
                logger.info(f"✅ Detected: {detected_format}, {len(audio_array)} samples")
            else:
                audio_array = None
        except Exception as e:
            logger.debug(f"Not int16 PCM: {e}")
            pass

        # Try as raw PCM float32
        if audio_array is None:
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                if len(audio_array) > 100:
                    detected_format = "float32 PCM"
                    logger.info(f"✅ Detected: {detected_format}, {len(audio_array)} samples")
                else:
                    audio_array = None
            except Exception as e:
                logger.debug(f"Not float32 PCM: {e}")
                pass

        # Try as raw PCM int8
        if audio_array is None:
            try:
                audio_array = np.frombuffer(audio_bytes, dtype=np.int8)
                if len(audio_array) > 100:
                    audio_array = audio_array.astype(np.float32) / 128.0
                    detected_format = "int8 PCM"
                    logger.info(f"✅ Detected: {detected_format}, {len(audio_array)} samples")
                else:
                    audio_array = None
            except Exception as e:
                logger.debug(f"Not int8 PCM: {e}")
                pass

        # If we still don't have audio, FAIL instead of creating silence
        if audio_array is None:
            logger.error("❌ Could not interpret audio bytes - invalid audio format")
            raise ValueError("Audio format not recognized - cannot decode audio bytes")

        # Ensure audio is the right length
        if len(audio_array) == 0:
            logger.error("❌ Empty audio array - no audio data to transcribe")
            raise ValueError("Empty audio data - cannot transcribe silence")

        # Validate audio has actual content (not just silence)
        audio_energy = np.abs(audio_array).mean()
        if audio_energy < 0.001:  # Threshold for silence detection
            logger.error(f"❌ Audio appears to be silence (energy: {audio_energy:.6f})")
            raise ValueError("Audio contains only silence - cannot transcribe")

        # Save to temp WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_array, sample_rate)
            return tmp.name

    async def transcribe_any_format(self, audio_data):
        """Transcribe audio in any format"""

        # Load model if needed
        model = self.load_model()

        try:
            # Convert to bytes
            audio_bytes = self.decode_audio_data(audio_data)

            # Create WAV file
            wav_path = self.create_wav_from_bytes(audio_bytes)

            # Transcribe with Whisper
            result = model.transcribe(wav_path)
            text = result["text"].strip()

            # Clean up temp file
            import os
            os.unlink(wav_path)

            logger.info(f"✅ Transcribed: '{text}'")
            return text

        except Exception as e:
            logger.error(f"❌ Whisper transcription failed: {e}")
            logger.error("   This indicates audio format issues or invalid audio data")
            # Return None to signal failure - do NOT return hardcoded text
            return None

# Global instance
_whisper_handler = WhisperAudioHandler()

async def transcribe_with_whisper(audio_data):
    """Global transcription function"""
    return await _whisper_handler.transcribe_any_format(audio_data)