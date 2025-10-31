#!/usr/bin/env python3
"""
Audio Format Converter
Ensures audio data is in the correct format for STT processing
"""

import base64
import numpy as np
import logging
import json
import struct
import wave
import io

logger = logging.getLogger(__name__)

class AudioFormatConverter:
    """Convert various audio formats to standard format for STT"""

    @staticmethod
    def convert_to_bytes(audio_data) -> bytes:
        """
        Convert any audio format to proper bytes for processing.

        Args:
            audio_data: Audio in any format (string, base64, bytes, list, etc.)

        Returns:
            bytes: PCM audio data as bytes
        """

        if audio_data is None:
            logger.warning("Audio data is None, returning empty bytes")
            return b''

        # Already bytes - validate it's proper audio
        if isinstance(audio_data, bytes):
            logger.debug(f"Audio is already bytes: {len(audio_data)} bytes")
            return audio_data

        # Base64 encoded string
        if isinstance(audio_data, str):
            # Check if it's JSON
            if audio_data.startswith('{') or audio_data.startswith('['):
                try:
                    data = json.loads(audio_data)
                    if isinstance(data, list):
                        # Array of samples
                        return AudioFormatConverter._array_to_bytes(data)
                    elif isinstance(data, dict) and 'data' in data:
                        # Nested data object
                        return AudioFormatConverter.convert_to_bytes(data['data'])
                except:
                    pass

            # Try base64 decoding
            try:
                # Standard base64
                decoded = base64.b64decode(audio_data)
                logger.debug(f"Decoded base64: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # Try base64 with padding fix
            try:
                # Add padding if needed
                padding = 4 - len(audio_data) % 4
                if padding != 4:
                    audio_data += '=' * padding
                decoded = base64.b64decode(audio_data)
                logger.debug(f"Decoded base64 with padding: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # URL-safe base64
            try:
                decoded = base64.urlsafe_b64decode(audio_data)
                logger.debug(f"Decoded URL-safe base64: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # Hex string
            try:
                decoded = bytes.fromhex(audio_data)
                logger.debug(f"Decoded hex: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # Comma-separated values
            if ',' in audio_data:
                try:
                    values = [int(x.strip()) for x in audio_data.split(',')]
                    return AudioFormatConverter._array_to_bytes(values)
                except:
                    pass

            logger.warning(f"Could not decode string of length {len(audio_data)}")
            return b''

        # List or array of samples
        if isinstance(audio_data, (list, tuple)):
            return AudioFormatConverter._array_to_bytes(audio_data)

        # NumPy array
        if hasattr(audio_data, 'dtype'):
            return AudioFormatConverter._numpy_to_bytes(audio_data)

        # Dictionary with audio data
        if isinstance(audio_data, dict):
            # Check common keys
            for key in ['data', 'audio', 'audio_data', 'samples', 'buffer']:
                if key in audio_data:
                    return AudioFormatConverter.convert_to_bytes(audio_data[key])

        # Try to convert to bytes directly
        try:
            return bytes(audio_data)
        except:
            logger.error(f"Cannot convert audio data of type {type(audio_data)}")
            return b''

    @staticmethod
    def _array_to_bytes(array) -> bytes:
        """Convert array of samples to bytes"""
        try:
            # Determine data type
            if all(isinstance(x, float) for x in array[:100]):
                # Float samples (-1.0 to 1.0)
                samples = np.array(array, dtype=np.float32)
                # Convert to int16
                samples = (samples * 32767).astype(np.int16)
            else:
                # Integer samples
                samples = np.array(array, dtype=np.int16)

            logger.debug(f"Converted array of {len(samples)} samples to bytes")
            return samples.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert array to bytes: {e}")
            return b''

    @staticmethod
    def _numpy_to_bytes(arr) -> bytes:
        """Convert numpy array to bytes"""
        try:
            if arr.dtype in [np.float32, np.float64]:
                # Convert float to int16
                arr = (arr * 32767).astype(np.int16)
            elif arr.dtype != np.int16:
                # Convert to int16
                arr = arr.astype(np.int16)

            logger.debug(f"Converted numpy array {arr.shape} to bytes")
            return arr.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert numpy array to bytes: {e}")
            return b''

    @staticmethod
    def ensure_pcm_format(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """
        Ensure audio bytes are in PCM format suitable for STT.

        Args:
            audio_bytes: Raw audio bytes
            sample_rate: Target sample rate (default 16000 Hz)

        Returns:
            bytes: PCM audio data at 16kHz, 16-bit, mono
        """

        if not audio_bytes:
            return b''

        # Check if it's a WAV file
        if audio_bytes.startswith(b'RIFF'):
            try:
                # Parse WAV file
                with io.BytesIO(audio_bytes) as wav_io:
                    with wave.open(wav_io, 'rb') as wav:
                        frames = wav.readframes(wav.getnframes())
                        return frames
            except:
                pass

        # Check if it's already proper PCM
        # PCM should have even number of bytes (16-bit samples)
        if len(audio_bytes) % 2 == 0:
            return audio_bytes

        # Pad with zero if odd number of bytes
        return audio_bytes + b'\x00'

    @staticmethod
    def create_wav_header(audio_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
        """
        Add WAV header to raw PCM data.

        Args:
            audio_bytes: Raw PCM audio data
            sample_rate: Sample rate in Hz
            channels: Number of channels

        Returns:
            bytes: Complete WAV file with header
        """

        # WAV header parameters
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(audio_bytes)
        file_size = data_size + 44 - 8  # Total file size minus RIFF header

        # Create WAV header
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',           # ChunkID
            file_size,         # ChunkSize
            b'WAVE',           # Format
            b'fmt ',           # Subchunk1ID
            16,                # Subchunk1Size (16 for PCM)
            1,                 # AudioFormat (1 for PCM)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            byte_rate,         # ByteRate
            block_align,       # BlockAlign
            bits_per_sample,   # BitsPerSample
            b'data',           # Subchunk2ID
            data_size          # Subchunk2Size
        )

        return header + audio_bytes


# Global converter instance
audio_converter = AudioFormatConverter()

def prepare_audio_for_stt(audio_data) -> bytes:
    """
    Prepare any audio format for STT processing.

    Args:
        audio_data: Audio in any format

    Returns:
        bytes: PCM audio ready for STT
    """

    # Convert to bytes
    audio_bytes = audio_converter.convert_to_bytes(audio_data)

    # Ensure PCM format
    pcm_bytes = audio_converter.ensure_pcm_format(audio_bytes)

    logger.info(f"✅ Prepared audio: {len(pcm_bytes)} bytes of PCM data")
    return pcm_bytes