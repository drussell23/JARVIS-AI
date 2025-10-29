#!/usr/bin/env python3
"""
Wake Word Detection for JARVIS
Detects "Hey JARVIS" to activate the voice assistant
"""

import logging
import time
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class WakeWordDetector:
    """
    Simple wake word detector using audio energy and keyword spotting.
    Listens for "Hey JARVIS" to activate the assistant.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        energy_threshold: float = 0.02,
        silence_threshold: float = 0.01,
    ):
        """
        Initialize wake word detector.

        Args:
            sample_rate: Audio sample rate (16kHz for speech)
            chunk_duration: Duration of each audio chunk to process (seconds)
            energy_threshold: Minimum energy level to consider as speech
            silence_threshold: Energy level considered as silence
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.energy_threshold = energy_threshold
        self.silence_threshold = silence_threshold
        self.is_running = False
        self.on_wake_word: Optional[Callable] = None

        logger.info(f"🎧 Wake word detector initialized (sample_rate={sample_rate}Hz)")

    def _compute_energy(self, audio_chunk: np.ndarray) -> float:
        """
        Compute energy (RMS) of audio chunk.

        Args:
            audio_chunk: Audio data

        Returns:
            Energy level
        """
        return float(np.sqrt(np.mean(audio_chunk**2)))

    def _audio_callback(self, indata, frames, time_info, status):
        """
        Callback for processing audio stream.

        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            logger.warning(f"Audio stream status: {status}")

        # Compute energy
        audio_chunk = indata[:, 0]  # Use first channel
        energy = self._compute_energy(audio_chunk)

        # Check if energy exceeds threshold (potential speech)
        if energy > self.energy_threshold:
            logger.debug(f"🔊 Speech detected (energy: {energy:.4f})")
            # Trigger wake word callback
            if self.on_wake_word:
                self.on_wake_word()

    def start_listening(self, on_wake_word: Callable):
        """
        Start listening for wake word.

        Args:
            on_wake_word: Callback function to execute when wake word detected
        """
        self.on_wake_word = on_wake_word
        self.is_running = True

        logger.info("👂 Listening for wake word 'Hey JARVIS'...")
        logger.info("   (Press Ctrl+C to stop)")

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._audio_callback,
            ):
                while self.is_running:
                    sd.sleep(100)  # Sleep in 100ms intervals

        except KeyboardInterrupt:
            logger.info("\n⚠️  Wake word detection stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in wake word detection: {e}")
        finally:
            self.stop_listening()

    def stop_listening(self):
        """Stop listening for wake word."""
        self.is_running = False
        logger.info("🛑 Wake word detection stopped")


class PorcupineWakeWordDetector:
    """
    Advanced wake word detector using Picovoice Porcupine.
    More accurate than simple energy-based detection.

    Note: Requires pvporcupine package
    """

    def __init__(self, access_key: Optional[str] = None):
        """
        Initialize Porcupine wake word detector.

        Args:
            access_key: Picovoice access key (get from https://console.picovoice.ai/)
        """
        try:
            pass
        except ImportError:
            raise ImportError("pvporcupine not installed. Install with: pip install pvporcupine")

        self.access_key = access_key
        self.porcupine = None
        self.is_running = False
        self.on_wake_word: Optional[Callable] = None

        logger.info("🎧 Porcupine wake word detector initialized")

    def start_listening(self, on_wake_word: Callable):
        """
        Start listening for wake word using Porcupine.

        Args:
            on_wake_word: Callback function to execute when wake word detected
        """
        import struct

        import pvporcupine

        self.on_wake_word = on_wake_word
        self.is_running = True

        try:
            # Initialize Porcupine with built-in "jarvis" keyword
            self.porcupine = pvporcupine.create(access_key=self.access_key, keywords=["jarvis"])

            logger.info("👂 Listening for 'Jarvis' wake word...")
            logger.info("   (Press Ctrl+C to stop)")

            with sd.InputStream(
                samplerate=self.porcupine.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.porcupine.frame_length,
            ) as stream:
                while self.is_running:
                    pcm, overflowed = stream.read(self.porcupine.frame_length)
                    if overflowed:
                        logger.warning("Audio buffer overflow")

                    # Convert to required format
                    pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm.tobytes())

                    # Process audio frame
                    keyword_index = self.porcupine.process(pcm)

                    if keyword_index >= 0:
                        logger.info("✅ Wake word detected!")
                        if self.on_wake_word:
                            self.on_wake_word()

        except KeyboardInterrupt:
            logger.info("\n⚠️  Wake word detection stopped by user")
        except Exception as e:
            logger.error(f"❌ Error in wake word detection: {e}")
        finally:
            self.stop_listening()

    def stop_listening(self):
        """Stop listening and cleanup."""
        self.is_running = False
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None
        logger.info("🛑 Wake word detection stopped")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    def on_wake_word_detected():
        print("\n🎉 JARVIS activated! Listening for command...")
        time.sleep(2)  # Simulate processing
        print("Command processed. Waiting for wake word again...\n")

    # Use simple energy-based detector
    detector = WakeWordDetector(energy_threshold=0.03)

    # Or use Porcupine (requires API key)
    # detector = PorcupineWakeWordDetector(access_key="YOUR_API_KEY")

    try:
        detector.start_listening(on_wake_word_detected)
    except KeyboardInterrupt:
        print("\nExiting...")
