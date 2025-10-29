#!/usr/bin/env python3
"""
JARVIS Voice Assistant - Main Conversation Loop
Complete STT → Intent Recognition → Action Execution → TTS pipeline
"""

import asyncio
import logging
from typing import Optional

import numpy as np
import sounddevice as sd
from intelligence.learning_database import JARVISLearningDatabase
from voice.engines.speechbrain_engine import SpeechBrainEngine
from voice.model_config import ModelConfig, STTEngine
from voice.wake_word_detector import WakeWordDetector

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class JARVISVoiceAssistant:
    """
    Complete JARVIS Voice Assistant with wake word detection,
    speech recognition, intent understanding, and voice response.
    """

    def __init__(self, speaker_name: str = "Derek J. Russell"):
        """
        Initialize JARVIS voice assistant.

        Args:
            speaker_name: Primary user's name for speaker verification
        """
        self.speaker_name = speaker_name
        self.sample_rate = 16000
        self.learning_db: Optional[JARVISLearningDatabase] = None
        self.stt_engine: Optional[SpeechBrainEngine] = None
        self.wake_word_detector: Optional[WakeWordDetector] = None
        self.is_listening_for_command = False
        self.speaker_id: Optional[int] = None

        logger.info("🤖 Initializing JARVIS Voice Assistant...")

    async def initialize(self):
        """Initialize all components asynchronously."""
        # Initialize learning database
        self.learning_db = JARVISLearningDatabase()
        await self.learning_db.initialize()

        # Get speaker ID
        self.speaker_id = await self.learning_db.get_or_create_speaker_profile(self.speaker_name)
        logger.info(f"👤 Speaker profile loaded: {self.speaker_name} (ID: {self.speaker_id})")

        # Initialize STT engine
        model_config = ModelConfig(
            name="speechbrain-wav2vec2",
            engine=STTEngine.SPEECHBRAIN,
            disk_size_mb=380,
            ram_required_gb=2.0,
            vram_required_gb=1.8,
            expected_accuracy=0.96,
        )
        self.stt_engine = SpeechBrainEngine(model_config)
        await self.stt_engine.initialize()

        # Initialize wake word detector
        self.wake_word_detector = WakeWordDetector(
            sample_rate=self.sample_rate,
            energy_threshold=0.03,  # Adjust based on environment
        )

        logger.info("✅ JARVIS Voice Assistant initialized\n")

    async def record_command(self, duration: float = 5.0) -> bytes:
        """
        Record audio command from user.

        Args:
            duration: Recording duration in seconds

        Returns:
            Audio data as WAV bytes
        """
        logger.info(f"🎙️  Recording for {duration} seconds...")

        # Record audio
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        logger.info("✅ Recording complete")

        # Convert to WAV bytes
        import io
        import wave

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            # Convert float32 to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            wav_file.writeframes(audio_int16.tobytes())

        return wav_buffer.getvalue()

    async def process_command(self, audio_bytes: bytes) -> str:
        """
        Process voice command through STT and intent recognition.

        Args:
            audio_bytes: Audio data

        Returns:
            Recognized text command
        """
        # Transcribe audio
        logger.info("🧠 Processing command...")
        result = await self.stt_engine.transcribe(audio_bytes)

        command_text = result.text
        confidence = result.confidence

        logger.info(f"📝 Recognized: '{command_text}' (confidence: {confidence:.2%})")

        # Record interaction in learning database
        if self.learning_db:
            await self.learning_db.record_conversation_interaction(
                user_query=command_text,
                jarvis_response="",  # Will be filled after action execution
                context_before={},
                context_after={},
                user_feedback=None,
            )

        return command_text

    async def execute_action(self, command: str) -> str:
        """
        Execute action based on recognized command.

        Args:
            command: Recognized command text

        Returns:
            Response text
        """
        # Simple command matching for now
        command_lower = command.lower()

        # Weather commands
        if "weather" in command_lower:
            return "The weather today is sunny with a high of 75 degrees."

        # Time commands
        elif "time" in command_lower:
            from datetime import datetime

            current_time = datetime.now().strftime("%I:%M %p")
            return f"The current time is {current_time}."

        # Greeting
        elif any(greeting in command_lower for greeting in ["hello", "hey", "hi", "jarvis"]):
            return f"Hello {self.speaker_name}! How can I assist you today?"

        # Calendar
        elif "calendar" in command_lower or "schedule" in command_lower:
            return "You have no events scheduled for today."

        # Lights
        elif "light" in command_lower:
            if "on" in command_lower:
                return "Turning on the lights."
            elif "off" in command_lower:
                return "Turning off the lights."
            else:
                return "I can turn lights on or off. Which would you like?"

        # Music
        elif "music" in command_lower or "play" in command_lower:
            return "Playing your favorite playlist."

        # Default response
        else:
            return f"I heard you say: {command}. I'm still learning how to handle this request."

    async def speak_response(self, response_text: str):
        """
        Convert text to speech and play audio.

        Args:
            response_text: Text to speak
        """
        logger.info(f"🔊 JARVIS: {response_text}")
        # TODO: Integrate TTS engine
        # For now, just print the response

    def on_wake_word_detected(self):
        """Callback when wake word is detected."""
        if self.is_listening_for_command:
            return  # Already processing a command

        logger.info("\n" + "=" * 60)
        logger.info("✨ JARVIS ACTIVATED ✨")
        logger.info("=" * 60)

        # Set flag to prevent multiple activations
        self.is_listening_for_command = True

        # Process command in async context
        asyncio.create_task(self.handle_voice_command())

    async def handle_voice_command(self):
        """Handle complete voice command interaction."""
        try:
            # Record command
            audio_bytes = await self.record_command(duration=5.0)

            # Process through STT
            command_text = await self.process_command(audio_bytes)

            # Execute action
            response_text = await self.execute_action(command_text)

            # Speak response
            await self.speak_response(response_text)

            logger.info("=" * 60)
            logger.info("👂 Listening for wake word...\n")

        except Exception as e:
            logger.error(f"❌ Error processing command: {e}")
        finally:
            self.is_listening_for_command = False

    async def run(self):
        """Run the main JARVIS assistant loop."""
        logger.info("\n" + "=" * 60)
        logger.info("🎤 JARVIS VOICE ASSISTANT")
        logger.info("=" * 60)
        logger.info(f"User: {self.speaker_name}")
        logger.info('Wake word: "Hey JARVIS" or speak loudly')
        logger.info("=" * 60)
        logger.info("\n👂 Listening for wake word...")
        logger.info("   (Press Ctrl+C to exit)\n")

        try:
            # Start wake word detection (blocking)
            self.wake_word_detector.start_listening(self.on_wake_word_detected)

        except KeyboardInterrupt:
            logger.info("\n⚠️  JARVIS shutting down...")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources."""
        if self.wake_word_detector:
            self.wake_word_detector.stop_listening()

        if self.stt_engine:
            await self.stt_engine.cleanup()

        if self.learning_db:
            await self.learning_db.close()

        logger.info("✅ JARVIS shutdown complete")


async def main():
    """Main entry point."""
    # Create JARVIS instance
    jarvis = JARVISVoiceAssistant(speaker_name="Derek J. Russell")

    # Initialize
    await jarvis.initialize()

    # Run main loop
    await jarvis.run()


if __name__ == "__main__":
    asyncio.run(main())
