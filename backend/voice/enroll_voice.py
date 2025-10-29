#!/usr/bin/env python3
"""
Voice Enrollment System for JARVIS
Collects voice samples and creates speaker profile for biometric verification

Usage:
    python backend/voice/enroll_voice.py --speaker "Derek J. Russell" --samples 25
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from intelligence.learning_database import JARVISLearningDatabase
from voice.engines.speechbrain_engine import SpeechBrainEngine
from voice.stt_config import ModelConfig, STTEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class VoiceEnrollment:
    """Voice enrollment system for speaker verification"""

    def __init__(self, speaker_name: str, num_samples: int = 25):
        self.speaker_name = speaker_name
        self.num_samples = num_samples
        self.sample_rate = 16000  # 16kHz for SpeechBrain
        self.duration_seconds = 10  # 10 seconds per sample
        self.learning_db = None
        self.speechbrain_engine = None
        self.collected_samples = []
        self.collected_embeddings = []

    async def initialize(self):
        """Initialize database and SpeechBrain engine"""
        logger.info("🚀 Initializing JARVIS Voice Enrollment System...")

        # Initialize learning database
        self.learning_db = JARVISLearningDatabase()
        await self.learning_db.initialize()

        # Initialize SpeechBrain engine for speaker embeddings
        model_config = ModelConfig(
            name="speechbrain-wav2vec2",
            engine=STTEngine.SPEECHBRAIN,
            disk_size_mb=380,
            ram_required_gb=2.0,
            vram_required_gb=1.8,
            expected_accuracy=0.96,
            avg_latency_ms=150,
            supports_fine_tuning=True,
            model_path="speechbrain/asr-wav2vec2-commonvoice-en",
        )

        self.speechbrain_engine = SpeechBrainEngine(model_config)
        await self.speechbrain_engine.initialize()

        logger.info("✅ System initialized\n")

    async def record_sample(self, sample_num: int, phrase: str) -> bytes:
        """
        Record a single voice sample

        Args:
            sample_num: Sample number (1-indexed)
            phrase: Phrase for user to say

        Returns:
            Audio data as bytes
        """
        print(f"\n{'='*60}")
        print(f"Sample {sample_num}/{self.num_samples}")
        print(f"{'='*60}")
        print(f'\n📝 Please say: "{phrase}"')
        print(f"\n⏳ Get ready! Recording will start in...")

        # Countdown
        for i in range(5, 0, -1):
            print(f"   {i}...")
            await asyncio.sleep(1)

        print(f"\n🎙️  RECORDING NOW... (speak clearly for {self.duration_seconds} seconds)")
        print("=" * 60)

        # Record audio
        audio_data = sd.rec(
            int(self.duration_seconds * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()  # Wait until recording is finished

        print("✅ Recording complete!")

        # Convert to bytes (WAV format)
        audio_bytes = self._audio_to_wav_bytes(audio_data)

        return audio_bytes

    def _audio_to_wav_bytes(self, audio_data: np.ndarray) -> bytes:
        """Convert numpy audio to WAV bytes"""
        import io

        buffer = io.BytesIO()
        sf.write(buffer, audio_data, self.sample_rate, format="WAV")
        buffer.seek(0)
        return buffer.read()

    async def extract_speaker_embedding(self, audio_bytes: bytes) -> np.ndarray:
        """
        Extract speaker embedding from audio using SpeechBrain

        Args:
            audio_bytes: Audio data as WAV bytes

        Returns:
            Speaker embedding as numpy array
        """
        # Convert audio to tensor
        audio_tensor, sample_rate = await self.speechbrain_engine._audio_bytes_to_tensor(
            audio_bytes
        )

        # For now, use simple statistical features as embedding
        # TODO: Use actual SpeechBrain speaker encoder when available
        embedding = np.array(
            [
                float(np.mean(audio_tensor.numpy())),
                float(np.std(audio_tensor.numpy())),
                float(np.max(audio_tensor.numpy())),
                float(np.min(audio_tensor.numpy())),
                float(np.median(audio_tensor.numpy())),
                float(np.percentile(audio_tensor.numpy(), 25)),
                float(np.percentile(audio_tensor.numpy(), 75)),
                float(len(audio_tensor.numpy())),
            ]
        )

        return embedding

    async def enroll_speaker(self):
        """Main enrollment flow"""
        print("\n" + "=" * 60)
        print("🎤 JARVIS VOICE ENROLLMENT")
        print("=" * 60)
        print(f"\nEnrolling speaker: {self.speaker_name}")
        print(f"Number of samples: {self.num_samples}")
        print(f"Sample duration: {self.duration_seconds} seconds each")
        print("\nTips for best results:")
        print("  • Speak naturally and clearly")
        print("  • Use your normal speaking volume")
        print("  • Vary your tone across samples")
        print("  • Record in a quiet environment")
        print("  • Position microphone consistently")
        print("\n" + "=" * 60)

        input("\nPress ENTER to start enrollment...")

        # Phrases for enrollment (varied to capture different phonemes)
        phrases = [
            "Hey JARVIS, what's the weather today?",
            "Open Safari and search for dogs",
            "Connect to the living room TV",
            "Set a timer for 10 minutes",
            "What's on my calendar tomorrow?",
            "Turn on the lights in the bedroom",
            "Play some jazz music",
            "Send an email to my assistant",
            "What time is my next meeting?",
            "Navigate to the nearest coffee shop",
            "Add milk to my shopping list",
            "What's the latest news?",
            "Set an alarm for 7 AM",
            "Call my mom on speaker",
            "Show me photos from last week",
            "What's the stock price of Apple?",
            "Turn up the volume",
            "Pause the music",
            "What's the capital of France?",
            "Calculate 15 percent of 200",
            "Remind me to call John at 3 PM",
            "What's the traffic like to work?",
            "Turn off all the lights",
            "What's my heart rate?",
            "Read my latest messages",
        ]

        # Collect samples
        for i in range(self.num_samples):
            phrase = phrases[i % len(phrases)]

            try:
                audio_bytes = await self.record_sample(i + 1, phrase)
                self.collected_samples.append(audio_bytes)

                # Extract embedding
                embedding = await self.extract_speaker_embedding(audio_bytes)
                self.collected_embeddings.append(embedding)

                # Transcribe to store with sample
                result = await self.speechbrain_engine.transcribe(audio_bytes)
                transcription = result.text

                # Record in learning database
                await self.learning_db.record_voice_sample(
                    speaker_name=self.speaker_name,
                    audio_data=audio_bytes,
                    transcription=transcription,
                    audio_duration_ms=self.duration_seconds * 1000,
                    quality_score=result.confidence,
                )

                logger.info(f"✅ Sample {i+1} recorded and stored")

            except KeyboardInterrupt:
                logger.warning("\n⚠️  Enrollment interrupted by user")
                return False
            except Exception as e:
                logger.error(f"❌ Error recording sample {i+1}: {e}")
                retry = input("Retry this sample? (y/n): ")
                if retry.lower() == "y":
                    continue
                else:
                    return False

        # Compute average embedding
        print("\n" + "=" * 60)
        print("🧠 Computing speaker profile...")
        print("=" * 60)

        avg_embedding = np.mean(self.collected_embeddings, axis=0)
        std_embedding = np.std(self.collected_embeddings, axis=0)

        # Compute confidence based on consistency
        consistency_score = 1.0 - np.mean(std_embedding / (np.abs(avg_embedding) + 1e-8))
        confidence = max(0.5, min(1.0, consistency_score))

        logger.info(f"\n📊 Enrollment Statistics:")
        logger.info(f"   Total samples: {len(self.collected_samples)}")
        logger.info(f"   Embedding dimensions: {len(avg_embedding)}")
        logger.info(f"   Consistency score: {consistency_score:.2%}")
        logger.info(f"   Confidence: {confidence:.2%}")

        # Store speaker profile in database
        speaker_id = await self.learning_db.get_or_create_speaker_profile(self.speaker_name)

        # Serialize embedding
        embedding_bytes = avg_embedding.tobytes()

        await self.learning_db.update_speaker_embedding(
            speaker_id=speaker_id,
            embedding=embedding_bytes,
            confidence=confidence,
            is_primary_user=True,  # Derek is the primary user
        )

        print("\n" + "=" * 60)
        print("✅ ENROLLMENT COMPLETE!")
        print("=" * 60)
        print(f"\nSpeaker Profile Created:")
        print(f"  Name: {self.speaker_name}")
        print(f"  Speaker ID: {speaker_id}")
        print(f"  Confidence: {confidence:.1%}")
        print(f"  Samples: {len(self.collected_samples)}")
        print(f"\n🎉 JARVIS can now recognize your voice!")
        print("=" * 60)

        return True

    async def cleanup(self):
        """Cleanup resources"""
        if self.speechbrain_engine:
            await self.speechbrain_engine.cleanup()
        if self.learning_db:
            await self.learning_db.close()


async def main():
    parser = argparse.ArgumentParser(description="Enroll speaker for JARVIS voice verification")
    parser.add_argument(
        "--speaker",
        type=str,
        default="Derek J. Russell",
        help="Speaker name (default: Derek J. Russell)",
    )
    parser.add_argument(
        "--samples", type=int, default=25, help="Number of voice samples to collect (default: 25)"
    )

    args = parser.parse_args()

    enrollment = VoiceEnrollment(speaker_name=args.speaker, num_samples=args.samples)

    try:
        await enrollment.initialize()
        success = await enrollment.enroll_speaker()

        if success:
            print("\n🎊 Voice enrollment successful!")
            return 0
        else:
            print("\n❌ Voice enrollment failed or was cancelled")
            return 1

    except Exception as e:
        logger.error(f"❌ Enrollment failed: {e}", exc_info=True)
        return 1

    finally:
        await enrollment.cleanup()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n\n⚠️  Enrollment cancelled by user")
        sys.exit(1)
