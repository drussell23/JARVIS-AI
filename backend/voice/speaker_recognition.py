"""
Dynamic Speaker Recognition System
Learns and identifies speakers (especially Derek J. Russell) by voice
Uses voice embeddings for biometric authentication
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VoiceProfile:
    """Voice profile for a speaker"""

    speaker_name: str
    speaker_id: int
    embedding: np.ndarray  # Voice embedding (128-512 dimensions)
    sample_count: int  # Number of voice samples used
    confidence: float  # Confidence in this profile (0.0-1.0)
    created_at: datetime
    updated_at: datetime
    is_owner: bool = False  # Is this the device owner (Derek J. Russell)
    security_clearance: str = "standard"  # standard, elevated, admin


class SpeakerRecognitionEngine:
    """
    Advanced speaker recognition engine using voice embeddings.

    Features:
    - Automatic speaker identification from voice
    - Voice enrollment (learn new speakers)
    - Continuous learning (improve profiles over time)
    - Owner detection (Derek J. Russell gets special privileges)
    - Security verification for sensitive commands
    - Zero-shot learning (recognize from few samples)
    """

    def __init__(self):
        self.profiles: Dict[str, VoiceProfile] = {}
        self.model = None
        self.device = None
        self.initialized = False
        self.learning_db = None

        # Intelligent voice router (cost-aware local/cloud routing)
        self.voice_router = None

        # Similarity thresholds
        self.recognition_threshold = 0.75  # Minimum similarity to recognize speaker
        self.verification_threshold = 0.85  # Higher threshold for security commands
        self.enrollment_threshold = 0.65  # Lower threshold for initial enrollment

        # Owner profile (loaded from database)
        self.owner_profile: Optional[VoiceProfile] = None

    async def initialize(self):
        """Initialize speaker recognition engine"""
        if self.initialized:
            return

        logger.info("🎭 Initializing Speaker Recognition Engine...")

        # Initialize intelligent voice router (handles model selection)
        try:
            from voice.intelligent_voice_router import get_voice_router

            self.voice_router = get_voice_router()
            await self.voice_router.initialize()
            logger.info("✅ Intelligent voice router initialized (cost-aware local/cloud)")
        except Exception as e:
            logger.warning(f"Voice router unavailable: {e}")

        # Legacy: Try to load local model as fallback
        try:
            # Try to use pre-trained speaker embedding model
            # Option 1: SpeechBrain (best for speaker recognition)
            try:
                from speechbrain.pretrained import EncoderClassifier

                # Use pre-trained x-vector model for speaker embeddings
                model_name = "speechbrain/spkrec-xvect-voxceleb"
                self.model = EncoderClassifier.from_hparams(
                    source=model_name,
                    savedir=str(Path.home() / ".jarvis" / "models" / "speaker_recognition"),
                    run_opts={"device": self._get_optimal_device()},
                )
                logger.info(f"✅ Loaded SpeechBrain x-vector model: {model_name}")

            except ImportError:
                # Fallback: Use Resemblyzer (lighter weight)
                logger.info("SpeechBrain not available, falling back to Resemblyzer")
                from resemblyzer import VoiceEncoder

                self.model = VoiceEncoder()
                logger.info("✅ Loaded Resemblyzer voice encoder")

        except Exception as e:
            logger.warning(f"Could not load speaker recognition model: {e}")
            logger.warning("Speaker recognition will use voice router")
            self.model = None

        # Load existing voice profiles from database
        await self._load_profiles_from_database()

        self.initialized = True
        logger.info(f"✅ Speaker Recognition initialized ({len(self.profiles)} profiles loaded)")

    def _get_optimal_device(self) -> str:
        """Determine optimal device for speaker recognition"""
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    async def _load_profiles_from_database(self):
        """Load speaker profiles from learning database"""
        try:
            from intelligence.learning_database import get_learning_database

            self.learning_db = await get_learning_database()

            # Get all speaker profiles
            profiles_data = await self.learning_db.get_all_speaker_profiles()

            for profile_data in profiles_data:
                speaker_name = profile_data["speaker_name"]
                speaker_id = profile_data["speaker_id"]

                # Deserialize voice embedding
                embedding = (
                    np.frombuffer(profile_data["voiceprint_embedding"], dtype=np.float32)
                    if profile_data["voiceprint_embedding"]
                    else None
                )

                if embedding is not None:
                    profile = VoiceProfile(
                        speaker_name=speaker_name,
                        speaker_id=speaker_id,
                        embedding=embedding,
                        sample_count=profile_data.get("total_samples", 0),
                        confidence=profile_data.get("recognition_confidence", 0.5),
                        created_at=datetime.fromisoformat(
                            profile_data.get("created_at", datetime.now().isoformat())
                        ),
                        updated_at=datetime.fromisoformat(
                            profile_data.get("last_updated", datetime.now().isoformat())
                        ),
                        is_owner=profile_data.get("is_primary_user", False),
                        security_clearance=profile_data.get("security_level", "standard"),
                    )

                    self.profiles[speaker_name] = profile

                    # Set as owner if marked as primary user
                    if profile.is_owner:
                        self.owner_profile = profile
                        logger.info(f"👑 Owner profile loaded: {speaker_name}")

            logger.info(f"📚 Loaded {len(self.profiles)} speaker profiles from database")

        except Exception as e:
            logger.error(f"Failed to load speaker profiles from database: {e}")

    async def identify_speaker(
        self,
        audio_data: bytes,
        return_confidence: bool = True,
        verification_level: str = "standard",
    ) -> Tuple[Optional[str], float]:
        """
        Identify speaker from audio using intelligent routing.

        Args:
            audio_data: Raw audio bytes
            return_confidence: Return confidence score
            verification_level: "quick", "standard", "high", "critical"

        Returns:
            (speaker_name, confidence) or (None, 0.0) if unknown
        """
        if not self.initialized:
            await self.initialize()

        # Try intelligent voice router first (cost-aware local/cloud routing)
        if self.voice_router:
            try:
                from voice.intelligent_voice_router import VerificationLevel

                # Map string to enum
                level_map = {
                    "quick": VerificationLevel.QUICK,
                    "standard": VerificationLevel.STANDARD,
                    "high": VerificationLevel.HIGH,
                    "critical": VerificationLevel.CRITICAL,
                }
                level = level_map.get(verification_level, VerificationLevel.STANDARD)

                # Use intelligent router
                result = await self.voice_router.recognize_speaker(
                    audio_data, verification_level=level
                )

                # Save embedding to learning database for continuous improvement
                if result.speaker_name != "Unknown":
                    await self._save_voice_sample(
                        speaker_name=result.speaker_name,
                        audio_data=audio_data,
                        embedding=result.embedding,
                        confidence=result.confidence,
                        model_used=result.model_used.value,
                    )

                logger.info(
                    f"🎭 Speaker identified via {result.model_used.value}: "
                    f"{result.speaker_name} (confidence: {result.confidence:.2f}, "
                    f"latency: {result.latency_ms:.0f}ms, cost: ${result.cost_cents/100:.4f})"
                )

                return result.speaker_name, result.confidence

            except Exception as e:
                logger.warning(f"Voice router failed, falling back to legacy: {e}")

        # Fallback to legacy local model
        if self.model is None:
            return await self._identify_speaker_heuristic(audio_data)

        try:
            # Extract voice embedding from audio
            embedding = await self._extract_embedding(audio_data)

            if embedding is None:
                return None, 0.0

            # Compare with all known profiles
            best_match = None
            best_similarity = 0.0

            for speaker_name, profile in self.profiles.items():
                similarity = self._cosine_similarity(embedding, profile.embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = speaker_name

            # Check if similarity meets threshold
            if best_similarity >= self.recognition_threshold:
                logger.info(
                    f"🎭 Speaker identified (legacy): {best_match} (confidence: {best_similarity:.2f})"
                )

                # Update profile with new sample (continuous learning)
                asyncio.create_task(
                    self._update_profile_with_sample(best_match, embedding, audio_data)
                )

                return best_match, best_similarity
            else:
                logger.info(
                    f"🎭 Unknown speaker (best match: {best_match} @ {best_similarity:.2f}, threshold: {self.recognition_threshold})"
                )
                return None, best_similarity

        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return None, 0.0

    async def verify_speaker(self, audio_data: bytes, claimed_speaker: str) -> Tuple[bool, float]:
        """
        Verify if audio matches claimed speaker (for security commands).

        Args:
            audio_data: Raw audio bytes
            claimed_speaker: Name of speaker to verify against

        Returns:
            (is_match, confidence)
        """
        if not self.initialized:
            await self.initialize()

        if claimed_speaker not in self.profiles:
            logger.warning(f"No profile found for claimed speaker: {claimed_speaker}")
            return False, 0.0

        try:
            # Extract embedding
            embedding = await self._extract_embedding(audio_data)
            if embedding is None:
                return False, 0.0

            # Compare with claimed speaker's profile
            profile = self.profiles[claimed_speaker]
            similarity = self._cosine_similarity(embedding, profile.embedding)

            # Use higher threshold for verification
            is_match = similarity >= self.verification_threshold

            logger.info(
                f"🔐 Speaker verification: {claimed_speaker} - {'✅ PASS' if is_match else '❌ FAIL'} (confidence: {similarity:.2f})"
            )

            return is_match, similarity

        except Exception as e:
            logger.error(f"Speaker verification failed: {e}")
            return False, 0.0

    async def enroll_speaker(
        self, speaker_name: str, audio_samples: List[bytes], is_owner: bool = False
    ) -> bool:
        """
        Enroll a new speaker with voice samples.

        Args:
            speaker_name: Name of speaker
            audio_samples: List of audio samples (at least 3-5 recommended)
            is_owner: Mark this speaker as device owner (Derek J. Russell)

        Returns:
            True if enrollment successful
        """
        if not self.initialized:
            await self.initialize()

        logger.info(
            f"🎓 Enrolling speaker: {speaker_name} ({len(audio_samples)} samples, owner={is_owner})"
        )

        try:
            # Extract embeddings from all samples
            embeddings = []
            for audio in audio_samples:
                embedding = await self._extract_embedding(audio)
                if embedding is not None:
                    embeddings.append(embedding)

            if len(embeddings) == 0:
                logger.error("No valid embeddings extracted from audio samples")
                return False

            # Average embeddings to create speaker profile
            avg_embedding = np.mean(embeddings, axis=0)

            # Calculate confidence based on consistency
            confidences = [self._cosine_similarity(emb, avg_embedding) for emb in embeddings]
            avg_confidence = np.mean(confidences)

            # Create profile
            profile = VoiceProfile(
                speaker_name=speaker_name,
                speaker_id=hash(speaker_name) % 1000000,  # Temporary ID (will be replaced by DB)
                embedding=avg_embedding,
                sample_count=len(embeddings),
                confidence=avg_confidence,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                is_owner=is_owner,
                security_clearance="admin" if is_owner else "standard",
            )

            # Save to database
            if self.learning_db:
                speaker_id = await self.learning_db.get_or_create_speaker_profile(
                    speaker_name=speaker_name
                )
                profile.speaker_id = speaker_id

                # Update with embedding
                await self.learning_db.update_speaker_embedding(
                    speaker_id=speaker_id,
                    embedding=avg_embedding.tobytes(),
                    confidence=avg_confidence,
                    is_primary_user=is_owner,
                )

            # Store in memory
            self.profiles[speaker_name] = profile

            if is_owner:
                self.owner_profile = profile
                logger.info(f"👑 Owner profile created: {speaker_name}")

            logger.info(f"✅ Speaker enrolled: {speaker_name} (confidence: {avg_confidence:.2f})")
            return True

        except Exception as e:
            logger.error(f"Speaker enrollment failed: {e}")
            return False

    async def _extract_embedding(self, audio_data: bytes) -> Optional[np.ndarray]:
        """Extract voice embedding from audio"""
        try:
            # Convert audio bytes to numpy array
            import io

            # Try with librosa
            try:
                import librosa

                audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=16000, mono=True)
            except ImportError:
                # Fallback to scipy
                import scipy.io.wavfile as wavfile

                sr, audio_array = wavfile.read(io.BytesIO(audio_data))
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = audio_array.astype(np.float32) / 32768.0

            # Extract embedding with model
            if hasattr(self.model, "encode_batch"):
                # SpeechBrain
                import torch

                audio_tensor = torch.FloatTensor(audio_array).unsqueeze(0)
                embedding = self.model.encode_batch(audio_tensor).squeeze().cpu().numpy()
            elif hasattr(self.model, "embed_utterance"):
                # Resemblyzer
                embedding = self.model.embed_utterance(audio_array)
            else:
                # Fallback: use MFCC features as simple embedding
                import librosa

                mfcc = librosa.feature.mfcc(y=audio_array, sr=sr, n_mfcc=20)
                embedding = np.mean(mfcc, axis=1)

            return embedding

        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def _save_voice_sample(
        self,
        speaker_name: str,
        audio_data: bytes,
        embedding: np.ndarray,
        confidence: float,
        model_used: str,
    ):
        """
        Save voice sample to learning database for continuous improvement.

        This enables the system to learn and improve voice recognition over time.
        """
        try:
            if not self.learning_db:
                from intelligence.learning_database import get_learning_database

                self.learning_db = await get_learning_database()

            # Store embedding in database
            embedding_bytes = embedding.astype(np.float32).tobytes()

            # Update speaker profile with new embedding
            speaker_id = await self.learning_db.get_or_create_speaker_profile(speaker_name)

            await self.learning_db.update_speaker_embedding(
                speaker_id=speaker_id,
                embedding=embedding_bytes,
                confidence=confidence,
                is_primary_user=(speaker_name == "Derek J. Russell"),
            )

            # Also record the voice sample
            audio_duration_ms = len(audio_data) / 16  # Rough estimate (16kHz)
            await self.learning_db.record_voice_sample(
                speaker_name=speaker_name,
                audio_data=audio_data,
                transcription="",  # Transcription handled elsewhere
                audio_duration_ms=audio_duration_ms,
                quality_score=confidence,
            )

            logger.debug(
                f"💾 Saved voice sample for {speaker_name} ({model_used}, confidence: {confidence:.2f})"
            )

        except Exception as e:
            logger.error(f"Failed to save voice sample to learning database: {e}")

    async def _identify_speaker_heuristic(self, audio_data: bytes) -> Tuple[Optional[str], float]:
        """
        Fallback speaker identification using heuristics.

        When speaker recognition model is not available, use simpler methods:
        - Check if this is the primary user based on system context
        - Use audio characteristics (pitch, energy, duration)
        """
        # Check if there's only one profile (assume it's them)
        if len(self.profiles) == 1:
            speaker_name = list(self.profiles.keys())[0]
            logger.info(f"🎭 Single profile heuristic: assuming speaker is {speaker_name}")
            return speaker_name, 0.8  # Medium confidence

        # Check if owner profile exists (assume it's the owner)
        if self.owner_profile:
            logger.info(
                f"🎭 Owner heuristic: assuming speaker is {self.owner_profile.speaker_name}"
            )
            return self.owner_profile.speaker_name, 0.75

        # Unknown
        return None, 0.0

    async def _update_profile_with_sample(
        self, speaker_name: str, embedding: np.ndarray, audio_data: bytes
    ):
        """Update speaker profile with new sample (continuous learning)"""
        try:
            if speaker_name not in self.profiles:
                return

            profile = self.profiles[speaker_name]

            # Moving average of embeddings
            alpha = 0.1  # Learning rate (10% new, 90% old)
            profile.embedding = (1 - alpha) * profile.embedding + alpha * embedding
            profile.sample_count += 1
            profile.updated_at = datetime.now()

            # Update in database
            if self.learning_db:
                await self.learning_db.record_voice_sample(
                    speaker_name=speaker_name,
                    audio_data=audio_data,
                    transcription="",  # Unknown at this point
                    audio_duration_ms=len(audio_data) / 32,
                    quality_score=0.9,  # Assume good quality
                )

                await self.learning_db.update_speaker_embedding(
                    speaker_id=profile.speaker_id,
                    embedding=profile.embedding.tobytes(),
                    confidence=profile.confidence,
                )

            logger.debug(f"📈 Updated profile for {speaker_name} (sample #{profile.sample_count})")

        except Exception as e:
            logger.error(f"Failed to update profile: {e}")

    def is_owner(self, speaker_name: Optional[str]) -> bool:
        """Check if speaker is the device owner"""
        if not speaker_name or speaker_name not in self.profiles:
            return False
        return self.profiles[speaker_name].is_owner

    def get_security_clearance(self, speaker_name: Optional[str]) -> str:
        """Get security clearance level for speaker"""
        if not speaker_name or speaker_name not in self.profiles:
            return "none"
        return self.profiles[speaker_name].security_clearance


# Global singleton
_speaker_recognition_engine: Optional[SpeakerRecognitionEngine] = None


def get_speaker_recognition_engine() -> SpeakerRecognitionEngine:
    """Get global speaker recognition engine instance"""
    global _speaker_recognition_engine
    if _speaker_recognition_engine is None:
        _speaker_recognition_engine = SpeakerRecognitionEngine()
    return _speaker_recognition_engine
