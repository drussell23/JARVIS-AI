"""
Speaker Verification Service for JARVIS
Provides voice biometric verification for security-sensitive operations

Features:
- Speaker identification from audio
- Confidence scoring
- Primary user (owner) detection
- Integration with learning database
- Background pre-loading for instant unlock
"""

import asyncio
import logging
import threading
from typing import Optional

import numpy as np
from intelligence.learning_database import JARVISLearningDatabase
from voice.engines.speechbrain_engine import SpeechBrainEngine
from voice.stt_config import ModelConfig, STTEngine

logger = logging.getLogger(__name__)

# Global instance for pre-loaded service (set by start_system.py)
_global_speaker_service = None


class SpeakerVerificationService:
    """
    Speaker verification service for JARVIS

    Verifies speaker identity using voice biometrics
    """

    def __init__(self, learning_db: Optional[JARVISLearningDatabase] = None):
        """
        Initialize speaker verification service

        Args:
            learning_db: LearningDatabase instance (optional, will create if not provided)
        """
        self.learning_db = learning_db
        self.speechbrain_engine = None
        self.initialized = False
        self.speaker_profiles = {}  # Cache of speaker profiles
        self.verification_threshold = 0.75  # 75% confidence for verification (native profiles)
        self.legacy_threshold = 0.50  # 50% for legacy profiles with dimension mismatch
        self.profile_quality_scores = {}  # Track profile quality (1.0 = native, <1.0 = legacy)
        self._preload_thread = None
        self._encoder_preloading = False
        self._encoder_preloaded = False

    async def initialize_fast(self):
        """
        Fast initialization with background encoder pre-loading.

        Loads profiles immediately, starts encoder loading in background.
        JARVIS starts fast (~2s), encoder ready in ~10s total.
        """
        if self.initialized:
            return

        logger.info("🔐 Initializing Speaker Verification Service (fast mode)...")

        # Initialize learning database if not provided - use singleton
        if self.learning_db is None:
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

        # Initialize SpeechBrain engine for embeddings
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

        # Load speaker profiles from database
        await self._load_speaker_profiles()

        self.initialized = True
        logger.info(
            f"✅ Speaker Verification Service ready ({len(self.speaker_profiles)} profiles loaded)"
        )

        # Start background pre-loading of encoder
        logger.info("🔄 Pre-loading speaker encoder in background...")
        self._start_background_preload()

    def _start_background_preload(self):
        """Start background thread to pre-load speaker encoder"""
        if self._encoder_preloading or self._encoder_preloaded:
            return

        self._encoder_preloading = True

        def preload_worker():
            """Worker function to pre-load encoder in background thread"""
            try:
                # Run async function in thread's event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.speechbrain_engine._load_speaker_encoder())
                loop.close()

                self._encoder_preloaded = True
                self._encoder_preloading = False
                logger.info("✅ Speaker encoder pre-loaded in background - unlock is now instant!")

            except Exception as e:
                logger.error(f"Background encoder pre-loading failed: {e}", exc_info=True)
                self._encoder_preloading = False

        self._preload_thread = threading.Thread(target=preload_worker, daemon=True)
        self._preload_thread.start()

    async def initialize(self, preload_encoder: bool = True):
        """
        Initialize service and load speaker profiles

        Args:
            preload_encoder: If True, pre-loads ECAPA-TDNN encoder during initialization
                           for instant unlock (adds ~10s to startup, but unlock is instant)
        """
        if self.initialized:
            return

        logger.info("🔐 Initializing Speaker Verification Service...")

        # Initialize learning database if not provided - use singleton
        if self.learning_db is None:
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

        # Initialize SpeechBrain engine for embeddings
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

        # Load speaker profiles from database
        await self._load_speaker_profiles()

        # Pre-load speaker encoder for instant unlock (if enabled)
        if preload_encoder:
            logger.info("🔄 Pre-loading speaker encoder (ECAPA-TDNN) for instant unlock...")
            await self.speechbrain_engine._load_speaker_encoder()
            logger.info("✅ Speaker encoder pre-loaded - unlock will be instant!")

        self.initialized = True
        logger.info(
            f"✅ Speaker Verification Service ready ({len(self.speaker_profiles)} profiles loaded)"
        )

    async def _load_speaker_profiles(self):
        """Load speaker profiles from learning database"""
        try:
            logger.info("🔄 Loading speaker profiles from database...")

            # Use singleton to get the shared database instance
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

            profiles = await self.learning_db.get_all_speaker_profiles()
            logger.info(f"📊 Found {len(profiles)} speaker profiles in database")

            for profile in profiles:
                speaker_id = profile["speaker_id"]
                speaker_name = profile["speaker_name"]

                # Deserialize embedding
                embedding_bytes = profile.get("voiceprint_embedding")
                if embedding_bytes:
                    embedding = np.frombuffer(embedding_bytes, dtype=np.float64)

                    # Assess profile quality based on embedding dimension
                    # Current ECAPA-TDNN uses 192 dimensions
                    is_native = embedding.shape[0] == 192 # Native profile if 192D embedding matches encoder dimension (192D) 
                    total_samples = profile.get("total_samples", 0) # Total audio samples used for profile creation 

                    # Determine quality and threshold based on samples and native status 
                    if is_native and total_samples >= 100: 
                        quality = "excellent" # High-quality native profile 
                        threshold = self.verification_threshold  # 0.75 
                    elif is_native and total_samples >= 50:
                        quality = "good" # Medium-quality native profile
                        threshold = self.verification_threshold  # 0.75
                    elif total_samples >= 50:
                        quality = "fair" # Medium-quality legacy profile 
                        threshold = self.legacy_threshold  # 0.50
                    else: 
                        quality = "legacy" # Low-quality legacy profile 
                        threshold = self.legacy_threshold  # 0.50

                    # Store profile in cache and quality scores for adaptive thresholding and verification 
                    self.speaker_profiles[speaker_name] = {
                        "speaker_id": speaker_id,
                        "embedding": embedding,
                        "confidence": profile.get("recognition_confidence", 0.0),
                        "is_primary_user": profile.get("is_primary_user", False),
                        "security_level": profile.get("security_level", "standard"),
                        "total_samples": total_samples,
                        "is_native": is_native,
                        "quality": quality,
                        "threshold": threshold,
                    }

                    # Store quality score for adaptive threshold
                    self.profile_quality_scores[speaker_name] = {
                        "is_native": is_native,
                        "quality": quality,
                        "threshold": threshold,
                        "samples": total_samples,
                    }

                    logger.info(
                        f"✅ Loaded speaker profile: {speaker_name} "
                        f"(ID: {speaker_id}, Primary: {profile.get('is_primary_user', False)}, "
                        f"Embedding: {embedding.shape[0]}D, Quality: {quality}, "
                        f"Threshold: {threshold*100:.0f}%, Samples: {total_samples})"
                    )
                else:
                    logger.warning(f"⚠️ Speaker profile {speaker_name} has no embedding!")

            logger.info(f"✅ Successfully loaded {len(self.speaker_profiles)} speaker profiles")

        except Exception as e:
            logger.error(f"❌ Failed to load speaker profiles: {e}", exc_info=True)
            logger.warning("⚠️ Continuing with 0 profiles - voice verification will fail until profiles are loaded")

    async def verify_speaker(self, audio_data: bytes, speaker_name: Optional[str] = None) -> dict:
        """
        Verify speaker from audio

        Args:
            audio_data: Audio bytes (WAV format)
            speaker_name: Expected speaker name (if None, identifies from all profiles)

        Returns:
            Verification result dict with:
                - verified: bool
                - confidence: float
                - speaker_name: str
                - is_owner: bool
                - security_level: str
        """
        if not self.initialized:
            await self.initialize()

        try:
            # If speaker name provided, verify against that profile
            if speaker_name and speaker_name in self.speaker_profiles:
                profile = self.speaker_profiles[speaker_name]
                known_embedding = profile["embedding"]
                profile_threshold = profile.get("threshold", self.verification_threshold)

                is_verified, confidence = await self.speechbrain_engine.verify_speaker(
                    audio_data, known_embedding, threshold=profile_threshold
                )

                return {
                    "verified": is_verified,
                    "confidence": confidence,
                    "speaker_name": speaker_name,
                    "speaker_id": profile["speaker_id"],
                    "is_owner": profile["is_primary_user"],
                    "security_level": profile["security_level"],
                }

            # Otherwise, identify speaker from all profiles
            best_match = None
            best_confidence = 0.0

            for profile_name, profile in self.speaker_profiles.items():
                known_embedding = profile["embedding"]
                profile_threshold = profile.get("threshold", self.verification_threshold)
                is_verified, confidence = await self.speechbrain_engine.verify_speaker(
                    audio_data, known_embedding, threshold=profile_threshold
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        "verified": is_verified,
                        "confidence": confidence,
                        "speaker_name": profile_name,
                        "speaker_id": profile["speaker_id"],
                        "is_owner": profile["is_primary_user"],
                        "security_level": profile["security_level"],
                    }

            if best_match:
                logger.info(
                    f"🎤 Speaker identified: {best_match['speaker_name']} "
                    f"(confidence: {best_match['confidence']:.1%}, "
                    f"owner: {best_match['is_owner']})"
                )
                return best_match

            # No match found
            return {
                "verified": False,
                "confidence": 0.0,
                "speaker_name": "unknown",
                "speaker_id": None,
                "is_owner": False,
                "security_level": "none",
            }

        except Exception as e:
            logger.error(f"Speaker verification error: {e}", exc_info=True)
            return {
                "verified": False,
                "confidence": 0.0,
                "speaker_name": "error",
                "speaker_id": None,
                "is_owner": False,
                "security_level": "none",
                "error": str(e),
            }

    async def is_owner(self, audio_data: bytes) -> tuple[bool, float]:
        """
        Check if audio is from the device owner (Derek J. Russell)

        Args:
            audio_data: Audio bytes

        Returns:
            Tuple of (is_owner, confidence)
        """
        result = await self.verify_speaker(audio_data)
        return result["is_owner"], result["confidence"]

    async def get_speaker_name(self, audio_data: bytes) -> str:
        """
        Get speaker name from audio

        Args:
            audio_data: Audio bytes

        Returns:
            Speaker name or "unknown"
        """
        result = await self.verify_speaker(audio_data)
        return result["speaker_name"]

    async def refresh_profiles(self):
        """Reload speaker profiles from database"""
        logger.info("🔄 Refreshing speaker profiles...")
        self.speaker_profiles.clear()
        await self._load_speaker_profiles()

    async def cleanup(self):
        """Cleanup resources"""
        if self.speechbrain_engine:
            await self.speechbrain_engine.cleanup()

        self.speaker_profiles.clear()
        self.initialized = False
        logger.info("🧹 Speaker Verification Service cleaned up")


# Global singleton instance
_speaker_verification_service: Optional[SpeakerVerificationService] = None


async def get_speaker_verification_service(
    learning_db: Optional[JARVISLearningDatabase] = None,
) -> SpeakerVerificationService:
    """
    Get global speaker verification service instance

    First checks for pre-loaded service from start_system.py,
    then falls back to creating new instance if needed.

    Args:
        learning_db: LearningDatabase instance (optional)

    Returns:
        SpeakerVerificationService instance
    """
    global _speaker_verification_service, _global_speaker_service

    # First check if there's a pre-loaded service from start_system.py
    if _global_speaker_service is not None:
        logger.info("✅ Using pre-loaded speaker verification service")
        _speaker_verification_service = _global_speaker_service
        return _global_speaker_service

    # Otherwise use the singleton pattern
    if _speaker_verification_service is None:
        logger.info("🔐 Creating new speaker verification service...")
        _speaker_verification_service = SpeakerVerificationService(learning_db)
        # Use fast initialization to avoid blocking (encoder loads in background)
        await _speaker_verification_service.initialize_fast()

    return _speaker_verification_service


async def reset_speaker_verification_service():
    """Reset service (for testing)"""
    global _speaker_verification_service
    if _speaker_verification_service:
        await _speaker_verification_service.cleanup()
    _speaker_verification_service = None
