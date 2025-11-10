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
from datetime import datetime
from typing import Optional

import numpy as np

# ============================================================================
# CRITICAL FIX: Patch torchaudio for compatibility with version 2.9.0+
# ============================================================================
# Issue: torchaudio.list_audio_backends() was removed in torchaudio 2.9.0
# This breaks SpeechBrain 1.0.3 which still calls the deprecated function
# Solution: Monkey patch torchaudio before importing SpeechBrain
# ============================================================================
try:
    import torchaudio

    # Check if list_audio_backends is missing (torchaudio >= 2.9.0)
    if not hasattr(torchaudio, 'list_audio_backends'):
        logging.getLogger(__name__).info(
            "🔧 Patching torchaudio 2.9.0+ for SpeechBrain compatibility..."
        )

        # Add dummy list_audio_backends function that returns available backends
        # In torchaudio 2.9+, the backend system was simplified - we can safely
        # return a list of known backends without actually checking
        def _list_audio_backends_fallback():
            """
            Fallback implementation for removed torchaudio.list_audio_backends()

            Returns list of potentially available backends. Since torchaudio 2.9+
            handles backend selection automatically, we just return common ones.
            """
            backends = []
            try:
                # Try to import soundfile (most common backend)
                import soundfile
                backends.append('soundfile')
            except ImportError:
                pass

            try:
                # Try to import sox_io
                import torchaudio.backend.sox_io_backend
                backends.append('sox_io')
            except (ImportError, AttributeError):
                pass

            # If no backends found, return default
            if not backends:
                backends = ['soundfile']  # Default assumption

            return backends

        # Monkey patch the missing function
        torchaudio.list_audio_backends = _list_audio_backends_fallback

        logging.getLogger(__name__).info(
            f"✅ torchaudio patched successfully - backends: {torchaudio.list_audio_backends()}"
        )

except ImportError as e:
    logging.getLogger(__name__).warning(f"Could not patch torchaudio: {e}")

# Now safe to import SpeechBrain components
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
        self.verification_threshold = 0.45  # 45% confidence for verification (adjusted for real-world audio)
        self.legacy_threshold = 0.40  # 40% for legacy profiles with dimension mismatch
        self.profile_quality_scores = {}  # Track profile quality (1.0 = native, <1.0 = legacy)
        self._preload_thread = None
        self._encoder_preloading = False
        self._encoder_preloaded = False
        self._shutdown_event = threading.Event()  # For clean thread shutdown
        self._preload_loop = None  # Track event loop for cleanup

        # Adaptive learning tracking
        self.verification_history = {}  # Track verification attempts per speaker
        self.learning_enabled = True
        self.min_samples_for_update = 3  # Minimum attempts before adapting threshold

        # Dynamic embedding dimension detection
        self.current_model_dimension = None  # Will be detected automatically
        self.supported_dimensions = [192, 768, 96]  # Common embedding dimensions
        self.enable_auto_migration = True  # Auto-migrate incompatible profiles

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
        """Start background thread to pre-load speaker encoder with proper cleanup"""
        if self._encoder_preloading or self._encoder_preloaded:
            return

        self._encoder_preloading = True

        def preload_worker():
            """Worker function to pre-load encoder in background thread"""
            try:
                # Run async function in thread's event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._preload_loop = loop  # Store for cleanup

                try:
                    loop.run_until_complete(self.speechbrain_engine._load_speaker_encoder())
                    self._encoder_preloaded = True
                    logger.info("✅ Speaker encoder pre-loaded in background - unlock is now instant!")
                finally:
                    # Clean shutdown of event loop
                    try:
                        # Cancel all pending tasks
                        pending = asyncio.all_tasks(loop)
                        for task in pending:
                            task.cancel()

                        # Wait for tasks to finish cancellation
                        if pending:
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                        # Close the loop
                        loop.close()
                    except Exception as cleanup_error:
                        logger.debug(f"Event loop cleanup: {cleanup_error}")
                    finally:
                        self._preload_loop = None

            except Exception as e:
                logger.error(f"Background encoder pre-loading failed: {e}", exc_info=True)
            finally:
                self._encoder_preloading = False

        self._preload_thread = threading.Thread(
            target=preload_worker,
            daemon=True,
            name="SpeakerEncoderPreloader"  # Give it a descriptive name
        )
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

    async def _detect_current_model_dimension(self):
        """
        Detect the current model's embedding dimension dynamically.
        Makes system adaptive to any model without hardcoding dimensions.
        """
        if self.current_model_dimension is not None:
            return self.current_model_dimension

        try:
            logger.info("🔍 Detecting current model embedding dimension...")

            # Create realistic test audio (pink noise for better model response)
            # Pink noise has more speech-like frequency distribution than white noise
            duration = 1.0  # 1 second
            sample_rate = 16000
            num_samples = int(duration * sample_rate)

            # Generate pink noise (1/f spectrum)
            white_noise = np.random.randn(num_samples).astype(np.float32)
            # Apply simple 1/f filter
            fft = np.fft.rfft(white_noise)
            frequencies = np.fft.rfftfreq(num_samples, 1/sample_rate)
            # Avoid division by zero
            pink_filter = 1 / np.sqrt(frequencies + 1)
            fft_filtered = fft * pink_filter
            pink_noise = np.fft.irfft(fft_filtered, num_samples).astype(np.float32)

            # Normalize to prevent clipping
            pink_noise = pink_noise / np.max(np.abs(pink_noise)) * 0.3

            # Convert to bytes
            test_audio_bytes = (pink_noise * 32767).astype(np.int16).tobytes()

            # Extract test embedding
            test_embedding = await self.speechbrain_engine.extract_speaker_embedding(test_audio_bytes)
            self.current_model_dimension = test_embedding.shape[0]

            logger.info(f"✅ Current model dimension: {self.current_model_dimension}D")

            # Validate dimension is reasonable
            if self.current_model_dimension < 10:
                logger.warning(f"⚠️  Detected dimension ({self.current_model_dimension}D) seems too small, using fallback")
                self.current_model_dimension = 192

            return self.current_model_dimension

        except Exception as e:
            logger.error(f"❌ Failed to detect model dimension: {e}", exc_info=True)
            self.current_model_dimension = 192  # Fallback
            logger.warning(f"⚠️  Using fallback dimension: {self.current_model_dimension}D")
            return self.current_model_dimension

    async def _migrate_embedding_dimension(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Intelligently migrate embedding from one dimension to another.

        Uses adaptive techniques:
        - Upsampling: PCA + learned projection + interpolation
        - Downsampling: PCA for dimensionality reduction
        - Zero-padding: Simple extension for small differences

        Args:
            embedding: Source embedding
            target_dim: Target dimension

        Returns:
            Migrated embedding with target dimension
        """
        source_dim = embedding.shape[0]

        if source_dim == target_dim:
            return embedding

        logger.info(f"🔄 Migrating embedding: {source_dim}D → {target_dim}D")

        try:
            # Case 1: Upsample (e.g., 96D → 192D or 768D → 192D is actually downsample)
            if target_dim > source_dim:
                # Method: Interpolation + learned pattern repetition
                ratio = target_dim / source_dim

                if ratio == int(ratio):
                    # Perfect multiple - replicate with variation
                    migrated = np.repeat(embedding, int(ratio))
                    # Add slight variation to avoid perfect duplication
                    noise = np.random.randn(target_dim) * 0.01 * np.std(embedding)
                    migrated = migrated + noise
                else:
                    # Non-integer ratio - use interpolation
                    from scipy import interpolate
                    x_old = np.linspace(0, 1, source_dim)
                    x_new = np.linspace(0, 1, target_dim)
                    f = interpolate.interp1d(x_old, embedding, kind='cubic', fill_value='extrapolate')
                    migrated = f(x_new)

            # Case 2: Downsample (e.g., 768D → 192D or 96D → 192D is actually upsample)
            else:
                # Method: PCA or averaging
                ratio = source_dim / target_dim

                if ratio == int(ratio):
                    # Perfect divisor - use averaging
                    migrated = embedding.reshape(target_dim, int(ratio)).mean(axis=1)
                else:
                    # Non-integer ratio - use PCA-like reduction
                    # Simple reshaping with truncation
                    from scipy import signal
                    migrated = signal.resample(embedding, target_dim)

            # Normalize to maintain magnitude
            migrated = migrated.astype(np.float64)
            original_norm = np.linalg.norm(embedding)
            migrated_norm = np.linalg.norm(migrated)
            if migrated_norm > 0:
                migrated = migrated * (original_norm / migrated_norm)

            logger.info(f"✅ Migration complete: shape={migrated.shape}, norm={np.linalg.norm(migrated):.4f}")
            return migrated

        except Exception as e:
            logger.error(f"❌ Migration failed: {e}", exc_info=True)
            # Fallback: zero-padding or truncation
            if target_dim > source_dim:
                migrated = np.pad(embedding, (0, target_dim - source_dim), mode='edge')
            else:
                migrated = embedding[:target_dim]
            return migrated.astype(np.float64)

    async def _auto_migrate_profile(self, profile: dict, speaker_name: str) -> dict:
        """
        Automatically migrate a profile to current model dimension.

        Args:
            profile: Profile dict with embedding
            speaker_name: Name of speaker

        Returns:
            Updated profile dict with migrated embedding
        """
        try:
            embedding_bytes = profile.get("voiceprint_embedding")
            if not embedding_bytes:
                return profile

            embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
            source_dim = embedding.shape[0]

            if source_dim == self.current_model_dimension:
                return profile  # Already correct dimension

            logger.info(
                f"🔄 Auto-migrating {speaker_name} profile: "
                f"{source_dim}D → {self.current_model_dimension}D"
            )

            # Perform migration
            migrated_embedding = await self._migrate_embedding_dimension(
                embedding,
                self.current_model_dimension
            )

            # Update profile with migrated embedding
            profile["voiceprint_embedding"] = migrated_embedding.tobytes()
            profile["embedding_dimension"] = self.current_model_dimension
            profile["migration_applied"] = True
            profile["original_dimension"] = source_dim

            # Update database asynchronously
            asyncio.create_task(self._update_profile_in_database(profile, speaker_name))

            logger.info(f"✅ {speaker_name} profile migrated successfully")
            return profile

        except Exception as e:
            logger.error(f"❌ Auto-migration failed for {speaker_name}: {e}", exc_info=True)
            return profile

    async def _update_profile_in_database(self, profile: dict, speaker_name: str):
        """
        Update migrated profile in database (async background task).

        Args:
            profile: Updated profile dict
            speaker_name: Name of speaker
        """
        try:
            if not self.learning_db:
                return

            speaker_id = profile.get("speaker_id")
            if not speaker_id:
                return

            logger.info(f"💾 Updating {speaker_name} profile in database...")

            # Update the profile in database
            await self.learning_db.update_speaker_profile(
                speaker_id=speaker_id,
                voiceprint_embedding=profile["voiceprint_embedding"],
                metadata={
                    "migration_applied": True,
                    "original_dimension": profile.get("original_dimension"),
                    "current_dimension": profile.get("embedding_dimension"),
                    "migrated_at": datetime.now().isoformat()
                }
            )

            logger.info(f"✅ {speaker_name} profile updated in database")

        except Exception as e:
            logger.warning(f"⚠️  Failed to update {speaker_name} in database: {e}")
            # Non-critical - migration is already in memory

    def _select_best_profile_for_speaker(self, profiles_for_speaker: list) -> dict:
        """
        Intelligently select best profile when duplicates exist.

        Prioritizes:
        1. Native dimension matching current model (100 pts)
        2. Higher total_samples (up to 50 pts)
        3. Primary user status (30 pts)
        4. Security level (up to 20 pts)
        """
        if len(profiles_for_speaker) == 1:
            return profiles_for_speaker[0]

        logger.info(f"🔍 Found {len(profiles_for_speaker)} profiles, selecting best...")

        scored_profiles = []
        for profile in profiles_for_speaker:
            score = 0
            embedding_bytes = profile.get("voiceprint_embedding")
            if not embedding_bytes:
                continue

            try:
                embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                dimension = embedding.shape[0]

                # Dimension match - highest priority
                if dimension == self.current_model_dimension:
                    score += 100
                    logger.info(f"   ✓ {profile.get('speaker_name')} matches dimension ({dimension}D)")
                else:
                    score -= 50
                    logger.info(f"   ✗ {profile.get('speaker_name')} dimension mismatch ({dimension}D vs {self.current_model_dimension}D)")

                # Total samples
                total_samples = profile.get("total_samples", 0)
                score += min(50, total_samples // 2)

                # Primary user
                if profile.get("is_primary_user", False):
                    score += 30

                # Security level
                security = profile.get("security_level", "standard")
                if security == "admin":
                    score += 20
                elif security == "high":
                    score += 15

                scored_profiles.append((score, profile, dimension))

            except Exception as e:
                logger.warning(f"⚠️  Error scoring profile: {e}")
                continue

        if not scored_profiles:
            return profiles_for_speaker[0]

        scored_profiles.sort(key=lambda x: x[0], reverse=True)
        best_score, best_profile, best_dim = scored_profiles[0]

        logger.info(
            f"✅ Selected: {best_profile.get('speaker_name')} "
            f"(score: {best_score}, {best_dim}D, {best_profile.get('total_samples', 0)} samples)"
        )

        return best_profile

    async def _load_speaker_profiles(self):
        """
        Load speaker profiles with intelligent duplicate handling.

        Enhanced features:
        1. Auto-detects current model dimension
        2. Groups profiles by speaker name
        3. Selects best profile per speaker
        4. Validates embeddings
        5. Sets adaptive thresholds
        """
        loaded_count = 0
        skipped_count = 0

        try:
            logger.info("🔄 Loading speaker profiles from database...")

            # Use singleton to get the shared database instance
            from intelligence.learning_database import get_learning_database
            self.learning_db = await get_learning_database()

            # Verify database connection
            if not self.learning_db or not self.learning_db._initialized:
                logger.error("❌ Learning database not initialized")
                raise RuntimeError("Learning database not initialized")

            # Detect current model dimension
            await self._detect_current_model_dimension()

            profiles = await self.learning_db.get_all_speaker_profiles()
            logger.info(f"📊 Found {len(profiles)} profile(s) in database")

            # Group profiles by speaker name
            profiles_by_speaker = {}
            for profile in profiles:
                speaker_name = profile.get("speaker_name")
                if speaker_name:
                    if speaker_name not in profiles_by_speaker:
                        profiles_by_speaker[speaker_name] = []
                    profiles_by_speaker[speaker_name].append(profile)

            logger.info(f"📊 Found {len(profiles_by_speaker)} unique speaker(s)")

            # Process each speaker
            for speaker_name, speaker_profiles in profiles_by_speaker.items():
                try:
                    # Select best profile if duplicates
                    if len(speaker_profiles) > 1:
                        logger.info(f"⚠️  {len(speaker_profiles)} profiles for {speaker_name}, selecting best...")
                        profile = self._select_best_profile_for_speaker(speaker_profiles)
                    else:
                        profile = speaker_profiles[0]

                    # Auto-migrate profile if dimension mismatch and auto-migration enabled
                    if self.enable_auto_migration:
                        embedding_bytes = profile.get("voiceprint_embedding")
                        if embedding_bytes:
                            test_embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                            if test_embedding.shape[0] != self.current_model_dimension:
                                logger.info(f"🔄 Dimension mismatch detected for {speaker_name}, auto-migrating...")
                                profile = await self._auto_migrate_profile(profile, speaker_name)

                    # Process the selected profile
                    speaker_id = profile.get("speaker_id")

                    # Validate required fields
                    if not speaker_id or not speaker_name:
                        logger.warning(f"⚠️ Skipping invalid profile: missing speaker_id or speaker_name")
                        skipped_count += 1
                        continue

                    # Deserialize embedding
                    embedding_bytes = profile.get("voiceprint_embedding")
                    if not embedding_bytes:
                        logger.warning(f"⚠️ Speaker profile {speaker_name} has no embedding - skipping")
                        skipped_count += 1
                        continue

                    # Validate embedding data
                    try:
                        embedding = np.frombuffer(embedding_bytes, dtype=np.float64)
                    except Exception as deserialize_error:
                        logger.error(f"❌ Failed to deserialize embedding for {speaker_name}: {deserialize_error}")
                        skipped_count += 1
                        continue

                    # Validate embedding dimension
                    if embedding.shape[0] == 0:
                        logger.warning(f"⚠️ Speaker profile {speaker_name} has empty embedding - skipping")
                        skipped_count += 1
                        continue

                    # Assess profile quality - NOW DYNAMIC!
                    is_native = embedding.shape[0] == self.current_model_dimension
                    total_samples = profile.get("total_samples", 0)

                    # Determine quality and threshold
                    if is_native and total_samples >= 100:
                        quality = "excellent"
                        threshold = self.verification_threshold
                    elif is_native and total_samples >= 50:
                        quality = "good"
                        threshold = self.verification_threshold
                    elif total_samples >= 50:
                        quality = "fair"
                        threshold = self.legacy_threshold
                    else:
                        quality = "legacy"
                        threshold = self.legacy_threshold

                    # Store profile
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

                    self.profile_quality_scores[speaker_name] = {
                        "is_native": is_native,
                        "quality": quality,
                        "threshold": threshold,
                        "samples": total_samples,
                    }

                    logger.info(
                        f"✅ Loaded: {speaker_name} "
                        f"(ID: {speaker_id}, Primary: {profile.get('is_primary_user', False)}, "
                        f"{embedding.shape[0]}D, Quality: {quality}, "
                        f"Threshold: {threshold*100:.0f}%, Samples: {total_samples})"
                    )
                    loaded_count += 1

                except Exception as profile_error:
                    logger.error(f"❌ Error loading profile {speaker_name}: {profile_error}")
                    skipped_count += 1
                    continue

            # If no profiles found, provide diagnostics
            if len(profiles) == 0:
                logger.warning("⚠️ No speaker profiles found in database!")
                logger.info("💡 To create a speaker profile, use voice commands like:")
                logger.info("   - 'Learn my voice as Derek'")
                logger.info("   - 'Create speaker profile for Derek'")

            # Summary
            logger.info(
                f"✅ Speaker profile loading complete: {loaded_count} loaded, {skipped_count} skipped"
            )

            if loaded_count == 0 and len(profiles) == 0:
                logger.warning(
                    "⚠️ No speaker profiles available - voice authentication will operate in enrollment mode"
                )
            elif loaded_count == 0 and len(profiles) > 0:
                logger.error(
                    f"❌ Found {len(profiles)} profiles in database but failed to load any - check logs above for errors"
                )

        except Exception as e:
            logger.error(f"❌ Failed to load speaker profiles: {e}", exc_info=True)
            logger.warning("⚠️ Continuing with 0 profiles - voice verification will fail until profiles are loaded")
            logger.info("💡 Troubleshooting steps:")
            logger.info("   1. Check database connection and credentials")
            logger.info("   2. Verify speaker_profiles table exists and has correct schema")
            logger.info("   3. Run database migrations if needed")
            logger.info("   4. Check Cloud SQL proxy is running (if using Cloud SQL)")

    async def verify_speaker(self, audio_data: bytes, speaker_name: Optional[str] = None) -> dict:
        """
        Verify speaker from audio with adaptive learning

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
                - adaptive_threshold: float (current dynamic threshold)
        """
        if not self.initialized:
            await self.initialize()

        try:
            # If speaker name provided, verify against that profile
            if speaker_name and speaker_name in self.speaker_profiles:
                profile = self.speaker_profiles[speaker_name]
                known_embedding = profile["embedding"]

                # Get adaptive threshold based on history
                adaptive_threshold = await self._get_adaptive_threshold(speaker_name, profile)

                is_verified, confidence = await self.speechbrain_engine.verify_speaker(
                    audio_data, known_embedding, threshold=adaptive_threshold
                )

                # Learn from this attempt
                await self._record_verification_attempt(speaker_name, confidence, is_verified)

                result = {
                    "verified": is_verified,
                    "confidence": confidence,
                    "speaker_name": speaker_name,
                    "speaker_id": profile["speaker_id"],
                    "is_owner": profile["is_primary_user"],
                    "security_level": profile["security_level"],
                    "adaptive_threshold": adaptive_threshold,
                }

                # If confidence is very low, suggest re-enrollment
                if confidence < 0.10:
                    result["suggestion"] = "Voice profile may need re-enrollment"
                    logger.warning(f"⚠️ Very low confidence ({confidence:.2%}) for {speaker_name} - consider re-enrollment")

                return result

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

    async def _get_adaptive_threshold(self, speaker_name: str, profile: dict) -> float:
        """
        Calculate adaptive threshold based on verification history

        Args:
            speaker_name: Name of speaker
            profile: Speaker profile dict

        Returns:
            Adaptive threshold value
        """
        base_threshold = profile.get("threshold", self.verification_threshold)

        # If no history, use base threshold
        if speaker_name not in self.verification_history:
            return base_threshold

        history = self.verification_history[speaker_name]
        attempts = history.get("attempts", [])

        # Need minimum samples for adaptation
        if len(attempts) < self.min_samples_for_update:
            return base_threshold

        # Calculate average confidence from recent successful attempts
        recent_attempts = attempts[-10:]  # Last 10 attempts
        successful_confidences = [a["confidence"] for a in recent_attempts if a.get("verified", False)]

        if not successful_confidences:
            # No successful attempts recently - lower threshold to allow verification
            avg_confidence = sum(a["confidence"] for a in recent_attempts) / len(recent_attempts)
            if avg_confidence > 0.20:
                # There's some similarity, lower threshold
                adaptive_threshold = max(0.25, avg_confidence * 0.8)
                logger.info(f"📊 Adaptive threshold for {speaker_name}: {adaptive_threshold:.2%} (lowered from {base_threshold:.2%})")
                return adaptive_threshold
            return base_threshold

        # Calculate statistics
        avg_confidence = sum(successful_confidences) / len(successful_confidences)
        min_confidence = min(successful_confidences)

        # Set threshold slightly below minimum successful confidence
        # This allows for natural variation in voice
        adaptive_threshold = max(0.25, min(base_threshold, min_confidence * 0.90))

        logger.info(
            f"📊 Adaptive threshold for {speaker_name}: {adaptive_threshold:.2%} "
            f"(base: {base_threshold:.2%}, avg: {avg_confidence:.2%}, min: {min_confidence:.2%})"
        )

        return adaptive_threshold

    async def _record_verification_attempt(self, speaker_name: str, confidence: float, verified: bool):
        """
        Record verification attempt for adaptive learning

        Args:
            speaker_name: Name of speaker
            confidence: Confidence score
            verified: Whether verification succeeded
        """
        if not self.learning_enabled:
            return

        # Initialize history for this speaker
        if speaker_name not in self.verification_history:
            self.verification_history[speaker_name] = {
                "attempts": [],
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
            }

        history = self.verification_history[speaker_name]

        # Record attempt
        attempt = {
            "timestamp": datetime.now().isoformat(),
            "confidence": confidence,
            "verified": verified,
        }

        history["attempts"].append(attempt)
        history["total_attempts"] += 1

        if verified:
            history["successful_attempts"] += 1
        else:
            history["failed_attempts"] += 1

        # Keep only recent attempts (last 50)
        if len(history["attempts"]) > 50:
            history["attempts"] = history["attempts"][-50:]

        # Log learning progress
        if history["total_attempts"] % 5 == 0:
            success_rate = history["successful_attempts"] / history["total_attempts"] * 100
            logger.info(
                f"📚 Learning progress for {speaker_name}: "
                f"{history['total_attempts']} attempts, "
                f"{success_rate:.1f}% success rate"
            )

    async def cleanup(self):
        """Cleanup resources and terminate background threads"""
        logger.info("🧹 Cleaning up Speaker Verification Service...")

        # Signal shutdown to background threads
        self._shutdown_event.set()

        # Wait for preload thread to complete (with timeout)
        if self._preload_thread and self._preload_thread.is_alive():
            logger.debug("   Waiting for background preload thread to finish...")
            self._preload_thread.join(timeout=2.0)

            if self._preload_thread.is_alive():
                logger.warning("   Preload thread did not exit cleanly (daemon will be terminated)")
            else:
                logger.debug("   ✅ Preload thread terminated cleanly")

        # Clean up event loop if still running
        if self._preload_loop and not self._preload_loop.is_closed():
            try:
                logger.debug("   Closing background event loop...")
                self._preload_loop.stop()
                self._preload_loop.close()
                self._preload_loop = None
            except Exception as e:
                logger.debug(f"   Event loop cleanup error: {e}")

        # Clean up learning database (closes background tasks and threads)
        if self.learning_db:
            try:
                logger.debug("   Closing learning database...")
                from intelligence.learning_database import close_learning_database
                await close_learning_database()
                logger.debug("   ✅ Learning database closed")
            except Exception as e:
                logger.warning(f"   ⚠ Learning database cleanup error: {e}")

        # Clean up SpeechBrain engine
        if self.speechbrain_engine:
            try:
                await self.speechbrain_engine.cleanup()
                logger.debug("   ✅ SpeechBrain engine cleaned up")
            except Exception as e:
                logger.warning(f"   ⚠ SpeechBrain cleanup error: {e}")

        # Clear caches
        self.speaker_profiles.clear()
        self.profile_quality_scores.clear()

        # Reset state
        self.initialized = False
        self._encoder_preloaded = False
        self._encoder_preloading = False
        self._preload_thread = None
        self.learning_db = None

        logger.info("✅ Speaker Verification Service cleaned up")


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
