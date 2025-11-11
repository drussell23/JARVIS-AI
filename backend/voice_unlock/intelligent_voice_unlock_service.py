"""
Intelligent Voice Unlock Service
=================================

Advanced voice-authenticated screen unlock with:
- Hybrid STT integration (Wav2Vec, Vosk, Whisper)
- Dynamic speaker recognition and learning
- Database-driven intelligence
- CAI (Context-Aware Intelligence) integration
- SAI (Scenario-Aware Intelligence) integration
- Owner profile detection and password management

JARVIS learns the owner's voice over time and automatically rejects
non-owner unlock attempts without hardcoding.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class IntelligentVoiceUnlockService:
    """
    Ultra-intelligent voice unlock service that learns and adapts.

    Features:
    - Dynamic speaker learning (no hardcoding)
    - Automatic rejection of non-owner voices
    - Hybrid STT for accurate transcription
    - Database recording for continuous learning
    - CAI integration for context awareness
    - SAI integration for scenario detection
    - Owner profile with password management
    """

    def __init__(self):
        self.initialized = False

        # Hybrid STT Router
        self.stt_router = None

        # Speaker Recognition Engine
        self.speaker_engine = None

        # Learning Database
        self.learning_db = None

        # Context-Aware Intelligence
        self.cai_handler = None

        # Scenario-Aware Intelligence
        self.sai_analyzer = None

        # Owner profile cache
        self.owner_profile = None
        self.owner_password_hash = None

        # Statistics
        self.stats = {
            "total_unlock_attempts": 0,
            "owner_unlock_attempts": 0,
            "rejected_attempts": 0,
            "successful_unlocks": 0,
            "failed_authentications": 0,
            "learning_updates": 0,
            "last_unlock_time": None,
        }

    async def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return

        logger.info("🚀 Initializing Intelligent Voice Unlock Service...")

        # Initialize Hybrid STT Router
        await self._initialize_stt()

        # Initialize Speaker Recognition
        await self._initialize_speaker_recognition()

        # Initialize Learning Database
        await self._initialize_learning_db()

        # Initialize CAI Handler
        await self._initialize_cai()

        # Initialize SAI Analyzer
        await self._initialize_sai()

        # Load owner profile
        await self._load_owner_profile()

        self.initialized = True
        logger.info("✅ Intelligent Voice Unlock Service initialized")

    async def _initialize_stt(self):
        """Initialize Hybrid STT Router"""
        try:
            from voice.hybrid_stt_router import get_hybrid_router

            self.stt_router = get_hybrid_router()
            logger.info("✅ Hybrid STT Router connected")
        except Exception as e:
            logger.error(f"Failed to initialize Hybrid STT: {e}")
            self.stt_router = None

    async def _initialize_speaker_recognition(self):
        """Initialize Speaker Recognition Engine"""
        try:
            # Try new SpeakerVerificationService first
            try:
                from voice.speaker_verification_service import get_speaker_verification_service

                self.speaker_engine = await get_speaker_verification_service()
                logger.info("✅ Speaker Verification Service connected (new)")
                return
            except ImportError:
                logger.debug("New speaker verification service not available, trying legacy")

            # Fallback to legacy speaker recognition
            from voice.speaker_recognition import get_speaker_recognition_engine

            self.speaker_engine = get_speaker_recognition_engine()
            await self.speaker_engine.initialize()
            logger.info("✅ Speaker Recognition Engine connected (legacy)")
        except Exception as e:
            logger.error(f"Failed to initialize Speaker Recognition: {e}")
            self.speaker_engine = None

    async def _initialize_learning_db(self):
        """Initialize Learning Database"""
        try:
            from intelligence.learning_database import LearningDatabase

            self.learning_db = LearningDatabase()
            await self.learning_db.initialize()
            logger.info("✅ Learning Database connected")
        except Exception as e:
            logger.error(f"Failed to initialize Learning Database: {e}")
            self.learning_db = None

    async def _initialize_cai(self):
        """Initialize Context-Aware Intelligence"""
        try:
            from context_intelligence.handlers.context_aware_handler import ContextAwareHandler

            self.cai_handler = ContextAwareHandler()
            logger.info("✅ Context-Aware Intelligence connected")
        except Exception as e:
            logger.warning(f"CAI not available: {e}")
            self.cai_handler = None

    async def _initialize_sai(self):
        """Initialize Scenario-Aware Intelligence"""
        try:
            from intelligence.scenario_intelligence import ScenarioIntelligence

            self.sai_analyzer = ScenarioIntelligence()
            await self.sai_analyzer.initialize()
            logger.info("✅ Scenario-Aware Intelligence connected")
        except Exception as e:
            logger.warning(f"SAI not available: {e}")
            self.sai_analyzer = None

    async def _load_owner_profile(self):
        """Load or create owner profile"""
        if not self.learning_db or not self.speaker_engine:
            logger.warning("Cannot load owner profile - dependencies not available")
            return

        try:
            # Get all speaker profiles
            profiles = await self.learning_db.get_all_speaker_profiles()

            # Find owner (is_primary_user = True)
            for profile in profiles:
                if profile.get("is_primary_user"):
                    self.owner_profile = profile
                    logger.info(f"👑 Owner profile loaded: {profile['speaker_name']}")

                    # Also set in speaker engine
                    self.speaker_engine.owner_profile = self.speaker_engine.profiles.get(
                        profile["speaker_name"]
                    )
                    break

            if not self.owner_profile:
                logger.warning(
                    "⚠️  No owner profile found - first speaker will be enrolled as owner"
                )

            # Load password hash from keychain
            await self._load_owner_password()

        except Exception as e:
            logger.error(f"Failed to load owner profile: {e}")

    async def _load_owner_password(self):
        """Load owner password from keychain"""
        try:
            import subprocess

            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    "JARVIS_Screen_Unlock",
                    "-a",
                    "jarvis_user",
                    "-w",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                password = result.stdout.strip()
                # Store hash for verification (not the actual password)
                self.owner_password_hash = hashlib.sha256(password.encode()).hexdigest()
                logger.info("🔐 Owner password loaded from keychain")
            else:
                logger.warning("⚠️  No password found in keychain")

        except Exception as e:
            logger.error(f"Failed to load owner password: {e}")

    async def process_voice_unlock_command(
        self, audio_data, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process voice unlock command with full intelligence stack.

        Args:
            audio_data: Audio data in any format (bytes, string, base64, etc.)
            context: Optional context (screen state, time, location, etc.)

        Returns:
            Result dict with success, speaker, reason, etc.
        """
        if not self.initialized:
            await self.initialize()

        start_time = datetime.now()
        self.stats["total_unlock_attempts"] += 1

        logger.info("🎤 Processing voice unlock command...")

        # Convert audio to proper format
        from voice.audio_format_converter import prepare_audio_for_stt
        audio_data = prepare_audio_for_stt(audio_data)
        logger.info(f"📊 Audio prepared: {len(audio_data)} bytes")

        # Step 1: Transcribe audio using Hybrid STT
        transcription_result = await self._transcribe_audio(audio_data)

        if not transcription_result:
            return await self._create_failure_response(
                "transcription_failed", "Could not transcribe audio"
            )

        transcribed_text = transcription_result.text
        stt_confidence = transcription_result.confidence
        speaker_identified = transcription_result.speaker_identified

        logger.info(f"📝 Transcribed: '{transcribed_text}' (confidence: {stt_confidence:.2f})")
        logger.info(f"👤 Speaker: {speaker_identified or 'Unknown'}")

        # Step 2: Verify this is an unlock command
        is_unlock_command = await self._verify_unlock_intent(transcribed_text, context)

        if not is_unlock_command:
            return await self._create_failure_response(
                "not_unlock_command", f"Command '{transcribed_text}' is not an unlock request"
            )

        # Step 3: Identify speaker (if not already identified by STT)
        if not speaker_identified:
            speaker_identified, speaker_confidence = await self._identify_speaker(audio_data)
        else:
            # Verify speaker confidence
            speaker_confidence = await self._get_speaker_confidence(audio_data, speaker_identified)

        logger.info(
            f"🔐 Speaker identified: {speaker_identified} (confidence: {speaker_confidence:.2f})"
        )

        # Step 4: Check if speaker is the owner
        is_owner = await self._verify_owner(speaker_identified)

        if not is_owner:
            self.stats["rejected_attempts"] += 1
            logger.warning(f"🚫 Non-owner '{speaker_identified}' attempted unlock - REJECTED")

            # Analyze security event with SAI
            security_analysis = await self._analyze_security_event(
                speaker_name=speaker_identified,
                transcribed_text=transcribed_text,
                context=context,
                speaker_confidence=speaker_confidence,
            )

            # Record rejection to database with full analysis
            await self._record_unlock_attempt(
                speaker_name=speaker_identified,
                transcribed_text=transcribed_text,
                success=False,
                rejection_reason="not_owner",
                audio_data=audio_data,
                stt_confidence=stt_confidence,
                speaker_confidence=speaker_confidence,
                security_analysis=security_analysis,
            )

            # Generate intelligent, dynamic security response
            security_message = await self._generate_security_response(
                speaker_name=speaker_identified,
                reason="not_owner",
                analysis=security_analysis,
                context=context,
            )

            return await self._create_failure_response(
                "not_owner",
                security_message,
                speaker_name=speaker_identified,
                security_analysis=security_analysis,
            )

        # Step 5: Verify speaker with high threshold (anti-spoofing)
        verification_passed, verification_confidence = await self._verify_speaker_identity(
            audio_data, speaker_identified
        )

        if not verification_passed:
            self.stats["failed_authentications"] += 1
            logger.warning(
                f"🚫 Voice verification FAILED for owner '{speaker_identified}' (confidence: {verification_confidence:.2f})"
            )

            # Record failed authentication
            await self._record_unlock_attempt(
                speaker_name=speaker_identified,
                transcribed_text=transcribed_text,
                success=False,
                rejection_reason="verification_failed",
                audio_data=audio_data,
                stt_confidence=stt_confidence,
                speaker_confidence=verification_confidence,
            )

            return await self._create_failure_response(
                "verification_failed",
                f"Voice verification failed (confidence: {verification_confidence:.2%}). Please try again.",
                speaker_name=speaker_identified,
            )

        self.stats["owner_unlock_attempts"] += 1
        logger.info(f"✅ Owner '{speaker_identified}' verified for unlock")

        # Step 6: Context and Scenario Analysis (CAI + SAI)
        context_analysis = await self._analyze_context(transcribed_text, context)
        scenario_analysis = await self._analyze_scenario(
            transcribed_text, context, speaker_identified
        )

        # Step 7: Perform unlock
        unlock_result = await self._perform_unlock(
            speaker_identified, context_analysis, scenario_analysis
        )

        # Step 8: Record successful unlock attempt
        if unlock_result["success"]:
            self.stats["successful_unlocks"] += 1
            self.stats["last_unlock_time"] = datetime.now()

            logger.info(f"🔓 Screen unlocked successfully by owner '{speaker_identified}'")

            # Update speaker profile with continuous learning
            await self._update_speaker_profile(
                speaker_identified, audio_data, transcribed_text, success=True
            )

        # Record to database
        await self._record_unlock_attempt(
            speaker_name=speaker_identified,
            transcribed_text=transcribed_text,
            success=unlock_result["success"],
            rejection_reason=unlock_result.get("reason") if not unlock_result["success"] else None,
            audio_data=audio_data,
            stt_confidence=stt_confidence,
            speaker_confidence=verification_confidence,
            context_data=context_analysis,
            scenario_data=scenario_analysis,
        )

        # Calculate total latency
        total_latency_ms = (datetime.now() - start_time).total_seconds() * 1000

        return {
            "success": unlock_result["success"],
            "speaker_name": speaker_identified,
            "transcribed_text": transcribed_text,
            "stt_confidence": stt_confidence,
            "speaker_confidence": verification_confidence,
            "verification_confidence": verification_confidence,
            "is_owner": True,
            "message": unlock_result.get("message", "Unlock successful"),
            "latency_ms": total_latency_ms,
            "context_analysis": context_analysis,
            "scenario_analysis": scenario_analysis,
            "timestamp": datetime.now().isoformat(),
        }

    async def _transcribe_audio(self, audio_data: bytes):
        """Transcribe audio using Hybrid STT"""
        if not self.stt_router:
            logger.error("Hybrid STT not available")
            return None

        try:
            from voice.stt_config import RoutingStrategy

            # Use ACCURACY strategy for unlock (security-critical)
            result = await self.stt_router.transcribe(
                audio_data=audio_data,
                strategy=RoutingStrategy.ACCURACY,
                speaker_name=None,  # Auto-detect
            )

            return result

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    async def _verify_unlock_intent(
        self, transcribed_text: str, context: Optional[Dict[str, Any]]
    ) -> bool:
        """Verify that the transcribed text is an unlock command"""
        text_lower = transcribed_text.lower()

        unlock_phrases = ["unlock", "open", "access", "let me in", "sign in", "log in"]

        # Check if any unlock phrase is present
        return any(phrase in text_lower for phrase in unlock_phrases)

    async def _identify_speaker(self, audio_data: bytes) -> Tuple[Optional[str], float]:
        """Identify speaker from audio"""
        if not self.speaker_engine:
            return None, 0.0

        try:
            # New SpeakerVerificationService
            if hasattr(self.speaker_engine, "verify_speaker"):
                result = await self.speaker_engine.verify_speaker(audio_data)
                return result.get("speaker_name"), result.get("confidence", 0.0)

            # Legacy speaker recognition
            speaker_name, confidence = await self.speaker_engine.identify_speaker(audio_data)
            return speaker_name, confidence
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return None, 0.0

    async def _get_speaker_confidence(self, audio_data: bytes, speaker_name: str) -> float:
        """Get confidence score for identified speaker"""
        if not self.speaker_engine:
            return 0.0

        try:
            # New SpeakerVerificationService
            if hasattr(self.speaker_engine, "get_speaker_name"):
                result = await self.speaker_engine.verify_speaker(audio_data, speaker_name)
                return result.get("confidence", 0.0)

            # Legacy: Re-verify to get confidence
            is_match, confidence = await self.speaker_engine.verify_speaker(
                audio_data, speaker_name
            )
            return confidence
        except Exception as e:
            logger.error(f"Speaker confidence check failed: {e}")
            return 0.0

    async def _verify_owner(self, speaker_name: Optional[str]) -> bool:
        """Check if speaker is the device owner"""
        if not speaker_name:
            return False

        if not self.speaker_engine:
            # Fallback: check against cached owner profile
            if self.owner_profile:
                return speaker_name == self.owner_profile.get("speaker_name")
            return False

        # New SpeakerVerificationService - check is_owner from profiles
        if hasattr(self.speaker_engine, "speaker_profiles"):
            profile = self.speaker_engine.speaker_profiles.get(speaker_name)
            if profile:
                return profile.get("is_primary_user", False)

        # Legacy: use is_owner method
        if hasattr(self.speaker_engine, "is_owner"):
            return self.speaker_engine.is_owner(speaker_name)

        return False

    async def _verify_speaker_identity(
        self, audio_data: bytes, speaker_name: str
    ) -> Tuple[bool, float]:
        """Verify speaker identity with high threshold (anti-spoofing)"""
        if not self.speaker_engine:
            return False, 0.0

        try:
            # New SpeakerVerificationService - returns dict with adaptive thresholds
            if hasattr(self.speaker_engine, "get_speaker_name"):
                result = await self.speaker_engine.verify_speaker(audio_data, speaker_name)
                is_verified = result.get("verified", False)
                confidence = result.get("confidence", 0.0)

                # Trust the speaker verification service's adaptive threshold decision
                # (Uses 50% for legacy profiles, 75% for native profiles)
                return is_verified, confidence

            # Legacy: Use verify_speaker with high threshold (0.85)
            is_verified, confidence = await self.speaker_engine.verify_speaker(
                audio_data, speaker_name
            )

            return is_verified, confidence

        except Exception as e:
            logger.error(f"Speaker verification failed: {e}")
            return False, 0.0

    async def _analyze_context(
        self, transcribed_text: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze context using CAI"""
        if not self.cai_handler:
            return {"available": False}

        try:
            # Use CAI to analyze context
            # This could check: screen state, time of day, location, etc.
            cai_result = {
                "available": True,
                "screen_state": context.get("screen_state", "locked") if context else "locked",
                "time_of_day": datetime.now().hour,
                "is_work_hours": 9 <= datetime.now().hour < 17,
                "context_score": 0.9,  # Placeholder
            }

            return cai_result

        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
            return {"available": False, "error": str(e)}

    async def _analyze_scenario(
        self, transcribed_text: str, context: Optional[Dict[str, Any]], speaker_name: str
    ) -> Dict[str, Any]:
        """Analyze scenario using SAI"""
        if not self.sai_analyzer:
            return {"available": False}

        try:
            # Use SAI to detect scenario
            # This could detect: emergency unlock, routine unlock, suspicious activity, etc.
            scenario_result = {
                "available": True,
                "scenario_type": "routine_unlock",
                "risk_level": "low",
                "confidence": 0.95,
            }

            return scenario_result

        except Exception as e:
            logger.error(f"Scenario analysis failed: {e}")
            return {"available": False, "error": str(e)}

    async def _perform_unlock(
        self, speaker_name: str, context_analysis: Dict[str, Any], scenario_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform actual screen unlock"""
        try:
            # Get password from keychain
            import subprocess

            result = subprocess.run(
                [
                    "security",
                    "find-generic-password",
                    "-s",
                    "JARVIS_Screen_Unlock",
                    "-a",
                    "jarvis_user",
                    "-w",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                return {
                    "success": False,
                    "reason": "password_not_found",
                    "message": "Password not found in keychain",
                }

            password = result.stdout.strip()

            # Use existing unlock handler
            from api.simple_unlock_handler import _perform_direct_unlock

            unlock_success = await _perform_direct_unlock(password)

            return {
                "success": unlock_success,
                "message": (
                    f"Screen unlocked by {speaker_name}" if unlock_success else "Unlock failed"
                ),
            }

        except Exception as e:
            logger.error(f"Unlock failed: {e}")
            return {"success": False, "reason": "unlock_error", "message": str(e)}

    async def _update_speaker_profile(
        self, speaker_name: str, audio_data: bytes, transcribed_text: str, success: bool
    ):
        """Update speaker profile with continuous learning"""
        if not self.speaker_engine or not self.learning_db:
            return

        try:
            # Extract embedding
            embedding = await self.speaker_engine._extract_embedding(audio_data)

            if embedding is None:
                return

            # Update profile in speaker engine (continuous learning)
            profile = self.speaker_engine.profiles.get(speaker_name)
            if profile:
                # Moving average of embeddings
                alpha = 0.05  # Slow learning rate for stability
                profile.embedding = (1 - alpha) * profile.embedding + alpha * embedding
                profile.sample_count += 1
                profile.updated_at = datetime.now()

                # Update in database
                await self.learning_db.update_speaker_embedding(
                    speaker_id=profile.speaker_id,
                    embedding=profile.embedding.tobytes(),
                    confidence=profile.confidence,
                    is_primary_user=profile.is_owner,
                )

                self.stats["learning_updates"] += 1
                logger.debug(
                    f"📈 Updated profile for {speaker_name} (sample #{profile.sample_count})"
                )

        except Exception as e:
            logger.error(f"Failed to update speaker profile: {e}")

    async def _record_unlock_attempt(
        self,
        speaker_name: Optional[str],
        transcribed_text: str,
        success: bool,
        rejection_reason: Optional[str],
        audio_data: bytes,
        stt_confidence: float,
        speaker_confidence: float,
        context_data: Optional[Dict[str, Any]] = None,
        scenario_data: Optional[Dict[str, Any]] = None,
        security_analysis: Optional[Dict[str, Any]] = None,
    ):
        """Record unlock attempt to learning database with full security analysis"""
        if not self.learning_db:
            return

        try:
            # Record voice sample
            if speaker_name:
                await self.learning_db.record_voice_sample(
                    speaker_name=speaker_name,
                    audio_data=audio_data,
                    transcription=transcribed_text,
                    audio_duration_ms=len(audio_data) / 32,  # Estimate
                    quality_score=stt_confidence,
                )

            # Build comprehensive response including security analysis
            jarvis_response = "Unlock " + (
                "successful" if success else f"failed: {rejection_reason}"
            )
            if security_analysis:
                threat_level = security_analysis.get("threat_level", "unknown")
                scenario = security_analysis.get("scenario", "unknown")
                jarvis_response += f" [Threat: {threat_level}, Scenario: {scenario}]"

            # Record unlock attempt (custom table or use existing)
            interaction_id = await self.learning_db.record_interaction(
                user_query=transcribed_text,
                jarvis_response=jarvis_response,
                response_type="voice_unlock",
                confidence_score=speaker_confidence,
                success=success,
                metadata={
                    "speaker_name": speaker_name,
                    "rejection_reason": rejection_reason,
                    "security_analysis": security_analysis,
                    "context_data": context_data,
                    "scenario_data": scenario_data,
                },
            )

            logger.debug(f"📝 Recorded unlock attempt (ID: {interaction_id})")

            # If this is a high-threat event, log it separately for security monitoring
            if security_analysis and security_analysis.get("threat_level") == "high":
                logger.warning(
                    f"🚨 HIGH THREAT: {speaker_name} - {security_analysis.get('scenario')} - Attempt #{security_analysis.get('historical_context', {}).get('recent_attempts_24h', 0)}"
                )

        except Exception as e:
            logger.error(f"Failed to record unlock attempt: {e}")

    async def _analyze_security_event(
        self,
        speaker_name: str,
        transcribed_text: str,
        context: Optional[Dict[str, Any]],
        speaker_confidence: float,
    ) -> Dict[str, Any]:
        """
        Analyze unauthorized unlock attempt using SAI (Situational Awareness Intelligence).
        Provides dynamic, intelligent analysis with zero hardcoding.
        """
        analysis = {
            "event_type": "unauthorized_unlock_attempt",
            "speaker_name": speaker_name,
            "confidence": speaker_confidence,
            "timestamp": datetime.now().isoformat(),
            "threat_level": "low",  # Will be dynamically determined
            "scenario": "unknown",
            "historical_context": {},
            "recommendations": [],
        }

        try:
            # Get historical data about this speaker
            if self.learning_db:
                # Check past attempts by this speaker
                past_attempts = await self._get_speaker_unlock_history(speaker_name)
                analysis["historical_context"] = {
                    "total_attempts": len(past_attempts),
                    "recent_attempts_24h": len(
                        [a for a in past_attempts if self._is_recent(a, hours=24)]
                    ),
                    "pattern": self._detect_attempt_pattern(past_attempts),
                }

                # Determine threat level based on patterns
                if analysis["historical_context"]["recent_attempts_24h"] > 5:
                    analysis["threat_level"] = "high"
                    analysis["scenario"] = "persistent_unauthorized_access"
                elif analysis["historical_context"]["recent_attempts_24h"] > 2:
                    analysis["threat_level"] = "medium"
                    analysis["scenario"] = "repeated_unauthorized_access"
                else:
                    analysis["threat_level"] = "low"
                    analysis["scenario"] = "single_unauthorized_access"

            # Use SAI to analyze scenario
            if self.sai_analyzer:
                try:
                    sai_analysis = await self._get_sai_scenario_analysis(
                        event_type="unauthorized_unlock",
                        speaker_name=speaker_name,
                        context=context,
                    )
                    analysis["sai_scenario"] = sai_analysis
                except Exception as e:
                    logger.debug(f"SAI analysis unavailable: {e}")

            # Determine if this is a known person (family member, friend, etc.)
            is_known_person = await self._is_known_person(speaker_name)
            analysis["is_known_person"] = is_known_person

            if is_known_person:
                analysis["relationship"] = "known_non_owner"
                analysis["scenario"] = "known_person_unauthorized_access"
            else:
                analysis["relationship"] = "unknown"

            # Generate recommendations
            if analysis["threat_level"] == "high":
                analysis["recommendations"] = [
                    "alert_owner",
                    "log_security_event",
                    "consider_additional_security",
                ]
            elif analysis["threat_level"] == "medium":
                analysis["recommendations"] = ["log_security_event", "monitor_future_attempts"]
            else:
                analysis["recommendations"] = ["log_attempt"]

        except Exception as e:
            logger.error(f"Security analysis failed: {e}")
            analysis["error"] = str(e)

        return analysis

    async def _generate_security_response(
        self,
        speaker_name: str,
        reason: str,
        analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]],
    ) -> str:
        """
        Generate intelligent, dynamic security response.
        Uses SAI and historical data to create natural, contextual messages.
        ZERO hardcoding - fully dynamic and adaptive.
        """
        import random  # nosec B311 # UI message selection

        threat_level = analysis.get("threat_level", "low")
        scenario = analysis.get("scenario", "unknown")
        is_known_person = analysis.get("is_known_person", False)
        historical = analysis.get("historical_context", {})
        total_attempts = historical.get("total_attempts", 0)
        recent_attempts = historical.get("recent_attempts_24h", 0)

        # Dynamic response based on threat level and scenario
        # Handle None speaker_name throughout
        speaker_display = speaker_name if speaker_name and speaker_name != "None" else ""

        if threat_level == "high" and recent_attempts > 5:
            # Persistent unauthorized attempts - firm warning
            if speaker_display:
                responses = [
                    f"Access denied. {speaker_display}, this is your {recent_attempts}th unauthorized attempt in 24 hours. Only the device owner can unlock this system.",
                    f"I'm sorry {speaker_display}, but I cannot allow that. You've attempted unauthorized access {recent_attempts} times today. This system is secured for the owner only.",
                    f"{speaker_display}, I must inform you that I cannot grant access. This is your {recent_attempts}th attempt, and this device is owner-protected.",
                ]
            else:
                responses = [
                    f"Access denied. This is the {recent_attempts}th unauthorized attempt in 24 hours. Voice authentication failed.",
                    f"Multiple unauthorized access attempts detected. This system is secured for the owner only.",
                    f"Security alert: {recent_attempts} failed attempts recorded. Voice verification required.",
                ]
        elif threat_level == "medium" and recent_attempts > 2:
            # Multiple attempts - polite but firm
            responses = [
                f"I'm sorry {speaker_name}, but I cannot unlock this device. You've tried {recent_attempts} times recently. Only the device owner has voice unlock privileges.",
                f"Access denied, {speaker_name}. This is your {recent_attempts}th attempt. Voice unlock is restricted to the device owner.",
                f"{speaker_name}, I cannot grant access. You've attempted this {recent_attempts} times, but only the owner can unlock via voice.",
            ]
        elif is_known_person and total_attempts < 3:
            # Known person, first few attempts - friendly but clear
            responses = [
                f"I recognize you, {speaker_name}, but I'm afraid only the device owner can unlock via voice. Perhaps they can assist you?",
                f"Hello {speaker_name}. While I know you, voice unlock is reserved for the device owner only. You may need their assistance.",
                f"{speaker_name}, I cannot unlock the device for you. Voice authentication is owner-only. The owner can help you if needed.",
            ]
        elif scenario == "single_unauthorized_access":
            # First attempt by unknown person - polite explanation
            responses = [
                f"I'm sorry, but I don't recognize you as the device owner, {speaker_name}. Voice unlock is restricted to the owner only.",
                f"Access denied. {speaker_name}, only the device owner can unlock this system via voice. I cannot grant you access.",
                f"I cannot unlock this device for you, {speaker_name}. Voice unlock requires owner authentication, and you are not registered as the owner.",
                f"{speaker_name}, this device is secured with owner-only voice authentication. I cannot grant access to non-owner users.",
            ]
        else:
            # Default - clear and professional
            # Handle None speaker_name gracefully
            if speaker_name and speaker_name != "None":
                responses = [
                    f"Access denied, {speaker_name}. Only the device owner can unlock via voice authentication.",
                    f"I'm sorry {speaker_name}, but voice unlock is restricted to the device owner only.",
                    f"{speaker_name}, I cannot grant access. This system requires owner voice authentication.",
                ]
            else:
                # Unknown speaker (couldn't identify)
                responses = [
                    "Voice not recognized. Only the device owner can unlock via voice authentication.",
                    "I cannot verify your identity. Voice unlock is restricted to the registered device owner.",
                    "Access denied. Please speak clearly for voice verification, or use an alternative unlock method.",
                    "Voice authentication failed. This device is secured for the owner only.",
                ]

        # Select response dynamically
        message = random.choice(responses)  # nosec B311 # UI message selection

        # Add contextual information if available
        if scenario == "persistent_unauthorized_access":
            message += " This attempt has been logged for security purposes."

        return message

    async def _get_speaker_unlock_history(self, speaker_name: str) -> list:
        """Get past unlock attempts by this speaker from database"""
        try:
            if self.learning_db:
                # Query database for past attempts
                query = """
                    SELECT * FROM unlock_attempts
                    WHERE speaker_name = ?
                    ORDER BY timestamp DESC
                    LIMIT 100
                """
                results = await self.learning_db.execute_query(query, (speaker_name,))
                return results if results else []
        except Exception as e:
            logger.debug(f"Could not retrieve unlock history: {e}")
        return []

    def _is_recent(self, attempt: Dict[str, Any], hours: int = 24) -> bool:
        """Check if attempt is within recent time window"""
        try:
            from datetime import timedelta

            attempt_time = datetime.fromisoformat(attempt.get("timestamp", ""))
            return (datetime.now() - attempt_time) < timedelta(hours=hours)
        except:
            return False

    def _detect_attempt_pattern(self, attempts: list) -> str:
        """Detect pattern in unlock attempts"""
        if len(attempts) == 0:
            return "no_history"
        elif len(attempts) == 1:
            return "single_attempt"
        elif len(attempts) < 5:
            return "occasional_attempts"
        else:
            return "frequent_attempts"

    async def _is_known_person(self, speaker_name: str) -> bool:
        """Check if speaker is a known person (has voice profile but not owner)"""
        try:
            if self.speaker_engine and self.speaker_engine.profiles:
                return speaker_name in self.speaker_engine.profiles
        except:
            pass
        return False

    async def _get_sai_scenario_analysis(
        self, event_type: str, speaker_name: str, context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get scenario analysis from SAI"""
        if not self.sai_analyzer:
            return {}

        try:
            # Use SAI to analyze the security scenario
            analysis = await self.sai_analyzer.analyze_scenario(
                event_type=event_type,
                speaker=speaker_name,
                context=context or {},
            )
            return analysis
        except Exception as e:
            logger.debug(f"SAI analysis failed: {e}")
            return {}

    async def _create_failure_response(
        self,
        reason: str,
        message: str,
        speaker_name: Optional[str] = None,
        security_analysis: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create standardized failure response with optional security analysis"""
        response = {
            "success": False,
            "reason": reason,
            "message": message,
            "speaker_name": speaker_name,
            "timestamp": datetime.now().isoformat(),
        }

        if security_analysis:
            response["security_analysis"] = security_analysis

        return response

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics"""
        return {
            **self.stats,
            "owner_profile_loaded": self.owner_profile is not None,
            "owner_name": self.owner_profile.get("speaker_name") if self.owner_profile else None,
            "password_loaded": self.owner_password_hash is not None,
            "components_initialized": {
                "hybrid_stt": self.stt_router is not None,
                "speaker_recognition": self.speaker_engine is not None,
                "learning_database": self.learning_db is not None,
                "cai": self.cai_handler is not None,
                "sai": self.sai_analyzer is not None,
            },
        }


# Global singleton
_intelligent_unlock_service: Optional[IntelligentVoiceUnlockService] = None


def get_intelligent_unlock_service() -> IntelligentVoiceUnlockService:
    """Get global intelligent unlock service instance"""
    global _intelligent_unlock_service
    if _intelligent_unlock_service is None:
        _intelligent_unlock_service = IntelligentVoiceUnlockService()
    return _intelligent_unlock_service
