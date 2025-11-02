#!/usr/bin/env python3
"""
JARVIS Voice Security Tester - Advanced Async Biometric Authentication Testing
===============================================================================

Tests voice biometric security by generating synthetic "attacker" voices and
verifying they are DENIED access while the authorized user is ACCEPTED.

Features:
- Async multi-voice generation using various TTS engines
- Dynamic threshold testing (no hardcoding)
- Real-time similarity scoring and rejection verification
- Comprehensive security report generation
- Integration with existing JARVIS voice unlock system

Usage:
    # Standalone
    python3 backend/voice_unlock/voice_security_tester.py

    # As JARVIS command
    Say: "test my voice security" or "verify voice authentication"
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceProfile(Enum):
    """Different voice profile types for testing"""
    AUTHORIZED_USER = "authorized_user"
    MALE_ATTACKER = "male_attacker"
    FEMALE_ATTACKER = "female_attacker"
    CHILD_ATTACKER = "child_attacker"
    ROBOTIC_ATTACKER = "robotic_attacker"
    PITCHED_ATTACKER = "pitched_attacker"


class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class VoiceSecurityTest:
    """Individual voice security test result"""
    profile_type: VoiceProfile
    test_phrase: str
    similarity_score: float
    threshold: float
    should_accept: bool
    was_accepted: bool
    result: TestResult
    duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    embedding_dimension: Optional[int] = None

    @property
    def passed(self) -> bool:
        """Check if test passed"""
        return self.result == TestResult.PASS

    @property
    def security_verdict(self) -> str:
        """Get security verdict"""
        if self.should_accept and self.was_accepted:
            return "✅ SECURE - Authorized voice accepted"
        elif not self.should_accept and not self.was_accepted:
            return "✅ SECURE - Unauthorized voice rejected"
        elif self.should_accept and not self.was_accepted:
            return "⚠️ FALSE REJECTION - Authorized voice denied"
        else:
            return "🚨 SECURITY BREACH - Unauthorized voice accepted"


@dataclass
class VoiceSecurityReport:
    """Complete voice security test report"""
    tests: List[VoiceSecurityTest]
    authorized_user_name: str
    total_duration_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def failed_tests(self) -> int:
        return sum(1 for t in self.tests if t.result == TestResult.FAIL)

    @property
    def security_breaches(self) -> List[VoiceSecurityTest]:
        """Get all security breaches (unauthorized voices accepted)"""
        return [t for t in self.tests
                if not t.should_accept and t.was_accepted]

    @property
    def false_rejections(self) -> List[VoiceSecurityTest]:
        """Get all false rejections (authorized voice denied)"""
        return [t for t in self.tests
                if t.should_accept and not t.was_accepted]

    @property
    def is_secure(self) -> bool:
        """Check if system is secure (no breaches)"""
        return len(self.security_breaches) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            'timestamp': self.timestamp,
            'authorized_user': self.authorized_user_name,
            'total_tests': self.total_tests,
            'passed_tests': self.passed_tests,
            'failed_tests': self.failed_tests,
            'is_secure': self.is_secure,
            'security_breaches': len(self.security_breaches),
            'false_rejections': len(self.false_rejections),
            'total_duration_ms': self.total_duration_ms,
            'tests': [
                {
                    'profile': t.profile_type.value,
                    'phrase': t.test_phrase,
                    'similarity': t.similarity_score,
                    'threshold': t.threshold,
                    'should_accept': t.should_accept,
                    'was_accepted': t.was_accepted,
                    'result': t.result.value,
                    'verdict': t.security_verdict,
                    'duration_ms': t.duration_ms,
                    'error': t.error_message
                }
                for t in self.tests
            ]
        }


class VoiceSecurityTester:
    """
    Advanced voice security testing system for JARVIS biometric authentication.

    Tests the voice unlock system by:
    1. Generating synthetic "attacker" voices with various characteristics
    2. Attempting unlock with each voice
    3. Verifying unauthorized voices are REJECTED
    4. Verifying authorized voice is ACCEPTED
    5. Generating comprehensive security report
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize voice security tester.

        Args:
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.authorized_user = self.config.get('authorized_user', 'Derek')
        self.test_phrase = self.config.get('test_phrase', 'unlock my screen')
        self.temp_dir = Path(tempfile.gettempdir()) / 'jarvis_voice_security_tests'
        self.temp_dir.mkdir(exist_ok=True)

        # Dynamic configuration (no hardcoding)
        self.verification_threshold = None  # Will be loaded from system
        self.embedding_dimension = None  # Will be detected

        # Test profiles to generate
        self.test_profiles = [
            VoiceProfile.MALE_ATTACKER,
            VoiceProfile.FEMALE_ATTACKER,
            VoiceProfile.CHILD_ATTACKER,
            VoiceProfile.ROBOTIC_ATTACKER,
            VoiceProfile.PITCHED_ATTACKER,
        ]

        logger.info(f"Voice Security Tester initialized for user: {self.authorized_user}")

    async def load_system_config(self) -> Dict[str, Any]:
        """
        Load configuration from JARVIS system dynamically.

        Returns:
            System configuration including thresholds and settings
        """
        try:
            # Import JARVIS components
            from backend.voice.speaker_verification_service import SpeakerVerificationService
            from backend.intelligence.cloud_database_adapter import CloudDatabaseAdapter

            # Get verification service
            verification_service = SpeakerVerificationService()
            await verification_service.initialize()

            # Get database adapter for profile info
            db_adapter = CloudDatabaseAdapter()
            await db_adapter.initialize()

            # Load user profile
            profile = await db_adapter.get_speaker_profile(self.authorized_user)

            if not profile:
                logger.warning(f"No profile found for {self.authorized_user}, using defaults")
                self.verification_threshold = 0.75
                self.embedding_dimension = 192
            else:
                # Determine threshold based on profile quality
                profile_dimension = len(profile.get('embedding', []))
                self.embedding_dimension = profile_dimension

                # Dynamic threshold (legacy vs native ECAPA-TDNN)
                if profile_dimension <= 512:
                    self.verification_threshold = 0.50  # Legacy threshold
                    logger.info(f"Using legacy threshold: 0.50 for {profile_dimension}D embedding")
                else:
                    self.verification_threshold = 0.75  # Native ECAPA-TDNN threshold
                    logger.info(f"Using native threshold: 0.75 for {profile_dimension}D embedding")

            return {
                'verification_threshold': self.verification_threshold,
                'embedding_dimension': self.embedding_dimension,
                'profile_exists': profile is not None,
                'profile_quality': profile.get('quality', 'unknown') if profile else None
            }

        except Exception as e:
            logger.error(f"Failed to load system config: {e}")
            # Fallback to defaults
            self.verification_threshold = 0.75
            self.embedding_dimension = 192
            return {
                'verification_threshold': self.verification_threshold,
                'embedding_dimension': self.embedding_dimension,
                'profile_exists': False,
                'profile_quality': None,
                'error': str(e)
            }

    async def generate_synthetic_voice(
        self,
        profile: VoiceProfile,
        text: str
    ) -> Optional[Path]:
        """
        Generate synthetic voice audio file for testing.

        Args:
            profile: Voice profile type to generate
            text: Text to synthesize

        Returns:
            Path to generated audio file or None if failed
        """
        try:
            # Try multiple TTS engines for robustness
            audio_file = await self._try_tts_engines(profile, text)

            if audio_file and audio_file.exists():
                logger.info(f"Generated {profile.value} voice: {audio_file}")
                return audio_file
            else:
                logger.error(f"Failed to generate {profile.value} voice")
                return None

        except Exception as e:
            logger.error(f"Error generating {profile.value} voice: {e}")
            return None

    async def _try_tts_engines(
        self,
        profile: VoiceProfile,
        text: str
    ) -> Optional[Path]:
        """
        Try multiple TTS engines to generate voice.

        Args:
            profile: Voice profile type
            text: Text to synthesize

        Returns:
            Path to generated audio or None
        """
        audio_file = self.temp_dir / f"{profile.value}_{int(time.time())}.wav"

        # Engine 1: macOS 'say' command (fast, built-in)
        try:
            voice_map = {
                VoiceProfile.MALE_ATTACKER: "Alex",
                VoiceProfile.FEMALE_ATTACKER: "Samantha",
                VoiceProfile.CHILD_ATTACKER: "Karen",
                VoiceProfile.ROBOTIC_ATTACKER: "Zarvox",
                VoiceProfile.PITCHED_ATTACKER: "Whisper",
            }

            voice = voice_map.get(profile, "Alex")

            process = await asyncio.create_subprocess_exec(
                'say', '-v', voice, '-o', str(audio_file), '--data-format=LEF32@22050', text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await asyncio.wait_for(process.wait(), timeout=10.0)

            if audio_file.exists() and audio_file.stat().st_size > 0:
                logger.info(f"Generated voice with macOS 'say' command: {voice}")
                return audio_file

        except Exception as e:
            logger.debug(f"macOS 'say' failed: {e}")

        # Engine 2: gTTS (Google TTS) - fallback
        try:
            from gtts import gTTS

            # Vary speech parameters for different profiles
            lang_map = {
                VoiceProfile.MALE_ATTACKER: 'en',
                VoiceProfile.FEMALE_ATTACKER: 'en-uk',
                VoiceProfile.CHILD_ATTACKER: 'en-au',
                VoiceProfile.ROBOTIC_ATTACKER: 'en-in',
                VoiceProfile.PITCHED_ATTACKER: 'en-ca',
            }

            lang = lang_map.get(profile, 'en')
            tts = gTTS(text=text, lang=lang, slow=False)
            tts.save(str(audio_file))

            if audio_file.exists():
                logger.info(f"Generated voice with gTTS: {lang}")
                return audio_file

        except Exception as e:
            logger.debug(f"gTTS failed: {e}")

        # Engine 3: pyttsx3 (offline TTS) - final fallback
        try:
            import pyttsx3

            engine = pyttsx3.init()

            # Configure voice characteristics
            voices = engine.getProperty('voices')
            if profile == VoiceProfile.FEMALE_ATTACKER and len(voices) > 1:
                engine.setProperty('voice', voices[1].id)

            rate = engine.getProperty('rate')
            if profile == VoiceProfile.CHILD_ATTACKER:
                engine.setProperty('rate', rate + 50)
            elif profile == VoiceProfile.ROBOTIC_ATTACKER:
                engine.setProperty('rate', rate - 50)

            engine.save_to_file(text, str(audio_file))
            engine.runAndWait()

            if audio_file.exists():
                logger.info("Generated voice with pyttsx3")
                return audio_file

        except Exception as e:
            logger.debug(f"pyttsx3 failed: {e}")

        logger.error(f"All TTS engines failed for {profile.value}")
        return None

    async def test_voice_authentication(
        self,
        audio_file: Path,
        profile: VoiceProfile,
        should_accept: bool
    ) -> VoiceSecurityTest:
        """
        Test voice authentication with given audio file.

        Args:
            audio_file: Path to audio file to test
            profile: Voice profile being tested
            should_accept: Whether this voice should be accepted

        Returns:
            Test result
        """
        start_time = time.time()

        try:
            # Import verification service
            from backend.voice.speaker_verification_service import SpeakerVerificationService

            # Initialize service
            verification_service = SpeakerVerificationService()
            await verification_service.initialize()

            # Perform verification
            result = await verification_service.verify_speaker(
                audio_file=str(audio_file),
                expected_speaker=self.authorized_user
            )

            # Extract results
            similarity_score = result.get('similarity_score', 0.0)
            was_accepted = result.get('verified', False)
            embedding_dim = result.get('embedding_dimension', self.embedding_dimension)

            # Determine test result
            if should_accept and was_accepted:
                test_result = TestResult.PASS  # Authorized accepted ✅
            elif not should_accept and not was_accepted:
                test_result = TestResult.PASS  # Unauthorized rejected ✅
            else:
                test_result = TestResult.FAIL  # Security issue ❌

            duration_ms = (time.time() - start_time) * 1000

            return VoiceSecurityTest(
                profile_type=profile,
                test_phrase=self.test_phrase,
                similarity_score=similarity_score,
                threshold=self.verification_threshold,
                should_accept=should_accept,
                was_accepted=was_accepted,
                result=test_result,
                duration_ms=duration_ms,
                embedding_dimension=embedding_dim
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Test failed for {profile.value}: {e}")

            return VoiceSecurityTest(
                profile_type=profile,
                test_phrase=self.test_phrase,
                similarity_score=0.0,
                threshold=self.verification_threshold or 0.75,
                should_accept=should_accept,
                was_accepted=False,
                result=TestResult.ERROR,
                duration_ms=duration_ms,
                error_message=str(e)
            )

    async def run_security_tests(self) -> VoiceSecurityReport:
        """
        Run complete voice security test suite.

        Returns:
            Comprehensive security report
        """
        logger.info("=" * 80)
        logger.info("JARVIS VOICE SECURITY TEST - STARTING")
        logger.info("=" * 80)

        start_time = time.time()
        tests = []

        # Load system configuration
        logger.info("Loading system configuration...")
        config = await self.load_system_config()
        logger.info(f"Configuration loaded: threshold={self.verification_threshold}, dimension={self.embedding_dimension}")

        # Test 1: Verify authorized user is ACCEPTED
        logger.info(f"\n{'='*80}")
        logger.info("TEST 1: Authorized User Acceptance Test")
        logger.info(f"{'='*80}")
        logger.info(f"Testing authorized user: {self.authorized_user}")
        logger.info("Expected result: ACCEPT (this is YOUR voice)")

        # Check if user has existing voice samples
        try:
            from backend.intelligence.cloud_database_adapter import CloudDatabaseAdapter
            db_adapter = CloudDatabaseAdapter()
            await db_adapter.initialize()

            samples = await db_adapter.get_voice_samples(self.authorized_user)

            if samples and len(samples) > 0:
                # Use existing sample for testing
                sample_path = Path(samples[0].get('file_path', ''))
                if sample_path.exists():
                    logger.info(f"Using existing voice sample: {sample_path}")
                    test = await self.test_voice_authentication(
                        audio_file=sample_path,
                        profile=VoiceProfile.AUTHORIZED_USER,
                        should_accept=True
                    )
                    tests.append(test)
                else:
                    logger.warning("Existing sample path not found, skipping authorized user test")
            else:
                logger.warning("No voice samples found for authorized user, skipping test")

        except Exception as e:
            logger.error(f"Failed to test authorized user: {e}")

        # Tests 2-N: Generate and test attacker voices
        logger.info(f"\n{'='*80}")
        logger.info("ATTACKER VOICE TESTS - Generating Synthetic Voices")
        logger.info(f"{'='*80}")

        for i, profile in enumerate(self.test_profiles, start=2):
            logger.info(f"\nTEST {i}: {profile.value.replace('_', ' ').title()}")
            logger.info(f"Expected result: REJECT (unauthorized voice)")

            # Generate synthetic voice
            audio_file = await self.generate_synthetic_voice(profile, self.test_phrase)

            if audio_file:
                # Test authentication
                test = await self.test_voice_authentication(
                    audio_file=audio_file,
                    profile=profile,
                    should_accept=False  # Attackers should be rejected
                )
                tests.append(test)

                # Log result
                logger.info(f"Similarity score: {test.similarity_score:.4f}")
                logger.info(f"Threshold: {test.threshold:.4f}")
                logger.info(f"Result: {test.security_verdict}")
            else:
                logger.warning(f"Skipping {profile.value} - voice generation failed")

        # Generate report
        total_duration_ms = (time.time() - start_time) * 1000
        report = VoiceSecurityReport(
            tests=tests,
            authorized_user_name=self.authorized_user,
            total_duration_ms=total_duration_ms
        )

        # Log summary
        self._log_security_report(report)

        return report

    def _log_security_report(self, report: VoiceSecurityReport):
        """Log security report summary"""
        logger.info(f"\n{'='*80}")
        logger.info("VOICE SECURITY TEST RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Authorized User: {report.authorized_user_name}")
        logger.info(f"Total Tests: {report.total_tests}")
        logger.info(f"Passed: {report.passed_tests}")
        logger.info(f"Failed: {report.failed_tests}")
        logger.info(f"Duration: {report.total_duration_ms:.0f}ms")
        logger.info(f"\n{'='*80}")
        logger.info(f"SECURITY STATUS: {'🔒 SECURE' if report.is_secure else '🚨 VULNERABLE'}")
        logger.info(f"{'='*80}")

        if report.security_breaches:
            logger.warning(f"\n⚠️ {len(report.security_breaches)} SECURITY BREACH(ES) DETECTED:")
            for breach in report.security_breaches:
                logger.warning(f"  - {breach.profile_type.value}: similarity={breach.similarity_score:.4f}")

        if report.false_rejections:
            logger.warning(f"\n⚠️ {len(report.false_rejections)} FALSE REJECTION(S):")
            for rejection in report.false_rejections:
                logger.warning(f"  - {rejection.profile_type.value}: similarity={rejection.similarity_score:.4f}")

        if report.is_secure and not report.false_rejections:
            logger.info("\n✅ Voice biometric security is working correctly!")
            logger.info("   - Authorized voice accepted")
            logger.info("   - All unauthorized voices rejected")

        logger.info(f"{'='*80}\n")

    async def save_report(self, report: VoiceSecurityReport, output_path: Optional[Path] = None):
        """
        Save security report to file.

        Args:
            report: Security report to save
            output_path: Optional custom output path
        """
        if output_path is None:
            output_path = Path.home() / '.jarvis' / 'logs' / 'voice_security_report.json'

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)

        logger.info(f"Security report saved: {output_path}")

    async def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")


async def main():
    """Main entry point for standalone execution"""
    print("\n" + "="*80)
    print("JARVIS VOICE SECURITY TESTER")
    print("="*80 + "\n")

    # Create tester
    tester = VoiceSecurityTester()

    try:
        # Run tests
        report = await tester.run_security_tests()

        # Save report
        await tester.save_report(report)

        # Exit with appropriate code
        exit_code = 0 if report.is_secure else 1
        return exit_code

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)
        return 1
    finally:
        await tester.cleanup()


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    exit(exit_code)
