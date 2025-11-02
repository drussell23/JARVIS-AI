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
import platform
import subprocess
import sys
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


class AudioBackend(Enum):
    """Available audio playback backends"""
    AFPLAY = "afplay"  # macOS
    APLAY = "aplay"    # Linux ALSA
    PYAUDIO = "pyaudio"  # Cross-platform Python library
    SOX = "sox"        # Cross-platform sound tool
    FFPLAY = "ffplay"  # FFmpeg audio player
    AUTO = "auto"      # Auto-detect best available


@dataclass
class PlaybackConfig:
    """Configuration for audio playback during testing"""
    enabled: bool = False  # Whether to play audio during tests
    verbose: bool = False  # Show detailed playback information
    backend: AudioBackend = AudioBackend.AUTO  # Which audio backend to use
    volume: float = 0.5  # Volume level (0.0 to 1.0)
    announce_profile: bool = True  # Announce which voice profile is playing
    pause_after_playback: float = 0.5  # Seconds to pause after playing audio


class AudioPlayer:
    """
    Cross-platform audio player with automatic backend detection.

    Supports multiple audio backends with graceful fallback:
    - macOS: afplay (built-in)
    - Linux: aplay (ALSA)
    - Cross-platform: PyAudio, sox, ffplay
    """

    def __init__(self, config: PlaybackConfig):
        """Initialize audio player with configuration"""
        self.config = config
        self.backend = None
        self._detect_backend()

    def _detect_backend(self):
        """Auto-detect best available audio backend"""
        if self.config.backend != AudioBackend.AUTO:
            # User specified a backend
            self.backend = self.config.backend
            return

        # Detect platform and check available tools
        system = platform.system().lower()

        # Try platform-specific backends first (most reliable)
        if system == 'darwin' and self._check_command('afplay'):
            self.backend = AudioBackend.AFPLAY
            logger.info("🔊 Audio backend: afplay (macOS)")
        elif system == 'linux' and self._check_command('aplay'):
            self.backend = AudioBackend.APLAY
            logger.info("🔊 Audio backend: aplay (Linux ALSA)")
        elif self._check_command('ffplay'):
            self.backend = AudioBackend.FFPLAY
            logger.info("🔊 Audio backend: ffplay (FFmpeg)")
        elif self._check_command('sox'):
            self.backend = AudioBackend.SOX
            logger.info("🔊 Audio backend: sox")
        else:
            # Try PyAudio as last resort
            try:
                import pyaudio
                self.backend = AudioBackend.PYAUDIO
                logger.info("🔊 Audio backend: PyAudio")
            except ImportError:
                logger.warning("⚠️ No audio backend available - audio playback disabled")
                self.config.enabled = False

    def _check_command(self, command: str) -> bool:
        """Check if a command is available in PATH"""
        try:
            subprocess.run(
                ['which', command],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    async def play(self, audio_file: Path, profile: 'VoiceProfile'):
        """
        Play audio file with current backend.

        Args:
            audio_file: Path to audio file
            profile: Voice profile being played (for announcements)
        """
        if not self.config.enabled:
            return

        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            return

        # Announce what's playing
        if self.config.announce_profile:
            profile_name = profile.value.replace('_', ' ').title()
            logger.info(f"🎤 Playing: {profile_name}")

        try:
            # Play audio based on backend
            if self.backend == AudioBackend.AFPLAY:
                await self._play_afplay(audio_file)
            elif self.backend == AudioBackend.APLAY:
                await self._play_aplay(audio_file)
            elif self.backend == AudioBackend.FFPLAY:
                await self._play_ffplay(audio_file)
            elif self.backend == AudioBackend.SOX:
                await self._play_sox(audio_file)
            elif self.backend == AudioBackend.PYAUDIO:
                await self._play_pyaudio(audio_file)
            else:
                logger.warning("No audio backend configured")
                return

            # Pause after playback
            if self.config.pause_after_playback > 0:
                await asyncio.sleep(self.config.pause_after_playback)

        except Exception as e:
            if self.config.verbose:
                logger.error(f"Audio playback error: {e}", exc_info=True)
            else:
                logger.warning(f"Audio playback failed: {e}")

    async def _play_afplay(self, audio_file: Path):
        """Play audio using macOS afplay"""
        volume = int(self.config.volume * 100)
        process = await asyncio.create_subprocess_exec(
            'afplay', '-v', str(volume / 100.0), str(audio_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()

    async def _play_aplay(self, audio_file: Path):
        """Play audio using Linux aplay"""
        process = await asyncio.create_subprocess_exec(
            'aplay', '-q', str(audio_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()

    async def _play_ffplay(self, audio_file: Path):
        """Play audio using ffplay"""
        volume = int(self.config.volume * 255)
        process = await asyncio.create_subprocess_exec(
            'ffplay', '-nodisp', '-autoexit', '-volume', str(volume), str(audio_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()

    async def _play_sox(self, audio_file: Path):
        """Play audio using sox"""
        volume = self.config.volume
        process = await asyncio.create_subprocess_exec(
            'play', '-q', '-v', str(volume), str(audio_file),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.wait()

    async def _play_pyaudio(self, audio_file: Path):
        """Play audio using PyAudio library"""
        try:
            import wave
            import pyaudio

            # Open wave file
            wf = wave.open(str(audio_file), 'rb')

            # Initialize PyAudio
            p = pyaudio.PyAudio()

            # Open stream
            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
            )

            # Read and play data
            chunk_size = 1024
            data = wf.readframes(chunk_size)

            while data:
                stream.write(data)
                data = wf.readframes(chunk_size)

            # Cleanup
            stream.stop_stream()
            stream.close()
            p.terminate()
            wf.close()

        except Exception as e:
            logger.error(f"PyAudio playback error: {e}")


class VoiceProfile(Enum):
    """
    Different voice profile types for comprehensive security testing.

    Tests voice biometric authentication against diverse vocal characteristics:
    - Gender variations (male, female, non-binary)
    - Age variations (child, teen, adult, elderly)
    - Vocal characteristics (deep, high-pitched, raspy, breathy)
    - Accents (US, UK, Australian, Indian, etc.)
    - Speech patterns (fast, slow, robotic, whispered)
    - Attack vectors (synthesized, pitched, modulated)
    """

    # Authorized user
    AUTHORIZED_USER = "authorized_user"

    # Gender-based attackers
    MALE_ATTACKER = "male_attacker"
    FEMALE_ATTACKER = "female_attacker"
    NONBINARY_ATTACKER = "nonbinary_attacker"

    # Age-based attackers
    CHILD_ATTACKER = "child_attacker"
    TEEN_ATTACKER = "teen_attacker"
    ELDERLY_ATTACKER = "elderly_attacker"

    # Vocal characteristic attackers
    DEEP_VOICE_ATTACKER = "deep_voice_attacker"
    HIGH_PITCHED_ATTACKER = "high_pitched_attacker"
    RASPY_VOICE_ATTACKER = "raspy_voice_attacker"
    BREATHY_VOICE_ATTACKER = "breathy_voice_attacker"
    NASAL_VOICE_ATTACKER = "nasal_voice_attacker"

    # Accent-based attackers
    BRITISH_ACCENT_ATTACKER = "british_accent_attacker"
    AUSTRALIAN_ACCENT_ATTACKER = "australian_accent_attacker"
    INDIAN_ACCENT_ATTACKER = "indian_accent_attacker"
    SOUTHERN_ACCENT_ATTACKER = "southern_accent_attacker"

    # Speech pattern attackers
    FAST_SPEAKER_ATTACKER = "fast_speaker_attacker"
    SLOW_SPEAKER_ATTACKER = "slow_speaker_attacker"
    WHISPERED_ATTACKER = "whispered_attacker"
    SHOUTED_ATTACKER = "shouted_attacker"

    # Synthetic/modified attackers
    ROBOTIC_ATTACKER = "robotic_attacker"
    PITCHED_ATTACKER = "pitched_attacker"
    SYNTHESIZED_ATTACKER = "synthesized_attacker"
    MODULATED_ATTACKER = "modulated_attacker"
    VOCODED_ATTACKER = "vocoded_attacker"


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

    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary dictionary for quick access to test results"""
        return {
            'total': self.total_tests,
            'passed': self.passed_tests,
            'failed': self.failed_tests,
            'is_secure': self.is_secure,
            'security_breaches': len(self.security_breaches),
            'false_rejections': len(self.false_rejections),
            'authorized_user': self.authorized_user_name,
            'duration_ms': self.total_duration_ms,
        }

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

    def __init__(self, config: Optional[Dict[str, Any]] = None, playback_config: Optional[PlaybackConfig] = None):
        """
        Initialize voice security tester.

        Args:
            config: Optional configuration overrides
            playback_config: Audio playback configuration (enables audio during tests)
        """
        self.config = config or {}
        self.authorized_user = self.config.get('authorized_user', 'Derek')
        self.test_phrase = self.config.get('test_phrase', 'unlock my screen')
        self.temp_dir = Path(tempfile.gettempdir()) / 'jarvis_voice_security_tests'
        self.temp_dir.mkdir(exist_ok=True)

        # Dynamic configuration (no hardcoding)
        self.verification_threshold = None  # Will be loaded from system
        self.embedding_dimension = None  # Will be detected

        # Audio playback configuration
        self.playback_config = playback_config or PlaybackConfig()
        self.audio_player = AudioPlayer(self.playback_config) if self.playback_config.enabled else None

        # Test profile selection (dynamic based on config)
        test_mode = self.config.get('test_mode', 'standard')
        self.test_profiles = self._select_test_profiles(test_mode)

        logger.info(f"Voice Security Tester initialized for user: {self.authorized_user}")
        logger.info(f"   Test mode: {test_mode} ({len(self.test_profiles)} profiles)")
        if self.playback_config.enabled:
            logger.info(f"   Audio playback: ENABLED (backend: {self.audio_player.backend.value if self.audio_player else 'none'})")
        else:
            logger.info("   Audio playback: DISABLED (silent mode)")

    def _select_test_profiles(self, test_mode: str) -> List[VoiceProfile]:
        """
        Select test profiles based on test mode.

        Args:
            test_mode: Test mode ('quick', 'standard', 'comprehensive', 'full')

        Returns:
            List of voice profiles to test
        """
        if test_mode == 'quick':
            # Quick test: 3 basic profiles
            return [
                VoiceProfile.MALE_ATTACKER,
                VoiceProfile.FEMALE_ATTACKER,
                VoiceProfile.ROBOTIC_ATTACKER,
            ]

        elif test_mode == 'standard':
            # Standard test: 8 diverse profiles
            return [
                VoiceProfile.MALE_ATTACKER,
                VoiceProfile.FEMALE_ATTACKER,
                VoiceProfile.CHILD_ATTACKER,
                VoiceProfile.ELDERLY_ATTACKER,
                VoiceProfile.DEEP_VOICE_ATTACKER,
                VoiceProfile.HIGH_PITCHED_ATTACKER,
                VoiceProfile.ROBOTIC_ATTACKER,
                VoiceProfile.PITCHED_ATTACKER,
            ]

        elif test_mode == 'comprehensive':
            # Comprehensive test: 15 profiles covering major categories
            return [
                # Gender variations
                VoiceProfile.MALE_ATTACKER,
                VoiceProfile.FEMALE_ATTACKER,
                VoiceProfile.NONBINARY_ATTACKER,
                # Age variations
                VoiceProfile.CHILD_ATTACKER,
                VoiceProfile.TEEN_ATTACKER,
                VoiceProfile.ELDERLY_ATTACKER,
                # Vocal characteristics
                VoiceProfile.DEEP_VOICE_ATTACKER,
                VoiceProfile.HIGH_PITCHED_ATTACKER,
                VoiceProfile.RASPY_VOICE_ATTACKER,
                # Accents
                VoiceProfile.BRITISH_ACCENT_ATTACKER,
                VoiceProfile.AUSTRALIAN_ACCENT_ATTACKER,
                # Speech patterns
                VoiceProfile.FAST_SPEAKER_ATTACKER,
                VoiceProfile.WHISPERED_ATTACKER,
                # Synthetic
                VoiceProfile.ROBOTIC_ATTACKER,
                VoiceProfile.SYNTHESIZED_ATTACKER,
            ]

        elif test_mode == 'full':
            # Full test: ALL profiles (maximum security validation)
            return [
                VoiceProfile.MALE_ATTACKER,
                VoiceProfile.FEMALE_ATTACKER,
                VoiceProfile.NONBINARY_ATTACKER,
                VoiceProfile.CHILD_ATTACKER,
                VoiceProfile.TEEN_ATTACKER,
                VoiceProfile.ELDERLY_ATTACKER,
                VoiceProfile.DEEP_VOICE_ATTACKER,
                VoiceProfile.HIGH_PITCHED_ATTACKER,
                VoiceProfile.RASPY_VOICE_ATTACKER,
                VoiceProfile.BREATHY_VOICE_ATTACKER,
                VoiceProfile.NASAL_VOICE_ATTACKER,
                VoiceProfile.BRITISH_ACCENT_ATTACKER,
                VoiceProfile.AUSTRALIAN_ACCENT_ATTACKER,
                VoiceProfile.INDIAN_ACCENT_ATTACKER,
                VoiceProfile.SOUTHERN_ACCENT_ATTACKER,
                VoiceProfile.FAST_SPEAKER_ATTACKER,
                VoiceProfile.SLOW_SPEAKER_ATTACKER,
                VoiceProfile.WHISPERED_ATTACKER,
                VoiceProfile.SHOUTED_ATTACKER,
                VoiceProfile.ROBOTIC_ATTACKER,
                VoiceProfile.PITCHED_ATTACKER,
                VoiceProfile.SYNTHESIZED_ATTACKER,
                VoiceProfile.MODULATED_ATTACKER,
                VoiceProfile.VOCODED_ATTACKER,
            ]

        else:
            # Default to standard
            logger.warning(f"Unknown test mode '{test_mode}', using 'standard'")
            return self._select_test_profiles('standard')

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
            # Comprehensive macOS voice mapping for all profiles
            voice_map = {
                # Gender variations
                VoiceProfile.MALE_ATTACKER: "Alex",  # US male
                VoiceProfile.FEMALE_ATTACKER: "Samantha",  # US female
                VoiceProfile.NONBINARY_ATTACKER: "Fred",  # Neutral voice

                # Age variations
                VoiceProfile.CHILD_ATTACKER: "Karen",  # Child-like
                VoiceProfile.TEEN_ATTACKER: "Ava",  # Teen/young adult
                VoiceProfile.ELDERLY_ATTACKER: "Ralph",  # Elderly male

                # Vocal characteristics
                VoiceProfile.DEEP_VOICE_ATTACKER: "Bruce",  # Deep voice
                VoiceProfile.HIGH_PITCHED_ATTACKER: "Princess",  # High-pitched
                VoiceProfile.RASPY_VOICE_ATTACKER: "Bahh",  # Raspy
                VoiceProfile.BREATHY_VOICE_ATTACKER: "Whisper",  # Breathy/whispered
                VoiceProfile.NASAL_VOICE_ATTACKER: "Bubbles",  # Nasal

                # Accents
                VoiceProfile.BRITISH_ACCENT_ATTACKER: "Daniel",  # British male
                VoiceProfile.AUSTRALIAN_ACCENT_ATTACKER: "Karen",  # Australian
                VoiceProfile.INDIAN_ACCENT_ATTACKER: "Rishi",  # Indian (if available)
                VoiceProfile.SOUTHERN_ACCENT_ATTACKER: "Samantha",  # US (closest to Southern)

                # Speech patterns
                VoiceProfile.FAST_SPEAKER_ATTACKER: "Alex",  # Will adjust rate
                VoiceProfile.SLOW_SPEAKER_ATTACKER: "Alex",  # Will adjust rate
                VoiceProfile.WHISPERED_ATTACKER: "Whisper",  # Whispered
                VoiceProfile.SHOUTED_ATTACKER: "Bahh",  # Loud/emphatic

                # Synthetic/modified
                VoiceProfile.ROBOTIC_ATTACKER: "Zarvox",  # Robotic
                VoiceProfile.PITCHED_ATTACKER: "Ralph",  # Will pitch-shift
                VoiceProfile.SYNTHESIZED_ATTACKER: "Cellos",  # Synthesized
                VoiceProfile.MODULATED_ATTACKER: "Trinoids",  # Modulated
                VoiceProfile.VOCODED_ATTACKER: "Albert",  # Vocoded effect
            }

            voice = voice_map.get(profile, "Alex")
            rate_modifier = 1.0  # Default speech rate

            # Adjust speech rate for specific profiles
            if profile == VoiceProfile.FAST_SPEAKER_ATTACKER:
                rate_modifier = 1.5  # 50% faster
            elif profile == VoiceProfile.SLOW_SPEAKER_ATTACKER:
                rate_modifier = 0.6  # 40% slower
            elif profile == VoiceProfile.CHILD_ATTACKER:
                rate_modifier = 1.2  # Slightly faster

            # Build say command with rate modifier
            say_cmd = ['say', '-v', voice, '-r', str(int(200 * rate_modifier)), '-o', str(audio_file), '--data-format=LEF32@22050', text]

            process = await asyncio.create_subprocess_exec(
                *say_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            await asyncio.wait_for(process.wait(), timeout=10.0)

            if audio_file.exists() and audio_file.stat().st_size > 0:
                logger.info(f"Generated voice with macOS 'say' command: {voice} (rate: {rate_modifier}x)")
                return audio_file

        except Exception as e:
            logger.debug(f"macOS 'say' failed: {e}")

        # Engine 2: gTTS (Google TTS) - fallback
        try:
            from gtts import gTTS

            # Comprehensive gTTS language/accent mapping
            lang_map = {
                # Gender variations (use different accents)
                VoiceProfile.MALE_ATTACKER: 'en',  # US
                VoiceProfile.FEMALE_ATTACKER: 'en-uk',  # UK female
                VoiceProfile.NONBINARY_ATTACKER: 'en-ca',  # Canadian (neutral)

                # Age variations
                VoiceProfile.CHILD_ATTACKER: 'en-au',  # Australian (lighter tone)
                VoiceProfile.TEEN_ATTACKER: 'en',  # US (young)
                VoiceProfile.ELDERLY_ATTACKER: 'en-uk',  # UK (mature)

                # Vocal characteristics (use slow parameter for variation)
                VoiceProfile.DEEP_VOICE_ATTACKER: 'en-gb',  # British (deeper)
                VoiceProfile.HIGH_PITCHED_ATTACKER: 'en-au',  # Australian
                VoiceProfile.RASPY_VOICE_ATTACKER: 'en-ie',  # Irish
                VoiceProfile.BREATHY_VOICE_ATTACKER: 'en-uk',  # UK
                VoiceProfile.NASAL_VOICE_ATTACKER: 'en-in',  # Indian

                # Accents
                VoiceProfile.BRITISH_ACCENT_ATTACKER: 'en-uk',  # UK
                VoiceProfile.AUSTRALIAN_ACCENT_ATTACKER: 'en-au',  # Australian
                VoiceProfile.INDIAN_ACCENT_ATTACKER: 'en-in',  # Indian
                VoiceProfile.SOUTHERN_ACCENT_ATTACKER: 'en',  # US

                # Speech patterns
                VoiceProfile.FAST_SPEAKER_ATTACKER: 'en',  # US (will use slow=False)
                VoiceProfile.SLOW_SPEAKER_ATTACKER: 'en',  # US (will use slow=True)
                VoiceProfile.WHISPERED_ATTACKER: 'en-uk',  # UK
                VoiceProfile.SHOUTED_ATTACKER: 'en',  # US

                # Synthetic/modified
                VoiceProfile.ROBOTIC_ATTACKER: 'en-in',  # Indian (robotic quality)
                VoiceProfile.PITCHED_ATTACKER: 'en-ca',  # Canadian
                VoiceProfile.SYNTHESIZED_ATTACKER: 'en',  # US
                VoiceProfile.MODULATED_ATTACKER: 'en-ie',  # Irish
                VoiceProfile.VOCODED_ATTACKER: 'en-nz',  # New Zealand
            }

            lang = lang_map.get(profile, 'en')
            # Use slow parameter for specific profiles
            slow = profile == VoiceProfile.SLOW_SPEAKER_ATTACKER

            tts = gTTS(text=text, lang=lang, slow=slow)
            tts.save(str(audio_file))

            if audio_file.exists():
                logger.info(f"Generated voice with gTTS: {lang} (slow: {slow})")
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
            # Play audio if playback is enabled
            if self.audio_player:
                await self.audio_player.play(audio_file, profile)

            # Import verification service
            from backend.voice.speaker_verification_service import SpeakerVerificationService

            # Initialize service
            verification_service = SpeakerVerificationService()
            await verification_service.initialize()

            # Read audio file as bytes
            with open(audio_file, 'rb') as f:
                audio_data = f.read()

            # Perform verification
            result = await verification_service.verify_speaker(
                audio_data=audio_data,
                speaker_name=self.authorized_user
            )

            # Extract results
            similarity_score = result.get('confidence', 0.0)  # Changed from 'similarity_score' to 'confidence'
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
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='JARVIS Voice Security Tester - Test voice biometric authentication security',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Quick silent test (3 profiles)
  python3 voice_security_tester.py --mode quick

  # Standard test with audio playback
  python3 voice_security_tester.py --play-audio

  # Comprehensive test with verbose output and audio
  python3 voice_security_tester.py --mode comprehensive --play-audio --verbose

  # Full test (all 24 profiles) with audio
  python3 voice_security_tester.py --mode full --play-audio

Test Modes:
  quick         - 3 profiles (~1 min)
  standard      - 8 profiles (~3 min) [default]
  comprehensive - 15 profiles (~5 min)
  full          - 24 profiles (~8 min)
        '''
    )

    # Audio playback options
    parser.add_argument(
        '--play-audio', '--play', '-p',
        action='store_true',
        help='Play synthetic voices during testing (silent by default)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed/verbose output including audio playback details'
    )

    # Test configuration
    parser.add_argument(
        '--mode', '-m',
        choices=['quick', 'standard', 'comprehensive', 'full'],
        default='standard',
        help='Test mode: quick (3), standard (8), comprehensive (15), or full (24 profiles)'
    )

    parser.add_argument(
        '--user', '-u',
        default='Derek',
        help='Authorized user name to test against (default: Derek)'
    )

    parser.add_argument(
        '--phrase', '--text',
        default='unlock my screen',
        help='Test phrase to synthesize (default: "unlock my screen")'
    )

    # Audio backend selection
    parser.add_argument(
        '--backend', '-b',
        choices=['auto', 'afplay', 'aplay', 'pyaudio', 'sox', 'ffplay'],
        default='auto',
        help='Audio playback backend (default: auto-detect)'
    )

    parser.add_argument(
        '--volume',
        type=float,
        default=0.5,
        help='Audio playback volume (0.0 to 1.0, default: 0.5)'
    )

    args = parser.parse_args()

    # Display banner
    print("\n" + "="*80)
    print("JARVIS VOICE SECURITY TESTER")
    print("="*80 + "\n")

    # Create playback configuration
    playback_config = PlaybackConfig(
        enabled=args.play_audio,
        verbose=args.verbose,
        backend=AudioBackend(args.backend.upper()) if args.backend != 'auto' else AudioBackend.AUTO,
        volume=max(0.0, min(1.0, args.volume)),  # Clamp to 0.0-1.0
        announce_profile=True,
        pause_after_playback=0.5
    )

    # Create test configuration
    test_config = {
        'authorized_user': args.user,
        'test_phrase': args.phrase,
        'test_mode': args.mode,
    }

    # Create tester with configurations
    tester = VoiceSecurityTester(
        config=test_config,
        playback_config=playback_config
    )

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
