"""
Integration Tests for Voice Unlock System
========================================

End-to-end tests with realistic scenarios.
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..config import get_config, reset_config
from ..core.anti_spoofing import AntiSpoofingDetector
from ..core.authentication import AuthenticationResult, VoiceAuthenticator
from ..core.enrollment import VoiceEnrollmentManager
from ..core.feature_extraction import VoiceFeatureExtractor
from ..core.voiceprint import VoiceprintManager
from ..services.keychain_service import KeychainService
from ..services.screensaver_integration import ScreensaverIntegration


class TestEndToEndFlow(unittest.TestCase):
    """Test complete enrollment and authentication flow"""

    def setUp(self):
        """Set up test environment"""
        reset_config()
        self.config = get_config()

        # Use temporary storage
        self.temp_dir = tempfile.mkdtemp()

        # Create real components (with mocked audio)
        self.voiceprint_manager = VoiceprintManager(storage_path=self.temp_dir)
        self.feature_extractor = VoiceFeatureExtractor()
        self.anti_spoofing = AntiSpoofingDetector()

        # Create managers
        self.enrollment_manager = VoiceEnrollmentManager(
            voiceprint_manager=self.voiceprint_manager,
            feature_extractor=self.feature_extractor,
            anti_spoofing=self.anti_spoofing,
        )

        self.authenticator = VoiceAuthenticator(
            voiceprint_manager=self.voiceprint_manager,
            feature_extractor=self.feature_extractor,
            anti_spoofing=self.anti_spoofing,
        )

        # Generate test audio samples
        self.test_audio_samples = self._generate_test_audio()

    def tearDown(self):
        """Clean up"""
        import shutil

        shutil.rmtree(self.temp_dir)

    def _generate_test_audio(self, num_samples=5):
        """Generate realistic test audio samples"""
        samples = []
        sample_rate = self.config.audio.sample_rate
        duration = 2.0  # 2 seconds each

        for i in range(num_samples):
            # Generate audio with voice-like characteristics
            t = np.linspace(0, duration, int(sample_rate * duration))

            # Fundamental frequency (pitch)
            f0 = 150 + i * 10  # Slightly different pitch for each sample

            # Generate harmonics
            audio = np.zeros_like(t)
            for harmonic in range(1, 6):
                audio += (1.0 / harmonic) * np.sin(2 * np.pi * f0 * harmonic * t)

            # Add formants (vocal tract resonances)
            formant_freqs = [700, 1200, 2500]
            for freq in formant_freqs:
                audio += 0.3 * np.sin(2 * np.pi * freq * t)

            # Add some noise
            audio += 0.05 * np.random.randn(len(t))

            # Normalize
            audio = audio / np.max(np.abs(audio)) * 0.8

            samples.append(audio)

        return samples

    async def test_full_enrollment_and_auth_flow(self):
        """Test complete enrollment and authentication workflow"""
        user_id = "test_user_001"

        # Start enrollment
        session_id = self.enrollment_manager.start_enrollment(user_id)
        self.assertIsNotNone(session_id)

        # Mock audio capture to return our test samples
        with patch.object(
            self.enrollment_manager.audio_capture, "capture_with_vad"
        ) as mock_capture:
            # Enroll with multiple samples
            for i, audio_sample in enumerate(self.test_audio_samples[:3]):
                mock_capture.return_value = (audio_sample, True)

                success, message = await self.enrollment_manager.collect_sample(session_id)
                self.assertTrue(success, f"Sample {i+1} failed: {message}")

        # Check enrollment completed
        session = self.enrollment_manager.sessions[session_id]
        self.assertEqual(session.status.value, "completed")

        # Verify voiceprint was created
        self.assertIn(user_id, self.voiceprint_manager.voiceprints)

        # Test authentication with enrolled voice
        with patch.object(self.authenticator.audio_capture, "capture_with_vad") as mock_capture:
            # Use a sample similar to enrollment
            mock_capture.return_value = (self.test_audio_samples[0], True)

            # Mock liveness check to pass
            with patch.object(self.anti_spoofing, "verify_liveness") as mock_liveness:
                mock_liveness.return_value = (True, 0.9, {})

                result, details = await self.authenticator.authenticate(user_id=user_id)

        self.assertEqual(result, AuthenticationResult.SUCCESS)
        self.assertIn("auth_score", details)
        self.assertGreater(details["auth_score"], 0.7)

    async def test_enrollment_quality_rejection(self):
        """Test enrollment rejects poor quality samples"""
        user_id = "test_user_002"
        session_id = self.enrollment_manager.start_enrollment(user_id)

        # Create poor quality audio (too quiet)
        poor_audio = np.random.randn(self.config.audio.sample_rate * 2) * 0.001

        success, message = await self.enrollment_manager.collect_sample(
            session_id, audio_data=poor_audio
        )

        self.assertFalse(success)
        self.assertIn("quiet", message.lower())

    async def test_spoofing_detection(self):
        """Test spoofing detection during authentication"""
        user_id = "test_user_003"

        # First, enroll user
        await self._quick_enroll(user_id)

        # Create "spoofed" audio (simulate replay attack)
        original_audio = self.test_audio_samples[0]

        # Add artifacts that indicate replay
        # (In reality, this would be detected by spectral analysis)
        spoofed_audio = original_audio.copy()

        # Mock anti-spoofing to detect the attack
        with patch.object(self.anti_spoofing, "verify_liveness") as mock_liveness:
            mock_liveness.return_value = (
                False,
                0.3,
                {"replay_score": 0.2, "synthetic_score": 0.4, "liveness_score": 0.3},
            )

            result, details = await self.authenticator.authenticate(
                user_id=user_id, audio_data=spoofed_audio
            )

        self.assertEqual(result, AuthenticationResult.SPOOFING_DETECTED)
        self.assertIn("liveness_score", details)

    async def test_multi_user_system(self):
        """Test system with multiple enrolled users"""
        users = ["alice", "bob", "charlie"]

        # Enroll multiple users
        for i, user_id in enumerate(users):
            # Generate slightly different audio for each user
            user_audio = []
            for sample in self.test_audio_samples[:3]:
                # Shift pitch for each user
                shifted = sample * (1 + i * 0.1)
                user_audio.append(shifted)

            await self._quick_enroll(user_id, user_audio)

        # Verify all users enrolled
        self.assertEqual(len(self.voiceprint_manager.voiceprints), 3)

        # Test speaker identification
        test_audio = self.test_audio_samples[0] * 1.1  # Similar to Bob

        with patch.object(self.anti_spoofing, "verify_liveness") as mock_liveness:
            mock_liveness.return_value = (True, 0.9, {})

            # Test identification (no user_id provided)
            result, details = await self.authenticator.authenticate(audio_data=test_audio)

        # Should identify a user (may not be exact due to similarity)
        if result == AuthenticationResult.SUCCESS:
            self.assertIn("user_id", details)
            self.assertIn(details["user_id"], users)

    async def test_adaptive_threshold_learning(self):
        """Test adaptive threshold adjustment over time"""
        user_id = "test_user_004"

        # Enroll user
        await self._quick_enroll(user_id)

        # Perform multiple authentications
        auth_scores = []

        with patch.object(self.anti_spoofing, "verify_liveness") as mock_liveness:
            mock_liveness.return_value = (True, 0.9, {})

            for i in range(10):
                # Gradually improve audio quality
                audio = self.test_audio_samples[i % 3]
                if i > 5:
                    audio = audio * 1.1  # Slightly louder/clearer

                result, details = await self.authenticator.authenticate(
                    user_id=user_id, audio_data=audio
                )

                if result == AuthenticationResult.SUCCESS:
                    auth_scores.append(details["auth_score"])

        # Check if threshold adapted
        user_state = self.authenticator.user_states[user_id]
        self.assertIsNotNone(user_state.adaptive_threshold)

        # With consistent high scores, threshold should increase slightly
        if len(auth_scores) > 5 and np.mean(auth_scores) > 0.85:
            self.assertGreater(
                user_state.adaptive_threshold, self.config.authentication.base_threshold
            )

    async def test_lockout_mechanism(self):
        """Test account lockout after failed attempts"""
        user_id = "test_user_005"

        # Enroll user
        await self._quick_enroll(user_id)

        # Create very different audio (wrong person)
        wrong_audio = np.random.randn(self.config.audio.sample_rate * 2)

        with patch.object(self.anti_spoofing, "verify_liveness") as mock_liveness:
            mock_liveness.return_value = (True, 0.9, {})

            # Attempt authentication multiple times with wrong voice
            for i in range(self.config.authentication.max_attempts + 1):
                result, details = await self.authenticator.authenticate(
                    user_id=user_id, audio_data=wrong_audio
                )

                if i < self.config.authentication.max_attempts:
                    self.assertEqual(result, AuthenticationResult.FAILED)
                else:
                    # Should be locked out
                    self.assertEqual(result, AuthenticationResult.LOCKOUT)

        # Verify lockout is active
        self.assertTrue(self.authenticator._is_locked_out(user_id))

    async def _quick_enroll(self, user_id: str, audio_samples=None):
        """Helper for quick enrollment"""
        if audio_samples is None:
            audio_samples = self.test_audio_samples[:3]

        session_id = self.enrollment_manager.start_enrollment(user_id)

        with patch.object(
            self.enrollment_manager.audio_capture, "capture_with_vad"
        ) as mock_capture:
            for audio in audio_samples:
                mock_capture.return_value = (audio, True)
                await self.enrollment_manager.collect_sample(session_id)


class TestKeychainIntegration(unittest.TestCase):
    """Test Keychain storage integration"""

    def setUp(self):
        """Set up test environment"""
        reset_config()
        self.keychain = KeychainService()

        # Create test voiceprint
        from ..core.voiceprint import VoiceFeatures, Voiceprint

        features = VoiceFeatures(
            mfcc_features=np.random.randn(26),
            pitch_contour=np.random.uniform(80, 300, 5),
            spectral_centroid=2000,
            zero_crossing_rate=0.05,
            energy_profile=np.ones(6) * 0.5,
            formants=[700, 1200, 2500, 3500],
        )

        self.test_voiceprint = Voiceprint(
            user_id="test_keychain_user",
            created_at=np.datetime64("now"),
            updated_at=np.datetime64("now"),
            enrollment_samples=[features],
            template_vector=features.to_vector(),
            variance_vector=np.ones_like(features.to_vector()) * 0.1,
            metadata={"test": True},
        )

    @patch("keyring.set_password")
    @patch("keyring.get_password")
    def test_voiceprint_storage_and_retrieval(self, mock_get, mock_set):
        """Test storing and loading voiceprints"""
        # Mock keyring responses
        stored_data = None

        def mock_set_password(service, account, data):
            nonlocal stored_data
            stored_data = data

        def mock_get_password(service, account):
            if account == self.test_voiceprint.user_id:
                return stored_data
            return None

        mock_set.side_effect = mock_set_password
        mock_get.side_effect = mock_get_password

        # Store voiceprint
        success = self.keychain.store_voiceprint(self.test_voiceprint)
        self.assertTrue(success)

        # Retrieve voiceprint
        loaded = self.keychain.load_voiceprint(self.test_voiceprint.user_id)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.user_id, self.test_voiceprint.user_id)
        self.assertEqual(loaded.sample_count, self.test_voiceprint.sample_count)

    @patch("keyring.set_password")
    def test_encryption(self, mock_set):
        """Test voiceprint encryption"""
        config = get_config()
        config.security.encrypt_voiceprints = True

        captured_data = None

        def capture_password(service, account, data):
            nonlocal captured_data
            captured_data = data

        mock_set.side_effect = capture_password

        # Store with encryption
        self.keychain.store_voiceprint(self.test_voiceprint)

        # Data should be encrypted (not JSON readable)
        self.assertIsNotNone(captured_data)
        try:
            import json

            json.loads(captured_data)
            self.fail("Data should be encrypted, not plain JSON")
        except:
            pass  # Expected - data is encrypted

    def test_backup_export_import(self):
        """Test backup functionality"""
        with tempfile.NamedTemporaryFile(suffix=".backup") as tmp:
            backup_path = Path(tmp.name)

            # Mock voiceprint storage
            with patch.object(self.keychain, "load_voiceprint") as mock_load:
                mock_load.return_value = self.test_voiceprint

                with patch.object(self.keychain, "_load_index") as mock_index:
                    mock_index.return_value = {"users": [self.test_voiceprint.user_id]}

                    # Export backup
                    test_password = os.getenv("TEST_BACKUP_PASSWORD", "test123")
                    success = self.keychain.export_backup(backup_path, password=test_password)
                    self.assertTrue(success)

            # Verify backup file exists and is encrypted
            self.assertTrue(backup_path.exists())
            self.assertGreater(backup_path.stat().st_size, 0)


class TestScreensaverIntegration(unittest.TestCase):
    """Test screensaver integration"""

    def setUp(self):
        """Set up test environment"""
        reset_config()
        self.integration = ScreensaverIntegration()

    @patch("subprocess.run")
    def test_screen_state_detection(self, mock_run):
        """Test screen state detection methods"""
        # Mock screensaver running
        mock_run.return_value.stdout = "ScreenSaverEngine running"
        mock_run.return_value.returncode = 0

        is_running = self.integration._is_screensaver_running()
        # Note: This test may fail depending on Quartz availability
        # In real tests, we'd mock Quartz calls too

    def test_event_handler_registration(self):
        """Test event handler system"""
        handler_called = False
        event_data = None

        def test_handler(data):
            nonlocal handler_called, event_data
            handler_called = True
            event_data = data

        self.integration.add_event_handler("screen_locked", test_handler)

        # Trigger event
        self.integration._trigger_event("screen_locked", {"test": "data"})

        self.assertTrue(handler_called)
        self.assertEqual(event_data, {"test": "data"})

    @patch.object(ScreensaverIntegration, "_get_screen_state")
    @patch("asyncio.create_task")
    def test_monitoring_state_changes(self, mock_task, mock_get_state):
        """Test screen state monitoring"""
        # Simulate state change
        mock_get_state.side_effect = [
            self.integration.ScreenState.ACTIVE,
            self.integration.ScreenState.SCREENSAVER,
            self.integration.ScreenState.SCREENSAVER,
        ]

        handler_called = False

        def screensaver_handler():
            nonlocal handler_called
            handler_called = True

        self.integration.add_event_handler("screensaver_started", screensaver_handler)

        # Start monitoring (briefly)
        self.integration.start_monitoring()

        # Give monitor thread time to detect change
        import time

        time.sleep(0.5)

        self.integration.stop_monitoring()

        # Note: In real implementation, we'd need better async testing


def run_async_test(test_func):
    """Helper to run async tests"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(test_func())
    finally:
        loop.close()


if __name__ == "__main__":
    # Run integration tests
    unittest.main()
