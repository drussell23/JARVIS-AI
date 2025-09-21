"""
Unit Tests for Voice Authentication Service
=========================================

Dynamic tests with mocked dependencies.
"""

import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import numpy as np
from datetime import datetime, timedelta
from collections import deque

from ..core.authentication import (
    VoiceAuthenticator, AuthenticationResult, AuthenticationAttempt,
    UserAuthState, ChallengeResponse
)
from ..core.voiceprint import VoiceFeatures, Voiceprint
from ..config import get_config, reset_config


class TestAuthenticationResult(unittest.TestCase):
    """Test AuthenticationResult enum"""
    
    def test_result_types(self):
        """Test all authentication result types exist"""
        expected_results = [
            'SUCCESS', 'FAILED', 'LOCKOUT', 'NO_VOICEPRINT',
            'POOR_QUALITY', 'SPOOFING_DETECTED', 'CHALLENGE_REQUIRED', 'ERROR'
        ]
        
        for result in expected_results:
            self.assertTrue(hasattr(AuthenticationResult, result))


class TestUserAuthState(unittest.TestCase):
    """Test UserAuthState tracking"""
    
    def test_state_initialization(self):
        """Test auth state initialization"""
        state = UserAuthState(user_id="test_user")
        
        self.assertEqual(state.user_id, "test_user")
        self.assertEqual(state.failed_attempts, 0)
        self.assertIsNone(state.lockout_until)
        self.assertIsNone(state.last_attempt)
        self.assertEqual(state.trust_score, 0.5)
        self.assertIsInstance(state.success_history, deque)
        
    def test_success_history_maxlen(self):
        """Test success history has max length"""
        state = UserAuthState(user_id="test_user")
        
        # Add more than maxlen items
        for i in range(150):
            state.success_history.append(0.8 + i * 0.001)
            
        self.assertEqual(len(state.success_history), 100)  # maxlen=100


class TestVoiceAuthenticator(unittest.TestCase):
    """Test VoiceAuthenticator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        reset_config()
        
        # Mock dependencies
        self.mock_voiceprint_manager = Mock()
        self.mock_feature_extractor = Mock()
        self.mock_anti_spoofing = Mock()
        
        # Create authenticator with mocked dependencies
        self.authenticator = VoiceAuthenticator(
            voiceprint_manager=self.mock_voiceprint_manager,
            feature_extractor=self.mock_feature_extractor,
            anti_spoofing=self.mock_anti_spoofing
        )
        
        # Create test features
        self.test_features = VoiceFeatures(
            mfcc_features=np.random.randn(26),
            pitch_contour=np.array([150, 155, 160, 155, 150]),
            spectral_centroid=2000,
            zero_crossing_rate=0.05,
            energy_profile=np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.4]),
            formants=[700, 1200, 2500, 3500]
        )
        
    def test_callback_registration(self):
        """Test callback registration"""
        callback = Mock()
        
        self.authenticator.add_callback('auth_success', callback)
        
        self.assertIn(callback, self.authenticator.callbacks['auth_success'])
        
    def test_lockout_checking(self):
        """Test lockout status checking"""
        user_id = "test_user"
        
        # Create locked out user
        state = UserAuthState(user_id=user_id)
        state.lockout_until = datetime.now() + timedelta(seconds=60)
        self.authenticator.user_states[user_id] = state
        
        # Check lockout
        self.assertTrue(self.authenticator._is_locked_out(user_id))
        
        # Check remaining time
        remaining = self.authenticator._get_lockout_remaining(user_id)
        self.assertGreater(remaining, 0)
        self.assertLessEqual(remaining, 60)
        
    def test_lockout_expiration(self):
        """Test lockout expiration"""
        user_id = "test_user"
        
        # Create expired lockout
        state = UserAuthState(user_id=user_id)
        state.lockout_until = datetime.now() - timedelta(seconds=1)
        state.failed_attempts = 5
        self.authenticator.user_states[user_id] = state
        
        # Should not be locked out
        self.assertFalse(self.authenticator._is_locked_out(user_id))
        
        # Failed attempts should be reset
        self.assertEqual(state.failed_attempts, 0)
        self.assertIsNone(state.lockout_until)
        
    @patch('backend.voice_unlock.core.authentication.AudioCapture')
    async def test_authenticate_no_audio(self, mock_audio_capture_class):
        """Test authentication with no audio detected"""
        # Mock audio capture
        mock_capture = Mock()
        mock_capture.calibrated = True
        mock_capture.capture_with_vad = Mock(return_value=(np.array([]), False))
        mock_audio_capture_class.return_value = mock_capture
        
        # Reset authenticator to use mocked audio capture
        self.authenticator.audio_capture = mock_capture
        
        result, details = await self.authenticator.authenticate()
        
        self.assertEqual(result, AuthenticationResult.FAILED)
        self.assertEqual(details['message'], 'No voice detected')
        
    async def test_authenticate_poor_quality(self):
        """Test authentication with poor quality audio"""
        # Mock audio capture
        audio_data = np.random.randn(16000)  # 1 second of audio
        
        # Mock validation failure
        self.mock_feature_extractor.validate_audio.return_value = (False, "Audio too quiet")
        
        result, details = await self.authenticator.authenticate(audio_data=audio_data)
        
        self.assertEqual(result, AuthenticationResult.POOR_QUALITY)
        self.assertEqual(details['message'], "Audio too quiet")
        
    async def test_authenticate_no_voiceprint(self):
        """Test authentication with no voiceprint"""
        user_id = "unknown_user"
        audio_data = np.random.randn(16000)
        
        # Mock successful validation and feature extraction
        self.mock_feature_extractor.validate_audio.return_value = (True, "Valid")
        self.mock_feature_extractor.extract_features.return_value = self.test_features
        
        # Mock no voiceprint found
        self.mock_voiceprint_manager.voiceprints = {}
        
        result, details = await self.authenticator.authenticate(
            user_id=user_id,
            audio_data=audio_data
        )
        
        self.assertEqual(result, AuthenticationResult.NO_VOICEPRINT)
        
    async def test_authenticate_spoofing_detected(self):
        """Test authentication with spoofing detected"""
        user_id = "test_user"
        audio_data = np.random.randn(16000)
        
        # Mock successful validation and feature extraction
        self.mock_feature_extractor.validate_audio.return_value = (True, "Valid")
        self.mock_feature_extractor.extract_features.return_value = self.test_features
        
        # Mock voiceprint exists
        self.mock_voiceprint_manager.voiceprints = {user_id: Mock()}
        
        # Mock spoofing detected
        self.mock_anti_spoofing.verify_liveness.return_value = (
            False, 0.3, {'replay_score': 0.2, 'synthetic_score': 0.4}
        )
        
        result, details = await self.authenticator.authenticate(
            user_id=user_id,
            audio_data=audio_data
        )
        
        self.assertEqual(result, AuthenticationResult.SPOOFING_DETECTED)
        self.assertIn('liveness_score', details)
        
    async def test_authenticate_success(self):
        """Test successful authentication"""
        user_id = "test_user"
        audio_data = np.random.randn(16000)
        
        # Mock successful validation and feature extraction
        self.mock_feature_extractor.validate_audio.return_value = (True, "Valid")
        self.mock_feature_extractor.extract_features.return_value = self.test_features
        
        # Mock voiceprint exists
        mock_voiceprint = Mock()
        self.mock_voiceprint_manager.voiceprints = {user_id: mock_voiceprint}
        
        # Mock liveness check passes
        self.mock_anti_spoofing.verify_liveness.return_value = (
            True, 0.9, {'liveness_score': 0.9}
        )
        
        # Mock voiceprint verification passes
        self.mock_voiceprint_manager.verify_user.return_value = (True, 0.85)
        
        # Mock callbacks
        success_callback = Mock()
        self.authenticator.add_callback('auth_success', success_callback)
        
        result, details = await self.authenticator.authenticate(
            user_id=user_id,
            audio_data=audio_data
        )
        
        self.assertEqual(result, AuthenticationResult.SUCCESS)
        self.assertIn('auth_score', details)
        self.assertIn('threshold', details)
        self.assertIn('factors', details)
        
        # Check callback was triggered
        success_callback.assert_called_once()
        
    async def test_authenticate_failed(self):
        """Test failed authentication"""
        user_id = "test_user"
        audio_data = np.random.randn(16000)
        
        # Set up mocks for failed authentication
        self.mock_feature_extractor.validate_audio.return_value = (True, "Valid")
        self.mock_feature_extractor.extract_features.return_value = self.test_features
        self.mock_voiceprint_manager.voiceprints = {user_id: Mock()}
        self.mock_anti_spoofing.verify_liveness.return_value = (True, 0.9, {})
        
        # Mock voiceprint verification fails
        self.mock_voiceprint_manager.verify_user.return_value = (False, 0.5)
        
        result, details = await self.authenticator.authenticate(
            user_id=user_id,
            audio_data=audio_data
        )
        
        self.assertEqual(result, AuthenticationResult.FAILED)
        self.assertIn('attempts_remaining', details)
        
    async def test_authenticate_lockout(self):
        """Test authentication during lockout"""
        user_id = "test_user"
        
        # Create locked out user
        state = UserAuthState(user_id=user_id)
        state.lockout_until = datetime.now() + timedelta(seconds=300)
        self.authenticator.user_states[user_id] = state
        
        result, details = await self.authenticator.authenticate(user_id=user_id)
        
        self.assertEqual(result, AuthenticationResult.LOCKOUT)
        self.assertIn('lockout_remaining', details)
        
    def test_calculate_auth_factors(self):
        """Test multi-factor calculation"""
        user_id = "test_user"
        
        # Create user with history
        state = UserAuthState(user_id=user_id)
        state.success_history.extend([0.8, 0.82, 0.79, 0.81, 0.83])
        state.last_attempt = datetime.now() - timedelta(hours=2)
        self.authenticator.user_states[user_id] = state
        
        factors = self.authenticator._calculate_auth_factors(
            user_id=user_id,
            features=self.test_features,
            match_score=0.82,
            identification_score=0.9,
            spoofing_results={'liveness_score': 0.95}
        )
        
        self.assertIn('voice_match', factors)
        self.assertIn('identification', factors)
        self.assertIn('liveness', factors)
        self.assertIn('consistency', factors)
        self.assertIn('temporal', factors)
        self.assertIn('audio_quality', factors)
        
        # Check factor ranges
        for factor, score in factors.items():
            self.assertTrue(0 <= score <= 1, f"{factor} score out of range: {score}")
            
    def test_calculate_final_score(self):
        """Test final score calculation"""
        factors = {
            'voice_match': 0.85,
            'liveness': 0.9,
            'consistency': 0.8,
            'audio_quality': 0.7
        }
        
        score = self.authenticator._calculate_final_score(factors)
        
        self.assertTrue(0 <= score <= 1)
        # Score should be weighted average
        self.assertLess(score, max(factors.values()))
        self.assertGreater(score, min(factors.values()))
        
    def test_adaptive_threshold(self):
        """Test adaptive threshold behavior"""
        user_id = "test_user"
        
        # Initially no adaptive threshold
        threshold = self.authenticator._get_adaptive_threshold(user_id)
        config = get_config()
        self.assertEqual(threshold, config.authentication.base_threshold)
        
        # Create user with adaptive threshold
        state = UserAuthState(user_id=user_id)
        state.adaptive_threshold = 0.8
        self.authenticator.user_states[user_id] = state
        
        threshold = self.authenticator._get_adaptive_threshold(user_id)
        self.assertEqual(threshold, 0.8)
        
    def test_threshold_adaptation(self):
        """Test threshold adaptation logic"""
        user_id = "test_user"
        state = UserAuthState(user_id=user_id)
        self.authenticator.user_states[user_id] = state
        
        # Test upward adaptation on consistent high scores
        self.authenticator._adapt_threshold(user_id, 0.95, success=True)
        self.assertIsNotNone(state.adaptive_threshold)
        initial_threshold = state.adaptive_threshold
        
        # Multiple high scores should increase threshold slightly
        for _ in range(5):
            self.authenticator._adapt_threshold(user_id, 0.95, success=True)
            
        self.assertGreater(state.adaptive_threshold, initial_threshold)
        
        # Test bounds
        config = get_config()
        self.assertLessEqual(
            state.adaptive_threshold,
            config.authentication.high_quality_threshold
        )
        
    def test_challenge_creation(self):
        """Test challenge-response creation"""
        user_id = "test_user"
        
        challenge = self.authenticator._create_challenge(user_id)
        
        self.assertIsInstance(challenge, ChallengeResponse)
        self.assertIn(challenge.challenge_type, ['repeat', 'math', 'random_words'])
        self.assertIsNotNone(challenge.challenge_text)
        self.assertEqual(challenge.timeout, 30.0)
        self.assertIn(user_id, self.authenticator.active_challenges)
        
    def test_snr_calculation(self):
        """Test SNR calculation from features"""
        # Create features with known energy distribution
        features = VoiceFeatures(
            mfcc_features=np.zeros(26),
            pitch_contour=np.zeros(5),
            spectral_centroid=2000,
            zero_crossing_rate=0.05,
            energy_profile=np.array([0.1, 0.8, 0.9, 0.7, 0.2, 0.1]),
            formants=[700, 1200, 2500, 3500]
        )
        
        snr = self.authenticator._calculate_snr(features)
        
        self.assertIsInstance(snr, float)
        self.assertGreater(snr, 0)  # Should be positive for good signal
        
    def test_environment_comparison(self):
        """Test environment comparison"""
        features1 = VoiceFeatures(
            mfcc_features=np.zeros(26),
            pitch_contour=np.zeros(5),
            spectral_centroid=2000,
            zero_crossing_rate=0.05,
            energy_profile=np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.4]),
            formants=[700, 1200, 2500, 3500]
        )
        
        # Similar environment
        features2 = VoiceFeatures(
            mfcc_features=np.zeros(26),
            pitch_contour=np.zeros(5),
            spectral_centroid=2100,  # Slightly different
            zero_crossing_rate=0.05,
            energy_profile=np.array([0.5, 0.65, 0.7, 0.65, 0.5, 0.4]),
            formants=[700, 1200, 2500, 3500]
        )
        
        similarity = self.authenticator._compare_environments(features1, features2)
        
        self.assertTrue(0 <= similarity <= 1)
        self.assertGreater(similarity, 0.8)  # Should be high for similar environments
        
    def test_user_stats(self):
        """Test user statistics retrieval"""
        user_id = "test_user"
        
        # No stats for unknown user
        stats = self.authenticator.get_user_stats("unknown_user")
        self.assertIsNone(stats)
        
        # Create user with history
        state = UserAuthState(user_id=user_id)
        state.trust_score = 0.75
        state.failed_attempts = 1
        state.last_attempt = datetime.now()
        state.success_history.extend([0.8, 0.82, 0.85, 0.83])
        state.adaptive_threshold = 0.78
        
        self.authenticator.user_states[user_id] = state
        
        stats = self.authenticator.get_user_stats(user_id)
        
        self.assertIsNotNone(stats)
        self.assertEqual(stats['user_id'], user_id)
        self.assertEqual(stats['trust_score'], 0.75)
        self.assertEqual(stats['failed_attempts'], 1)
        self.assertFalse(stats['is_locked_out'])
        self.assertIn('success_history', stats)
        self.assertEqual(stats['success_history']['count'], 4)
        
    @patch('backend.voice_unlock.core.authentication.Path')
    @patch('backend.voice_unlock.core.authentication.open', new_callable=unittest.mock.mock_open)
    def test_audit_logging(self, mock_open, mock_path):
        """Test audit log writing"""
        config = get_config()
        config.security.audit_enabled = True
        
        # Mock path exists
        mock_path.return_value.expanduser.return_value.parent.mkdir.return_value = None
        mock_path.return_value.expanduser.return_value = 'test_audit.log'
        
        attempt = AuthenticationAttempt(
            timestamp=datetime.now(),
            user_id="test_user",
            result=AuthenticationResult.SUCCESS,
            score=0.85,
            audio_quality=0.9,
            liveness_score=0.95,
            factors={'voice_match': 0.85},
            metadata={'test': 'data'}
        )
        
        self.authenticator._save_auth_attempt(attempt)
        
        # Check file was opened for append
        mock_open.assert_called_with('test_audit.log', 'a')
        
        # Check JSON was written
        written_data = mock_open().write.call_args[0][0]
        self.assertIn('"result": "success"', written_data)
        self.assertIn('"score": 0.85', written_data)


class TestContinuousAuthentication(unittest.TestCase):
    """Test continuous authentication feature"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.authenticator = VoiceAuthenticator()
        
    @patch('asyncio.sleep', new_callable=AsyncMock)
    async def test_continuous_auth_loop(self, mock_sleep):
        """Test continuous authentication loop"""
        user_id = "test_user"
        callback = Mock()
        
        # Limit iterations for testing
        iteration_count = 0
        
        async def mock_authenticate(*args, **kwargs):
            nonlocal iteration_count
            iteration_count += 1
            if iteration_count > 2:
                raise asyncio.CancelledError()
            return AuthenticationResult.SUCCESS, {'score': 0.85}
            
        self.authenticator.authenticate = mock_authenticate
        
        # Run continuous auth (will be cancelled after 2 iterations)
        with self.assertRaises(asyncio.CancelledError):
            await self.authenticator.continuous_authentication(
                user_id=user_id,
                interval=10.0,
                callback=callback
            )
            
        # Check callback was called
        self.assertEqual(callback.call_count, 2)
        
        # Check sleep was called with appropriate interval
        mock_sleep.assert_called()


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()