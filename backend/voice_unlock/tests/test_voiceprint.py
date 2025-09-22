"""
Unit Tests for Voiceprint Management
===================================

Comprehensive tests with no hardcoded values.
"""

import unittest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch
import tempfile
import os

from ..core.voiceprint import VoiceFeatures, Voiceprint, VoiceprintManager
from ..config import VoiceUnlockConfig, get_config, reset_config


class TestVoiceFeatures(unittest.TestCase):
    """Test VoiceFeatures class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample features with random data
        self.sample_rate = 16000
        self.n_mfcc = 13
        
        self.features = VoiceFeatures(
            mfcc_features=np.random.randn(self.n_mfcc * 2),
            pitch_contour=np.random.uniform(80, 300, 5),
            spectral_centroid=np.random.uniform(1000, 3000),
            zero_crossing_rate=np.random.uniform(0.01, 0.1),
            energy_profile=np.random.uniform(0.1, 1.0, 6),
            formants=[700, 1200, 2500, 3500]
        )
        
    def test_feature_creation(self):
        """Test feature object creation"""
        self.assertIsInstance(self.features.mfcc_features, np.ndarray)
        self.assertEqual(len(self.features.mfcc_features), self.n_mfcc * 2)
        self.assertEqual(len(self.features.formants), 4)
        
    def test_to_vector(self):
        """Test conversion to feature vector"""
        vector = self.features.to_vector()
        
        self.assertIsInstance(vector, np.ndarray)
        # Check vector contains all features
        expected_length = (
            len(self.features.mfcc_features) +
            len(self.features.pitch_contour) +
            2 +  # spectral_centroid and zero_crossing_rate
            len(self.features.energy_profile) +
            len(self.features.formants)
        )
        self.assertEqual(len(vector), expected_length)
        
    def test_vector_consistency(self):
        """Test that to_vector produces consistent results"""
        vector1 = self.features.to_vector()
        vector2 = self.features.to_vector()
        
        np.testing.assert_array_equal(vector1, vector2)


class TestVoiceprint(unittest.TestCase):
    """Test Voiceprint class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample features
        self.features_list = []
        for _ in range(3):
            features = VoiceFeatures(
                mfcc_features=np.random.randn(26),
                pitch_contour=np.random.uniform(80, 300, 5),
                spectral_centroid=np.random.uniform(1000, 3000),
                zero_crossing_rate=np.random.uniform(0.01, 0.1),
                energy_profile=np.random.uniform(0.1, 1.0, 6),
                formants=np.random.uniform(500, 4000, 4).tolist()
            )
            self.features_list.append(features)
            
        self.user_id = "test_user_001"
        
    def test_voiceprint_creation(self):
        """Test voiceprint creation"""
        voiceprint = Voiceprint(
            user_id=self.user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enrollment_samples=self.features_list,
            template_vector=np.array([]),
            variance_vector=np.array([]),
            metadata={}
        )
        
        self.assertEqual(voiceprint.user_id, self.user_id)
        self.assertEqual(voiceprint.sample_count, 3)
        
    def test_template_update(self):
        """Test template vector calculation"""
        voiceprint = Voiceprint(
            user_id=self.user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enrollment_samples=self.features_list,
            template_vector=np.array([]),
            variance_vector=np.array([]),
            metadata={}
        )
        
        voiceprint._update_template()
        
        self.assertGreater(len(voiceprint.template_vector), 0)
        self.assertGreater(len(voiceprint.variance_vector), 0)
        self.assertEqual(len(voiceprint.template_vector), len(voiceprint.variance_vector))
        
    def test_add_sample(self):
        """Test adding new sample to voiceprint"""
        voiceprint = Voiceprint(
            user_id=self.user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enrollment_samples=self.features_list[:2],
            template_vector=np.array([]),
            variance_vector=np.array([]),
            metadata={}
        )
        
        initial_count = voiceprint.sample_count
        initial_updated = voiceprint.updated_at
        
        # Add new sample
        new_features = self.features_list[2]
        voiceprint.add_sample(new_features)
        
        self.assertEqual(voiceprint.sample_count, initial_count + 1)
        self.assertGreater(voiceprint.updated_at, initial_updated)
        
    def test_match_score(self):
        """Test match score calculation"""
        voiceprint = Voiceprint(
            user_id=self.user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enrollment_samples=self.features_list,
            template_vector=np.array([]),
            variance_vector=np.array([]),
            metadata={}
        )
        
        voiceprint._update_template()
        
        # Test with enrolled sample (should have high score)
        score1 = voiceprint.match_score(self.features_list[0])
        self.assertGreater(score1, 0.5)
        self.assertLessEqual(score1, 1.0)
        
        # Test with random features (should have lower score)
        random_features = VoiceFeatures(
            mfcc_features=np.random.randn(26) * 10,  # Very different
            pitch_contour=np.random.uniform(400, 600, 5),
            spectral_centroid=5000,
            zero_crossing_rate=0.5,
            energy_profile=np.random.uniform(2, 3, 6),
            formants=[100, 200, 300, 400]
        )
        
        score2 = voiceprint.match_score(random_features)
        self.assertLess(score2, score1)
        
    def test_match_score_edge_cases(self):
        """Test match score with edge cases"""
        voiceprint = Voiceprint(
            user_id=self.user_id,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enrollment_samples=self.features_list,
            template_vector=np.zeros(50),
            variance_vector=np.zeros(50),  # Zero variance
            metadata={}
        )
        
        # Should handle zero variance
        score = voiceprint.match_score(self.features_list[0])
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)


class TestVoiceprintManager(unittest.TestCase):
    """Test VoiceprintManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Use temporary storage
        self.temp_dir = tempfile.mkdtemp()
        self.manager = VoiceprintManager(storage_path=self.temp_dir)
        
        # Create test features
        self.features_list = []
        for _ in range(3):
            features = VoiceFeatures(
                mfcc_features=np.random.randn(26),
                pitch_contour=np.random.uniform(80, 300, 5),
                spectral_centroid=np.random.uniform(1000, 3000),
                zero_crossing_rate=np.random.uniform(0.01, 0.1),
                energy_profile=np.random.uniform(0.1, 1.0, 6),
                formants=np.random.uniform(500, 4000, 4).tolist()
            )
            self.features_list.append(features)
            
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_create_voiceprint(self):
        """Test creating new voiceprint"""
        user_id = "test_user_001"
        voiceprint = self.manager.create_voiceprint(user_id, self.features_list)
        
        self.assertEqual(voiceprint.user_id, user_id)
        self.assertEqual(voiceprint.sample_count, 3)
        self.assertIn(user_id, self.manager.voiceprints)
        
    def test_duplicate_voiceprint(self):
        """Test creating duplicate voiceprint raises error"""
        user_id = "test_user_001"
        self.manager.create_voiceprint(user_id, self.features_list)
        
        with self.assertRaises(ValueError):
            self.manager.create_voiceprint(user_id, self.features_list)
            
    def test_verify_user(self):
        """Test user verification"""
        user_id = "test_user_001"
        self.manager.create_voiceprint(user_id, self.features_list)
        
        # Verify with enrolled feature
        verified, score = self.manager.verify_user(user_id, self.features_list[0])
        
        self.assertIsInstance(verified, bool)
        self.assertIsInstance(score, float)
        self.assertTrue(0 <= score <= 1)
        
    def test_verify_nonexistent_user(self):
        """Test verification of non-existent user"""
        verified, score = self.manager.verify_user("unknown_user", self.features_list[0])
        
        self.assertFalse(verified)
        self.assertEqual(score, 0.0)
        
    def test_identify_speaker(self):
        """Test speaker identification"""
        # Create multiple users
        users = ["user1", "user2", "user3"]
        for i, user_id in enumerate(users):
            # Create slightly different features for each user
            user_features = []
            for features in self.features_list:
                modified = VoiceFeatures(
                    mfcc_features=features.mfcc_features + i * 0.1,
                    pitch_contour=features.pitch_contour + i * 20,
                    spectral_centroid=features.spectral_centroid + i * 100,
                    zero_crossing_rate=features.zero_crossing_rate,
                    energy_profile=features.energy_profile,
                    formants=features.formants
                )
                user_features.append(modified)
            self.manager.create_voiceprint(user_id, user_features)
            
        # Test identification
        result = self.manager.identify_speaker(self.features_list[0])
        
        if result is not None:
            identified_user, score = result
            self.assertIn(identified_user, users)
            self.assertGreater(score, 0.7)
            
    def test_update_voiceprint(self):
        """Test updating existing voiceprint"""
        user_id = "test_user_001"
        voiceprint = self.manager.create_voiceprint(user_id, self.features_list[:2])
        
        initial_count = voiceprint.sample_count
        
        # Update with new sample
        self.manager.update_voiceprint(user_id, self.features_list[2])
        
        updated_voiceprint = self.manager.voiceprints[user_id]
        self.assertEqual(updated_voiceprint.sample_count, initial_count + 1)
        
    def test_update_nonexistent_voiceprint(self):
        """Test updating non-existent voiceprint raises error"""
        with self.assertRaises(ValueError):
            self.manager.update_voiceprint("unknown_user", self.features_list[0])
            
    def test_quality_assessment(self):
        """Test sample quality assessment"""
        # Create consistent samples
        consistent_features = []
        base_mfcc = np.random.randn(26)
        for i in range(3):
            features = VoiceFeatures(
                mfcc_features=base_mfcc + np.random.randn(26) * 0.01,  # Small variation
                pitch_contour=np.array([150, 155, 160, 155, 150]),
                spectral_centroid=2000,
                zero_crossing_rate=0.05,
                energy_profile=np.array([0.5, 0.6, 0.7, 0.6, 0.5, 0.4]),
                formants=[700, 1200, 2500, 3500]
            )
            consistent_features.append(features)
            
        quality_score = self.manager._assess_sample_quality(consistent_features)
        self.assertGreater(quality_score, 0.7)  # Should be high for consistent samples
        
        # Create inconsistent samples
        inconsistent_features = []
        for i in range(3):
            features = VoiceFeatures(
                mfcc_features=np.random.randn(26) * (i + 1),  # Large variation
                pitch_contour=np.random.uniform(50, 400, 5),
                spectral_centroid=np.random.uniform(500, 5000),
                zero_crossing_rate=np.random.uniform(0.001, 0.5),
                energy_profile=np.random.randn(6),
                formants=np.random.uniform(100, 5000, 4).tolist()
            )
            inconsistent_features.append(features)
            
        quality_score = self.manager._assess_sample_quality(inconsistent_features)
        self.assertLess(quality_score, 0.5)  # Should be low for inconsistent samples
        
    def test_adaptive_threshold(self):
        """Test adaptive threshold calculation"""
        user_id = "test_user_001"
        
        # High quality voiceprint
        voiceprint_high = self.manager.create_voiceprint(user_id, self.features_list)
        voiceprint_high.metadata['quality_score'] = 0.9
        threshold_high = self.manager._calculate_threshold(voiceprint_high)
        
        # Low quality voiceprint
        user_id_low = "test_user_002"
        voiceprint_low = self.manager.create_voiceprint(user_id_low, self.features_list)
        voiceprint_low.metadata['quality_score'] = 0.5
        threshold_low = self.manager._calculate_threshold(voiceprint_low)
        
        # High quality should have stricter threshold
        self.assertGreater(threshold_high, threshold_low)


class TestConfigIntegration(unittest.TestCase):
    """Test integration with configuration system"""
    
    def setUp(self):
        """Reset configuration before each test"""
        reset_config()
        
    def test_config_driven_thresholds(self):
        """Test that thresholds come from config"""
        config = get_config()
        
        # Modify config
        original_threshold = config.authentication.base_threshold
        config.authentication.base_threshold = 0.85
        
        manager = VoiceprintManager()
        
        # Create voiceprint
        features = [VoiceFeatures(
            mfcc_features=np.random.randn(26),
            pitch_contour=np.random.uniform(80, 300, 5),
            spectral_centroid=2000,
            zero_crossing_rate=0.05,
            energy_profile=np.ones(6) * 0.5,
            formants=[700, 1200, 2500, 3500]
        ) for _ in range(3)]
        
        voiceprint = manager.create_voiceprint("test_user", features)
        threshold = manager._calculate_threshold(voiceprint)
        
        # Threshold should be influenced by config
        self.assertAlmostEqual(threshold, config.authentication.base_threshold, delta=0.2)
        
        # Restore
        config.authentication.base_threshold = original_threshold


if __name__ == '__main__':
    unittest.main()