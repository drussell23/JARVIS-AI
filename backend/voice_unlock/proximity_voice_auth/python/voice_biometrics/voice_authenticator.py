"""
Voice Biometric Authenticator
============================

Advanced voice authentication with anti-spoofing and continuous learning.
"""

import numpy as np
import librosa
import joblib
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
import json
import hashlib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .feature_extractor import VoiceFeatureExtractor
from .liveness_detector import LivenessDetector

logger = logging.getLogger(__name__)


class VoiceAuthenticator:
    """
    Advanced voice biometric authentication system with continuous learning.
    """
    
    def __init__(self, user_id: str, model_dir: Path = None):
        self.user_id = user_id
        self.model_dir = model_dir or Path.home() / ".jarvis" / "voice_models"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_extractor = VoiceFeatureExtractor()
        self.liveness_detector = LivenessDetector()
        
        # Voice model parameters
        self.model = None
        self.scaler = StandardScaler()
        self.voice_samples = []
        self.last_update = None
        
        # Authentication thresholds
        self.min_confidence = 85.0
        self.liveness_threshold = 80.0
        self.min_samples = 3
        
        # Learning parameters
        self.max_samples = 100
        self.update_interval = timedelta(days=1)
        
        # Load existing model if available
        self.load_model()
    
    def enroll_voice(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Enroll a new voice sample for the user.
        
        Args:
            audio_data: Audio waveform data
            sample_rate: Sample rate of the audio
            
        Returns:
            Enrollment result with status and metrics
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(
                audio_data, sample_rate
            )
            
            # Check liveness
            liveness_score = self.liveness_detector.check_liveness(
                audio_data, sample_rate
            )
            
            if liveness_score < self.liveness_threshold:
                return {
                    'success': False,
                    'reason': 'Liveness check failed',
                    'liveness_score': liveness_score
                }
            
            # Add to voice samples
            self.voice_samples.append({
                'features': features,
                'timestamp': datetime.now().isoformat(),
                'liveness_score': liveness_score
            })
            
            # Limit sample size
            if len(self.voice_samples) > self.max_samples:
                self.voice_samples = self.voice_samples[-self.max_samples:]
            
            # Update model if we have enough samples
            if len(self.voice_samples) >= self.min_samples:
                self._update_model()
                self.save_model()
            
            return {
                'success': True,
                'samples_collected': len(self.voice_samples),
                'model_ready': self.model is not None,
                'liveness_score': liveness_score
            }
            
        except Exception as e:
            logger.error(f"Error enrolling voice: {e}")
            return {
                'success': False,
                'reason': str(e)
            }
    
    def authenticate(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """
        Authenticate a voice sample against the enrolled model.
        
        Args:
            audio_data: Audio waveform data to authenticate
            sample_rate: Sample rate of the audio
            
        Returns:
            Authentication result with confidence score
        """
        try:
            # Check if model is ready
            if self.model is None:
                return {
                    'success': False,
                    'reason': 'No voice model enrolled',
                    'confidence': 0.0
                }
            
            # Extract features
            features = self.feature_extractor.extract_all_features(
                audio_data, sample_rate
            )
            
            # Check liveness
            liveness_score = self.liveness_detector.check_liveness(
                audio_data, sample_rate
            )
            
            if liveness_score < self.liveness_threshold:
                return {
                    'success': False,
                    'reason': 'Liveness check failed - possible replay attack',
                    'confidence': 0.0,
                    'liveness_score': liveness_score,
                    'threat_type': 'replay_attack'
                }
            
            # Prepare features for model
            feature_vector = self._prepare_features(features)
            feature_scaled = self.scaler.transform([feature_vector])
            
            # Get authentication score
            decision_score = self.model.decision_function(feature_scaled)[0]
            
            # Convert to confidence percentage (0-100)
            # One-class SVM scores: positive = inlier, negative = outlier
            confidence = self._score_to_confidence(decision_score)
            
            # Make authentication decision
            authenticated = confidence >= self.min_confidence
            
            # Continuous learning - add successful authentications
            if authenticated and self._should_update():
                self.voice_samples.append({
                    'features': features,
                    'timestamp': datetime.now().isoformat(),
                    'liveness_score': liveness_score
                })
                self._update_model()
                self.save_model()
            
            return {
                'success': authenticated,
                'confidence': confidence,
                'liveness_score': liveness_score,
                'decision_score': float(decision_score),
                'reason': 'Voice authenticated' if authenticated else 'Voice not recognized'
            }
            
        except Exception as e:
            logger.error(f"Error authenticating voice: {e}")
            return {
                'success': False,
                'reason': str(e),
                'confidence': 0.0
            }
    
    def _prepare_features(self, features: Dict) -> np.ndarray:
        """Convert feature dictionary to feature vector."""
        feature_vector = []
        
        # MFCC features
        if 'mfcc' in features:
            feature_vector.extend(features['mfcc'])
        
        # Spectral features
        for key in ['spectral_centroid', 'spectral_bandwidth', 
                   'spectral_rolloff', 'zero_crossing_rate']:
            if key in features:
                feature_vector.append(features[key])
        
        # Prosodic features
        for key in ['pitch_mean', 'pitch_std', 'energy_mean', 'energy_std']:
            if key in features:
                feature_vector.append(features[key])
        
        return np.array(feature_vector)
    
    def _update_model(self):
        """Update the voice model with current samples."""
        if len(self.voice_samples) < self.min_samples:
            logger.warning("Not enough samples to update model")
            return
        
        # Prepare training data
        X = []
        for sample in self.voice_samples:
            feature_vector = self._prepare_features(sample['features'])
            X.append(feature_vector)
        
        X = np.array(X)
        
        # Fit scaler
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Train One-Class SVM for anomaly detection
        self.model = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.05,  # Expected outlier fraction
            cache_size=200
        )
        
        self.model.fit(X_scaled)
        self.last_update = datetime.now()
        
        logger.info(f"Voice model updated with {len(X)} samples")
    
    def _score_to_confidence(self, score: float) -> float:
        """Convert SVM decision score to confidence percentage."""
        # Sigmoid transformation for smoother confidence scores
        # Adjust these parameters based on your model's score distribution
        confidence = 100 / (1 + np.exp(-2 * score))
        return max(0, min(100, confidence))
    
    def _should_update(self) -> bool:
        """Check if model should be updated."""
        if self.last_update is None:
            return True
        
        time_since_update = datetime.now() - self.last_update
        return time_since_update > self.update_interval
    
    def save_model(self):
        """Save the voice model to disk."""
        model_path = self.model_dir / f"{self.user_id}_voice_model.pkl"
        scaler_path = self.model_dir / f"{self.user_id}_scaler.pkl"
        samples_path = self.model_dir / f"{self.user_id}_samples.json"
        
        try:
            # Save model and scaler
            if self.model is not None:
                joblib.dump(self.model, model_path)
                joblib.dump(self.scaler, scaler_path)
            
            # Save voice samples
            with open(samples_path, 'w') as f:
                json.dump(self.voice_samples, f, indent=2)
            
            logger.info(f"Voice model saved for user {self.user_id}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load the voice model from disk."""
        model_path = self.model_dir / f"{self.user_id}_voice_model.pkl"
        scaler_path = self.model_dir / f"{self.user_id}_scaler.pkl"
        samples_path = self.model_dir / f"{self.user_id}_samples.json"
        
        try:
            if model_path.exists() and scaler_path.exists():
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Voice model loaded for user {self.user_id}")
            
            if samples_path.exists():
                with open(samples_path, 'r') as f:
                    self.voice_samples = json.load(f)
                logger.info(f"Loaded {len(self.voice_samples)} voice samples")
                
                # Parse last update time
                if self.voice_samples:
                    last_sample_time = self.voice_samples[-1]['timestamp']
                    self.last_update = datetime.fromisoformat(last_sample_time)
                    
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def get_statistics(self) -> Dict:
        """Get voice model statistics."""
        return {
            'user_id': self.user_id,
            'model_ready': self.model is not None,
            'samples_collected': len(self.voice_samples),
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'min_confidence': self.min_confidence,
            'liveness_threshold': self.liveness_threshold
        }