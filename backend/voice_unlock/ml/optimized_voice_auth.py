"""
Optimized Voice Authentication with ML Manager
=============================================

Memory-efficient voice authentication using dynamic model loading
and optimization techniques for 16GB RAM systems.
"""

import numpy as np
import joblib
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import sounddevice as sd

from ..core.feature_extraction import VoiceFeatureExtractor
from ..core.anti_spoofing import AntiSpoofingDetector
from .ml_manager import get_ml_manager, MLModelManager
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class UserModel:
    """Lightweight user model metadata"""
    user_id: str
    model_id: str
    created: datetime
    last_updated: datetime
    sample_count: int
    model_path: Path
    scaler_path: Path
    pca_path: Optional[Path] = None
    feature_version: str = "v1"


class OptimizedVoiceAuthenticator:
    """
    Memory-optimized voice authenticator using ML Manager
    """
    
    def __init__(self):
        self.config = get_config()
        self.ml_manager = get_ml_manager({
            'max_memory_mb': self.config.performance.max_memory_mb,
            'max_cache_memory_mb': self.config.performance.cache_size_mb,
            'enable_quantization': True,
            'enable_compression': True,
            'enable_monitoring': self.config.performance.background_monitoring,
            'enable_lazy_loading': True,
            'enable_predictive_loading': True,
            'preload_threshold': 0.7,
            'aggressive_unload_timeout': 60
        })
        
        # Feature extraction (lazy loaded)
        self._feature_extractor = None
        self._anti_spoofing = None
        
        # User models metadata (lightweight)
        self.user_models: Dict[str, UserModel] = {}
        self._load_user_metadata()
        
        # Register lazy loaders for all known users
        self._register_user_model_loaders()
        
        # Feature dimension reduction
        self.use_pca = True
        self.pca_components = 50  # Reduce feature dimensions
        
    @property
    def feature_extractor(self):
        """Lazy load feature extractor"""
        if self._feature_extractor is None:
            self._feature_extractor = VoiceFeatureExtractor()
        return self._feature_extractor
        
    @property
    def anti_spoofing(self):
        """Lazy load anti-spoofing detector"""
        if self._anti_spoofing is None:
            self._anti_spoofing = AntiSpoofingDetector()
        return self._anti_spoofing
        
    def _load_user_metadata(self):
        """Load user model metadata (not the models themselves)"""
        metadata_file = Path(self.config.security.storage_path).expanduser() / 'user_models.json'
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                    
                for user_id, model_data in data.items():
                    self.user_models[user_id] = UserModel(
                        user_id=user_id,
                        model_id=model_data['model_id'],
                        created=datetime.fromisoformat(model_data['created']),
                        last_updated=datetime.fromisoformat(model_data['last_updated']),
                        sample_count=model_data['sample_count'],
                        model_path=Path(model_data['model_path']),
                        scaler_path=Path(model_data['scaler_path']),
                        pca_path=Path(model_data.get('pca_path')) if model_data.get('pca_path') else None,
                        feature_version=model_data.get('feature_version', 'v1')
                    )
                    
                logger.info(f"Loaded metadata for {len(self.user_models)} users")
            except Exception as e:
                logger.error(f"Failed to load user metadata: {e}")
                
    def _register_user_model_loaders(self):
        """Register lazy loaders for all known user models"""
        for user_id, user_model in self.user_models.items():
            # Register model loader
            self.ml_manager.register_lazy_loader(
                f"{user_model.model_id}_model",
                None,  # Will use default loader
                user_model.model_path,
                'sklearn'
            )
            
            # Register scaler loader
            self.ml_manager.register_lazy_loader(
                f"{user_model.model_id}_scaler",
                None,
                user_model.scaler_path,
                'sklearn'
            )
            
            # Register PCA loader if exists
            if user_model.pca_path and user_model.pca_path.exists():
                self.ml_manager.register_lazy_loader(
                    f"{user_model.model_id}_pca",
                    None,
                    user_model.pca_path,
                    'sklearn'
                )
                
        logger.info(f"Registered lazy loaders for {len(self.user_models)} users")
                
    def _save_user_metadata(self):
        """Save user model metadata"""
        metadata_file = Path(self.config.security.storage_path).expanduser() / 'user_models.json'
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        data = {}
        for user_id, model in self.user_models.items():
            data[user_id] = {
                'model_id': model.model_id,
                'created': model.created.isoformat(),
                'last_updated': model.last_updated.isoformat(),
                'sample_count': model.sample_count,
                'model_path': str(model.model_path),
                'scaler_path': str(model.scaler_path),
                'pca_path': str(model.pca_path) if model.pca_path else None,
                'feature_version': model.feature_version
            }
            
        with open(metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def enroll_user(self, user_id: str, audio_samples: List[np.ndarray], 
                    sample_rate: int = 16000) -> bool:
        """
        Enroll user with memory-efficient processing
        """
        try:
            logger.info(f"Enrolling user {user_id} with {len(audio_samples)} samples")
            
            # Extract features from all samples
            all_features = []
            for i, audio in enumerate(audio_samples):
                # Anti-spoofing check
                is_live = self.anti_spoofing.detect_spoofing(audio, sample_rate)
                if not is_live['is_live']:
                    logger.warning(f"Sample {i} failed liveness check")
                    continue
                    
                # Extract features
                features = self.feature_extractor.extract_all_features(audio, sample_rate)
                all_features.append(features)
                
            if len(all_features) < self.config.enrollment.min_samples:
                logger.error(f"Not enough valid samples for enrollment")
                return False
                
            # Convert to numpy array
            X = np.array(all_features)
            
            # Feature scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Dimensionality reduction with PCA
            pca = None
            if self.use_pca and X.shape[1] > self.pca_components:
                pca = PCA(n_components=self.pca_components)
                X_scaled = pca.fit_transform(X_scaled)
                logger.info(f"Reduced features from {X.shape[1]} to {X_scaled.shape[1]} dimensions")
                
            # Train One-Class SVM (lightweight model)
            model = OneClassSVM(
                kernel='rbf',
                gamma='auto',
                nu=0.05,
                cache_size=50  # Limit cache size
            )
            model.fit(X_scaled)
            
            # Generate unique model ID
            model_id = f"voice_model_{user_id}_{hashlib.md5(user_id.encode()).hexdigest()[:8]}"
            
            # Save model components
            model_dir = Path(self.config.security.storage_path).expanduser() / 'models' / user_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f'{model_id}.joblib'
            scaler_path = model_dir / f'{model_id}_scaler.joblib'
            pca_path = model_dir / f'{model_id}_pca.joblib' if pca else None
            
            # Save with compression
            joblib.dump(model, model_path, compress=3)
            joblib.dump(scaler, scaler_path, compress=3)
            if pca:
                joblib.dump(pca, pca_path, compress=3)
                
            # Create metadata
            user_model = UserModel(
                user_id=user_id,
                model_id=model_id,
                created=datetime.now(),
                last_updated=datetime.now(),
                sample_count=len(all_features),
                model_path=model_path,
                scaler_path=scaler_path,
                pca_path=pca_path,
                feature_version="v1"
            )
            
            self.user_models[user_id] = user_model
            self._save_user_metadata()
            
            # Register lazy loaders for new user
            self.ml_manager.register_lazy_loader(
                f"{model_id}_model",
                None,
                model_path,
                'sklearn'
            )
            self.ml_manager.register_lazy_loader(
                f"{model_id}_scaler",
                None,
                scaler_path,
                'sklearn'
            )
            if pca:
                self.ml_manager.register_lazy_loader(
                    f"{model_id}_pca",
                    None,
                    pca_path,
                    'sklearn'
                )
            
            logger.info(f"Successfully enrolled user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Enrollment failed for user {user_id}: {e}")
            return False
            
    def authenticate(self, user_id: str, audio_data: np.ndarray, 
                    sample_rate: int = 16000) -> Dict[str, Any]:
        """
        Authenticate user with dynamic model loading
        """
        result = {
            'authenticated': False,
            'confidence': 0.0,
            'reason': None,
            'processing_time': 0.0
        }
        
        start_time = datetime.now()
        
        try:
            # Check if user exists
            if user_id not in self.user_models:
                result['reason'] = "User not enrolled"
                return result
                
            user_model = self.user_models[user_id]
            
            # Anti-spoofing check
            spoofing_result = self.anti_spoofing.detect_spoofing(audio_data, sample_rate)
            if not spoofing_result['is_live']:
                result['reason'] = f"Spoofing detected: {spoofing_result['detection_type']}"
                return result
                
            # Extract features
            features = self.feature_extractor.extract_all_features(audio_data, sample_rate)
            
            # Load model components using lazy loading
            model = self.ml_manager.get_model_lazy(f"{user_model.model_id}_model")
            if model is None:
                # Fallback to direct loading if lazy loading fails
                model = self.ml_manager.load_model(
                    f"{user_model.model_id}_model",
                    user_model.model_path,
                    'sklearn'
                )
            
            scaler = self.ml_manager.get_model_lazy(f"{user_model.model_id}_scaler")
            if scaler is None:
                scaler = self.ml_manager.load_model(
                    f"{user_model.model_id}_scaler",
                    user_model.scaler_path,
                    'sklearn'
                )
            
            # Apply transformations
            features_scaled = scaler.transform([features])
            
            # Apply PCA if available
            if user_model.pca_path:
                pca = self.ml_manager.get_model_lazy(f"{user_model.model_id}_pca")
                if pca is None:
                    pca = self.ml_manager.load_model(
                        f"{user_model.model_id}_pca",
                        user_model.pca_path,
                        'sklearn'
                    )
                features_scaled = pca.transform(features_scaled)
                
            # Predict
            decision_score = model.decision_function(features_scaled)[0]
            prediction = model.predict(features_scaled)[0]
            
            # Calculate confidence (normalize decision score)
            confidence = 1 / (1 + np.exp(-decision_score))  # Sigmoid
            
            # Determine authentication result
            threshold = self._get_adaptive_threshold(user_id, spoofing_result['confidence'])
            
            result['authenticated'] = prediction == 1 and confidence >= threshold
            result['confidence'] = float(confidence)
            result['processing_time'] = (datetime.now() - start_time).total_seconds()
            
            if not result['authenticated']:
                result['reason'] = f"Low confidence: {confidence:.2f} < {threshold:.2f}"
                
            # Log authentication attempt
            self._log_authentication(user_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Authentication failed for user {user_id}: {e}")
            result['reason'] = "Internal error"
            return result
            
    def _get_adaptive_threshold(self, user_id: str, liveness_confidence: float) -> float:
        """
        Get adaptive threshold based on various factors
        """
        base_threshold = self.config.authentication.base_threshold
        
        # Adjust based on liveness confidence
        if liveness_confidence > 0.9:
            threshold = base_threshold * 0.95
        elif liveness_confidence < 0.7:
            threshold = base_threshold * 1.1
        else:
            threshold = base_threshold
            
        # Could add more factors like:
        # - Time of day
        # - Recent authentication history
        # - Environmental noise level
        
        return min(0.95, max(0.6, threshold))  # Clamp to reasonable range
        
    def update_user_model(self, user_id: str, audio_data: np.ndarray, 
                         sample_rate: int = 16000):
        """
        Update user model with new sample (continuous learning)
        """
        if user_id not in self.user_models:
            logger.error(f"User {user_id} not found")
            return
            
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(audio_data, sample_rate)
            
            # Load existing model
            user_model = self.user_models[user_id]
            
            # For now, just log - full implementation would:
            # 1. Load existing training data
            # 2. Add new sample
            # 3. Retrain model
            # 4. Save updated model
            
            logger.info(f"Model update logged for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update model for user {user_id}: {e}")
            
    def remove_user(self, user_id: str):
        """
        Remove user and unload their models
        """
        if user_id not in self.user_models:
            return
            
        user_model = self.user_models[user_id]
        
        # Unload models from ML manager
        self.ml_manager.unload_model(f"{user_model.model_id}_model")
        self.ml_manager.unload_model(f"{user_model.model_id}_scaler")
        if user_model.pca_path:
            self.ml_manager.unload_model(f"{user_model.model_id}_pca")
            
        # Remove files
        try:
            user_model.model_path.unlink(missing_ok=True)
            user_model.scaler_path.unlink(missing_ok=True)
            if user_model.pca_path:
                user_model.pca_path.unlink(missing_ok=True)
        except Exception as e:
            logger.error(f"Failed to remove model files: {e}")
            
        # Remove metadata
        del self.user_models[user_id]
        self._save_user_metadata()
        
        logger.info(f"Removed user {user_id}")
        
    def _log_authentication(self, user_id: str, result: Dict[str, Any]):
        """Log authentication attempt"""
        if not self.config.security.audit_enabled:
            return
            
        audit_path = Path(self.config.security.audit_path).expanduser()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': hashlib.sha256(user_id.encode()).hexdigest() if self.config.security.anonymize_logs else user_id,
            'authenticated': result['authenticated'],
            'confidence': result['confidence'],
            'reason': result['reason'],
            'processing_time': result['processing_time']
        }
        
        with open(audit_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics from ML manager"""
        return self.ml_manager.get_performance_report()
        
    def cleanup(self):
        """Cleanup resources"""
        self.ml_manager.cleanup()


# Example usage for testing
def test_optimized_auth():
    """Test the optimized authentication system"""
    auth = OptimizedVoiceAuthenticator()
    
    # Simulate recording audio
    print("Recording 3 seconds of audio for enrollment...")
    duration = 3
    sample_rate = 16000
    
    audio_samples = []
    for i in range(3):
        print(f"Recording sample {i+1}/3...")
        audio = sd.rec(int(duration * sample_rate), 
                      samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio_samples.append(audio.flatten())
        
    # Enroll user
    success = auth.enroll_user("test_user", audio_samples, sample_rate)
    print(f"Enrollment: {'Success' if success else 'Failed'}")
    
    # Test authentication
    print("\nRecording 3 seconds for authentication...")
    test_audio = sd.rec(int(duration * sample_rate), 
                       samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    
    result = auth.authenticate("test_user", test_audio.flatten(), sample_rate)
    print(f"Authentication result: {result}")
    
    # Show memory stats
    stats = auth.get_memory_stats()
    print(f"\nMemory stats: {json.dumps(stats, indent=2)}")
    
    # Cleanup
    auth.cleanup()


if __name__ == "__main__":
    test_optimized_auth()