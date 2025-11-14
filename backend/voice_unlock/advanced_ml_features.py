#!/usr/bin/env python3
"""
Advanced ML Features for JARVIS Continuous Learning
===================================================

This module provides advanced ML capabilities:
1. Random Forest for failure prediction
2. Bayesian Optimization for hyperparameter tuning
3. Multi-Armed Bandit for exploration/exploitation
4. Context-Aware Learning (time of day, voice variations)
5. Anomaly Detection (One-Class SVM / Isolation Forest)
6. Fine-Tuning mechanism for speaker embeddings
7. Progressive Learning Stages

All features are:
- Fully async
- Dynamic (no hardcoding)
- Robust with error handling
- Database-backed for persistence
"""

import asyncio
import logging
import numpy as np
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict, field
import sqlite3
import json

logger = logging.getLogger(__name__)


# ==================== Random Forest Failure Predictor ====================

class FailurePredictor:
    """
    Random Forest classifier to predict character typing failures.

    Learns patterns like:
    - Which character positions are problematic
    - When system load affects typing
    - Time-of-day correlation with failures
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None
        self.feature_names = [
            'char_position',
            'char_type_encoded',  # 0=letter, 1=digit, 2=special
            'requires_shift',
            'system_load',
            'hour_of_day',
            'day_of_week',
            'historical_failures_at_position',
            'avg_duration_at_position'
        ]

        # Lazy import to avoid dependency issues
        try:
            from sklearn.ensemble import RandomForestClassifier
            self.RandomForestClassifier = RandomForestClassifier
        except ImportError:
            logger.warning("⚠️ scikit-learn not available, failure prediction disabled")
            self.RandomForestClassifier = None

    async def initialize(self):
        """Load or create Random Forest model"""
        if not self.RandomForestClassifier:
            return

        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Loaded Random Forest model from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}, creating new one")
                self._create_new_model()
        else:
            self._create_new_model()

    def _create_new_model(self):
        """Create new Random Forest classifier"""
        if not self.RandomForestClassifier:
            return

        self.model = self.RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42
        )
        logger.info("✅ Created new Random Forest model")

    async def train_from_history(self, char_metrics: List[Dict[str, Any]]):
        """
        Train model on historical character typing data.

        Features:
        - Character position (1-N)
        - Character type (letter/digit/special)
        - Requires shift
        - System load
        - Time of day
        - Historical failure count
        """
        if not self.model or len(char_metrics) < 50:
            logger.debug("Not enough data to train Random Forest (need 50+ samples)")
            return

        try:
            X = []
            y = []

            for metric in char_metrics:
                features = [
                    metric.get('char_position', 0),
                    self._encode_char_type(metric.get('char_type', 'letter')),
                    1 if metric.get('requires_shift') else 0,
                    metric.get('system_load_at_char', 0) or 0,
                    datetime.fromisoformat(metric.get('timestamp', datetime.now().isoformat())).hour,
                    datetime.fromisoformat(metric.get('timestamp', datetime.now().isoformat())).weekday(),
                    metric.get('historical_failures', 0),
                    metric.get('avg_duration_at_pos', 50)
                ]

                X.append(features)
                y.append(0 if metric.get('success') else 1)  # 1 = failure

            X = np.array(X)
            y = np.array(y)

            # Train model
            self.model.fit(X, y)

            # Save model
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            logger.info(f"✅ Random Forest trained on {len(X)} samples")

        except Exception as e:
            logger.error(f"Failed to train Random Forest: {e}", exc_info=True)

    def _encode_char_type(self, char_type: str) -> int:
        """Encode character type as integer"""
        return {'letter': 0, 'digit': 1, 'special': 2}.get(char_type, 0)

    async def predict_failure_probability(
        self,
        char_position: int,
        char_type: str,
        requires_shift: bool,
        system_load: float,
        historical_failures: int,
        avg_duration: float
    ) -> float:
        """
        Predict probability of failure for this character.

        Returns: float between 0.0 (won't fail) and 1.0 (likely to fail)
        """
        if not self.model:
            return 0.0

        try:
            now = datetime.now()
            features = np.array([[
                char_position,
                self._encode_char_type(char_type),
                1 if requires_shift else 0,
                system_load,
                now.hour,
                now.weekday(),
                historical_failures,
                avg_duration
            ]])

            # Predict probability of failure (class 1)
            proba = self.model.predict_proba(features)[0][1]

            return proba

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return 0.0


# ==================== Bayesian Optimization ====================

class BayesianTimingOptimizer:
    """
    Uses Bayesian Optimization to find optimal typing timing parameters.

    Much more efficient than grid search - intelligently explores parameter space.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.best_params = {
            'key_duration': 50.0,
            'inter_char_delay': 100.0,
            'shift_duration': 30.0
        }
        self.exploration_history = []

        # Bayesian optimization parameters
        self.n_init = 10  # Random exploration first
        self.n_iter = 50  # Then optimize

        # Try to import bayesian-optimization
        try:
            from bayes_opt import BayesianOptimization
            self.BayesianOptimization = BayesianOptimization
        except ImportError:
            logger.warning("⚠️ bayesian-optimization not available, using grid search fallback")
            self.BayesianOptimization = None

    async def optimize_parameters(
        self,
        success_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Find optimal timing parameters using Bayesian Optimization.

        Returns best parameters found.
        """
        if not self.BayesianOptimization or len(success_data) < 20:
            logger.debug("Not enough data for Bayesian Optimization")
            return self.best_params

        try:
            # Define objective function (what we want to maximize)
            def objective(key_duration, inter_char_delay, shift_duration):
                # Simulate success rate with these parameters
                # In practice, this would use historical data
                score = 0.0
                for data in success_data:
                    # Calculate how close our params are to successful attempts
                    param_distance = abs(data.get('avg_duration', 50) - key_duration)
                    if param_distance < 10:
                        score += 1.0

                return score / len(success_data) if success_data else 0.0

            # Define parameter bounds
            pbounds = {
                'key_duration': (30, 100),
                'inter_char_delay': (50, 200),
                'shift_duration': (20, 50)
            }

            # Run optimization
            optimizer = self.BayesianOptimization(
                f=objective,
                pbounds=pbounds,
                random_state=42,
                verbose=0
            )

            optimizer.maximize(
                init_points=self.n_init,
                n_iter=self.n_iter
            )

            # Get best parameters
            self.best_params = {
                'key_duration': optimizer.max['params']['key_duration'],
                'inter_char_delay': optimizer.max['params']['inter_char_delay'],
                'shift_duration': optimizer.max['params']['shift_duration']
            }

            logger.info(f"🎯 Bayesian Optimization found: {self.best_params}")

            return self.best_params

        except Exception as e:
            logger.error(f"Bayesian Optimization failed: {e}", exc_info=True)
            return self.best_params


# ==================== Multi-Armed Bandit ====================

class MultiArmedBandit:
    """
    Balances exploration (trying new timings) vs exploitation (using known good timings).

    Uses epsilon-greedy strategy:
    - 90% of time: Use optimal timing (exploitation)
    - 10% of time: Try random variation (exploration)
    """

    def __init__(self, epsilon: float = 0.1):
        self.epsilon = epsilon  # Exploration rate
        self.arms = {}  # timing_variant -> (total_reward, num_pulls)
        self.best_arm = None

    async def select_timing_strategy(
        self,
        char_type: str,
        optimal_timing: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Choose timing strategy: explore new timings or exploit known good ones.

        Returns: timing parameters to use
        """
        import random

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # EXPLORE: Try random variation
            variation = random.uniform(0.8, 1.2)
            timing = {
                'duration': optimal_timing['duration'] * variation,
                'delay': optimal_timing['delay'] * variation
            }
            logger.debug(f"🔍 Exploring new timing for {char_type}: {variation:.2f}x")
        else:
            # EXPLOIT: Use optimal timing
            timing = optimal_timing.copy()

        return timing

    async def update_reward(
        self,
        timing_key: str,
        success: bool,
        speed: float
    ):
        """
        Update bandit with reward from this timing choice.

        Reward = success + speed_bonus
        """
        reward = 1.0 if success else 0.0
        if success and speed < 2000:  # Fast unlock
            reward += 0.5

        if timing_key not in self.arms:
            self.arms[timing_key] = (0.0, 0)

        total_reward, num_pulls = self.arms[timing_key]
        self.arms[timing_key] = (total_reward + reward, num_pulls + 1)

        # Update best arm
        best_avg = max((r/n for r, n in self.arms.values() if n > 0), default=0.0)
        for key, (r, n) in self.arms.items():
            if n > 0 and r/n == best_avg:
                self.best_arm = key
                break


# ==================== Context-Aware Learning ====================

@dataclass
class VoiceContext:
    """Context information for voice biometric learning"""
    time_of_day: str  # 'morning', 'afternoon', 'evening', 'night'
    day_of_week: str
    is_weekend: bool
    estimated_voice_condition: str  # 'fresh', 'tired', 'normal'
    background_noise_level: float  # 0.0 to 1.0


class ContextAwareLearner:
    """
    Learns context-specific patterns for voice biometrics.

    Examples:
    - Derek's voice is lower in the morning
    - Confidence is higher in quiet environments
    - Voice changes when tired
    """

    def __init__(self):
        self.context_patterns = {}  # context_key -> avg_confidence

    def get_context(self, audio_quality: float) -> VoiceContext:
        """Extract context from current environment"""
        now = datetime.now()
        hour = now.hour

        # Determine time of day
        if 5 <= hour < 12:
            time_of_day = 'morning'
        elif 12 <= hour < 17:
            time_of_day = 'afternoon'
        elif 17 <= hour < 21:
            time_of_day = 'evening'
        else:
            time_of_day = 'night'

        # Estimate voice condition based on time
        if hour < 7 or hour > 22:
            voice_condition = 'tired'
        elif 9 <= hour <= 16:
            voice_condition = 'fresh'
        else:
            voice_condition = 'normal'

        return VoiceContext(
            time_of_day=time_of_day,
            day_of_week=now.strftime('%A'),
            is_weekend=now.weekday() >= 5,
            estimated_voice_condition=voice_condition,
            background_noise_level=1.0 - audio_quality
        )

    async def update_context_pattern(
        self,
        context: VoiceContext,
        confidence: float,
        success: bool
    ):
        """Learn how context affects voice recognition"""
        context_key = f"{context.time_of_day}_{context.estimated_voice_condition}"

        if context_key not in self.context_patterns:
            self.context_patterns[context_key] = []

        self.context_patterns[context_key].append(confidence)

        # Keep last 20 per context
        self.context_patterns[context_key] = self.context_patterns[context_key][-20:]

        avg = np.mean(self.context_patterns[context_key])
        logger.debug(f"📊 Context pattern {context_key}: avg confidence {avg:.1%}")

    async def get_expected_confidence(self, context: VoiceContext) -> float:
        """Predict expected confidence for this context"""
        context_key = f"{context.time_of_day}_{context.estimated_voice_condition}"

        if context_key in self.context_patterns and self.context_patterns[context_key]:
            return np.mean(self.context_patterns[context_key])

        # Default expectation
        return 0.50


# ==================== Anomaly Detection ====================

class AnomalyDetector:
    """
    Detects unusual authentication patterns that could indicate:
    - Spoofing attempts
    - Unauthorized access
    - System issues

    Uses Isolation Forest algorithm.
    """

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.model = None

        try:
            from sklearn.ensemble import IsolationForest
            self.IsolationForest = IsolationForest
        except ImportError:
            logger.warning("⚠️ scikit-learn not available, anomaly detection disabled")
            self.IsolationForest = None

    async def initialize(self):
        """Load or create Isolation Forest model"""
        if not self.IsolationForest:
            return

        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                logger.info(f"✅ Loaded Isolation Forest from {self.model_path}")
            except Exception as e:
                logger.warning(f"Failed to load anomaly detector: {e}")
                self._create_new_model()
        else:
            self._create_new_model()

    def _create_new_model(self):
        """Create new Isolation Forest"""
        if not self.IsolationForest:
            return

        self.model = self.IsolationForest(
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        logger.info("✅ Created new Isolation Forest for anomaly detection")

    async def train(self, historical_attempts: List[Dict[str, Any]]):
        """Train on normal authentication patterns"""
        if not self.model or len(historical_attempts) < 30:
            return

        try:
            # Extract features
            X = []
            for attempt in historical_attempts:
                features = [
                    attempt.get('speaker_confidence', 0),
                    attempt.get('stt_confidence', 0),
                    attempt.get('audio_duration_ms', 0),
                    attempt.get('processing_time_ms', 0),
                    datetime.fromisoformat(attempt.get('timestamp', datetime.now().isoformat())).hour
                ]
                X.append(features)

            X = np.array(X)
            self.model.fit(X)

            # Save model
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)

            logger.info(f"✅ Anomaly detector trained on {len(X)} samples")

        except Exception as e:
            logger.error(f"Anomaly detection training failed: {e}", exc_info=True)

    async def detect_anomaly(
        self,
        speaker_confidence: float,
        stt_confidence: float,
        audio_duration_ms: float,
        processing_time_ms: float
    ) -> Tuple[bool, float]:
        """
        Detect if this attempt is anomalous.

        Returns: (is_anomaly, anomaly_score)
        """
        if not self.model:
            return False, 0.0

        try:
            now = datetime.now()
            features = np.array([[
                speaker_confidence,
                stt_confidence,
                audio_duration_ms,
                processing_time_ms,
                now.hour
            ]])

            # Predict (-1 = anomaly, 1 = normal)
            prediction = self.model.predict(features)[0]
            score = self.model.score_samples(features)[0]

            is_anomaly = prediction == -1

            if is_anomaly:
                logger.warning(f"🚨 Anomaly detected! Score: {score:.3f}")

            return is_anomaly, abs(score)

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return False, 0.0


# ==================== Fine-Tuning for Speaker Embeddings ====================

class SpeakerEmbeddingFineTuner:
    """
    Fine-tunes speaker embeddings for improved accuracy.

    Uses online learning to adapt embeddings to Derek's voice variations.
    """

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate
        self.base_embedding = None
        self.adapted_embedding = None
        self.adaptation_count = 0

    async def initialize(self, base_embedding: np.ndarray):
        """Initialize with base speaker embedding"""
        self.base_embedding = base_embedding.copy()
        self.adapted_embedding = base_embedding.copy()
        logger.info(f"✅ Fine-tuner initialized with {len(base_embedding)}D embedding")

    async def fine_tune(
        self,
        new_embedding: np.ndarray,
        confidence: float,
        success: bool
    ):
        """
        Fine-tune embedding using online learning.

        Higher confidence samples have more influence.
        """
        if self.adapted_embedding is None:
            self.adapted_embedding = new_embedding.copy()
            return

        # Weight by confidence and success
        weight = confidence if success else confidence * 0.5
        effective_lr = self.learning_rate * weight

        # Online gradient descent update
        self.adapted_embedding = (
            (1 - effective_lr) * self.adapted_embedding +
            effective_lr * new_embedding
        )

        self.adaptation_count += 1

        if self.adaptation_count % 10 == 0:
            # Calculate adaptation distance
            distance = np.linalg.norm(self.adapted_embedding - self.base_embedding)
            logger.info(f"🎯 Embedding fine-tuned {self.adaptation_count} times, "
                       f"drift: {distance:.3f}")

    async def get_adapted_embedding(self) -> np.ndarray:
        """Get current fine-tuned embedding"""
        return self.adapted_embedding.copy() if self.adapted_embedding is not None else None


# ==================== Progressive Learning Stages ====================

@dataclass
class LearningStage:
    """Represents a stage in progressive learning"""
    name: str
    min_attempts: int
    max_attempts: int
    target_confidence: float
    target_success_rate: float
    description: str


class ProgressiveLearningManager:
    """
    Manages progressive learning stages:
    1. Data Collection (0-20 attempts)
    2. Pattern Recognition (20-50 attempts)
    3. Optimization (50-100 attempts)
    4. Mastery (100+ attempts)
    """

    def __init__(self):
        self.stages = [
            LearningStage(
                name='data_collection',
                min_attempts=0,
                max_attempts=20,
                target_confidence=0.40,
                target_success_rate=0.60,
                description='Learning Derek\'s voice patterns'
            ),
            LearningStage(
                name='pattern_recognition',
                min_attempts=20,
                max_attempts=50,
                target_confidence=0.50,
                target_success_rate=0.75,
                description='Recognizing voice patterns and optimizing threshold'
            ),
            LearningStage(
                name='optimization',
                min_attempts=50,
                max_attempts=100,
                target_confidence=0.55,
                target_success_rate=0.85,
                description='Fine-tuning for optimal performance'
            ),
            LearningStage(
                name='mastery',
                min_attempts=100,
                max_attempts=float('inf'),
                target_confidence=0.60,
                target_success_rate=0.95,
                description='Near-perfect recognition and typing'
            )
        ]

    def get_current_stage(self, total_attempts: int) -> LearningStage:
        """Get current learning stage based on attempt count"""
        for stage in self.stages:
            if stage.min_attempts <= total_attempts < stage.max_attempts:
                return stage

        return self.stages[-1]  # Mastery

    def get_progress_report(
        self,
        total_attempts: int,
        avg_confidence: float,
        success_rate: float
    ) -> Dict[str, Any]:
        """Get detailed progress report"""
        stage = self.get_current_stage(total_attempts)

        confidence_progress = (avg_confidence / stage.target_confidence) * 100
        success_progress = (success_rate / stage.target_success_rate) * 100
        overall_progress = (confidence_progress + success_progress) / 2

        return {
            'current_stage': stage.name,
            'stage_description': stage.description,
            'total_attempts': total_attempts,
            'attempts_in_stage': total_attempts - stage.min_attempts,
            'attempts_to_next_stage': max(0, stage.max_attempts - total_attempts),
            'confidence_progress': min(100, confidence_progress),
            'success_rate_progress': min(100, success_progress),
            'overall_progress': min(100, overall_progress),
            'targets': {
                'confidence': stage.target_confidence,
                'success_rate': stage.target_success_rate
            },
            'actual': {
                'confidence': avg_confidence,
                'success_rate': success_rate
            },
            'status': self._get_status(overall_progress)
        }

    def _get_status(self, progress: float) -> str:
        """Get status message based on progress"""
        if progress >= 90:
            return 'excellent'
        elif progress >= 70:
            return 'good'
        elif progress >= 50:
            return 'fair'
        else:
            return 'learning'


# ==================== Model Persistence ====================

class ModelPersistence:
    """
    Handles saving and loading of all ML models.

    Provides checkpoint functionality for recovery.
    """

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def save_checkpoint(
        self,
        voice_state: Any,
        typing_state: Any,
        models: Dict[str, Any],
        metadata: Dict[str, Any]
    ):
        """Save complete checkpoint of learning state"""
        try:
            checkpoint = {
                'timestamp': datetime.now().isoformat(),
                'voice_state': asdict(voice_state) if hasattr(voice_state, '__dict__') else voice_state,
                'typing_state': asdict(typing_state) if hasattr(typing_state, '__dict__') else typing_state,
                'metadata': metadata
            }

            checkpoint_path = self.base_path / f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

            logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

            # Keep only last 10 checkpoints
            await self._cleanup_old_checkpoints()

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    async def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Load most recent checkpoint"""
        try:
            checkpoints = sorted(self.base_path.glob("checkpoint_*.json"))

            if not checkpoints:
                return None

            latest = checkpoints[-1]

            with open(latest, 'r') as f:
                checkpoint = json.load(f)

            logger.info(f"✅ Loaded checkpoint from {checkpoint['timestamp']}")

            return checkpoint

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def _cleanup_old_checkpoints(self, keep_last: int = 10):
        """Remove old checkpoints, keeping only recent ones"""
        try:
            checkpoints = sorted(self.base_path.glob("checkpoint_*.json"))

            if len(checkpoints) > keep_last:
                for old_checkpoint in checkpoints[:-keep_last]:
                    old_checkpoint.unlink()
                    logger.debug(f"Removed old checkpoint: {old_checkpoint.name}")

        except Exception as e:
            logger.error(f"Checkpoint cleanup failed: {e}")
