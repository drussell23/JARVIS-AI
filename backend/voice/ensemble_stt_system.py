"""
Ensemble STT System with Multi-Model Voting and Confidence-Based Auto-Retry

Enterprise-grade ensemble that runs multiple STT models in parallel and uses
intelligent voting to select the best transcription.

Features:
- Parallel multi-model inference (async)
- Weighted voting based on model confidence and historical accuracy
- Confidence-based auto-retry with escalation
- Learning from user corrections
- Adaptive model selection
- Performance tracking and optimization
"""

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .engines.base_engine import STTResult
from .stt_config import STTEngine

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from ensemble voting"""

    text: str
    confidence: float
    votes: int  # How many models agreed
    total_models: int  # How many models participated
    model_results: List[STTResult] = field(default_factory=list)
    voting_strategy: str = "weighted"
    retry_count: int = 0
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelPerformance:
    """Track model performance over time"""

    engine: STTEngine
    model_name: str
    total_inferences: int = 0
    correct_inferences: int = 0
    avg_confidence: float = 0.0
    avg_latency_ms: float = 0.0
    last_used: datetime = field(default_factory=datetime.now)


class EnsembleSTTSystem:
    """
    Advanced ensemble STT system with voting and auto-retry.

    Voting Strategies:
    1. MAJORITY: Simple majority vote
    2. WEIGHTED: Confidence-weighted voting
    3. ACCURACY: Historical accuracy-weighted
    4. ADAPTIVE: Learn optimal weights over time

    Auto-Retry Logic:
    - confidence < 0.6 → retry with better models
    - confidence < 0.8 → run ensemble if not already
    - confidence >= 0.8 → accept result
    """

    def __init__(
        self,
        stt_router,
        learning_db=None,
        confidence_threshold=0.75,
        retry_threshold=0.60,
        max_retries=2,
    ):
        """
        Initialize ensemble system

        Args:
            stt_router: HybridSTTRouter instance
            learning_db: LearningDatabase instance (optional)
            confidence_threshold: Minimum confidence for ensemble voting
            retry_threshold: Minimum confidence before retry
            max_retries: Maximum retry attempts
        """
        self.stt_router = stt_router
        self.learning_db = learning_db
        self.confidence_threshold = confidence_threshold
        self.retry_threshold = retry_threshold
        self.max_retries = max_retries

        # Performance tracking
        self.model_performance: Dict[str, ModelPerformance] = {}
        self.ensemble_history = []

        # Adaptive weights (learned over time)
        self.model_weights = defaultdict(lambda: 1.0)

        logger.info("🎯 Ensemble STT System initialized")
        logger.info(f"   Confidence threshold: {confidence_threshold}")
        logger.info(f"   Retry threshold: {retry_threshold}")
        logger.info(f"   Max retries: {max_retries}")

    async def transcribe_with_ensemble(
        self,
        audio_data: bytes,
        models: Optional[List[str]] = None,
        voting_strategy: str = "weighted",
    ) -> EnsembleResult:
        """
        Transcribe with ensemble voting

        Args:
            audio_data: Audio bytes
            models: List of model names to use (None = auto-select)
            voting_strategy: Voting strategy (majority, weighted, accuracy, adaptive)

        Returns:
            EnsembleResult with voted transcription
        """
        # Auto-select models if not provided
        if models is None:
            models = await self._select_ensemble_models(audio_data)

        logger.info(f"[ENSEMBLE] Running {len(models)} models in parallel: {models}")

        # Run all models in parallel
        tasks = []
        for model_name in models:
            task = self._transcribe_single_model(model_name, audio_data)
            tasks.append(task)

        # Gather results
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, STTResult) and r.confidence > 0]

        if not valid_results:
            logger.error("[ENSEMBLE] No valid results from ensemble")
            return EnsembleResult(
                text="",
                confidence=0.0,
                votes=0,
                total_models=len(models),
                voting_strategy=voting_strategy,
            )

        # Vote on best result
        if voting_strategy == "majority":
            voted_result = await self._vote_majority(valid_results)
        elif voting_strategy == "weighted":
            voted_result = await self._vote_weighted(valid_results)
        elif voting_strategy == "accuracy":
            voted_result = await self._vote_accuracy_weighted(valid_results)
        elif voting_strategy == "adaptive":
            voted_result = await self._vote_adaptive(valid_results)
        else:
            voted_result = await self._vote_weighted(valid_results)  # Default

        logger.info(
            f"[ENSEMBLE] Voted result: '{voted_result.text}' "
            f"(confidence={voted_result.confidence:.2f}, votes={voted_result.votes}/{voted_result.total_models})"
        )

        return voted_result

    async def transcribe_with_auto_retry(
        self,
        audio_data: bytes,
        initial_model: Optional[str] = None,
    ) -> EnsembleResult:
        """
        Transcribe with automatic retry on low confidence

        Flow:
        1. Try single model (fast)
        2. If confidence < retry_threshold → retry with ensemble
        3. If still low → escalate to better models
        4. Return best result

        Args:
            audio_data: Audio bytes
            initial_model: Initial model to try (None = auto-select)

        Returns:
            EnsembleResult with best transcription
        """
        retry_count = 0
        best_result = None

        # Attempt 1: Single model (fast path)
        if initial_model is None:
            initial_model = await self._select_fast_model()

        logger.info(f"[AUTO-RETRY] Attempt 1: {initial_model}")
        result = await self._transcribe_single_model(initial_model, audio_data)

        if result.confidence >= self.confidence_threshold:
            # Success on first try!
            logger.info(
                f"[AUTO-RETRY] ✅ Success on first try (confidence={result.confidence:.2f})"
            )
            return self._convert_to_ensemble_result([result], retry_count=0)

        best_result = result
        logger.warning(
            f"[AUTO-RETRY] Low confidence ({result.confidence:.2f}), will retry with ensemble"
        )

        # Attempt 2: Ensemble with multiple models
        retry_count += 1
        if retry_count <= self.max_retries:
            logger.info(f"[AUTO-RETRY] Attempt 2: Ensemble voting")
            ensemble_result = await self.transcribe_with_ensemble(
                audio_data,
                models=await self._select_ensemble_models(audio_data),
                voting_strategy="weighted",
            )

            if ensemble_result.confidence >= self.retry_threshold:
                logger.info(
                    f"[AUTO-RETRY] ✅ Ensemble success (confidence={ensemble_result.confidence:.2f})"
                )
                ensemble_result.retry_count = retry_count
                return ensemble_result

            if ensemble_result.confidence > best_result.confidence:
                best_result = ensemble_result

        # Attempt 3: Escalate to high-accuracy models
        retry_count += 1
        if retry_count <= self.max_retries and best_result.confidence < self.retry_threshold:
            logger.info(f"[AUTO-RETRY] Attempt 3: Escalating to high-accuracy models")
            accurate_models = await self._select_accurate_models()

            ensemble_result = await self.transcribe_with_ensemble(
                audio_data,
                models=accurate_models,
                voting_strategy="accuracy",
            )

            if ensemble_result.confidence > best_result.confidence:
                best_result = ensemble_result

        logger.info(
            f"[AUTO-RETRY] Final result: '{best_result.text if isinstance(best_result, EnsembleResult) else best_result.text}' (retries={retry_count})"
        )

        if isinstance(best_result, EnsembleResult):
            best_result.retry_count = retry_count
            return best_result
        else:
            return self._convert_to_ensemble_result([best_result], retry_count)

    async def _vote_majority(self, results: List[STTResult]) -> EnsembleResult:
        """Simple majority voting - most common transcription wins"""
        text_counts = defaultdict(list)

        for result in results:
            text_counts[result.text.lower().strip()].append(result)

        # Find most common
        winner_text, winner_results = max(text_counts.items(), key=lambda x: len(x[1]))

        # Average confidence of winning results
        avg_confidence = sum(r.confidence for r in winner_results) / len(winner_results)

        return EnsembleResult(
            text=winner_results[0].text,  # Use original casing
            confidence=avg_confidence,
            votes=len(winner_results),
            total_models=len(results),
            model_results=results,
            voting_strategy="majority",
        )

    async def _vote_weighted(self, results: List[STTResult]) -> EnsembleResult:
        """Confidence-weighted voting"""
        text_scores = defaultdict(float)
        text_results = defaultdict(list)

        for result in results:
            key = result.text.lower().strip()
            text_scores[key] += result.confidence
            text_results[key].append(result)

        # Find highest weighted score
        winner_text = max(text_scores.keys(), key=lambda x: text_scores[x])
        winner_results = text_results[winner_text]

        # Normalized confidence
        total_confidence = sum(text_scores.values())
        normalized_confidence = (
            text_scores[winner_text] / total_confidence if total_confidence > 0 else 0
        )

        return EnsembleResult(
            text=winner_results[0].text,
            confidence=normalized_confidence,
            votes=len(winner_results),
            total_models=len(results),
            model_results=results,
            voting_strategy="weighted",
        )

    async def _vote_accuracy_weighted(self, results: List[STTResult]) -> EnsembleResult:
        """Historical accuracy-weighted voting"""
        text_scores = defaultdict(float)
        text_results = defaultdict(list)

        for result in results:
            key = result.text.lower().strip()

            # Get model's historical accuracy
            perf = self.model_performance.get(result.model_name)
            accuracy_weight = (
                perf.correct_inferences / max(perf.total_inferences, 1) if perf else 0.5
            )

            # Combine confidence with accuracy
            score = result.confidence * (0.7 + 0.3 * accuracy_weight)

            text_scores[key] += score
            text_results[key].append(result)

        winner_text = max(text_scores.keys(), key=lambda x: text_scores[x])
        winner_results = text_results[winner_text]

        total_score = sum(text_scores.values())
        normalized_confidence = text_scores[winner_text] / total_score if total_score > 0 else 0

        return EnsembleResult(
            text=winner_results[0].text,
            confidence=normalized_confidence,
            votes=len(winner_results),
            total_models=len(results),
            model_results=results,
            voting_strategy="accuracy",
        )

    async def _vote_adaptive(self, results: List[STTResult]) -> EnsembleResult:
        """Adaptive voting using learned weights"""
        text_scores = defaultdict(float)
        text_results = defaultdict(list)

        for result in results:
            key = result.text.lower().strip()

            # Use learned model weight
            model_weight = self.model_weights.get(result.model_name, 1.0)

            # Combined score
            score = result.confidence * model_weight

            text_scores[key] += score
            text_results[key].append(result)

        winner_text = max(text_scores.keys(), key=lambda x: text_scores[x])
        winner_results = text_results[winner_text]

        total_score = sum(text_scores.values())
        normalized_confidence = text_scores[winner_text] / total_score if total_score > 0 else 0

        return EnsembleResult(
            text=winner_results[0].text,
            confidence=normalized_confidence,
            votes=len(winner_results),
            total_models=len(results),
            model_results=results,
            voting_strategy="adaptive",
            metadata={"model_weights": dict(self.model_weights)},
        )

    async def _transcribe_single_model(self, model_name: str, audio_data: bytes) -> STTResult:
        """Transcribe using a single model via router"""
        try:
            result = await self.stt_router.transcribe(
                audio_data,
                preferred_model=model_name,
                strategy="speed",  # Don't auto-escalate
            )
            return result
        except Exception as e:
            logger.error(f"Error transcribing with {model_name}: {e}")
            # Return empty result
            from .engines.base_engine import STTResult

            return STTResult(
                text="",
                confidence=0.0,
                engine=STTEngine.VOSK,  # Placeholder
                model_name=model_name,
                latency_ms=0,
                audio_duration_ms=0,
            )

    async def _select_ensemble_models(self, audio_data: bytes) -> List[str]:
        """Auto-select best models for ensemble based on audio characteristics"""
        # For now, use a balanced set
        # TODO: Analyze audio (length, noise, etc.) to select optimal models
        return [
            "speechbrain-asr-crdnn",  # Noise-robust
            "wav2vec2-base",  # Accurate
            "vosk-small",  # Fast fallback
        ]

    async def _select_fast_model(self) -> str:
        """Select fastest model for initial attempt"""
        return "vosk-small"

    async def _select_accurate_models(self) -> List[str]:
        """Select most accurate models for escalation"""
        return [
            "speechbrain-wav2vec2",
            "whisper-small",
            "wav2vec2-large",
        ]

    def _convert_to_ensemble_result(
        self, results: List[STTResult], retry_count: int = 0
    ) -> EnsembleResult:
        """Convert STTResult to EnsembleResult"""
        if not results:
            return EnsembleResult(text="", confidence=0.0, votes=0, total_models=0)

        return EnsembleResult(
            text=results[0].text,
            confidence=results[0].confidence,
            votes=1,
            total_models=1,
            model_results=results,
            voting_strategy="single",
            retry_count=retry_count,
        )

    async def learn_from_correction(
        self,
        audio_hash: str,
        incorrect_text: str,
        correct_text: str,
        model_results: List[STTResult],
    ):
        """
        Learn from user corrections to improve future performance

        Updates:
        - Model weights (models that were correct get higher weight)
        - Model performance tracking
        - Learning database (if available)
        """
        logger.info(f"[ENSEMBLE] Learning from correction: '{incorrect_text}' → '{correct_text}'")

        # Calculate which models were closest to correct
        for result in model_results:
            similarity = self._text_similarity(result.text, correct_text)

            # Update model performance
            if result.model_name not in self.model_performance:
                self.model_performance[result.model_name] = ModelPerformance(
                    engine=result.engine,
                    model_name=result.model_name,
                )

            perf = self.model_performance[result.model_name]
            perf.total_inferences += 1

            if similarity > 0.8:  # Close enough
                perf.correct_inferences += 1
                # Increase weight for good models
                self.model_weights[result.model_name] *= 1.05
            else:
                # Decrease weight for poor models
                self.model_weights[result.model_name] *= 0.95

            # Normalize weights to prevent unbounded growth
            self._normalize_weights()

        # Store in learning database
        if self.learning_db:
            try:
                await self.learning_db.record_stt_correction(
                    audio_hash=audio_hash,
                    incorrect_text=incorrect_text,
                    correct_text=correct_text,
                    model_results=[r.model_name for r in model_results],
                )
            except Exception as e:
                logger.error(f"Failed to record correction in learning DB: {e}")

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _normalize_weights(self):
        """Normalize all model weights to prevent unbounded growth"""
        if not self.model_weights:
            return

        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for model in self.model_weights:
                self.model_weights[model] /= total_weight
                self.model_weights[model] *= len(self.model_weights)  # Scale back up

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            "total_ensembles": len(self.ensemble_history),
            "model_performance": {
                name: {
                    "accuracy": perf.correct_inferences / max(perf.total_inferences, 1),
                    "total_inferences": perf.total_inferences,
                    "avg_confidence": perf.avg_confidence,
                    "avg_latency_ms": perf.avg_latency_ms,
                }
                for name, perf in self.model_performance.items()
            },
            "model_weights": dict(self.model_weights),
        }
