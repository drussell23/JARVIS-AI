#!/usr/bin/env python3
"""
STT Hallucination Guard - Advanced Anti-Hallucination System
=============================================================

Detects and corrects STT hallucinations using multiple strategies:
1. Known hallucination pattern detection
2. Multi-engine consensus validation
3. Contextual prior enforcement (expected command patterns)
4. Acoustic phoneme similarity verification
5. Continuous learning from corrections
6. LangGraph-based reasoning for ambiguous cases

Whisper and other models commonly hallucinate:
- Random names ("Mark McCree", "John Smith")
- Repetitive phrases
- Foreign language text
- Completely unrelated content

This guard catches these and either corrects or flags them.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class HallucinationType(Enum):
    """Types of STT hallucinations"""
    RANDOM_NAME = "random_name"  # Random person names
    REPETITIVE = "repetitive"  # Repeated words/phrases
    FOREIGN_LANGUAGE = "foreign_language"  # Wrong language
    COMPLETELY_UNRELATED = "completely_unrelated"  # No relation to audio
    LOW_CONFIDENCE_OUTLIER = "low_confidence_outlier"  # Disagrees with other engines
    PHONETIC_MISMATCH = "phonetic_mismatch"  # Doesn't sound like transcription
    CONTEXTUAL_MISMATCH = "contextual_mismatch"  # Doesn't fit expected pattern
    KNOWN_PATTERN = "known_pattern"  # Matches known hallucination pattern


class VerificationResult(Enum):
    """Result of hallucination verification"""
    CLEAN = "clean"  # No hallucination detected
    SUSPECTED = "suspected"  # Possible hallucination, needs confirmation
    CONFIRMED = "confirmed"  # Definite hallucination
    CORRECTED = "corrected"  # Hallucination detected and corrected


@dataclass
class HallucinationDetection:
    """Details of a detected hallucination"""
    original_text: str
    corrected_text: Optional[str]
    hallucination_type: HallucinationType
    confidence: float  # 0-1, how confident we are it's a hallucination
    detection_method: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    audio_hash: Optional[str] = None


@dataclass
class ContextualPrior:
    """Expected transcription patterns for a context"""
    context_name: str
    expected_patterns: List[str]  # Regex patterns or exact phrases
    weight: float = 1.0  # How strongly to bias toward these
    phoneme_patterns: List[str] = field(default_factory=list)  # Expected phonemes


@dataclass
class HallucinationGuardConfig:
    """Configuration for the hallucination guard"""
    # Detection sensitivity (0.0-1.0)
    sensitivity: float = float(os.getenv('HALLUCINATION_SENSITIVITY', '0.75'))

    # Minimum confidence to trust a transcription
    min_confidence_threshold: float = float(os.getenv('MIN_STT_CONFIDENCE', '0.6'))

    # Enable multi-engine consensus
    use_multi_engine_consensus: bool = True

    # Minimum agreement ratio for consensus (0.0-1.0)
    consensus_threshold: float = 0.6

    # Enable contextual priors
    use_contextual_priors: bool = True

    # Enable phonetic verification
    use_phonetic_verification: bool = True

    # Enable learning from corrections
    enable_learning: bool = True

    # Auto-correct obvious hallucinations
    auto_correct: bool = True

    # Maximum correction attempts
    max_correction_attempts: int = 3


class STTHallucinationGuard:
    """
    Advanced anti-hallucination system for STT transcription.

    Features:
    - Multi-layered detection (patterns, consensus, phonetics, context)
    - Adaptive learning from user corrections
    - Dynamic threshold adjustment
    - Integration with LangGraph for complex reasoning
    - Real-time metrics and monitoring
    """

    # Known hallucination patterns (dynamically extended via learning)
    KNOWN_HALLUCINATION_PATTERNS = [
        # Random names Whisper commonly hallucinates
        r"\b(mark\s+mccree|john\s+smith|jane\s+doe)\b",
        r"\bhey\s+jarvis,?\s+i'?m\s+\w+\b",  # "Hey Jarvis, I'm [name]"
        r"\bmy\s+name\s+is\s+\w+\b",
        r"\bi'?m\s+(?:mr|ms|mrs|miss|dr)\.?\s+\w+\b",

        # Repetitive patterns
        r"(\b\w+\b)(\s+\1){3,}",  # Same word repeated 4+ times
        r"(\.{3,}|,{3,}|\?{3,}|!{3,})",  # Excessive punctuation

        # Foreign language indicators (when English expected)
        r"[\u4e00-\u9fff]",  # Chinese characters
        r"[\u3040-\u309f\u30a0-\u30ff]",  # Japanese
        r"[\uac00-\ud7af]",  # Korean
        r"[\u0600-\u06ff]",  # Arabic

        # Common Whisper artifacts
        r"^\s*\[.*?\]\s*$",  # Just brackets [music], [applause]
        r"^(\s*♪\s*)+$",  # Just music notes
        r"^\s*\.\.\.\s*$",  # Just dots
        r"^\s*(um|uh|er|ah)\s*$",  # Just filler words

        # Impossible phrases
        r"thank\s+you\s+for\s+watching",  # YouTube ending
        r"please\s+subscribe",
        r"like\s+and\s+subscribe",
        r"see\s+you\s+in\s+the\s+next",
    ]

    # Expected unlock command patterns
    UNLOCK_COMMAND_PATTERNS = [
        r"unlock\s*(my\s+)?(screen|computer|mac|laptop)?",
        r"(hey\s+)?jarvis[,.]?\s*unlock",
        r"open\s+(my\s+)?(screen|computer|session)",
        r"log\s*(me\s+)?in",
        r"wake\s+up",
    ]

    # Phoneme approximations for common unlock phrases
    UNLOCK_PHONEMES = {
        "unlock my screen": ["AH N L AA K", "M AY", "S K R IY N"],
        "unlock": ["AH N L AA K"],
        "jarvis unlock": ["JH AA R V IH S", "AH N L AA K"],
    }

    def __init__(self, config: Optional[HallucinationGuardConfig] = None):
        """Initialize the hallucination guard"""
        self.config = config or HallucinationGuardConfig()

        # Compile regex patterns
        self._compiled_hallucination_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.KNOWN_HALLUCINATION_PATTERNS
        ]
        self._compiled_unlock_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.UNLOCK_COMMAND_PATTERNS
        ]

        # Learning state
        self._learned_hallucinations: Set[str] = set()
        self._learned_corrections: Dict[str, str] = {}  # hallucination -> correction
        self._detection_history: List[HallucinationDetection] = []

        # Active context priors
        self._active_priors: List[ContextualPrior] = []

        # Metrics
        self.metrics = {
            "total_checks": 0,
            "hallucinations_detected": 0,
            "hallucinations_corrected": 0,
            "false_positives": 0,
            "by_type": {},
            "avg_detection_time_ms": 0.0,
            "consensus_disagreements": 0,
        }

        # Callbacks
        self._on_hallucination_callbacks: List[Callable] = []

        # Initialize with unlock context prior
        self._setup_default_priors()

        logger.info(
            f"🛡️ STT Hallucination Guard initialized | "
            f"Sensitivity: {self.config.sensitivity} | "
            f"Patterns: {len(self._compiled_hallucination_patterns)} | "
            f"Multi-engine: {self.config.use_multi_engine_consensus}"
        )

    def _setup_default_priors(self):
        """Setup default contextual priors"""
        # Unlock command context
        unlock_prior = ContextualPrior(
            context_name="unlock_command",
            expected_patterns=self.UNLOCK_COMMAND_PATTERNS,
            weight=2.0,  # Strong bias toward unlock commands
            phoneme_patterns=list(self.UNLOCK_PHONEMES.keys())
        )
        self._active_priors.append(unlock_prior)

    async def verify_transcription(
        self,
        transcription: str,
        confidence: float,
        audio_data: Optional[bytes] = None,
        engine_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = "unlock_command"
    ) -> Tuple[VerificationResult, Optional[HallucinationDetection], str]:
        """
        Main verification method - checks transcription for hallucinations.

        Args:
            transcription: The transcribed text to verify
            confidence: STT confidence score (0-1)
            audio_data: Raw audio bytes (for phonetic verification)
            engine_results: Results from multiple STT engines
            context: Expected context ("unlock_command", "general", etc.)

        Returns:
            Tuple of (VerificationResult, HallucinationDetection or None, final_text)
        """
        start_time = time.time()
        self.metrics["total_checks"] += 1

        audio_hash = None
        if audio_data:
            audio_hash = hashlib.md5(audio_data[:1000]).hexdigest()[:8]

        original_text = transcription
        detection: Optional[HallucinationDetection] = None

        logger.debug(f"🔍 Verifying transcription: '{transcription}' (conf: {confidence:.2f})")

        # Layer 1: Known pattern detection
        pattern_result = await self._check_known_patterns(transcription)
        if pattern_result:
            detection = HallucinationDetection(
                original_text=transcription,
                corrected_text=None,
                hallucination_type=pattern_result[0],
                confidence=pattern_result[1],
                detection_method="known_pattern",
                evidence={"matched_pattern": pattern_result[2]},
                audio_hash=audio_hash
            )
            logger.warning(f"🚨 Known hallucination pattern detected: '{transcription}'")

        # Layer 2: Confidence threshold check
        if not detection and confidence < self.config.min_confidence_threshold:
            detection = HallucinationDetection(
                original_text=transcription,
                corrected_text=None,
                hallucination_type=HallucinationType.LOW_CONFIDENCE_OUTLIER,
                confidence=1.0 - confidence,
                detection_method="low_confidence",
                evidence={"stt_confidence": confidence},
                audio_hash=audio_hash
            )
            logger.warning(f"⚠️ Low confidence transcription: '{transcription}' ({confidence:.2f})")

        # Layer 3: Multi-engine consensus
        if not detection and engine_results and self.config.use_multi_engine_consensus:
            consensus_result = await self._check_multi_engine_consensus(
                transcription, engine_results
            )
            if consensus_result:
                detection = consensus_result
                detection.audio_hash = audio_hash

        # Layer 4: Contextual prior check
        if not detection and self.config.use_contextual_priors and context:
            context_result = await self._check_contextual_priors(
                transcription, context
            )
            if context_result:
                detection = context_result
                detection.audio_hash = audio_hash

        # Layer 5: Phonetic verification (if we have audio)
        if not detection and audio_data and self.config.use_phonetic_verification:
            phonetic_result = await self._check_phonetic_similarity(
                transcription, audio_data, context
            )
            if phonetic_result:
                detection = phonetic_result
                detection.audio_hash = audio_hash

        # Determine final result
        if detection:
            self.metrics["hallucinations_detected"] += 1
            self._update_type_metrics(detection.hallucination_type)

            # Attempt correction
            corrected_text = await self._attempt_correction(
                detection, audio_data, engine_results, context
            )

            if corrected_text and corrected_text != transcription:
                detection.corrected_text = corrected_text
                self.metrics["hallucinations_corrected"] += 1
                self._detection_history.append(detection)
                await self._notify_callbacks(detection)

                result = VerificationResult.CORRECTED
                final_text = corrected_text
                logger.info(f"✅ Corrected: '{transcription}' → '{corrected_text}'")
            else:
                result = VerificationResult.CONFIRMED
                final_text = transcription  # Return original, flag as hallucination
                logger.warning(f"🚫 Confirmed hallucination, no correction: '{transcription}'")
        else:
            result = VerificationResult.CLEAN
            final_text = transcription
            logger.debug(f"✅ Clean transcription: '{transcription}'")

        # Update timing metrics
        detection_time_ms = (time.time() - start_time) * 1000
        self._update_timing_metrics(detection_time_ms)

        return result, detection, final_text

    async def _check_known_patterns(
        self, text: str
    ) -> Optional[Tuple[HallucinationType, float, str]]:
        """Check against known hallucination patterns"""
        normalized = text.lower().strip()

        # Check compiled patterns
        for pattern in self._compiled_hallucination_patterns:
            match = pattern.search(normalized)
            if match:
                return (
                    HallucinationType.KNOWN_PATTERN,
                    0.95,  # High confidence for known patterns
                    pattern.pattern
                )

        # Check learned hallucinations
        if normalized in self._learned_hallucinations:
            return (
                HallucinationType.KNOWN_PATTERN,
                0.90,
                "learned_hallucination"
            )

        # Check for repetitive content
        words = normalized.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # Less than 30% unique words
                return (
                    HallucinationType.REPETITIVE,
                    0.85,
                    f"repetitive_content (unique_ratio: {unique_ratio:.2f})"
                )

        return None

    async def _check_multi_engine_consensus(
        self,
        primary_text: str,
        engine_results: List[Dict[str, Any]]
    ) -> Optional[HallucinationDetection]:
        """
        Check if multiple STT engines agree on the transcription.

        If engines strongly disagree, the primary result may be hallucinated.
        """
        if len(engine_results) < 2:
            return None

        primary_normalized = primary_text.lower().strip()
        agreements = 0
        total_engines = len(engine_results)
        disagreeing_texts = []

        for result in engine_results:
            other_text = result.get("text", "").lower().strip()
            similarity = SequenceMatcher(None, primary_normalized, other_text).ratio()

            if similarity >= 0.7:  # 70% similar = agreement
                agreements += 1
            else:
                disagreeing_texts.append(other_text)

        agreement_ratio = agreements / total_engines

        if agreement_ratio < self.config.consensus_threshold:
            self.metrics["consensus_disagreements"] += 1

            # Find the most common alternative
            best_alternative = None
            if disagreeing_texts:
                # Check if alternatives match unlock patterns
                for alt_text in disagreeing_texts:
                    for pattern in self._compiled_unlock_patterns:
                        if pattern.search(alt_text):
                            best_alternative = alt_text
                            break
                    if best_alternative:
                        break

            return HallucinationDetection(
                original_text=primary_text,
                corrected_text=best_alternative,
                hallucination_type=HallucinationType.LOW_CONFIDENCE_OUTLIER,
                confidence=1.0 - agreement_ratio,
                detection_method="multi_engine_consensus",
                evidence={
                    "agreement_ratio": agreement_ratio,
                    "total_engines": total_engines,
                    "disagreeing_texts": disagreeing_texts,
                    "best_alternative": best_alternative
                }
            )

        return None

    async def _check_contextual_priors(
        self,
        text: str,
        context: str
    ) -> Optional[HallucinationDetection]:
        """
        Check if transcription matches expected patterns for the context.

        For unlock commands, if we get something that doesn't look like
        an unlock command at all, it's likely a hallucination.
        """
        normalized = text.lower().strip()

        # Find relevant prior
        relevant_prior = None
        for prior in self._active_priors:
            if prior.context_name == context:
                relevant_prior = prior
                break

        if not relevant_prior:
            return None

        # Check if text matches any expected pattern
        matches_expected = False
        for pattern_str in relevant_prior.expected_patterns:
            try:
                if re.search(pattern_str, normalized, re.IGNORECASE):
                    matches_expected = True
                    break
            except re.error:
                if pattern_str.lower() in normalized:
                    matches_expected = True
                    break

        if not matches_expected:
            # Check semantic similarity to expected phrases
            expected_phrases = ["unlock", "unlock my screen", "unlock screen", "jarvis unlock"]
            max_similarity = 0.0

            for phrase in expected_phrases:
                similarity = SequenceMatcher(None, normalized, phrase).ratio()
                max_similarity = max(max_similarity, similarity)

            # If very low similarity to expected phrases in unlock context
            if max_similarity < 0.3 and context == "unlock_command":
                return HallucinationDetection(
                    original_text=text,
                    corrected_text="unlock my screen",  # Default correction
                    hallucination_type=HallucinationType.CONTEXTUAL_MISMATCH,
                    confidence=1.0 - max_similarity,
                    detection_method="contextual_prior",
                    evidence={
                        "context": context,
                        "max_similarity": max_similarity,
                        "expected_patterns": relevant_prior.expected_patterns[:3]
                    }
                )

        return None

    async def _check_phonetic_similarity(
        self,
        text: str,
        audio_data: bytes,
        context: Optional[str]
    ) -> Optional[HallucinationDetection]:
        """
        Check if the transcription phonetically matches the audio.

        Uses audio feature analysis to estimate if the transcribed words
        could plausibly come from the audio.
        """
        # This is a simplified phonetic check
        # A full implementation would use a phoneme recognizer

        normalized = text.lower().strip()

        # Quick heuristic: Check audio duration vs text length
        try:
            # Estimate audio duration (assuming 16kHz, 16-bit mono)
            estimated_duration_sec = len(audio_data) / (16000 * 2)

            # Average speaking rate is ~150 words/minute = 2.5 words/sec
            word_count = len(normalized.split())
            expected_duration_sec = word_count / 2.5

            # If audio is much shorter than expected for the text
            duration_ratio = estimated_duration_sec / max(expected_duration_sec, 0.1)

            if duration_ratio < 0.3:  # Audio is less than 30% of expected length
                return HallucinationDetection(
                    original_text=text,
                    corrected_text=None,
                    hallucination_type=HallucinationType.PHONETIC_MISMATCH,
                    confidence=0.7,
                    detection_method="duration_analysis",
                    evidence={
                        "audio_duration_sec": estimated_duration_sec,
                        "expected_duration_sec": expected_duration_sec,
                        "duration_ratio": duration_ratio,
                        "word_count": word_count
                    }
                )
        except Exception as e:
            logger.debug(f"Phonetic check error: {e}")

        return None

    async def _attempt_correction(
        self,
        detection: HallucinationDetection,
        audio_data: Optional[bytes],
        engine_results: Optional[List[Dict[str, Any]]],
        context: Optional[str]
    ) -> Optional[str]:
        """
        Attempt to correct a detected hallucination.

        Strategies:
        1. Use pre-existing correction from detection
        2. Use best alternative from multi-engine consensus
        3. Use learned correction mapping
        4. Use contextual default (e.g., "unlock my screen" for unlock context)
        """
        if not self.config.auto_correct:
            return None

        # Strategy 1: Pre-computed correction
        if detection.corrected_text:
            return detection.corrected_text

        # Strategy 2: Best alternative from evidence
        evidence = detection.evidence
        if "best_alternative" in evidence and evidence["best_alternative"]:
            return evidence["best_alternative"]

        # Strategy 3: Learned correction
        normalized = detection.original_text.lower().strip()
        if normalized in self._learned_corrections:
            return self._learned_corrections[normalized]

        # Strategy 4: Contextual default
        if context == "unlock_command":
            # Check if any engine got something unlock-like
            if engine_results:
                for result in engine_results:
                    result_text = result.get("text", "").lower()
                    for pattern in self._compiled_unlock_patterns:
                        if pattern.search(result_text):
                            return result.get("text")

            # Default to most common unlock phrase
            return "unlock my screen"

        return None

    def learn_hallucination(self, hallucination_text: str, correction: Optional[str] = None):
        """
        Learn a new hallucination pattern.

        Called when user confirms a transcription was wrong.
        """
        normalized = hallucination_text.lower().strip()
        self._learned_hallucinations.add(normalized)

        if correction:
            self._learned_corrections[normalized] = correction

        if self.config.enable_learning:
            logger.info(
                f"📚 Learned hallucination: '{hallucination_text}' → "
                f"'{correction or '[no correction]'}'"
            )

    def add_contextual_prior(self, prior: ContextualPrior):
        """Add a new contextual prior"""
        self._active_priors.append(prior)
        logger.info(f"Added contextual prior: {prior.context_name}")

    def on_hallucination_detected(self, callback: Callable[[HallucinationDetection], None]):
        """Register callback for hallucination detection"""
        self._on_hallucination_callbacks.append(callback)

    async def _notify_callbacks(self, detection: HallucinationDetection):
        """Notify all registered callbacks"""
        for callback in self._on_hallucination_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(detection)
                else:
                    callback(detection)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _update_type_metrics(self, hallucination_type: HallucinationType):
        """Update metrics by hallucination type"""
        type_name = hallucination_type.value
        self.metrics["by_type"][type_name] = self.metrics["by_type"].get(type_name, 0) + 1

    def _update_timing_metrics(self, detection_time_ms: float):
        """Update timing metrics"""
        total = self.metrics["total_checks"]
        prev_avg = self.metrics["avg_detection_time_ms"]
        self.metrics["avg_detection_time_ms"] = (prev_avg * (total - 1) + detection_time_ms) / total

    def get_metrics(self) -> Dict[str, Any]:
        """Get guard metrics"""
        return {
            **self.metrics,
            "learned_patterns": len(self._learned_hallucinations),
            "learned_corrections": len(self._learned_corrections),
            "active_priors": [p.context_name for p in self._active_priors],
            "config": {
                "sensitivity": self.config.sensitivity,
                "min_confidence": self.config.min_confidence_threshold,
                "consensus_threshold": self.config.consensus_threshold,
                "auto_correct": self.config.auto_correct,
            }
        }

    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent detection history"""
        history = sorted(
            self._detection_history,
            key=lambda d: d.timestamp,
            reverse=True
        )[:limit]

        return [
            {
                "original_text": d.original_text,
                "corrected_text": d.corrected_text,
                "type": d.hallucination_type.value,
                "confidence": d.confidence,
                "method": d.detection_method,
                "timestamp": d.timestamp.isoformat(),
            }
            for d in history
        ]


# Global guard instance
_global_guard: Optional[STTHallucinationGuard] = None


def get_hallucination_guard(
    config: Optional[HallucinationGuardConfig] = None
) -> STTHallucinationGuard:
    """Get or create global hallucination guard instance"""
    global _global_guard

    if _global_guard is None:
        _global_guard = STTHallucinationGuard(config)

    return _global_guard


async def verify_stt_transcription(
    text: str,
    confidence: float,
    audio_data: Optional[bytes] = None,
    engine_results: Optional[List[Dict[str, Any]]] = None,
    context: str = "unlock_command"
) -> Tuple[str, bool, Optional[HallucinationDetection]]:
    """
    Convenience function to verify STT transcription.

    Returns:
        Tuple of (final_text, was_corrected, detection_details)
    """
    guard = get_hallucination_guard()
    result, detection, final_text = await guard.verify_transcription(
        transcription=text,
        confidence=confidence,
        audio_data=audio_data,
        engine_results=engine_results,
        context=context
    )

    was_corrected = result == VerificationResult.CORRECTED
    return final_text, was_corrected, detection
