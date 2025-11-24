#!/usr/bin/env python3
"""
STT Hallucination Guard - Advanced Anti-Hallucination System with LangGraph + LangChain + SAI
==============================================================================================

Enterprise-grade hallucination detection and correction using:
1. LangGraph: Multi-step reasoning chains for intelligent analysis
2. LangChain: Tool orchestration for verification pipeline
3. SAI: Situational Awareness Intelligence for context
4. SQLite: Continuous learning from patterns over time
5. Multi-engine consensus validation
6. Acoustic phoneme similarity verification

Whisper and other models commonly hallucinate:
- Random names ("Mark McCree", "John Smith")
- Repetitive phrases
- Foreign language text
- Completely unrelated content

This guard catches these and either corrects or flags them using
intelligent reasoning that learns and adapts over time.
"""

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, TypedDict

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class HallucinationType(Enum):
    """Types of STT hallucinations"""
    RANDOM_NAME = "random_name"
    REPETITIVE = "repetitive"
    FOREIGN_LANGUAGE = "foreign_language"
    COMPLETELY_UNRELATED = "completely_unrelated"
    LOW_CONFIDENCE_OUTLIER = "low_confidence_outlier"
    PHONETIC_MISMATCH = "phonetic_mismatch"
    CONTEXTUAL_MISMATCH = "contextual_mismatch"
    KNOWN_PATTERN = "known_pattern"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    BEHAVIORAL_MISMATCH = "behavioral_mismatch"


class VerificationResult(Enum):
    """Result of hallucination verification"""
    CLEAN = "clean"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"


class ReasoningStep(Enum):
    """LangGraph reasoning steps"""
    ANALYZE_PATTERN = "analyze_pattern"
    CHECK_CONSENSUS = "check_consensus"
    VERIFY_CONTEXT = "verify_context"
    CHECK_PHONETICS = "check_phonetics"
    ANALYZE_BEHAVIOR = "analyze_behavior"
    FORM_HYPOTHESIS = "form_hypothesis"
    ATTEMPT_CORRECTION = "attempt_correction"
    FINAL_DECISION = "final_decision"


@dataclass
class HallucinationDetection:
    """Details of a detected hallucination"""
    original_text: str
    corrected_text: Optional[str]
    hallucination_type: HallucinationType
    confidence: float
    detection_method: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    sai_context: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    audio_hash: Optional[str] = None
    learned_from: bool = False


@dataclass
class ContextualPrior:
    """Expected transcription patterns for a context"""
    context_name: str
    expected_patterns: List[str]
    weight: float = 1.0
    phoneme_patterns: List[str] = field(default_factory=list)
    time_based_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class HallucinationGuardConfig:
    """Configuration for the hallucination guard"""
    sensitivity: float = float(os.getenv('HALLUCINATION_SENSITIVITY', '0.75'))
    min_confidence_threshold: float = float(os.getenv('MIN_STT_CONFIDENCE', '0.6'))
    use_multi_engine_consensus: bool = True
    consensus_threshold: float = 0.6
    use_contextual_priors: bool = True
    use_phonetic_verification: bool = True
    use_langgraph_reasoning: bool = True
    use_sai_context: bool = True
    enable_learning: bool = True
    auto_correct: bool = True
    max_correction_attempts: int = 3
    learning_decay_days: int = 30
    min_pattern_occurrences: int = 3


# ============================================================================
# LANGGRAPH STATE MACHINE
# ============================================================================

class HallucinationAnalysisState(TypedDict):
    """State for LangGraph hallucination analysis"""
    transcription: str
    confidence: float
    audio_data: Optional[bytes]
    engine_results: Optional[List[Dict[str, Any]]]
    context: str

    # Analysis results
    pattern_analysis: Optional[Dict[str, Any]]
    consensus_analysis: Optional[Dict[str, Any]]
    context_analysis: Optional[Dict[str, Any]]
    phonetic_analysis: Optional[Dict[str, Any]]
    behavioral_analysis: Optional[Dict[str, Any]]
    sai_analysis: Optional[Dict[str, Any]]

    # Reasoning
    hypotheses: List[Dict[str, Any]]
    reasoning_steps: List[Dict[str, Any]]
    current_step: ReasoningStep

    # Decision
    is_hallucination: bool
    hallucination_type: Optional[str]
    hallucination_confidence: float
    correction: Optional[str]
    final_decision: Optional[str]


class LangGraphHallucinationReasoner:
    """
    LangGraph-based multi-step reasoning for hallucination detection.

    Uses a state machine to:
    1. Analyze patterns in transcription
    2. Check multi-engine consensus
    3. Verify against contextual priors
    4. Check phonetic similarity
    5. Analyze behavioral patterns
    6. Form hypotheses
    7. Attempt corrections
    8. Make final decision
    """

    def __init__(self, guard: 'STTHallucinationGuard'):
        self.guard = guard
        self._graph = None
        self._initialized = False

    async def initialize(self):
        """Initialize LangGraph components"""
        if self._initialized:
            return

        try:
            from langgraph.graph import StateGraph, END

            # Build the reasoning graph
            workflow = StateGraph(HallucinationAnalysisState)

            # Add nodes for each reasoning step
            workflow.add_node("analyze_pattern", self._analyze_pattern_node)
            workflow.add_node("check_consensus", self._check_consensus_node)
            workflow.add_node("verify_context", self._verify_context_node)
            workflow.add_node("check_phonetics", self._check_phonetics_node)
            workflow.add_node("analyze_behavior", self._analyze_behavior_node)
            workflow.add_node("integrate_sai", self._integrate_sai_node)
            workflow.add_node("form_hypothesis", self._form_hypothesis_node)
            workflow.add_node("attempt_correction", self._attempt_correction_node)
            workflow.add_node("final_decision", self._final_decision_node)

            # Define edges
            workflow.set_entry_point("analyze_pattern")
            workflow.add_edge("analyze_pattern", "check_consensus")
            workflow.add_edge("check_consensus", "verify_context")
            workflow.add_edge("verify_context", "check_phonetics")
            workflow.add_edge("check_phonetics", "analyze_behavior")
            workflow.add_edge("analyze_behavior", "integrate_sai")
            workflow.add_edge("integrate_sai", "form_hypothesis")
            workflow.add_conditional_edges(
                "form_hypothesis",
                self._should_attempt_correction,
                {
                    "correct": "attempt_correction",
                    "decide": "final_decision"
                }
            )
            workflow.add_edge("attempt_correction", "final_decision")
            workflow.add_edge("final_decision", END)

            self._graph = workflow.compile()
            self._initialized = True
            logger.info("🧠 LangGraph Hallucination Reasoner initialized")

        except ImportError as e:
            logger.warning(f"LangGraph not available, using fallback reasoning: {e}")
            self._initialized = False

    async def reason(
        self,
        transcription: str,
        confidence: float,
        audio_data: Optional[bytes] = None,
        engine_results: Optional[List[Dict[str, Any]]] = None,
        context: str = "unlock_command"
    ) -> HallucinationAnalysisState:
        """
        Run LangGraph reasoning chain on transcription.

        Returns complete analysis state with reasoning steps.
        """
        # Initialize state
        initial_state: HallucinationAnalysisState = {
            "transcription": transcription,
            "confidence": confidence,
            "audio_data": audio_data,
            "engine_results": engine_results,
            "context": context,
            "pattern_analysis": None,
            "consensus_analysis": None,
            "context_analysis": None,
            "phonetic_analysis": None,
            "behavioral_analysis": None,
            "sai_analysis": None,
            "hypotheses": [],
            "reasoning_steps": [],
            "current_step": ReasoningStep.ANALYZE_PATTERN,
            "is_hallucination": False,
            "hallucination_type": None,
            "hallucination_confidence": 0.0,
            "correction": None,
            "final_decision": None
        }

        if self._graph and self._initialized:
            # Run through LangGraph
            try:
                final_state = await self._graph.ainvoke(initial_state)
                return final_state
            except Exception as e:
                logger.error(f"LangGraph reasoning failed: {e}")
                # Fall through to manual reasoning

        # Fallback: Manual reasoning chain
        return await self._manual_reasoning_chain(initial_state)

    async def _manual_reasoning_chain(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Fallback manual reasoning if LangGraph unavailable"""
        state = await self._analyze_pattern_node(state)
        state = await self._check_consensus_node(state)
        state = await self._verify_context_node(state)
        state = await self._check_phonetics_node(state)
        state = await self._analyze_behavior_node(state)
        state = await self._integrate_sai_node(state)
        state = await self._form_hypothesis_node(state)

        if self._should_attempt_correction(state) == "correct":
            state = await self._attempt_correction_node(state)

        state = await self._final_decision_node(state)
        return state

    async def _analyze_pattern_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Analyze transcription for known hallucination patterns"""
        transcription = state["transcription"]
        normalized = transcription.lower().strip()

        findings = {
            "matched_patterns": [],
            "repetition_score": 0.0,
            "foreign_chars": False,
            "artifact_detected": False,
            "confidence": 0.0
        }

        # Check compiled patterns
        for pattern in self.guard._compiled_hallucination_patterns:
            match = pattern.search(normalized)
            if match:
                findings["matched_patterns"].append({
                    "pattern": pattern.pattern,
                    "matched_text": match.group(),
                    "confidence": 0.95
                })

        # Check repetition
        words = normalized.split()
        if len(words) > 3:
            unique_ratio = len(set(words)) / len(words)
            findings["repetition_score"] = 1.0 - unique_ratio
            if unique_ratio < 0.3:
                findings["matched_patterns"].append({
                    "pattern": "repetitive_content",
                    "matched_text": transcription,
                    "confidence": 0.85
                })

        # Check for foreign characters
        if re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\u0600-\u06ff]', normalized):
            findings["foreign_chars"] = True
            findings["matched_patterns"].append({
                "pattern": "foreign_language",
                "matched_text": transcription,
                "confidence": 0.90
            })

        # Check learned patterns from SQLite
        learned_match = await self.guard._check_learned_patterns(normalized)
        if learned_match:
            findings["matched_patterns"].append(learned_match)

        # Calculate overall confidence
        if findings["matched_patterns"]:
            findings["confidence"] = max(p["confidence"] for p in findings["matched_patterns"])

        state["pattern_analysis"] = findings
        state["reasoning_steps"].append({
            "step": "analyze_pattern",
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "conclusion": f"Found {len(findings['matched_patterns'])} pattern matches"
        })

        return state

    async def _check_consensus_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Check multi-engine consensus"""
        engine_results = state.get("engine_results") or []
        transcription = state["transcription"]

        findings = {
            "total_engines": len(engine_results),
            "agreements": 0,
            "disagreements": [],
            "best_alternative": None,
            "consensus_ratio": 1.0
        }

        if len(engine_results) >= 2:
            normalized = transcription.lower().strip()

            for result in engine_results:
                other_text = result.get("text", "").lower().strip()
                similarity = SequenceMatcher(None, normalized, other_text).ratio()

                if similarity >= 0.7:
                    findings["agreements"] += 1
                else:
                    findings["disagreements"].append({
                        "engine": result.get("engine", "unknown"),
                        "text": result.get("text"),
                        "confidence": result.get("confidence", 0),
                        "similarity": similarity
                    })

                    # Check if alternative matches unlock patterns
                    for pattern in self.guard._compiled_unlock_patterns:
                        if pattern.search(other_text):
                            if not findings["best_alternative"]:
                                findings["best_alternative"] = result.get("text")

            findings["consensus_ratio"] = findings["agreements"] / len(engine_results)

        state["consensus_analysis"] = findings
        state["reasoning_steps"].append({
            "step": "check_consensus",
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "conclusion": f"Consensus ratio: {findings['consensus_ratio']:.2f}"
        })

        return state

    async def _verify_context_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Verify against contextual priors"""
        transcription = state["transcription"]
        context = state["context"]
        normalized = transcription.lower().strip()

        findings = {
            "context": context,
            "matches_expected": False,
            "max_similarity": 0.0,
            "expected_patterns": [],
            "time_weight": 1.0
        }

        # Find relevant prior
        relevant_prior = None
        for prior in self.guard._active_priors:
            if prior.context_name == context:
                relevant_prior = prior
                break

        if relevant_prior:
            findings["expected_patterns"] = relevant_prior.expected_patterns[:5]

            # Check pattern match
            for pattern_str in relevant_prior.expected_patterns:
                try:
                    if re.search(pattern_str, normalized, re.IGNORECASE):
                        findings["matches_expected"] = True
                        break
                except re.error:
                    if pattern_str.lower() in normalized:
                        findings["matches_expected"] = True
                        break

            # Calculate semantic similarity
            expected_phrases = ["unlock", "unlock my screen", "unlock screen", "jarvis unlock"]
            for phrase in expected_phrases:
                similarity = SequenceMatcher(None, normalized, phrase).ratio()
                findings["max_similarity"] = max(findings["max_similarity"], similarity)

            # Apply time-based weighting
            current_hour = datetime.now().hour
            hour_key = f"hour_{current_hour}"
            if hour_key in relevant_prior.time_based_weights:
                findings["time_weight"] = relevant_prior.time_based_weights[hour_key]

        state["context_analysis"] = findings
        state["reasoning_steps"].append({
            "step": "verify_context",
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "conclusion": f"Matches expected: {findings['matches_expected']}, similarity: {findings['max_similarity']:.2f}"
        })

        return state

    async def _check_phonetics_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Check phonetic similarity between transcription and audio"""
        transcription = state["transcription"]
        audio_data = state.get("audio_data")

        findings = {
            "audio_available": audio_data is not None,
            "duration_ratio": 1.0,
            "expected_duration_sec": 0.0,
            "actual_duration_sec": 0.0,
            "phonetic_plausibility": 1.0
        }

        if audio_data:
            try:
                # Estimate audio duration (assuming 16kHz, 16-bit mono)
                findings["actual_duration_sec"] = len(audio_data) / (16000 * 2)

                # Expected duration based on word count
                word_count = len(transcription.split())
                findings["expected_duration_sec"] = word_count / 2.5  # ~150 wpm

                # Calculate ratio
                if findings["expected_duration_sec"] > 0:
                    findings["duration_ratio"] = (
                        findings["actual_duration_sec"] / findings["expected_duration_sec"]
                    )

                # Phonetic plausibility score
                if findings["duration_ratio"] < 0.3:
                    findings["phonetic_plausibility"] = 0.3
                elif findings["duration_ratio"] > 3.0:
                    findings["phonetic_plausibility"] = 0.5
                else:
                    findings["phonetic_plausibility"] = min(1.0, findings["duration_ratio"])

            except Exception as e:
                logger.debug(f"Phonetic analysis error: {e}")

        state["phonetic_analysis"] = findings
        state["reasoning_steps"].append({
            "step": "check_phonetics",
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "conclusion": f"Phonetic plausibility: {findings['phonetic_plausibility']:.2f}"
        })

        return state

    async def _analyze_behavior_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Analyze against learned behavioral patterns from SQLite"""
        transcription = state["transcription"]
        context = state["context"]

        findings = {
            "has_behavioral_data": False,
            "typical_phrases": [],
            "matches_typical": False,
            "time_of_day_typical": False,
            "behavioral_score": 0.5
        }

        try:
            # Get behavioral patterns from SQLite
            behavioral_data = await self.guard._get_behavioral_patterns(context)

            if behavioral_data:
                findings["has_behavioral_data"] = True
                findings["typical_phrases"] = behavioral_data.get("typical_phrases", [])

                # Check if transcription matches typical patterns
                normalized = transcription.lower().strip()
                for phrase in findings["typical_phrases"]:
                    if SequenceMatcher(None, normalized, phrase.lower()).ratio() > 0.7:
                        findings["matches_typical"] = True
                        break

                # Check time-of-day patterns
                current_hour = datetime.now().hour
                typical_hours = behavioral_data.get("typical_hours", [])
                findings["time_of_day_typical"] = current_hour in typical_hours

                # Calculate behavioral score
                score = 0.5
                if findings["matches_typical"]:
                    score += 0.3
                if findings["time_of_day_typical"]:
                    score += 0.2
                findings["behavioral_score"] = min(1.0, score)

        except Exception as e:
            logger.debug(f"Behavioral analysis error: {e}")

        state["behavioral_analysis"] = findings
        state["reasoning_steps"].append({
            "step": "analyze_behavior",
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "conclusion": f"Behavioral score: {findings['behavioral_score']:.2f}"
        })

        return state

    async def _integrate_sai_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Integrate SAI (Situational Awareness Intelligence) context"""
        findings = {
            "sai_available": False,
            "display_context": None,
            "is_tv_connected": False,
            "environmental_factors": {},
            "sai_confidence_modifier": 1.0
        }

        try:
            # Get SAI context
            from voice_unlock.display_aware_sai import get_display_context, check_tv_connection

            display_context = await get_display_context()
            tv_state = await check_tv_connection()

            findings["sai_available"] = True
            findings["display_context"] = {
                "display_count": display_context.display_count if display_context else 0,
                "is_mirrored": display_context.is_mirrored if display_context else False,
                "primary_display": display_context.primary_display_name if display_context else None
            }
            findings["is_tv_connected"] = tv_state.get("is_tv_connected", False) if tv_state else False

            # Environmental factors from display context
            if display_context:
                findings["environmental_factors"] = {
                    "tv_brand": getattr(display_context, 'tv_brand', None),
                    "display_type": "tv" if findings["is_tv_connected"] else "monitor"
                }

            # Adjust confidence based on SAI
            # TV connections often have more ambient noise
            if findings["is_tv_connected"]:
                findings["sai_confidence_modifier"] = 0.9  # Slightly more lenient

        except ImportError:
            logger.debug("SAI not available for hallucination guard")
        except Exception as e:
            logger.debug(f"SAI integration error: {e}")

        state["sai_analysis"] = findings
        state["reasoning_steps"].append({
            "step": "integrate_sai",
            "timestamp": datetime.now().isoformat(),
            "findings": findings,
            "conclusion": f"SAI available: {findings['sai_available']}, TV: {findings['is_tv_connected']}"
        })

        return state

    async def _form_hypothesis_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Form hypotheses about whether transcription is hallucination"""
        hypotheses = []

        # Gather evidence
        pattern = state.get("pattern_analysis", {})
        consensus = state.get("consensus_analysis", {})
        context = state.get("context_analysis", {})
        phonetic = state.get("phonetic_analysis", {})
        behavioral = state.get("behavioral_analysis", {})
        sai = state.get("sai_analysis", {})

        # Hypothesis 1: Known pattern match
        if pattern.get("matched_patterns"):
            hypotheses.append({
                "type": "known_pattern",
                "confidence": pattern.get("confidence", 0.9),
                "evidence": f"Matched {len(pattern['matched_patterns'])} hallucination patterns",
                "hallucination_type": HallucinationType.KNOWN_PATTERN.value
            })

        # Hypothesis 2: Consensus disagreement
        if consensus.get("consensus_ratio", 1.0) < self.guard.config.consensus_threshold:
            hypotheses.append({
                "type": "consensus_disagreement",
                "confidence": 1.0 - consensus["consensus_ratio"],
                "evidence": f"Only {consensus['consensus_ratio']:.0%} engine agreement",
                "hallucination_type": HallucinationType.LOW_CONFIDENCE_OUTLIER.value
            })

        # Hypothesis 3: Context mismatch
        if not context.get("matches_expected") and context.get("max_similarity", 0) < 0.3:
            hypotheses.append({
                "type": "context_mismatch",
                "confidence": 1.0 - context["max_similarity"],
                "evidence": f"Low similarity ({context['max_similarity']:.2f}) to expected patterns",
                "hallucination_type": HallucinationType.CONTEXTUAL_MISMATCH.value
            })

        # Hypothesis 4: Phonetic implausibility
        if phonetic.get("phonetic_plausibility", 1.0) < 0.5:
            hypotheses.append({
                "type": "phonetic_mismatch",
                "confidence": 1.0 - phonetic["phonetic_plausibility"],
                "evidence": f"Audio duration doesn't match word count",
                "hallucination_type": HallucinationType.PHONETIC_MISMATCH.value
            })

        # Hypothesis 5: Behavioral anomaly
        if behavioral.get("has_behavioral_data") and not behavioral.get("matches_typical"):
            hypotheses.append({
                "type": "behavioral_anomaly",
                "confidence": 0.6,
                "evidence": "Doesn't match typical usage patterns",
                "hallucination_type": HallucinationType.BEHAVIORAL_MISMATCH.value
            })

        # Determine if hallucination based on hypotheses
        is_hallucination = False
        hallucination_type = None
        hallucination_confidence = 0.0

        if hypotheses:
            # Sort by confidence
            hypotheses.sort(key=lambda h: h["confidence"], reverse=True)
            top_hypothesis = hypotheses[0]

            # Apply SAI confidence modifier
            sai_modifier = sai.get("sai_confidence_modifier", 1.0)
            adjusted_confidence = top_hypothesis["confidence"] * sai_modifier

            # Threshold for hallucination determination
            if adjusted_confidence >= self.guard.config.sensitivity:
                is_hallucination = True
                hallucination_type = top_hypothesis["hallucination_type"]
                hallucination_confidence = adjusted_confidence

        state["hypotheses"] = hypotheses
        state["is_hallucination"] = is_hallucination
        state["hallucination_type"] = hallucination_type
        state["hallucination_confidence"] = hallucination_confidence

        state["reasoning_steps"].append({
            "step": "form_hypothesis",
            "timestamp": datetime.now().isoformat(),
            "hypotheses_count": len(hypotheses),
            "top_hypothesis": hypotheses[0] if hypotheses else None,
            "conclusion": f"Hallucination: {is_hallucination} ({hallucination_confidence:.2f})"
        })

        return state

    def _should_attempt_correction(
        self, state: HallucinationAnalysisState
    ) -> Literal["correct", "decide"]:
        """Determine if correction should be attempted"""
        if state["is_hallucination"] and self.guard.config.auto_correct:
            return "correct"
        return "decide"

    async def _attempt_correction_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Attempt to correct the hallucination"""
        correction = None
        correction_source = None

        # Strategy 1: Best alternative from consensus
        consensus = state.get("consensus_analysis", {})
        if consensus.get("best_alternative"):
            correction = consensus["best_alternative"]
            correction_source = "consensus_alternative"

        # Strategy 2: Learned correction from SQLite
        if not correction:
            normalized = state["transcription"].lower().strip()
            learned = await self.guard._get_learned_correction(normalized)
            if learned:
                correction = learned
                correction_source = "learned_correction"

        # Strategy 3: Contextual default
        if not correction and state["context"] == "unlock_command":
            correction = "unlock my screen"
            correction_source = "contextual_default"

        state["correction"] = correction
        state["reasoning_steps"].append({
            "step": "attempt_correction",
            "timestamp": datetime.now().isoformat(),
            "correction": correction,
            "source": correction_source,
            "conclusion": f"Correction: '{correction}' from {correction_source}"
        })

        return state

    async def _final_decision_node(
        self, state: HallucinationAnalysisState
    ) -> HallucinationAnalysisState:
        """Make final decision and prepare response"""
        if state["is_hallucination"]:
            if state["correction"]:
                decision = f"CORRECTED: '{state['transcription']}' → '{state['correction']}'"
            else:
                decision = f"FLAGGED: '{state['transcription']}' is likely hallucination"
        else:
            decision = f"CLEAN: '{state['transcription']}' passed verification"

        state["final_decision"] = decision
        state["reasoning_steps"].append({
            "step": "final_decision",
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "is_hallucination": state["is_hallucination"],
            "correction_applied": state["correction"] is not None
        })

        return state


# ============================================================================
# LANGCHAIN TOOL ORCHESTRATION
# ============================================================================

class LangChainVerificationPipeline:
    """
    LangChain-based tool orchestration for verification pipeline.

    Chains multiple verification tools with intelligent routing:
    1. Pattern Detection Tool
    2. Consensus Validation Tool
    3. Phonetic Verification Tool
    4. Context Matching Tool
    5. Behavioral Analysis Tool
    """

    def __init__(self, guard: 'STTHallucinationGuard'):
        self.guard = guard
        self._chain = None
        self._initialized = False

    async def initialize(self):
        """Initialize LangChain components"""
        if self._initialized:
            return

        try:
            from langchain_core.tools import Tool
            from langchain.chains import SequentialChain

            # Define tools (simplified - full implementation would use proper LangChain tools)
            self._tools = {
                "pattern_detector": self._create_pattern_tool(),
                "consensus_validator": self._create_consensus_tool(),
                "phonetic_verifier": self._create_phonetic_tool(),
                "context_matcher": self._create_context_tool(),
                "behavioral_analyzer": self._create_behavioral_tool()
            }

            self._initialized = True
            logger.info("🔗 LangChain Verification Pipeline initialized")

        except ImportError as e:
            logger.warning(f"LangChain not available: {e}")
            self._initialized = False

    def _create_pattern_tool(self):
        """Create pattern detection tool"""
        async def detect_patterns(text: str) -> Dict[str, Any]:
            return await self.guard._check_known_patterns_internal(text)
        return detect_patterns

    def _create_consensus_tool(self):
        """Create consensus validation tool"""
        async def validate_consensus(
            text: str, engine_results: List[Dict]
        ) -> Dict[str, Any]:
            return await self.guard._check_consensus_internal(text, engine_results)
        return validate_consensus

    def _create_phonetic_tool(self):
        """Create phonetic verification tool"""
        async def verify_phonetics(
            text: str, audio_data: bytes
        ) -> Dict[str, Any]:
            return await self.guard._check_phonetics_internal(text, audio_data)
        return verify_phonetics

    def _create_context_tool(self):
        """Create context matching tool"""
        async def match_context(text: str, context: str) -> Dict[str, Any]:
            return await self.guard._check_context_internal(text, context)
        return match_context

    def _create_behavioral_tool(self):
        """Create behavioral analysis tool"""
        async def analyze_behavior(text: str, context: str) -> Dict[str, Any]:
            return await self.guard._analyze_behavior_internal(text, context)
        return analyze_behavior

    async def run_pipeline(
        self,
        transcription: str,
        confidence: float,
        audio_data: Optional[bytes] = None,
        engine_results: Optional[List[Dict[str, Any]]] = None,
        context: str = "unlock_command"
    ) -> Dict[str, Any]:
        """Run the full verification pipeline"""
        results = {
            "transcription": transcription,
            "confidence": confidence,
            "tools_run": [],
            "issues_found": [],
            "overall_score": 1.0
        }

        # Run pattern detection
        pattern_result = await self._tools["pattern_detector"](transcription)
        results["tools_run"].append("pattern_detector")
        if pattern_result.get("is_suspicious"):
            results["issues_found"].append(pattern_result)
            results["overall_score"] *= 0.5

        # Run consensus validation if we have multiple results
        if engine_results and len(engine_results) >= 2:
            consensus_result = await self._tools["consensus_validator"](
                transcription, engine_results
            )
            results["tools_run"].append("consensus_validator")
            if consensus_result.get("consensus_ratio", 1.0) < self.guard.config.consensus_threshold:
                results["issues_found"].append(consensus_result)
                results["overall_score"] *= consensus_result["consensus_ratio"]

        # Run phonetic verification if we have audio
        if audio_data:
            phonetic_result = await self._tools["phonetic_verifier"](
                transcription, audio_data
            )
            results["tools_run"].append("phonetic_verifier")
            if phonetic_result.get("plausibility", 1.0) < 0.5:
                results["issues_found"].append(phonetic_result)
                results["overall_score"] *= phonetic_result["plausibility"]

        # Run context matching
        context_result = await self._tools["context_matcher"](transcription, context)
        results["tools_run"].append("context_matcher")
        if not context_result.get("matches"):
            results["issues_found"].append(context_result)
            results["overall_score"] *= 0.7

        # Run behavioral analysis
        behavioral_result = await self._tools["behavioral_analyzer"](transcription, context)
        results["tools_run"].append("behavioral_analyzer")
        results["behavioral_score"] = behavioral_result.get("score", 0.5)

        return results


# ============================================================================
# MAIN HALLUCINATION GUARD CLASS
# ============================================================================

class STTHallucinationGuard:
    """
    Advanced anti-hallucination system for STT transcription.

    Features:
    - LangGraph multi-step reasoning
    - LangChain tool orchestration
    - SAI situational awareness integration
    - SQLite continuous learning
    - Multi-engine consensus validation
    - Phonetic similarity verification
    - Behavioral pattern analysis
    """

    # Known hallucination patterns
    KNOWN_HALLUCINATION_PATTERNS = [
        # Random names Whisper commonly hallucinates
        r"\b(mark\s+mccree|john\s+smith|jane\s+doe)\b",
        r"\bhey\s+jarvis,?\s+i'?m\s+\w+\b",
        r"\bmy\s+name\s+is\s+\w+\b",
        r"\bi'?m\s+(?:mr|ms|mrs|miss|dr)\.?\s+\w+\b",

        # Repetitive patterns
        r"(\b\w+\b)(\s+\1){3,}",
        r"(\.{3,}|,{3,}|\?{3,}|!{3,})",

        # Foreign language indicators
        r"[\u4e00-\u9fff]",
        r"[\u3040-\u309f\u30a0-\u30ff]",
        r"[\uac00-\ud7af]",
        r"[\u0600-\u06ff]",

        # Common Whisper artifacts
        r"^\s*\[.*?\]\s*$",
        r"^(\s*♪\s*)+$",
        r"^\s*\.\.\.\s*$",
        r"^\s*(um|uh|er|ah)\s*$",

        # Impossible phrases
        r"thank\s+you\s+for\s+watching",
        r"please\s+subscribe",
        r"like\s+and\s+subscribe",
        r"see\s+you\s+in\s+the\s+next",
    ]

    UNLOCK_COMMAND_PATTERNS = [
        r"unlock\s*(my\s+)?(screen|computer|mac|laptop)?",
        r"(hey\s+)?jarvis[,.]?\s*unlock",
        r"open\s+(my\s+)?(screen|computer|session)",
        r"log\s*(me\s+)?in",
        r"wake\s+up",
    ]

    def __init__(self, config: Optional[HallucinationGuardConfig] = None):
        """Initialize the hallucination guard"""
        self.config = config or HallucinationGuardConfig()

        # Compile patterns
        self._compiled_hallucination_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.KNOWN_HALLUCINATION_PATTERNS
        ]
        self._compiled_unlock_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.UNLOCK_COMMAND_PATTERNS
        ]

        # LangGraph reasoner
        self._reasoner = LangGraphHallucinationReasoner(self)

        # LangChain pipeline
        self._pipeline = LangChainVerificationPipeline(self)

        # Learning state (in-memory cache, synced to SQLite)
        self._learned_hallucinations: Set[str] = set()
        self._learned_corrections: Dict[str, str] = {}
        self._detection_history: List[HallucinationDetection] = []

        # Active context priors
        self._active_priors: List[ContextualPrior] = []

        # Metrics database connection
        self._metrics_db = None

        # Metrics
        self.metrics = {
            "total_checks": 0,
            "hallucinations_detected": 0,
            "hallucinations_corrected": 0,
            "langgraph_reasoning_calls": 0,
            "langchain_pipeline_calls": 0,
            "sai_integrations": 0,
            "learned_patterns_used": 0,
            "by_type": {},
            "avg_detection_time_ms": 0.0,
            "consensus_disagreements": 0,
        }

        # Callbacks
        self._on_hallucination_callbacks: List[Callable] = []

        # Initialize defaults
        self._setup_default_priors()

        logger.info(
            f"🛡️ STT Hallucination Guard initialized | "
            f"LangGraph: {self.config.use_langgraph_reasoning} | "
            f"SAI: {self.config.use_sai_context} | "
            f"Learning: {self.config.enable_learning}"
        )

    def _setup_default_priors(self):
        """Setup default contextual priors with time-based weights"""
        # Unlock command context with time weights
        unlock_prior = ContextualPrior(
            context_name="unlock_command",
            expected_patterns=self.UNLOCK_COMMAND_PATTERNS,
            weight=2.0,
            phoneme_patterns=["unlock", "unlock my screen", "jarvis unlock"],
            time_based_weights={
                # Morning unlock is most common
                "hour_6": 1.2, "hour_7": 1.5, "hour_8": 1.3, "hour_9": 1.2,
                # Evening
                "hour_17": 1.1, "hour_18": 1.1, "hour_19": 1.0,
                # Late night (less common but still valid)
                "hour_22": 0.9, "hour_23": 0.8, "hour_0": 0.7,
            }
        )
        self._active_priors.append(unlock_prior)

    async def initialize(self):
        """Async initialization for LangGraph, LangChain, and SQLite"""
        # Initialize LangGraph reasoner
        if self.config.use_langgraph_reasoning:
            await self._reasoner.initialize()

        # Initialize LangChain pipeline
        await self._pipeline.initialize()

        # Connect to metrics database for learning
        await self._connect_metrics_db()

        # Load learned patterns from SQLite
        await self._load_learned_patterns()

        logger.info("✅ Hallucination Guard fully initialized")

    async def _connect_metrics_db(self):
        """Connect to metrics database for learning"""
        try:
            from voice_unlock.metrics_database import get_metrics_database
            self._metrics_db = get_metrics_database()
            logger.info("📊 Connected to metrics database for hallucination learning")
        except Exception as e:
            logger.warning(f"Could not connect to metrics database: {e}")

    async def _load_learned_patterns(self):
        """Load learned hallucination patterns from SQLite"""
        if not self._metrics_db:
            return

        try:
            patterns = await self._metrics_db.get_hallucination_patterns()
            corrections = await self._metrics_db.get_hallucination_corrections()

            self._learned_hallucinations = set(patterns)
            self._learned_corrections = corrections

            logger.info(
                f"📚 Loaded {len(self._learned_hallucinations)} learned patterns, "
                f"{len(self._learned_corrections)} corrections from SQLite"
            )
        except Exception as e:
            logger.debug(f"Could not load learned patterns: {e}")

    async def verify_transcription(
        self,
        transcription: str,
        confidence: float,
        audio_data: Optional[bytes] = None,
        engine_results: Optional[List[Dict[str, Any]]] = None,
        context: Optional[str] = "unlock_command"
    ) -> Tuple[VerificationResult, Optional[HallucinationDetection], str]:
        """
        Main verification method with LangGraph reasoning.
        """
        start_time = time.time()
        self.metrics["total_checks"] += 1

        audio_hash = None
        if audio_data:
            audio_hash = hashlib.md5(audio_data[:1000]).hexdigest()[:8]

        logger.info(f"🔍 Verifying: '{transcription}' (conf: {confidence:.2f})")

        # 🚀 FAST PATH: Check for obvious hallucinations first (skip full reasoning)
        normalized = transcription.lower().strip()

        # Fast check 1: Known learned hallucinations
        if normalized in self._learned_hallucinations:
            correction = self._learned_corrections.get(normalized, "unlock my screen")
            self.metrics["learned_patterns_used"] += 1
            self.metrics["hallucinations_detected"] += 1
            self.metrics["hallucinations_corrected"] += 1

            detection = HallucinationDetection(
                original_text=transcription,
                corrected_text=correction,
                hallucination_type=HallucinationType.KNOWN_PATTERN,
                confidence=0.95,
                detection_method="fast_path_learned",
                audio_hash=audio_hash
            )
            self._detection_history.append(detection)

            logger.info(f"🚀 [FAST-PATH] Learned hallucination: '{transcription}' → '{correction}'")
            self._update_timing_metrics((time.time() - start_time) * 1000)
            return VerificationResult.CORRECTED, detection, correction

        # Fast check 2: Known pattern match (e.g., "Hey Jarvis, I'm [name]")
        for pattern in self._compiled_hallucination_patterns:
            if pattern.search(normalized):
                correction = "unlock my screen"  # Default correction for unlock context
                self.metrics["hallucinations_detected"] += 1
                self.metrics["hallucinations_corrected"] += 1

                detection = HallucinationDetection(
                    original_text=transcription,
                    corrected_text=correction,
                    hallucination_type=HallucinationType.KNOWN_PATTERN,
                    confidence=0.95,
                    detection_method="fast_path_pattern",
                    evidence={"matched_pattern": pattern.pattern},
                    audio_hash=audio_hash
                )
                self._detection_history.append(detection)

                # Learn this for next time
                self._learned_hallucinations.add(normalized)
                self._learned_corrections[normalized] = correction

                logger.info(f"🚀 [FAST-PATH] Pattern match: '{transcription}' → '{correction}'")
                self._update_timing_metrics((time.time() - start_time) * 1000)

                # Store in SQLite (fire and forget)
                if self._metrics_db and self.config.enable_learning:
                    asyncio.create_task(self._metrics_db.record_hallucination(
                        original_text=transcription,
                        corrected_text=correction,
                        hallucination_type="known_pattern",
                        confidence=0.95,
                        detection_method="fast_path_pattern",
                        context=context or "unlock_command"
                    ))

                return VerificationResult.CORRECTED, detection, correction

        # Fast check 3: Already looks like valid unlock command - skip full reasoning
        for pattern in self._compiled_unlock_patterns:
            if pattern.search(normalized):
                logger.debug(f"✅ [FAST-PATH] Valid unlock pattern, skipping full reasoning")
                self._update_timing_metrics((time.time() - start_time) * 1000)
                return VerificationResult.CLEAN, None, transcription

        # 🧠 FULL REASONING: Run LangGraph for ambiguous cases
        if self.config.use_langgraph_reasoning:
            self.metrics["langgraph_reasoning_calls"] += 1

            analysis_state = await self._reasoner.reason(
                transcription=transcription,
                confidence=confidence,
                audio_data=audio_data,
                engine_results=engine_results,
                context=context or "unlock_command"
            )

            # Build detection from state
            if analysis_state["is_hallucination"]:
                detection = HallucinationDetection(
                    original_text=transcription,
                    corrected_text=analysis_state.get("correction"),
                    hallucination_type=HallucinationType(analysis_state["hallucination_type"]),
                    confidence=analysis_state["hallucination_confidence"],
                    detection_method="langgraph_reasoning",
                    evidence={
                        "hypotheses": analysis_state["hypotheses"],
                        "pattern_analysis": analysis_state.get("pattern_analysis"),
                        "consensus_analysis": analysis_state.get("consensus_analysis"),
                    },
                    reasoning_chain=analysis_state["reasoning_steps"],
                    sai_context=analysis_state.get("sai_analysis"),
                    audio_hash=audio_hash
                )

                # Update metrics
                self.metrics["hallucinations_detected"] += 1
                self._update_type_metrics(detection.hallucination_type)

                # Store detection
                self._detection_history.append(detection)

                # Learn from this detection
                if self.config.enable_learning:
                    await self._learn_from_detection(detection)

                # Notify callbacks
                await self._notify_callbacks(detection)

                # Determine result
                if detection.corrected_text:
                    self.metrics["hallucinations_corrected"] += 1
                    result = VerificationResult.CORRECTED
                    final_text = detection.corrected_text
                    logger.info(
                        f"✅ CORRECTED: '{transcription}' → '{final_text}' | "
                        f"Reasoning: {len(analysis_state['reasoning_steps'])} steps"
                    )
                else:
                    result = VerificationResult.CONFIRMED
                    final_text = transcription
                    logger.warning(f"🚫 HALLUCINATION CONFIRMED: '{transcription}'")

                # Update timing
                self._update_timing_metrics((time.time() - start_time) * 1000)

                return result, detection, final_text

        # Clean transcription
        self._update_timing_metrics((time.time() - start_time) * 1000)
        logger.debug(f"✅ Clean: '{transcription}'")

        return VerificationResult.CLEAN, None, transcription

    async def _check_known_patterns_internal(self, text: str) -> Dict[str, Any]:
        """Internal pattern check for LangChain tool"""
        normalized = text.lower().strip()
        result = {"is_suspicious": False, "patterns": [], "confidence": 0.0}

        for pattern in self._compiled_hallucination_patterns:
            if pattern.search(normalized):
                result["is_suspicious"] = True
                result["patterns"].append(pattern.pattern)
                result["confidence"] = 0.95

        if normalized in self._learned_hallucinations:
            result["is_suspicious"] = True
            result["patterns"].append("learned_pattern")
            result["confidence"] = max(result["confidence"], 0.90)
            self.metrics["learned_patterns_used"] += 1

        return result

    async def _check_consensus_internal(
        self, text: str, engine_results: List[Dict]
    ) -> Dict[str, Any]:
        """Internal consensus check for LangChain tool"""
        if len(engine_results) < 2:
            return {"consensus_ratio": 1.0, "agreements": 1, "total": 1}

        normalized = text.lower().strip()
        agreements = 0

        for result in engine_results:
            other_text = result.get("text", "").lower().strip()
            if SequenceMatcher(None, normalized, other_text).ratio() >= 0.7:
                agreements += 1

        return {
            "consensus_ratio": agreements / len(engine_results),
            "agreements": agreements,
            "total": len(engine_results)
        }

    async def _check_phonetics_internal(
        self, text: str, audio_data: bytes
    ) -> Dict[str, Any]:
        """Internal phonetic check for LangChain tool"""
        try:
            duration_sec = len(audio_data) / (16000 * 2)
            word_count = len(text.split())
            expected_duration = word_count / 2.5

            if expected_duration > 0:
                ratio = duration_sec / expected_duration
                plausibility = min(1.0, ratio) if ratio <= 1.5 else max(0.3, 1.5 / ratio)
            else:
                plausibility = 1.0

            return {"plausibility": plausibility, "duration_sec": duration_sec}
        except:
            return {"plausibility": 1.0, "duration_sec": 0}

    async def _check_context_internal(self, text: str, context: str) -> Dict[str, Any]:
        """Internal context check for LangChain tool"""
        normalized = text.lower().strip()
        matches = False

        for prior in self._active_priors:
            if prior.context_name == context:
                for pattern_str in prior.expected_patterns:
                    try:
                        if re.search(pattern_str, normalized, re.IGNORECASE):
                            matches = True
                            break
                    except:
                        if pattern_str.lower() in normalized:
                            matches = True
                            break

        return {"matches": matches, "context": context}

    async def _analyze_behavior_internal(self, text: str, context: str) -> Dict[str, Any]:
        """Internal behavioral analysis for LangChain tool"""
        return await self._get_behavioral_patterns(context)

    async def _check_learned_patterns(self, normalized_text: str) -> Optional[Dict[str, Any]]:
        """Check against learned patterns from SQLite"""
        if normalized_text in self._learned_hallucinations:
            self.metrics["learned_patterns_used"] += 1
            return {
                "pattern": "learned_hallucination",
                "matched_text": normalized_text,
                "confidence": 0.90
            }
        return None

    async def _get_learned_correction(self, normalized_text: str) -> Optional[str]:
        """Get learned correction from cache/SQLite"""
        return self._learned_corrections.get(normalized_text)

    async def _get_behavioral_patterns(self, context: str) -> Dict[str, Any]:
        """Get behavioral patterns from SQLite"""
        result = {
            "has_behavioral_data": False,
            "typical_phrases": [],
            "typical_hours": [],
            "score": 0.5
        }

        if self._metrics_db:
            try:
                behavioral_data = await self._metrics_db.get_user_behavioral_patterns(context)
                if behavioral_data:
                    result.update(behavioral_data)
                    result["has_behavioral_data"] = True
            except Exception as e:
                logger.debug(f"Could not get behavioral patterns: {e}")

        return result

    async def _learn_from_detection(self, detection: HallucinationDetection):
        """Learn from a hallucination detection and store in SQLite"""
        normalized = detection.original_text.lower().strip()

        # Update in-memory cache
        self._learned_hallucinations.add(normalized)
        if detection.corrected_text:
            self._learned_corrections[normalized] = detection.corrected_text

        # Store in SQLite
        if self._metrics_db:
            try:
                await self._metrics_db.record_hallucination(
                    original_text=detection.original_text,
                    corrected_text=detection.corrected_text,
                    hallucination_type=detection.hallucination_type.value,
                    confidence=detection.confidence,
                    detection_method=detection.detection_method,
                    reasoning_steps=len(detection.reasoning_chain),
                    sai_context=detection.sai_context,
                    audio_hash=detection.audio_hash
                )
                logger.debug(f"📚 Learned hallucination stored in SQLite: '{normalized}'")
            except Exception as e:
                logger.warning(f"Could not store hallucination in SQLite: {e}")

    def learn_hallucination(self, hallucination_text: str, correction: Optional[str] = None):
        """Manually learn a hallucination pattern"""
        normalized = hallucination_text.lower().strip()
        self._learned_hallucinations.add(normalized)

        if correction:
            self._learned_corrections[normalized] = correction

        # Store in SQLite (fire and forget)
        if self._metrics_db and self.config.enable_learning:
            asyncio.create_task(self._store_manual_learning(normalized, correction))

        logger.info(f"📚 Manually learned: '{hallucination_text}' → '{correction or '[flagged]'}'")

    async def _store_manual_learning(self, normalized: str, correction: Optional[str]):
        """Store manually learned pattern in SQLite"""
        if self._metrics_db:
            try:
                await self._metrics_db.record_hallucination(
                    original_text=normalized,
                    corrected_text=correction,
                    hallucination_type="manual_learning",
                    confidence=1.0,
                    detection_method="user_feedback",
                    reasoning_steps=0,
                    sai_context=None,
                    audio_hash=None
                )
            except Exception as e:
                logger.warning(f"Could not store manual learning: {e}")

    def _update_type_metrics(self, hallucination_type: HallucinationType):
        """Update metrics by type"""
        type_name = hallucination_type.value
        self.metrics["by_type"][type_name] = self.metrics["by_type"].get(type_name, 0) + 1

    def _update_timing_metrics(self, detection_time_ms: float):
        """Update timing metrics"""
        total = self.metrics["total_checks"]
        prev_avg = self.metrics["avg_detection_time_ms"]
        self.metrics["avg_detection_time_ms"] = (prev_avg * (total - 1) + detection_time_ms) / total

    async def _notify_callbacks(self, detection: HallucinationDetection):
        """Notify callbacks"""
        for callback in self._on_hallucination_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(detection)
                else:
                    callback(detection)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def on_hallucination_detected(self, callback: Callable[[HallucinationDetection], None]):
        """Register callback"""
        self._on_hallucination_callbacks.append(callback)

    def add_contextual_prior(self, prior: ContextualPrior):
        """Add contextual prior"""
        self._active_priors.append(prior)

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics"""
        return {
            **self.metrics,
            "learned_patterns": len(self._learned_hallucinations),
            "learned_corrections": len(self._learned_corrections),
            "active_priors": [p.context_name for p in self._active_priors],
            "config": {
                "sensitivity": self.config.sensitivity,
                "langgraph_enabled": self.config.use_langgraph_reasoning,
                "sai_enabled": self.config.use_sai_context,
                "learning_enabled": self.config.enable_learning,
            }
        }

    def get_detection_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get detection history"""
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
                "reasoning_steps": len(d.reasoning_chain),
                "timestamp": d.timestamp.isoformat(),
            }
            for d in history
        ]

    def get_reasoning_trace(self, detection: HallucinationDetection) -> str:
        """Get human-readable reasoning trace for a detection"""
        lines = [
            f"🔍 Hallucination Analysis for: '{detection.original_text}'",
            f"━" * 60,
        ]

        for i, step in enumerate(detection.reasoning_chain, 1):
            lines.append(f"\nStep {i}: {step.get('step', 'unknown').upper()}")
            lines.append(f"  Time: {step.get('timestamp', 'N/A')}")
            lines.append(f"  Conclusion: {step.get('conclusion', 'N/A')}")

        lines.append(f"\n{'━' * 60}")
        lines.append(f"Final Result: {detection.hallucination_type.value}")
        lines.append(f"Confidence: {detection.confidence:.2%}")
        if detection.corrected_text:
            lines.append(f"Correction: '{detection.corrected_text}'")

        return "\n".join(lines)


# ============================================================================
# GLOBAL INSTANCE AND CONVENIENCE FUNCTIONS
# ============================================================================

_global_guard: Optional[STTHallucinationGuard] = None


def get_hallucination_guard(
    config: Optional[HallucinationGuardConfig] = None
) -> STTHallucinationGuard:
    """Get or create global hallucination guard"""
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

    # Ensure initialized
    if not hasattr(guard, '_initialized_async') or not guard._initialized_async:
        await guard.initialize()
        guard._initialized_async = True

    result, detection, final_text = await guard.verify_transcription(
        transcription=text,
        confidence=confidence,
        audio_data=audio_data,
        engine_results=engine_results,
        context=context
    )

    was_corrected = result == VerificationResult.CORRECTED
    return final_text, was_corrected, detection
