"""
Enhanced Unified Awareness Engine with LangGraph Chain-of-Thought Reasoning

This module extends the UAE with advanced LangGraph-based reasoning capabilities:
- Multi-step decision reasoning with explicit thought chains
- Adaptive strategy selection based on task complexity
- Self-reflection and confidence calibration
- Parallel hypothesis evaluation for uncertainty handling
- Continuous learning from decision outcomes

The enhanced UAE provides:
- Transparent reasoning traces for all decisions
- Dynamic confidence adjustment based on evidence
- Cross-layer reasoning fusion
- Predictive decision making with uncertainty quantification

Author: JARVIS AI System
Version: 2.1.0 - LangGraph Integration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Deque, Dict, List, Optional, Set, Tuple, Union
)
from uuid import uuid4
from collections import deque

import numpy as np

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = "end"
    MemorySaver = None

from pydantic import BaseModel, Field

# Import base UAE components
try:
    from .unified_awareness_engine import (
        UnifiedAwarenessEngine,
        ContextIntelligenceLayer,
        SituationalAwarenessLayer,
        AwarenessIntegrationLayer,
        UnifiedDecision,
        ExecutionResult,
        DecisionSource,
        ContextualData,
        SituationalData,
        ElementPriority
    )
except ImportError:
    from unified_awareness_engine import (
        UnifiedAwarenessEngine,
        ContextIntelligenceLayer,
        SituationalAwarenessLayer,
        AwarenessIntegrationLayer,
        UnifiedDecision,
        ExecutionResult,
        DecisionSource,
        ContextualData,
        SituationalData,
        ElementPriority
    )

# Import chain-of-thought
try:
    from .chain_of_thought import (
        ChainOfThoughtEngine,
        ChainOfThoughtMixin,
        ChainOfThoughtState,
        ReasoningStrategy,
        ThoughtType,
        Thought,
        ReasoningChain,
        create_cot_engine
    )
except ImportError:
    from chain_of_thought import (
        ChainOfThoughtEngine,
        ChainOfThoughtMixin,
        ChainOfThoughtState,
        ReasoningStrategy,
        ThoughtType,
        Thought,
        ReasoningChain,
        create_cot_engine
    )

logger = logging.getLogger(__name__)


# ============================================================================
# Enhanced State Models
# ============================================================================

class UAEReasoningPhase(str, Enum):
    """Phases specific to UAE reasoning."""
    CONTEXT_ANALYSIS = "context_analysis"
    SITUATION_PERCEPTION = "situation_perception"
    FUSION_REASONING = "fusion_reasoning"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    DECISION_SYNTHESIS = "decision_synthesis"
    OUTCOME_PREDICTION = "outcome_prediction"
    STRATEGY_SELECTION = "strategy_selection"


@dataclass
class ReasonedDecision(UnifiedDecision):
    """Extended UnifiedDecision with reasoning trace."""
    reasoning_chain_id: str = ""
    thought_count: int = 0
    reasoning_trace: str = ""
    hypotheses_evaluated: int = 0
    prediction_confidence: float = 0.0
    predicted_outcome: str = ""
    alternative_decisions: List[Dict[str, Any]] = field(default_factory=list)


class UAEGraphState(BaseModel):
    """State for UAE LangGraph reasoning."""
    # Identity
    reasoning_id: str = Field(default_factory=lambda: str(uuid4()))
    element_id: str = ""

    # Phase
    phase: str = UAEReasoningPhase.CONTEXT_ANALYSIS.value

    # Layer data
    context_data: Optional[Dict[str, Any]] = None
    situation_data: Optional[Dict[str, Any]] = None
    context_confidence: float = 0.0
    situation_confidence: float = 0.0

    # Reasoning
    thoughts: List[Dict[str, Any]] = Field(default_factory=list)
    hypotheses: List[Dict[str, Any]] = Field(default_factory=list)

    # Fusion
    fusion_weights: Dict[str, float] = Field(default_factory=dict)
    fusion_strategy: str = "adaptive"
    position_agreement: bool = False

    # Decision
    chosen_position: Optional[Tuple[int, int]] = None
    decision_source: str = DecisionSource.FUSION.value
    confidence: float = 0.0
    reasoning: str = ""

    # Prediction
    predicted_success_rate: float = 0.0
    risk_factors: List[str] = Field(default_factory=list)

    # Alternatives
    alternative_positions: List[Dict[str, Any]] = Field(default_factory=list)

    # Control
    iterations: int = 0
    max_iterations: int = 10
    should_continue: bool = True

    # Timestamps
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# UAE Reasoning Nodes
# ============================================================================

class ContextAnalysisNode:
    """Analyze historical context with chain-of-thought."""

    def __init__(self, context_layer: ContextIntelligenceLayer):
        self.context_layer = context_layer
        self.logger = logging.getLogger(f"{__name__}.context_analysis")

    async def process(self, state: UAEGraphState) -> UAEGraphState:
        self.logger.debug(f"Analyzing context for element: {state.element_id}")

        state.phase = UAEReasoningPhase.CONTEXT_ANALYSIS.value

        # Get contextual data
        try:
            context_data = await self.context_layer.get_contextual_data(state.element_id)

            if context_data:
                state.context_data = {
                    "element_id": context_data.element_id,
                    "expected_position": context_data.expected_position,
                    "confidence": context_data.confidence,
                    "usage_count": context_data.usage_count,
                    "pattern_strength": context_data.pattern_strength,
                    "last_success": context_data.last_success
                }
                state.context_confidence = context_data.confidence

                # Create reasoning thought
                thought = {
                    "thought_id": str(uuid4()),
                    "type": ThoughtType.ANALYSIS.value,
                    "content": f"Context analysis: Element '{state.element_id}' has {context_data.usage_count} historical observations with pattern strength {context_data.pattern_strength:.2f}",
                    "confidence": context_data.confidence,
                    "timestamp": datetime.utcnow().isoformat()
                }
                state.thoughts.append(thought)

                # Generate hypothesis about position
                if context_data.expected_position:
                    hyp = {
                        "hypothesis_id": str(uuid4()),
                        "source": "context",
                        "position": context_data.expected_position,
                        "confidence": context_data.confidence,
                        "reasoning": f"Historical pattern suggests position {context_data.expected_position}"
                    }
                    state.hypotheses.append(hyp)

        except Exception as e:
            self.logger.warning(f"Context analysis failed: {e}")
            thought = {
                "thought_id": str(uuid4()),
                "type": ThoughtType.OBSERVATION.value,
                "content": f"Context analysis unavailable: {str(e)}",
                "confidence": 0.1,
                "timestamp": datetime.utcnow().isoformat()
            }
            state.thoughts.append(thought)

        return state


class SituationPerceptionNode:
    """Perceive current situation with chain-of-thought."""

    def __init__(self, situation_layer: SituationalAwarenessLayer):
        self.situation_layer = situation_layer
        self.logger = logging.getLogger(f"{__name__}.situation_perception")

    async def process(self, state: UAEGraphState) -> UAEGraphState:
        self.logger.debug(f"Perceiving situation for element: {state.element_id}")

        state.phase = UAEReasoningPhase.SITUATION_PERCEPTION.value

        # Get situational data
        try:
            situation_data = await self.situation_layer.get_situational_data(state.element_id)

            if situation_data:
                state.situation_data = {
                    "element_id": situation_data.element_id,
                    "detected_position": situation_data.detected_position,
                    "confidence": situation_data.confidence,
                    "detection_method": situation_data.detection_method,
                    "detection_time": situation_data.detection_time
                }
                state.situation_confidence = situation_data.confidence

                # Create reasoning thought
                thought = {
                    "thought_id": str(uuid4()),
                    "type": ThoughtType.OBSERVATION.value,
                    "content": f"Situation perception: Element '{state.element_id}' detected at {situation_data.detected_position} via {situation_data.detection_method}",
                    "confidence": situation_data.confidence,
                    "timestamp": datetime.utcnow().isoformat()
                }
                state.thoughts.append(thought)

                # Generate hypothesis
                if situation_data.detected_position:
                    hyp = {
                        "hypothesis_id": str(uuid4()),
                        "source": "situation",
                        "position": situation_data.detected_position,
                        "confidence": situation_data.confidence,
                        "reasoning": f"Real-time detection at {situation_data.detected_position}"
                    }
                    state.hypotheses.append(hyp)

        except Exception as e:
            self.logger.warning(f"Situation perception failed: {e}")
            thought = {
                "thought_id": str(uuid4()),
                "type": ThoughtType.OBSERVATION.value,
                "content": f"Situation perception unavailable: {str(e)}",
                "confidence": 0.1,
                "timestamp": datetime.utcnow().isoformat()
            }
            state.thoughts.append(thought)

        return state


class FusionReasoningNode:
    """Reason about fusing context and situation data."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.fusion_reasoning")

    async def process(self, state: UAEGraphState) -> UAEGraphState:
        self.logger.debug("Performing fusion reasoning")

        state.phase = UAEReasoningPhase.FUSION_REASONING.value

        # Check for position agreement
        context_pos = state.context_data.get("expected_position") if state.context_data else None
        situation_pos = state.situation_data.get("detected_position") if state.situation_data else None

        if context_pos and situation_pos:
            # Calculate distance between positions
            distance = np.sqrt(
                (context_pos[0] - situation_pos[0]) ** 2 +
                (context_pos[1] - situation_pos[1]) ** 2
            )
            state.position_agreement = distance < 50  # Within 50 pixels

            # Create reasoning thought about agreement
            if state.position_agreement:
                thought = {
                    "thought_id": str(uuid4()),
                    "type": ThoughtType.INFERENCE.value,
                    "content": f"Position agreement detected: context {context_pos} and situation {situation_pos} within {distance:.1f}px",
                    "confidence": 0.9,
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                thought = {
                    "thought_id": str(uuid4()),
                    "type": ThoughtType.INFERENCE.value,
                    "content": f"Position disagreement: context {context_pos} vs situation {situation_pos}, distance {distance:.1f}px",
                    "confidence": 0.7,
                    "timestamp": datetime.utcnow().isoformat()
                }
            state.thoughts.append(thought)

        # Calculate fusion weights
        state.fusion_weights = self._calculate_fusion_weights(state)

        # Create thought about fusion strategy
        fusion_thought = {
            "thought_id": str(uuid4()),
            "type": ThoughtType.ANALYSIS.value,
            "content": f"Fusion weights: context={state.fusion_weights.get('context', 0):.2f}, situation={state.fusion_weights.get('situation', 0):.2f}",
            "confidence": 0.8,
            "timestamp": datetime.utcnow().isoformat()
        }
        state.thoughts.append(fusion_thought)

        return state

    def _calculate_fusion_weights(self, state: UAEGraphState) -> Dict[str, float]:
        """Calculate dynamic fusion weights based on data quality."""
        context_weight = 0.0
        situation_weight = 0.0

        context_conf = state.context_confidence
        situation_conf = state.situation_confidence

        if state.position_agreement:
            # Agreement boosts both weights
            context_weight = context_conf * 0.5
            situation_weight = situation_conf * 0.5
        elif context_conf > 0 and situation_conf > 0:
            # Weighted by confidence
            total = context_conf + situation_conf
            context_weight = context_conf / total
            situation_weight = situation_conf / total

            # Recency bias for situation
            if state.situation_data:
                detection_time = state.situation_data.get("detection_time", 0)
                age_seconds = time.time() - detection_time
                if age_seconds < 30:
                    situation_weight *= 1.2
                    context_weight *= 0.8
        elif situation_conf > 0:
            situation_weight = 1.0
        elif context_conf > 0:
            context_weight = 1.0
        else:
            # Default fallback
            context_weight = 0.5
            situation_weight = 0.5

        # Normalize
        total = context_weight + situation_weight
        if total > 0:
            context_weight /= total
            situation_weight /= total

        return {
            "context": context_weight,
            "situation": situation_weight
        }


class ConfidenceCalibrationNode:
    """Calibrate confidence with self-reflection."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.confidence_calibration")

    async def process(self, state: UAEGraphState) -> UAEGraphState:
        self.logger.debug("Calibrating confidence")

        state.phase = UAEReasoningPhase.CONFIDENCE_CALIBRATION.value

        # Evaluate hypothesis quality
        hypothesis_quality = self._evaluate_hypotheses(state)

        # Evaluate reasoning chain quality
        reasoning_quality = self._evaluate_reasoning(state)

        # Calculate calibrated confidence
        base_confidence = max(state.context_confidence, state.situation_confidence)
        calibrated_confidence = base_confidence * hypothesis_quality * reasoning_quality

        # Apply agreement bonus
        if state.position_agreement:
            calibrated_confidence = min(1.0, calibrated_confidence * 1.2)

        state.confidence = calibrated_confidence

        # Create calibration thought
        thought = {
            "thought_id": str(uuid4()),
            "type": ThoughtType.REFLECTION.value,
            "content": f"Confidence calibrated from {base_confidence:.2f} to {calibrated_confidence:.2f} (hyp_quality={hypothesis_quality:.2f}, reasoning_quality={reasoning_quality:.2f})",
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat()
        }
        state.thoughts.append(thought)

        return state

    def _evaluate_hypotheses(self, state: UAEGraphState) -> float:
        """Evaluate quality of hypotheses."""
        if not state.hypotheses:
            return 0.5

        # More hypotheses with higher confidence = better
        confidences = [h.get("confidence", 0.5) for h in state.hypotheses]
        avg_confidence = sum(confidences) / len(confidences)

        # Diversity bonus
        sources = set(h.get("source") for h in state.hypotheses)
        diversity_bonus = 0.1 * (len(sources) - 1)

        return min(1.0, avg_confidence + diversity_bonus)

    def _evaluate_reasoning(self, state: UAEGraphState) -> float:
        """Evaluate quality of reasoning chain."""
        if not state.thoughts:
            return 0.5

        # Check for variety of thought types
        thought_types = [t.get("type") for t in state.thoughts]
        unique_types = len(set(thought_types))

        # More diverse = better
        type_score = min(1.0, unique_types / 4)

        # Check for reflections and corrections
        has_reflection = any(t == ThoughtType.REFLECTION.value for t in thought_types)
        reflection_bonus = 0.1 if has_reflection else 0

        return min(1.0, type_score + reflection_bonus)


class DecisionSynthesisNode:
    """Synthesize final decision with reasoning."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.decision_synthesis")

    async def process(self, state: UAEGraphState) -> UAEGraphState:
        self.logger.debug("Synthesizing decision")

        state.phase = UAEReasoningPhase.DECISION_SYNTHESIS.value

        # Select best position based on fusion weights
        position, source, reasoning = self._select_position(state)

        state.chosen_position = position
        state.decision_source = source
        state.reasoning = reasoning

        # Generate alternative decisions
        state.alternative_positions = self._generate_alternatives(state)

        # Create decision thought
        thought = {
            "thought_id": str(uuid4()),
            "type": ThoughtType.DECISION.value,
            "content": f"Decision: Position {position} selected via {source} with confidence {state.confidence:.2f}. Reasoning: {reasoning}",
            "confidence": state.confidence,
            "timestamp": datetime.utcnow().isoformat()
        }
        state.thoughts.append(thought)

        return state

    def _select_position(
        self,
        state: UAEGraphState
    ) -> Tuple[Optional[Tuple[int, int]], str, str]:
        """Select the best position."""
        context_pos = state.context_data.get("expected_position") if state.context_data else None
        situation_pos = state.situation_data.get("detected_position") if state.situation_data else None

        context_weight = state.fusion_weights.get("context", 0)
        situation_weight = state.fusion_weights.get("situation", 0)

        if state.position_agreement and context_pos and situation_pos:
            # Weighted average
            x = int(context_pos[0] * context_weight + situation_pos[0] * situation_weight)
            y = int(context_pos[1] * context_weight + situation_pos[1] * situation_weight)
            return (x, y), DecisionSource.FUSION.value, "Positions agree, using weighted fusion"

        elif situation_weight > context_weight and situation_pos:
            return situation_pos, DecisionSource.SITUATION.value, "Real-time situation data preferred"

        elif context_pos:
            return context_pos, DecisionSource.CONTEXT.value, "Historical context data preferred"

        elif situation_pos:
            return situation_pos, DecisionSource.SITUATION.value, "Only situation data available"

        else:
            return None, DecisionSource.FALLBACK.value, "No position data available"

    def _generate_alternatives(self, state: UAEGraphState) -> List[Dict[str, Any]]:
        """Generate alternative decision options."""
        alternatives = []

        for hyp in state.hypotheses:
            if hyp.get("position") != state.chosen_position:
                alternatives.append({
                    "position": hyp.get("position"),
                    "source": hyp.get("source"),
                    "confidence": hyp.get("confidence"),
                    "reasoning": hyp.get("reasoning")
                })

        return alternatives[:3]  # Top 3 alternatives


class OutcomePredictionNode:
    """Predict outcome of the decision."""

    def __init__(self, context_layer: Optional[ContextIntelligenceLayer] = None):
        self.context_layer = context_layer
        self.logger = logging.getLogger(f"{__name__}.outcome_prediction")

    async def process(self, state: UAEGraphState) -> UAEGraphState:
        self.logger.debug("Predicting outcome")

        state.phase = UAEReasoningPhase.OUTCOME_PREDICTION.value

        # Predict success rate based on historical data
        success_rate = await self._predict_success(state)
        state.predicted_success_rate = success_rate

        # Identify risk factors
        state.risk_factors = self._identify_risks(state)

        # Create prediction thought
        thought = {
            "thought_id": str(uuid4()),
            "type": ThoughtType.PREDICTION.value,
            "content": f"Predicted success rate: {success_rate:.0%}. Risk factors: {', '.join(state.risk_factors) if state.risk_factors else 'None identified'}",
            "confidence": 0.7,
            "timestamp": datetime.utcnow().isoformat()
        }
        state.thoughts.append(thought)

        # Mark complete
        state.completed_at = datetime.utcnow().isoformat()
        state.should_continue = False

        return state

    async def _predict_success(self, state: UAEGraphState) -> float:
        """Predict success rate based on historical patterns."""
        base_rate = state.confidence

        # Adjust based on decision source
        source_modifiers = {
            DecisionSource.FUSION.value: 1.1,
            DecisionSource.SITUATION.value: 1.05,
            DecisionSource.CONTEXT.value: 0.95,
            DecisionSource.FALLBACK.value: 0.7
        }

        modifier = source_modifiers.get(state.decision_source, 1.0)

        # Check historical success if context layer available
        if self.context_layer and state.context_data:
            pattern_strength = state.context_data.get("pattern_strength", 0.5)
            base_rate = (base_rate + pattern_strength) / 2

        return min(1.0, base_rate * modifier)

    def _identify_risks(self, state: UAEGraphState) -> List[str]:
        """Identify risk factors for the decision."""
        risks = []

        if state.confidence < 0.5:
            risks.append("Low overall confidence")

        if not state.position_agreement:
            risks.append("Position disagreement between sources")

        if state.decision_source == DecisionSource.FALLBACK.value:
            risks.append("Using fallback decision")

        if len(state.hypotheses) < 2:
            risks.append("Limited hypothesis diversity")

        if state.context_confidence < 0.3:
            risks.append("Weak historical patterns")

        if state.situation_confidence < 0.3:
            risks.append("Low situation detection confidence")

        return risks


# ============================================================================
# Router Functions
# ============================================================================

def route_after_context(state: UAEGraphState) -> str:
    return "situation_perception"


def route_after_situation(state: UAEGraphState) -> str:
    return "fusion_reasoning"


def route_after_fusion(state: UAEGraphState) -> str:
    return "confidence_calibration"


def route_after_calibration(state: UAEGraphState) -> str:
    # Check if confidence is acceptable
    if state.confidence < 0.3 and state.iterations < state.max_iterations:
        state.iterations += 1
        return "situation_perception"  # Try to get better data
    return "decision_synthesis"


def route_after_decision(state: UAEGraphState) -> str:
    return "outcome_prediction"


# ============================================================================
# Enhanced UAE with LangGraph
# ============================================================================

class EnhancedUAE:
    """
    Enhanced Unified Awareness Engine with LangGraph Chain-of-Thought Reasoning.

    This class wraps the original UAE and adds:
    - Explicit reasoning traces for decisions
    - Multi-step reasoning with hypothesis evaluation
    - Confidence calibration and self-reflection
    - Outcome prediction with risk assessment
    - Alternative decision generation

    Usage:
        ```python
        enhanced_uae = EnhancedUAE()
        await enhanced_uae.start()

        # Get position with reasoning
        result = await enhanced_uae.get_element_position_with_reasoning("element_id")
        print(result.reasoning_trace)
        print(f"Confidence: {result.confidence}")
        print(f"Predicted success: {result.predicted_success_rate}")
        ```
    """

    def __init__(
        self,
        base_uae: Optional[UnifiedAwarenessEngine] = None,
        enable_checkpointing: bool = True
    ):
        self.base_uae = base_uae
        self.enable_checkpointing = enable_checkpointing

        # Initialize layers (will use base_uae's if available)
        self.context_layer: Optional[ContextIntelligenceLayer] = None
        self.situation_layer: Optional[SituationalAwarenessLayer] = None

        # LangGraph components
        self.graph: Optional[StateGraph] = None
        self.compiled_graph = None

        # Reasoning history
        self._reasoning_history: Deque[ReasonedDecision] = deque(maxlen=100)

        # Callbacks
        self._decision_callbacks: List[Callable] = []

        self.logger = logging.getLogger(__name__)

    async def start(self) -> None:
        """Start the enhanced UAE."""
        self.logger.info("Starting Enhanced UAE with LangGraph...")

        # Start base UAE if provided
        if self.base_uae:
            await self.base_uae.start()
            self.context_layer = self.base_uae.context_layer
            self.situation_layer = self.base_uae.situation_layer
        else:
            # Create standalone layers
            self.context_layer = ContextIntelligenceLayer()
            self.situation_layer = SituationalAwarenessLayer()

        # Build LangGraph
        self._build_graph()

        self.logger.info("Enhanced UAE started successfully")

    async def stop(self) -> None:
        """Stop the enhanced UAE."""
        if self.base_uae:
            await self.base_uae.stop()
        self.logger.info("Enhanced UAE stopped")

    def _build_graph(self) -> None:
        """Build the LangGraph for UAE reasoning."""
        if not LANGGRAPH_AVAILABLE:
            self.logger.warning("LangGraph not available")
            return

        graph = StateGraph(UAEGraphState)

        # Create nodes
        context_node = ContextAnalysisNode(self.context_layer)
        situation_node = SituationPerceptionNode(self.situation_layer)
        fusion_node = FusionReasoningNode()
        calibration_node = ConfidenceCalibrationNode()
        decision_node = DecisionSynthesisNode()
        prediction_node = OutcomePredictionNode(self.context_layer)

        # Add nodes
        graph.add_node("context_analysis", self._wrap_node(context_node))
        graph.add_node("situation_perception", self._wrap_node(situation_node))
        graph.add_node("fusion_reasoning", self._wrap_node(fusion_node))
        graph.add_node("confidence_calibration", self._wrap_node(calibration_node))
        graph.add_node("decision_synthesis", self._wrap_node(decision_node))
        graph.add_node("outcome_prediction", self._wrap_node(prediction_node))

        # Set entry point
        graph.set_entry_point("context_analysis")

        # Add edges
        graph.add_conditional_edges("context_analysis", route_after_context, {
            "situation_perception": "situation_perception"
        })
        graph.add_conditional_edges("situation_perception", route_after_situation, {
            "fusion_reasoning": "fusion_reasoning"
        })
        graph.add_conditional_edges("fusion_reasoning", route_after_fusion, {
            "confidence_calibration": "confidence_calibration"
        })
        graph.add_conditional_edges("confidence_calibration", route_after_calibration, {
            "situation_perception": "situation_perception",
            "decision_synthesis": "decision_synthesis"
        })
        graph.add_conditional_edges("decision_synthesis", route_after_decision, {
            "outcome_prediction": "outcome_prediction"
        })
        graph.add_edge("outcome_prediction", END)

        self.graph = graph

        # Compile
        compile_kwargs = {}
        if self.enable_checkpointing:
            compile_kwargs["checkpointer"] = MemorySaver()

        self.compiled_graph = graph.compile(**compile_kwargs)

    def _wrap_node(self, node):
        """Wrap node for LangGraph."""
        async def wrapped(state: UAEGraphState) -> UAEGraphState:
            return await node.process(state)
        return wrapped

    async def get_element_position_with_reasoning(
        self,
        element_id: str,
        force_detect: bool = False
    ) -> ReasonedDecision:
        """
        Get element position with full chain-of-thought reasoning.

        Args:
            element_id: ID of element to locate
            force_detect: Force fresh detection

        Returns:
            ReasonedDecision with full reasoning trace
        """
        # Initialize state
        initial_state = UAEGraphState(
            element_id=element_id,
            started_at=datetime.utcnow().isoformat()
        )

        # Run reasoning
        if self.compiled_graph:
            try:
                config = {"configurable": {"thread_id": element_id}}
                final_state = await self.compiled_graph.ainvoke(initial_state, config=config)
            except Exception as e:
                self.logger.error(f"Graph execution failed: {e}")
                final_state = await self._fallback_reasoning(initial_state)
        else:
            final_state = await self._fallback_reasoning(initial_state)

        # Create ReasonedDecision
        decision = ReasonedDecision(
            element_id=element_id,
            chosen_position=final_state.chosen_position or (0, 0),
            confidence=final_state.confidence,
            decision_source=DecisionSource(final_state.decision_source),
            context_weight=final_state.fusion_weights.get("context", 0),
            situation_weight=final_state.fusion_weights.get("situation", 0),
            reasoning=final_state.reasoning,
            timestamp=time.time(),
            reasoning_chain_id=final_state.reasoning_id,
            thought_count=len(final_state.thoughts),
            reasoning_trace=self._generate_trace(final_state),
            hypotheses_evaluated=len(final_state.hypotheses),
            prediction_confidence=final_state.predicted_success_rate,
            predicted_outcome=f"Predicted {final_state.predicted_success_rate:.0%} success",
            alternative_decisions=final_state.alternative_positions
        )

        # Store in history
        self._reasoning_history.append(decision)

        # Notify callbacks
        await self._notify_callbacks(decision)

        return decision

    async def _fallback_reasoning(self, state: UAEGraphState) -> UAEGraphState:
        """Fallback sequential reasoning."""
        context_node = ContextAnalysisNode(self.context_layer)
        situation_node = SituationPerceptionNode(self.situation_layer)
        fusion_node = FusionReasoningNode()
        calibration_node = ConfidenceCalibrationNode()
        decision_node = DecisionSynthesisNode()
        prediction_node = OutcomePredictionNode(self.context_layer)

        state = await context_node.process(state)
        state = await situation_node.process(state)
        state = await fusion_node.process(state)
        state = await calibration_node.process(state)
        state = await decision_node.process(state)
        state = await prediction_node.process(state)

        return state

    def _generate_trace(self, state: UAEGraphState) -> str:
        """Generate human-readable reasoning trace."""
        lines = [
            "=== UAE Chain-of-Thought Reasoning ===",
            f"Element: {state.element_id}",
            f"Reasoning ID: {state.reasoning_id}",
            "",
            "Thought Process:"
        ]

        for i, thought in enumerate(state.thoughts, 1):
            lines.append(f"  {i}. [{thought.get('type', 'unknown').upper()}]")
            lines.append(f"     {thought.get('content', '')}")
            lines.append(f"     Confidence: {thought.get('confidence', 0):.2f}")

        lines.extend([
            "",
            f"Hypotheses Evaluated: {len(state.hypotheses)}",
            f"Position Agreement: {state.position_agreement}",
            f"Fusion Weights: context={state.fusion_weights.get('context', 0):.2f}, situation={state.fusion_weights.get('situation', 0):.2f}",
            "",
            f"DECISION: Position {state.chosen_position}",
            f"SOURCE: {state.decision_source}",
            f"CONFIDENCE: {state.confidence:.2f}",
            f"PREDICTED SUCCESS: {state.predicted_success_rate:.0%}",
            "",
            f"Risk Factors: {', '.join(state.risk_factors) if state.risk_factors else 'None'}"
        ])

        return "\n".join(lines)

    def register_decision_callback(self, callback: Callable) -> None:
        """Register callback for decisions."""
        self._decision_callbacks.append(callback)

    async def _notify_callbacks(self, decision: ReasonedDecision) -> None:
        """Notify all callbacks of decision."""
        for callback in self._decision_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(decision)
                else:
                    callback(decision)
            except Exception as e:
                self.logger.warning(f"Callback error: {e}")

    def get_reasoning_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent reasoning history."""
        history = list(self._reasoning_history)[-limit:]
        return [
            {
                "element_id": d.element_id,
                "position": d.chosen_position,
                "confidence": d.confidence,
                "source": d.decision_source.value,
                "thought_count": d.thought_count,
                "predicted_success": d.prediction_confidence
            }
            for d in history
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """Get enhanced UAE metrics."""
        base_metrics = {}
        if self.base_uae:
            base_metrics = self.base_uae.get_comprehensive_metrics()

        return {
            **base_metrics,
            "enhanced_reasoning": {
                "total_reasoned_decisions": len(self._reasoning_history),
                "avg_thought_count": np.mean([d.thought_count for d in self._reasoning_history]) if self._reasoning_history else 0,
                "avg_confidence": np.mean([d.confidence for d in self._reasoning_history]) if self._reasoning_history else 0,
                "avg_predicted_success": np.mean([d.prediction_confidence for d in self._reasoning_history]) if self._reasoning_history else 0,
                "langgraph_available": LANGGRAPH_AVAILABLE
            }
        }


# ============================================================================
# Factory Functions
# ============================================================================

def create_enhanced_uae(
    base_uae: Optional[UnifiedAwarenessEngine] = None,
    **kwargs
) -> EnhancedUAE:
    """Create an enhanced UAE with LangGraph."""
    return EnhancedUAE(base_uae=base_uae, **kwargs)


# Global instance
_enhanced_uae: Optional[EnhancedUAE] = None


async def get_enhanced_uae() -> EnhancedUAE:
    """Get or create global enhanced UAE."""
    global _enhanced_uae
    if _enhanced_uae is None:
        _enhanced_uae = create_enhanced_uae()
        await _enhanced_uae.start()
    return _enhanced_uae
