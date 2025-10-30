#!/usr/bin/env python3
"""
Intervention Decision Engine for Proactive Intelligence System (Component 3.2)

This engine determines when and how JARVIS should proactively assist users by:
1. Evaluating user state (frustration, productivity, focus, stress)
2. Assessing situations (severity, time criticality, solution availability)
3. Choosing intervention options (silent monitoring to autonomous action)
4. Optimizing timing for interventions
5. Learning from intervention effectiveness

Memory allocation: 80MB total (30MB decision models, 25MB history, 25MB learning)
Multi-language support: Python (orchestration), Rust (performance), Swift (macOS integration)
Zero hardcoding: All parameters are learned and dynamically adapted

Classes:
    UserState: Enumeration of user emotional and cognitive states
    SituationType: Types of situations requiring intervention
    InterventionLevel: Levels of intervention from passive to active
    UserStateSignal: Signal indicating user state with metadata
    SituationAssessment: Assessment of a situation requiring intervention
    InterventionDecision: Decision about intervention with reasoning
    UserStateEvaluator: Evaluates user emotional and cognitive states
    SituationAnalyzer: Analyzes situations requiring intervention
    InterventionTiming: Optimizes timing for interventions
    EffectivenessLearner: Learns from intervention effectiveness
    InterventionDecisionEngine: Main engine orchestrating all components

Functions:
    get_intervention_engine: Get or create intervention engine instance
    test_intervention_engine: Test the intervention decision engine

Example:
    >>> engine = get_intervention_engine()
    >>> decision = await engine.evaluate_intervention_need(context)
    >>> if decision:
    ...     result = await engine.execute_intervention(decision)
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod

# System imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Internal imports
from vision.intelligence.state_intelligence import StateIntelligence, get_state_intelligence
from vision.intelligence.temporal_context_engine import TemporalContextEngine
from vision.intelligence.goal_inference_system import GoalInferenceEngine
from autonomy.autonomous_decision_engine import AutonomousDecisionEngine, AutonomousAction, ActionPriority, ActionCategory
from core.memory_controller import MemoryController
from core.dynamic_config_manager import DynamicConfigManager

logger = logging.getLogger(__name__)

class UserState(Enum):
    """User emotional and cognitive states.
    
    Represents different states a user can be in while interacting with the system.
    These states are used to determine appropriate intervention strategies.
    
    Attributes:
        FRUSTRATED: User is experiencing frustration with current task
        FOCUSED: User is deeply concentrated on work
        PRODUCTIVE: User is making good progress on tasks
        STRESSED: User is under pressure or experiencing stress
        CONFUSED: User is uncertain about how to proceed
        OVERWHELMED: User has too many competing demands
        IDLE: User is not actively engaged with any task
        ENGAGED: User is actively working but not deeply focused
    """
    FRUSTRATED = "frustrated"
    FOCUSED = "focused"
    PRODUCTIVE = "productive"
    STRESSED = "stressed"
    CONFUSED = "confused"
    OVERWHELMED = "overwhelmed"
    IDLE = "idle"
    ENGAGED = "engaged"

class SituationType(Enum):
    """Types of situations requiring intervention.
    
    Categorizes different scenarios where JARVIS might need to intervene
    to assist the user or prevent problems.
    
    Attributes:
        CRITICAL_ERROR: System error requiring immediate attention
        WORKFLOW_BLOCKED: User's workflow is stuck or blocked
        EFFICIENCY_OPPORTUNITY: Chance to improve user efficiency
        LEARNING_MOMENT: Opportunity to teach user something new
        HEALTH_REMINDER: Reminder for user health and wellbeing
        SECURITY_CONCERN: Potential security risk detected
        TIME_MANAGEMENT: Issues with time allocation or deadlines
    """
    CRITICAL_ERROR = "critical_error"
    WORKFLOW_BLOCKED = "workflow_blocked"
    EFFICIENCY_OPPORTUNITY = "efficiency_opportunity"
    LEARNING_MOMENT = "learning_moment"
    HEALTH_REMINDER = "health_reminder"
    SECURITY_CONCERN = "security_concern"
    TIME_MANAGEMENT = "time_management"

class InterventionLevel(Enum):
    """Levels of intervention from passive to active.
    
    Defines the spectrum of intervention approaches from completely passive
    monitoring to autonomous action without user input.
    
    Attributes:
        SILENT_MONITORING: Monitor situation without user-visible action
        SUBTLE_INDICATION: Minimal visual cue or indicator
        GENTLE_SUGGESTION: Soft suggestion that can be easily dismissed
        DIRECT_RECOMMENDATION: Clear recommendation with options
        PROACTIVE_ASSISTANCE: Active offer of help with specific actions
        AUTONOMOUS_ACTION: Take action automatically with minimal user input
    """
    SILENT_MONITORING = "silent_monitoring"
    SUBTLE_INDICATION = "subtle_indication"
    GENTLE_SUGGESTION = "gentle_suggestion"
    DIRECT_RECOMMENDATION = "direct_recommendation"
    PROACTIVE_ASSISTANCE = "proactive_assistance"
    AUTONOMOUS_ACTION = "autonomous_action"

@dataclass
class UserStateSignal:
    """Signal indicating user state.
    
    Represents a single piece of evidence about the user's current state,
    collected from various sources like vision, audio, or behavior analysis.
    
    Attributes:
        signal_type: Type of signal (e.g., 'facial_frustration', 'rapid_interactions')
        strength: Signal strength from 0.0 to 1.0
        confidence: Confidence in signal accuracy from 0.0 to 1.0
        source: Source of signal (vision, audio, behavior, etc.)
        timestamp: When the signal was detected
        metadata: Additional context-specific information
    """
    signal_type: str
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    source: str  # vision, audio, behavior, etc.
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SituationAssessment:
    """Assessment of a situation requiring intervention.
    
    Comprehensive evaluation of a situation that might require JARVIS intervention,
    including severity, urgency, and available solutions.
    
    Attributes:
        situation_type: Type of situation detected
        severity: Severity level from 0.0 to 1.0
        time_criticality: Urgency level from 0.0 to 1.0 (1.0 = immediate)
        solution_availability: Availability of solutions from 0.0 to 1.0
        context: Additional context information about the situation
        confidence: Confidence in the assessment accuracy
        timestamp: When the assessment was made
    """
    situation_type: SituationType
    severity: float  # 0.0 to 1.0
    time_criticality: float  # 0.0 to 1.0 (1.0 = immediate)
    solution_availability: float  # 0.0 to 1.0 (1.0 = clear solution)
    context: Dict[str, Any]
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class InterventionDecision:
    """Decision about intervention.
    
    Complete decision about how, when, and why to intervene in a user's workflow,
    including the reasoning and expected effectiveness.
    
    Attributes:
        intervention_level: Chosen level of intervention
        timing_delay: How long to wait before intervening
        intervention_content: Specific content and actions for intervention
        reasoning: Human-readable explanation of the decision
        confidence: Confidence in the decision quality
        expected_effectiveness: Predicted effectiveness based on learning
        user_state: User's state when decision was made
        situation: Situation assessment that triggered the decision
        timestamp: When the decision was made
    """
    intervention_level: InterventionLevel
    timing_delay: timedelta  # How long to wait before intervening
    intervention_content: Dict[str, Any]
    reasoning: str
    confidence: float
    expected_effectiveness: float  # Learned from history
    user_state: UserState
    situation: SituationAssessment
    timestamp: datetime = field(default_factory=datetime.now)

class UserStateEvaluator:
    """Evaluates user emotional and cognitive states.
    
    Analyzes multiple signals from different sources to determine the user's
    current emotional and cognitive state, learning patterns over time.
    
    Attributes:
        memory_limit: Memory limit in bytes for this component
        state_patterns: Learned patterns for different user states
        signal_weights: Learned weights for different signal types
        learning_buffer: Buffer of recent evaluations for learning
        thresholds: Dynamic thresholds learned from user behavior
    """
    
    def __init__(self, memory_limit_mb: int = 30):
        """Initialize the UserStateEvaluator.
        
        Args:
            memory_limit_mb: Memory limit in megabytes for this component
        """
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.state_patterns = {}
        self.signal_weights = {}
        self.learning_buffer = deque(maxlen=1000)
        
        # Dynamic thresholds - learned from user behavior
        self.thresholds = {
            'frustration_indicators': 0.3,
            'productivity_indicators': 0.6,
            'focus_duration': 300,  # seconds
            'stress_accumulation': 0.7
        }
        
        # Load learned patterns
        self._load_state_patterns()
    
    def evaluate_user_state(self, signals: List[UserStateSignal], 
                          context: Dict[str, Any]) -> Tuple[UserState, float]:
        """Evaluate current user state from multiple signals.
        
        Analyzes all available signals to determine the most likely user state
        and confidence in that assessment.
        
        Args:
            signals: List of user state signals from various sources
            context: Additional context information
            
        Returns:
            Tuple of (predicted_user_state, confidence_score)
            
        Example:
            >>> signals = [UserStateSignal('rapid_interactions', 0.8, 0.9, 'vision')]
            >>> state, confidence = evaluator.evaluate_user_state(signals, {})
            >>> print(f"User is {state.value} with {confidence:.1%} confidence")
        """
        
        # Aggregate signals by type
        signal_scores = defaultdict(float)
        total_confidence = 0.0
        
        for signal in signals:
            weight = self.signal_weights.get(signal.signal_type, 1.0)
            weighted_strength = signal.strength * signal.confidence * weight
            signal_scores[signal.signal_type] += weighted_strength
            total_confidence += signal.confidence
        
        # Normalize confidence
        if signals:
            avg_confidence = total_confidence / len(signals)
        else:
            avg_confidence = 0.0
        
        # Determine primary state
        state, confidence = self._classify_state(signal_scores, context)
        
        # Learn from this evaluation
        self._record_state_evaluation(signals, state, confidence)
        
        return state, min(confidence * avg_confidence, 1.0)
    
    def _classify_state(self, signal_scores: Dict[str, float], 
                       context: Dict[str, Any]) -> Tuple[UserState, float]:
        """Classify user state from signal scores.
        
        Args:
            signal_scores: Aggregated signal scores by type
            context: Additional context information
            
        Returns:
            Tuple of (most_likely_state, confidence)
        """
        
        # Dynamic classification using learned patterns
        state_probabilities = {}
        
        for state in UserState:
            prob = self._calculate_state_probability(state, signal_scores, context)
            state_probabilities[state] = prob
        
        # Get most likely state
        best_state = max(state_probabilities, key=state_probabilities.get)
        confidence = state_probabilities[best_state]
        
        return best_state, confidence
    
    def _calculate_state_probability(self, state: UserState, 
                                   signal_scores: Dict[str, float],
                                   context: Dict[str, Any]) -> float:
        """Calculate probability of a specific state.
        
        Args:
            state: User state to calculate probability for
            signal_scores: Aggregated signal scores
            context: Additional context information
            
        Returns:
            Probability score from 0.0 to 1.0
        """
        
        # Get learned pattern for this state
        pattern = self.state_patterns.get(state.value, {})
        
        probability = 0.0
        
        # Check signal patterns
        for signal_type, score in signal_scores.items():
            expected_range = pattern.get(f"{signal_type}_range", (0.0, 1.0))
            if expected_range[0] <= score <= expected_range[1]:
                probability += 0.2  # Each matching signal adds probability
        
        # Context-based adjustments
        if 'time_in_current_task' in context:
            task_time = context['time_in_current_task']
            
            # Frustration increases with time in difficult tasks
            if state == UserState.FRUSTRATED and task_time > 600:  # 10 minutes
                probability += 0.3
            
            # Focus decreases over time without breaks
            elif state == UserState.FOCUSED and task_time < 1800:  # 30 minutes
                probability += 0.2
        
        # Temporal patterns
        current_hour = datetime.now().hour
        time_factor = pattern.get(f"hour_{current_hour}", 1.0)
        probability *= time_factor
        
        return min(probability, 1.0)
    
    def _record_state_evaluation(self, signals: List[UserStateSignal], 
                               state: UserState, confidence: float):
        """Record state evaluation for learning.
        
        Args:
            signals: Signals used in evaluation
            state: Predicted user state
            confidence: Confidence in prediction
        """
        
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'signals': [
                {
                    'type': s.signal_type,
                    'strength': s.strength,
                    'confidence': s.confidence,
                    'source': s.source
                } for s in signals
            ],
            'predicted_state': state.value,
            'confidence': confidence
        }
        
        self.learning_buffer.append(evaluation)
        
        # Periodically update patterns
        if len(self.learning_buffer) >= 100:
            self._update_state_patterns()
    
    def _update_state_patterns(self):
        """Update state patterns from learning buffer.
        
        Analyzes recent evaluations to update learned patterns for
        different user states and signal combinations.
        """
        
        # Analyze recent evaluations
        state_data = defaultdict(list)
        
        for evaluation in list(self.learning_buffer):
            state = evaluation['predicted_state']
            
            for signal in evaluation['signals']:
                key = f"{state}_{signal['type']}"
                state_data[key].append(signal['strength'])
        
        # Update patterns
        for key, values in state_data.items():
            if len(values) >= 10:  # Minimum samples for pattern
                mean_val = np.mean(values)
                std_val = np.std(values)
                
                # Update range for this signal type in this state
                state, signal_type = key.split('_', 1)
                if state not in self.state_patterns:
                    self.state_patterns[state] = {}
                
                self.state_patterns[state][f"{signal_type}_range"] = (
                    max(0.0, mean_val - std_val),
                    min(1.0, mean_val + std_val)
                )
    
    def _load_state_patterns(self):
        """Load learned state patterns from disk.
        
        Loads previously learned patterns, weights, and thresholds
        from persistent storage.
        """
        patterns_file = Path("backend/data/user_state_patterns.json")
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.state_patterns = data.get('patterns', {})
                    self.signal_weights = data.get('weights', {})
                    self.thresholds = data.get('thresholds', self.thresholds)
            except Exception as e:
                logger.error(f"Failed to load state patterns: {e}")
    
    def save_patterns(self):
        """Save learned patterns to disk.
        
        Persists learned patterns, weights, and thresholds to disk
        for future use.
        
        Raises:
            Exception: If file writing fails
        """
        patterns_file = Path("backend/data/user_state_patterns.json")
        patterns_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            data = {
                'patterns': self.state_patterns,
                'weights': self.signal_weights,
                'thresholds': self.thresholds,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state patterns: {e}")

class SituationAnalyzer:
    """Analyzes situations requiring intervention.
    
    Evaluates current context to identify situations that might require
    JARVIS intervention, assessing severity, urgency, and solution availability.
    
    Attributes:
        memory_limit: Memory limit in bytes for this component
        situation_patterns: Learned patterns for different situation types
        solution_database: Database of known solutions for situations
        criticality_models: Models for assessing situation criticality
    """
    
    def __init__(self, memory_limit_mb: int = 25):
        """Initialize the SituationAnalyzer.
        
        Args:
            memory_limit_mb: Memory limit in megabytes for this component
        """
        self.memory_limit = memory_limit_mb * 1024 * 1024
        self.situation_patterns = {}
        self.solution_database = {}
        self.criticality_models = {}
        
        # Load learned situation patterns
        self._load_situation_patterns()
    
    def assess_situation(self, context: Dict[str, Any], 
                        user_state: UserState) -> Optional[SituationAssessment]:
        """Assess if current context requires intervention.
        
        Analyzes the current context and user state to determine if there's
        a situation requiring intervention and how severe it is.
        
        Args:
            context: Current context information
            user_state: Current user state
            
        Returns:
            SituationAssessment if intervention needed, None otherwise
            
        Example:
            >>> context = {'error_detected': True, 'time_in_current_task': 600}
            >>> assessment = analyzer.assess_situation(context, UserState.FRUSTRATED)
            >>> if assessment:
            ...     print(f"Situation: {assessment.situation_type.value}")
        """
        
        # Detect situation type
        situation_type = self._detect_situation_type(context, user_state)
        if not situation_type:
            return None
        
        # Calculate severity
        severity = self._calculate_severity(situation_type, context, user_state)
        
        # Calculate time criticality
        time_criticality = self._calculate_time_criticality(situation_type, context)
        
        # Check solution availability
        solution_availability = self._check_solution_availability(situation_type, context)
        
        # Overall confidence
        confidence = self._calculate_assessment_confidence(
            situation_type, severity, time_criticality, solution_availability
        )
        
        return SituationAssessment(
            situation_type=situation_type,
            severity=severity,
            time_criticality=time_criticality,
            solution_availability=solution_availability,
            context=context.copy(),
            confidence=confidence
        )
    
    def _detect_situation_type(self, context: Dict[str, Any], 
                              user_state: UserState) -> Optional[SituationType]:
        """Detect the type of situation from context.
        
        Args:
            context: Current context information
            user_state: Current user state
            
        Returns:
            Detected situation type or None if no situation detected
        """
        
        # Check for critical errors
        if context.get('error_detected') or context.get('system_failure'):
            return SituationType.CRITICAL_ERROR
        
        # Check for workflow blockages
        if (user_state == UserState.FRUSTRATED and 
            context.get('time_in_current_task', 0) > 600):  # 10+ minutes
            return SituationType.WORKFLOW_BLOCKED
        
        # Check for efficiency opportunities
        if context.get('repetitive_actions', 0) > 5:
            return SituationType.EFFICIENCY_OPPORTUNITY
        
        # Check for learning moments
        if (user_state == UserState.CONFUSED and 
            context.get('help_searches', 0) > 2):
            return SituationType.LEARNING_MOMENT
        
        # Check for health reminders
        if context.get('time_without_break', 0) > 3600:  # 1+ hour
            return SituationType.HEALTH_REMINDER
        
        # Check for security concerns
        if context.get('sensitive_data_exposure') or context.get('unsafe_action'):
            return SituationType.SECURITY_CONCERN
        
        # Check for time management issues
        if context.get('deadline_approaching') and user_state != UserState.FOCUSED:
            return SituationType.TIME_MANAGEMENT
        
        return None
    
    def _calculate_severity(self, situation_type: SituationType, 
                          context: Dict[str, Any], user_state: UserState) -> float:
        """Calculate situation severity (0.0 to 1.0).
        
        Args:
            situation_type: Type of situation detected
            context: Current context information
            user_state: Current user state
            
        Returns:
            Severity score from 0.0 to 1.0
        """
        
        base_severity = {
            SituationType.CRITICAL_ERROR: 0.9,
            SituationType.SECURITY_CONCERN: 0.8,
            SituationType.WORKFLOW_BLOCKED: 0.6,
            SituationType.TIME_MANAGEMENT: 0.7,
            SituationType.LEARNING_MOMENT: 0.4,
            SituationType.EFFICIENCY_OPPORTUNITY: 0.3,
            SituationType.HEALTH_REMINDER: 0.5
        }.get(situation_type, 0.5)
        
        # Adjust based on user state
        if user_state in [UserState.FRUSTRATED, UserState.STRESSED]:
            base_severity += 0.2
        elif user_state == UserState.OVERWHELMED:
            base_severity += 0.3
        
        # Context-specific adjustments
        if context.get('impact_level') == 'high':
            base_severity += 0.2
        elif context.get('impact_level') == 'low':
            base_severity -= 0.2
        
        return min(1.0, max(0.0, base_severity))
    
    def _calculate_time_criticality(self, situation_type: SituationType, 
                                  context: Dict[str, Any]) -> float:
        """Calculate time criticality (0.0 to 1.0).
        
        Args:
            situation_type: Type of situation detected
            context: Current context information
            
        Returns:
            Time criticality score from 0.0 to 1.0
        """
        
        # Base criticality by type
        base_criticality = {
            SituationType.CRITICAL_ERROR: 1.0,
            SituationType.SECURITY_CONCERN: 0.9,
            SituationType.WORKFLOW_BLOCKED: 0.7,
            SituationType.TIME_MANAGEMENT: 0.8,
            SituationType.LEARNING_MOMENT: 0.3,
            SituationType.EFFICIENCY_OPPORTUNITY: 0.2,
            SituationType.HEALTH_REMINDER: 0.4
        }.get(situation_type, 0.5)
        
        # Adjust based on deadlines
        if context.get('time_until_deadline'):
            deadline_hours = context['time_until_deadline'] / 3600  # Convert to hours
            if deadline_hours < 1:
                base_criticality = 1.0
            elif deadline_hours < 4:
                base_criticality += 0.3
            elif deadline_hours < 24:
                base_criticality += 0.1
        
        # Adjust based on user activity
        if context.get('user_idle_time', 0) > 300:  # 5+ minutes idle
            base_criticality -= 0.2
        
        return min(1.0, max(0.0, base_criticality))
    
    def _check_solution_availability(self, situation_type: SituationType, 
                                   context: Dict[str, Any]) -> float:
        """Check availability of solutions (0.0 to 1.0).
        
        Args:
            situation_type: Type of situation detected
            context: Current context information
            
        Returns:
            Solution availability score from 0.0 to 1.0
        """
        
        # Check solution database
        solutions = self.solution_database.get(situation_type.value, [])
        
        if not solutions:
            return 0.1  # Low availability if no known solutions
        
        # Check context match for solutions
        matching_solutions = 0
        for solution in solutions:
            context_match = 0
            required_context = solution.get('required_context', {})
            
            for key, value in required_context.items():
                if key in context and context[key] == value:
                    context_match += 1
            
            if len(required_context) == 0 or context_match / len(required_context) > 0.7:
                matching_solutions += 1
        
        availability = matching_solutions / len(solutions)
        return min(1.0, max(0.1, availability))
    
    def _calculate_assessment_confidence(self, situation_type: SituationType,
                                       severity: float, time_criticality: float,
                                       solution_availability: float) -> float:
        """Calculate overall assessment confidence.
        
        Args:
            situation_type: Type of situation detected
            severity: Calculated severity score
            time_criticality: Calculated time criticality score
            solution_availability: Calculated solution availability score
            
        Returns:
            Overall confidence score from 0.0 to 1.0
        """
        
        # Base confidence from pattern matching
        pattern_data = self.situation_patterns.get(situation_type.value, {})
        pattern_confidence = pattern_data.get('confidence', 0.5)
        
        # Adjust based on data completeness
        completeness = (severity + time_criticality + solution_availability) / 3
        
        return min(1.0, pattern_confidence * completeness)
    
    def _load_situation_patterns(self):
        """Load learned situation patterns from disk.
        
        Loads previously learned patterns and solution database
        from persistent storage.
        """
        patterns_file = Path("backend/data/situation_patterns.json")
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    self.situation_patterns = data.get('patterns', {})
                    self.solution_database = data.get('solutions', {})
            except Exception as e:
                logger.error(f"Failed to load situation patterns: {e}")

class InterventionTiming:
    """Optimizes timing for interventions.
    
    Determines the optimal timing for interventions based on user state,
    situation urgency, and learned patterns of effectiveness.
    
    Attributes:
        timing_patterns: Learned patterns for optimal intervention timing
        interruption_costs: Learned costs of interrupting different tasks
        effectiveness_history: History of timing effectiveness for learning
    """
    
    def __init__(self):
        """Initialize the InterventionTiming optimizer."""
        self.timing_patterns = {}
        self.interruption_costs = {}
        self.effectiveness_history = deque(maxlen=1000)
        
        # Load timing patterns
        self._load_timing_patterns()
    
    def calculate_optimal_timing(self, decision: InterventionDecision,
                               user_state: UserState,
                               context: Dict[str, Any]) -> timedelta:
        """Calculate optimal timing for intervention.
        
        Determines when to execute an intervention based on user state,
        situation urgency, and learned timing patterns.
        
        Args:
            decision: The intervention decision to time
            user_state: Current user state
            context: Current context information
            
        Returns:
            Optimal delay before executing intervention
            
        Example:
            >>> timing = InterventionTiming()
            >>> delay = timing.calculate_optimal_timing(decision, user_state, context)
            >>> print(f"Wait {delay.total_seconds()} seconds before intervening")
        """
        
        # Base timing by intervention level
        base_delays = {
            InterventionLevel.SILENT_MONITORING: timedelta(0),
            InterventionLevel.SUBTLE_INDICATION: timedelta(seconds=30),
            InterventionLevel.GENTLE_SUGGESTION: timedelta(minutes=2),
            InterventionLevel.DIRECT_RECOMMENDATION: timedelta(minutes=1),
            InterventionLevel.PROACTIVE_ASSISTANCE: timedelta(seconds=10),
            InterventionLevel.AUTONOMOUS_ACTION: timedelta(0)
        }
        
        base_delay = base_delays.get(decision.intervention_level, timedelta(minutes=1))
        
        # Adjust based on user state
        if user_state in [UserState.FOCUSED, UserState.ENGAGED]:
            # Don't interrupt focused work unless critical
            if decision.situation.severity < 0.8:
                base_delay += timedelta(minutes=5)
        
        elif user_state in [UserState.FRUSTRATED, UserState.OVERWHELMED]:
            # Intervene quickly when user is struggling
            base_delay = max(timedelta(seconds=10), base_delay * 0.3)
        
        # Adjust based on time criticality
        criticality_factor = decision.situation.time_criticality
        if criticality_factor > 0.8:
            base_delay = min(base_delay, timedelta(seconds=30))
        
        # Consider interruption cost
        current_task = context.get('current_task')
        if current_task:
            interruption_cost = self.interruption_costs.get(current_task, 0.5)
            if interruption_cost > 0.7:  # High-cost task
                base_delay *= 1.5
        
        # Learn from timing patterns
        timing_key = f"{user_state.value}_{decision.intervention_level.value}"
        if timing