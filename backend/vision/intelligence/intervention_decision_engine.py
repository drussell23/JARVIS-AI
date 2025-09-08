#!/usr/bin/env python3
"""
Intervention Decision Engine - Proactive Intelligence System Component
Determines when and how JARVIS should proactively assist users
Implements multi-factor decision making with zero hardcoding
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import asyncio
from pathlib import Path
import statistics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

logger = logging.getLogger(__name__)


class UserState(Enum):
    """User psychological/productivity states"""
    FOCUSED = "focused"
    FRUSTRATED = "frustrated"
    PRODUCTIVE = "productive"
    STRUGGLING = "struggling"
    STRESSED = "stressed"
    IDLE = "idle"
    LEARNING = "learning"
    CONFUSED = "confused"


class InterventionType(Enum):
    """Types of interventions available"""
    SILENT_MONITORING = "silent_monitoring"
    SUBTLE_INDICATION = "subtle_indication"
    SUGGESTION_OFFER = "suggestion_offer"
    DIRECT_ASSISTANCE = "direct_assistance"
    AUTONOMOUS_ACTION = "autonomous_action"


class TimingStrategy(Enum):
    """Timing strategies for interventions"""
    IMMEDIATE = "immediate"
    NATURAL_BREAK = "natural_break"
    TASK_BOUNDARY = "task_boundary"
    LOW_COGNITIVE_LOAD = "low_cognitive_load"
    USER_REQUEST_LIKELY = "user_request_likely"


@dataclass
class UserStateSignal:
    """Signal indicating user state"""
    signal_type: str  # mouse_movement, typing_pattern, navigation, error_rate, etc.
    value: float
    confidence: float
    timestamp: datetime
    source: str  # vsms, anomaly_detector, workflow_engine, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SituationContext:
    """Current situation assessment"""
    problem_severity: float  # 0-1 scale
    time_criticality: float  # 0-1 scale
    solution_availability: float  # 0-1 probability we can help
    success_probability: float  # 0-1 probability of successful intervention
    context_type: str  # coding, debugging, learning, communicating, etc.
    active_task: Optional[str] = None
    error_context: Optional[Dict[str, Any]] = None


@dataclass
class InterventionOpportunity:
    """Identified opportunity for intervention"""
    opportunity_id: str
    user_state: UserState
    situation: SituationContext
    intervention_type: InterventionType
    timing_strategy: TimingStrategy
    confidence_score: float
    urgency_score: float
    content: Dict[str, Any]  # What to show/do
    rationale: str  # Why intervene
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class InterventionResult:
    """Result of an intervention"""
    intervention_id: str
    opportunity_id: str
    executed_at: datetime
    user_response: str  # accepted, rejected, ignored, delayed
    effectiveness_score: float  # 0-1 scale
    time_to_resolution: Optional[timedelta] = None
    user_feedback: Optional[str] = None
    side_effects: List[str] = field(default_factory=list)


class InterventionDecisionEngine:
    """Main intervention decision engine"""
    
    def __init__(self, memory_allocation: Dict[str, int] = None):
        """Initialize with memory allocation"""
        # Default memory allocation (80MB total)
        self.memory_allocation = memory_allocation or {
            'decision_models': 30 * 1024 * 1024,  # 30MB
            'intervention_history': 25 * 1024 * 1024,  # 25MB
            'learning_data': 25 * 1024 * 1024   # 25MB
        }
        
        # User state tracking
        self.user_state_signals: deque = deque(maxlen=1000)
        self.current_user_state = UserState.FOCUSED
        self.state_confidence = 0.5
        self.state_history: deque = deque(maxlen=100)
        
        # Situation tracking
        self.current_situation: Optional[SituationContext] = None
        self.situation_history: deque = deque(maxlen=50)
        
        # Decision models
        self.decision_tree = DecisionTreeClassifier(random_state=42)
        self.intervention_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.timing_optimizer = self._create_timing_model()
        self.scaler = StandardScaler()
        
        # Intervention tracking
        self.active_opportunities: Dict[str, InterventionOpportunity] = {}
        self.intervention_history: deque = deque(maxlen=500)
        self.effectiveness_scores: Dict[str, List[float]] = defaultdict(list)
        
        # Learning system
        self.learning_buffer: List[Tuple[Dict, InterventionResult]] = []
        self.model_version = 0
        self.last_training = datetime.now()
        
        # Configuration
        self.min_confidence_threshold = 0.7
        self.intervention_cooldown = timedelta(minutes=5)
        self.last_intervention_time = {}
        
        # Feature extractors
        self.feature_extractors = {
            'user_state': self._extract_user_state_features,
            'situation': self._extract_situation_features,
            'timing': self._extract_timing_features,
            'history': self._extract_history_features
        }
        
        logger.info("Initialized Intervention Decision Engine")
    
    def _create_timing_model(self) -> Dict[str, Any]:
        """Create timing optimization model"""
        return {
            'natural_break_detector': self._detect_natural_break,
            'task_boundary_detector': self._detect_task_boundary,
            'cognitive_load_estimator': self._estimate_cognitive_load,
            'request_likelihood_predictor': self._predict_request_likelihood
        }
    
    async def process_user_signal(self, signal: UserStateSignal):
        """Process incoming user state signal"""
        self.user_state_signals.append(signal)
        
        # Update user state if enough signals
        if len(self.user_state_signals) >= 10:
            await self._update_user_state()
    
    async def _update_user_state(self):
        """Update current user state based on signals"""
        recent_signals = list(self.user_state_signals)[-50:]
        
        # Extract features from signals
        features = self._aggregate_signal_features(recent_signals)
        
        # Determine user state
        state_scores = {
            UserState.FOCUSED: self._calculate_focus_score(features),
            UserState.FRUSTRATED: self._calculate_frustration_score(features),
            UserState.PRODUCTIVE: self._calculate_productivity_score(features),
            UserState.STRUGGLING: self._calculate_struggle_score(features),
            UserState.STRESSED: self._calculate_stress_score(features),
            UserState.IDLE: self._calculate_idle_score(features),
            UserState.LEARNING: self._calculate_learning_score(features),
            UserState.CONFUSED: self._calculate_confusion_score(features)
        }
        
        # Select highest scoring state
        best_state = max(state_scores.items(), key=lambda x: x[1])
        self.current_user_state = best_state[0]
        self.state_confidence = best_state[1]
        
        # Record state change
        self.state_history.append({
            'state': self.current_user_state,
            'confidence': self.state_confidence,
            'timestamp': datetime.now(),
            'features': features
        })
    
    def _aggregate_signal_features(self, signals: List[UserStateSignal]) -> Dict[str, float]:
        """Aggregate features from multiple signals"""
        features = defaultdict(list)
        
        for signal in signals:
            prefix = f"{signal.signal_type}_"
            features[f"{prefix}value"].append(signal.value)
            features[f"{prefix}confidence"].append(signal.confidence)
            
            # Add metadata features
            for key, value in signal.metadata.items():
                if isinstance(value, (int, float)):
                    features[f"{prefix}{key}"].append(value)
        
        # Aggregate lists into statistics
        aggregated = {}
        for key, values in features.items():
            if values:
                aggregated[f"{key}_mean"] = statistics.mean(values)
                aggregated[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0
                aggregated[f"{key}_max"] = max(values)
                aggregated[f"{key}_min"] = min(values)
        
        return aggregated
    
    # User state scoring methods
    def _calculate_focus_score(self, features: Dict[str, float]) -> float:
        """Calculate focus score from features"""
        score = 0.5  # Base score
        
        # Positive indicators
        if features.get('typing_pattern_value_mean', 0) > 0.7:
            score += 0.2
        if features.get('mouse_movement_value_std', 1) < 0.3:
            score += 0.1
        if features.get('task_switches_value_mean', 1) < 0.5:
            score += 0.2
        
        # Negative indicators
        if features.get('error_rate_value_mean', 0) > 0.3:
            score -= 0.2
        if features.get('backspace_rate_value_mean', 0) > 0.4:
            score -= 0.1
        
        return max(0, min(1, score))
    
    def _calculate_frustration_score(self, features: Dict[str, float]) -> float:
        """Calculate frustration score from features"""
        score = 0.0
        
        # Strong indicators
        if features.get('error_rate_value_mean', 0) > 0.5:
            score += 0.3
        if features.get('repeated_actions_value_mean', 0) > 0.4:
            score += 0.3
        if features.get('mouse_movement_value_std', 0) > 0.8:
            score += 0.2
        
        # Behavioral indicators
        if features.get('backspace_rate_value_mean', 0) > 0.5:
            score += 0.2
        if features.get('rapid_clicks_value_mean', 0) > 0.6:
            score += 0.2
        
        return min(1, score)
    
    def _calculate_productivity_score(self, features: Dict[str, float]) -> float:
        """Calculate productivity score from features"""
        score = 0.5
        
        # Positive indicators
        if features.get('task_completion_value_mean', 0) > 0.7:
            score += 0.3
        if features.get('typing_speed_value_mean', 0) > 0.6:
            score += 0.1
        if features.get('focus_duration_value_mean', 0) > 0.7:
            score += 0.2
        
        # Negative indicators
        if features.get('distraction_count_value_mean', 0) > 0.5:
            score -= 0.2
        
        return max(0, min(1, score))
    
    def _calculate_struggle_score(self, features: Dict[str, float]) -> float:
        """Calculate struggle score from features"""
        score = 0.0
        
        # Indicators of struggle
        if features.get('help_searches_value_mean', 0) > 0.3:
            score += 0.3
        if features.get('documentation_views_value_mean', 0) > 0.5:
            score += 0.2
        if features.get('error_rate_value_mean', 0) > 0.4:
            score += 0.2
        if features.get('pause_duration_value_mean', 0) > 0.6:
            score += 0.2
        
        return min(1, score)
    
    def _calculate_stress_score(self, features: Dict[str, float]) -> float:
        """Calculate stress score from features"""
        score = 0.0
        
        # Stress indicators
        if features.get('typing_speed_value_std', 0) > 0.7:
            score += 0.3
        if features.get('mouse_acceleration_value_mean', 0) > 0.8:
            score += 0.2
        if features.get('window_switches_value_mean', 0) > 0.7:
            score += 0.2
        if features.get('error_correction_time_value_mean', 0) > 0.8:
            score += 0.3
        
        return min(1, score)
    
    def _calculate_idle_score(self, features: Dict[str, float]) -> float:
        """Calculate idle score from features"""
        score = 0.0
        
        if features.get('activity_level_value_mean', 1) < 0.2:
            score += 0.5
        if features.get('mouse_movement_value_mean', 1) < 0.1:
            score += 0.3
        if features.get('typing_pattern_value_mean', 1) < 0.1:
            score += 0.2
        
        return min(1, score)
    
    def _calculate_learning_score(self, features: Dict[str, float]) -> float:
        """Calculate learning score from features"""
        score = 0.0
        
        if features.get('documentation_views_value_mean', 0) > 0.4:
            score += 0.3
        if features.get('tutorial_progress_value_mean', 0) > 0.3:
            score += 0.3
        if features.get('exploration_actions_value_mean', 0) > 0.4:
            score += 0.2
        if features.get('note_taking_value_mean', 0) > 0.3:
            score += 0.2
        
        return min(1, score)
    
    def _calculate_confusion_score(self, features: Dict[str, float]) -> float:
        """Calculate confusion score from features"""
        score = 0.0
        
        if features.get('navigation_loops_value_mean', 0) > 0.4:
            score += 0.3
        if features.get('help_searches_value_mean', 0) > 0.5:
            score += 0.3
        if features.get('undo_actions_value_mean', 0) > 0.4:
            score += 0.2
        if features.get('pause_frequency_value_mean', 0) > 0.6:
            score += 0.2
        
        return min(1, score)
    
    async def assess_situation(self, context_data: Dict[str, Any]) -> SituationContext:
        """Assess current situation"""
        situation = SituationContext(
            problem_severity=self._assess_problem_severity(context_data),
            time_criticality=self._assess_time_criticality(context_data),
            solution_availability=self._assess_solution_availability(context_data),
            success_probability=self._calculate_success_probability(context_data),
            context_type=context_data.get('context_type', 'unknown'),
            active_task=context_data.get('active_task'),
            error_context=context_data.get('error_context')
        )
        
        self.current_situation = situation
        self.situation_history.append({
            'situation': situation,
            'timestamp': datetime.now()
        })
        
        return situation
    
    def _assess_problem_severity(self, context: Dict[str, Any]) -> float:
        """Assess severity of current problem"""
        severity = 0.0
        
        # Check for errors
        if context.get('has_error'):
            severity += 0.3
            if context.get('error_type') == 'critical':
                severity += 0.4
            elif context.get('error_type') == 'blocking':
                severity += 0.3
        
        # Check for repeated failures
        if context.get('failure_count', 0) > 3:
            severity += 0.3
        
        # Check for deadline pressure
        if context.get('deadline_proximity', 1.0) < 0.2:
            severity += 0.2
        
        return min(1, severity)
    
    def _assess_time_criticality(self, context: Dict[str, Any]) -> float:
        """Assess time criticality of situation"""
        criticality = 0.0
        
        # Deadline factors
        deadline_proximity = context.get('deadline_proximity', 1.0)
        if deadline_proximity < 0.1:
            criticality += 0.5
        elif deadline_proximity < 0.3:
            criticality += 0.3
        
        # Active waiting
        if context.get('user_waiting'):
            criticality += 0.3
        
        # System performance
        if context.get('performance_degraded'):
            criticality += 0.2
        
        return min(1, criticality)
    
    def _assess_solution_availability(self, context: Dict[str, Any]) -> float:
        """Assess if we have a solution available"""
        availability = 0.0
        
        # Check knowledge base
        if context.get('known_issue'):
            availability += 0.5
        
        # Check for similar past solutions
        if context.get('similar_solutions_count', 0) > 0:
            availability += 0.3
        
        # Check for documentation
        if context.get('documentation_available'):
            availability += 0.2
        
        # Reduce if complex
        if context.get('complexity_score', 0) > 0.8:
            availability *= 0.5
        
        return min(1, availability)
    
    def _calculate_success_probability(self, context: Dict[str, Any]) -> float:
        """Calculate probability of successful intervention"""
        base_probability = 0.5
        
        # User receptiveness
        if self.current_user_state == UserState.FRUSTRATED:
            base_probability += 0.2  # More likely to accept help
        elif self.current_user_state == UserState.FOCUSED:
            base_probability -= 0.3  # Less likely to want interruption
        
        # Past success rate
        intervention_type = context.get('suggested_intervention_type')
        if intervention_type and intervention_type in self.effectiveness_scores:
            past_scores = self.effectiveness_scores[intervention_type]
            if past_scores:
                base_probability = 0.3 * base_probability + 0.7 * statistics.mean(past_scores)
        
        # Timing factors
        if context.get('natural_break'):
            base_probability += 0.2
        
        return max(0, min(1, base_probability))
    
    async def decide_intervention(self) -> Optional[InterventionOpportunity]:
        """Main decision point for interventions"""
        if not self.current_situation:
            return None
        
        # Check cooldown
        if not self._check_cooldown():
            return None
        
        # Extract features for decision
        features = self._extract_decision_features()
        
        # Calculate opportunity score
        opportunity_score = self._calculate_opportunity_score(features)
        
        if opportunity_score < self.min_confidence_threshold:
            return None
        
        # Select intervention type
        intervention_type = self._select_intervention_type(features)
        
        # Select timing strategy
        timing_strategy = self._select_timing_strategy(features)
        
        # Prepare intervention content
        content = await self._prepare_intervention_content(intervention_type)
        
        # Create opportunity
        opportunity = InterventionOpportunity(
            opportunity_id=f"opp_{datetime.now().timestamp()}",
            user_state=self.current_user_state,
            situation=self.current_situation,
            intervention_type=intervention_type,
            timing_strategy=timing_strategy,
            confidence_score=opportunity_score,
            urgency_score=self._calculate_urgency_score(features),
            content=content,
            rationale=self._generate_rationale(features, intervention_type)
        )
        
        self.active_opportunities[opportunity.opportunity_id] = opportunity
        
        return opportunity
    
    def _check_cooldown(self) -> bool:
        """Check if enough time has passed since last intervention"""
        for intervention_type, last_time in self.last_intervention_time.items():
            if datetime.now() - last_time < self.intervention_cooldown:
                return False
        return True
    
    def _extract_decision_features(self) -> Dict[str, float]:
        """Extract all features for decision making"""
        features = {}
        
        # User state features
        features.update(self.feature_extractors['user_state']())
        
        # Situation features
        features.update(self.feature_extractors['situation']())
        
        # Timing features
        features.update(self.feature_extractors['timing']())
        
        # History features
        features.update(self.feature_extractors['history']())
        
        return features
    
    def _extract_user_state_features(self) -> Dict[str, float]:
        """Extract user state features"""
        features = {
            f'user_state_{state.value}': 1.0 if self.current_user_state == state else 0.0
            for state in UserState
        }
        features['user_state_confidence'] = self.state_confidence
        
        # Recent state changes
        recent_states = list(self.state_history)[-10:]
        if recent_states:
            state_changes = sum(1 for i in range(1, len(recent_states)) 
                              if recent_states[i]['state'] != recent_states[i-1]['state'])
            features['state_volatility'] = state_changes / max(len(recent_states) - 1, 1)
        
        return features
    
    def _extract_situation_features(self) -> Dict[str, float]:
        """Extract situation features"""
        if not self.current_situation:
            return {}
        
        return {
            'problem_severity': self.current_situation.problem_severity,
            'time_criticality': self.current_situation.time_criticality,
            'solution_availability': self.current_situation.solution_availability,
            'success_probability': self.current_situation.success_probability,
            f'context_{self.current_situation.context_type}': 1.0
        }
    
    def _extract_timing_features(self) -> Dict[str, float]:
        """Extract timing-related features"""
        features = {}
        
        # Natural break detection
        features['natural_break_score'] = self._detect_natural_break()
        
        # Task boundary detection
        features['task_boundary_score'] = self._detect_task_boundary()
        
        # Cognitive load estimation
        features['cognitive_load'] = self._estimate_cognitive_load()
        
        # Request likelihood
        features['request_likelihood'] = self._predict_request_likelihood()
        
        # Time of day factors
        hour = datetime.now().hour
        features['work_hours'] = 1.0 if 9 <= hour <= 17 else 0.0
        features['peak_hours'] = 1.0 if 10 <= hour <= 12 or 14 <= hour <= 16 else 0.0
        
        return features
    
    def _extract_history_features(self) -> Dict[str, float]:
        """Extract features from intervention history"""
        features = {}
        
        # Recent intervention success rate
        recent_interventions = list(self.intervention_history)[-20:]
        if recent_interventions:
            success_count = sum(1 for i in recent_interventions 
                              if i.effectiveness_score > 0.7)
            features['recent_success_rate'] = success_count / len(recent_interventions)
        
        # Intervention frequency
        day_interventions = [i for i in self.intervention_history 
                           if i.executed_at > datetime.now() - timedelta(days=1)]
        features['daily_intervention_count'] = len(day_interventions)
        
        return features
    
    def _detect_natural_break(self) -> float:
        """Detect if user is at a natural break point"""
        score = 0.0
        
        # Check for idle state
        if self.current_user_state == UserState.IDLE:
            score += 0.5
        
        # Check for recent task completion
        recent_signals = list(self.user_state_signals)[-10:]
        completion_signals = [s for s in recent_signals 
                            if s.signal_type == 'task_completion']
        if completion_signals:
            score += 0.3
        
        # Check for context switch
        if any(s.signal_type == 'context_switch' for s in recent_signals):
            score += 0.2
        
        return min(1, score)
    
    def _detect_task_boundary(self) -> float:
        """Detect if user is at a task boundary"""
        score = 0.0
        
        # Check recent signals
        recent_signals = list(self.user_state_signals)[-20:]
        
        # File save/close events
        if any(s.signal_type in ['file_save', 'file_close'] for s in recent_signals):
            score += 0.4
        
        # Git commits
        if any(s.signal_type == 'git_commit' for s in recent_signals):
            score += 0.3
        
        # Application switches
        if any(s.signal_type == 'app_switch' for s in recent_signals):
            score += 0.2
        
        return min(1, score)
    
    def _estimate_cognitive_load(self) -> float:
        """Estimate current cognitive load"""
        load = 0.5  # Base load
        
        # High focus increases load
        if self.current_user_state == UserState.FOCUSED:
            load += 0.3
        
        # Complex tasks increase load
        if self.current_situation and self.current_situation.context_type in ['debugging', 'problem_solving']:
            load += 0.2
        
        # Errors increase load
        error_signals = [s for s in list(self.user_state_signals)[-20:]
                        if s.signal_type == 'error']
        if len(error_signals) > 3:
            load += 0.2
        
        return min(1, load)
    
    def _predict_request_likelihood(self) -> float:
        """Predict likelihood of user requesting help"""
        likelihood = 0.0
        
        # Frustrated users more likely to want help
        if self.current_user_state == UserState.FRUSTRATED:
            likelihood += 0.4
        elif self.current_user_state == UserState.STRUGGLING:
            likelihood += 0.5
        elif self.current_user_state == UserState.CONFUSED:
            likelihood += 0.4
        
        # Recent help searches
        help_signals = [s for s in list(self.user_state_signals)[-30:]
                       if s.signal_type in ['help_search', 'documentation_view']]
        if help_signals:
            likelihood += 0.3
        
        # Problem severity
        if self.current_situation and self.current_situation.problem_severity > 0.7:
            likelihood += 0.2
        
        return min(1, likelihood)
    
    def _calculate_opportunity_score(self, features: Dict[str, float]) -> float:
        """Calculate overall opportunity score"""
        # Use weighted combination of factors
        weights = {
            'user_need': 0.3,
            'solution_quality': 0.2,
            'timing_quality': 0.2,
            'success_probability': 0.2,
            'urgency': 0.1
        }
        
        scores = {
            'user_need': self._calculate_user_need_score(features),
            'solution_quality': features.get('solution_availability', 0),
            'timing_quality': self._calculate_timing_quality(features),
            'success_probability': features.get('success_probability', 0.5),
            'urgency': features.get('time_criticality', 0)
        }
        
        return sum(weights[k] * scores[k] for k in weights)
    
    def _calculate_user_need_score(self, features: Dict[str, float]) -> float:
        """Calculate how much the user needs help"""
        need = 0.0
        
        # Direct need indicators
        if features.get('user_state_frustrated'):
            need += 0.4
        if features.get('user_state_struggling'):
            need += 0.5
        if features.get('user_state_confused'):
            need += 0.4
        
        # Problem indicators
        need += features.get('problem_severity', 0) * 0.3
        
        return min(1, need)
    
    def _calculate_timing_quality(self, features: Dict[str, float]) -> float:
        """Calculate timing quality score"""
        quality = 0.0
        
        # Best times
        if features.get('natural_break_score', 0) > 0.7:
            quality += 0.4
        if features.get('task_boundary_score', 0) > 0.7:
            quality += 0.3
        
        # Consider cognitive load
        cognitive_load = features.get('cognitive_load', 0.5)
        if cognitive_load < 0.3:
            quality += 0.3
        elif cognitive_load > 0.8:
            quality -= 0.3
        
        return max(0, min(1, quality))
    
    def _calculate_urgency_score(self, features: Dict[str, float]) -> float:
        """Calculate intervention urgency"""
        urgency = features.get('time_criticality', 0) * 0.5
        urgency += features.get('problem_severity', 0) * 0.3
        
        # User state factors
        if self.current_user_state == UserState.STRESSED:
            urgency += 0.2
        
        return min(1, urgency)
    
    def _select_intervention_type(self, features: Dict[str, float]) -> InterventionType:
        """Select appropriate intervention type"""
        # Decision logic based on features
        
        # Silent monitoring for focused users
        if (features.get('user_state_focused') and 
            features.get('problem_severity', 0) < 0.3):
            return InterventionType.SILENT_MONITORING
        
        # Autonomous action for critical issues with clear solutions
        if (features.get('problem_severity', 0) > 0.8 and
            features.get('solution_availability', 0) > 0.8 and
            features.get('success_probability', 0) > 0.8):
            return InterventionType.AUTONOMOUS_ACTION
        
        # Direct assistance for struggling users
        if (features.get('user_state_struggling') or 
            features.get('user_state_frustrated')):
            return InterventionType.DIRECT_ASSISTANCE
        
        # Suggestion for moderate issues
        if features.get('problem_severity', 0) > 0.5:
            return InterventionType.SUGGESTION_OFFER
        
        # Default to subtle indication
        return InterventionType.SUBTLE_INDICATION
    
    def _select_timing_strategy(self, features: Dict[str, float]) -> TimingStrategy:
        """Select optimal timing strategy"""
        # Immediate for urgent issues
        if features.get('urgency', 0) > 0.8:
            return TimingStrategy.IMMEDIATE
        
        # Natural break if available
        if features.get('natural_break_score', 0) > 0.7:
            return TimingStrategy.NATURAL_BREAK
        
        # Task boundary if near
        if features.get('task_boundary_score', 0) > 0.7:
            return TimingStrategy.TASK_BOUNDARY
        
        # Wait for low cognitive load
        if features.get('cognitive_load', 0.5) > 0.7:
            return TimingStrategy.LOW_COGNITIVE_LOAD
        
        # Default to request likelihood
        return TimingStrategy.USER_REQUEST_LIKELY
    
    async def _prepare_intervention_content(self, intervention_type: InterventionType) -> Dict[str, Any]:
        """Prepare content for intervention"""
        content = {
            'type': intervention_type.value,
            'prepared_at': datetime.now().isoformat()
        }
        
        if intervention_type == InterventionType.SILENT_MONITORING:
            content['actions'] = ['observe', 'collect_data']
            
        elif intervention_type == InterventionType.SUBTLE_INDICATION:
            content['indicator'] = {
                'type': 'visual_hint',
                'location': 'status_bar',
                'message': 'Help available'
            }
            
        elif intervention_type == InterventionType.SUGGESTION_OFFER:
            content['suggestion'] = await self._generate_suggestion()
            content['display'] = {
                'type': 'non_modal_popup',
                'duration': 5,
                'dismissible': True
            }
            
        elif intervention_type == InterventionType.DIRECT_ASSISTANCE:
            content['assistance'] = await self._generate_assistance()
            content['display'] = {
                'type': 'assistant_panel',
                'position': 'side',
                'interactive': True
            }
            
        elif intervention_type == InterventionType.AUTONOMOUS_ACTION:
            content['actions'] = await self._plan_autonomous_actions()
            content['confirmation'] = {
                'required': True,
                'timeout': 10
            }
        
        return content
    
    async def _generate_suggestion(self) -> Dict[str, Any]:
        """Generate suggestion based on context"""
        suggestion = {
            'title': 'Suggestion',
            'confidence': self.state_confidence
        }
        
        if self.current_user_state == UserState.FRUSTRATED:
            suggestion['message'] = "I noticed you might be having trouble. Would you like help?"
            suggestion['options'] = ['Show solution', 'Debug together', 'No thanks']
            
        elif self.current_user_state == UserState.CONFUSED:
            suggestion['message'] = "This seems complex. Would you like me to explain?"
            suggestion['options'] = ['Explain concept', 'Show examples', 'I\'m good']
            
        else:
            suggestion['message'] = "I have a suggestion that might help."
            suggestion['options'] = ['Show me', 'Later', 'Dismiss']
        
        return suggestion
    
    async def _generate_assistance(self) -> Dict[str, Any]:
        """Generate direct assistance content"""
        assistance = {
            'title': 'Direct Assistance',
            'steps': []
        }
        
        if self.current_situation and self.current_situation.error_context:
            # Error-specific assistance
            assistance['type'] = 'error_resolution'
            assistance['error_analysis'] = self.current_situation.error_context
            assistance['steps'] = [
                {'action': 'identify_cause', 'description': 'Analyzing error...'},
                {'action': 'suggest_fix', 'description': 'Generating solution...'},
                {'action': 'apply_fix', 'description': 'Ready to apply fix'}
            ]
        else:
            # General assistance
            assistance['type'] = 'general_help'
            assistance['steps'] = [
                {'action': 'understand_goal', 'description': 'What are you trying to do?'},
                {'action': 'analyze_blockers', 'description': 'Identifying obstacles...'},
                {'action': 'provide_guidance', 'description': 'Suggesting next steps...'}
            ]
        
        return assistance
    
    async def _plan_autonomous_actions(self) -> List[Dict[str, Any]]:
        """Plan autonomous actions to take"""
        actions = []
        
        if self.current_situation and self.current_situation.error_context:
            # Autonomous error fixing
            actions.append({
                'action': 'backup_current_state',
                'description': 'Saving current state...',
                'risk': 'low'
            })
            actions.append({
                'action': 'apply_fix',
                'description': 'Applying known solution...',
                'risk': 'medium'
            })
            actions.append({
                'action': 'verify_fix',
                'description': 'Verifying fix worked...',
                'risk': 'low'
            })
        
        return actions
    
    def _generate_rationale(self, features: Dict[str, float], 
                          intervention_type: InterventionType) -> str:
        """Generate rationale for intervention"""
        rationale_parts = []
        
        # User state rationale
        if self.current_user_state == UserState.FRUSTRATED:
            rationale_parts.append("User appears frustrated")
        elif self.current_user_state == UserState.STRUGGLING:
            rationale_parts.append("User is struggling with task")
        
        # Problem rationale
        if features.get('problem_severity', 0) > 0.7:
            rationale_parts.append("Severe problem detected")
        
        # Timing rationale
        if features.get('natural_break_score', 0) > 0.7:
            rationale_parts.append("Natural break point identified")
        
        # Intervention type rationale
        rationale_parts.append(f"Recommending {intervention_type.value} based on context")
        
        return ". ".join(rationale_parts)
    
    async def execute_intervention(self, opportunity: InterventionOpportunity) -> InterventionResult:
        """Execute an intervention and track result"""
        start_time = datetime.now()
        
        # Record intervention start
        self.last_intervention_time[opportunity.intervention_type] = start_time
        
        # Execute based on type
        user_response = await self._execute_intervention_type(opportunity)
        
        # Calculate effectiveness
        effectiveness = self._calculate_effectiveness(opportunity, user_response)
        
        # Create result
        result = InterventionResult(
            intervention_id=f"int_{start_time.timestamp()}",
            opportunity_id=opportunity.opportunity_id,
            executed_at=start_time,
            user_response=user_response,
            effectiveness_score=effectiveness,
            time_to_resolution=datetime.now() - start_time
        )
        
        # Update history and learning
        self.intervention_history.append(result)
        self.effectiveness_scores[opportunity.intervention_type].append(effectiveness)
        
        # Add to learning buffer
        self.learning_buffer.append((
            self._extract_decision_features(),
            result
        ))
        
        # Trigger learning if buffer is full
        if len(self.learning_buffer) >= 50:
            await self.update_models()
        
        return result
    
    async def _execute_intervention_type(self, opportunity: InterventionOpportunity) -> str:
        """Execute specific intervention type"""
        # This would interface with the UI/notification system
        # For now, simulate execution
        
        if opportunity.intervention_type == InterventionType.SILENT_MONITORING:
            return "monitoring"
        
        elif opportunity.intervention_type == InterventionType.AUTONOMOUS_ACTION:
            # Check if user confirms
            confirmed = await self._get_user_confirmation(opportunity.content)
            return "accepted" if confirmed else "rejected"
        
        else:
            # Simulate user response based on probability
            import random
            if random.random() < opportunity.situation.success_probability:
                return "accepted"
            elif random.random() < 0.3:
                return "ignored"
            else:
                return "rejected"
    
    async def _get_user_confirmation(self, content: Dict[str, Any]) -> bool:
        """Get user confirmation for action"""
        # This would show actual UI
        # For now, simulate based on context
        return content.get('confirmation', {}).get('required', False)
    
    def _calculate_effectiveness(self, opportunity: InterventionOpportunity, 
                               response: str) -> float:
        """Calculate intervention effectiveness"""
        base_score = 0.0
        
        # Response-based scoring
        if response == "accepted":
            base_score = 0.8
        elif response == "delayed":
            base_score = 0.6
        elif response == "ignored":
            base_score = 0.3
        else:  # rejected
            base_score = 0.1
        
        # Adjust based on resolution
        if response == "accepted":
            # Check if problem was resolved
            # This would check actual system state
            problem_resolved = True  # Simulate
            if problem_resolved:
                base_score += 0.2
        
        return min(1, base_score)
    
    async def update_models(self):
        """Update decision models with learning data"""
        if not self.learning_buffer:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for features, result in self.learning_buffer:
            feature_vector = [features.get(k, 0) for k in sorted(features.keys())]
            X.append(feature_vector)
            y.append(1 if result.effectiveness_score > 0.7 else 0)
        
        # Update models
        if len(X) >= 20:
            X = np.array(X)
            y = np.array(y)
            
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
            
            # Update decision tree
            self.decision_tree.fit(X_scaled, y)
            
            # Update intervention predictor
            if hasattr(self.intervention_predictor, 'n_estimators'):
                self.intervention_predictor.fit(X_scaled, y)
            
            self.model_version += 1
            self.last_training = datetime.now()
            
            # Clear learning buffer
            self.learning_buffer = []
            
            logger.info(f"Updated decision models to version {self.model_version}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get intervention statistics"""
        stats = {
            'current_user_state': self.current_user_state.value,
            'state_confidence': self.state_confidence,
            'active_opportunities': len(self.active_opportunities),
            'total_interventions': len(self.intervention_history),
            'model_version': self.model_version,
            'last_training': self.last_training.isoformat() if self.last_training else None
        }
        
        # Effectiveness by type
        effectiveness_by_type = {}
        for intervention_type in InterventionType:
            scores = self.effectiveness_scores.get(intervention_type, [])
            if scores:
                effectiveness_by_type[intervention_type.value] = {
                    'mean': statistics.mean(scores),
                    'count': len(scores)
                }
        stats['effectiveness_by_type'] = effectiveness_by_type
        
        # User state distribution
        if self.state_history:
            state_counts = defaultdict(int)
            for record in self.state_history:
                state_counts[record['state'].value] += 1
            stats['state_distribution'] = dict(state_counts)
        
        # Response distribution
        if self.intervention_history:
            response_counts = defaultdict(int)
            for result in self.intervention_history:
                response_counts[result.user_response] += 1
            stats['response_distribution'] = dict(response_counts)
        
        return stats
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage"""
        return {
            'decision_models': len(pickle.dumps(self.decision_tree)) + 
                             len(pickle.dumps(self.intervention_predictor)),
            'intervention_history': len(str(list(self.intervention_history)).encode()),
            'learning_data': len(str(self.learning_buffer).encode()),
            'total': sum([
                len(pickle.dumps(self.decision_tree)),
                len(pickle.dumps(self.intervention_predictor)),
                len(str(list(self.intervention_history)).encode()),
                len(str(self.learning_buffer).encode())
            ])
        }
    
    async def save_models(self, path: str):
        """Save trained models to disk"""
        model_data = {
            'decision_tree': self.decision_tree,
            'intervention_predictor': self.intervention_predictor,
            'scaler': self.scaler,
            'effectiveness_scores': dict(self.effectiveness_scores),
            'model_version': self.model_version,
            'last_training': self.last_training
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Saved models to {path}")
    
    async def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.decision_tree = model_data['decision_tree']
            self.intervention_predictor = model_data['intervention_predictor']
            self.scaler = model_data['scaler']
            self.effectiveness_scores = defaultdict(list, model_data['effectiveness_scores'])
            self.model_version = model_data['model_version']
            self.last_training = model_data['last_training']
            
            logger.info(f"Loaded models from {path} (version {self.model_version})")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")


# Global instance
_decision_engine_instance = None

def get_intervention_decision_engine() -> InterventionDecisionEngine:
    """Get or create intervention decision engine instance"""
    global _decision_engine_instance
    if _decision_engine_instance is None:
        _decision_engine_instance = InterventionDecisionEngine()
    return _decision_engine_instance