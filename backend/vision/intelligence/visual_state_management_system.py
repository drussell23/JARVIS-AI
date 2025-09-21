"""
Visual State Management System (VSMS) - Core Intelligence Foundation
Dynamic, learning-based state understanding without hardcoding

This system transforms raw visual data into meaningful state understanding
by learning patterns, signatures, and transitions from observation.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from pathlib import Path
import numpy as np
from collections import defaultdict, deque
import hashlib

logger = logging.getLogger(__name__)


class StateType(Enum):
    """Dynamic state types discovered through observation"""
    UNKNOWN = auto()
    IDLE = auto()
    ACTIVE = auto()
    LOADING = auto()
    ERROR = auto()
    MODAL = auto()
    TRANSITION = auto()
    CUSTOM = auto()  # For learned states


@dataclass
class VisualSignature:
    """Learned visual patterns that identify states"""
    signature_id: str
    feature_vectors: List[np.ndarray] = field(default_factory=list)
    color_patterns: Dict[str, Any] = field(default_factory=dict)
    structural_patterns: Dict[str, Any] = field(default_factory=dict)
    text_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0
    occurrence_count: int = 0
    last_seen: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_confidence(self, match_score: float):
        """Dynamically update confidence based on matches"""
        self.confidence = (self.confidence * 0.9) + (match_score * 0.1)
        self.occurrence_count += 1
        self.last_seen = datetime.now()


@dataclass
class ApplicationState:
    """Dynamic application state representation"""
    state_id: str
    state_type: StateType
    signatures: List[VisualSignature] = field(default_factory=list)
    transitions_to: Dict[str, float] = field(default_factory=dict)  # state_id -> probability
    transitions_from: Dict[str, float] = field(default_factory=dict)
    duration_stats: Dict[str, float] = field(default_factory=dict)  # min, max, avg
    context_patterns: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_observed: Optional[datetime] = None
    observation_count: int = 0
    
    def add_transition(self, to_state: str, from_state: Optional[str] = None):
        """Learn state transitions"""
        if to_state:
            self.transitions_to[to_state] = self.transitions_to.get(to_state, 0) + 1
        if from_state:
            self.transitions_from[from_state] = self.transitions_from.get(from_state, 0) + 1
        self._normalize_transitions()
    
    def _normalize_transitions(self):
        """Convert counts to probabilities"""
        for transitions in [self.transitions_to, self.transitions_from]:
            total = sum(transitions.values())
            if total > 0:
                for state_id in transitions:
                    transitions[state_id] /= total


@dataclass
class StateObservation:
    """Single observation of application state"""
    timestamp: datetime
    visual_features: Dict[str, Any]
    detected_elements: List[Dict[str, Any]]
    text_content: List[str]
    interaction_hints: List[str]
    screenshot_hash: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateDetector(ABC):
    """Abstract base for pluggable state detection strategies"""
    
    @abstractmethod
    async def detect_state(self, observation: StateObservation) -> Tuple[Optional[str], float]:
        """Detect state from observation, return (state_id, confidence)"""
        pass
    
    @abstractmethod
    def learn_from_observation(self, observation: StateObservation, state_id: str):
        """Learn patterns from confirmed state observation"""
        pass


class PatternBasedStateDetector(StateDetector):
    """Detects states by learning visual patterns"""
    
    def __init__(self):
        self.learned_patterns = defaultdict(list)
        self.pattern_confidence = defaultdict(float)
        
    async def detect_state(self, observation: StateObservation) -> Tuple[Optional[str], float]:
        """Match observation against learned patterns"""
        best_match = None
        best_confidence = 0.0
        
        # Extract pattern features
        pattern_key = self._extract_pattern_key(observation)
        
        # Compare against learned patterns
        for state_id, patterns in self.learned_patterns.items():
            similarity = self._calculate_similarity(pattern_key, patterns)
            if similarity > best_confidence:
                best_confidence = similarity
                best_match = state_id
        
        return best_match, best_confidence
    
    def learn_from_observation(self, observation: StateObservation, state_id: str):
        """Learn pattern from observation"""
        pattern_key = self._extract_pattern_key(observation)
        self.learned_patterns[state_id].append(pattern_key)
        
        # Keep only recent patterns to adapt to changes
        max_patterns = 100
        if len(self.learned_patterns[state_id]) > max_patterns:
            self.learned_patterns[state_id] = self.learned_patterns[state_id][-max_patterns:]
    
    def _extract_pattern_key(self, observation: StateObservation) -> Dict[str, Any]:
        """Extract pattern features from observation"""
        return {
            'element_count': len(observation.detected_elements),
            'text_patterns': [text[:50] for text in observation.text_content[:10]],
            'visual_summary': observation.visual_features.get('summary', ''),
            'interaction_types': observation.interaction_hints,
            'timestamp': observation.timestamp.isoformat()
        }
    
    def _calculate_similarity(self, pattern: Dict[str, Any], 
                            learned_patterns: List[Dict[str, Any]]) -> float:
        """Calculate similarity between patterns"""
        if not learned_patterns:
            return 0.0
        
        similarities = []
        for learned in learned_patterns:
            sim = 0.0
            # Compare element counts
            if pattern['element_count'] == learned['element_count']:
                sim += 0.3
            
            # Compare text patterns
            text_sim = len(set(pattern['text_patterns']) & set(learned['text_patterns']))
            text_sim /= max(len(pattern['text_patterns']), 1)
            sim += text_sim * 0.4
            
            # Compare interaction types
            int_sim = len(set(pattern['interaction_types']) & set(learned['interaction_types']))
            int_sim /= max(len(pattern['interaction_types']), 1)
            sim += int_sim * 0.3
            
            similarities.append(sim)
        
        return max(similarities)


class ApplicationStateTracker:
    """Tracks and learns application states dynamically"""
    
    def __init__(self, app_identifier: str):
        self.app_id = app_identifier
        self.states: Dict[str, ApplicationState] = {}
        self.current_state: Optional[str] = None
        self.state_history: deque = deque(maxlen=1000)
        self.detectors: List[StateDetector] = [PatternBasedStateDetector()]
        self.learning_mode = True
        self.confidence_threshold = 0.7
        
        # Learning buffers
        self.observation_buffer: deque = deque(maxlen=50)
        self.transition_buffer: List[Tuple[str, str, datetime]] = []
        
        # State persistence
        self.state_db_path = Path(f"learned_states/{app_identifier}_states.json")
        self.state_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_learned_states()
    
    async def observe(self, visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual observation and detect state"""
        # Create observation
        observation = self._create_observation(visual_data)
        self.observation_buffer.append(observation)
        
        # Detect state
        detected_state, confidence = await self._detect_state(observation)
        
        # Handle state detection
        if detected_state:
            if confidence >= self.confidence_threshold:
                await self._handle_confirmed_state(detected_state, observation)
            else:
                await self._handle_uncertain_state(detected_state, observation, confidence)
        else:
            # No state detected - potential new state
            if self.learning_mode:
                await self._discover_new_state(observation)
        
        return {
            'state_id': detected_state,
            'confidence': confidence,
            'current_state': self.current_state,
            'learning_active': self.learning_mode,
            'total_states': len(self.states),
            'observation_id': observation.timestamp.isoformat()
        }
    
    async def _detect_state(self, observation: StateObservation) -> Tuple[Optional[str], float]:
        """Use multiple detectors to identify state"""
        results = []
        
        for detector in self.detectors:
            state_id, confidence = await detector.detect_state(observation)
            if state_id:
                results.append((state_id, confidence))
        
        if not results:
            return None, 0.0
        
        # Aggregate results (could be more sophisticated)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[0]
    
    async def _handle_confirmed_state(self, state_id: str, observation: StateObservation):
        """Handle state with high confidence"""
        previous_state = self.current_state
        self.current_state = state_id
        
        # Update state information
        if state_id not in self.states:
            self.states[state_id] = ApplicationState(
                state_id=state_id,
                state_type=StateType.CUSTOM
            )
        
        state = self.states[state_id]
        state.last_observed = datetime.now()
        state.observation_count += 1
        
        # Record transition
        if previous_state and previous_state != state_id:
            state.add_transition(state_id, previous_state)
            self.transition_buffer.append((previous_state, state_id, datetime.now()))
        
        # Learn from observation
        for detector in self.detectors:
            detector.learn_from_observation(observation, state_id)
        
        # Add to history
        self.state_history.append({
            'state_id': state_id,
            'timestamp': datetime.now(),
            'confidence': observation.confidence
        })
    
    async def _handle_uncertain_state(self, state_id: str, 
                                    observation: StateObservation, 
                                    confidence: float):
        """Handle state detection with low confidence"""
        # Store for later analysis
        uncertain_observation = {
            'potential_state': state_id,
            'confidence': confidence,
            'observation': observation,
            'timestamp': datetime.now()
        }
        
        # Could implement confirmation mechanisms here
        logger.info(f"Uncertain state detected: {state_id} (confidence: {confidence:.2f})")
    
    async def _discover_new_state(self, observation: StateObservation):
        """Discover and learn new states"""
        # Generate unique state ID based on observation
        state_features = {
            'element_count': len(observation.detected_elements),
            'primary_text': observation.text_content[:3] if observation.text_content else [],
            'visual_hash': observation.screenshot_hash
        }
        
        state_id = self._generate_state_id(state_features)
        
        # Check if this is truly new or a variation
        if state_id not in self.states:
            logger.info(f"Discovered new state: {state_id}")
            self.states[state_id] = ApplicationState(
                state_id=state_id,
                state_type=StateType.CUSTOM
            )
            
            # Learn this new state
            for detector in self.detectors:
                detector.learn_from_observation(observation, state_id)
    
    def _create_observation(self, visual_data: Dict[str, Any]) -> StateObservation:
        """Create observation from visual data"""
        return StateObservation(
            timestamp=datetime.now(),
            visual_features=visual_data.get('features', {}),
            detected_elements=visual_data.get('elements', []),
            text_content=visual_data.get('text', []),
            interaction_hints=visual_data.get('interactions', []),
            screenshot_hash=self._hash_screenshot(visual_data.get('screenshot')),
            confidence=visual_data.get('confidence', 0.0),
            metadata=visual_data.get('metadata', {})
        )
    
    def _generate_state_id(self, features: Dict[str, Any]) -> str:
        """Generate unique state ID from features"""
        feature_str = json.dumps(features, sort_keys=True)
        return f"state_{hashlib.md5(feature_str.encode()).hexdigest()[:8]}"
    
    def _hash_screenshot(self, screenshot_data: Any) -> Optional[str]:
        """Generate hash of screenshot for comparison"""
        if screenshot_data is None:
            return None
        # Implement perceptual hashing for better similarity detection
        # For now, simple hash
        return hashlib.md5(str(screenshot_data).encode()).hexdigest()[:16]
    
    def _load_learned_states(self):
        """Load previously learned states"""
        if self.state_db_path.exists():
            try:
                with open(self.state_db_path, 'r') as f:
                    data = json.load(f)
                    # Reconstruct states from saved data
                    logger.info(f"Loaded {len(data.get('states', {}))} learned states")
            except Exception as e:
                logger.error(f"Failed to load learned states: {e}")
    
    def save_learned_states(self):
        """Persist learned states"""
        try:
            data = {
                'app_id': self.app_id,
                'states': {
                    state_id: {
                        'type': state.state_type.name,
                        'observation_count': state.observation_count,
                        'transitions': state.transitions_to,
                        'created': state.created_at.isoformat(),
                        'last_seen': state.last_observed.isoformat() if state.last_observed else None
                    }
                    for state_id, state in self.states.items()
                },
                'saved_at': datetime.now().isoformat()
            }
            
            with open(self.state_db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved {len(self.states)} learned states")
        except Exception as e:
            logger.error(f"Failed to save learned states: {e}")
    
    def get_state_insights(self) -> Dict[str, Any]:
        """Get insights about learned states"""
        if not self.states:
            return {'message': 'No states learned yet'}
        
        return {
            'total_states': len(self.states),
            'current_state': self.current_state,
            'most_common_states': self._get_most_common_states(),
            'state_transitions': self._get_transition_insights(),
            'learning_progress': {
                'observations': len(self.observation_buffer),
                'confidence_avg': self._calculate_avg_confidence()
            }
        }
    
    def _get_most_common_states(self) -> List[Dict[str, Any]]:
        """Get most frequently observed states"""
        sorted_states = sorted(
            self.states.items(), 
            key=lambda x: x[1].observation_count, 
            reverse=True
        )[:5]
        
        return [
            {
                'state_id': state_id,
                'count': state.observation_count,
                'type': state.state_type.name
            }
            for state_id, state in sorted_states
        ]
    
    def _get_transition_insights(self) -> Dict[str, Any]:
        """Analyze state transitions"""
        total_transitions = len(self.transition_buffer)
        if total_transitions == 0:
            return {'message': 'No transitions observed yet'}
        
        # Find common transition patterns
        transition_counts = defaultdict(int)
        for from_state, to_state, _ in self.transition_buffer:
            transition_counts[f"{from_state} → {to_state}"] += 1
        
        common_transitions = sorted(
            transition_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            'total_transitions': total_transitions,
            'common_patterns': [
                {'transition': t, 'count': c} 
                for t, c in common_transitions
            ]
        }
    
    def _calculate_avg_confidence(self) -> float:
        """Calculate average confidence of recent detections"""
        if not self.state_history:
            return 0.0
        
        recent = list(self.state_history)[-10:]
        confidences = [h.get('confidence', 0) for h in recent]
        return sum(confidences) / len(confidences) if confidences else 0.0


class VisualStateManagementSystem:
    """Main VSMS coordinator - manages state tracking across applications"""
    
    def __init__(self):
        self.app_trackers: Dict[str, ApplicationStateTracker] = {}
        self.global_patterns: Dict[str, Any] = {}
        self.system_insights: Dict[str, Any] = {}
        self.learning_enabled = True
        
    async def process_visual_input(self, app_id: str, visual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process visual input for an application"""
        # Get or create tracker for this app
        if app_id not in self.app_trackers:
            self.app_trackers[app_id] = ApplicationStateTracker(app_id)
            logger.info(f"Created new state tracker for: {app_id}")
        
        tracker = self.app_trackers[app_id]
        
        # Process observation
        result = await tracker.observe(visual_data)
        
        # Update global patterns if in learning mode
        if self.learning_enabled:
            self._update_global_patterns(app_id, result)
        
        return {
            'app_id': app_id,
            'state_analysis': result,
            'tracker_insights': tracker.get_state_insights()
        }
    
    def _update_global_patterns(self, app_id: str, result: Dict[str, Any]):
        """Learn patterns that apply across applications"""
        # This could identify common UI patterns like:
        # - Loading states
        # - Error dialogs
        # - Modal windows
        # - Navigation patterns
        pass
    
    def get_system_insights(self) -> Dict[str, Any]:
        """Get insights across all tracked applications"""
        return {
            'tracked_applications': list(self.app_trackers.keys()),
            'total_states_learned': sum(
                len(tracker.states) for tracker in self.app_trackers.values()
            ),
            'learning_enabled': self.learning_enabled,
            'app_insights': {
                app_id: tracker.get_state_insights()
                for app_id, tracker in self.app_trackers.items()
            }
        }
    
    def save_all_states(self):
        """Save learned states for all applications"""
        for app_id, tracker in self.app_trackers.items():
            tracker.save_learned_states()
            
    def enable_learning(self, enabled: bool = True):
        """Enable or disable learning mode"""
        self.learning_enabled = enabled
        for tracker in self.app_trackers.values():
            tracker.learning_mode = enabled