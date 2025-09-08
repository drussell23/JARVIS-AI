"""
Visual State Management System (VSMS) Core
Complete implementation of the Application State Model architecture
Memory-optimized for 150MB allocation (50MB per component)
"""

import asyncio
import json
import logging
import mmap
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import psutil

logger = logging.getLogger(__name__)


# Memory management constants
MEMORY_LIMITS = {
    'state_definitions': 50 * 1024 * 1024,  # 50MB
    'transition_history': 50 * 1024 * 1024,  # 50MB
    'pattern_storage': 50 * 1024 * 1024,     # 50MB
}


class StateCategory(Enum):
    """Core state categories for all applications"""
    STARTUP = auto()
    IDLE = auto()
    ACTIVE = auto()
    LOADING = auto()
    MODAL = auto()
    ERROR = auto()
    TRANSITION = auto()
    SHUTDOWN = auto()
    CUSTOM = auto()


class ModalType(Enum):
    """Types of modal states"""
    DIALOG = auto()
    POPUP = auto()
    TOOLTIP = auto()
    MENU = auto()
    OVERLAY = auto()
    ALERT = auto()
    CONFIRMATION = auto()
    INPUT = auto()


@dataclass
class ApplicationIdentity:
    """Identity Layer - Application identification and versioning"""
    app_id: str
    name: str
    detected_version: Optional[str] = None
    icon_signature: Optional[str] = None
    color_palette: List[Tuple[int, int, int]] = field(default_factory=list)
    chrome_pattern: Dict[str, Any] = field(default_factory=dict)
    configuration_hash: Optional[str] = None
    customization_level: float = 0.0  # 0-1, how customized from default
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_signature(self, visual_data: Dict[str, Any]):
        """Update identity from visual observation"""
        if 'icon' in visual_data:
            self.icon_signature = self._hash_icon(visual_data['icon'])
        if 'colors' in visual_data:
            self.color_palette = visual_data['colors'][:5]  # Top 5 colors
        if 'chrome' in visual_data:
            self.chrome_pattern = visual_data['chrome']
        self.last_updated = datetime.now()
    
    def _hash_icon(self, icon_data: Any) -> str:
        """Create perceptual hash of icon"""
        # Simplified - in production use proper perceptual hashing
        import hashlib
        return hashlib.md5(str(icon_data).encode()).hexdigest()[:16]


@dataclass
class ApplicationState:
    """State Layer - Current operational state"""
    state_id: str
    category: StateCategory
    name: str
    visual_signatures: List[Dict[str, Any]] = field(default_factory=list)
    confidence_threshold: float = 0.7
    
    # Modal state information
    is_modal: bool = False
    modal_type: Optional[ModalType] = None
    modal_parent_state: Optional[str] = None
    
    # State characteristics
    is_interruptible: bool = True
    is_terminal: bool = False
    expected_duration: Optional[timedelta] = None
    timeout_duration: Optional[timedelta] = None
    
    # Transition information
    valid_transitions: Set[str] = field(default_factory=set)
    common_next_states: List[Tuple[str, float]] = field(default_factory=list)  # (state_id, probability)
    
    # Detection metadata
    detection_count: int = 0
    false_positive_count: int = 0
    last_detected: Optional[datetime] = None
    average_duration: Optional[timedelta] = None
    
    def add_visual_signature(self, signature: Dict[str, Any]):
        """Add a visual signature for this state"""
        self.visual_signatures.append({
            'signature': signature,
            'added_at': datetime.now(),
            'confidence': 1.0
        })
        # Keep only most recent signatures
        if len(self.visual_signatures) > 10:
            self.visual_signatures = self.visual_signatures[-10:]
    
    def update_statistics(self, duration: Optional[timedelta] = None):
        """Update state statistics"""
        self.detection_count += 1
        self.last_detected = datetime.now()
        
        if duration and self.average_duration:
            # Running average
            alpha = 0.1
            self.average_duration = timedelta(
                seconds=(1 - alpha) * self.average_duration.total_seconds() + 
                        alpha * duration.total_seconds()
            )
        elif duration:
            self.average_duration = duration


@dataclass
class ContentContext:
    """Content Layer - What's being worked on"""
    content_id: str
    content_type: str  # document, webpage, media, etc.
    title: Optional[str] = None
    path: Optional[str] = None
    summary: Optional[str] = None
    
    # Modification tracking
    is_modified: bool = False
    last_modification: Optional[datetime] = None
    modification_indicators: List[str] = field(default_factory=list)
    
    # Interaction possibilities
    available_actions: List[str] = field(default_factory=list)
    keyboard_shortcuts: Dict[str, str] = field(default_factory=dict)
    clickable_regions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Content understanding
    key_elements: List[str] = field(default_factory=list)
    extracted_text: Optional[str] = None
    semantic_tags: List[str] = field(default_factory=list)


@dataclass
class StateTransition:
    """State transition record"""
    from_state: str
    to_state: str
    timestamp: datetime
    trigger: Optional[str] = None  # What caused the transition
    duration_in_from_state: Optional[timedelta] = None
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateHistory:
    """Historical Layer - Manages state history with memory constraints"""
    
    def __init__(self, max_memory_mb: int = 50):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.transitions: Deque[StateTransition] = deque(maxlen=10000)
        self.state_durations: Dict[str, List[timedelta]] = defaultdict(list)
        self.patterns: Dict[str, int] = defaultdict(int)  # Pattern -> count
        self.anomalies: List[Dict[str, Any]] = []
        self.recovery_paths: Dict[str, List[str]] = defaultdict(list)
        
        # Memory-mapped file for large history
        self.history_file = Path("vsms_history.dat")
        self._init_memory_map()
    
    def _init_memory_map(self):
        """Initialize memory-mapped file for efficient storage"""
        if not self.history_file.exists():
            # Create file with initial size
            with open(self.history_file, 'wb') as f:
                f.write(b'\x00' * 1024 * 1024)  # 1MB initial
        
        self.mmap_file = open(self.history_file, 'r+b')
        self.mmap = mmap.mmap(self.mmap_file.fileno(), 0)
    
    def add_transition(self, transition: StateTransition):
        """Add a state transition to history"""
        self.transitions.append(transition)
        
        # Update patterns
        pattern_key = f"{transition.from_state}->{transition.to_state}"
        self.patterns[pattern_key] += 1
        
        # Track duration
        if transition.duration_in_from_state:
            self.state_durations[transition.from_state].append(transition.duration_in_from_state)
            # Keep only recent durations
            if len(self.state_durations[transition.from_state]) > 100:
                self.state_durations[transition.from_state] = self.state_durations[transition.from_state][-100:]
        
        # Check for anomalies
        if not transition.success or self._is_anomalous(transition):
            self.anomalies.append({
                'transition': transition,
                'reason': self._get_anomaly_reason(transition)
            })
        
        # Update recovery paths for error states
        if transition.from_state.startswith('error_') and transition.success:
            self.recovery_paths[transition.from_state].append(transition.to_state)
    
    def _is_anomalous(self, transition: StateTransition) -> bool:
        """Detect anomalous transitions"""
        # Quick duration check
        if transition.duration_in_from_state:
            avg_duration = self.get_average_duration(transition.from_state)
            if avg_duration:
                ratio = transition.duration_in_from_state / avg_duration
                if ratio < 0.1 or ratio > 10:  # 10x faster or slower
                    return True
        
        # Rare transition check
        pattern_key = f"{transition.from_state}->{transition.to_state}"
        if self.patterns[pattern_key] == 1 and len(self.transitions) > 100:
            return True
        
        return False
    
    def _get_anomaly_reason(self, transition: StateTransition) -> str:
        """Get reason for anomaly"""
        reasons = []
        
        if not transition.success:
            reasons.append("failed_transition")
        
        if transition.duration_in_from_state:
            avg_duration = self.get_average_duration(transition.from_state)
            if avg_duration:
                ratio = transition.duration_in_from_state / avg_duration
                if ratio < 0.1:
                    reasons.append("unusually_fast")
                elif ratio > 10:
                    reasons.append("unusually_slow")
        
        pattern_key = f"{transition.from_state}->{transition.to_state}"
        if self.patterns[pattern_key] == 1:
            reasons.append("rare_transition")
        
        return ", ".join(reasons) if reasons else "unknown"
    
    def get_average_duration(self, state: str) -> Optional[timedelta]:
        """Get average duration for a state"""
        durations = self.state_durations.get(state, [])
        if not durations:
            return None
        
        total_seconds = sum(d.total_seconds() for d in durations)
        return timedelta(seconds=total_seconds / len(durations))
    
    def get_common_patterns(self, min_count: int = 5) -> List[Tuple[str, int]]:
        """Get common transition patterns"""
        return [(pattern, count) for pattern, count in self.patterns.items() 
                if count >= min_count]
    
    def get_stuck_states(self, threshold: timedelta = timedelta(minutes=5)) -> List[str]:
        """Identify states where users get stuck"""
        stuck_states = []
        
        for state, durations in self.state_durations.items():
            if durations:
                avg_duration = sum(d.total_seconds() for d in durations) / len(durations)
                if avg_duration > threshold.total_seconds():
                    stuck_states.append(state)
        
        return stuck_states
    
    def predict_next_state(self, current_state: str, n: int = 3) -> List[Tuple[str, float]]:
        """Predict most likely next states"""
        transitions_from = [t for t in self.transitions if t.from_state == current_state]
        
        if not transitions_from:
            return []
        
        next_state_counts = defaultdict(int)
        for t in transitions_from:
            next_state_counts[t.to_state] += 1
        
        total = sum(next_state_counts.values())
        predictions = [(state, count/total) for state, count in next_state_counts.items()]
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:n]
    
    def save_to_disk(self):
        """Save history to memory-mapped file"""
        try:
            data = {
                'transitions': list(self.transitions)[-1000:],  # Keep last 1000
                'patterns': dict(self.patterns),
                'anomalies': self.anomalies[-100:],  # Keep last 100
                'recovery_paths': dict(self.recovery_paths)
            }
            
            serialized = pickle.dumps(data)
            if len(serialized) > self.mmap.size():
                # Need to resize
                self.mmap.close()
                self.mmap_file.close()
                
                with open(self.history_file, 'wb') as f:
                    f.write(serialized)
                
                self._init_memory_map()
            else:
                self.mmap.seek(0)
                self.mmap.write(serialized)
                self.mmap.flush()
                
        except Exception as e:
            logger.error(f"Failed to save history: {e}")


class VisualStateManager:
    """Main VSMS coordinator"""
    
    def __init__(self):
        self.applications: Dict[str, ApplicationIdentity] = {}
        self.states: Dict[str, Dict[str, ApplicationState]] = {}  # app_id -> state_id -> state
        self.current_states: Dict[str, str] = {}  # app_id -> current_state_id
        self.current_state_timestamps: Dict[str, datetime] = {}  # Track when entered current state
        self.state_history = StateHistory()
        self.content_contexts: Dict[str, ContentContext] = {}  # app_id -> content
        
        # State detection pipeline
        from .state_detection_pipeline import StateDetectionPipeline
        self.detection_pipeline = StateDetectionPipeline()
        
        # State intelligence
        from .state_intelligence import get_state_intelligence
        self.state_intelligence = get_state_intelligence()
        
        # Semantic Scene Graph
        from .semantic_scene_graph import get_scene_graph
        self.scene_graph = get_scene_graph()
        
        # Element detector for scene graph
        from .element_detector import ElementDetector
        self.element_detector = ElementDetector()
        
        # Memory monitoring
        self.memory_usage = {
            'state_definitions': 0,
            'transition_history': 0,
            'pattern_storage': 0,
            'scene_graph': 0
        }
        
        # Load saved state definitions
        self._load_state_definitions()
    
    def _load_state_definitions(self):
        """Load state definitions from disk"""
        state_file = Path("vsms_states.json")
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    # Reconstruct state objects
                    for app_id, states in data.get('states', {}).items():
                        self.states[app_id] = {}
                        for state_id, state_data in states.items():
                            state = ApplicationState(
                                state_id=state_id,
                                category=StateCategory[state_data['category']],
                                name=state_data['name']
                            )
                            # Restore other fields...
                            self.states[app_id][state_id] = state
                            
                logger.info(f"Loaded {len(self.states)} application states")
            except Exception as e:
                logger.error(f"Failed to load state definitions: {e}")
    
    async def process_visual_observation(self, 
                                       screenshot: np.ndarray,
                                       app_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a visual observation and update state"""
        # Memory check
        if not self._check_memory_limits():
            logger.warning("Memory limits exceeded, cleaning up...")
            self._cleanup_memory()
        
        # Extract application boundaries if not provided
        if not app_id:
            app_id = await self.detection_pipeline.detect_application(screenshot)
            if not app_id:
                app_id = "unknown"
        
        # Get or create application identity
        if app_id not in self.applications:
            self.applications[app_id] = ApplicationIdentity(
                app_id=app_id,
                name=app_id.replace('_', ' ').title()
            )
        
        app_identity = self.applications[app_id]
        
        # Update application identity from visual data
        identity_features = await self.detection_pipeline.extract_identity_features(screenshot)
        app_identity.update_signature(identity_features)
        
        # Detect current state
        detected_state, confidence = await self.detection_pipeline.detect_state(
            screenshot, 
            app_id,
            self.states.get(app_id, {})
        )
        
        # Handle state detection
        result = {
            'app_id': app_id,
            'app_identity': {
                'name': app_identity.name,
                'version': app_identity.detected_version,
                'customization': app_identity.customization_level
            },
            'detected_state': detected_state,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
        
        # Update current state if confident
        if detected_state and confidence > 0.7:
            previous_state = self.current_states.get(app_id)
            
            # Calculate duration in previous state
            duration = None
            if previous_state and app_id in self.current_state_timestamps:
                duration = datetime.now() - self.current_state_timestamps[app_id]
            
            # Record transition
            if previous_state and previous_state != detected_state:
                transition = StateTransition(
                    from_state=previous_state,
                    to_state=detected_state,
                    timestamp=datetime.now(),
                    duration_in_from_state=duration
                )
                self.state_history.add_transition(transition)
                
                # Record in state intelligence
                from .state_intelligence import StateVisit
                visit = StateVisit(
                    state_id=previous_state,
                    app_id=app_id,
                    timestamp=self.current_state_timestamps.get(app_id, datetime.now()),
                    duration=duration,
                    transition_to=detected_state
                )
                self.state_intelligence.record_visit(visit)
            
            self.current_states[app_id] = detected_state
            self.current_state_timestamps[app_id] = datetime.now()
            
            # Update state statistics
            if app_id in self.states and detected_state in self.states[app_id]:
                self.states[app_id][detected_state].update_statistics(duration)
            
            # Get intelligent recommendations
            recommendations = self.state_intelligence.get_state_recommendations(detected_state)
            result['recommendations'] = recommendations
        
        # Extract content context
        content_features = await self.detection_pipeline.extract_content_features(screenshot)
        if content_features:
            self.content_contexts[app_id] = ContentContext(
                content_id=f"{app_id}_{datetime.now().timestamp()}",
                content_type=content_features.get('type', 'unknown'),
                title=content_features.get('title'),
                summary=content_features.get('summary'),
                available_actions=content_features.get('actions', [])
            )
            result['content'] = {
                'type': self.content_contexts[app_id].content_type,
                'title': self.content_contexts[app_id].title,
                'modified': self.content_contexts[app_id].is_modified
            }
        
        # Add predictions
        if app_id in self.current_states:
            predictions = self.state_history.predict_next_state(self.current_states[app_id])
            result['predictions'] = predictions
        
        # Check for anomalies
        if app_id in self.current_states:
            stuck_states = self.state_history.get_stuck_states()
            if self.current_states[app_id] in stuck_states:
                result['warning'] = 'Currently in a state where users often get stuck'
        
        # Build semantic scene graph
        try:
            # Detect elements in screenshot
            detected_elements = await self.element_detector.detect_elements(screenshot)
            
            # Build scene graph
            scene_graph_result = await self.scene_graph.process_scene(screenshot, detected_elements)
            
            # Add scene graph insights to result
            result['scene_graph'] = {
                'node_count': scene_graph_result['node_count'],
                'relationship_count': scene_graph_result['relationship_count'],
                'key_nodes': scene_graph_result['key_nodes'][:3],  # Top 3 key nodes
                'interaction_patterns': scene_graph_result['interaction_patterns'],
                'anomalies': scene_graph_result.get('anomalies', [])
            }
            
            # Use scene graph to enhance state detection
            if scene_graph_result['key_nodes']:
                key_node_types = [node['node_type'] for node in scene_graph_result['key_nodes'][:3]]
                result['scene_context'] = {
                    'primary_elements': key_node_types,
                    'has_modal': any(node['node_type'] == 'UI_ELEMENT' and 
                                   node.get('properties', {}).get('is_modal') 
                                   for node in scene_graph_result['key_nodes']),
                    'information_density': len([n for n in scene_graph_result['key_nodes'] 
                                              if n['node_type'] == 'INFORMATION'])
                }
        except Exception as e:
            logger.error(f"Scene graph construction failed: {e}")
            result['scene_graph_error'] = str(e)
        
        # Update memory usage
        self._update_memory_usage()
        
        return result
    
    def create_state_definition(self, 
                              app_id: str,
                              state_id: str,
                              category: StateCategory,
                              name: str,
                              visual_signatures: List[Dict[str, Any]]) -> ApplicationState:
        """Create a new state definition"""
        if app_id not in self.states:
            self.states[app_id] = {}
        
        state = ApplicationState(
            state_id=state_id,
            category=category,
            name=name
        )
        
        for sig in visual_signatures:
            state.add_visual_signature(sig)
        
        self.states[app_id][state_id] = state
        
        # Save to disk
        self._save_state_definitions()
        
        return state
    
    def _check_memory_limits(self) -> bool:
        """Check if within memory limits"""
        total_usage = sum(self.memory_usage.values())
        total_limit = sum(MEMORY_LIMITS.values())
        return total_usage < total_limit * 0.9  # 90% threshold
    
    def _cleanup_memory(self):
        """Clean up memory by removing old data"""
        # Remove old transitions
        if len(self.state_history.transitions) > 5000:
            self.state_history.transitions = deque(
                list(self.state_history.transitions)[-5000:], 
                maxlen=10000
            )
        
        # Remove rarely used states
        for app_id, states in self.states.items():
            rarely_used = [
                state_id for state_id, state in states.items()
                if state.detection_count < 5 and 
                (not state.last_detected or 
                 (datetime.now() - state.last_detected) > timedelta(days=30))
            ]
            for state_id in rarely_used:
                del states[state_id]
        
        # Clear old anomalies
        if len(self.state_history.anomalies) > 100:
            self.state_history.anomalies = self.state_history.anomalies[-100:]
    
    def _update_memory_usage(self):
        """Update memory usage tracking"""
        # Estimate memory usage
        import sys
        
        self.memory_usage['state_definitions'] = sum(
            sys.getsizeof(states) for states in self.states.values()
        )
        
        self.memory_usage['transition_history'] = (
            sys.getsizeof(self.state_history.transitions) +
            sys.getsizeof(self.state_history.patterns) +
            sys.getsizeof(self.state_history.anomalies)
        )
        
        self.memory_usage['pattern_storage'] = sum(
            sys.getsizeof(state.visual_signatures) 
            for states in self.states.values()
            for state in states.values()
        )
        
        # Add scene graph memory usage
        if hasattr(self, 'scene_graph') and self.scene_graph:
            self.memory_usage['scene_graph'] = sum(
                self.scene_graph.memory_usage.values()
            )
    
    def _save_state_definitions(self):
        """Save state definitions to disk"""
        try:
            data = {
                'states': {},
                'applications': {}
            }
            
            # Serialize states
            for app_id, states in self.states.items():
                data['states'][app_id] = {}
                for state_id, state in states.items():
                    data['states'][app_id][state_id] = {
                        'category': state.category.name,
                        'name': state.name,
                        'is_modal': state.is_modal,
                        'detection_count': state.detection_count,
                        'confidence_threshold': state.confidence_threshold
                    }
            
            # Serialize applications
            for app_id, identity in self.applications.items():
                data['applications'][app_id] = {
                    'name': identity.name,
                    'version': identity.detected_version,
                    'customization_level': identity.customization_level
                }
            
            with open("vsms_states.json", 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save state definitions: {e}")
    
    def get_insights(self) -> Dict[str, Any]:
        """Get VSMS insights"""
        # Get state intelligence insights
        intelligence_insights = self.state_intelligence.get_productivity_insights()
        
        return {
            'tracked_applications': len(self.applications),
            'total_states': sum(len(states) for states in self.states.values()),
            'current_states': self.current_states,
            'common_patterns': self.state_history.get_common_patterns(),
            'stuck_states': self.state_history.get_stuck_states(),
            'memory_usage': self.memory_usage,
            'anomalies_detected': len(self.state_history.anomalies),
            'intelligence': intelligence_insights,
            'personalization_score': self.state_intelligence.user_preference.personalization_score,
            'preferred_states': list(self.state_intelligence.user_preference.preferred_states),
            'workflow_sequences': self.state_intelligence.user_preference.workflow_sequences[:5]
        }
    
    def get_application_insights(self, app_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific application"""
        if app_id not in self.applications:
            return {'error': f'Application {app_id} not tracked'}
        
        app_identity = self.applications[app_id]
        app_states = self.states.get(app_id, {})
        
        # Get state-specific insights
        state_insights = []
        for state_id, state in app_states.items():
            avg_duration = self.state_history.get_average_duration(state_id)
            state_insights.append({
                'state_id': state_id,
                'name': state.name,
                'category': state.category.name,
                'detection_count': state.detection_count,
                'average_duration': avg_duration,
                'is_modal': state.is_modal,
                'last_detected': state.last_detected
            })
        
        # Sort by detection count
        state_insights.sort(key=lambda x: x['detection_count'], reverse=True)
        
        return {
            'identity': {
                'name': app_identity.name,
                'version': app_identity.detected_version,
                'customization_level': app_identity.customization_level,
                'color_palette': app_identity.color_palette
            },
            'states': state_insights[:10],  # Top 10 states
            'current_state': self.current_states.get(app_id),
            'total_states': len(app_states),
            'transition_patterns': self._get_app_transition_patterns(app_id),
            'anomalies': [a for a in self.state_history.anomalies 
                         if a['transition'].from_state in app_states or 
                         a['transition'].to_state in app_states][:5]
        }
    
    def _get_app_transition_patterns(self, app_id: str) -> List[Dict[str, Any]]:
        """Get transition patterns for an application"""
        app_states = set(self.states.get(app_id, {}).keys())
        patterns = []
        
        for pattern, count in self.state_history.patterns.items():
            states = pattern.split('->')
            if all(state in app_states for state in states):
                patterns.append({
                    'pattern': pattern,
                    'count': count,
                    'probability': count / sum(1 for t in self.state_history.transitions 
                                             if t.from_state == states[0])
                    if any(t.from_state == states[0] for t in self.state_history.transitions) else 0
                })
        
        # Sort by count
        patterns.sort(key=lambda x: x['count'], reverse=True)
        return patterns[:10]


class StateDetectionPipeline:
    """Pipeline for detecting application states"""
    
    async def detect_application(self, screenshot: np.ndarray) -> Optional[str]:
        """Detect which application is shown"""
        # Simplified - in production use window detection
        # and application identification
        return "detected_app"
    
    async def extract_identity_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract application identity features"""
        # Extract visual features for application identification
        return {
            'colors': self._extract_color_palette(screenshot),
            'chrome': self._extract_chrome_pattern(screenshot)
        }
    
    async def detect_state(self, 
                         screenshot: np.ndarray,
                         app_id: str,
                         known_states: Dict[str, ApplicationState]) -> Tuple[Optional[str], float]:
        """Detect current state from screenshot"""
        # Extract state features
        features = self._extract_state_features(screenshot)
        
        # Match against known states
        best_match = None
        best_confidence = 0.0
        
        for state_id, state in known_states.items():
            confidence = self._match_state(features, state)
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = state_id
        
        return best_match, best_confidence
    
    async def extract_content_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract content context features"""
        # Simplified content extraction
        return {
            'type': 'document',
            'title': 'Extracted Title',
            'summary': 'Content summary...',
            'actions': ['save', 'edit', 'close']
        }
    
    def _extract_color_palette(self, screenshot: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors"""
        # Simplified - use k-means clustering in production
        return [(255, 255, 255), (0, 0, 0), (128, 128, 128)]
    
    def _extract_chrome_pattern(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract UI chrome pattern"""
        # Detect window decorations, toolbars, etc.
        return {
            'has_toolbar': True,
            'has_sidebar': False,
            'layout_type': 'standard'
        }
    
    def _extract_state_features(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Extract features for state detection"""
        return {
            'layout_hash': 'simplified_hash',
            'element_count': 42,
            'text_density': 0.3,
            'has_modal': False
        }
    
    def _match_state(self, features: Dict[str, Any], state: ApplicationState) -> float:
        """Match features against state signatures"""
        if not state.visual_signatures:
            return 0.0
        
        # Simplified matching - in production use proper similarity metrics
        return 0.8  # Placeholder confidence


# Global VSMS instance
_vsms_instance = None

def get_vsms() -> VisualStateManager:
    """Get or create the global VSMS instance"""
    global _vsms_instance
    if _vsms_instance is None:
        _vsms_instance = VisualStateManager()
    return _vsms_instance