"""
Temporal Context Engine - Understanding causality and patterns across time
Connects events to build comprehensive temporal understanding
Memory-optimized for 200MB allocation
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Deque, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict, Counter
from enum import Enum, auto
import json
import logging
import mmap
import pickle
import weakref
from pathlib import Path
import heapq
import threading
import time

logger = logging.getLogger(__name__)

# Memory allocation constants
MEMORY_LIMITS = {
    'event_buffer': 100 * 1024 * 1024,    # 100MB
    'pattern_storage': 50 * 1024 * 1024,   # 50MB
    'context_index': 50 * 1024 * 1024     # 50MB
}


class EventType(Enum):
    """Types of events we track"""
    # Visual events
    SCREENSHOT_CAPTURED = auto()
    STATE_CHANGE = auto()
    ELEMENT_INTERACTION = auto()
    WINDOW_FOCUS = auto()
    CONTENT_CHANGE = auto()
    
    # User events
    MOUSE_CLICK = auto()
    KEYBOARD_INPUT = auto()
    SCROLL = auto()
    GESTURE = auto()
    
    # System events
    APPLICATION_LAUNCH = auto()
    APPLICATION_CLOSE = auto()
    ERROR_OCCURRED = auto()
    NOTIFICATION = auto()
    
    # Workflow events
    TASK_START = auto()
    TASK_COMPLETE = auto()
    WORKFLOW_STEP = auto()
    CONTEXT_SWITCH = auto()


class ContextLayer(Enum):
    """Temporal context layers"""
    IMMEDIATE = "immediate"      # Last 30 seconds
    SHORT_TERM = "short_term"    # Last 5 minutes
    LONG_TERM = "long_term"      # Hours/days
    PERSISTENT = "persistent"    # Permanent patterns


@dataclass
class TemporalEvent:
    """Represents an event in time"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    app_id: str
    state_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    related_events: List[str] = field(default_factory=list)
    confidence: float = 1.0
    
    def time_since(self, other_time: datetime) -> timedelta:
        """Calculate time difference"""
        return self.timestamp - other_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'timestamp': self.timestamp.isoformat(),
            'app_id': self.app_id,
            'state_id': self.state_id,
            'data': self.data,
            'related_events': self.related_events,
            'confidence': self.confidence
        }


@dataclass
class TemporalPattern:
    """Represents a pattern across time"""
    pattern_id: str
    pattern_type: str  # sequence, periodic, causality, workflow
    events: List[str]  # Event IDs in order
    confidence: float
    occurrences: int
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> timedelta:
        """Average duration of this pattern"""
        if 'durations' in self.metadata:
            durations = self.metadata['durations']
            if durations:
                avg_seconds = sum(d.total_seconds() for d in durations) / len(durations)
                return timedelta(seconds=avg_seconds)
        return timedelta()
    
    @property
    def frequency(self) -> float:
        """How often this pattern occurs per day"""
        if self.first_seen and self.last_seen and self.occurrences > 1:
            days = (self.last_seen - self.first_seen).days or 1
            return self.occurrences / days
        return 0.0


@dataclass
class CausalityChain:
    """Represents cause-effect relationships"""
    cause_event: str
    effect_events: List[str]
    confidence: float
    time_delta: timedelta  # Typical time between cause and effect
    evidence_count: int
    conditions: Dict[str, Any] = field(default_factory=dict)


class EventStreamProcessor:
    """Processes incoming event streams"""
    
    def __init__(self, buffer_size: int = 10000):
        self.event_buffer: Deque[TemporalEvent] = deque(maxlen=buffer_size)
        self.event_index: Dict[str, TemporalEvent] = {}
        self.app_events: Dict[str, List[str]] = defaultdict(list)
        self.type_events: Dict[EventType, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        
    async def process_event(self, event: TemporalEvent) -> None:
        """Process incoming event"""
        with self._lock:
            # Add to buffer
            self.event_buffer.append(event)
            self.event_index[event.event_id] = event
            
            # Index by app and type
            self.app_events[event.app_id].append(event.event_id)
            self.type_events[event.event_type].append(event.event_id)
            
            # Link related events
            await self._link_related_events(event)
            
            # Trigger pattern detection if buffer is getting full
            if len(self.event_buffer) > self.event_buffer.maxlen * 0.9:
                await self._compact_old_events()
    
    async def _link_related_events(self, event: TemporalEvent) -> None:
        """Link events that are related"""
        # Find events in same app within 5 seconds
        cutoff_time = event.timestamp - timedelta(seconds=5)
        
        related = []
        for event_id in reversed(self.app_events[event.app_id]):
            other_event = self.event_index.get(event_id)
            if other_event and other_event.timestamp < cutoff_time:
                break
            if other_event and other_event.event_id != event.event_id:
                related.append(event_id)
                # Bidirectional linking
                if event.event_id not in other_event.related_events:
                    other_event.related_events.append(event.event_id)
        
        event.related_events.extend(related[:5])  # Limit to 5 related events
    
    async def _compact_old_events(self) -> List[TemporalEvent]:
        """Compact old events to save memory"""
        cutoff = datetime.now() - timedelta(minutes=30)
        old_events = []
        
        with self._lock:
            # Find old events
            for event in list(self.event_buffer):
                if event.timestamp < cutoff:
                    old_events.append(event)
                    self.event_index.pop(event.event_id, None)
        
        return old_events
    
    def get_recent_events(self, seconds: int = 30, 
                         app_id: Optional[str] = None,
                         event_type: Optional[EventType] = None) -> List[TemporalEvent]:
        """Get recent events with optional filtering"""
        cutoff = datetime.now() - timedelta(seconds=seconds)
        events = []
        
        with self._lock:
            for event in reversed(self.event_buffer):
                if event.timestamp < cutoff:
                    break
                
                if app_id and event.app_id != app_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                    
                events.append(event)
        
        return list(reversed(events))
    
    def get_event_sequence(self, start_event_id: str, max_length: int = 10) -> List[TemporalEvent]:
        """Get sequence of events starting from given event"""
        sequence = []
        visited = set()
        
        with self._lock:
            queue = [start_event_id]
            
            while queue and len(sequence) < max_length:
                event_id = queue.pop(0)
                if event_id in visited:
                    continue
                    
                visited.add(event_id)
                event = self.event_index.get(event_id)
                
                if event:
                    sequence.append(event)
                    queue.extend(event.related_events)
        
        return sorted(sequence, key=lambda e: e.timestamp)


class PatternExtractor:
    """Extracts patterns from event streams"""
    
    def __init__(self):
        self.patterns: Dict[str, TemporalPattern] = {}
        self.sequence_trie = {}  # Trie for efficient sequence matching
        self.periodic_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.causality_chains: List[CausalityChain] = []
        
    async def extract_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Extract various types of patterns from events"""
        patterns = []
        
        # Extract different pattern types in parallel
        tasks = [
            self._extract_sequences(events),
            self._extract_periodic_patterns(events),
            self._extract_causality_patterns(events),
            self._extract_workflow_patterns(events)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                patterns.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Pattern extraction error: {result}")
        
        # Update pattern database
        for pattern in patterns:
            self.patterns[pattern.pattern_id] = pattern
        
        return patterns
    
    async def _extract_sequences(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Extract repeated sequences of events"""
        patterns = []
        min_sequence_length = 3
        max_sequence_length = 10
        min_occurrences = 2
        
        # Build sequences
        sequences = defaultdict(int)
        
        for i in range(len(events) - min_sequence_length + 1):
            for length in range(min_sequence_length, 
                              min(max_sequence_length + 1, len(events) - i + 1)):
                # Create sequence key from event types
                seq_key = tuple(e.event_type.name for e in events[i:i+length])
                sequences[seq_key] += 1
        
        # Find patterns
        for seq_key, count in sequences.items():
            if count >= min_occurrences:
                pattern_id = f"seq_{hash(seq_key)}_{len(seq_key)}"
                
                # Find all occurrences
                occurrences = []
                for i in range(len(events) - len(seq_key) + 1):
                    if all(events[i+j].event_type.name == seq_key[j] 
                          for j in range(len(seq_key))):
                        occurrences.append(i)
                
                if occurrences:
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type="sequence",
                        events=[events[i].event_id for i in occurrences[0:1] 
                               for j in range(len(seq_key))],
                        confidence=min(1.0, count / 10),
                        occurrences=count,
                        first_seen=events[occurrences[0]].timestamp,
                        last_seen=events[occurrences[-1]].timestamp,
                        metadata={'sequence': seq_key, 'positions': occurrences}
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _extract_periodic_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Extract patterns that occur periodically"""
        patterns = []
        
        # Group events by type and app
        type_app_events = defaultdict(list)
        for event in events:
            key = (event.event_type, event.app_id)
            type_app_events[key].append(event)
        
        # Look for periodic occurrences
        for (event_type, app_id), event_list in type_app_events.items():
            if len(event_list) < 3:
                continue
            
            # Calculate time intervals
            intervals = []
            for i in range(1, len(event_list)):
                interval = (event_list[i].timestamp - event_list[i-1].timestamp).total_seconds()
                intervals.append(interval)
            
            if not intervals:
                continue
            
            # Check for periodicity using coefficient of variation
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval > 0:
                cv = std_interval / mean_interval
                
                # Low CV indicates periodic pattern
                if cv < 0.3:  # 30% variation threshold
                    pattern_id = f"periodic_{event_type.name}_{app_id}_{int(mean_interval)}"
                    
                    pattern = TemporalPattern(
                        pattern_id=pattern_id,
                        pattern_type="periodic",
                        events=[e.event_id for e in event_list],
                        confidence=1.0 - cv,
                        occurrences=len(event_list),
                        first_seen=event_list[0].timestamp,
                        last_seen=event_list[-1].timestamp,
                        metadata={
                            'period_seconds': mean_interval,
                            'period_variance': std_interval,
                            'event_type': event_type.name,
                            'app_id': app_id
                        }
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _extract_causality_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Extract cause-effect relationships"""
        patterns = []
        max_time_window = timedelta(seconds=10)
        
        # Look for consistent cause-effect pairs
        causality_candidates = defaultdict(lambda: defaultdict(int))
        
        for i, cause_event in enumerate(events):
            # Look at subsequent events within time window
            j = i + 1
            while j < len(events):
                effect_event = events[j]
                time_delta = effect_event.timestamp - cause_event.timestamp
                
                if time_delta > max_time_window:
                    break
                
                # Record potential causality
                cause_key = (cause_event.event_type, cause_event.app_id)
                effect_key = (effect_event.event_type, effect_event.app_id)
                causality_candidates[cause_key][effect_key] += 1
                
                j += 1
        
        # Find strong causality relationships
        for cause_key, effects in causality_candidates.items():
            for effect_key, count in effects.items():
                if count >= 3:  # Minimum evidence threshold
                    pattern_id = f"causal_{cause_key[0].name}_{effect_key[0].name}_{count}"
                    
                    # Find example events
                    cause_events = [e for e in events 
                                  if e.event_type == cause_key[0] and e.app_id == cause_key[1]]
                    effect_events = [e for e in events 
                                   if e.event_type == effect_key[0] and e.app_id == effect_key[1]]
                    
                    if cause_events and effect_events:
                        pattern = TemporalPattern(
                            pattern_id=pattern_id,
                            pattern_type="causality",
                            events=[cause_events[0].event_id, effect_events[0].event_id],
                            confidence=min(1.0, count / len(cause_events)),
                            occurrences=count,
                            first_seen=cause_events[0].timestamp,
                            last_seen=cause_events[-1].timestamp,
                            metadata={
                                'cause': cause_key,
                                'effect': effect_key,
                                'evidence_count': count
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    async def _extract_workflow_patterns(self, events: List[TemporalEvent]) -> List[TemporalPattern]:
        """Extract workflow patterns (task sequences)"""
        patterns = []
        
        # Group events by app and look for task sequences
        app_sequences = defaultdict(list)
        
        for event in events:
            if event.event_type in [EventType.TASK_START, EventType.WORKFLOW_STEP, 
                                  EventType.TASK_COMPLETE]:
                app_sequences[event.app_id].append(event)
        
        # Analyze sequences
        for app_id, sequence in app_sequences.items():
            if len(sequence) < 3:
                continue
            
            # Look for complete workflows
            workflow_segments = []
            current_segment = []
            
            for event in sequence:
                current_segment.append(event)
                
                if event.event_type == EventType.TASK_COMPLETE:
                    if len(current_segment) >= 3:
                        workflow_segments.append(current_segment)
                    current_segment = []
            
            # Create patterns from workflow segments
            for i, segment in enumerate(workflow_segments):
                pattern_id = f"workflow_{app_id}_{i}_{len(segment)}"
                
                pattern = TemporalPattern(
                    pattern_id=pattern_id,
                    pattern_type="workflow",
                    events=[e.event_id for e in segment],
                    confidence=0.9,  # High confidence for complete workflows
                    occurrences=1,
                    first_seen=segment[0].timestamp,
                    last_seen=segment[-1].timestamp,
                    metadata={
                        'app_id': app_id,
                        'steps': len(segment),
                        'duration': (segment[-1].timestamp - segment[0].timestamp).total_seconds()
                    }
                )
                patterns.append(pattern)
        
        return patterns


class ContextBuilder:
    """Builds temporal context from events and patterns"""
    
    def __init__(self):
        self.immediate_context: Deque[TemporalEvent] = deque(maxlen=100)
        self.short_term_context: Dict[str, Any] = {}
        self.long_term_context: Dict[str, Any] = {}
        self.persistent_context: Dict[str, Any] = {}
        
        # Context windows
        self.immediate_window = timedelta(seconds=30)
        self.short_term_window = timedelta(minutes=5)
        self.long_term_window = timedelta(hours=24)
        
        # Pattern scores
        self.pattern_significance: Dict[str, float] = {}
        
        # Load persistent context
        self._load_persistent_context()
    
    async def update_context(self, 
                           events: List[TemporalEvent],
                           patterns: List[TemporalPattern]) -> Dict[str, Any]:
        """Update all context layers"""
        now = datetime.now()
        
        # Update immediate context
        self._update_immediate_context(events, now)
        
        # Update short-term context
        self._update_short_term_context(events, patterns, now)
        
        # Update long-term context
        await self._update_long_term_context(patterns, now)
        
        # Update persistent context
        self._update_persistent_context(patterns)
        
        # Build comprehensive context
        context = {
            'immediate': self._get_immediate_summary(),
            'short_term': self.short_term_context,
            'long_term': self.long_term_context,
            'persistent': self.persistent_context,
            'active_patterns': self._get_active_patterns(patterns, now),
            'predictions': self._generate_predictions(patterns, now)
        }
        
        return context
    
    def _update_immediate_context(self, events: List[TemporalEvent], now: datetime) -> None:
        """Update immediate context (last 30 seconds)"""
        cutoff = now - self.immediate_window
        
        # Add recent events
        for event in events:
            if event.timestamp > cutoff:
                self.immediate_context.append(event)
        
        # Remove old events
        while self.immediate_context and self.immediate_context[0].timestamp <= cutoff:
            self.immediate_context.popleft()
    
    def _update_short_term_context(self, 
                                  events: List[TemporalEvent],
                                  patterns: List[TemporalPattern],
                                  now: datetime) -> None:
        """Update short-term context (last 5 minutes)"""
        cutoff = now - self.short_term_window
        
        # Task sequences in progress
        task_sequences = []
        for event in events:
            if event.timestamp > cutoff and event.event_type in [
                EventType.TASK_START, EventType.WORKFLOW_STEP
            ]:
                task_sequences.append({
                    'event_id': event.event_id,
                    'app_id': event.app_id,
                    'state_id': event.state_id,
                    'timestamp': event.timestamp.isoformat()
                })
        
        # Recent errors
        recent_errors = [
            event for event in events 
            if event.timestamp > cutoff and event.event_type == EventType.ERROR_OCCURRED
        ]
        
        # Navigation patterns
        nav_patterns = [
            p for p in patterns 
            if p.pattern_type == "sequence" and p.last_seen > cutoff
        ]
        
        self.short_term_context = {
            'task_sequences': task_sequences[-10:],  # Last 10 tasks
            'error_count': len(recent_errors),
            'navigation_patterns': [p.pattern_id for p in nav_patterns],
            'focus_changes': self._count_focus_changes(events, cutoff),
            'active_apps': self._get_active_apps(events, cutoff)
        }
    
    async def _update_long_term_context(self, 
                                      patterns: List[TemporalPattern],
                                      now: datetime) -> None:
        """Update long-term context (hours/days)"""
        cutoff = now - self.long_term_window
        
        # Workflow patterns
        workflow_patterns = [
            p for p in patterns 
            if p.pattern_type == "workflow" and p.last_seen > cutoff
        ]
        
        # Recurring issues
        recurring_issues = [
            p for p in patterns
            if p.pattern_type == "causality" and 
            "ERROR" in str(p.metadata.get('effect', ''))
        ]
        
        # Productivity patterns
        productivity_metrics = self._calculate_productivity_metrics(patterns, cutoff)
        
        self.long_term_context = {
            'workflow_count': len(workflow_patterns),
            'common_workflows': [p.pattern_id for p in sorted(
                workflow_patterns, key=lambda x: x.occurrences, reverse=True
            )[:5]],
            'recurring_issues': [
                {
                    'pattern_id': p.pattern_id,
                    'occurrences': p.occurrences,
                    'last_seen': p.last_seen.isoformat()
                }
                for p in recurring_issues
            ],
            'productivity_metrics': productivity_metrics
        }
    
    def _update_persistent_context(self, patterns: List[TemporalPattern]) -> None:
        """Update persistent context (permanent patterns)"""
        # Update pattern significance scores
        for pattern in patterns:
            current_score = self.pattern_significance.get(pattern.pattern_id, 0.0)
            
            # Increase score based on occurrences and recency
            age_factor = 1.0 / (1.0 + (datetime.now() - pattern.last_seen).days)
            new_score = current_score * 0.9 + (pattern.occurrences * age_factor) * 0.1
            
            self.pattern_significance[pattern.pattern_id] = new_score
        
        # Keep top patterns
        top_patterns = sorted(
            self.pattern_significance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:100]
        
        self.persistent_context['learned_patterns'] = dict(top_patterns)
        
        # Save periodically
        if len(patterns) % 10 == 0:
            self._save_persistent_context()
    
    def _get_immediate_summary(self) -> Dict[str, Any]:
        """Summarize immediate context"""
        if not self.immediate_context:
            return {'empty': True}
        
        event_types = Counter(e.event_type.name for e in self.immediate_context)
        apps = Counter(e.app_id for e in self.immediate_context)
        
        return {
            'event_count': len(self.immediate_context),
            'dominant_event_types': event_types.most_common(3),
            'active_apps': apps.most_common(3),
            'latest_event': self.immediate_context[-1].to_dict() if self.immediate_context else None,
            'time_span': (
                self.immediate_context[-1].timestamp - self.immediate_context[0].timestamp
            ).total_seconds() if len(self.immediate_context) > 1 else 0
        }
    
    def _get_active_patterns(self, patterns: List[TemporalPattern], 
                           now: datetime) -> List[Dict[str, Any]]:
        """Get currently active patterns"""
        active = []
        
        for pattern in patterns:
            # Check if pattern is likely active
            if pattern.pattern_type == "periodic":
                period = pattern.metadata.get('period_seconds', 0)
                if period > 0:
                    expected_next = pattern.last_seen + timedelta(seconds=period)
                    if now <= expected_next + timedelta(seconds=period * 0.2):
                        active.append({
                            'pattern_id': pattern.pattern_id,
                            'type': pattern.pattern_type,
                            'expected_next': expected_next.isoformat(),
                            'confidence': pattern.confidence
                        })
            
            elif pattern.last_seen > now - timedelta(minutes=1):
                active.append({
                    'pattern_id': pattern.pattern_id,
                    'type': pattern.pattern_type,
                    'last_seen': pattern.last_seen.isoformat(),
                    'confidence': pattern.confidence
                })
        
        return active
    
    def _generate_predictions(self, patterns: List[TemporalPattern], 
                            now: datetime) -> Dict[str, Any]:
        """Generate predictions based on patterns"""
        predictions = {
            'next_likely_events': [],
            'pattern_continuations': [],
            'time_based_suggestions': []
        }
        
        # Predict based on periodic patterns
        for pattern in patterns:
            if pattern.pattern_type == "periodic":
                period = pattern.metadata.get('period_seconds', 0)
                if period > 0:
                    next_occurrence = pattern.last_seen + timedelta(seconds=period)
                    if now < next_occurrence < now + timedelta(minutes=5):
                        predictions['next_likely_events'].append({
                            'event_type': pattern.metadata.get('event_type', 'unknown'),
                            'app_id': pattern.metadata.get('app_id', 'unknown'),
                            'expected_time': next_occurrence.isoformat(),
                            'confidence': pattern.confidence
                        })
        
        # Predict workflow continuations
        if self.immediate_context:
            recent_types = [e.event_type for e in list(self.immediate_context)[-5:]]
            
            for pattern in patterns:
                if pattern.pattern_type == "sequence":
                    seq = pattern.metadata.get('sequence', [])
                    if len(seq) > len(recent_types):
                        # Check if recent events match pattern start
                        if all(recent_types[i] == seq[i] for i in range(len(recent_types))):
                            predictions['pattern_continuations'].append({
                                'pattern_id': pattern.pattern_id,
                                'next_events': seq[len(recent_types):],
                                'confidence': pattern.confidence
                            })
        
        return predictions
    
    def _count_focus_changes(self, events: List[TemporalEvent], 
                           cutoff: datetime) -> int:
        """Count focus changes after cutoff"""
        return sum(
            1 for e in events 
            if e.timestamp > cutoff and e.event_type == EventType.WINDOW_FOCUS
        )
    
    def _get_active_apps(self, events: List[TemporalEvent], 
                        cutoff: datetime) -> List[str]:
        """Get active apps after cutoff"""
        apps = set()
        for event in events:
            if event.timestamp > cutoff:
                apps.add(event.app_id)
        return list(apps)
    
    def _calculate_productivity_metrics(self, patterns: List[TemporalPattern], 
                                     cutoff: datetime) -> Dict[str, Any]:
        """Calculate productivity metrics from patterns"""
        workflow_patterns = [
            p for p in patterns 
            if p.pattern_type == "workflow" and p.last_seen > cutoff
        ]
        
        if not workflow_patterns:
            return {'no_data': True}
        
        # Calculate metrics
        total_workflows = len(workflow_patterns)
        avg_duration = np.mean([
            p.metadata.get('duration', 0) for p in workflow_patterns
        ]) if workflow_patterns else 0
        
        completion_rate = sum(
            1 for p in workflow_patterns 
            if p.metadata.get('completed', False)
        ) / total_workflows if total_workflows > 0 else 0
        
        return {
            'workflow_count': total_workflows,
            'average_duration_seconds': avg_duration,
            'completion_rate': completion_rate,
            'peak_hours': self._find_peak_hours(workflow_patterns)
        }
    
    def _find_peak_hours(self, patterns: List[TemporalPattern]) -> List[int]:
        """Find peak productivity hours"""
        hour_counts = Counter()
        
        for pattern in patterns:
            hour = pattern.first_seen.hour
            hour_counts[hour] += 1
        
        return [hour for hour, _ in hour_counts.most_common(3)]
    
    def _save_persistent_context(self) -> None:
        """Save persistent context to disk"""
        try:
            save_path = Path("temporal_persistent_context.json")
            with open(save_path, 'w') as f:
                json.dump(self.persistent_context, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save persistent context: {e}")
    
    def _load_persistent_context(self) -> None:
        """Load persistent context from disk"""
        try:
            load_path = Path("temporal_persistent_context.json")
            if load_path.exists():
                with open(load_path, 'r') as f:
                    self.persistent_context = json.load(f)
                    
                # Reconstruct pattern significance
                if 'learned_patterns' in self.persistent_context:
                    self.pattern_significance = dict(self.persistent_context['learned_patterns'])
        except Exception as e:
            logger.error(f"Failed to load persistent context: {e}")


class TemporalContextEngine:
    """Main Temporal Context Engine coordinating all components"""
    
    def __init__(self):
        self.event_processor = EventStreamProcessor()
        self.pattern_extractor = PatternExtractor()
        self.context_builder = ContextBuilder()
        
        # Memory management
        self.memory_usage = {
            'event_buffer': 0,
            'pattern_storage': 0,
            'context_index': 0
        }
        
        # Background tasks
        self._running = False
        self._pattern_extraction_task = None
        self._cleanup_task = None
        
    async def start(self) -> None:
        """Start the temporal context engine"""
        self._running = True
        
        # Start background tasks
        self._pattern_extraction_task = asyncio.create_task(self._pattern_extraction_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Temporal Context Engine started")
    
    async def stop(self) -> None:
        """Stop the temporal context engine"""
        self._running = False
        
        # Cancel background tasks
        if self._pattern_extraction_task:
            self._pattern_extraction_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Save state
        self.context_builder._save_persistent_context()
        
        logger.info("Temporal Context Engine stopped")
    
    async def process_visual_event(self, 
                                 event_type: EventType,
                                 app_id: str,
                                 state_id: Optional[str] = None,
                                 data: Optional[Dict[str, Any]] = None) -> str:
        """Process a visual event"""
        import uuid
        
        event = TemporalEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            app_id=app_id,
            state_id=state_id,
            data=data or {}
        )
        
        await self.event_processor.process_event(event)
        
        return event.event_id
    
    async def get_temporal_context(self, 
                                 app_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current temporal context"""
        # Get recent events
        recent_events = self.event_processor.get_recent_events(seconds=300, app_id=app_id)
        
        # Extract patterns
        patterns = await self.pattern_extractor.extract_patterns(recent_events)
        
        # Build context
        context = await self.context_builder.update_context(recent_events, patterns)
        
        # Add memory usage
        context['memory_usage'] = self._calculate_memory_usage()
        
        return context
    
    async def get_event_history(self, 
                              app_id: Optional[str] = None,
                              event_type: Optional[EventType] = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history with optional filtering"""
        events = self.event_processor.get_recent_events(
            seconds=3600,  # Last hour
            app_id=app_id,
            event_type=event_type
        )
        
        return [event.to_dict() for event in events[:limit]]
    
    async def get_active_patterns(self) -> List[Dict[str, Any]]:
        """Get currently active patterns"""
        patterns = list(self.pattern_extractor.patterns.values())
        now = datetime.now()
        
        active = self.context_builder._get_active_patterns(patterns, now)
        
        # Add pattern details
        for item in active:
            pattern = self.pattern_extractor.patterns.get(item['pattern_id'])
            if pattern:
                item['occurrences'] = pattern.occurrences
                item['frequency'] = pattern.frequency
        
        return active
    
    async def predict_next_events(self, 
                                lookahead_seconds: int = 60) -> List[Dict[str, Any]]:
        """Predict likely next events"""
        patterns = list(self.pattern_extractor.patterns.values())
        now = datetime.now()
        
        predictions = self.context_builder._generate_predictions(patterns, now)
        
        # Filter by lookahead window
        lookahead_cutoff = now + timedelta(seconds=lookahead_seconds)
        
        filtered_predictions = []
        for pred in predictions.get('next_likely_events', []):
            expected_time = datetime.fromisoformat(pred['expected_time'])
            if expected_time <= lookahead_cutoff:
                filtered_predictions.append(pred)
        
        return filtered_predictions
    
    async def _pattern_extraction_loop(self) -> None:
        """Background task for pattern extraction"""
        while self._running:
            try:
                # Extract patterns every 30 seconds
                await asyncio.sleep(30)
                
                # Get recent events
                events = self.event_processor.get_recent_events(seconds=600)  # Last 10 minutes
                
                # Extract patterns
                patterns = await self.pattern_extractor.extract_patterns(events)
                
                logger.info(f"Extracted {len(patterns)} patterns from {len(events)} events")
                
            except Exception as e:
                logger.error(f"Pattern extraction error: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background task for memory cleanup"""
        while self._running:
            try:
                # Cleanup every minute
                await asyncio.sleep(60)
                
                # Compact old events
                old_events = await self.event_processor._compact_old_events()
                
                # Update memory usage
                self._calculate_memory_usage()
                
                # Check memory limits
                total_usage = sum(self.memory_usage.values())
                total_limit = sum(MEMORY_LIMITS.values())
                
                if total_usage > total_limit * 0.9:
                    logger.warning(f"Memory usage high: {total_usage / 1024 / 1024:.1f}MB")
                    # Trigger more aggressive cleanup
                    await self._aggressive_cleanup()
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _aggressive_cleanup(self) -> None:
        """Aggressive cleanup when memory is high"""
        # Remove low-significance patterns
        if hasattr(self.pattern_extractor, 'patterns'):
            patterns = list(self.pattern_extractor.patterns.values())
            patterns.sort(key=lambda p: p.occurrences)
            
            # Remove bottom 25%
            remove_count = len(patterns) // 4
            for pattern in patterns[:remove_count]:
                self.pattern_extractor.patterns.pop(pattern.pattern_id, None)
        
        # Force garbage collection
        import gc
        gc.collect()
    
    def _calculate_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage of components"""
        import sys
        
        # Event buffer
        self.memory_usage['event_buffer'] = sum(
            sys.getsizeof(event) for event in self.event_processor.event_buffer
        )
        
        # Pattern storage
        self.memory_usage['pattern_storage'] = sum(
            sys.getsizeof(pattern) for pattern in self.pattern_extractor.patterns.values()
        )
        
        # Context index
        self.memory_usage['context_index'] = (
            sys.getsizeof(self.context_builder.immediate_context) +
            sys.getsizeof(self.context_builder.short_term_context) +
            sys.getsizeof(self.context_builder.long_term_context) +
            sys.getsizeof(self.context_builder.persistent_context)
        )
        
        return self.memory_usage


# Global instance
_temporal_engine_instance = None

def get_temporal_engine() -> TemporalContextEngine:
    """Get or create the global temporal context engine"""
    global _temporal_engine_instance
    if _temporal_engine_instance is None:
        _temporal_engine_instance = TemporalContextEngine()
    return _temporal_engine_instance