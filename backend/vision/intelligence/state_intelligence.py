"""
State Intelligence - Learning personal state patterns and preferences
Identifies frequently visited states, stuck states, and builds preference models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from collections import defaultdict, Counter
from enum import Enum
import json
import logging
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


class TimeOfDay(Enum):
    """Time of day categories"""
    EARLY_MORNING = "early_morning"  # 5-8 AM
    MORNING = "morning"              # 8-12 PM
    AFTERNOON = "afternoon"          # 12-5 PM
    EVENING = "evening"              # 5-9 PM
    NIGHT = "night"                  # 9PM-12AM
    LATE_NIGHT = "late_night"        # 12AM-5AM


class DayType(Enum):
    """Day type categories"""
    WEEKDAY = "weekday"
    WEEKEND = "weekend"
    HOLIDAY = "holiday"


@dataclass
class StateVisit:
    """Record of a state visit"""
    state_id: str
    app_id: str
    timestamp: datetime
    duration: Optional[timedelta] = None
    transition_to: Optional[str] = None
    user_triggered: bool = True
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_of_day(self) -> TimeOfDay:
        """Get time of day category"""
        hour = self.timestamp.hour
        if 5 <= hour < 8:
            return TimeOfDay.EARLY_MORNING
        elif 8 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        elif 21 <= hour < 24:
            return TimeOfDay.NIGHT
        else:
            return TimeOfDay.LATE_NIGHT
    
    @property
    def day_type(self) -> DayType:
        """Get day type"""
        if self.timestamp.weekday() < 5:
            return DayType.WEEKDAY
        return DayType.WEEKEND


@dataclass 
class StatePattern:
    """Identified state pattern"""
    pattern_type: str  # frequent, stuck, error_prone, etc.
    states: List[str]
    confidence: float
    occurrences: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """User preference for states"""
    preferred_states: Set[str]
    avoided_states: Set[str]
    time_preferences: Dict[TimeOfDay, Set[str]]  # Preferred states by time
    workflow_sequences: List[List[str]]  # Common workflows
    personalization_score: float = 0.0  # How well we understand the user


class StateIntelligence:
    """Learns and analyzes personal state patterns"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.state_visits: List[StateVisit] = []
        self.state_frequencies: Dict[str, int] = defaultdict(int)
        self.state_durations: Dict[str, List[timedelta]] = defaultdict(list)
        self.transition_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.patterns: List[StatePattern] = []
        self.user_preference = UserPreference(
            preferred_states=set(),
            avoided_states=set(),
            time_preferences=defaultdict(set),
            workflow_sequences=[]
        )
        
        # Pattern detection parameters
        self.min_pattern_occurrences = 3
        self.stuck_threshold = timedelta(minutes=5)
        self.frequent_threshold = 10  # visits
        
        # Load historical data
        self._load_intelligence_data()
    
    def record_visit(self, state_visit: StateVisit):
        """Record a state visit"""
        self.state_visits.append(state_visit)
        self.state_frequencies[state_visit.state_id] += 1
        
        # Update duration if available
        if state_visit.duration:
            self.state_durations[state_visit.state_id].append(state_visit.duration)
        
        # Update transition matrix
        if state_visit.transition_to:
            self.transition_matrix[state_visit.state_id][state_visit.transition_to] += 1
        
        # Trigger pattern analysis periodically
        if len(self.state_visits) % 50 == 0:
            self._analyze_patterns()
    
    def _analyze_patterns(self):
        """Analyze state visits for patterns"""
        self.patterns.clear()
        
        # Identify frequent states
        self._identify_frequent_states()
        
        # Identify stuck states
        self._identify_stuck_states()
        
        # Identify error-prone states
        self._identify_error_prone_states()
        
        # Identify time-based patterns
        self._identify_time_patterns()
        
        # Identify workflow sequences
        self._identify_workflows()
        
        # Update user preferences
        self._update_preferences()
    
    def _identify_frequent_states(self):
        """Identify frequently visited states"""
        for state_id, count in self.state_frequencies.items():
            if count >= self.frequent_threshold:
                pattern = StatePattern(
                    pattern_type="frequent",
                    states=[state_id],
                    confidence=min(1.0, count / 50),  # Cap at 50 visits
                    occurrences=count,
                    metadata={
                        'avg_duration': self._get_average_duration(state_id),
                        'last_visit': self._get_last_visit(state_id)
                    }
                )
                self.patterns.append(pattern)
                self.user_preference.preferred_states.add(state_id)
    
    def _identify_stuck_states(self):
        """Identify states where users get stuck"""
        for state_id, durations in self.state_durations.items():
            if not durations:
                continue
            
            avg_duration = sum(d.total_seconds() for d in durations) / len(durations)
            stuck_visits = [d for d in durations if d > self.stuck_threshold]
            
            if len(stuck_visits) >= self.min_pattern_occurrences:
                pattern = StatePattern(
                    pattern_type="stuck",
                    states=[state_id],
                    confidence=len(stuck_visits) / len(durations),
                    occurrences=len(stuck_visits),
                    metadata={
                        'avg_stuck_duration': timedelta(seconds=statistics.mean([d.total_seconds() for d in stuck_visits])),
                        'normal_duration': timedelta(seconds=avg_duration),
                        'stuck_ratio': len(stuck_visits) / len(durations)
                    }
                )
                self.patterns.append(pattern)
    
    def _identify_error_prone_states(self):
        """Identify states that often lead to errors"""
        error_transitions = defaultdict(int)
        total_transitions = defaultdict(int)
        
        for from_state, transitions in self.transition_matrix.items():
            for to_state, count in transitions.items():
                total_transitions[from_state] += count
                if 'error' in to_state.lower() or 'fail' in to_state.lower():
                    error_transitions[from_state] += count
        
        for state_id, error_count in error_transitions.items():
            error_rate = error_count / total_transitions[state_id]
            if error_rate > 0.2 and error_count >= self.min_pattern_occurrences:
                pattern = StatePattern(
                    pattern_type="error_prone",
                    states=[state_id],
                    confidence=error_rate,
                    occurrences=error_count,
                    metadata={
                        'error_rate': error_rate,
                        'total_transitions': total_transitions[state_id],
                        'common_errors': self._get_common_error_transitions(state_id)
                    }
                )
                self.patterns.append(pattern)
                self.user_preference.avoided_states.add(state_id)
    
    def _identify_time_patterns(self):
        """Identify time-based usage patterns"""
        time_state_map: Dict[TimeOfDay, Counter] = defaultdict(Counter)
        
        for visit in self.state_visits:
            time_state_map[visit.time_of_day][visit.state_id] += 1
        
        # Find preferred states for each time period
        for time_period, state_counts in time_state_map.items():
            if not state_counts:
                continue
            
            # Get top states for this time period
            top_states = state_counts.most_common(5)
            total_visits = sum(state_counts.values())
            
            for state_id, count in top_states:
                if count >= self.min_pattern_occurrences:
                    self.user_preference.time_preferences[time_period].add(state_id)
                    
                    pattern = StatePattern(
                        pattern_type="time_preference",
                        states=[state_id],
                        confidence=count / total_visits,
                        occurrences=count,
                        metadata={
                            'time_period': time_period.value,
                            'preference_strength': count / total_visits
                        }
                    )
                    self.patterns.append(pattern)
    
    def _identify_workflows(self):
        """Identify common workflow sequences"""
        # Extract sequences of 3+ states
        sequences = []
        
        for i in range(len(self.state_visits) - 2):
            if i + 2 < len(self.state_visits):
                seq = [
                    self.state_visits[i].state_id,
                    self.state_visits[i + 1].state_id,
                    self.state_visits[i + 2].state_id
                ]
                
                # Check if transitions were quick (likely same workflow)
                time_diff1 = self.state_visits[i + 1].timestamp - self.state_visits[i].timestamp
                time_diff2 = self.state_visits[i + 2].timestamp - self.state_visits[i + 1].timestamp
                
                if time_diff1 < timedelta(minutes=5) and time_diff2 < timedelta(minutes=5):
                    sequences.append(tuple(seq))
        
        # Count sequence occurrences
        seq_counter = Counter(sequences)
        
        for seq, count in seq_counter.most_common(10):
            if count >= self.min_pattern_occurrences:
                pattern = StatePattern(
                    pattern_type="workflow",
                    states=list(seq),
                    confidence=min(1.0, count / 10),
                    occurrences=count,
                    metadata={
                        'sequence_length': len(seq),
                        'avg_completion_time': self._calculate_workflow_duration(seq)
                    }
                )
                self.patterns.append(pattern)
                self.user_preference.workflow_sequences.append(list(seq))
    
    def _update_preferences(self):
        """Update user preference model"""
        # Calculate personalization score based on data quality
        data_points = len(self.state_visits)
        unique_states = len(self.state_frequencies)
        pattern_count = len(self.patterns)
        
        # Score based on data richness
        self.user_preference.personalization_score = min(1.0, (
            (data_points / 1000) * 0.3 +  # Visit count
            (unique_states / 50) * 0.3 +   # State diversity
            (pattern_count / 20) * 0.4     # Pattern detection
        ))
    
    def predict_next_state(self, current_state: str, context: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Predict likely next states with personalized scoring"""
        predictions = []
        
        # Base predictions from transition matrix
        if current_state in self.transition_matrix:
            transitions = self.transition_matrix[current_state]
            total = sum(transitions.values())
            
            for next_state, count in transitions.items():
                base_prob = count / total
                
                # Adjust based on preferences
                if next_state in self.user_preference.preferred_states:
                    base_prob *= 1.2
                elif next_state in self.user_preference.avoided_states:
                    base_prob *= 0.8
                
                # Adjust based on time preferences
                if context and 'time_of_day' in context:
                    tod = TimeOfDay(context['time_of_day'])
                    if next_state in self.user_preference.time_preferences[tod]:
                        base_prob *= 1.1
                
                predictions.append((next_state, min(1.0, base_prob)))
        
        # Sort by probability
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions[:5]
    
    def get_state_recommendations(self, current_state: str, 
                                 current_time: datetime = None) -> Dict[str, Any]:
        """Get intelligent state recommendations"""
        if current_time is None:
            current_time = datetime.now()
        
        recommendations = {
            'next_states': [],
            'warnings': [],
            'suggestions': [],
            'workflow_hint': None
        }
        
        # Get next state predictions
        context = {'time_of_day': self._get_time_of_day(current_time).value}
        predictions = self.predict_next_state(current_state, context)
        recommendations['next_states'] = predictions
        
        # Check if in stuck state
        stuck_patterns = [p for p in self.patterns 
                         if p.pattern_type == "stuck" and current_state in p.states]
        if stuck_patterns:
            avg_stuck_duration = stuck_patterns[0].metadata.get('avg_stuck_duration', timedelta(minutes=5))
            recommendations['warnings'].append({
                'type': 'stuck_state',
                'message': f"Users typically spend {avg_stuck_duration} in this state",
                'suggestion': "Consider if you need help or want to switch tasks"
            })
        
        # Check if in error-prone state
        error_patterns = [p for p in self.patterns 
                         if p.pattern_type == "error_prone" and current_state in p.states]
        if error_patterns:
            error_rate = error_patterns[0].metadata.get('error_rate', 0)
            recommendations['warnings'].append({
                'type': 'error_prone',
                'message': f"This state has a {error_rate:.0%} error rate",
                'suggestion': "Double-check your inputs before proceeding"
            })
        
        # Check for workflow matches
        for workflow in self.user_preference.workflow_sequences:
            if len(workflow) > 1 and current_state == workflow[0]:
                recommendations['workflow_hint'] = {
                    'detected_workflow': workflow,
                    'next_in_workflow': workflow[1],
                    'completion_steps': len(workflow) - 1
                }
                break
        
        # Time-based suggestions
        tod = self._get_time_of_day(current_time)
        preferred_for_time = self.user_preference.time_preferences[tod]
        if preferred_for_time and current_state not in preferred_for_time:
            recommendations['suggestions'].append({
                'type': 'time_preference',
                'message': f"You usually prefer these states during {tod.value}: {', '.join(list(preferred_for_time)[:3])}"
            })
        
        return recommendations
    
    def get_productivity_insights(self) -> Dict[str, Any]:
        """Get productivity insights from state patterns"""
        insights = {
            'total_states_visited': len(self.state_frequencies),
            'total_visits': len(self.state_visits),
            'personalization_score': self.user_preference.personalization_score,
            'patterns_found': len(self.patterns),
            'top_states': [],
            'stuck_states': [],
            'efficient_workflows': [],
            'time_analysis': {},
            'improvement_suggestions': []
        }
        
        # Top states by frequency
        top_states = sorted(self.state_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
        insights['top_states'] = [
            {
                'state': state,
                'visits': count,
                'avg_duration': self._get_average_duration(state)
            }
            for state, count in top_states
        ]
        
        # Stuck states
        stuck_patterns = [p for p in self.patterns if p.pattern_type == "stuck"]
        insights['stuck_states'] = [
            {
                'state': p.states[0],
                'avg_stuck_duration': p.metadata.get('avg_stuck_duration'),
                'frequency': p.occurrences
            }
            for p in stuck_patterns
        ]
        
        # Efficient workflows
        workflow_patterns = [p for p in self.patterns if p.pattern_type == "workflow"]
        insights['efficient_workflows'] = [
            {
                'workflow': p.states,
                'frequency': p.occurrences,
                'avg_duration': p.metadata.get('avg_completion_time')
            }
            for p in sorted(workflow_patterns, key=lambda x: x.occurrences, reverse=True)[:5]
        ]
        
        # Time analysis
        for time_period in TimeOfDay:
            time_visits = [v for v in self.state_visits if v.time_of_day == time_period]
            if time_visits:
                insights['time_analysis'][time_period.value] = {
                    'visit_count': len(time_visits),
                    'preferred_states': list(self.user_preference.time_preferences[time_period])[:3]
                }
        
        # Generate improvement suggestions
        if stuck_patterns:
            insights['improvement_suggestions'].append({
                'type': 'reduce_stuck_time',
                'message': f"You have {len(stuck_patterns)} states where you frequently get stuck",
                'action': "Consider creating shortcuts or automations for these states"
            })
        
        if insights['personalization_score'] < 0.5:
            insights['improvement_suggestions'].append({
                'type': 'need_more_data',
                'message': "Limited usage data available",
                'action': "Continue using the system to get more personalized insights"
            })
        
        return insights
    
    def _get_average_duration(self, state_id: str) -> Optional[timedelta]:
        """Get average duration for a state"""
        durations = self.state_durations.get(state_id, [])
        if not durations:
            return None
        
        avg_seconds = statistics.mean([d.total_seconds() for d in durations])
        return timedelta(seconds=avg_seconds)
    
    def _get_last_visit(self, state_id: str) -> Optional[datetime]:
        """Get last visit time for a state"""
        visits = [v for v in self.state_visits if v.state_id == state_id]
        if visits:
            return visits[-1].timestamp
        return None
    
    def _get_common_error_transitions(self, state_id: str) -> List[str]:
        """Get common error states transitioned to"""
        if state_id not in self.transition_matrix:
            return []
        
        error_transitions = [
            (to_state, count) 
            for to_state, count in self.transition_matrix[state_id].items()
            if 'error' in to_state.lower() or 'fail' in to_state.lower()
        ]
        
        error_transitions.sort(key=lambda x: x[1], reverse=True)
        return [t[0] for t in error_transitions[:3]]
    
    def _calculate_workflow_duration(self, workflow_seq: Tuple[str, ...]) -> Optional[timedelta]:
        """Calculate average duration for a workflow sequence"""
        durations = []
        
        # Find matching sequences in history
        for i in range(len(self.state_visits) - len(workflow_seq) + 1):
            match = True
            for j, state in enumerate(workflow_seq):
                if i + j >= len(self.state_visits) or self.state_visits[i + j].state_id != state:
                    match = False
                    break
            
            if match:
                # Calculate total duration
                start_time = self.state_visits[i].timestamp
                end_time = self.state_visits[i + len(workflow_seq) - 1].timestamp
                durations.append(end_time - start_time)
        
        if durations:
            avg_seconds = statistics.mean([d.total_seconds() for d in durations])
            return timedelta(seconds=avg_seconds)
        
        return None
    
    def _get_time_of_day(self, dt: datetime) -> TimeOfDay:
        """Get time of day category for a datetime"""
        hour = dt.hour
        if 5 <= hour < 8:
            return TimeOfDay.EARLY_MORNING
        elif 8 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        elif 21 <= hour < 24:
            return TimeOfDay.NIGHT
        else:
            return TimeOfDay.LATE_NIGHT
    
    def _save_intelligence_data(self):
        """Save intelligence data to disk"""
        try:
            data = {
                'user_id': self.user_id,
                'visits': [
                    {
                        'state_id': v.state_id,
                        'app_id': v.app_id,
                        'timestamp': v.timestamp.isoformat(),
                        'duration': v.duration.total_seconds() if v.duration else None,
                        'transition_to': v.transition_to,
                        'user_triggered': v.user_triggered,
                        'context': v.context
                    }
                    for v in self.state_visits[-1000:]  # Keep last 1000 visits
                ],
                'patterns': [
                    {
                        'pattern_type': p.pattern_type,
                        'states': p.states,
                        'confidence': p.confidence,
                        'occurrences': p.occurrences,
                        'metadata': p.metadata
                    }
                    for p in self.patterns
                ],
                'preferences': {
                    'preferred_states': list(self.user_preference.preferred_states),
                    'avoided_states': list(self.user_preference.avoided_states),
                    'workflow_sequences': self.user_preference.workflow_sequences,
                    'personalization_score': self.user_preference.personalization_score
                }
            }
            
            save_path = Path(f"state_intelligence_{self.user_id}.json")
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved state intelligence data for user {self.user_id}")
        
        except Exception as e:
            logger.error(f"Failed to save intelligence data: {e}")
    
    def _load_intelligence_data(self):
        """Load intelligence data from disk"""
        try:
            load_path = Path(f"state_intelligence_{self.user_id}.json")
            if not load_path.exists():
                return
            
            with open(load_path, 'r') as f:
                data = json.load(f)
            
            # Reconstruct visits
            for visit_data in data.get('visits', []):
                visit = StateVisit(
                    state_id=visit_data['state_id'],
                    app_id=visit_data['app_id'],
                    timestamp=datetime.fromisoformat(visit_data['timestamp']),
                    duration=timedelta(seconds=visit_data['duration']) if visit_data['duration'] else None,
                    transition_to=visit_data.get('transition_to'),
                    user_triggered=visit_data.get('user_triggered', True),
                    context=visit_data.get('context', {})
                )
                self.state_visits.append(visit)
                self.state_frequencies[visit.state_id] += 1
                
                if visit.duration:
                    self.state_durations[visit.state_id].append(visit.duration)
                
                if visit.transition_to:
                    self.transition_matrix[visit.state_id][visit.transition_to] += 1
            
            # Reconstruct preferences
            prefs = data.get('preferences', {})
            self.user_preference.preferred_states = set(prefs.get('preferred_states', []))
            self.user_preference.avoided_states = set(prefs.get('avoided_states', []))
            self.user_preference.workflow_sequences = prefs.get('workflow_sequences', [])
            self.user_preference.personalization_score = prefs.get('personalization_score', 0.0)
            
            # Re-analyze patterns
            if self.state_visits:
                self._analyze_patterns()
            
            logger.info(f"Loaded {len(self.state_visits)} visits for user {self.user_id}")
        
        except Exception as e:
            logger.error(f"Failed to load intelligence data: {e}")


# Global instance
_state_intelligence_instance = None

def get_state_intelligence(user_id: str = "default") -> StateIntelligence:
    """Get or create state intelligence instance"""
    global _state_intelligence_instance
    if _state_intelligence_instance is None or _state_intelligence_instance.user_id != user_id:
        _state_intelligence_instance = StateIntelligence(user_id)
    return _state_intelligence_instance