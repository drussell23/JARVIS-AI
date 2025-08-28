#!/usr/bin/env python3
"""
Workflow Learning System for JARVIS Multi-Window Intelligence
Learns user patterns and predicts window relationships over time
"""

import json
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from pathlib import Path
import pickle
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from .window_detector import WindowDetector, WindowInfo
from .window_relationship_detector import WindowRelationshipDetector, WindowRelationship

logger = logging.getLogger(__name__)

@dataclass
class WorkflowPattern:
    """Represents a learned workflow pattern"""
    pattern_id: str
    window_apps: List[str]  # Apps commonly used together
    window_titles_keywords: List[str]  # Common keywords in titles
    time_of_day: str  # morning, afternoon, evening, night
    day_of_week: List[str]  # weekdays when pattern occurs
    frequency: int  # How often this pattern occurs
    confidence: float  # Confidence in this pattern
    last_seen: datetime
    created_at: datetime

@dataclass
class WindowSession:
    """Represents a window configuration session"""
    session_id: str
    timestamp: datetime
    windows: List[WindowInfo]
    relationships: List[WindowRelationship]
    duration_minutes: int = 0
    user_activity: str = "unknown"  # coding, meeting, research, etc.

@dataclass
class WorkflowPrediction:
    """Prediction about missing windows or relationships"""
    prediction_type: str  # 'missing_window', 'likely_relationship', 'workflow_suggestion'
    confidence: float
    description: str
    suggested_apps: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)

class WorkflowLearningSystem:
    """Learns and predicts user workflow patterns"""
    
    def __init__(self, data_dir: str = "~/.jarvis/workflow_data"):
        self.window_detector = WindowDetector()
        self.relationship_detector = WindowRelationshipDetector()
        
        # Data storage
        self.data_dir = Path(data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.sessions_file = self.data_dir / "sessions.json"
        self.patterns_file = self.data_dir / "patterns.json"
        self.model_file = self.data_dir / "workflow_model.pkl"
        
        # Load existing data
        self.sessions: List[WindowSession] = self._load_sessions()
        self.patterns: List[WorkflowPattern] = self._load_patterns()
        
        # Learning parameters
        self.min_pattern_frequency = 3  # Minimum occurrences to consider a pattern
        self.pattern_confidence_threshold = 0.7
        self.session_timeout_minutes = 30  # New session after 30 min inactivity
        
        # Current session tracking
        self.current_session: Optional[WindowSession] = None
        self.last_window_state: Optional[List[WindowInfo]] = None
        self.last_activity_time: Optional[datetime] = None
        
        # Text vectorizer for title analysis
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    def record_window_state(self, windows: List[WindowInfo] = None) -> None:
        """Record current window state for learning"""
        if windows is None:
            windows = self.window_detector.get_all_windows()
        
        current_time = datetime.now()
        
        # Check if we need a new session
        if self._should_start_new_session(current_time):
            self._save_current_session()
            self._start_new_session(current_time)
        
        # Update current session
        if self.current_session:
            # Detect relationships
            relationships = self.relationship_detector.detect_relationships(windows)
            
            # Update session
            self.current_session.windows = windows
            self.current_session.relationships = relationships
            
            # Calculate duration
            if self.last_activity_time:
                duration = (current_time - self.current_session.timestamp).total_seconds() / 60
                self.current_session.duration_minutes = int(duration)
        
        self.last_window_state = windows
        self.last_activity_time = current_time
    
    def predict_workflow(self, current_windows: List[WindowInfo] = None) -> List[WorkflowPrediction]:
        """Predict likely next windows or missing components"""
        if current_windows is None:
            current_windows = self.window_detector.get_all_windows()
        
        predictions = []
        
        # Get current context
        current_apps = {w.app_name for w in current_windows}
        current_time = datetime.now()
        time_of_day = self._get_time_of_day(current_time)
        day_of_week = current_time.strftime("%A")
        
        # Find matching patterns
        matching_patterns = self._find_matching_patterns(
            current_apps, time_of_day, day_of_week
        )
        
        # Generate predictions from patterns
        for pattern in matching_patterns:
            # Find missing apps from pattern
            missing_apps = set(pattern.window_apps) - current_apps
            
            if missing_apps:
                predictions.append(WorkflowPrediction(
                    prediction_type='missing_window',
                    confidence=pattern.confidence,
                    description=f"You usually also have {', '.join(missing_apps)} open",
                    suggested_apps=list(missing_apps),
                    evidence=[
                        f"Pattern seen {pattern.frequency} times",
                        f"Usually at {pattern.time_of_day}",
                        f"Common on {', '.join(pattern.day_of_week[:2])}"
                    ]
                ))
        
        # Predict relationships
        relationship_predictions = self._predict_relationships(current_windows)
        predictions.extend(relationship_predictions)
        
        # Workflow suggestions
        workflow_suggestions = self._generate_workflow_suggestions(
            current_windows, matching_patterns
        )
        predictions.extend(workflow_suggestions)
        
        # Sort by confidence
        predictions.sort(key=lambda p: p.confidence, reverse=True)
        
        return predictions[:5]  # Return top 5 predictions
    
    def learn_patterns(self) -> List[WorkflowPattern]:
        """Analyze sessions to learn patterns"""
        if len(self.sessions) < self.min_pattern_frequency:
            return []
        
        # Group sessions by time and apps
        session_groups = defaultdict(list)
        
        for session in self.sessions[-100:]:  # Last 100 sessions
            # Create session key
            apps_key = tuple(sorted(set(w.app_name for w in session.windows)))
            time_key = self._get_time_of_day(session.timestamp)
            day_key = session.timestamp.strftime("%A")
            
            key = (apps_key, time_key)
            session_groups[key].append(session)
        
        # Create patterns from groups
        new_patterns = []
        
        for (apps, time_of_day), sessions in session_groups.items():
            if len(sessions) >= self.min_pattern_frequency:
                # Extract common keywords from titles
                all_titles = []
                days_seen = []
                
                for session in sessions:
                    for window in session.windows:
                        if window.window_title:
                            all_titles.append(window.window_title)
                    days_seen.append(session.timestamp.strftime("%A"))
                
                # Find common keywords
                keywords = self._extract_common_keywords(all_titles)
                
                # Count day frequency
                day_counter = Counter(days_seen)
                common_days = [day for day, count in day_counter.most_common(3)]
                
                # Create pattern
                pattern = WorkflowPattern(
                    pattern_id=f"pattern_{len(self.patterns) + len(new_patterns)}",
                    window_apps=list(apps),
                    window_titles_keywords=keywords,
                    time_of_day=time_of_day,
                    day_of_week=common_days,
                    frequency=len(sessions),
                    confidence=min(len(sessions) / 10, 1.0),  # Max confidence at 10 occurrences
                    last_seen=max(s.timestamp for s in sessions),
                    created_at=datetime.now()
                )
                
                new_patterns.append(pattern)
        
        # Add new patterns to existing ones
        self.patterns.extend(new_patterns)
        self._save_patterns()
        
        return new_patterns
    
    def get_workflow_insights(self) -> Dict[str, any]:
        """Get insights about learned workflows"""
        if not self.sessions:
            return {"status": "No data collected yet"}
        
        insights = {
            "total_sessions": len(self.sessions),
            "total_patterns": len(self.patterns),
            "most_common_workflows": [],
            "time_based_patterns": {},
            "app_combinations": []
        }
        
        # Most common workflows
        pattern_by_freq = sorted(self.patterns, key=lambda p: p.frequency, reverse=True)
        for pattern in pattern_by_freq[:5]:
            insights["most_common_workflows"].append({
                "apps": pattern.window_apps,
                "frequency": pattern.frequency,
                "time_of_day": pattern.time_of_day,
                "confidence": pattern.confidence
            })
        
        # Time-based patterns
        for time_period in ["morning", "afternoon", "evening", "night"]:
            time_patterns = [p for p in self.patterns if p.time_of_day == time_period]
            if time_patterns:
                most_common = max(time_patterns, key=lambda p: p.frequency)
                insights["time_based_patterns"][time_period] = {
                    "apps": most_common.window_apps,
                    "frequency": most_common.frequency
                }
        
        # Common app combinations
        app_pairs = defaultdict(int)
        for session in self.sessions:
            apps = [w.app_name for w in session.windows]
            for i in range(len(apps)):
                for j in range(i + 1, len(apps)):
                    pair = tuple(sorted([apps[i], apps[j]]))
                    app_pairs[pair] += 1
        
        top_pairs = sorted(app_pairs.items(), key=lambda x: x[1], reverse=True)[:5]
        insights["app_combinations"] = [
            {"apps": list(pair), "frequency": freq}
            for pair, freq in top_pairs
        ]
        
        return insights
    
    def _should_start_new_session(self, current_time: datetime) -> bool:
        """Check if we should start a new session"""
        if not self.current_session:
            return True
        
        if not self.last_activity_time:
            return True
        
        # Check timeout
        time_since_activity = (current_time - self.last_activity_time).total_seconds() / 60
        return time_since_activity > self.session_timeout_minutes
    
    def _start_new_session(self, timestamp: datetime) -> None:
        """Start a new session"""
        self.current_session = WindowSession(
            session_id=f"session_{timestamp.timestamp()}",
            timestamp=timestamp,
            windows=[],
            relationships=[]
        )
    
    def _save_current_session(self) -> None:
        """Save current session to storage"""
        if self.current_session and self.current_session.windows:
            self.sessions.append(self.current_session)
            self._save_sessions()
            
            # Trigger pattern learning periodically
            if len(self.sessions) % 10 == 0:
                self.learn_patterns()
    
    def _find_matching_patterns(self, current_apps: Set[str], 
                               time_of_day: str, day_of_week: str) -> List[WorkflowPattern]:
        """Find patterns matching current context"""
        matching = []
        
        for pattern in self.patterns:
            # Check if current apps are subset of pattern
            if current_apps.issubset(set(pattern.window_apps)):
                # Boost confidence if time matches
                time_match = pattern.time_of_day == time_of_day
                day_match = day_of_week in pattern.day_of_week
                
                confidence_boost = 0
                if time_match:
                    confidence_boost += 0.2
                if day_match:
                    confidence_boost += 0.1
                
                # Create adjusted pattern
                adjusted_pattern = WorkflowPattern(
                    **{k: v for k, v in asdict(pattern).items() if k != 'confidence'},
                    confidence=min(pattern.confidence + confidence_boost, 1.0)
                )
                
                if adjusted_pattern.confidence >= self.pattern_confidence_threshold:
                    matching.append(adjusted_pattern)
        
        return matching
    
    def _predict_relationships(self, windows: List[WindowInfo]) -> List[WorkflowPrediction]:
        """Predict likely relationships between windows"""
        predictions = []
        
        # Get current relationships
        current_relationships = self.relationship_detector.detect_relationships(windows)
        current_pairs = {(r.window1_id, r.window2_id) for r in current_relationships}
        
        # Analyze historical relationships
        relationship_frequency = defaultdict(int)
        
        for session in self.sessions[-50:]:  # Last 50 sessions
            for rel in session.relationships:
                # Find windows by app name (since IDs change)
                window1_app = next((w.app_name for w in session.windows 
                                  if w.window_id == rel.window1_id), None)
                window2_app = next((w.app_name for w in session.windows 
                                  if w.window_id == rel.window2_id), None)
                
                if window1_app and window2_app:
                    app_pair = tuple(sorted([window1_app, window2_app]))
                    relationship_frequency[app_pair] += 1
        
        # Check for missing relationships
        current_apps = {w.app_name for w in windows}
        
        for (app1, app2), frequency in relationship_frequency.items():
            if app1 in current_apps and app2 in current_apps:
                # Check if relationship exists
                has_relationship = any(
                    (w1.app_name == app1 and w2.app_name == app2) or 
                    (w1.app_name == app2 and w2.app_name == app1)
                    for r in current_relationships
                    for w1 in windows if w1.window_id == r.window1_id
                    for w2 in windows if w2.window_id == r.window2_id
                )
                
                if not has_relationship and frequency >= 3:
                    predictions.append(WorkflowPrediction(
                        prediction_type='likely_relationship',
                        confidence=min(frequency / 10, 0.9),
                        description=f"{app1} and {app2} usually work together",
                        evidence=[f"Seen together {frequency} times"]
                    ))
        
        return predictions
    
    def _generate_workflow_suggestions(self, windows: List[WindowInfo], 
                                     patterns: List[WorkflowPattern]) -> List[WorkflowPrediction]:
        """Generate workflow improvement suggestions"""
        suggestions = []
        
        # Check for incomplete workflows
        if patterns:
            most_likely = patterns[0]
            if len(windows) < len(most_likely.window_apps) * 0.7:
                suggestions.append(WorkflowPrediction(
                    prediction_type='workflow_suggestion',
                    confidence=0.8,
                    description="Your usual workflow seems incomplete",
                    suggested_apps=most_likely.window_apps,
                    evidence=[f"Typical workflow uses {len(most_likely.window_apps)} apps"]
                ))
        
        return suggestions
    
    def _get_time_of_day(self, timestamp: datetime) -> str:
        """Get time of day category"""
        hour = timestamp.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _extract_common_keywords(self, titles: List[str]) -> List[str]:
        """Extract common keywords from window titles"""
        if not titles:
            return []
        
        try:
            # Fit vectorizer
            self.vectorizer.fit(titles)
            
            # Get feature names (keywords)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Calculate importance scores
            tfidf_matrix = self.vectorizer.transform(titles)
            importance = tfidf_matrix.sum(axis=0).A1
            
            # Get top keywords
            top_indices = importance.argsort()[-10:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            
            return keywords[:5]  # Return top 5
        except:
            # Fallback to simple word frequency
            word_freq = defaultdict(int)
            for title in titles:
                words = title.lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        word_freq[word] += 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            return [word for word, _ in top_words[:5]]
    
    def _load_sessions(self) -> List[WindowSession]:
        """Load sessions from file"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    data = json.load(f)
                    # Convert back to WindowSession objects
                    sessions = []
                    for s_data in data:
                        # Recreate WindowInfo objects
                        windows = [WindowInfo(**w) for w in s_data['windows']]
                        relationships = [WindowRelationship(**r) for r in s_data['relationships']]
                        
                        session = WindowSession(
                            session_id=s_data['session_id'],
                            timestamp=datetime.fromisoformat(s_data['timestamp']),
                            windows=windows,
                            relationships=relationships,
                            duration_minutes=s_data.get('duration_minutes', 0),
                            user_activity=s_data.get('user_activity', 'unknown')
                        )
                        sessions.append(session)
                    return sessions
            except Exception as e:
                logger.error(f"Error loading sessions: {e}")
        return []
    
    def _save_sessions(self) -> None:
        """Save sessions to file"""
        try:
            # Convert to JSON-serializable format
            data = []
            for session in self.sessions[-200:]:  # Keep last 200 sessions
                s_data = {
                    'session_id': session.session_id,
                    'timestamp': session.timestamp.isoformat(),
                    'windows': [asdict(w) for w in session.windows],
                    'relationships': [asdict(r) for r in session.relationships],
                    'duration_minutes': session.duration_minutes,
                    'user_activity': session.user_activity
                }
                data.append(s_data)
            
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
    
    def _load_patterns(self) -> List[WorkflowPattern]:
        """Load patterns from file"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    patterns = []
                    for p_data in data:
                        pattern = WorkflowPattern(
                            pattern_id=p_data['pattern_id'],
                            window_apps=p_data['window_apps'],
                            window_titles_keywords=p_data['window_titles_keywords'],
                            time_of_day=p_data['time_of_day'],
                            day_of_week=p_data['day_of_week'],
                            frequency=p_data['frequency'],
                            confidence=p_data['confidence'],
                            last_seen=datetime.fromisoformat(p_data['last_seen']),
                            created_at=datetime.fromisoformat(p_data['created_at'])
                        )
                        patterns.append(pattern)
                    return patterns
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
        return []
    
    def _save_patterns(self) -> None:
        """Save patterns to file"""
        try:
            data = []
            for pattern in self.patterns:
                p_data = asdict(pattern)
                p_data['last_seen'] = pattern.last_seen.isoformat()
                p_data['created_at'] = pattern.created_at.isoformat()
                data.append(p_data)
            
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

def test_workflow_learning():
    """Test workflow learning system"""
    print("🧠 Testing Workflow Learning System")
    print("=" * 50)
    
    learning_system = WorkflowLearningSystem()
    
    # Record current state
    print("\n📊 Recording current window state...")
    learning_system.record_window_state()
    
    # Get predictions
    print("\n🔮 Generating workflow predictions...")
    predictions = learning_system.predict_workflow()
    
    if predictions:
        print(f"\nFound {len(predictions)} predictions:")
        for i, pred in enumerate(predictions, 1):
            print(f"\n   Prediction #{i}:")
            print(f"   Type: {pred.prediction_type}")
            print(f"   Confidence: {pred.confidence:.0%}")
            print(f"   Description: {pred.description}")
            if pred.suggested_apps:
                print(f"   Suggested Apps: {', '.join(pred.suggested_apps)}")
            if pred.evidence:
                print(f"   Evidence:")
                for evidence in pred.evidence:
                    print(f"     • {evidence}")
    else:
        print("\n   No predictions yet (need more data)")
    
    # Get insights
    print("\n📈 Workflow Insights:")
    insights = learning_system.get_workflow_insights()
    
    print(f"   Total Sessions: {insights['total_sessions']}")
    print(f"   Learned Patterns: {insights['total_patterns']}")
    
    if insights['most_common_workflows']:
        print(f"\n   Most Common Workflows:")
        for workflow in insights['most_common_workflows'][:3]:
            print(f"     • {' + '.join(workflow['apps'])} (seen {workflow['frequency']} times)")
    
    if insights['app_combinations']:
        print(f"\n   Common App Pairs:")
        for combo in insights['app_combinations'][:3]:
            print(f"     • {' + '.join(combo['apps'])} (seen {combo['frequency']} times)")
    
    # Learn patterns
    print("\n🎓 Learning new patterns...")
    new_patterns = learning_system.learn_patterns()
    if new_patterns:
        print(f"   Learned {len(new_patterns)} new patterns!")
    else:
        print("   No new patterns found (need more sessions)")
    
    print("\n✅ Workflow learning test complete!")

if __name__ == "__main__":
    test_workflow_learning()