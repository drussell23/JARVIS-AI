#!/usr/bin/env python3
"""
Goal Inference System - Part of Intelligent Understanding System (IUS)
Understands user objectives beyond immediate actions
Memory allocation: 80MB total
"""

import asyncio
import json
import logging
import pickle
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Deque
from collections import defaultdict, deque
import numpy as np

logger = logging.getLogger(__name__)

# Memory management constants (80MB total)
MEMORY_LIMITS = {
    'goal_templates': 20 * 1024 * 1024,     # 20MB
    'inference_engine': 30 * 1024 * 1024,   # 30MB
    'tracking_data': 30 * 1024 * 1024,      # 30MB
}


class GoalLevel(Enum):
    """Goal hierarchy levels"""
    HIGH_LEVEL = auto()      # Long-term objectives
    INTERMEDIATE = auto()    # Session/task goals
    IMMEDIATE = auto()       # Current action goals


class HighLevelGoalType(Enum):
    """High-level goal categories"""
    PROJECT_COMPLETION = "project_completion"
    PROBLEM_SOLVING = "problem_solving"
    INFORMATION_GATHERING = "information_gathering"
    COMMUNICATION = "communication"
    LEARNING_RESEARCH = "learning_research"


class IntermediateGoalType(Enum):
    """Intermediate goal categories"""
    FEATURE_IMPLEMENTATION = "feature_implementation"
    BUG_FIXING = "bug_fixing"
    DOCUMENT_PREPARATION = "document_preparation"
    MEETING_PREPARATION = "meeting_preparation"
    RESPONSE_COMPOSITION = "response_composition"


class ImmediateGoalType(Enum):
    """Immediate goal categories"""
    FIND_INFORMATION = "find_information"
    FIX_ERROR = "fix_error"
    COMPLETE_FORM = "complete_form"
    SEND_MESSAGE = "send_message"
    REVIEW_CONTENT = "review_content"


@dataclass
class Goal:
    """Represents a user goal at any level"""
    goal_id: str
    level: GoalLevel
    goal_type: str  # One of the goal type enums
    description: str
    confidence: float = 0.0
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    progress: float = 0.0  # 0-1
    is_active: bool = True
    is_completed: bool = False
    
    # Relationships
    parent_goal_id: Optional[str] = None
    child_goal_ids: List[str] = field(default_factory=list)
    related_task_ids: List[str] = field(default_factory=list)
    
    # Learning
    success_patterns: List[Dict[str, Any]] = field(default_factory=list)
    completion_time: Optional[timedelta] = None
    
    def update_progress(self, new_progress: float):
        """Update goal progress"""
        self.progress = max(0.0, min(1.0, new_progress))
        self.last_updated = datetime.now()
        
        if self.progress >= 1.0:
            self.is_completed = True
            self.is_active = False
            self.completion_time = self.last_updated - self.created_at


@dataclass
class GoalEvidence:
    """Evidence supporting a goal hypothesis"""
    source: str  # 'action', 'application', 'content', 'time', 'history'
    data: Dict[str, Any]
    weight: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


class GoalTemplate:
    """Template for recognizing specific goal patterns"""
    
    def __init__(self, goal_type: str, level: GoalLevel):
        self.goal_type = goal_type
        self.level = level
        self.required_evidence: List[str] = []
        self.optional_evidence: List[str] = []
        self.negative_evidence: List[str] = []  # Evidence that contradicts this goal
        self.confidence_threshold: float = 0.7
        self.learned_patterns: List[Dict[str, Any]] = []
        
    def match_evidence(self, evidence: List[GoalEvidence]) -> float:
        """Calculate confidence score based on evidence"""
        score = 0.0
        evidence_types = {e.source for e in evidence}
        
        # Check required evidence
        required_met = sum(1 for req in self.required_evidence if req in evidence_types)
        if required_met < len(self.required_evidence):
            return 0.0  # Missing required evidence
        
        score += 0.5  # Base score for meeting requirements
        
        # Add score for optional evidence
        optional_met = sum(1 for opt in self.optional_evidence if opt in evidence_types)
        if self.optional_evidence:
            score += 0.3 * (optional_met / len(self.optional_evidence))
        
        # Check negative evidence
        negative_present = sum(1 for neg in self.negative_evidence if neg in evidence_types)
        if self.negative_evidence and negative_present > 0:
            score *= (1 - 0.2 * negative_present)  # Reduce score
        
        # Apply evidence weights
        total_weight = sum(e.weight for e in evidence)
        if total_weight > 0:
            weighted_score = sum(e.weight for e in evidence if e.source in 
                               self.required_evidence + self.optional_evidence) / total_weight
            score *= weighted_score
        
        # Check learned patterns
        pattern_boost = self._check_learned_patterns(evidence)
        score = min(1.0, score + pattern_boost)
        
        return score
    
    def _check_learned_patterns(self, evidence: List[GoalEvidence]) -> float:
        """Check if evidence matches learned patterns"""
        if not self.learned_patterns:
            return 0.0
        
        boost = 0.0
        for pattern in self.learned_patterns:
            if self._matches_pattern(evidence, pattern):
                boost += pattern.get('confidence_boost', 0.1)
        
        return min(0.2, boost)  # Cap pattern boost
    
    def _matches_pattern(self, evidence: List[GoalEvidence], pattern: Dict[str, Any]) -> bool:
        """Check if evidence matches a specific pattern"""
        # Simplified pattern matching - can be enhanced
        required_sources = pattern.get('sources', [])
        evidence_sources = {e.source for e in evidence}
        return all(src in evidence_sources for src in required_sources)


class GoalInferenceEngine:
    """Main engine for inferring user goals"""
    
    def __init__(self):
        self.templates: Dict[str, GoalTemplate] = {}
        self.active_goals: Dict[str, Goal] = {}
        self.completed_goals: List[Goal] = []
        self.goal_history: Deque[Goal] = deque(maxlen=1000)
        
        # Memory management
        self.memory_usage = {
            'templates': 0,
            'active_goals': 0,
            'history': 0
        }
        
        # Initialize templates
        self._initialize_goal_templates()
        
        # Learning data
        self.success_patterns: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.goal_transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        logger.info("Goal Inference Engine initialized")
    
    def _initialize_goal_templates(self):
        """Initialize goal recognition templates"""
        # High-level goals
        self._create_template(
            HighLevelGoalType.PROJECT_COMPLETION,
            GoalLevel.HIGH_LEVEL,
            required=['application', 'content'],
            optional=['time', 'history'],
            negative=['communication']
        )
        
        self._create_template(
            HighLevelGoalType.PROBLEM_SOLVING,
            GoalLevel.HIGH_LEVEL,
            required=['action', 'content'],
            optional=['application', 'history'],
            negative=[]
        )
        
        self._create_template(
            HighLevelGoalType.INFORMATION_GATHERING,
            GoalLevel.HIGH_LEVEL,
            required=['action', 'content'],
            optional=['application', 'time'],
            negative=['communication']
        )
        
        self._create_template(
            HighLevelGoalType.COMMUNICATION,
            GoalLevel.HIGH_LEVEL,
            required=['application', 'action'],
            optional=['content', 'time'],
            negative=[]
        )
        
        self._create_template(
            HighLevelGoalType.LEARNING_RESEARCH,
            GoalLevel.HIGH_LEVEL,
            required=['content', 'action'],
            optional=['application', 'history'],
            negative=[]
        )
        
        # Intermediate goals
        self._create_template(
            IntermediateGoalType.FEATURE_IMPLEMENTATION,
            GoalLevel.INTERMEDIATE,
            required=['application', 'action'],
            optional=['content', 'history'],
            negative=['communication']
        )
        
        self._create_template(
            IntermediateGoalType.BUG_FIXING,
            GoalLevel.INTERMEDIATE,
            required=['content', 'action'],
            optional=['application', 'history'],
            negative=[]
        )
        
        self._create_template(
            IntermediateGoalType.DOCUMENT_PREPARATION,
            GoalLevel.INTERMEDIATE,
            required=['application', 'content'],
            optional=['action', 'time'],
            negative=['communication']
        )
        
        self._create_template(
            IntermediateGoalType.MEETING_PREPARATION,
            GoalLevel.INTERMEDIATE,
            required=['time', 'content'],
            optional=['application', 'action'],
            negative=[]
        )
        
        self._create_template(
            IntermediateGoalType.RESPONSE_COMPOSITION,
            GoalLevel.INTERMEDIATE,
            required=['communication', 'content'],
            optional=['application', 'action'],
            negative=[]
        )
        
        # Immediate goals
        self._create_template(
            ImmediateGoalType.FIND_INFORMATION,
            GoalLevel.IMMEDIATE,
            required=['action'],
            optional=['content', 'application'],
            negative=[]
        )
        
        self._create_template(
            ImmediateGoalType.FIX_ERROR,
            GoalLevel.IMMEDIATE,
            required=['content', 'action'],
            optional=['application'],
            negative=['communication']
        )
        
        self._create_template(
            ImmediateGoalType.COMPLETE_FORM,
            GoalLevel.IMMEDIATE,
            required=['action', 'content'],
            optional=['application'],
            negative=[]
        )
        
        self._create_template(
            ImmediateGoalType.SEND_MESSAGE,
            GoalLevel.IMMEDIATE,
            required=['communication', 'action'],
            optional=['content'],
            negative=[]
        )
        
        self._create_template(
            ImmediateGoalType.REVIEW_CONTENT,
            GoalLevel.IMMEDIATE,
            required=['content'],
            optional=['action', 'application'],
            negative=['communication']
        )
    
    def _create_template(self, goal_type: Enum, level: GoalLevel, 
                        required: List[str], optional: List[str], 
                        negative: List[str]):
        """Create a goal template"""
        template = GoalTemplate(goal_type.value, level)
        template.required_evidence = required
        template.optional_evidence = optional
        template.negative_evidence = negative
        
        key = f"{level.name}_{goal_type.value}"
        self.templates[key] = template
    
    async def collect_evidence(self, context: Dict[str, Any]) -> List[GoalEvidence]:
        """Collect evidence from current context"""
        evidence = []
        
        # Recent actions
        if 'recent_actions' in context:
            evidence.append(GoalEvidence(
                source='action',
                data={'actions': context['recent_actions']},
                weight=1.0
            ))
        
        # Open applications
        if 'active_applications' in context:
            evidence.append(GoalEvidence(
                source='application',
                data={'apps': context['active_applications']},
                weight=0.9
            ))
        
        # Content being viewed
        if 'content' in context:
            evidence.append(GoalEvidence(
                source='content',
                data=context['content'],
                weight=0.8
            ))
        
        # Time context
        if 'time_context' in context:
            evidence.append(GoalEvidence(
                source='time',
                data=context['time_context'],
                weight=0.7
            ))
        
        # Communication context
        if any(app in context.get('active_applications', []) 
               for app in ['mail', 'slack', 'teams', 'messages']):
            evidence.append(GoalEvidence(
                source='communication',
                data={'communication_active': True},
                weight=0.8
            ))
        
        # Historical context
        if 'history' in context:
            evidence.append(GoalEvidence(
                source='history',
                data=context['history'],
                weight=0.6
            ))
        
        return evidence
    
    async def infer_goals(self, context: Dict[str, Any]) -> Dict[GoalLevel, List[Goal]]:
        """Infer user goals from context"""
        # Collect evidence
        evidence = await self.collect_evidence(context)
        
        # Generate hypotheses for each level
        hypotheses = {
            GoalLevel.HIGH_LEVEL: [],
            GoalLevel.INTERMEDIATE: [],
            GoalLevel.IMMEDIATE: []
        }
        
        # Test each template
        for template_key, template in self.templates.items():
            confidence = template.match_evidence(evidence)
            
            if confidence >= template.confidence_threshold:
                goal = Goal(
                    goal_id=f"goal_{datetime.now().timestamp()}_{template.goal_type}",
                    level=template.level,
                    goal_type=template.goal_type,
                    description=self._generate_goal_description(template.goal_type, evidence),
                    confidence=confidence,
                    evidence=[{'source': e.source, 'data': e.data} for e in evidence]
                )
                
                hypotheses[template.level].append(goal)
        
        # Sort by confidence and limit
        for level in hypotheses:
            hypotheses[level].sort(key=lambda g: g.confidence, reverse=True)
            hypotheses[level] = hypotheses[level][:3]  # Top 3 per level
        
        # Update active goals
        await self._update_active_goals(hypotheses)
        
        return hypotheses
    
    def _generate_goal_description(self, goal_type: str, evidence: List[GoalEvidence]) -> str:
        """Generate human-readable goal description"""
        # Extract key information from evidence
        apps = []
        actions = []
        
        for e in evidence:
            if e.source == 'application':
                apps.extend(e.data.get('apps', []))
            elif e.source == 'action':
                actions.extend(e.data.get('actions', []))
        
        # Generate description based on goal type
        if goal_type == HighLevelGoalType.PROJECT_COMPLETION.value:
            return f"Complete project work in {', '.join(apps[:2]) if apps else 'development environment'}"
        elif goal_type == HighLevelGoalType.PROBLEM_SOLVING.value:
            return f"Solve technical problem{'s' if len(actions) > 3 else ''}"
        elif goal_type == HighLevelGoalType.INFORMATION_GATHERING.value:
            return f"Research and gather information"
        elif goal_type == HighLevelGoalType.COMMUNICATION.value:
            return f"Communicate via {apps[0] if apps else 'messaging'}"
        elif goal_type == HighLevelGoalType.LEARNING_RESEARCH.value:
            return f"Learn about new concepts or technologies"
        elif goal_type == IntermediateGoalType.FEATURE_IMPLEMENTATION.value:
            return f"Implement new feature"
        elif goal_type == IntermediateGoalType.BUG_FIXING.value:
            return f"Fix bug or error"
        elif goal_type == IntermediateGoalType.DOCUMENT_PREPARATION.value:
            return f"Prepare documentation"
        elif goal_type == IntermediateGoalType.MEETING_PREPARATION.value:
            return f"Prepare for upcoming meeting"
        elif goal_type == IntermediateGoalType.RESPONSE_COMPOSITION.value:
            return f"Compose response or message"
        else:
            return f"Perform {goal_type.replace('_', ' ')}"
    
    async def _update_active_goals(self, new_hypotheses: Dict[GoalLevel, List[Goal]]):
        """Update active goals with new hypotheses"""
        # Add high-confidence new goals
        for level, goals in new_hypotheses.items():
            for goal in goals:
                if goal.confidence >= 0.8:  # High confidence threshold
                    # Check if similar goal exists
                    similar = self._find_similar_active_goal(goal)
                    if similar:
                        # Update existing goal
                        similar.confidence = max(similar.confidence, goal.confidence)
                        similar.last_updated = datetime.now()
                        similar.evidence.extend(goal.evidence)
                    else:
                        # Add new goal
                        self.active_goals[goal.goal_id] = goal
                        
                        # Link to parent goals
                        if level == GoalLevel.IMMEDIATE:
                            parent = self._find_parent_goal(goal, GoalLevel.INTERMEDIATE)
                            if parent:
                                goal.parent_goal_id = parent.goal_id
                                parent.child_goal_ids.append(goal.goal_id)
                        elif level == GoalLevel.INTERMEDIATE:
                            parent = self._find_parent_goal(goal, GoalLevel.HIGH_LEVEL)
                            if parent:
                                goal.parent_goal_id = parent.goal_id
                                parent.child_goal_ids.append(goal.goal_id)
        
        # Prune inactive goals
        await self._prune_inactive_goals()
    
    def _find_similar_active_goal(self, new_goal: Goal) -> Optional[Goal]:
        """Find similar active goal"""
        for goal_id, active_goal in self.active_goals.items():
            if (active_goal.level == new_goal.level and 
                active_goal.goal_type == new_goal.goal_type and
                active_goal.is_active):
                return active_goal
        return None
    
    def _find_parent_goal(self, child_goal: Goal, parent_level: GoalLevel) -> Optional[Goal]:
        """Find appropriate parent goal"""
        candidates = [g for g in self.active_goals.values() 
                     if g.level == parent_level and g.is_active]
        
        if not candidates:
            return None
        
        # Simple heuristic - return most recent high-confidence goal
        candidates.sort(key=lambda g: (g.confidence, g.last_updated), reverse=True)
        return candidates[0]
    
    async def _prune_inactive_goals(self):
        """Remove inactive or old goals"""
        current_time = datetime.now()
        to_remove = []
        
        for goal_id, goal in self.active_goals.items():
            # Remove completed goals
            if goal.is_completed:
                self.completed_goals.append(goal)
                self.goal_history.append(goal)
                to_remove.append(goal_id)
                continue
            
            # Remove stale goals
            if current_time - goal.last_updated > timedelta(minutes=30):
                goal.is_active = False
                self.goal_history.append(goal)
                to_remove.append(goal_id)
        
        for goal_id in to_remove:
            del self.active_goals[goal_id]
    
    def track_goal_progress(self, goal_id: str, progress_delta: float):
        """Update goal progress"""
        if goal_id in self.active_goals:
            goal = self.active_goals[goal_id]
            goal.update_progress(goal.progress + progress_delta)
            
            # Update parent progress
            if goal.parent_goal_id and goal.parent_goal_id in self.active_goals:
                parent = self.active_goals[goal.parent_goal_id]
                # Parent progress is average of children
                if parent.child_goal_ids:
                    child_progress = []
                    for child_id in parent.child_goal_ids:
                        if child_id in self.active_goals:
                            child_progress.append(self.active_goals[child_id].progress)
                    if child_progress:
                        parent.update_progress(sum(child_progress) / len(child_progress))
    
    def learn_from_completion(self, goal: Goal, success: bool):
        """Learn from goal completion"""
        if success:
            # Store success pattern
            pattern = {
                'goal_type': goal.goal_type,
                'evidence_types': [e['source'] for e in goal.evidence],
                'completion_time': goal.completion_time.total_seconds() if goal.completion_time else None,
                'confidence_boost': 0.05
            }
            self.success_patterns[goal.goal_type].append(pattern)
            
            # Update template with learned pattern
            template_key = f"{goal.level.name}_{goal.goal_type}"
            if template_key in self.templates:
                self.templates[template_key].learned_patterns.append(pattern)
        
        # Track goal transitions
        if goal.parent_goal_id:
            self.goal_transitions[goal.parent_goal_id][goal.goal_id] += 1
    
    def get_active_goals_summary(self) -> Dict[str, Any]:
        """Get summary of active goals"""
        summary = {
            'total_active': len(self.active_goals),
            'by_level': {
                GoalLevel.HIGH_LEVEL.name: 0,
                GoalLevel.INTERMEDIATE.name: 0,
                GoalLevel.IMMEDIATE.name: 0
            },
            'high_confidence': [],
            'recently_updated': [],
            'near_completion': []
        }
        
        for goal in self.active_goals.values():
            if goal.is_active:
                summary['by_level'][goal.level.name] += 1
                
                if goal.confidence >= 0.9:
                    summary['high_confidence'].append({
                        'goal_id': goal.goal_id,
                        'type': goal.goal_type,
                        'description': goal.description,
                        'confidence': goal.confidence
                    })
                
                if datetime.now() - goal.last_updated < timedelta(minutes=5):
                    summary['recently_updated'].append({
                        'goal_id': goal.goal_id,
                        'type': goal.goal_type,
                        'description': goal.description
                    })
                
                if goal.progress >= 0.8:
                    summary['near_completion'].append({
                        'goal_id': goal.goal_id,
                        'type': goal.goal_type,
                        'description': goal.description,
                        'progress': goal.progress
                    })
        
        # Limit lists
        for key in ['high_confidence', 'recently_updated', 'near_completion']:
            summary[key] = summary[key][:5]
        
        return summary
    
    def get_goal_insights(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed insights for a specific goal"""
        if goal_id not in self.active_goals:
            # Check completed goals
            for goal in self.completed_goals:
                if goal.goal_id == goal_id:
                    return self._generate_goal_insights(goal)
            return None
        
        goal = self.active_goals[goal_id]
        return self._generate_goal_insights(goal)
    
    def _generate_goal_insights(self, goal: Goal) -> Dict[str, Any]:
        """Generate insights for a goal"""
        insights = {
            'goal_id': goal.goal_id,
            'type': goal.goal_type,
            'level': goal.level.name,
            'description': goal.description,
            'confidence': goal.confidence,
            'progress': goal.progress,
            'is_active': goal.is_active,
            'is_completed': goal.is_completed,
            'duration': (datetime.now() - goal.created_at).total_seconds(),
            'evidence_summary': self._summarize_evidence(goal.evidence),
            'child_goals': [],
            'parent_goal': None
        }
        
        # Add child goals
        for child_id in goal.child_goal_ids:
            if child_id in self.active_goals:
                child = self.active_goals[child_id]
                insights['child_goals'].append({
                    'goal_id': child.goal_id,
                    'type': child.goal_type,
                    'progress': child.progress
                })
        
        # Add parent goal
        if goal.parent_goal_id and goal.parent_goal_id in self.active_goals:
            parent = self.active_goals[goal.parent_goal_id]
            insights['parent_goal'] = {
                'goal_id': parent.goal_id,
                'type': parent.goal_type,
                'description': parent.description
            }
        
        # Add predictions
        if goal.goal_type in self.success_patterns:
            patterns = self.success_patterns[goal.goal_type]
            if patterns:
                avg_time = np.mean([p['completion_time'] for p in patterns 
                                  if p['completion_time'] is not None])
                insights['predicted_completion_time'] = avg_time
        
        return insights
    
    def _summarize_evidence(self, evidence: List[Dict[str, Any]]) -> Dict[str, int]:
        """Summarize evidence types"""
        summary = defaultdict(int)
        for e in evidence:
            summary[e['source']] += 1
        return dict(summary)
    
    def save_state(self, path: Path):
        """Save inference engine state"""
        state = {
            'templates': {k: {
                'goal_type': t.goal_type,
                'level': t.level.name,
                'learned_patterns': t.learned_patterns
            } for k, t in self.templates.items()},
            'success_patterns': dict(self.success_patterns),
            'goal_transitions': dict(self.goal_transitions),
            'completed_goals': len(self.completed_goals)
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self, path: Path):
        """Load inference engine state"""
        if not path.exists():
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        # Update templates with learned patterns
        for template_key, template_data in state.get('templates', {}).items():
            if template_key in self.templates:
                self.templates[template_key].learned_patterns = template_data.get('learned_patterns', [])
        
        # Load success patterns
        self.success_patterns = defaultdict(list, state.get('success_patterns', {}))
        self.goal_transitions = defaultdict(lambda: defaultdict(int), state.get('goal_transitions', {}))


# Global instance management
_goal_inference_instance = None

def get_goal_inference_engine() -> GoalInferenceEngine:
    """Get or create the global Goal Inference Engine instance"""
    global _goal_inference_instance
    if _goal_inference_instance is None:
        _goal_inference_instance = GoalInferenceEngine()
    return _goal_inference_instance


async def test_goal_inference():
    """Test the Goal Inference System"""
    print("🎯 Testing Goal Inference System")
    print("=" * 50)
    
    engine = get_goal_inference_engine()
    
    # Test context 1: Development work
    print("\n1️⃣ Testing development context...")
    dev_context = {
        'active_applications': ['vscode', 'terminal', 'chrome'],
        'recent_actions': ['typing', 'switching_tabs', 'scrolling'],
        'content': {
            'type': 'code',
            'language': 'python',
            'has_errors': True
        },
        'time_context': {
            'time_of_day': 'afternoon',
            'day_of_week': 'weekday'
        }
    }
    
    goals = await engine.infer_goals(dev_context)
    print("\n   Inferred goals:")
    for level, level_goals in goals.items():
        if level_goals:
            print(f"\n   {level.name}:")
            for goal in level_goals:
                print(f"     - {goal.description} (confidence: {goal.confidence:.2f})")
    
    # Test context 2: Communication
    print("\n2️⃣ Testing communication context...")
    comm_context = {
        'active_applications': ['slack', 'mail'],
        'recent_actions': ['typing', 'reading'],
        'content': {
            'type': 'message',
            'draft_present': True
        }
    }
    
    goals = await engine.infer_goals(comm_context)
    print("\n   Inferred goals:")
    for level, level_goals in goals.items():
        if level_goals:
            print(f"\n   {level.name}:")
            for goal in level_goals:
                print(f"     - {goal.description} (confidence: {goal.confidence:.2f})")
    
    # Test goal tracking
    print("\n3️⃣ Testing goal progress tracking...")
    if engine.active_goals:
        first_goal_id = list(engine.active_goals.keys())[0]
        print(f"   Updating progress for goal: {first_goal_id[:20]}...")
        
        engine.track_goal_progress(first_goal_id, 0.3)
        engine.track_goal_progress(first_goal_id, 0.4)
        
        insights = engine.get_goal_insights(first_goal_id)
        if insights:
            print(f"   Goal progress: {insights['progress']:.1%}")
    
    # Test summary
    print("\n4️⃣ Getting active goals summary...")
    summary = engine.get_active_goals_summary()
    print(f"   Total active goals: {summary['total_active']}")
    print(f"   By level: {summary['by_level']}")
    
    print("\n✅ Goal Inference System test complete!")


if __name__ == "__main__":
    asyncio.run(test_goal_inference())