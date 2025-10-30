#!/usr/bin/env python3
"""
Advanced Contextual Understanding Module for JARVIS

This module provides deep emotional intelligence and adaptive personality capabilities
for JARVIS, enabling sophisticated understanding of user context, emotional states,
and behavioral patterns. It uses machine learning and AI to analyze user behavior
and adapt JARVIS's personality and responses accordingly.

The module includes:
- Emotional state detection and tracking
- Cognitive load assessment
- Work context understanding
- Behavioral pattern analysis
- Adaptive personality modeling
- Contextual insight generation

Example:
    >>> engine = ContextualUnderstandingEngine(api_key="your_key")
    >>> state = await engine.analyze_user_state(workspace_data, activity_data)
    >>> response = await engine.generate_empathetic_response("I'm stressed", state)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import anthropic
import re

logger = logging.getLogger(__name__)

class EmotionalState(Enum):
    """Comprehensive emotional states with nuanced understanding.
    
    Represents the various emotional states that can be detected and tracked
    for users, enabling JARVIS to respond appropriately to user emotions.
    
    Attributes:
        FOCUSED: User is concentrated and engaged
        DEEP_FLOW: User is in a state of deep focus and productivity
        STRESSED: User is experiencing stress or pressure
        OVERWHELMED: User is feeling overwhelmed by tasks or information
        RELAXED: User is in a calm, relaxed state
        ENERGETIC: User has high energy and enthusiasm
        TIRED: User is experiencing fatigue
        FRUSTRATED: User is feeling frustrated or annoyed
        CREATIVE: User is in a creative, innovative mindset
        COLLABORATIVE: User is engaged in collaborative work
        CONTEMPLATIVE: User is in a thoughtful, reflective state
        NEUTRAL: Default state with no strong emotional indicators
    """
    FOCUSED = "focused"
    DEEP_FLOW = "deep_flow"
    STRESSED = "stressed"
    OVERWHELMED = "overwhelmed"
    RELAXED = "relaxed"
    ENERGETIC = "energetic"
    TIRED = "tired"
    FRUSTRATED = "frustrated"
    CREATIVE = "creative"
    COLLABORATIVE = "collaborative"
    CONTEMPLATIVE = "contemplative"
    NEUTRAL = "neutral"

class CognitiveLoad(Enum):
    """User's cognitive load levels.
    
    Represents different levels of mental workload and cognitive demand
    that a user is experiencing, used to adapt JARVIS's behavior.
    
    Attributes:
        MINIMAL: Very low cognitive demand
        LOW: Light cognitive load
        MODERATE: Balanced cognitive load
        HIGH: Heavy cognitive load
        OVERLOAD: Excessive cognitive demand that may impair performance
    """
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    OVERLOAD = "overload"

class WorkContext(Enum):
    """Detailed work contexts.
    
    Represents different types of work activities and contexts that
    users engage in, enabling context-aware assistance.
    
    Attributes:
        DEEP_WORK: Focused, uninterrupted work requiring concentration
        CREATIVE_WORK: Creative tasks like design, writing, brainstorming
        ANALYTICAL_WORK: Data analysis, research, problem-solving
        MEETINGS: Video calls, conferences, collaborative sessions
        COMMUNICATION: Email, messaging, correspondence
        RESEARCH: Information gathering, reading, studying
        DEVELOPMENT: Programming, coding, software development
        PLANNING: Task planning, scheduling, organization
        LEARNING: Educational activities, skill development
        ADMINISTRATIVE: Administrative tasks, paperwork
        BREAK: Rest periods, breaks from work
        TRANSITION: Moving between different work contexts
    """
    DEEP_WORK = "deep_work"
    CREATIVE_WORK = "creative_work"
    ANALYTICAL_WORK = "analytical_work"
    MEETINGS = "meetings"
    COMMUNICATION = "communication"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    PLANNING = "planning"
    LEARNING = "learning"
    ADMINISTRATIVE = "administrative"
    BREAK = "break"
    TRANSITION = "transition"

@dataclass
class UserBehaviorSignal:
    """Behavioral signal for understanding user state.
    
    Represents a single behavioral data point that contributes to
    understanding the user's current state and context.
    
    Attributes:
        signal_type: Type of behavioral signal (e.g., 'typing_pattern')
        value: Normalized signal value (typically 0.0-1.0)
        timestamp: When the signal was recorded
        confidence: Confidence level in the signal accuracy (0.0-1.0)
        context: Additional contextual information about the signal
    """
    signal_type: str
    value: float
    timestamp: datetime
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EmotionalProfile:
    """Dynamic emotional profile that adapts over time.
    
    Stores learned patterns about a user's emotional tendencies,
    energy levels, and behavioral patterns to enable personalized
    understanding and responses.
    
    Attributes:
        baseline_mood: User's typical emotional state
        stress_threshold: Threshold for detecting stress (0.0-1.0)
        energy_patterns: Energy levels by hour of day (hour -> energy)
        trigger_patterns: Patterns that trigger specific emotions
        coping_mechanisms: User's preferred coping strategies
        personality_traits: Learned personality characteristics
    """
    baseline_mood: EmotionalState = EmotionalState.NEUTRAL
    stress_threshold: float = 0.7
    energy_patterns: Dict[int, float] = field(default_factory=dict)  # Hour -> energy level
    trigger_patterns: Dict[str, List[str]] = field(default_factory=dict)
    coping_mechanisms: List[str] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)

@dataclass
class ContextualInsight:
    """Deep insight about user's current context.
    
    Represents an analytical insight about the user's current state,
    including recommendations and predictions about impact.
    
    Attributes:
        insight_type: Category of insight (e.g., 'stress_overload')
        description: Human-readable description of the insight
        confidence: Confidence level in the insight (0.0-1.0)
        supporting_evidence: List of evidence supporting the insight
        recommended_actions: Suggested actions with urgency levels
        impact_prediction: Predicted impacts on various metrics
    """
    insight_type: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    recommended_actions: List[Dict[str, Any]]
    impact_prediction: Dict[str, float]

class ContextualUnderstandingEngine:
    """Advanced contextual understanding with emotional intelligence.
    
    Main engine for analyzing user behavior, detecting emotional states,
    understanding work context, and generating contextual insights to
    enable JARVIS to provide emotionally intelligent assistance.
    
    This engine combines behavioral signal analysis, machine learning,
    and AI-powered understanding to create a comprehensive model of
    user state and context.
    
    Attributes:
        claude: Anthropic Claude client for AI analysis
        use_intelligent_selection: Whether to use intelligent model selection
        emotional_history: Recent emotional state history
        emotional_profile: Learned emotional patterns
        emotion_transitions: Patterns of emotional state changes
        behavior_signals: Recent behavioral signals
        behavior_patterns: Learned behavioral patterns
        context_history: Recent work context history
        personality_model: Adaptive personality model
        understanding_accuracy: Current accuracy of understanding
    """
    
    def __init__(self, anthropic_api_key: str, use_intelligent_selection: bool = True):
        """Initialize the contextual understanding engine.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
            use_intelligent_selection: Whether to use intelligent model selection
            
        Raises:
            ValueError: If API key is invalid or missing
        """
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self.use_intelligent_selection = use_intelligent_selection

        # Emotional understanding
        self.emotional_history = deque(maxlen=1000)
        self.emotional_profile = EmotionalProfile()
        self.emotion_transitions = defaultdict(lambda: defaultdict(int))

        # Behavioral analysis
        self.behavior_signals = deque(maxlen=5000)
        self.behavior_patterns = {}
        self.anomaly_detector = None

        # Context understanding
        self.context_history = deque(maxlen=2000)
        self.context_embeddings = {}
        self.context_clusters = []

        # Personality adaptation
        self.personality_model = self._initialize_personality_model()
        self.interaction_styles = {}

        # Learning components
        self.understanding_accuracy = 0.7
        self.adaptation_rate = 0.1

        # Initialize ML components
        self._initialize_ml_models()
    
    def _initialize_personality_model(self) -> Dict[str, Any]:
        """Initialize adaptive personality model.
        
        Creates the base personality model with traits, adaptation factors,
        and communication styles that JARVIS can use to adapt its behavior.
        
        Returns:
            Dictionary containing personality model configuration with:
            - base_traits: Core personality characteristics
            - adaptation_factors: Factors that influence adaptation
            - communication_styles: Predefined communication styles
        """
        return {
            'base_traits': {
                'warmth': 0.7,
                'competence': 0.9,
                'enthusiasm': 0.6,
                'formality': 0.5,
                'humor': 0.4,
                'empathy': 0.8
            },
            'adaptation_factors': {
                'time_of_day': 1.0,
                'user_mood': 1.0,
                'task_urgency': 1.0,
                'relationship_depth': 1.0
            },
            'communication_styles': {
                'professional': {'formality': 0.8, 'warmth': 0.5, 'humor': 0.2},
                'friendly': {'formality': 0.3, 'warmth': 0.9, 'humor': 0.6},
                'focused': {'formality': 0.5, 'warmth': 0.4, 'humor': 0.1},
                'supportive': {'formality': 0.4, 'warmth': 0.9, 'empathy': 1.0}
            }
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models.
        
        Sets up the ML models used for behavioral pattern recognition,
        clustering, and anomaly detection.
        """
        # Clustering for behavior patterns
        self.behavior_clusterer = KMeans(n_clusters=10, random_state=42)
        
        # PCA for dimensionality reduction
        self.context_pca = PCA(n_components=50)
        
        # Pattern recognition models
        self.pattern_models = {
            'stress_detection': {'threshold': 0.7, 'weights': {}},
            'flow_state': {'threshold': 0.8, 'weights': {}},
            'fatigue_detection': {'threshold': 0.6, 'weights': {}}
        }
    
    async def analyze_user_state(self, workspace_state: Dict[str, Any],
                               activity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis of user's current state.
        
        Performs a complete analysis of the user's current state including
        emotional state, cognitive load, work context, and generates
        contextual insights.
        
        Args:
            workspace_state: Current workspace information including windows,
                           notifications, and application states
            activity_data: User activity metrics including typing patterns,
                         click patterns, and interaction data
                         
        Returns:
            Dictionary containing:
            - emotional_state: Detected emotional state
            - cognitive_load: Assessed cognitive load level
            - work_context: Identified work context
            - insights: List of contextual insights
            - confidence: Overall confidence in analysis
            - personality_adaptation: Adapted personality traits
            
        Raises:
            ValueError: If input data is malformed
            RuntimeError: If analysis fails due to system errors
            
        Example:
            >>> engine = ContextualUnderstandingEngine(api_key)
            >>> workspace = {"windows": [...], "notifications": {...}}
            >>> activity = {"typing_speed": 65, "window_switches": 12}
            >>> state = await engine.analyze_user_state(workspace, activity)
            >>> print(state['emotional_state'])
            EmotionalState.FOCUSED
        """
        # Collect behavioral signals
        signals = await self._extract_behavioral_signals(workspace_state, activity_data)
        
        # Detect emotional state
        emotional_state = await self._detect_emotional_state(signals)
        
        # Assess cognitive load
        cognitive_load = await self._assess_cognitive_load(workspace_state, activity_data)
        
        # Understand work context
        work_context = await self._understand_work_context(workspace_state)
        
        # Generate contextual insights
        insights = await self._generate_contextual_insights(
            emotional_state, cognitive_load, work_context, signals
        )
        
        # Update learning
        await self._update_understanding(emotional_state, work_context, signals)
        
        return {
            'emotional_state': emotional_state,
            'cognitive_load': cognitive_load,
            'work_context': work_context,
            'insights': insights,
            'confidence': self._calculate_confidence(signals),
            'personality_adaptation': await self._adapt_personality(emotional_state, work_context)
        }
    
    async def _extract_behavioral_signals(self, workspace_state: Dict[str, Any],
                                        activity_data: Dict[str, Any]) -> List[UserBehaviorSignal]:
        """Extract behavioral signals from user activity.
        
        Analyzes user activity data to extract meaningful behavioral signals
        that can be used to understand the user's current state.
        
        Args:
            workspace_state: Current workspace information
            activity_data: User activity metrics and patterns
            
        Returns:
            List of UserBehaviorSignal objects representing different
            aspects of user behavior
            
        Raises:
            KeyError: If required activity data is missing
        """
        signals = []
        timestamp = datetime.now()
        
        # Window switching patterns
        window_switches = activity_data.get('window_switches', 0)
        signals.append(UserBehaviorSignal(
            signal_type='window_switching',
            value=min(window_switches / 30, 1.0),  # Normalize to 0-1
            timestamp=timestamp,
            confidence=0.9,
            context={'raw_count': window_switches}
        ))
        
        # Typing patterns
        typing_speed = activity_data.get('typing_speed', 0)
        typing_variance = activity_data.get('typing_variance', 0)
        signals.append(UserBehaviorSignal(
            signal_type='typing_pattern',
            value=typing_speed / 100,  # Normalize WPM
            timestamp=timestamp,
            confidence=0.8,
            context={'variance': typing_variance}
        ))
        
        # Click patterns
        click_rate = activity_data.get('click_rate', 0)
        click_accuracy = activity_data.get('click_accuracy', 1.0)
        signals.append(UserBehaviorSignal(
            signal_type='click_pattern',
            value=click_rate / 60,  # Clicks per minute normalized
            timestamp=timestamp,
            confidence=0.85,
            context={'accuracy': click_accuracy}
        ))
        
        # Application usage patterns
        app_diversity = len(set(w.get('app_name', '') for w in workspace_state.get('windows', [])))
        signals.append(UserBehaviorSignal(
            signal_type='app_diversity',
            value=min(app_diversity / 10, 1.0),
            timestamp=timestamp,
            confidence=0.9,
            context={'active_apps': app_diversity}
        ))
        
        # Idle time patterns
        idle_time = activity_data.get('idle_time', 0)
        signals.append(UserBehaviorSignal(
            signal_type='idle_pattern',
            value=min(idle_time / 300, 1.0),  # 5 minutes max
            timestamp=timestamp,
            confidence=0.95,
            context={'seconds_idle': idle_time}
        ))
        
        # Error/correction patterns
        error_rate = activity_data.get('error_corrections', 0) / max(activity_data.get('total_actions', 1), 1)
        signals.append(UserBehaviorSignal(
            signal_type='error_pattern',
            value=min(error_rate * 10, 1.0),
            timestamp=timestamp,
            confidence=0.7,
            context={'correction_rate': error_rate}
        ))
        
        # Store signals for learning
        self.behavior_signals.extend(signals)
        
        return signals
    
    async def _detect_emotional_state_with_intelligent_selection(self, signals: List[UserBehaviorSignal]) -> EmotionalState:
        """Detect user's emotional state using intelligent model selection.
        
        Uses the hybrid orchestrator to intelligently select the best model
        for emotional state detection based on the current context.
        
        Args:
            signals: List of behavioral signals to analyze
            
        Returns:
            Detected emotional state
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            RuntimeError: If intelligent selection fails
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Create signal vector
            signal_vector = self._create_signal_vector(signals)

            # Build signal summary
            signal_summary = {
                sig.signal_type: {
                    'value': sig.value,
                    'context': sig.context
                }
                for sig in signals
            }

            # Build rich context for intelligent selection
            intelligent_context = {
                "task_type": "emotional_analysis",
                "user_state": {
                    "behavioral_signals": signal_summary,
                    "signal_patterns": {
                        "window_switching": next((s.value for s in signals if s.signal_type == 'window_switching'), 0),
                        "typing_consistency": next((s.value for s in signals if s.signal_type == 'typing_pattern'), 0),
                        "error_rate": next((s.value for s in signals if s.signal_type == 'error_pattern'), 0)
                    }
                },
                "cognitive_load_indicators": {
                    "high_switching": any(s.value > 0.7 for s in signals if s.signal_type == 'window_switching'),
                    "high_errors": any(s.value > 0.5 for s in signals if s.signal_type == 'error_pattern')
                },
                "analysis_target": "emotional_state_classification"
            }

            prompt = f"""Analyze these behavioral signals to determine the user's emotional state:

Signals:
{json.dumps(signal_summary, indent=2)}

Consider:
1. Window switching: High = stressed/overwhelmed, Low = focused
2. Typing patterns: Fast+consistent = flow state, Fast+erratic = stressed
3. Click patterns: Rapid = frustrated/stressed, Measured = focused
4. App diversity: High = scattered, Low = focused
5. Idle time: High = tired/contemplative, Low = engaged
6. Error rate: High = frustrated/tired, Low = confident

Based on these patterns, what is the user's most likely emotional state?
Options: focused, deep_flow, stressed, overwhelmed, relaxed, energetic,
tired, frustrated, creative, collaborative, contemplative, neutral

Provide your analysis with:
- Primary emotion (one of the above)
- Confidence (0-1)
- Supporting evidence (which signals support this)
- Secondary emotions if present"""

            result = await orchestrator.execute_with_intelligent_model_selection(
                query=prompt,
                intent="emotional_analysis",
                required_capabilities={"nlp_analysis", "emotional_intelligence", "context_understanding"},
                context=intelligent_context,
                max_tokens=500,
                temperature=0.7,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            analysis = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ Emotional state analysis generated using {model_used}")

            # Parse emotional state
            for state in EmotionalState:
                if state.value in analysis.lower():
                    # Update emotional history
                    self.emotional_history.append({
                        'state': state,
                        'timestamp': datetime.now(),
                        'signals': signal_vector.tolist(),
                        'confidence': self._extract_confidence(analysis)
                    })

                    # Update transitions
                    if len(self.emotional_history) > 1:
                        prev_state = self.emotional_history[-2]['state']
                        self.emotion_transitions[prev_state.value][state.value] += 1

                    return state

            # If no match found, return neutral
            return EmotionalState.NEUTRAL

        except ImportError:
            logger.warning("Hybrid orchestrator not available for emotional state detection")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent emotional detection: {e}")
            raise

    async def _detect_emotional_state(self, signals: List[UserBehaviorSignal]) -> EmotionalState:
        """Detect user's emotional state from behavioral signals.
        
        Analyzes behavioral signals to determine the user's current emotional
        state using AI analysis with fallback to rule-based detection.
        
        Args:
            signals: List of behavioral signals to analyze
            
        Returns:
            Detected emotional state
            
        Raises:
            RuntimeError: If both AI and rule-based detection fail
        """
        # Use intelligent selection first with fallback
        if self.use_intelligent_selection:
            try:
                return await self._detect_emotional_state_with_intelligent_selection(signals)
            except Exception as e:
                logger.warning(f"Intelligent selection failed for emotional state, falling back to direct Claude: {e}")

        # Fallback: Create signal vector
        signal_vector = self._create_signal_vector(signals)

        # Use Claude for nuanced emotional analysis
        signal_summary = {
            sig.signal_type: {
                'value': sig.value,
                'context': sig.context
            }
            for sig in signals
        }

        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze these behavioral signals to determine the user's emotional state:

Signals:
{json.dumps(signal_summary, indent=2)}

Consider:
1. Window switching: High = stressed/overwhelmed, Low = focused
2. Typing patterns: Fast+consistent = flow state, Fast+erratic = stressed
3. Click patterns: Rapid = frustrated/stressed, Measured = focused
4. App diversity: High = scattered, Low = focused
5. Idle time: High = tired/contemplative, Low = engaged
6. Error rate: High = frustrated/tired, Low = confident

Based on these patterns, what is the user's most likely emotional state?
Options: focused, deep_flow, stressed, overwhelmed, relaxed, energetic, 
tired, frustrated, creative, collaborative, contemplative, neutral

Provide your analysis with:
- Primary emotion (one of the above)
- Confidence (0-1)
- Supporting evidence (which signals support this)
- Secondary emotions if present"""
                }]
            )
            
            analysis = response.content[0].text
            
            # Parse emotional state
            for state in EmotionalState:
                if state.value in analysis.lower():
                    # Update emotional history
                    self.emotional_history.append({
                        'state': state,
                        'timestamp': datetime.now(),
                        'signals': signal_vector.tolist(),
                        'confidence': self._extract_confidence(analysis)
                    })
                    
                    # Update transitions
                    if len(self.emotional_history) > 1:
                        prev_state = self.emotional_history[-2]['state']
                        self.emotion_transitions[prev_state.value][state.value] += 1
                    
                    return state
                    
        except Exception as e:
            logger.error(f"Error in emotional detection: {e}")
        
        # Fallback to rule-based detection
        return self._rule_based_emotion_detection(signals)
    
    async def _assess_cognitive_load(self, workspace_state: Dict[str, Any],
                                   activity_data: Dict[str, Any]) -> CognitiveLoad:
        """Assess user's cognitive load.
        
        Evaluates the user's current cognitive load based on workspace
        complexity, task switching, and activity patterns.
        
        Args:
            workspace_state: Current workspace information
            activity_data: User activity metrics
            
        Returns:
            Assessed cognitive load level
            
        Example:
            >>> load = await engine._assess_cognitive_load(workspace, activity)
            >>> print(load)
            CognitiveLoad.MODERATE
        """
        load_score = 0.0
        
        # Window count factor
        window_count = len(workspace_state.get('windows', []))
        load_score += min(window_count / 10, 0.3)  # Max 0.3 contribution
        
        # Task switching factor
        switches = activity_data.get('window_switches', 0)
        load_score += min(switches / 50, 0.2)  # Max 0.2 contribution
        
        # Notification factor
        notifications = len(workspace_state.get('notifications', {}).get('badges', []))
        load_score += min(notifications / 10, 0.15)  # Max 0.15 contribution
        
        # Error rate factor
        error_rate = activity_data.get('error_corrections', 0) / max(activity_data.get('total_actions', 1), 1)
        load_score += min(error_rate * 5, 0.15)  # Max 0.15 contribution
        
        # Speed vs accuracy trade-off
        typing_speed = activity_data.get('typing_speed', 50)
        typing_accuracy = activity_data.get('typing_accuracy', 0.95)
        if typing_speed > 80 and typing_accuracy < 0.9:
            load_score += 0.1  # Rushing indicator
        
        # Time pressure factor
        if activity_data.get('deadline_approaching', False):
            load_score += 0.1
        
        # Map score to cognitive load level
        if load_score < 0.2:
            return CognitiveLoad.MINIMAL
        elif load_score < 0.4:
            return CognitiveLoad.LOW
        elif load_score < 0.6:
            return CognitiveLoad.MODERATE
        elif load_score < 0.8:
            return CognitiveLoad.HIGH
        else:
            return CognitiveLoad.OVERLOAD
    
    async def _understand_work_context(self, workspace_state: Dict[str, Any]) -> WorkContext:
        """Understand the current work context.
        
        Analyzes the active applications and workspace state to determine
        what type of work the user is currently engaged in.
        
        Args:
            workspace_state: Current workspace information including active windows
            
        Returns:
            Identified work context
            
        Example:
            >>> context = await engine._understand_work_context(workspace)
            >>> print(context)
            WorkContext.DEVELOPMENT
        """
        windows = workspace_state.get('windows', [])
        if not windows:
            return WorkContext.BREAK
        
        # Collect app categories
        app_categories = defaultdict(int)
        for window in windows:
            app_name = window.get('app_name', '').lower()
            
            # Categorize apps
            if any(dev in app_name for dev in ['code', 'xcode', 'terminal', 'git', 'docker']):
                app_categories['development'] += 1
            elif any(meet in app_name for meet in ['zoom', 'teams', 'meet', 'webex']):
                app_categories['meetings'] += 1
            elif any(comm in app_name for comm in ['slack', 'discord', 'mail', 'messages']):
                app_categories['communication'] += 1
            elif any(create in app_name for create in ['figma', 'sketch', 'photoshop', 'illustrator']):
                app_categories['creative'] += 1
            elif any(research in app_name for research in ['browser', 'safari', 'chrome', 'firefox']):
                app_categories['research'] += 1
            elif any(plan in app_name for plan in ['calendar', 'notion', 'todoist', 'trello']):
                app_categories['planning'] += 1
            elif any(learn in app_name for learn in ['coursera', 'udemy', 'youtube', 'reader']):
                app_categories['learning'] += 1
            elif any(analyze in app_name for analyze in ['excel', 'sheets', 'tableau', 'jupyter']):
                app_categories['analytical'] += 1
        
        # Determine primary context
        if not app_categories:
            return WorkContext.NEUTRAL
        
        primary_category = max(app_categories.items(), key=lambda x: x[1])[0]
        
        # Map to work context
        context_mapping = {
            'development': WorkContext.DEVELOPMENT,
            'meetings': WorkContext.MEETINGS,
            'communication': WorkContext.COMMUNICATION,
            'creative': WorkContext.CREATIVE_WORK,
            'research': WorkContext.RESEARCH,
            'planning': WorkContext.PLANNING,
            'learning': WorkContext.LEARNING,
            'analytical': WorkContext.ANALYTICAL_WORK
        }
        
        context = context_mapping.get(primary_category, WorkContext.DEEP_WORK)
        
        # Check for transitions
        if len(app_categories) > 3:
            context = WorkContext.TRANSITION
        
        # Update context history
        self.context_history.append({
            'context': context,
            'timestamp': datetime.now(),
            'app_distribution': dict(app_categories)
        })
        
        return context
    
    async def _generate_contextual_insights(self, emotional_state: EmotionalState,
                                          cognitive_load: CognitiveLoad,
                                          work_context: WorkContext,
                                          signals: List[UserBehaviorSignal]) -> List[ContextualInsight]:
        """Generate deep contextual insights.
        
        Analyzes the combination of emotional state, cognitive load, and work
        context to generate actionable insights and recommendations.
        
        Args:
            emotional_state: Current emotional state
            cognitive_load: Current cognitive load level
            work_context: Current work context
            signals: Behavioral signals used in analysis
            
        Returns:
            List of contextual insights with recommendations
            
        Example: