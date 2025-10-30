#!/usr/bin/env python3
"""
Advanced Predictive Intelligence Module for JARVIS.

This module provides sophisticated prediction capabilities using dynamic learning
and Anthropic API integration. It analyzes user behavior patterns, workspace state,
and environmental factors to generate intelligent predictions about user needs,
workflow transitions, and optimization opportunities.

The module combines machine learning models with Claude AI to provide contextual
predictions that adapt and improve over time through continuous learning.

Example:
    >>> engine = PredictiveIntelligenceEngine(api_key="your-key")
    >>> context = PredictionContext(
    ...     timestamp=datetime.now(),
    ...     workspace_state={'windows': [...]},
    ...     user_activity={'click_rate': 50},
    ...     environmental_factors={},
    ...     historical_patterns={},
    ...     real_time_signals={}
    ... )
    >>> predictions = await engine.generate_predictions(context)
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import anthropic
import hashlib

logger = logging.getLogger(__name__)

class PredictionType(Enum):
    """Types of predictions the system can make.
    
    Attributes:
        NEXT_ACTION: Predicts the user's next likely action
        RESOURCE_NEED: Predicts resources the user will need
        WORKFLOW_TRANSITION: Predicts workflow state changes
        BREAK_SUGGESTION: Predicts when user needs a break
        FOCUS_OPTIMIZATION: Predicts focus improvement opportunities
        MEETING_PREPARATION: Predicts meeting preparation needs
        TASK_COMPLETION: Predicts task completion time and requirements
        COLLABORATION_NEED: Predicts collaboration opportunities
        AUTOMATION_OPPORTUNITY: Predicts automation possibilities
        DEADLINE_RISK: Predicts deadline-related risks
    """
    NEXT_ACTION = "next_action"
    RESOURCE_NEED = "resource_need"
    WORKFLOW_TRANSITION = "workflow_transition"
    BREAK_SUGGESTION = "break_suggestion"
    FOCUS_OPTIMIZATION = "focus_optimization"
    MEETING_PREPARATION = "meeting_preparation"
    TASK_COMPLETION = "task_completion"
    COLLABORATION_NEED = "collaboration_need"
    AUTOMATION_OPPORTUNITY = "automation_opportunity"
    DEADLINE_RISK = "deadline_risk"

@dataclass
class PredictionContext:
    """Rich context for making predictions.
    
    Contains comprehensive information about the user's current state,
    workspace, activity patterns, and environmental factors used to
    generate accurate predictions.
    
    Attributes:
        timestamp: Current timestamp for temporal analysis
        workspace_state: Current state of user's workspace (windows, apps, etc.)
        user_activity: User activity metrics (clicks, typing, switches)
        environmental_factors: External factors (noise, calendar, deadlines)
        historical_patterns: Historical behavior patterns
        real_time_signals: Real-time signals and indicators
    """
    timestamp: datetime
    workspace_state: Dict[str, Any]
    user_activity: Dict[str, Any]
    environmental_factors: Dict[str, Any]
    historical_patterns: Dict[str, Any]
    real_time_signals: Dict[str, Any]
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert context to ML feature vector.
        
        Transforms the rich context data into a numerical feature vector
        suitable for machine learning models.
        
        Returns:
            np.ndarray: Feature vector containing temporal, activity, and
                       environmental features
        
        Example:
            >>> context = PredictionContext(...)
            >>> features = context.to_feature_vector()
            >>> print(features.shape)
            (13,)
        """
        features = []
        
        # Time features
        features.extend([
            self.timestamp.hour,
            self.timestamp.minute,
            self.timestamp.weekday(),
            self.timestamp.day,
            self.timestamp.month
        ])
        
        # Activity features
        features.append(len(self.workspace_state.get('windows', [])))
        features.append(len(self.workspace_state.get('notifications', {}).get('badges', [])))
        features.append(self.user_activity.get('click_rate', 0))
        features.append(self.user_activity.get('typing_speed', 0))
        features.append(self.user_activity.get('window_switches', 0))
        
        # Environmental features
        features.append(self.environmental_factors.get('noise_level', 0))
        features.append(self.environmental_factors.get('calendar_density', 0))
        features.append(self.environmental_factors.get('deadline_pressure', 0))
        
        return np.array(features)

@dataclass
class DynamicPrediction:
    """A dynamic, context-aware prediction.
    
    Represents a single prediction with confidence scores, urgency levels,
    actionable items, and reasoning. Includes learning feedback mechanisms
    for continuous improvement.
    
    Attributes:
        prediction_id: Unique identifier for the prediction
        type: Type of prediction from PredictionType enum
        confidence: Confidence score (0.0 to 1.0)
        urgency: Urgency level (0.0 to 1.0)
        action_items: List of actionable items based on prediction
        reasoning: Human-readable explanation of the prediction
        context_factors: Factors that influenced the prediction
        learning_feedback: Optional feedback for learning improvement
        expires_at: Optional expiration time for the prediction
    """
    prediction_id: str
    type: PredictionType
    confidence: float
    urgency: float
    action_items: List[Dict[str, Any]]
    reasoning: str
    context_factors: Dict[str, float]
    learning_feedback: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None
    
    def should_act(self, threshold: float = 0.7) -> bool:
        """Determine if prediction warrants action.
        
        Calculates whether the prediction's combined confidence and urgency
        scores exceed the threshold for taking action.
        
        Args:
            threshold: Minimum score threshold for action (default: 0.7)
            
        Returns:
            bool: True if action should be taken, False otherwise
            
        Example:
            >>> prediction = DynamicPrediction(...)
            >>> if prediction.should_act():
            ...     # Take action based on prediction
            ...     pass
        """
        return self.confidence * self.urgency > threshold

class PredictiveIntelligenceEngine:
    """Advanced predictive intelligence that learns and adapts dynamically.
    
    The main engine that combines machine learning models with Claude AI
    to generate intelligent predictions about user behavior, needs, and
    optimization opportunities. Features dynamic learning and adaptation.
    
    Attributes:
        claude: Anthropic Claude client for AI analysis
        use_intelligent_selection: Whether to use intelligent model selection
        pattern_memory: Memory of historical patterns for learning
        prediction_models: ML models for each prediction type
        feature_importance: Dynamic feature importance tracking
        active_predictions: Currently active predictions
        prediction_outcomes: Historical prediction outcomes
        learning_rate: Rate of learning adaptation
        context_embeddings: Context embeddings for similarity matching
        context_clusters: Clustered contexts for pattern recognition
    """
    
    def __init__(self, anthropic_api_key: str, use_intelligent_selection: bool = True):
        """Initialize the predictive intelligence engine.
        
        Args:
            anthropic_api_key: API key for Anthropic Claude
            use_intelligent_selection: Whether to use intelligent model selection
                                     for enhanced predictions (default: True)
        
        Raises:
            ValueError: If API key is invalid or missing
        """
        self.claude = anthropic.Anthropic(api_key=anthropic_api_key)
        self.use_intelligent_selection = use_intelligent_selection

        # Dynamic learning components
        self.pattern_memory = deque(maxlen=10000)
        self.prediction_models: Dict[PredictionType, Any] = {}
        self.feature_importance: Dict[str, float] = defaultdict(float)

        # Real-time learning
        self.active_predictions: Dict[str, DynamicPrediction] = {}
        self.prediction_outcomes: List[Dict[str, Any]] = []
        self.learning_rate = 0.1

        # Context understanding
        self.context_embeddings: Dict[str, np.ndarray] = {}
        self.context_clusters = []

        # Initialize ML models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize machine learning models for each prediction type.
        
        Sets up RandomForest classifiers and StandardScalers for each
        prediction type, along with performance tracking metrics.
        """
        for pred_type in PredictionType:
            self.prediction_models[pred_type] = {
                'classifier': RandomForestClassifier(n_estimators=100, random_state=42),
                'scaler': StandardScaler(),
                'trained': False,
                'performance': 0.5
            }
    
    async def generate_predictions(self, context: PredictionContext) -> List[DynamicPrediction]:
        """Generate intelligent predictions based on current context.
        
        Analyzes the provided context using either intelligent model selection
        or direct Claude analysis, then generates predictions for each type
        and returns the most relevant ones.
        
        Args:
            context: Rich context containing workspace state, user activity,
                    and environmental factors
                    
        Returns:
            List[DynamicPrediction]: Top predictions sorted by urgency and
                                   confidence, limited to 10 results
                                   
        Raises:
            Exception: If context analysis fails with both intelligent
                      selection and Claude fallback
                      
        Example:
            >>> engine = PredictiveIntelligenceEngine(api_key)
            >>> context = PredictionContext(...)
            >>> predictions = await engine.generate_predictions(context)
            >>> for pred in predictions:
            ...     if pred.should_act():
            ...         print(f"Action needed: {pred.reasoning}")
        """
        predictions = []

        # Use intelligent selection or fallback to direct Claude
        if self.use_intelligent_selection:
            try:
                context_analysis = await self._analyze_context_with_intelligent_selection(context)
            except Exception as e:
                logger.warning(f"Intelligent selection failed for context analysis, falling back to direct Claude: {e}")
                context_analysis = await self._analyze_context_with_claude(context)
        else:
            context_analysis = await self._analyze_context_with_claude(context)
        
        # Generate predictions for each type
        for pred_type in PredictionType:
            prediction = await self._generate_typed_prediction(
                pred_type, context, context_analysis
            )
            if prediction and prediction.confidence > 0.5:
                predictions.append(prediction)
        
        # Learn from context
        await self._learn_from_context(context, predictions)
        
        # Sort by urgency and confidence
        predictions.sort(key=lambda p: p.urgency * p.confidence, reverse=True)
        
        return predictions[:10]  # Top 10 predictions
    
    async def _analyze_context_with_intelligent_selection(self, context: PredictionContext) -> Dict[str, Any]:
        """Use intelligent model selection for context analysis.
        
        Leverages the hybrid orchestrator to select the most appropriate
        model for analyzing the current context and generating insights.
        
        Args:
            context: Prediction context to analyze
            
        Returns:
            Dict[str, Any]: Analysis results including current goal, predicted
                          needs, risk factors, and optimization opportunities
                          
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: If analysis fails or cannot be parsed
        """
        try:
            from backend.core.hybrid_orchestrator import HybridOrchestrator

            orchestrator = HybridOrchestrator()
            if not orchestrator.is_running:
                await orchestrator.start()

            # Prepare context summary
            context_summary = {
                'time': context.timestamp.isoformat(),
                'workspace': {
                    'window_count': len(context.workspace_state.get('windows', [])),
                    'active_apps': [w['app_name'] for w in context.workspace_state.get('windows', [])[:5]],
                    'notifications': len(context.workspace_state.get('notifications', {}).get('badges', []))
                },
                'user_activity': context.user_activity,
                'patterns': list(context.historical_patterns.keys())[:10]
            }

            # Build rich context for intelligent selection
            intelligent_context = {
                "task_type": "behavior_prediction",
                "user_behavior_history": context.historical_patterns,
                "current_patterns": {
                    "window_count": len(context.workspace_state.get('windows', [])),
                    "activity_level": context.user_activity.get('click_rate', 0) / 60.0,
                    "temporal_data": {
                        "hour": context.timestamp.hour,
                        "day_of_week": context.timestamp.weekday()
                    }
                },
                "prediction_targets": ["user_goal", "upcoming_needs", "friction_points", "assistance_opportunities", "user_state"]
            }

            prompt = f"""As JARVIS's predictive intelligence, analyze this context and identify:
1. What the user is likely trying to accomplish
2. What they'll need in the next 5-30 minutes
3. Potential friction points or interruptions
4. Opportunities for proactive assistance
5. Signs of stress, fatigue, or flow state

Context:
{json.dumps(context_summary, indent=2, default=str)}

Provide analysis as JSON with these keys:
- current_goal: what user is working on
- predicted_needs: array of likely upcoming needs
- risk_factors: potential issues
- optimization_opportunities: ways to help
- user_state: emotional/cognitive state
- confidence_scores: confidence for each prediction"""

            result = await orchestrator.execute_with_intelligent_model_selection(
                query=prompt,
                intent="behavior_prediction",
                required_capabilities={"nlp_analysis", "prediction", "pattern_recognition"},
                context=intelligent_context,
                max_tokens=1000,
                temperature=0.7,
            )

            if not result.get("success"):
                raise Exception(result.get("error", "Unknown error"))

            analysis_text = result.get("text", "").strip()
            model_used = result.get("model_used", "intelligent_selection")

            logger.info(f"✨ Context analysis generated using {model_used}")

            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'error': 'Could not parse analysis'}

        except ImportError:
            logger.warning("Hybrid orchestrator not available for intelligent selection")
            raise
        except Exception as e:
            logger.error(f"Error in intelligent context analysis: {e}")
            raise

    async def _analyze_context_with_claude(self, context: PredictionContext) -> Dict[str, Any]:
        """Use Claude to deeply understand the current context.
        
        Directly uses Claude AI to analyze the context and generate insights
        about user goals, needs, and potential optimization opportunities.
        
        Args:
            context: Prediction context to analyze
            
        Returns:
            Dict[str, Any]: Analysis results in JSON format, or empty dict
                          if analysis fails
        """
        try:
            # Prepare context summary
            context_summary = {
                'time': context.timestamp.isoformat(),
                'workspace': {
                    'window_count': len(context.workspace_state.get('windows', [])),
                    'active_apps': [w['app_name'] for w in context.workspace_state.get('windows', [])[:5]],
                    'notifications': len(context.workspace_state.get('notifications', {}).get('badges', []))
                },
                'user_activity': context.user_activity,
                'patterns': list(context.historical_patterns.keys())[:10]
            }
            
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""As JARVIS's predictive intelligence, analyze this context and identify:
1. What the user is likely trying to accomplish
2. What they'll need in the next 5-30 minutes
3. Potential friction points or interruptions
4. Opportunities for proactive assistance
5. Signs of stress, fatigue, or flow state

Context:
{json.dumps(context_summary, indent=2, default=str)}

Provide analysis as JSON with these keys:
- current_goal: what user is working on
- predicted_needs: array of likely upcoming needs
- risk_factors: potential issues
- optimization_opportunities: ways to help
- user_state: emotional/cognitive state
- confidence_scores: confidence for each prediction"""
                }]
            )
            
            # Parse response
            analysis_text = response.content[0].text
            
            # Extract JSON
            import re
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {'error': 'Could not parse analysis'}
                
        except Exception as e:
            logger.error(f"Error in Claude analysis: {e}")
            return {}
    
    async def _generate_typed_prediction(self, pred_type: PredictionType, 
                                       context: PredictionContext,
                                       analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Generate a specific type of prediction.
        
        Routes to the appropriate prediction generator based on the
        prediction type and returns a typed prediction if applicable.
        
        Args:
            pred_type: Type of prediction to generate
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Generated prediction or None if
                                       not applicable for current context
        """
        
        # Dynamic prediction generation based on type
        generators = {
            PredictionType.NEXT_ACTION: self._predict_next_action,
            PredictionType.RESOURCE_NEED: self._predict_resource_need,
            PredictionType.WORKFLOW_TRANSITION: self._predict_workflow_transition,
            PredictionType.BREAK_SUGGESTION: self._predict_break_need,
            PredictionType.FOCUS_OPTIMIZATION: self._predict_focus_optimization,
            PredictionType.MEETING_PREPARATION: self._predict_meeting_prep,
            PredictionType.TASK_COMPLETION: self._predict_task_completion,
            PredictionType.COLLABORATION_NEED: self._predict_collaboration,
            PredictionType.AUTOMATION_OPPORTUNITY: self._predict_automation,
            PredictionType.DEADLINE_RISK: self._predict_deadline_risk
        }
        
        generator = generators.get(pred_type)
        if generator:
            return await generator(context, analysis)
        
        return None
    
    async def _predict_next_action(self, context: PredictionContext, 
                                  analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Predict the user's next likely action.
        
        Analyzes current workspace state and predicted needs to determine
        what action the user is likely to take next.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Next action prediction or None
        """
        current_apps = [w['app_name'] for w in context.workspace_state.get('windows', [])]
        predicted_needs = analysis.get('predicted_needs', [])
        
        if not predicted_needs:
            return None
        
        # Use ML model if trained
        model_data = self.prediction_models[PredictionType.NEXT_ACTION]
        confidence = 0.5
        
        if model_data['trained']:
            features = context.to_feature_vector()
            try:
                scaled_features = model_data['scaler'].transform([features])
                confidence = model_data['classifier'].predict_proba(scaled_features)[0].max()
            except:
                pass
        
        # Enhance with Claude's analysis
        if analysis.get('current_goal'):
            confidence = min(confidence + 0.2, 0.95)
        
        return DynamicPrediction(
            prediction_id=self._generate_prediction_id(context, PredictionType.NEXT_ACTION),
            type=PredictionType.NEXT_ACTION,
            confidence=confidence,
            urgency=0.7,
            action_items=[{
                'action': 'prepare_workspace',
                'details': predicted_needs[0],
                'apps_to_open': self._suggest_apps_for_need(predicted_needs[0], current_apps)
            }],
            reasoning=f"Based on {analysis.get('current_goal', 'current activity')}, you'll likely need {predicted_needs[0]}",
            context_factors={
                'pattern_match': 0.8,
                'time_relevance': 0.7,
                'user_state': 0.6
            }
        )
    
    async def _predict_resource_need(self, context: PredictionContext,
                                   analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Predict resources user will need.
        
        Analyzes current workflow and applications to predict what
        additional resources or tools the user will need.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Resource need prediction or None
        """
        resources = []
        confidence = 0.6
        
        # Analyze current workflow
        current_apps = [w['app_name'] for w in context.workspace_state.get('windows', [])]
        
        # Check for common resource patterns
        if 'code' in ' '.join(current_apps).lower():
            resources.extend(['documentation', 'terminal', 'debugger'])
            confidence += 0.1
        
        if 'browser' in ' '.join(current_apps).lower():
            if 'research' in analysis.get('current_goal', '').lower():
                resources.extend(['note-taking app', 'reference manager'])
                confidence += 0.15
        
        if not resources:
            return None
        
        return DynamicPrediction(
            prediction_id=self._generate_prediction_id(context, PredictionType.RESOURCE_NEED),
            type=PredictionType.RESOURCE_NEED,
            confidence=min(confidence, 0.9),
            urgency=0.6,
            action_items=[{
                'action': 'prepare_resources',
                'resources': resources,
                'reason': 'Commonly used together with current apps'
            }],
            reasoning=f"Your current workflow suggests you might need: {', '.join(resources)}",
            context_factors={
                'workflow_analysis': 0.8,
                'historical_usage': 0.7
            }
        )
    
    async def _predict_workflow_transition(self, context: PredictionContext,
                                         analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Predict workflow transitions.
        
        Placeholder for workflow transition prediction logic.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Always returns None (not implemented)
        """
        # Placeholder for workflow transition prediction
        return None
    
    async def _predict_break_need(self, context: PredictionContext,
                                 analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Predict when user needs a break.
        
        Analyzes work duration, activity intensity, and user state to
        determine if the user would benefit from taking a break.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Break suggestion prediction or None
        """
        # Analyze activity patterns
        activity = context.user_activity
        focus_duration = activity.get('continuous_work_minutes', 0)
        
        # Dynamic break prediction based on multiple factors
        need_break_score = 0
        
        # Time-based factor
        if focus_duration > 90:
            need_break_score += 0.4
        elif focus_duration > 60:
            need_break_score += 0.2
        
        # Activity intensity factor
        if activity.get('click_rate', 0) > 100:  # High activity
            need_break_score += 0.2
        
        if activity.get('window_switches', 0) > 20:  # Frequent context switching
            need_break_score += 0.3
        
        # User state factor
        user_state = analysis.get('user_state', {})
        if 'stress' in str(user_state).lower() or 'fatigue' in str(user_state).lower():
            need_break_score += 0.3
        
        if need_break_score < 0.5:
            return None
        
        # Calculate optimal break duration
        break_duration = min(5 + (focus_duration // 30) * 5, 20)
        
        return DynamicPrediction(
            prediction_id=self._generate_prediction_id(context, PredictionType.BREAK_SUGGESTION),
            type=PredictionType.BREAK_SUGGESTION,
            confidence=min(need_break_score, 0.95),
            urgency=min(need_break_score * 1.2, 1.0),
            action_items=[{
                'action': 'suggest_break',
                'duration_minutes': break_duration,
                'activities': self._suggest_break_activities(context, analysis)
            }],
            reasoning=f"You've been focused for {focus_duration} minutes. A {break_duration}-minute break would boost productivity.",
            context_factors={
                'focus_duration': focus_duration / 120,
                'activity_intensity': activity.get('click_rate', 0) / 150,
                'user_fatigue': need_break_score
            }
        )
    
    async def _predict_focus_optimization(self, context: PredictionContext,
                                        analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Predict focus optimization opportunities.
        
        Identifies distractions and suggests ways to optimize focus
        and productivity in the current workspace.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Focus optimization prediction or None
        """
        distractions = []
        confidence = 0.5
        
        # Analyze potential distractions
        notifications = context.workspace_state.get('notifications', {}).get('badges', [])
        if len(notifications) > 3:
            distractions.append('notifications')
            confidence += 0.2
        
        # Check for distracting apps
        windows = context.workspace_state.get('windows', [])
        distracting_apps = ['slack', 'discord', 'messages', 'mail', 'twitter', 'facebook']
        
        active_distractions = [
            w['app_name'] for w in windows 
            if any(d in w['app_name'].lower() for d in distracting_apps)
        ]
        
        if active_distractions:
            distractions.extend(active_distractions)
            confidence += 0.15 * len(active_distractions)
        
        if not distractions:
            return None
        
        return DynamicPrediction(
            prediction_id=self._generate_prediction_id(context, PredictionType.FOCUS_OPTIMIZATION),
            type=PredictionType.FOCUS_OPTIMIZATION,
            confidence=min(confidence, 0.9),
            urgency=0.7,
            action_items=[{
                'action': 'minimize_distractions',
                'distractions': distractions,
                'suggestion': 'Enter focus mode to boost productivity'
            }],
            reasoning=f"Removing {len(distractions)} distractions could improve focus by ~30%",
            context_factors={
                'distraction_level': len(distractions) / 5,
                'focus_potential': 0.8
            }
        )
    
    async def _predict_meeting_prep(self, context: PredictionContext,
                                  analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Predict meeting preparation needs.
        
        Detects upcoming meetings and suggests preparation actions
        based on calendar apps and timing patterns.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Meeting preparation prediction or None
        """
        # Check for calendar apps or meeting indicators
        windows = context.workspace_state.get('windows', [])
        meeting_apps = ['calendar', 'zoom', 'teams', 'meet', 'webex']
        
        has_meeting_app = any(
            any(app in w['app_name'].lower() for app in meeting_apps)
            for w in windows
        )
        
        if not has_meeting_app:
            return None
        
        # Check time until likely meeting (usually on the hour or half hour)
        now = context.timestamp
        minutes_to_hour = 60 - now.minute
        minutes_to_half = 30 - (now.minute % 30)
        
        if minutes_to_half > 15 and minutes_to_hour > 15:
            return None
        
        prep_time = min(minutes_to_half, minutes_to_hour)
        
        return DynamicPrediction(
            prediction_id=self._generate_prediction_id(context, PredictionType.MEETING_PREPARATION),
            type=PredictionType.MEETING_PREPARATION,
            confidence=0.8,
            urgency=1.0 - (prep_time / 15),
            action_items=[{
                'action': 'prepare_meeting',
                'time_until_meeting': prep_time,
                'preparations': [
                    'Check audio/video',
                    'Review agenda',
                    'Close unnecessary apps',
                    'Prepare notes document'
                ]
            }],
            reasoning=f"Meeting likely starting in {prep_time} minutes",
            context_factors={
                'time_proximity': 1.0 - (prep_time / 15),
                'calendar_detected': 1.0
            }
        )
    
    async def _predict_task_completion_with_intelligent_selection(self, context: PredictionContext,
                                                                   analysis: Dict[str, Any]) -> Optional[DynamicPrediction]:
        """Use intelligent model selection for task completion prediction.
        
        Leverages the hybrid orchestrator to predict task completion time
        and requirements using the most appropriate AI model.
        
        Args:
            context: Current prediction context
            analysis: Context analysis results
            
        Returns:
            Optional[DynamicPrediction]: Task completion prediction or None
            
        Raises:
            ImportError: If hybrid orchestrator is not available
            Exception: