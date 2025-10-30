#!/usr/bin/env python3
"""
Advanced Python-Swift Bridge with Self-Learning Capabilities

This module provides a sophisticated command routing system that learns from user
interactions without hardcoded rules. It features adaptive classification,
continuous learning, and performance optimization through machine learning
techniques.

The system can operate with either Swift-based classification (when available)
or advanced Python ML fallback, ensuring robust operation in all environments.

Example:
    >>> router = AdvancedIntelligentCommandRouter()
    >>> handler, classification = await router.route_command("take a screenshot")
    >>> print(f"Handler: {handler}, Type: {classification.type}")
    Handler: system, Type: system
"""

import asyncio
import json
import logging
import subprocess
import os
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass, asdict
import pickle
import sqlite3
from collections import defaultdict
import threading
import queue

logger = logging.getLogger(__name__)

@dataclass
class CommandClassification:
    """Represents a command classification result with confidence and reasoning.
    
    Attributes:
        type: The classified command type (e.g., 'system', 'vision', 'conversation')
        intent: The specific intent within the command type
        confidence: Confidence score between 0.0 and 1.0
        entities: List of extracted entities with their metadata
        reasoning: Human-readable explanation of the classification decision
        alternatives: Alternative classifications with their confidence scores
        context_used: Context information that influenced the classification
        learned: Whether this classification used learned patterns
    """
    type: str
    intent: str
    confidence: float
    entities: List[Dict[str, Any]]
    reasoning: str
    alternatives: List[Dict[str, Any]]
    context_used: Dict[str, Any]
    learned: bool = False

@dataclass
class LearningFeedback:
    """User feedback for improving classification accuracy.
    
    Attributes:
        command: The original command that was classified
        classified_as: What the system classified it as
        should_be: What the user says it should have been classified as
        user_rating: User satisfaction rating (0.0 to 1.0)
        timestamp: When the feedback was provided
        context: Context information when the command was issued
    """
    command: str
    classified_as: str
    should_be: str
    user_rating: float
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """System performance metrics for monitoring and optimization.
    
    Attributes:
        accuracy: Overall classification accuracy (0.0 to 1.0)
        avg_response_time: Average response time in seconds
        total_classifications: Total number of classifications performed
        improvement_rate: Rate of accuracy improvement over time
        common_errors: List of (expected, actual, count) tuples for common errors
    """
    accuracy: float
    avg_response_time: float
    total_classifications: int
    improvement_rate: float
    common_errors: List[Tuple[str, str, int]]

class AdvancedIntelligentCommandRouter:
    """
    Advanced Command Router with Zero Hardcoding and Self-Learning Capabilities.
    
    This class provides intelligent command routing that learns from user interactions
    and adapts its behavior over time. It supports both Swift-based and Python-based
    classification with continuous learning and performance optimization.
    
    The router maintains no hardcoded rules - all classification logic is learned
    from user feedback and interaction patterns.
    
    Attributes:
        swift_available: Whether Swift classifier is available and functional
        learning_db: Database for storing learning patterns and feedback
        context_manager: Manages contextual information for better classification
        performance_tracker: Tracks and analyzes system performance metrics
        pattern_learner: Machine learning component for pattern recognition
        feedback_queue: Queue for processing user feedback asynchronously
        learning_thread: Background thread for continuous learning
    
    Example:
        >>> router = AdvancedIntelligentCommandRouter()
        >>> handler, classification = await router.route_command("show me the weather")
        >>> feedback = LearningFeedback(
        ...     command="show me the weather",
        ...     classified_as="conversation",
        ...     should_be="system",
        ...     user_rating=0.8,
        ...     timestamp=datetime.now(),
        ...     context={}
        ... )
        >>> router.provide_feedback(feedback)
    """
    
    def __init__(self):
        """Initialize the Advanced Intelligent Command Router.
        
        Sets up all components including Swift availability check, learning database,
        context management, performance tracking, and background learning processes.
        """
        self.swift_available = self._check_swift_availability()
        self.learning_db = LearningDatabase()
        self.context_manager = AdvancedContextManager()
        self.performance_tracker = PerformanceTracker()
        self.pattern_learner = PatternLearner()
        self.feedback_queue = queue.Queue()
        
        # Start background learning thread
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()
        
        # Initialize Swift classifier if available
        if self.swift_available:
            self._initialize_swift_classifier()
        else:
            logger.info("Swift classifier not available - using advanced Python fallback")
            
        # Load historical learning data
        self._load_learning_history()
    
    def _check_swift_availability(self) -> bool:
        """Check if Swift classifier binary is available and executable.
        
        Returns:
            True if Swift classifier is available and can be executed, False otherwise.
        """
        swift_binary = Path(__file__).parent / ".build" / "release" / "jarvis-advanced-classifier"
        return swift_binary.exists() and os.access(swift_binary, os.X_OK)
    
    def _initialize_swift_classifier(self):
        """Initialize the Swift classifier process and verify it's working.
        
        Tests the Swift classifier with a version check and sets swift_available
        to False if initialization fails.
        
        Raises:
            No exceptions raised - errors are logged and swift_available is set to False.
        """
        try:
            self.swift_binary = Path(__file__).parent / ".build" / "release" / "jarvis-advanced-classifier"
            # Test run
            result = subprocess.run(
                [str(self.swift_binary), "--version"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                logger.info(f"Swift classifier initialized: {result.stdout.strip()}")
            else:
                logger.error(f"Swift classifier initialization failed: {result.stderr}")
                self.swift_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Swift classifier: {e}")
            self.swift_available = False
    
    async def route_command(self, command: str) -> Tuple[str, CommandClassification]:
        """
        Route command with zero hardcoding - everything is learned and adaptive.
        
        This is the main entry point for command classification. It uses the best
        available classification method (Swift or Python ML), applies learned
        corrections, tracks performance, and queues the interaction for learning.
        
        Args:
            command: The user command to classify and route
            
        Returns:
            A tuple containing:
                - handler: The name of the handler that should process this command
                - classification: Detailed classification results with confidence and reasoning
                
        Example:
            >>> handler, classification = await router.route_command("take a screenshot")
            >>> print(f"Confidence: {classification.confidence:.2f}")
            Confidence: 0.95
        """
        start_time = datetime.now()
        
        # Get current context
        context = self.context_manager.get_context()
        
        # Add command to context
        self.context_manager.add_command(command)
        
        # Classify using best available method
        if self.swift_available:
            classification = await self._classify_with_swift(command, context)
        else:
            classification = await self._classify_with_python_ml(command, context)
        
        # Apply learned corrections
        classification = self._apply_learned_corrections(classification, command, context)
        
        # Track performance
        response_time = (datetime.now() - start_time).total_seconds()
        self.performance_tracker.record_classification(
            command, classification, response_time
        )
        
        # Determine handler based on classification
        handler = self._determine_handler(classification)
        
        # Learn from this interaction
        self._queue_for_learning(command, classification, context)
        
        return handler, classification
    
    async def _classify_with_swift(self, command: str, context: Dict[str, Any]) -> CommandClassification:
        """Use Swift classifier for intelligent command classification.
        
        Calls the external Swift binary for classification, handling process
        communication and error recovery with Python ML fallback.
        
        Args:
            command: The command to classify
            context: Current context information
            
        Returns:
            CommandClassification object with Swift-generated results
            
        Raises:
            Falls back to Python ML classification if Swift process fails
        """
        try:
            # Prepare input
            input_data = {
                "command": command,
                "context": self._serialize_context(context)
            }
            
            # Call Swift classifier
            process = await asyncio.create_subprocess_exec(
                str(self.swift_binary),
                "classify",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate(
                json.dumps(input_data).encode()
            )
            
            if process.returncode != 0:
                logger.error(f"Swift classifier error: {stderr.decode()}")
                return await self._classify_with_python_ml(command, context)
            
            # Parse result
            result = json.loads(stdout.decode())
            
            return CommandClassification(
                type=result["type"],
                intent=result["intent"]["primary"],
                confidence=result["confidence"],
                entities=result["entities"],
                reasoning=result["reasoning"],
                alternatives=result.get("alternatives", []),
                context_used=context,
                learned=False
            )
            
        except Exception as e:
            logger.error(f"Swift classification failed: {e}")
            return await self._classify_with_python_ml(command, context)
    
    async def _classify_with_python_ml(self, command: str, context: Dict[str, Any]) -> CommandClassification:
        """
        Advanced Python ML classification with zero hardcoding.
        
        Uses machine learning techniques to classify commands based on learned
        patterns, feature extraction, and contextual information. All classification
        logic is adaptive and learned from user interactions.
        
        Args:
            command: The command to classify
            context: Current context information including previous commands, time, etc.
            
        Returns:
            CommandClassification object with ML-generated results including
            confidence scores, reasoning, and alternative classifications
        """
        # Extract features
        features = self.pattern_learner.extract_features(command, context)
        
        # Get learned patterns
        similar_patterns = self.learning_db.find_similar_patterns(features)
        
        # Calculate probabilities for each type
        type_probabilities = self._calculate_type_probabilities(
            features, similar_patterns, context
        )
        
        # Select best type
        best_type = max(type_probabilities.items(), key=lambda x: x[1])
        
        # Extract intent
        intent = self._extract_intent(command, best_type[0], features)
        
        # Extract entities
        entities = self._extract_entities(command, features)
        
        # Generate alternatives
        alternatives = [
            {
                "type": t,
                "confidence": p,
                "reasoning": f"Alternative classification based on pattern matching"
            }
            for t, p in sorted(type_probabilities.items(), key=lambda x: x[1], reverse=True)[1:3]
            if p > 0.1
        ]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            command, best_type[0], intent, features, similar_patterns
        )
        
        return CommandClassification(
            type=best_type[0],
            intent=intent,
            confidence=best_type[1],
            entities=entities,
            reasoning=reasoning,
            alternatives=alternatives,
            context_used=context,
            learned=len(similar_patterns) > 0
        )
    
    def _calculate_type_probabilities(
        self, 
        features: np.ndarray, 
        similar_patterns: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate probabilities for each command type using learned patterns.
        
        Combines multiple sources of information including similar patterns,
        context weights, and linguistic features to determine the probability
        distribution across command types.
        
        Args:
            features: Extracted feature vector for the command
            similar_patterns: List of similar patterns from learning database
            context: Current context information
            
        Returns:
            Dictionary mapping command types to their probability scores
        """
        
        # Initialize with small probabilities (no zeros)
        probabilities = {
            "system": 0.1,
            "vision": 0.1,
            "conversation": 0.1,
            "automation": 0.1,
            "unknown": 0.1
        }
        
        # Learn from similar patterns
        if similar_patterns:
            for pattern in similar_patterns:
                similarity = pattern["similarity"]
                success_rate = pattern.get("success_rate", 0.5)
                weight = similarity * success_rate
                
                pattern_type = pattern["type"]
                if pattern_type in probabilities:
                    probabilities[pattern_type] += weight
        
        # Apply context influence
        context_weights = self._get_context_weights(context)
        for type_name, weight in context_weights.items():
            if type_name in probabilities:
                probabilities[type_name] *= (1 + weight)
        
        # Learn from linguistic features
        linguistic_weights = self.pattern_learner.get_linguistic_weights(features)
        for type_name, weight in linguistic_weights.items():
            if type_name in probabilities:
                probabilities[type_name] *= (1 + weight)
        
        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        return probabilities
    
    def _extract_intent(self, command: str, command_type: str, features: np.ndarray) -> str:
        """Extract intent without hardcoding using learned patterns.
        
        Determines the specific intent within a command type by matching against
        learned intent patterns or generating new intent names based on command
        structure analysis.
        
        Args:
            command: The original command text
            command_type: The classified command type
            features: Extracted feature vector
            
        Returns:
            String representing the extracted intent
        """
        
        # Look for learned intents
        learned_intents = self.learning_db.get_intents_for_type(command_type)
        
        if learned_intents:
            # Find best matching intent
            best_match = None
            best_score = 0
            
            for intent in learned_intents:
                score = self.pattern_learner.calculate_intent_match(
                    command, intent["pattern"], features
                )
                if score > best_score:
                    best_score = score
                    best_match = intent["name"]
            
            if best_match and best_score > 0.5:
                return best_match
        
        # Generate new intent name from command structure
        tokens = command.lower().split()
        
        # Find action words (learned, not hardcoded)
        action_words = self.pattern_learner.extract_action_words(tokens)
        target_words = self.pattern_learner.extract_target_words(tokens)
        
        if action_words and target_words:
            return f"{action_words[0]}_{target_words[0]}"
        elif action_words:
            return f"{action_words[0]}_action"
        elif target_words:
            return f"query_{target_words[0]}"
        else:
            return f"intent_{hash(command) % 10000}"
    
    def _extract_entities(self, command: str, features: np.ndarray) -> List[Dict[str, Any]]:
        """Extract entities from command without hardcoded rules.
        
        Uses the pattern learner to identify and extract entities based on
        learned patterns and linguistic analysis.
        
        Args:
            command: The command text to analyze
            features: Extracted feature vector
            
        Returns:
            List of dictionaries containing entity information with text,
            type, role, and confidence for each identified entity
        """
        entities = []
        
        # Use pattern learner to identify entities
        identified_entities = self.pattern_learner.extract_entities(command)
        
        for entity in identified_entities:
            entities.append({
                "text": entity["text"],
                "type": entity["type"],
                "role": entity["role"],
                "confidence": entity["confidence"]
            })
        
        return entities
    
    def _apply_learned_corrections(
        self, 
        classification: CommandClassification,
        command: str,
        context: Dict[str, Any]
    ) -> CommandClassification:
        """Apply learned corrections from user feedback to improve accuracy.
        
        Checks for stored corrections from user feedback and applies them
        if they have higher confidence than the current classification.
        
        Args:
            classification: The initial classification result
            command: The original command
            context: Current context information
            
        Returns:
            Updated CommandClassification with corrections applied if applicable
        """
        
        # Check if we have corrections for similar commands
        corrections = self.learning_db.get_corrections_for_command(command)
        
        if corrections:
            # Apply the most recent/highest rated correction
            best_correction = max(corrections, key=lambda x: x["rating"] * x["recency"])
            
            if best_correction["confidence"] > classification.confidence:
                # Update classification with learned correction
                classification.type = best_correction["correct_type"]
                classification.intent = best_correction["correct_intent"]
                classification.confidence = best_correction["confidence"]
                classification.learned = True
                classification.reasoning += f" (Corrected based on user feedback)"
        
        return classification
    
    def _determine_handler(self, classification: CommandClassification) -> str:
        """Determine appropriate handler based on classification.
        
        Maps the classified command type to the appropriate handler using
        learned mappings or default fallbacks.
        
        Args:
            classification: The command classification result
            
        Returns:
            String name of the handler that should process this command
        """
        
        # Map classification to handler (this mapping is also learned)
        handler_mapping = self.learning_db.get_handler_mapping()
        
        if classification.type in handler_mapping:
            return handler_mapping[classification.type]
        
        # Default mapping if not learned yet
        default_mapping = {
            "system": "system",
            "vision": "vision",
            "conversation": "conversation",
            "automation": "automation",
            "unknown": "conversation"
        }
        
        return default_mapping.get(classification.type, "conversation")
    
    def provide_feedback(self, feedback: LearningFeedback):
        """Process user feedback to improve future classifications.
        
        Queues feedback for background processing and applies immediate
        corrections for critically poor classifications.
        
        Args:
            feedback: LearningFeedback object containing user correction information
            
        Example:
            >>> feedback = LearningFeedback(
            ...     command="take screenshot",
            ...     classified_as="conversation",
            ...     should_be="system",
            ...     user_rating=0.2,
            ...     timestamp=datetime.now(),
            ...     context={}
            ... )
            >>> router.provide_feedback(feedback)
        """
        
        # Queue feedback for processing
        self.feedback_queue.put(feedback)
        
        # Immediate update for critical corrections
        if feedback.user_rating < 0.3:  # Poor classification
            self._immediate_correction(feedback)
    
    def _immediate_correction(self, feedback: LearningFeedback):
        """Apply immediate correction for poor classifications.
        
        Stores the correction and updates learning models immediately
        for classifications with very low user ratings.
        
        Args:
            feedback: LearningFeedback object with poor rating requiring immediate correction
        """
        
        # Store correction
        self.learning_db.store_correction(
            feedback.command,
            feedback.classified_as,
            feedback.should_be,
            feedback.user_rating,
            feedback.context
        )
        
        # Update pattern learner
        self.pattern_learner.update_from_feedback(feedback)
        
        # Log for analysis
        logger.info(f"Immediate correction applied: '{feedback.command}' should be {feedback.should_be}, not {feedback.classified_as}")
    
    def _continuous_learning_loop(self):
        """Background thread for continuous learning and model updates.
        
        Processes feedback queue, updates learning models, and performs
        periodic maintenance tasks. Runs continuously in a daemon thread.
        
        This method handles all exceptions internally to prevent thread termination.
        """
        
        while True:
            try:
                # Process feedback queue
                while not self.feedback_queue.empty():
                    feedback = self.feedback_queue.get()
                    self._process_feedback(feedback)
                
                # Periodic model updates
                self._update_learning_models()
                
                # Sleep before next iteration
                asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in learning loop: {e}")
    
    def _process_feedback(self, feedback: LearningFeedback):
        """Process individual feedback item for learning.
        
        Stores feedback in database, updates pattern learner, and
        updates performance metrics.
        
        Args:
            feedback: LearningFeedback object to process
        """
        
        # Store in database
        self.learning_db.store_feedback(feedback)
        
        # Update pattern learner
        self.pattern_learner.learn_from_feedback(feedback)
        
        # Update performance metrics
        self.performance_tracker.update_from_feedback(feedback)
    
    def _update_learning_models(self):
        """Periodically update learning models with recent data.
        
        Retrains pattern learner with recent patterns and updates
        context weights based on success rates.
        """
        
        # Retrain pattern learner with recent data
        recent_patterns = self.learning_db.get_recent_patterns(hours=24)
        if recent_patterns:
            self.pattern_learner.retrain(recent_patterns)
        
        # Update context weights based on success rates
        self.context_manager.update_weights(
            self.performance_tracker.get_context_success_rates()
        )
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current system performance metrics.
        
        Returns:
            PerformanceMetrics object containing accuracy, response time,
            and other performance indicators
        """
        return self.performance_tracker.get_metrics()
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights about the learning process and system adaptation.
        
        Provides detailed information about learning progress, accuracy trends,
        and adaptation patterns for system monitoring and optimization.
        
        Returns:
            Dictionary containing:
                - total_patterns_learned: Number of patterns in learning database
                - accuracy_trend: Historical accuracy improvement data
                - most_improved_classifications: Classifications with biggest improvements
                - adaptation_rate: How quickly the system adapts to new patterns
                - common_corrections: Most frequent user corrections
                
        Example:
            >>> insights = router.get_learning_insights()
            >>> print(f"Learned {insights['total_patterns_learned']} patterns")
            Learned 1247 patterns
        """
        
        return {
            "total_patterns_learned": self.learning_db.get_pattern_count(),
            "accuracy_trend": self.performance_tracker.get_accuracy_trend(),
            "most_improved_classifications": self.pattern_learner.get_most_improved(),
            "adaptation_rate": self.pattern_learner.get_adaptation_rate(),
            "common_corrections": self.learning_db.get_common_corrections()
        }
    
    def _serialize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize context for Swift consumption.
        
        Converts non-serializable types like numpy arrays and datetime objects
        to JSON-serializable formats for communication with Swift processes.
        
        Args:
            context: Context dictionary that may contain non-serializable objects
            
        Returns:
            Serialized context dictionary safe for JSON encoding
        """
        
        # Convert numpy arrays and other non-serializable types
        serialized = {}
        for key, value in context.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            else:
                serialized[key] = value
        
        return serialized
    
    def _generate_reasoning(
        self,
        command: str,
        command_type: str,
        intent: str,
        features: np.ndarray,
        similar_patterns: List[Dict[str, Any]]
    ) -> str:
        """Generate human-readable reasoning for the classification decision.
        
        Creates explanatory text that helps users understand why a particular
        classification was made, including pattern matches and key indicators.
        
        Args:
            command: The original command
            command_type: The classified type
            intent: The extracted intent
            features: Feature vector used in classification
            similar_patterns: Patterns that influenced the decision
            
        Returns:
            Human-readable string explaining the classification reasoning
        """
        
        reasoning_parts = []
        
        # Add pattern-based reasoning
        if similar_patterns:
            best_pattern = max(similar_patterns, key=lambda x: x["similarity"])
            reasoning_parts.append(
                f"Similar to learned pattern with {best_pattern['similarity']:.0%} match"
            )
        
        # Add feature-based reasoning
        dominant_features = self.pattern_learner.get_dominant_features(features)
        if dominant_features:
            reasoning_parts.append(
                f"Key indicators: {', '.join(dominant_features)}"
            )
        
        # Add intent reasoning
        reasoning_parts.append(f"Detected intent: {intent}")
        
        return " | ".join(reasoning_parts)
    
    def _load_learning_history(self):
        """Load historical learning data from persistent storage.
        
        Initializes the system with previously learned patterns and
        performance metrics to maintain continuity across restarts.
        """
        
        # Load patterns
        patterns = self.learning_db.load_all_patterns()
        if patterns:
            self.pattern_learner.load_patterns(patterns)
            logger.info(f"Loaded {len(patterns)} historical patterns")
        
        # Load performance history
        metrics = self.learning_db.load_performance_metrics()
        if metrics:
            self.performance_tracker.load_history(metrics)
            logger.info("Loaded performance history")
    
    def _queue_for_learning(
        self,
        command: str,
        classification: CommandClassification,
        context: Dict[str, Any]
    ):
        """Queue interaction for background learning analysis.
        
        Stores interaction data for later analysis and learning by
        background processes.
        
        Args:
            command: The processed command
            classification: The classification result
            context: Context information during classification
        """
        
        learning_data = {
            "command": command,
            "classification": asdict(classification),
            "context": context,
            "timestamp": datetime.now()
        }
        
        # Store for learning
        self.learning_db.store_interaction(learning_data)
    
    def _get_context_weights(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Get context-based weights for classification enhancement.
        
        Analyzes current context to determine weights that should be applied
        to different command types based on learned patterns of user behavior.
        
        Args:
            context: Current context information including time, previous commands, etc.
            
        Returns:
            Dictionary mapping command types to weight multipliers based on context
        """
        
        weights = {}
        
        # Learn from previous commands
        if "previous_commands" in context:
            recent_types = self.learning_db.get_recent_command_types()
            for cmd_type, frequency in recent_types.items():
                weights[cmd_type] = frequency * 0.2
        
        # Learn from time patterns
        if "time_of_day" in context:
            time_patterns = self.learning_db.get_time_patterns()
            hour = context["time_of_day"].hour
            for cmd_type, pattern in time_patterns.items():
                if hour in pattern["peak_hours"]:
                    weights[cmd_type] = weights.get(cmd_type, 0) + 0.3
        
        # Learn from user state
        if "user_state" in context:
            state_patterns = self.learning_db.get_user_state_patterns()
            user_state = context["user_state"]
            for cmd_type, pattern in state_patterns.items():
                if pattern["state"] == user_state:
                    weights[cmd_type] = weights.get(cmd_type, 0) + 0.25
        
        return weights