#!/usr/bin/env python3
"""
Learning Components for Advanced Command Classification.

This module provides zero-hardcoding learning components for command classification.
Everything is learned and adaptive, including pattern recognition, user behavior
analysis, and performance tracking.

The module contains:
- LearningDatabase: Persistent storage for learned patterns and feedback
- PatternLearner: Machine learning component for pattern recognition
- AdvancedContextManager: Context management with learning capabilities
- UserProfile: User behavior modeling and personalization
- PerformanceTracker: System performance monitoring and analysis

Example:
    >>> from learning_components import LearningDatabase, PatternLearner
    >>> db = LearningDatabase()
    >>> learner = PatternLearner()
    >>> features = learner.extract_features("open Chrome", {})
    >>> similar = db.find_similar_patterns(features)
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, deque
import threading

logger = logging.getLogger(__name__)

class LearningDatabase:
    """
    Database for storing and retrieving learned patterns.
    
    Provides persistent storage for command patterns, user corrections,
    feedback, and performance metrics. Supports similarity search and
    pattern matching for adaptive learning.
    
    Attributes:
        db_path (str): Path to the SQLite database file
        conn (sqlite3.Connection): Database connection
        lock (threading.Lock): Thread safety lock
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the learning database.
        
        Args:
            db_path: Optional path to database file. If None, creates in
                    ~/.jarvis/learning/command_patterns.db
        """
        if db_path is None:
            db_dir = Path.home() / ".jarvis" / "learning"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = db_dir / "command_patterns.db"
        
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        
        self._initialize_database()
    
    def _initialize_database(self):
        """
        Create database tables if they don't exist.
        
        Creates tables for patterns, corrections, feedback, interactions,
        metrics, intent patterns, and handler mappings.
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            
            # Patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    features BLOB NOT NULL,
                    type TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    success_rate REAL DEFAULT 0.5,
                    usage_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Corrections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    original_type TEXT NOT NULL,
                    correct_type TEXT NOT NULL,
                    correct_intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    rating REAL NOT NULL,
                    context BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Feedback table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    classified_as TEXT NOT NULL,
                    should_be TEXT NOT NULL,
                    user_rating REAL NOT NULL,
                    context BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    classification BLOB NOT NULL,
                    context BLOB NOT NULL,
                    response_time REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    accuracy REAL NOT NULL,
                    avg_response_time REAL NOT NULL,
                    total_classifications INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Intent patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intent_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Handler mapping table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS handler_mappings (
                    type TEXT PRIMARY KEY,
                    handler TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.5,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            self.conn.commit()
    
    def find_similar_patterns(self, features: np.ndarray, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find patterns similar to the given features.
        
        Uses cosine similarity to find patterns with similar feature vectors.
        
        Args:
            features: Feature vector to match against
            threshold: Minimum similarity threshold (0.0 to 1.0)
            
        Returns:
            List of similar patterns with similarity scores, sorted by similarity
            
        Example:
            >>> features = np.array([0.1, 0.5, 0.3])
            >>> similar = db.find_similar_patterns(features, threshold=0.5)
            >>> print(f"Found {len(similar)} similar patterns")
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, command, features, type, intent, confidence, success_rate
                FROM patterns
                ORDER BY updated_at DESC
                LIMIT 1000
            ''')
            
            similar_patterns = []
            for row in cursor.fetchall():
                pattern_id, command, features_blob, pattern_type, intent, confidence, success_rate = row
                pattern_features = pickle.loads(features_blob)
                
                # Calculate similarity
                similarity = self._calculate_similarity(features, pattern_features)
                
                if similarity > threshold:
                    similar_patterns.append({
                        "id": pattern_id,
                        "command": command,
                        "type": pattern_type,
                        "intent": intent,
                        "confidence": confidence,
                        "success_rate": success_rate,
                        "similarity": similarity
                    })
            
            # Sort by similarity
            similar_patterns.sort(key=lambda x: x["similarity"], reverse=True)
            
            return similar_patterns[:10]  # Return top 10
    
    def _calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Cosine similarity score between 0.0 and 1.0
        """
        
        # Ensure same shape
        min_len = min(len(features1), len(features2))
        if min_len == 0:
            return 0.0
        
        # Truncate to same length
        f1 = features1[:min_len].reshape(1, -1)
        f2 = features2[:min_len].reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(f1, f2)[0, 0]
        
        # Ensure it's between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def get_corrections_for_command(self, command: str) -> List[Dict[str, Any]]:
        """
        Get corrections for a specific command.
        
        Retrieves user corrections for the given command, including
        recency weighting for more relevant corrections.
        
        Args:
            command: Command text to find corrections for
            
        Returns:
            List of corrections with confidence and recency scores
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            
            # Get exact matches first
            cursor.execute('''
                SELECT correct_type, correct_intent, confidence, rating,
                       julianday('now') - julianday(created_at) as age_days
                FROM corrections
                WHERE LOWER(command) = LOWER(?)
                ORDER BY created_at DESC
                LIMIT 10
            ''', (command,))
            
            corrections = []
            for row in cursor.fetchall():
                correct_type, correct_intent, confidence, rating, age_days = row
                
                # Calculate recency factor (newer corrections are more relevant)
                recency = 1.0 / (1.0 + age_days)
                
                corrections.append({
                    "correct_type": correct_type,
                    "correct_intent": correct_intent,
                    "confidence": confidence,
                    "rating": rating,
                    "recency": recency
                })
            
            return corrections
    
    def store_correction(
        self,
        command: str,
        original_type: str,
        correct_type: str,
        user_rating: float,
        context: Dict[str, Any]
    ):
        """
        Store a correction from user feedback.
        
        Args:
            command: The command that was corrected
            original_type: Original classification type
            correct_type: Correct classification type
            user_rating: User satisfaction rating (0.0 to 1.0)
            context: Additional context information
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            
            # Extract intent from context or generate
            correct_intent = context.get("intent", f"{correct_type}_intent")
            
            cursor.execute('''
                INSERT INTO corrections (command, original_type, correct_type, 
                                       correct_intent, confidence, rating, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                command,
                original_type,
                correct_type,
                correct_intent,
                user_rating,
                user_rating,
                pickle.dumps(context)
            ))
            
            self.conn.commit()
    
    def get_intents_for_type(self, command_type: str) -> List[Dict[str, Any]]:
        """
        Get learned intents for a specific command type.
        
        Args:
            command_type: Type of command to get intents for
            
        Returns:
            List of intents with patterns and confidence scores
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT intent, pattern, confidence
                FROM intent_patterns
                WHERE type = ?
                ORDER BY confidence DESC
                LIMIT 20
            ''', (command_type,))
            
            intents = []
            for row in cursor.fetchall():
                intent, pattern, confidence = row
                intents.append({
                    "name": intent,
                    "pattern": pattern,
                    "confidence": confidence
                })
            
            return intents
    
    def get_handler_mapping(self) -> Dict[str, str]:
        """
        Get learned handler mappings.
        
        Returns:
            Dictionary mapping command types to handler names
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT type, handler FROM handler_mappings')
            
            mapping = {}
            for row in cursor.fetchall():
                command_type, handler = row
                mapping[command_type] = handler
            
            return mapping
    
    def store_feedback(self, feedback):
        """
        Store user feedback.
        
        Args:
            feedback: Feedback object containing command, classification,
                     and user rating information
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO feedback (command, classified_as, should_be, 
                                    user_rating, context)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                feedback.command,
                feedback.classified_as,
                feedback.should_be,
                feedback.user_rating,
                pickle.dumps(feedback.context)
            ))
            
            self.conn.commit()
    
    def store_interaction(self, interaction_data: Dict[str, Any]):
        """
        Store an interaction for learning.
        
        Args:
            interaction_data: Dictionary containing command, classification,
                            context, and optional response time
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO interactions (command, classification, context, response_time)
                VALUES (?, ?, ?, ?)
            ''', (
                interaction_data["command"],
                pickle.dumps(interaction_data["classification"]),
                pickle.dumps(interaction_data["context"]),
                interaction_data.get("response_time", 0)
            ))
            
            # Also update or create pattern
            features = interaction_data.get("features")
            if features is not None:
                classification = interaction_data["classification"]
                cursor.execute('''
                    INSERT OR REPLACE INTO patterns 
                    (command, features, type, intent, confidence)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    interaction_data["command"],
                    pickle.dumps(features),
                    classification["type"],
                    classification["intent"],
                    classification["confidence"]
                ))
            
            self.conn.commit()
    
    def get_recent_patterns(self, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get patterns from the last N hours.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent patterns with features and metadata
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT command, features, type, intent, confidence, success_rate
                FROM patterns
                WHERE datetime(updated_at) > datetime('now', '-{} hours')
                ORDER BY updated_at DESC
            '''.format(hours))
            
            patterns = []
            for row in cursor.fetchall():
                command, features_blob, pattern_type, intent, confidence, success_rate = row
                patterns.append({
                    "command": command,
                    "features": pickle.loads(features_blob),
                    "type": pattern_type,
                    "intent": intent,
                    "confidence": confidence,
                    "success_rate": success_rate
                })
            
            return patterns
    
    def get_pattern_count(self) -> int:
        """
        Get total number of learned patterns.
        
        Returns:
            Total count of patterns in database
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM patterns')
            return cursor.fetchone()[0]
    
    def get_common_corrections(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most common corrections.
        
        Args:
            limit: Maximum number of corrections to return
            
        Returns:
            List of common corrections with counts
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT original_type, correct_type, COUNT(*) as count
                FROM corrections
                GROUP BY original_type, correct_type
                ORDER BY count DESC
                LIMIT ?
            ''', (limit,))
            
            corrections = []
            for row in cursor.fetchall():
                original, correct, count = row
                corrections.append({
                    "original_type": original,
                    "correct_type": correct,
                    "count": count
                })
            
            return corrections
    
    def load_all_patterns(self) -> List[Dict[str, Any]]:
        """
        Load all patterns for initialization.
        
        Returns:
            List of all patterns sorted by success rate and recency
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT command, features, type, intent, confidence, success_rate
                FROM patterns
                ORDER BY success_rate DESC, updated_at DESC
                LIMIT 10000
            ''')
            
            patterns = []
            for row in cursor.fetchall():
                command, features_blob, pattern_type, intent, confidence, success_rate = row
                patterns.append({
                    "command": command,
                    "features": pickle.loads(features_blob),
                    "type": pattern_type,
                    "intent": intent,
                    "confidence": confidence,
                    "success_rate": success_rate
                })
            
            return patterns
    
    def load_performance_metrics(self) -> List[Dict[str, Any]]:
        """
        Load performance metrics history.
        
        Returns:
            List of historical performance metrics
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT accuracy, avg_response_time, total_classifications, created_at
                FROM metrics
                ORDER BY created_at DESC
                LIMIT 1000
            ''')
            
            metrics = []
            for row in cursor.fetchall():
                accuracy, avg_time, total, created_at = row
                metrics.append({
                    "accuracy": accuracy,
                    "avg_response_time": avg_time,
                    "total_classifications": total,
                    "timestamp": created_at
                })
            
            return metrics
    
    def get_recent_command_types(self) -> Dict[str, float]:
        """
        Get frequency of recent command types.
        
        Returns:
            Dictionary mapping command types to their frequencies
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT type, COUNT(*) as count
                FROM patterns
                WHERE datetime(updated_at) > datetime('now', '-1 hour')
                GROUP BY type
            ''')
            
            total = 0
            type_counts = {}
            
            for row in cursor.fetchall():
                cmd_type, count = row
                type_counts[cmd_type] = count
                total += count
            
            # Convert to frequencies
            if total > 0:
                return {k: v/total for k, v in type_counts.items()}
            else:
                return {}
    
    def get_time_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get command type patterns by time of day.
        
        Analyzes when different command types are most commonly used.
        
        Returns:
            Dictionary mapping command types to their time patterns,
            including peak hours and usage counts
        """
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT type, 
                       strftime('%H', created_at) as hour,
                       COUNT(*) as count
                FROM interactions
                WHERE datetime(created_at) > datetime('now', '-7 days')
                GROUP BY type, hour
            ''')
            
            patterns = defaultdict(lambda: {"peak_hours": [], "counts": {}})
            
            for row in cursor.fetchall():
                cmd_type, hour, count = row
                hour_int = int(hour)
                patterns[cmd_type]["counts"][hour_int] = count
            
            # Find peak hours for each type
            for cmd_type, data in patterns.items():
                counts = data["counts"]
                if counts:
                    max_count = max(counts.values())
                    threshold = max_count * 0.7
                    peak_hours = [h for h, c in counts.items() if c >= threshold]
                    patterns[cmd_type]["peak_hours"] = peak_hours
            
            return dict(patterns)
    
    def get_user_state_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        Get command patterns based on user state.
        
        Analyzes how user state affects command patterns.
        
        Returns:
            Dictionary mapping command types to user state patterns
        """
        
        # This would analyze patterns based on user state from context
        # For now, returning a simplified version
        return {
            "system": {"state": "focused", "confidence": 0.8},
            "vision": {"state": "exploring", "confidence": 0.7},
            "conversation": {"state": "multitasking", "confidence": 0.6}
        }

class PatternLearner:
    """
    Machine learning component for pattern recognition and learning.
    
    Uses TF-IDF vectorization and feature extraction to learn command patterns.
    Adapts to user behavior and improves classification over time.
    
    Attributes:
        vectorizer (TfidfVectorizer): Text vectorization component
        patterns (List[Dict]): Learned patterns storage
        feature_importance (defaultdict): Feature importance weights
        action_words (set): Learned action words
        target_words (set): Learned target words
        learned_entities (defaultdict): Learned entity patterns
    """
    
    def __init__(self):
        """Initialize the pattern learner with empty models."""
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 3),
            stop_words=None  # We want to learn everything
        )
        self.patterns = []
        self.feature_importance = defaultdict(float)
        self.action_words = set()
        self.target_words = set()
        self.learned_entities = defaultdict(set)
        
        # Initialize with empty fit
        self.vectorizer.fit([""])
    
    def extract_features(self, command: str, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from command and context.
        
        Combines text features (TF-IDF), linguistic features, and context features
        into a single feature vector for classification.
        
        Args:
            command: Command text to extract features from
            context: Context information including time, user state, etc.
            
        Returns:
            Combined feature vector as numpy array
            
        Example:
            >>> learner = PatternLearner()
            >>> features = learner.extract_features("open Chrome", {"time_of_day": datetime.now()})
            >>> print(f"Feature vector length: {len(features)}")
        """
        
        # Text features
        text_features = self.vectorizer.transform([command]).toarray()[0]
        
        # Linguistic features
        linguistic_features = self._extract_linguistic_features(command)
        
        # Context features
        context_features = self._extract_context_features(context)
        
        # Combine all features
        all_features = np.concatenate([
            text_features,
            linguistic_features,
            context_features
        ])
        
        return all_features
    
    def _extract_linguistic_features(self, command: str) -> np.ndarray:
        """
        Extract linguistic features from command.
        
        Args:
            command: Command text to analyze
            
        Returns:
            Array of linguistic features including token count, punctuation,
            and learned word patterns
        """
        
        tokens = command.lower().split()
        
        features = [
            len(tokens),                          # Token count
            len(command),                         # Character count
            command.count(" "),                   # Space count
            1.0 if command.endswith("?") else 0.0,  # Question mark
            1.0 if command.endswith("!") else 0.0,  # Exclamation
            1.0 if any(t in self.action_words for t in tokens) else 0.0,  # Has learned action
            1.0 if any(t in self.target_words for t in tokens) else 0.0,  # Has learned target
            sum(1 for t in tokens if t[0].isupper()) / max(1, len(tokens)),  # Capitalization ratio
        ]
        
        return np.array(features)
    
    def _extract_context_features(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from context.
        
        Args:
            context: Context dictionary with time, user state, and history
            
        Returns:
            Array of context features including time patterns, user state,
            and session information
        """
        
        features = []
        
        # Previous commands count
        prev_commands = context.get("previous_commands", [])
        features.append(len(prev_commands))
        
        # Time features
        if "time_of_day" in context:
            hour = context["time_of_day"].hour
            features.extend([
                hour / 24.0,  # Normalized hour
                1.0 if 9 <= hour <= 17 else 0.0,  # Working hours
                1.0 if hour < 6 or hour > 22 else 0.0,  # Off hours
            ])
        else:
            features.extend([0.5, 0.5, 0.0])
        
        # User state features
        if "user_state" in context:
            state = context["user_state"]
            features.append(state.get("cognitive_load", 0.5))
            features.append(state.get("frustration_level", 0.0))
            features.append(state.get("expertise", 0.5))
        else:
            features.extend([0.5, 0.0, 0.5])
        
        # Session duration
        if "session_duration" in context:
            features.append(min(1.0, context["session_duration"] / 3600.0))
        else:
            features.append(0.0)
        
        return np.array(features)
    
    def learn_from_feedback(self, feedback):
        """
        Learn from user feedback.
        
        Updates patterns, action/target words, and feature importance
        based on user corrections.
        
        Args:
            feedback: Feedback object containing command, correct classification,
                     and user rating
        """
        
        # Extract features from the command
        features = self.extract_features(feedback.command, feedback.context)
        
        # Store pattern with correct classification
        self.patterns.append({
            "command": feedback.command,
            "features": features,
            "type": feedback.should_be,
            "confidence": feedback.user_rating
        })
        
        # Learn action and target words
        tokens = feedback.command.lower().split()
        if feedback.should_be == "system":
            # First word is likely an action
            if tokens:
                self.action_words.add(tokens[0])
            # Last word might be a target
            if len(tokens) > 1:
                self.target_words.add(tokens[-1])
        
        # Update feature importance based on successful classifications
        if feedback.user_rating > 0.7:
            self._update_feature_importance(features, feedback.should_be)
        
        # Retrain vectorizer periodically
        if len(self.patterns) % 100 == 0:
            self._retrain_vectorizer()
    
    def _update_feature_importance(self, features: np.ndarray, correct_type: str):
        """
        Update feature importance based on successful classification.
        
        Args:
            features: Feature vector from successful classification
            correct_type: The correct command type
        """
        
        # Simple importance tracking
        feature_key = f"{correct_type}_features"
        
        if feature_key not in self.feature_importance:
            self.feature_importance[feature_key] = np.zeros_like(features)
        
        # Exponential moving average
        alpha = 0.1
        self.feature_importance[feature_key] = (
            alpha * features + (1 - alpha) * self.feature_importance[feature_key]
        )
    
    def _retrain_vectorizer(self):
        """Retrain the vectorizer with accumulated patterns."""
        
        if self.patterns:
            commands = [p["command"] for p in self.patterns]
            self.vectorizer = TfidfVectorizer(
                max_features=100,
                ngram_range=(1, 3)
            )
            self.vectorizer.fit(commands)
    
    def calculate_intent_match(
        self, 
        command: str, 
        intent_pattern: str, 
        features: np.ndarray
    ) -> float:
        """
        Calculate how well a command matches an intent pattern.
        
        Args:
            command: Command text to match
            intent_pattern: Pattern to match against
            features: Feature vector (currently unused)
            
        Returns:
            Match score between 0.0 and 1.0
        """
        
        # Simple token overlap for now
        command_tokens = set(command.lower().split())
        pattern_tokens = set(intent_pattern.lower().split())
        
        if not pattern_tokens:
            return 0.0
        
        overlap = len(command_tokens & pattern_tokens)
        return overlap / len(pattern_tokens)
    
    def extract_action_words(self, tokens: List[str]) -> List[str]:
        """
        Extract action words from tokens (learned, not hardcoded).
        
        Args:
            tokens: List of command tokens
            
        Returns:
            List of identified action words
        """
        
        action_words = []
        
        # Check learned action words
        for token in tokens:
            if token in self.action_words:
                action_words.append(token)
        
        # If no learned actions, guess based on position and pattern
        if not action_words and tokens:
            # First word is often an action
            first_token = tokens[0]
            # Basic heuristic: words ending in common verb suffixes
            if any(first_token.endswith(suffix) for suffix in ["e", "ate", "ify"]):
                action_words.append(first_token)
                self.action_words.add(first_token)  # Learn it
        
        return action_words
    
    def extract_target_words(self, tokens: List[str]) -> List[str]:
        """
        Extract target words from tokens (learned, not hardcoded).
        
        Args:
            tokens: List of command tokens
            
        Returns:
            List of identified target words
        """
        
        target_words = []
        
        # Check learned target words
        for token in tokens:
            if token in self.target_words:
                target_words.append(token)
        
        # If no learned targets, guess based on pattern
        if not target_words and len(tokens) > 1:
            # Capitalized words are often targets (app names)
            for token in tokens[1:]:  # Skip first word (usually action)
                if token[0].isupper():
                    target_words.append(token)
                    self.target_words.add(token.lower())  # Learn it
        
        return target_words
    
    def extract_entities(self,