#!/usr/bin/env python3
"""
Adaptive Learning System for Detection Patterns
Learns from user interactions to improve query detection and response accuracy
"""

import json
import os
import pickle
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class QueryPattern:
    """Represents a learned query pattern"""
    query_text: str
    intent: str
    keywords: List[str]
    success_rate: float
    usage_count: int
    last_used: datetime
    user_feedback: float  # -1 to 1 scale
    variations: List[str]
    context_features: Dict[str, Any]

@dataclass
class LearningRecord:
    """Record of a learning event"""
    timestamp: datetime
    query: str
    detected_intent: str
    actual_intent: str
    success: bool
    response_quality: float
    user_correction: Optional[str]
    context: Dict[str, Any]

class AdaptiveDetectionLearning:
    """
    Adaptive learning system that improves detection patterns over time
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path or "/tmp/jarvis_adaptive_model.pkl"
        self.patterns = {}
        self.learning_records = []
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.pattern_embeddings = {}
        self.confidence_threshold = 0.6
        self._init_base_patterns()
        self._load_model()
        logger.info("Adaptive Detection Learning initialized")

    def _init_base_patterns(self):
        """Initialize with base detection patterns"""
        self.base_patterns = {
            'multi_space_query': {
                'keywords': [
                    'across', 'desktop', 'spaces', 'all', 'every',
                    'multiple', 'what\'s happening', 'workspace',
                    'show me', 'tell me about'
                ],
                'variations': [
                    "what's happening across my desktop spaces",
                    "show me all my spaces",
                    "what am i doing in each space",
                    "analyze my workspace",
                    "what's on my desktop",
                    "tell me about my workspace"
                ],
                'confidence_boost': 0.2
            },
            'single_space_query': {
                'keywords': [
                    'current', 'this', 'here', 'now', 'active',
                    'focused', 'present'
                ],
                'variations': [
                    "what's on my screen",
                    "analyze current window",
                    "what am i looking at"
                ],
                'confidence_boost': 0.1
            },
            'application_query': {
                'keywords': [
                    'app', 'application', 'program', 'running',
                    'open', 'using'
                ],
                'variations': [
                    "what apps are open",
                    "which programs are running",
                    "show me running applications"
                ],
                'confidence_boost': 0.15
            }
        }

    def learn_from_interaction(
        self,
        query: str,
        detected_intent: str,
        success: bool,
        user_feedback: Optional[str] = None,
        context: Dict[str, Any] = None
    ) -> None:
        """
        Learn from a user interaction
        """
        # Create learning record
        record = LearningRecord(
            timestamp=datetime.now(),
            query=query.lower(),
            detected_intent=detected_intent,
            actual_intent=user_feedback if user_feedback else detected_intent,
            success=success,
            response_quality=1.0 if success else 0.0,
            user_correction=user_feedback,
            context=context or {}
        )

        self.learning_records.append(record)

        # Update patterns
        self._update_patterns(record)

        # Retrain if needed
        if len(self.learning_records) % 10 == 0:
            self._retrain_detection_model()

        # Save model periodically
        if len(self.learning_records) % 50 == 0:
            self._save_model()

    def _update_patterns(self, record: LearningRecord):
        """Update pattern database based on learning record"""
        query_key = self._generate_pattern_key(record.query)

        if query_key not in self.patterns:
            # Create new pattern
            self.patterns[query_key] = QueryPattern(
                query_text=record.query,
                intent=record.actual_intent,
                keywords=self._extract_keywords(record.query),
                success_rate=1.0 if record.success else 0.0,
                usage_count=1,
                last_used=record.timestamp,
                user_feedback=record.response_quality,
                variations=[record.query],
                context_features=record.context
            )
        else:
            # Update existing pattern
            pattern = self.patterns[query_key]
            pattern.usage_count += 1
            pattern.success_rate = (
                (pattern.success_rate * (pattern.usage_count - 1) +
                 (1.0 if record.success else 0.0)) / pattern.usage_count
            )
            pattern.last_used = record.timestamp
            pattern.user_feedback = (
                (pattern.user_feedback * (pattern.usage_count - 1) +
                 record.response_quality) / pattern.usage_count
            )

            # Add variation if significantly different
            if record.query not in pattern.variations:
                pattern.variations.append(record.query)

    def detect_intent_adaptive(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Tuple[str, float]:
        """
        Detect intent using adaptive learning
        """
        query_lower = query.lower()

        # Check learned patterns first
        learned_intent = self._check_learned_patterns(query_lower)
        if learned_intent[1] > self.confidence_threshold:
            return learned_intent

        # Check base patterns
        base_intent = self._check_base_patterns(query_lower)
        if base_intent[1] > self.confidence_threshold:
            return base_intent

        # Use similarity matching
        similar_intent = self._find_similar_pattern(query_lower)
        if similar_intent[1] > self.confidence_threshold:
            return similar_intent

        # Default fallback
        return self._intelligent_fallback(query_lower, context)

    def _check_learned_patterns(self, query: str) -> Tuple[str, float]:
        """Check against learned patterns"""
        best_match = None
        best_score = 0.0

        for pattern_key, pattern in self.patterns.items():
            # Direct match
            if query in pattern.variations:
                return (pattern.intent, pattern.success_rate)

            # Keyword matching
            keyword_score = self._calculate_keyword_score(query, pattern.keywords)
            adjusted_score = keyword_score * pattern.success_rate

            if adjusted_score > best_score:
                best_score = adjusted_score
                best_match = pattern.intent

        return (best_match, best_score) if best_match else ('unknown', 0.0)

    def _check_base_patterns(self, query: str) -> Tuple[str, float]:
        """Check against base patterns"""
        scores = {}

        for intent, pattern_data in self.base_patterns.items():
            score = 0.0

            # Check keywords
            for keyword in pattern_data['keywords']:
                if keyword in query:
                    score += 0.1

            # Check variations
            for variation in pattern_data['variations']:
                similarity = self._calculate_string_similarity(query, variation.lower())
                score = max(score, similarity)

            # Apply confidence boost
            score += pattern_data['confidence_boost']
            scores[intent] = min(score, 1.0)

        if scores:
            best_intent = max(scores, key=scores.get)
            return (best_intent, scores[best_intent])

        return ('unknown', 0.0)

    def _find_similar_pattern(self, query: str) -> Tuple[str, float]:
        """Find similar patterns using vector similarity"""
        if not self.pattern_embeddings:
            self._build_pattern_embeddings()

        try:
            # Vectorize query
            query_vector = self.vectorizer.transform([query])

            # Calculate similarities
            similarities = {}
            for pattern_key, pattern_vector in self.pattern_embeddings.items():
                similarity = cosine_similarity(query_vector, pattern_vector)[0][0]
                pattern = self.patterns.get(pattern_key)
                if pattern:
                    # Weight by success rate
                    weighted_similarity = similarity * pattern.success_rate
                    similarities[pattern.intent] = max(
                        similarities.get(pattern.intent, 0),
                        weighted_similarity
                    )

            if similarities:
                best_intent = max(similarities, key=similarities.get)
                return (best_intent, similarities[best_intent])

        except Exception as e:
            logger.warning(f"Similarity matching failed: {e}")

        return ('unknown', 0.0)

    def _intelligent_fallback(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Tuple[str, float]:
        """Intelligent fallback when no pattern matches"""
        # Enhanced keyword detection
        multi_space_keywords = ['across', 'desktop space', 'all space', 'multiple']
        single_space_keywords = ['current', 'this', 'here', 'now']

        multi_score = sum(1 for kw in multi_space_keywords if kw in query) / len(multi_space_keywords)
        single_score = sum(1 for kw in single_space_keywords if kw in query) / len(single_space_keywords)

        # Context-based adjustment
        if context:
            if context.get('previous_intent') == 'multi_space_query':
                multi_score += 0.2
            if context.get('space_count', 1) > 1:
                multi_score += 0.1

        if multi_score > single_score:
            return ('multi_space_query', min(multi_score + 0.3, 0.9))
        elif single_score > 0:
            return ('single_space_query', min(single_score + 0.3, 0.8))

        # Check for "what's happening" specifically
        if "what's happening" in query or "what is happening" in query:
            return ('multi_space_query', 0.75)

        return ('general_query', 0.5)

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        stop_words = {'the', 'a', 'an', 'is', 'are', 'in', 'on', 'at', 'to', 'for'}
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        return keywords

    def _calculate_keyword_score(self, query: str, keywords: List[str]) -> float:
        """Calculate keyword matching score"""
        if not keywords:
            return 0.0

        matches = sum(1 for kw in keywords if kw in query)
        return matches / len(keywords)

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using character overlap"""
        set1 = set(str1.split())
        set2 = set(str2.split())

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _generate_pattern_key(self, query: str) -> str:
        """Generate a key for pattern storage"""
        # Use first few words as key
        words = query.lower().split()[:3]
        return '_'.join(words)

    def _build_pattern_embeddings(self):
        """Build vector embeddings for patterns"""
        if not self.patterns:
            return

        pattern_texts = []
        pattern_keys = []

        for key, pattern in self.patterns.items():
            # Combine query and variations
            text = ' '.join([pattern.query_text] + pattern.variations)
            pattern_texts.append(text)
            pattern_keys.append(key)

        if pattern_texts:
            # Fit vectorizer
            vectors = self.vectorizer.fit_transform(pattern_texts)

            # Store embeddings
            for i, key in enumerate(pattern_keys):
                self.pattern_embeddings[key] = vectors[i]

    def _retrain_detection_model(self):
        """Retrain the detection model with accumulated data"""
        if len(self.learning_records) < 10:
            return

        # Rebuild pattern embeddings
        self._build_pattern_embeddings()

        # Adjust confidence threshold based on success rate
        success_rate = sum(r.success for r in self.learning_records[-50:]) / min(50, len(self.learning_records))
        if success_rate < 0.7:
            self.confidence_threshold = max(0.5, self.confidence_threshold - 0.05)
        elif success_rate > 0.9:
            self.confidence_threshold = min(0.8, self.confidence_threshold + 0.05)

        logger.info(f"Model retrained. Success rate: {success_rate:.2f}, Threshold: {self.confidence_threshold:.2f}")

    def get_pattern_suggestions(self, partial_query: str) -> List[str]:
        """Get query suggestions based on learned patterns"""
        suggestions = []

        # Check learned patterns
        for pattern in self.patterns.values():
            for variation in pattern.variations:
                if partial_query.lower() in variation.lower():
                    suggestions.append(variation)

        # Check base patterns
        for pattern_data in self.base_patterns.values():
            for variation in pattern_data['variations']:
                if partial_query.lower() in variation.lower():
                    suggestions.append(variation)

        return list(set(suggestions))[:5]

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics of the adaptive system"""
        if not self.learning_records:
            return {
                'total_interactions': 0,
                'success_rate': 0.0,
                'pattern_count': 0,
                'confidence_threshold': self.confidence_threshold
            }

        recent_records = self.learning_records[-100:]
        success_count = sum(r.success for r in recent_records)

        intent_accuracy = defaultdict(list)
        for record in recent_records:
            intent_accuracy[record.detected_intent].append(record.success)

        intent_stats = {}
        for intent, successes in intent_accuracy.items():
            intent_stats[intent] = {
                'accuracy': sum(successes) / len(successes) if successes else 0,
                'count': len(successes)
            }

        return {
            'total_interactions': len(self.learning_records),
            'recent_success_rate': success_count / len(recent_records),
            'pattern_count': len(self.patterns),
            'confidence_threshold': self.confidence_threshold,
            'intent_performance': intent_stats,
            'top_patterns': self._get_top_patterns()
        }

    def _get_top_patterns(self, n: int = 5) -> List[Dict]:
        """Get top performing patterns"""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.success_rate * p.usage_count,
            reverse=True
        )

        return [
            {
                'query': p.query_text,
                'intent': p.intent,
                'success_rate': p.success_rate,
                'usage_count': p.usage_count
            }
            for p in sorted_patterns[:n]
        ]

    def _save_model(self):
        """Save the adaptive model to disk"""
        try:
            model_data = {
                'patterns': self.patterns,
                'learning_records': self.learning_records[-1000:],  # Keep last 1000
                'confidence_threshold': self.confidence_threshold,
                'pattern_embeddings': self.pattern_embeddings
            }

            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def _load_model(self):
        """Load the adaptive model from disk"""
        if not os.path.exists(self.model_path):
            logger.info("No existing model found, starting fresh")
            return

        try:
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)

            self.patterns = model_data.get('patterns', {})
            self.learning_records = model_data.get('learning_records', [])
            self.confidence_threshold = model_data.get('confidence_threshold', 0.6)
            self.pattern_embeddings = model_data.get('pattern_embeddings', {})

            logger.info(f"Model loaded from {self.model_path}")
            logger.info(f"Loaded {len(self.patterns)} patterns, {len(self.learning_records)} records")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")

    def export_patterns(self, output_path: str):
        """Export learned patterns for analysis"""
        export_data = {
            'patterns': [asdict(p) for p in self.patterns.values()],
            'base_patterns': self.base_patterns,
            'metrics': self.get_performance_metrics()
        }

        # Convert datetime objects to strings
        for pattern in export_data['patterns']:
            pattern['last_used'] = pattern['last_used'].isoformat()

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Patterns exported to {output_path}")