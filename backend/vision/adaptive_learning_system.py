"""
Adaptive Learning System for JARVIS Query Classification
Learns from user feedback and improves classification accuracy over time.

Zero hardcoded patterns - learns entirely from usage.
"""

import asyncio
import logging
import sqlite3
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from .intelligent_query_classifier import QueryIntent, ClassificationResult

logger = logging.getLogger(__name__)


@dataclass
class FeedbackRecord:
    """Record of classification feedback"""
    query: str
    classified_intent: str  # What we classified it as
    actual_intent: str  # What it should have been
    confidence: float
    reasoning: str
    user_satisfied: bool  # Implicit/explicit feedback
    response_latency_ms: float
    timestamp: datetime
    context: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['context'] = json.dumps(self.context)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeedbackRecord':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['context'] = json.loads(data['context']) if isinstance(data['context'], str) else data['context']
        return cls(**data)


class AdaptiveLearningSystem:
    """
    Learns from user interactions to improve classification accuracy.
    Implements implicit and explicit feedback loops.
    """

    def __init__(self, db_path: str = None):
        """
        Initialize adaptive learning system

        Args:
            db_path: Path to SQLite database for persistent storage
        """
        # Setup database
        if db_path is None:
            db_dir = Path.home() / ".jarvis" / "vision"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "classification_feedback.db")

        self.db_path = db_path
        self._init_database()

        # In-memory tracking
        self._recent_classifications: List[Tuple[str, ClassificationResult]] = []
        self._query_history: List[str] = []

        # Learning metrics
        self._accuracy_history: List[Tuple[datetime, float]] = []
        self._misclassification_patterns: Dict[str, int] = {}

        logger.info(f"[LEARNING] Adaptive learning system initialized (db: {db_path})")

    def _init_database(self):
        """Initialize SQLite database for feedback storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    classified_intent TEXT NOT NULL,
                    actual_intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    user_satisfied INTEGER NOT NULL,
                    response_latency_ms REAL,
                    timestamp TEXT NOT NULL,
                    context TEXT
                )
            """)

            # Accuracy tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    accuracy REAL NOT NULL,
                    total_queries INTEGER NOT NULL,
                    correct_classifications INTEGER NOT NULL
                )
            """)

            # Pattern learning table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    occurrence_count INTEGER DEFAULT 1,
                    last_seen TEXT NOT NULL
                )
            """)

            conn.commit()
            conn.close()

            logger.info("[LEARNING] Database initialized successfully")

        except Exception as e:
            logger.error(f"[LEARNING] Database initialization failed: {e}")

    async def record_classification(
        self,
        query: str,
        result: ClassificationResult,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record a classification for potential feedback later

        Args:
            query: User's query
            result: Classification result
            context: Optional context information
        """
        self._recent_classifications.append((query, result))
        self._query_history.append(query)

        # Keep only recent history (last 100)
        if len(self._recent_classifications) > 100:
            self._recent_classifications = self._recent_classifications[-100:]
        if len(self._query_history) > 100:
            self._query_history = self._query_history[-100:]

    async def record_feedback(
        self,
        query: str,
        classified_intent: QueryIntent,
        actual_intent: QueryIntent,
        confidence: float,
        reasoning: str,
        user_satisfied: bool,
        response_latency_ms: float = 0,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record feedback on a classification

        Args:
            query: User's query
            classified_intent: What we classified it as
            actual_intent: What it should have been (from user behavior)
            confidence: Classification confidence
            reasoning: Classification reasoning
            user_satisfied: Whether user was satisfied with result
            response_latency_ms: Time to respond
            context: Additional context
        """
        record = FeedbackRecord(
            query=query,
            classified_intent=classified_intent.value,
            actual_intent=actual_intent.value,
            confidence=confidence,
            reasoning=reasoning,
            user_satisfied=user_satisfied,
            response_latency_ms=response_latency_ms,
            timestamp=datetime.now(),
            context=context or {}
        )

        # Store in database
        await self._store_feedback(record)

        # Track misclassification patterns
        if classified_intent != actual_intent:
            pattern = f"{classified_intent.value} -> {actual_intent.value}"
            self._misclassification_patterns[pattern] = (
                self._misclassification_patterns.get(pattern, 0) + 1
            )

        # Update accuracy metrics every 10 queries
        if len(self._query_history) % 10 == 0:
            await self._update_accuracy_metrics()

        logger.info(
            f"[LEARNING] Feedback recorded: {classified_intent.value} -> {actual_intent.value} "
            f"(satisfied: {user_satisfied}, confidence: {confidence:.2f})"
        )

    async def _store_feedback(self, record: FeedbackRecord):
        """Store feedback record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            data = record.to_dict()
            cursor.execute("""
                INSERT INTO feedback (
                    query, classified_intent, actual_intent, confidence,
                    reasoning, user_satisfied, response_latency_ms, timestamp, context
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['query'],
                data['classified_intent'],
                data['actual_intent'],
                data['confidence'],
                data['reasoning'],
                1 if data['user_satisfied'] else 0,
                data['response_latency_ms'],
                data['timestamp'],
                data['context']
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"[LEARNING] Failed to store feedback: {e}")

    async def detect_implicit_feedback(
        self,
        query: str,
        classified_intent: QueryIntent,
        user_action: str,
        response_latency_ms: float
    ) -> Optional[QueryIntent]:
        """
        Detect implicit feedback from user behavior

        Args:
            query: User's query
            classified_intent: How we classified it
            user_action: User's action after response (e.g., "accepted", "retry", "rephrase")
            response_latency_ms: Response time

        Returns:
            Actual intent if different from classification, None if correct
        """
        actual_intent = None

        # Implicit feedback signals
        if user_action == "accepted":
            # User accepted response without retry → classification was correct
            await self.record_feedback(
                query=query,
                classified_intent=classified_intent,
                actual_intent=classified_intent,  # Same = correct
                confidence=1.0,
                reasoning="User accepted response (implicit positive feedback)",
                user_satisfied=True,
                response_latency_ms=response_latency_ms
            )

        elif user_action == "retry" or user_action == "rephrase":
            # User retried or rephrased → likely misclassified
            # Infer actual intent from context
            actual_intent = await self._infer_actual_intent(query, classified_intent, user_action)

            if actual_intent and actual_intent != classified_intent:
                await self.record_feedback(
                    query=query,
                    classified_intent=classified_intent,
                    actual_intent=actual_intent,
                    confidence=0.7,  # Medium confidence on inference
                    reasoning=f"User {user_action} (implicit negative feedback)",
                    user_satisfied=False,
                    response_latency_ms=response_latency_ms
                )

        elif user_action == "followup":
            # User asked follow-up → neutral signal, classification likely correct
            await self.record_feedback(
                query=query,
                classified_intent=classified_intent,
                actual_intent=classified_intent,
                confidence=0.8,
                reasoning="User asked follow-up (implicit neutral feedback)",
                user_satisfied=True,
                response_latency_ms=response_latency_ms
            )

        return actual_intent

    async def _infer_actual_intent(
        self,
        query: str,
        classified_intent: QueryIntent,
        user_action: str
    ) -> Optional[QueryIntent]:
        """
        Infer what the actual intent should have been based on user behavior.
        Uses learned patterns and heuristics.
        """
        # Simple inference: If classified as METADATA_ONLY but user retried,
        # they probably wanted visual analysis
        if classified_intent == QueryIntent.METADATA_ONLY and user_action == "retry":
            return QueryIntent.VISUAL_ANALYSIS

        # If classified as VISUAL_ANALYSIS but user retried,
        # they might have wanted deeper analysis
        if classified_intent == QueryIntent.VISUAL_ANALYSIS and user_action == "retry":
            return QueryIntent.DEEP_ANALYSIS

        # If classified as DEEP_ANALYSIS but user retried,
        # maybe just visual was enough (too slow)
        if classified_intent == QueryIntent.DEEP_ANALYSIS and user_action == "retry":
            return QueryIntent.VISUAL_ANALYSIS

        return None

    async def _update_accuracy_metrics(self):
        """Update accuracy metrics based on recent feedback"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get feedback from last 100 queries
            cursor.execute("""
                SELECT classified_intent, actual_intent
                FROM feedback
                ORDER BY timestamp DESC
                LIMIT 100
            """)

            results = cursor.fetchall()
            if not results:
                conn.close()
                return

            # Calculate accuracy
            correct = sum(1 for c, a in results if c == a)
            total = len(results)
            accuracy = correct / total if total > 0 else 0

            # Store metrics
            cursor.execute("""
                INSERT INTO accuracy_metrics (timestamp, accuracy, total_queries, correct_classifications)
                VALUES (?, ?, ?, ?)
            """, (datetime.now().isoformat(), accuracy, total, correct))

            conn.commit()
            conn.close()

            # Track in memory
            self._accuracy_history.append((datetime.now(), accuracy))
            if len(self._accuracy_history) > 50:
                self._accuracy_history = self._accuracy_history[-50:]

            logger.info(f"[LEARNING] Accuracy updated: {accuracy:.2%} ({correct}/{total})")

        except Exception as e:
            logger.error(f"[LEARNING] Failed to update accuracy metrics: {e}")

    async def get_learned_patterns(self) -> List[Dict[str, Any]]:
        """Get learned patterns that can improve classification"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT pattern_type, pattern_data, intent, confidence, occurrence_count, last_seen
                FROM learned_patterns
                WHERE occurrence_count >= 3
                ORDER BY occurrence_count DESC
                LIMIT 50
            """)

            patterns = []
            for row in cursor.fetchall():
                patterns.append({
                    'pattern_type': row[0],
                    'pattern_data': json.loads(row[1]),
                    'intent': row[2],
                    'confidence': row[3],
                    'occurrence_count': row[4],
                    'last_seen': row[5]
                })

            conn.close()
            return patterns

        except Exception as e:
            logger.error(f"[LEARNING] Failed to get learned patterns: {e}")
            return []

    async def learn_from_feedback(self):
        """
        Analyze feedback to learn patterns and improve classification.
        Called periodically (e.g., every 100 queries)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find common misclassification patterns
            cursor.execute("""
                SELECT
                    query,
                    classified_intent,
                    actual_intent,
                    COUNT(*) as occurrence_count
                FROM feedback
                WHERE classified_intent != actual_intent
                GROUP BY query, classified_intent, actual_intent
                HAVING occurrence_count >= 2
                ORDER BY occurrence_count DESC
                LIMIT 20
            """)

            misclassifications = cursor.fetchall()

            # Store as learned patterns
            for query, classified, actual, count in misclassifications:
                # Extract pattern from query
                pattern_data = json.dumps({
                    'query_example': query,
                    'misclassified_as': classified,
                    'should_be': actual
                })

                cursor.execute("""
                    INSERT OR REPLACE INTO learned_patterns
                    (pattern_type, pattern_data, intent, confidence, occurrence_count, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    'misclassification',
                    pattern_data,
                    actual,
                    min(0.95, 0.7 + (count * 0.05)),  # Confidence increases with occurrences
                    count,
                    datetime.now().isoformat()
                ))

            conn.commit()
            conn.close()

            logger.info(f"[LEARNING] Learned from {len(misclassifications)} misclassification patterns")

        except Exception as e:
            logger.error(f"[LEARNING] Failed to learn from feedback: {e}")

    def get_accuracy_report(self) -> Dict[str, Any]:
        """Get accuracy and learning metrics report"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Overall accuracy
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN classified_intent = actual_intent THEN 1 ELSE 0 END) as correct
                FROM feedback
            """)
            total, correct = cursor.fetchone()
            overall_accuracy = (correct / total) if total > 0 else 0

            # Recent accuracy (last 100 queries)
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN classified_intent = actual_intent THEN 1 ELSE 0 END) as correct
                FROM (
                    SELECT * FROM feedback
                    ORDER BY timestamp DESC
                    LIMIT 100
                )
            """)
            recent_total, recent_correct = cursor.fetchone()
            recent_accuracy = (recent_correct / recent_total) if recent_total > 0 else 0

            # Accuracy by intent
            cursor.execute("""
                SELECT
                    actual_intent,
                    COUNT(*) as total,
                    SUM(CASE WHEN classified_intent = actual_intent THEN 1 ELSE 0 END) as correct
                FROM feedback
                GROUP BY actual_intent
            """)
            intent_accuracy = {}
            for intent, intent_total, intent_correct in cursor.fetchall():
                intent_accuracy[intent] = {
                    'accuracy': (intent_correct / intent_total) if intent_total > 0 else 0,
                    'total': intent_total,
                    'correct': intent_correct
                }

            # Most common misclassifications
            cursor.execute("""
                SELECT
                    classified_intent,
                    actual_intent,
                    COUNT(*) as count
                FROM feedback
                WHERE classified_intent != actual_intent
                GROUP BY classified_intent, actual_intent
                ORDER BY count DESC
                LIMIT 5
            """)
            misclassifications = [
                {'classified_as': c, 'should_be': a, 'count': cnt}
                for c, a, cnt in cursor.fetchall()
            ]

            conn.close()

            return {
                'overall_accuracy': overall_accuracy,
                'recent_accuracy': recent_accuracy,
                'total_queries': total or 0,
                'accuracy_by_intent': intent_accuracy,
                'common_misclassifications': misclassifications,
                'misclassification_patterns': dict(self._misclassification_patterns)
            }

        except Exception as e:
            logger.error(f"[LEARNING] Failed to generate accuracy report: {e}")
            return {
                'overall_accuracy': 0,
                'recent_accuracy': 0,
                'total_queries': 0,
                'accuracy_by_intent': {},
                'common_misclassifications': [],
                'error': str(e)
            }


# Singleton instance
_learning_system: Optional[AdaptiveLearningSystem] = None


def get_learning_system(db_path: str = None) -> AdaptiveLearningSystem:
    """Get or create the singleton learning system"""
    global _learning_system

    if _learning_system is None:
        _learning_system = AdaptiveLearningSystem(db_path)

    return _learning_system
