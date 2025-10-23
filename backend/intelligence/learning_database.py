#!/usr/bin/env python3
"""
Advanced Learning Database System for JARVIS Goal Inference
Hybrid architecture: SQLite (structured) + ChromaDB (embeddings) + Async + ML-powered insights
"""

import sqlite3
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict, field
import asyncio
import aiosqlite
from collections import defaultdict, deque
from enum import Enum
import time

# Async and ML dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ChromaDB for semantic search and embeddings
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logging.warning("ChromaDB not available - install with: pip install chromadb")

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Dynamic pattern types - extensible"""
    TEMPORAL = "temporal"  # Time-based patterns
    SEQUENTIAL = "sequential"  # Action sequences
    CONTEXTUAL = "contextual"  # Context-driven patterns
    HYBRID = "hybrid"  # Multi-factor patterns


class ConfidenceBoostStrategy(Enum):
    """Strategies for boosting pattern confidence"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    ADAPTIVE = "adaptive"  # Learns optimal strategy


@dataclass
class GoalPattern:
    """Represents a learned goal pattern with ML metadata"""
    pattern_id: str
    goal_type: str
    context_embedding: Optional[List[float]]
    action_sequence: List[str]
    confidence: float
    success_rate: float
    occurrence_count: int
    last_seen: datetime
    avg_execution_time: float = 0.0
    std_execution_time: float = 0.0
    decay_factor: float = 0.95  # For time-based decay
    boost_history: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Real-time learning performance metrics"""
    total_patterns: int
    active_patterns: int
    avg_confidence: float
    prediction_accuracy: float
    cache_hit_rate: float
    avg_inference_time_ms: float
    memory_usage_mb: float
    last_updated: datetime


class AdaptiveCache:
    """Smart LRU cache with TTL and size management"""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.access_count[key] += 1
                self.hits += 1
                return value
            else:
                del self.cache[key]

        self.misses += 1
        return None

    def set(self, key: str, value: Any):
        """Set value in cache with eviction"""
        # Evict if full
        if len(self.cache) >= self.max_size:
            # Remove least recently used
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[key] = (value, time.time())
        self.access_count[key] = 0

    def invalidate(self, key: str):
        """Remove from cache"""
        if key in self.cache:
            del self.cache[key]
            del self.access_count[key]

    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class PatternMatcher:
    """Advanced pattern matching with fuzzy matching and ML"""

    def __init__(self):
        self.embeddings_cache = AdaptiveCache(max_size=500, ttl_seconds=7200)

    async def compute_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """Compute similarity between patterns using multiple signals"""
        similarity_scores = []

        # Context similarity
        if 'context' in pattern1 and 'context' in pattern2:
            context_sim = self._jaccard_similarity(
                set(pattern1['context'].keys()),
                set(pattern2['context'].keys())
            )
            similarity_scores.append(context_sim * 0.3)

        # Action sequence similarity
        if 'actions' in pattern1 and 'actions' in pattern2:
            action_sim = self._sequence_similarity(
                pattern1['actions'],
                pattern2['actions']
            )
            similarity_scores.append(action_sim * 0.4)

        # Temporal similarity
        if 'timestamp' in pattern1 and 'timestamp' in pattern2:
            time_sim = self._temporal_similarity(
                pattern1['timestamp'],
                pattern2['timestamp']
            )
            similarity_scores.append(time_sim * 0.3)

        return sum(similarity_scores) if similarity_scores else 0.0

    @staticmethod
    def _jaccard_similarity(set1: Set, set2: Set) -> float:
        """Jaccard similarity coefficient"""
        if not set1 and not set2:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _sequence_similarity(seq1: List, seq2: List) -> float:
        """Levenshtein-based sequence similarity"""
        if not seq1 and not seq2:
            return 1.0

        # Simple edit distance
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        max_len = max(m, n)
        return 1.0 - (dp[m][n] / max_len) if max_len > 0 else 0.0

    @staticmethod
    def _temporal_similarity(time1: datetime, time2: datetime) -> float:
        """Temporal similarity based on time difference"""
        diff_seconds = abs((time1 - time2).total_seconds())

        # Decay similarity over time (1 hour window)
        if diff_seconds < 3600:
            return 1.0 - (diff_seconds / 3600)
        elif diff_seconds < 86400:  # 24 hours
            return 0.5 * (1.0 - (diff_seconds / 86400))
        else:
            return 0.0


class JARVISLearningDatabase:
    """
    Advanced hybrid database system for JARVIS learning
    - Async SQLite: Structured data with connection pooling
    - ChromaDB: Embeddings and semantic search
    - Adaptive caching: Smart LRU with TTL
    - ML-powered pattern matching and insights
    - Dynamic schema evolution
    - Real-time metrics and analytics
    """

    def __init__(self, db_path: Optional[Path] = None, config: Optional[Dict] = None):
        """Initialize the advanced learning database"""
        # Configuration with defaults
        self.config = config or {}
        self.cache_size = self.config.get('cache_size', 1000)
        self.cache_ttl = self.config.get('cache_ttl_seconds', 3600)
        self.enable_ml = self.config.get('enable_ml_features', True)
        self.auto_optimize = self.config.get('auto_optimize', True)
        self.batch_size = self.config.get('batch_insert_size', 100)

        # Set up paths
        self.db_dir = db_path or Path.home() / ".jarvis" / "learning"
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.sqlite_path = self.db_dir / "jarvis_learning.db"
        self.chroma_path = self.db_dir / "chroma_embeddings"

        # Async SQLite connection (will be initialized in async context)
        self.db: Optional[aiosqlite.Connection] = None
        self._db_lock = asyncio.Lock()

        # Adaptive caching
        self.pattern_cache = AdaptiveCache(self.cache_size, self.cache_ttl)
        self.goal_cache = AdaptiveCache(self.cache_size, self.cache_ttl)
        self.query_cache = AdaptiveCache(self.cache_size // 2, self.cache_ttl)

        # Pattern matching engine
        self.pattern_matcher = PatternMatcher()

        # Batch processing queues
        self.pending_goals: deque = deque(maxlen=self.batch_size)
        self.pending_actions: deque = deque(maxlen=self.batch_size)
        self.pending_patterns: deque = deque(maxlen=self.batch_size)

        # Performance metrics
        self.metrics = LearningMetrics(
            total_patterns=0,
            active_patterns=0,
            avg_confidence=0.0,
            prediction_accuracy=0.0,
            cache_hit_rate=0.0,
            avg_inference_time_ms=0.0,
            memory_usage_mb=0.0,
            last_updated=datetime.now()
        )

        # Initialize ChromaDB if available
        self.chroma_client = None
        self.goal_collection = None
        self.pattern_collection = None
        self.context_collection = None

        logger.info(f"Advanced JARVIS Learning Database initializing at {self.db_dir}")

    async def initialize(self):
        """Async initialization - call this after creating instance"""
        # Initialize async SQLite
        await self._init_sqlite()

        # Initialize ChromaDB
        if CHROMADB_AVAILABLE:
            await self._init_chromadb()

        # Load metrics
        await self._load_metrics()

        # Start background tasks
        asyncio.create_task(self._auto_flush_batches())
        asyncio.create_task(self._auto_optimize_task())

        logger.info(f"✅ Advanced Learning Database initialized")
        logger.info(f"   Cache: {self.cache_size} entries, {self.cache_ttl}s TTL")
        logger.info(f"   ML Features: {self.enable_ml}")
        logger.info(f"   Auto-optimize: {self.auto_optimize}")

    async def _init_sqlite(self):
        """Initialize async SQLite database with enhanced schema"""
        self.db = await aiosqlite.connect(str(self.sqlite_path))
        self.db.row_factory = aiosqlite.Row

        async with self.db.cursor() as cursor:
            # Enable WAL mode for better concurrency
            await cursor.execute("PRAGMA journal_mode=WAL")
            await cursor.execute("PRAGMA synchronous=NORMAL")
            await cursor.execute("PRAGMA cache_size=10000")
            await cursor.execute("PRAGMA temp_store=MEMORY")

            # Goals table with enhanced tracking
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    goal_type TEXT NOT NULL,
                    goal_level TEXT NOT NULL,
                    description TEXT,
                    confidence REAL,
                    progress REAL DEFAULT 0.0,
                    is_completed BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP,
                    completed_at TIMESTAMP,
                    predicted_duration REAL,
                    actual_duration REAL,
                    evidence JSON,
                    context_hash TEXT,
                    embedding_id TEXT,
                    metadata JSON
                )
            """)

            # Actions table with performance tracking
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS actions (
                    action_id TEXT PRIMARY KEY,
                    action_type TEXT NOT NULL,
                    target TEXT,
                    goal_id TEXT,
                    confidence REAL,
                    success BOOLEAN,
                    execution_time REAL,
                    timestamp TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    params JSON,
                    result JSON,
                    context_hash TEXT,
                    FOREIGN KEY (goal_id) REFERENCES goals(goal_id)
                )
            """)

            # Enhanced patterns table with ML metadata
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    pattern_hash TEXT UNIQUE,
                    pattern_data JSON,
                    confidence REAL,
                    success_rate REAL,
                    occurrence_count INTEGER DEFAULT 1,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    avg_execution_time REAL,
                    std_execution_time REAL,
                    decay_applied BOOLEAN DEFAULT 0,
                    boost_count INTEGER DEFAULT 0,
                    embedding_id TEXT,
                    metadata JSON
                )
            """)

            # User preferences with confidence tracking
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    preference_id TEXT PRIMARY KEY,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    confidence REAL,
                    learned_from TEXT,
                    update_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP,
                    updated_at TIMESTAMP,
                    UNIQUE(category, key)
                )
            """)

            # Goal-Action mappings with performance metrics
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS goal_action_mappings (
                    mapping_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal_type TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    avg_execution_time REAL,
                    std_execution_time REAL,
                    confidence REAL,
                    last_updated TIMESTAMP,
                    prediction_accuracy REAL,
                    UNIQUE(goal_type, action_type)
                )
            """)

            # Display patterns with temporal analysis
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS display_patterns (
                    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    display_name TEXT NOT NULL,
                    context JSON,
                    context_hash TEXT,
                    connection_time TIME,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    frequency INTEGER DEFAULT 1,
                    auto_connect BOOLEAN DEFAULT 0,
                    last_seen TIMESTAMP,
                    consecutive_successes INTEGER DEFAULT 0,
                    metadata JSON
                )
            """)

            # Learning metrics tracking
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_metrics (
                    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    timestamp TIMESTAMP,
                    context JSON
                )
            """)

            # Pattern similarity cache
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_similarity_cache (
                    cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern1_id TEXT NOT NULL,
                    pattern2_id TEXT NOT NULL,
                    similarity_score REAL,
                    computed_at TIMESTAMP,
                    UNIQUE(pattern1_id, pattern2_id)
                )
            """)

            # Context embeddings metadata
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS context_embeddings (
                    embedding_id TEXT PRIMARY KEY,
                    context_hash TEXT UNIQUE,
                    embedding_vector BLOB,
                    dimension INTEGER,
                    created_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP
                )
            """)

            # ============================================================================
            # 24/7 BEHAVIORAL LEARNING TABLES - Enhanced Workspace Tracking
            # ============================================================================

            # Workspace/Space tracking (Yabai integration)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS workspace_usage (
                    usage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    space_id INTEGER NOT NULL,
                    space_label TEXT,
                    app_name TEXT NOT NULL,
                    window_title TEXT,
                    window_position JSON,
                    focus_duration_seconds REAL,
                    timestamp TIMESTAMP,
                    day_of_week INTEGER,
                    hour_of_day INTEGER,
                    is_fullscreen BOOLEAN DEFAULT 0,
                    metadata JSON
                )
            """)

            # App usage patterns (24/7 tracking)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS app_usage_patterns (
                    pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    app_name TEXT NOT NULL,
                    space_id INTEGER,
                    usage_frequency INTEGER DEFAULT 1,
                    avg_session_duration REAL,
                    total_usage_time REAL,
                    typical_time_of_day INTEGER,
                    typical_day_of_week INTEGER,
                    last_used TIMESTAMP,
                    confidence REAL DEFAULT 0.5,
                    metadata JSON,
                    UNIQUE(app_name, space_id, typical_time_of_day, typical_day_of_week)
                )
            """)

            # User workflows (sequential action patterns)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_workflows (
                    workflow_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_name TEXT,
                    action_sequence JSON NOT NULL,
                    space_sequence JSON,
                    app_sequence JSON,
                    frequency INTEGER DEFAULT 1,
                    avg_duration REAL,
                    success_rate REAL DEFAULT 1.0,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    time_of_day_pattern JSON,
                    confidence REAL DEFAULT 0.5,
                    metadata JSON
                )
            """)

            # Space transitions (movement between Spaces)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS space_transitions (
                    transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    from_space_id INTEGER NOT NULL,
                    to_space_id INTEGER NOT NULL,
                    trigger_app TEXT,
                    trigger_action TEXT,
                    frequency INTEGER DEFAULT 1,
                    avg_time_between_seconds REAL,
                    timestamp TIMESTAMP,
                    hour_of_day INTEGER,
                    day_of_week INTEGER,
                    metadata JSON
                )
            """)

            # Behavioral patterns (high-level user habits)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS behavioral_patterns (
                    behavior_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    behavior_type TEXT NOT NULL,
                    behavior_description TEXT,
                    pattern_data JSON NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    temporal_pattern JSON,
                    contextual_triggers JSON,
                    first_observed TIMESTAMP,
                    last_observed TIMESTAMP,
                    prediction_accuracy REAL,
                    metadata JSON
                )
            """)

            # Temporal patterns (time-based behaviors for leap years too!)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_patterns (
                    temporal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    time_of_day INTEGER,
                    day_of_week INTEGER,
                    day_of_month INTEGER,
                    month_of_year INTEGER,
                    is_leap_year BOOLEAN DEFAULT 0,
                    action_type TEXT NOT NULL,
                    target TEXT,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 0.5,
                    last_occurrence TIMESTAMP,
                    metadata JSON
                )
            """)

            # Proactive suggestions (what JARVIS should suggest)
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS proactive_suggestions (
                    suggestion_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    suggestion_type TEXT NOT NULL,
                    suggestion_text TEXT NOT NULL,
                    trigger_pattern_id TEXT,
                    confidence REAL NOT NULL,
                    times_suggested INTEGER DEFAULT 0,
                    times_accepted INTEGER DEFAULT 0,
                    times_rejected INTEGER DEFAULT 0,
                    acceptance_rate REAL,
                    created_at TIMESTAMP,
                    last_suggested TIMESTAMP,
                    metadata JSON
                )
            """)

            # Performance indexes
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_type ON goals(goal_type)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_created ON goals(created_at)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_context_hash ON goals(context_hash)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_type ON actions(action_type)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_timestamp ON actions(timestamp)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_actions_context_hash ON actions(context_hash)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_hash ON patterns(pattern_hash)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_display_patterns_context ON display_patterns(context_hash)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_display_patterns_time ON display_patterns(connection_time, day_of_week)")

            # Indexes for 24/7 behavioral learning tables
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_workspace_usage_space ON workspace_usage(space_id, timestamp)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_workspace_usage_app ON workspace_usage(app_name, timestamp)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_workspace_usage_time ON workspace_usage(hour_of_day, day_of_week)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_usage_app ON app_usage_patterns(app_name)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_usage_space ON app_usage_patterns(space_id)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_app_usage_time ON app_usage_patterns(typical_time_of_day, typical_day_of_week)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflows_frequency ON user_workflows(frequency DESC)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_space_transitions_from ON space_transitions(from_space_id, timestamp)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_space_transitions_to ON space_transitions(to_space_id, timestamp)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_behavioral_patterns_type ON behavioral_patterns(behavior_type)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_patterns_time ON temporal_patterns(time_of_day, day_of_week)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_temporal_patterns_leap ON temporal_patterns(is_leap_year)")
            await cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_confidence ON proactive_suggestions(confidence DESC)")

        await self.db.commit()
        logger.info("SQLite database initialized with enhanced async schema")

    async def _init_chromadb(self):
        """Initialize ChromaDB for embeddings and semantic search"""
        try:
            # Initialize ChromaDB client with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            # Create or get collections with metadata
            self.goal_collection = self.chroma_client.get_or_create_collection(
                name="goal_embeddings",
                metadata={
                    "description": "Goal context embeddings for similarity search",
                    "created": datetime.now().isoformat()
                }
            )

            self.pattern_collection = self.chroma_client.get_or_create_collection(
                name="pattern_embeddings",
                metadata={
                    "description": "Pattern embeddings for matching",
                    "created": datetime.now().isoformat()
                }
            )

            self.context_collection = self.chroma_client.get_or_create_collection(
                name="context_embeddings",
                metadata={
                    "description": "Context state embeddings for prediction",
                    "created": datetime.now().isoformat()
                }
            )

            logger.info("ChromaDB initialized for semantic search")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.chroma_client = None

    # ==================== Goal Management (Async + Cached) ====================

    async def store_goal(self, goal: Dict[str, Any], batch: bool = False) -> str:
        """Store an inferred goal with batching support"""
        goal_id = goal.get('goal_id', self._generate_id('goal'))

        if batch:
            self.pending_goals.append((goal_id, goal))
            if len(self.pending_goals) >= self.batch_size:
                await self._flush_goal_batch()
            return goal_id

        # Compute context hash for deduplication
        context_hash = self._hash_context(goal.get('evidence', {}))

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("""
                    INSERT OR REPLACE INTO goals
                    (goal_id, goal_type, goal_level, description, confidence,
                     progress, is_completed, created_at, evidence, context_hash, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal_id,
                    goal['goal_type'],
                    goal.get('goal_level', 'UNKNOWN'),
                    goal.get('description', ''),
                    goal.get('confidence', 0.0),
                    goal.get('progress', 0.0),
                    goal.get('is_completed', False),
                    goal.get('created_at', datetime.now()),
                    json.dumps(goal.get('evidence', [])),
                    context_hash,
                    json.dumps(goal.get('metadata', {}))
                ))

            await self.db.commit()

        # Store embedding in ChromaDB if available
        if self.goal_collection and 'embedding' in goal:
            await self._store_embedding_async(
                self.goal_collection,
                goal_id,
                goal['embedding'],
                {
                    'goal_type': goal['goal_type'],
                    'confidence': goal.get('confidence', 0.0),
                    'timestamp': datetime.now().isoformat()
                }
            )

        # Update cache
        self.goal_cache.set(goal_id, goal)

        return goal_id

    async def get_similar_goals(self, embedding: List[float], top_k: int = 5,
                               min_confidence: float = 0.5) -> List[Dict]:
        """Find similar goals using semantic search with caching"""
        cache_key = f"similar_goals_{hashlib.md5(str(embedding).encode()).hexdigest()}_{top_k}"

        cached = self.query_cache.get(cache_key)
        if cached:
            return cached

        if not self.goal_collection:
            return []

        try:
            results = self.goal_collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"confidence": {"$gte": min_confidence}} if min_confidence > 0 else None
            )

            similar_goals = []
            async with self.db.cursor() as cursor:
                for i, goal_id in enumerate(results['ids'][0]):
                    await cursor.execute("SELECT * FROM goals WHERE goal_id = ?", (goal_id,))
                    row = await cursor.fetchone()

                    if row:
                        goal = dict(row)
                        goal['evidence'] = json.loads(goal['evidence']) if goal['evidence'] else []
                        goal['metadata'] = json.loads(goal['metadata']) if goal['metadata'] else {}
                        goal['similarity'] = 1.0 - results['distances'][0][i]
                        similar_goals.append(goal)

            self.query_cache.set(cache_key, similar_goals)
            return similar_goals

        except Exception as e:
            logger.error(f"Error finding similar goals: {e}")
            return []

    # ==================== Action Tracking (Async + Metrics) ====================

    async def store_action(self, action: Dict[str, Any], batch: bool = False) -> str:
        """Store an executed action with performance tracking"""
        action_id = action.get('action_id', self._generate_id('action'))

        if batch:
            self.pending_actions.append((action_id, action))
            if len(self.pending_actions) >= self.batch_size:
                await self._flush_action_batch()
            return action_id

        context_hash = self._hash_context(action.get('params', {}))

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO actions
                    (action_id, action_type, target, goal_id, confidence,
                     success, execution_time, timestamp, retry_count, params, result, context_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    action_id,
                    action['action_type'],
                    action.get('target', ''),
                    action.get('goal_id'),
                    action.get('confidence', 0.0),
                    action.get('success', False),
                    action.get('execution_time', 0.0),
                    action.get('timestamp', datetime.now()),
                    action.get('retry_count', 0),
                    json.dumps(action.get('params', {})),
                    json.dumps(action.get('result', {})),
                    context_hash
                ))

            await self.db.commit()

        # Update goal-action mapping
        if action.get('goal_id'):
            await self._update_goal_action_mapping(
                action['action_type'],
                action.get('goal_id'),
                action.get('success', False),
                action.get('execution_time', 0.0)
            )

        return action_id

    async def _update_goal_action_mapping(self, action_type: str, goal_id: Optional[str],
                                          success: bool, execution_time: float):
        """Update goal-action mapping with statistical tracking"""
        if not goal_id:
            return

        async with self.db.cursor() as cursor:
            # Get goal type
            await cursor.execute("SELECT goal_type FROM goals WHERE goal_id = ?", (goal_id,))
            row = await cursor.fetchone()

            if row:
                goal_type = row['goal_type']

                # Calculate new statistics
                success_inc = 1 if success else 0
                failure_inc = 0 if success else 1

                await cursor.execute("""
                    INSERT INTO goal_action_mappings
                    (goal_type, action_type, success_count, failure_count,
                     avg_execution_time, confidence, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(goal_type, action_type) DO UPDATE SET
                        success_count = success_count + ?,
                        failure_count = failure_count + ?,
                        avg_execution_time = (avg_execution_time * (success_count + failure_count) + ?)
                                            / (success_count + failure_count + 1),
                        confidence = CAST(success_count + ? AS REAL) / (success_count + failure_count + 1),
                        last_updated = ?
                """, (
                    goal_type, action_type,
                    success_inc, failure_inc,
                    execution_time, 0.5, datetime.now(),
                    success_inc, failure_inc,
                    execution_time, success_inc, datetime.now()
                ))

        await self.db.commit()

    # ==================== Pattern Learning (ML-Powered) ====================

    async def store_pattern(self, pattern: Dict[str, Any], auto_merge: bool = True) -> str:
        """Store pattern with automatic similarity-based merging"""
        pattern_id = pattern.get('pattern_id', self._generate_id('pattern'))
        pattern_hash = self._hash_pattern(pattern)

        # Check cache first
        cached_pattern = self.pattern_cache.get(pattern_hash)
        if cached_pattern:
            # Update occurrence count
            await self._increment_pattern_occurrence(cached_pattern['pattern_id'])
            return cached_pattern['pattern_id']

        # Check for similar patterns if auto_merge enabled
        if auto_merge and 'embedding' in pattern:
            similar = await self._find_similar_patterns(
                pattern['embedding'],
                pattern['pattern_type'],
                similarity_threshold=0.85
            )

            if similar:
                # Merge with most similar pattern
                await self._merge_patterns(similar[0]['pattern_id'], pattern)
                return similar[0]['pattern_id']

        # Store new pattern
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("""
                    INSERT OR REPLACE INTO patterns
                    (pattern_id, pattern_type, pattern_hash, pattern_data, confidence,
                     success_rate, occurrence_count, first_seen, last_seen, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern_id,
                    pattern['pattern_type'],
                    pattern_hash,
                    json.dumps(pattern.get('pattern_data', {})),
                    pattern.get('confidence', 0.5),
                    pattern.get('success_rate', 0.5),
                    1,
                    datetime.now(),
                    datetime.now(),
                    json.dumps(pattern.get('metadata', {}))
                ))

            await self.db.commit()

        # Store embedding
        if self.pattern_collection and 'embedding' in pattern:
            await self._store_embedding_async(
                self.pattern_collection,
                pattern_id,
                pattern['embedding'],
                {
                    'pattern_type': pattern['pattern_type'],
                    'confidence': pattern.get('confidence', 0.5),
                    'timestamp': datetime.now().isoformat()
                }
            )

        # Update cache
        self.pattern_cache.set(pattern_hash, {'pattern_id': pattern_id, **pattern})

        return pattern_id

    async def _find_similar_patterns(self, embedding: List[float], pattern_type: str,
                                    similarity_threshold: float = 0.8) -> List[Dict]:
        """Find similar patterns using embeddings"""
        if not self.pattern_collection:
            return []

        try:
            results = self.pattern_collection.query(
                query_embeddings=[embedding],
                n_results=5,
                where={"pattern_type": pattern_type}
            )

            similar_patterns = []
            async with self.db.cursor() as cursor:
                for i, pattern_id in enumerate(results['ids'][0]):
                    similarity = 1.0 - results['distances'][0][i]
                    if similarity >= similarity_threshold:
                        await cursor.execute("SELECT * FROM patterns WHERE pattern_id = ?", (pattern_id,))
                        row = await cursor.fetchone()
                        if row:
                            pattern = dict(row)
                            pattern['similarity'] = similarity
                            similar_patterns.append(pattern)

            return similar_patterns

        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return []

    async def _merge_patterns(self, target_pattern_id: str, new_pattern: Dict):
        """Merge new pattern into existing pattern"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Update occurrence count and confidence
                await cursor.execute("""
                    UPDATE patterns SET
                        occurrence_count = occurrence_count + 1,
                        confidence = (confidence + ?) / 2,
                        success_rate = (success_rate + ?) / 2,
                        last_seen = ?,
                        boost_count = boost_count + 1
                    WHERE pattern_id = ?
                """, (
                    new_pattern.get('confidence', 0.5),
                    new_pattern.get('success_rate', 0.5),
                    datetime.now(),
                    target_pattern_id
                ))

            await self.db.commit()

    async def _increment_pattern_occurrence(self, pattern_id: str):
        """Increment pattern occurrence count"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("""
                    UPDATE patterns SET
                        occurrence_count = occurrence_count + 1,
                        last_seen = ?
                    WHERE pattern_id = ?
                """, (datetime.now(), pattern_id))

            await self.db.commit()

    async def get_pattern_by_type(self, pattern_type: str,
                                  min_confidence: float = 0.5,
                                  limit: int = 10) -> List[Dict]:
        """Get patterns by type with caching"""
        cache_key = f"patterns_{pattern_type}_{min_confidence}_{limit}"

        cached = self.query_cache.get(cache_key)
        if cached:
            return cached

        async with self.db.cursor() as cursor:
            await cursor.execute("""
                SELECT * FROM patterns
                WHERE pattern_type = ? AND confidence >= ?
                ORDER BY confidence DESC, occurrence_count DESC
                LIMIT ?
            """, (pattern_type, min_confidence, limit))

            rows = await cursor.fetchall()
            patterns = [dict(row) for row in rows]

        self.query_cache.set(cache_key, patterns)
        return patterns

    # ==================== Display Patterns (Enhanced) ====================

    async def learn_display_pattern(self, display_name: str, context: Dict[str, Any]):
        """Learn display connection patterns with temporal analysis"""
        now = datetime.now()
        time_str = now.strftime("%H:%M")
        day_of_week = now.weekday()
        hour_of_day = now.hour
        context_hash = self._hash_context(context)

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Check if similar pattern exists
                await cursor.execute("""
                    SELECT * FROM display_patterns
                    WHERE display_name = ?
                    AND hour_of_day = ?
                    AND day_of_week = ?
                """, (display_name, hour_of_day, day_of_week))

                existing = await cursor.fetchone()

                if existing:
                    # Update frequency and consecutive successes
                    await cursor.execute("""
                        UPDATE display_patterns SET
                            frequency = frequency + 1,
                            consecutive_successes = consecutive_successes + 1,
                            last_seen = ?,
                            auto_connect = CASE WHEN frequency >= 3 THEN 1 ELSE auto_connect END,
                            context = ?
                        WHERE pattern_id = ?
                    """, (datetime.now(), json.dumps(context), existing['pattern_id']))

                    if existing['frequency'] >= 2:
                        logger.info(f"📊 Display pattern strengthened: {display_name} at {hour_of_day}:00 on day {day_of_week}")
                else:
                    # Insert new pattern
                    await cursor.execute("""
                        INSERT INTO display_patterns
                        (display_name, context, context_hash, connection_time, day_of_week,
                         hour_of_day, frequency, auto_connect, last_seen, consecutive_successes, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        display_name,
                        json.dumps(context),
                        context_hash,
                        time_str,
                        day_of_week,
                        hour_of_day,
                        1,
                        False,
                        datetime.now(),
                        1,
                        json.dumps({})
                    ))

            await self.db.commit()

    async def should_auto_connect_display(self, current_context: Optional[Dict] = None) -> Optional[Tuple[str, float]]:
        """Predict display connection with context awareness"""
        now = datetime.now()
        hour_of_day = now.hour
        day_of_week = now.weekday()

        async with self.db.cursor() as cursor:
            # Look for matching patterns with temporal proximity
            await cursor.execute("""
                SELECT display_name, frequency, consecutive_successes, auto_connect
                FROM display_patterns
                WHERE hour_of_day = ?
                AND day_of_week = ?
                AND frequency >= 2
                ORDER BY frequency DESC, consecutive_successes DESC
                LIMIT 1
            """, (hour_of_day, day_of_week))

            row = await cursor.fetchone()

            if row:
                # Calculate dynamic confidence
                base_confidence = min(row['frequency'] / 10.0, 0.85)
                consecutive_bonus = min(row['consecutive_successes'] * 0.05, 0.10)
                confidence = min(base_confidence + consecutive_bonus, 0.95)

                return row['display_name'], confidence

        return None

    # ==================== User Preferences (Enhanced) ====================

    async def learn_preference(self, category: str, key: str, value: Any,
                              confidence: float = 0.5, learned_from: str = 'implicit'):
        """Learn user preference with confidence averaging"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("""
                    INSERT INTO user_preferences
                    (preference_id, category, key, value, confidence,
                     learned_from, created_at, updated_at, update_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(category, key) DO UPDATE SET
                        value = ?,
                        confidence = (confidence * update_count + ?) / (update_count + 1),
                        update_count = update_count + 1,
                        updated_at = ?
                """, (
                    f"{category}_{key}",
                    category, key, str(value), confidence,
                    learned_from, datetime.now(), datetime.now(), 1,
                    str(value), confidence, datetime.now()
                ))

            await self.db.commit()

    async def get_preference(self, category: str, key: str,
                            min_confidence: float = 0.5) -> Optional[Dict]:
        """Get learned preference with confidence threshold"""
        cache_key = f"pref_{category}_{key}"

        cached = self.query_cache.get(cache_key)
        if cached:
            return cached

        async with self.db.cursor() as cursor:
            await cursor.execute("""
                SELECT * FROM user_preferences
                WHERE category = ? AND key = ? AND confidence >= ?
            """, (category, key, min_confidence))

            row = await cursor.fetchone()
            result = dict(row) if row else None

            if result:
                self.query_cache.set(cache_key, result)

            return result

    # ==================== Analytics & Metrics ====================

    async def get_learning_metrics(self) -> Dict[str, Any]:
        """Get comprehensive real-time learning metrics"""
        metrics = {}

        async with self.db.cursor() as cursor:
            # Goal metrics
            await cursor.execute("""
                SELECT
                    COUNT(*) as total_goals,
                    AVG(confidence) as avg_confidence,
                    SUM(is_completed) as completed_goals,
                    AVG(actual_duration) as avg_duration
                FROM goals
                WHERE created_at >= datetime('now', '-30 days')
            """)
            metrics['goals'] = dict(await cursor.fetchone())

            # Action metrics
            await cursor.execute("""
                SELECT
                    COUNT(*) as total_actions,
                    AVG(confidence) as avg_confidence,
                    SUM(success) * 100.0 / COUNT(*) as success_rate,
                    AVG(execution_time) as avg_execution_time,
                    AVG(retry_count) as avg_retries
                FROM actions
                WHERE timestamp >= datetime('now', '-30 days')
            """)
            metrics['actions'] = dict(await cursor.fetchone())

            # Pattern metrics
            await cursor.execute("""
                SELECT
                    COUNT(*) as total_patterns,
                    AVG(confidence) as avg_confidence,
                    AVG(success_rate) as avg_success_rate,
                    SUM(occurrence_count) as total_occurrences,
                    AVG(occurrence_count) as avg_occurrences_per_pattern
                FROM patterns
            """)
            metrics['patterns'] = dict(await cursor.fetchone())

            # Display pattern metrics
            await cursor.execute("""
                SELECT
                    COUNT(*) as total_display_patterns,
                    SUM(auto_connect) as auto_connect_enabled,
                    MAX(frequency) as max_frequency,
                    AVG(consecutive_successes) as avg_consecutive_successes
                FROM display_patterns
            """)
            metrics['display_patterns'] = dict(await cursor.fetchone())

            # Goal-action mapping insights
            await cursor.execute("""
                SELECT
                    goal_type,
                    action_type,
                    confidence,
                    success_count + failure_count as total_attempts,
                    avg_execution_time
                FROM goal_action_mappings
                WHERE confidence > 0.7
                ORDER BY confidence DESC
                LIMIT 10
            """)
            metrics['top_mappings'] = [dict(row) for row in await cursor.fetchall()]

        # Add cache metrics
        metrics['cache_performance'] = {
            'pattern_cache_hit_rate': self.pattern_cache.hit_rate(),
            'goal_cache_hit_rate': self.goal_cache.hit_rate(),
            'query_cache_hit_rate': self.query_cache.hit_rate()
        }

        # Update internal metrics
        self.metrics.total_patterns = metrics['patterns']['total_patterns']
        self.metrics.avg_confidence = metrics['patterns']['avg_confidence'] or 0.0
        self.metrics.cache_hit_rate = metrics['cache_performance']['pattern_cache_hit_rate']
        self.metrics.last_updated = datetime.now()

        return metrics

    async def analyze_patterns(self) -> List[Dict]:
        """Advanced pattern analysis with ML insights"""
        patterns = []

        async with self.db.cursor() as cursor:
            await cursor.execute("""
                SELECT *
                FROM patterns
                WHERE occurrence_count >= 3
                ORDER BY success_rate DESC, occurrence_count DESC
                LIMIT 50
            """)

            rows = await cursor.fetchall()

            for row in rows:
                pattern = dict(row)

                # Calculate pattern strength score
                strength = (
                    pattern['confidence'] * 0.4 +
                    pattern['success_rate'] * 0.4 +
                    min(pattern['occurrence_count'] / 100.0, 1.0) * 0.2
                )
                pattern['strength_score'] = strength

                # Time since last seen
                last_seen = datetime.fromisoformat(pattern['last_seen'])
                days_since = (datetime.now() - last_seen).days
                pattern['days_since_last_seen'] = days_since

                # Decay recommendation
                if days_since > 30 and pattern['occurrence_count'] < 5:
                    pattern['should_decay'] = True
                else:
                    pattern['should_decay'] = False

                patterns.append(pattern)

        return patterns

    async def boost_pattern_confidence(self, pattern_id: str, boost: float = 0.05,
                                      strategy: ConfidenceBoostStrategy = ConfidenceBoostStrategy.ADAPTIVE):
        """Boost pattern confidence using configurable strategy"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("SELECT confidence, boost_count FROM patterns WHERE pattern_id = ?", (pattern_id,))
                row = await cursor.fetchone()

                if row:
                    current_confidence = row['confidence']
                    boost_count = row['boost_count']

                    # Apply strategy
                    if strategy == ConfidenceBoostStrategy.LINEAR:
                        new_confidence = min(current_confidence + boost, 1.0)
                    elif strategy == ConfidenceBoostStrategy.EXPONENTIAL:
                        new_confidence = min(current_confidence * (1.0 + boost), 1.0)
                    elif strategy == ConfidenceBoostStrategy.LOGARITHMIC:
                        new_confidence = min(current_confidence + boost / (1 + boost_count), 1.0)
                    else:  # ADAPTIVE
                        # Reduces boost as confidence increases
                        adaptive_boost = boost * (1.0 - current_confidence)
                        new_confidence = min(current_confidence + adaptive_boost, 1.0)

                    await cursor.execute("""
                        UPDATE patterns SET
                            confidence = ?,
                            boost_count = boost_count + 1
                        WHERE pattern_id = ?
                    """, (new_confidence, pattern_id))

                await self.db.commit()

    # ==================== Maintenance & Optimization ====================

    async def cleanup_old_patterns(self, days: int = 30):
        """Clean up old unused patterns with decay"""
        cutoff_date = datetime.now() - timedelta(days=days)

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Decay old patterns instead of deleting
                await cursor.execute("""
                    UPDATE patterns SET
                        confidence = confidence * 0.9,
                        decay_applied = 1
                    WHERE last_seen < ? AND occurrence_count < 3
                """, (cutoff_date,))

                # Delete very old patterns with low occurrence
                await cursor.execute("""
                    DELETE FROM patterns
                    WHERE last_seen < ? AND occurrence_count = 1 AND confidence < 0.3
                """, (cutoff_date,))

                deleted_count = cursor.rowcount

            await self.db.commit()

        logger.info(f"🧹 Cleaned up {deleted_count} old patterns")

    async def optimize(self):
        """Optimize database performance"""
        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.execute("ANALYZE")
                await cursor.execute("VACUUM")

            await self.db.commit()

        logger.info("✨ Database optimized")

    async def _auto_flush_batches(self):
        """Auto-flush batch queues periodically"""
        while True:
            await asyncio.sleep(5)  # Flush every 5 seconds

            if self.pending_goals:
                await self._flush_goal_batch()

            if self.pending_actions:
                await self._flush_action_batch()

            if self.pending_patterns:
                await self._flush_pattern_batch()

    async def _flush_goal_batch(self):
        """Flush pending goals"""
        if not self.pending_goals:
            return

        goals_to_insert = list(self.pending_goals)
        self.pending_goals.clear()

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.executemany("""
                    INSERT OR REPLACE INTO goals
                    (goal_id, goal_type, goal_level, description, confidence,
                     progress, is_completed, created_at, evidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        goal_id,
                        goal['goal_type'],
                        goal.get('goal_level', 'UNKNOWN'),
                        goal.get('description', ''),
                        goal.get('confidence', 0.0),
                        goal.get('progress', 0.0),
                        goal.get('is_completed', False),
                        goal.get('created_at', datetime.now()),
                        json.dumps(goal.get('evidence', [])),
                        json.dumps(goal.get('metadata', {}))
                    )
                    for goal_id, goal in goals_to_insert
                ])

            await self.db.commit()

    async def _flush_action_batch(self):
        """Flush pending actions"""
        if not self.pending_actions:
            return

        actions_to_insert = list(self.pending_actions)
        self.pending_actions.clear()

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                await cursor.executemany("""
                    INSERT INTO actions
                    (action_id, action_type, target, goal_id, confidence,
                     success, execution_time, timestamp, params, result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    (
                        action_id,
                        action['action_type'],
                        action.get('target', ''),
                        action.get('goal_id'),
                        action.get('confidence', 0.0),
                        action.get('success', False),
                        action.get('execution_time', 0.0),
                        action.get('timestamp', datetime.now()),
                        json.dumps(action.get('params', {})),
                        json.dumps(action.get('result', {}))
                    )
                    for action_id, action in actions_to_insert
                ])

            await self.db.commit()

    async def _flush_pattern_batch(self):
        """Flush pending patterns with intelligent deduplication and merging"""
        if not self.pending_patterns:
            return

        patterns_to_process = list(self.pending_patterns)
        self.pending_patterns.clear()

        # Group patterns by hash for intelligent merging
        pattern_groups: Dict[str, List[Tuple[str, Dict]]] = defaultdict(list)

        for pattern_id, pattern in patterns_to_process:
            pattern_hash = self._hash_pattern(pattern)
            pattern_groups[pattern_hash].append((pattern_id, pattern))

        async with self._db_lock:
            async with self.db.cursor() as cursor:
                # Process each group
                for pattern_hash, group in pattern_groups.items():
                    if len(group) == 1:
                        # Single pattern - insert directly
                        pattern_id, pattern = group[0]

                        await cursor.execute("""
                            INSERT OR REPLACE INTO patterns
                            (pattern_id, pattern_type, pattern_hash, pattern_data, confidence,
                             success_rate, occurrence_count, first_seen, last_seen, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            pattern_id,
                            pattern['pattern_type'],
                            pattern_hash,
                            json.dumps(pattern.get('pattern_data', {})),
                            pattern.get('confidence', 0.5),
                            pattern.get('success_rate', 0.5),
                            1,
                            datetime.now(),
                            datetime.now(),
                            json.dumps(pattern.get('metadata', {}))
                        ))

                        # Store embedding if available
                        if self.pattern_collection and 'embedding' in pattern:
                            await self._store_embedding_async(
                                self.pattern_collection,
                                pattern_id,
                                pattern['embedding'],
                                {
                                    'pattern_type': pattern['pattern_type'],
                                    'confidence': pattern.get('confidence', 0.5),
                                    'timestamp': datetime.now().isoformat()
                                }
                            )
                    else:
                        # Multiple patterns with same hash - merge intelligently
                        # Use the first pattern as base
                        base_pattern_id, base_pattern = group[0]

                        # Calculate merged confidence (weighted average)
                        total_confidence = sum(p.get('confidence', 0.5) for _, p in group)
                        avg_confidence = total_confidence / len(group)

                        # Calculate merged success rate
                        total_success_rate = sum(p.get('success_rate', 0.5) for _, p in group)
                        avg_success_rate = total_success_rate / len(group)

                        # Merge metadata intelligently
                        merged_metadata = {}
                        for _, pattern in group:
                            pattern_meta = pattern.get('metadata', {})
                            for key, value in pattern_meta.items():
                                if key not in merged_metadata:
                                    merged_metadata[key] = []
                                if value not in merged_metadata[key]:
                                    merged_metadata[key].append(value)

                        # Flatten single-value lists
                        for key in merged_metadata:
                            if len(merged_metadata[key]) == 1:
                                merged_metadata[key] = merged_metadata[key][0]

                        # Insert merged pattern
                        await cursor.execute("""
                            INSERT OR REPLACE INTO patterns
                            (pattern_id, pattern_type, pattern_hash, pattern_data, confidence,
                             success_rate, occurrence_count, first_seen, last_seen, boost_count, metadata)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            base_pattern_id,
                            base_pattern['pattern_type'],
                            pattern_hash,
                            json.dumps(base_pattern.get('pattern_data', {})),
                            avg_confidence,
                            avg_success_rate,
                            len(group),  # Number of merged patterns
                            datetime.now(),
                            datetime.now(),
                            len(group) - 1,  # Boost count from merging
                            json.dumps(merged_metadata)
                        ))

                        # Store embedding for merged pattern
                        if self.pattern_collection and 'embedding' in base_pattern:
                            # Average all embeddings if multiple available
                            embeddings_to_merge = [
                                p.get('embedding') for _, p in group
                                if 'embedding' in p
                            ]

                            if embeddings_to_merge and NUMPY_AVAILABLE:
                                # Average embeddings
                                import numpy as np
                                avg_embedding = np.mean(embeddings_to_merge, axis=0).tolist()

                                await self._store_embedding_async(
                                    self.pattern_collection,
                                    base_pattern_id,
                                    avg_embedding,
                                    {
                                        'pattern_type': base_pattern['pattern_type'],
                                        'confidence': avg_confidence,
                                        'merged_count': len(group),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                )
                            elif embeddings_to_merge:
                                # Use first embedding if numpy not available
                                await self._store_embedding_async(
                                    self.pattern_collection,
                                    base_pattern_id,
                                    embeddings_to_merge[0],
                                    {
                                        'pattern_type': base_pattern['pattern_type'],
                                        'confidence': avg_confidence,
                                        'merged_count': len(group),
                                        'timestamp': datetime.now().isoformat()
                                    }
                                )

                        # Update cache with merged pattern
                        self.pattern_cache.set(pattern_hash, {
                            'pattern_id': base_pattern_id,
                            'confidence': avg_confidence,
                            'success_rate': avg_success_rate,
                            'occurrence_count': len(group)
                        })

                        logger.debug(f"🔀 Merged {len(group)} similar patterns into {base_pattern_id}")

            await self.db.commit()

        logger.debug(f"✅ Flushed {len(patterns_to_process)} patterns ({len(pattern_groups)} unique)")

    async def _auto_optimize_task(self):
        """Auto-optimize database periodically"""
        if not self.auto_optimize:
            return

        while True:
            await asyncio.sleep(3600)  # Every hour
            await self.optimize()

    async def _load_metrics(self):
        """Load initial metrics"""
        try:
            metrics = await self.get_learning_metrics()
            logger.info(f"📊 Loaded metrics: {metrics['patterns']['total_patterns']} patterns")
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")

    async def _store_embedding_async(self, collection, doc_id: str,
                                    embedding: List[float], metadata: Dict):
        """Store embedding asynchronously"""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: collection.upsert(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata]
                )
            )
        except Exception as e:
            logger.error(f"Failed to store embedding: {e}")

    # ==================== Utility Methods ====================

    @staticmethod
    def _generate_id(prefix: str) -> str:
        """Generate unique ID"""
        timestamp = datetime.now().timestamp()
        return f"{prefix}_{timestamp}_{hashlib.md5(str(timestamp).encode()).hexdigest()[:8]}"

    @staticmethod
    def _hash_context(context: Dict) -> str:
        """Generate hash for context deduplication"""
        context_str = json.dumps(context, sort_keys=True)
        return hashlib.sha256(context_str.encode()).hexdigest()[:16]

    @staticmethod
    def _hash_pattern(pattern: Dict) -> str:
        """Generate hash for pattern deduplication"""
        pattern_str = json.dumps({
            'type': pattern.get('pattern_type'),
            'data': pattern.get('pattern_data')
        }, sort_keys=True)
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]

    async def close(self):
        """Close database connections gracefully"""
        # Flush pending batches
        await self._flush_goal_batch()
        await self._flush_action_batch()
        await self._flush_pattern_batch()

        # Close SQLite
        if self.db:
            await self.db.close()

        logger.info("✅ Learning database closed gracefully")


# Global instance with async initialization
_db_instance = None
_db_lock = asyncio.Lock()


async def get_learning_database(config: Optional[Dict] = None) -> JARVISLearningDatabase:
    """Get or create the global async learning database"""
    global _db_instance

    async with _db_lock:
        if _db_instance is None:
            _db_instance = JARVISLearningDatabase(config=config)
            await _db_instance.initialize()
        return _db_instance


async def test_database():
    """Test the advanced learning database"""
    print("🗄️ Testing Advanced JARVIS Learning Database")
    print("=" * 60)

    db = await get_learning_database()

    # Test storing a goal
    goal = {
        'goal_id': 'test_goal_1',
        'goal_type': 'meeting_preparation',
        'goal_level': 'HIGH',
        'description': 'Prepare for team meeting',
        'confidence': 0.92,
        'evidence': [{'source': 'calendar', 'data': 'meeting in 10 min'}]
    }

    goal_id = await db.store_goal(goal)
    print(f"✅ Stored goal: {goal_id}")

    # Test batch storage
    for i in range(5):
        await db.store_goal({
            'goal_type': 'test',
            'confidence': 0.5 + i * 0.1
        }, batch=True)
    print(f"✅ Queued 5 goals for batch insert")

    # Test storing an action
    action = {
        'action_id': 'test_action_1',
        'action_type': 'connect_display',
        'target': 'Living Room TV',
        'goal_id': goal_id,
        'confidence': 0.85,
        'success': True,
        'execution_time': 0.45
    }

    action_id = await db.store_action(action)
    print(f"✅ Stored action: {action_id}")

    # Test learning display pattern
    await db.learn_display_pattern('Living Room TV', {
        'apps': ['keynote', 'calendar'],
        'time': '09:00'
    })
    print("✅ Learned display pattern")

    # Test learning preference
    await db.learn_preference('display', 'default', 'Living Room TV', 0.8)
    print("✅ Learned preference")

    # Get metrics
    metrics = await db.get_learning_metrics()
    print("\n📊 Learning Metrics:")
    print(f"   Total Goals: {metrics['goals']['total_goals']}")
    print(f"   Total Actions: {metrics['actions']['total_actions']}")
    print(f"   Total Patterns: {metrics['patterns']['total_patterns']}")
    print(f"   Pattern Cache Hit Rate: {metrics['cache_performance']['pattern_cache_hit_rate']:.2%}")

    # Test pattern analysis
    patterns = await db.analyze_patterns()
    print(f"\n🔍 Analyzed {len(patterns)} patterns")

    await db.close()
    print("\n✅ Advanced database test complete!")


if __name__ == "__main__":
    asyncio.run(test_database())
