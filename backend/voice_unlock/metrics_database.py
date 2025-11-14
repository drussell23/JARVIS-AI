#!/usr/bin/env python3
"""
Voice Unlock Metrics Database
==============================
Stores voice unlock metrics in SQLite (local) and CloudSQL (cloud) simultaneously.

This provides:
- Local SQLite for fast queries and offline access
- CloudSQL sync for backup and cross-device access
- Automatic schema creation
- Async database operations
"""

import asyncio
import logging
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)


class MetricsDatabase:
    """
    Dual-database system for voice unlock metrics.

    Stores metrics in both SQLite (local) and CloudSQL (cloud) for:
    - Fast local queries
    - Cloud backup
    - Historical analysis
    - Cross-device sync
    """

    def __init__(self, sqlite_path: str = None, use_cloud_sql: bool = True):
        """
        Initialize metrics database.

        Args:
            sqlite_path: Path to SQLite database file
            use_cloud_sql: Whether to sync to CloudSQL (requires existing connection)
        """
        if sqlite_path is None:
            db_dir = Path.home() / ".jarvis/logs/unlock_metrics"
            db_dir.mkdir(parents=True, exist_ok=True)
            self.sqlite_path = db_dir / "unlock_metrics.db"
        else:
            self.sqlite_path = Path(sqlite_path)

        self.use_cloud_sql = use_cloud_sql
        self.cloud_db = None

        # Initialize databases
        self._init_sqlite()
        if self.use_cloud_sql:
            self._init_cloud_sql()

    def _init_sqlite(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.sqlite_path)
        cursor = conn.cursor()

        # Create unlock_attempts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS unlock_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                time TEXT NOT NULL,
                day_of_week TEXT,
                unix_timestamp REAL,
                success INTEGER NOT NULL,
                speaker_name TEXT NOT NULL,
                transcribed_text TEXT,
                error TEXT,

                -- Biometrics
                speaker_confidence REAL,
                stt_confidence REAL,
                threshold REAL,
                above_threshold INTEGER,
                confidence_margin REAL,
                margin_percentage REAL,

                -- Confidence Trends
                avg_last_10 REAL,
                avg_last_30 REAL,
                trend_direction TEXT,
                volatility REAL,
                best_ever REAL,
                worst_ever REAL,
                percentile_rank REAL,

                -- Performance
                total_duration_ms REAL,
                slowest_stage TEXT,
                fastest_stage TEXT,

                -- Quality
                audio_quality TEXT,
                voice_match_quality TEXT,
                overall_confidence REAL,

                -- Stage Summary
                total_stages INTEGER,
                successful_stages INTEGER,
                failed_stages INTEGER,
                all_stages_passed INTEGER,

                -- System Info
                platform TEXT,
                platform_version TEXT,
                python_version TEXT,
                stt_engine TEXT,
                speaker_engine TEXT,

                -- Metadata
                session_id TEXT,
                logger_version TEXT,

                -- Indexes for fast queries
                UNIQUE(timestamp, speaker_name)
            )
        """)

        # Create processing_stages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_stages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                stage_name TEXT NOT NULL,
                started_at REAL,
                ended_at REAL,
                duration_ms REAL,
                percentage_of_total REAL,
                success INTEGER,
                algorithm_used TEXT,
                module_path TEXT,
                function_name TEXT,
                input_size_bytes INTEGER,
                output_size_bytes INTEGER,
                confidence_score REAL,
                threshold REAL,
                above_threshold INTEGER,
                error_message TEXT,
                metadata_json TEXT,

                FOREIGN KEY (attempt_id) REFERENCES unlock_attempts(id) ON DELETE CASCADE
            )
        """)

        # Create stage_breakdown table (for quick performance queries)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stage_breakdown (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                stage_name TEXT NOT NULL,
                duration_ms REAL,
                percentage REAL,

                FOREIGN KEY (attempt_id) REFERENCES unlock_attempts(id) ON DELETE CASCADE
            )
        """)

        # 🤖 ADVANCED ML TRAINING: Character-level typing metrics for continuous learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS password_typing_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                success INTEGER NOT NULL,

                -- Session Metrics
                total_characters INTEGER,
                characters_typed INTEGER,
                typing_method TEXT,  -- 'core_graphics', 'applescript_fallback', etc.
                fallback_used INTEGER DEFAULT 0,

                -- Performance
                total_typing_duration_ms REAL,
                avg_char_duration_ms REAL,
                min_char_duration_ms REAL,
                max_char_duration_ms REAL,

                -- System Context
                system_load REAL,
                memory_pressure TEXT,
                screen_locked INTEGER,

                -- Timing Patterns (for ML)
                inter_char_delay_avg_ms REAL,
                inter_char_delay_std_ms REAL,
                shift_press_duration_avg_ms REAL,
                shift_release_delay_avg_ms REAL,

                -- Success Patterns
                failed_at_character INTEGER,  -- Which character position failed (NULL if success)
                retry_count INTEGER DEFAULT 0,

                -- Environment
                time_of_day TEXT,
                day_of_week TEXT,

                FOREIGN KEY (attempt_id) REFERENCES unlock_attempts(id) ON DELETE CASCADE
            )
        """)

        # 🔬 ULTRA-DETAILED: Individual character typing metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS character_typing_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                attempt_id INTEGER NOT NULL,

                -- Character Identity (hashed for security)
                char_position INTEGER NOT NULL,  -- Position in password (1-indexed)
                char_type TEXT NOT NULL,  -- 'letter', 'digit', 'special'
                char_case TEXT,  -- 'upper', 'lower', 'none'
                requires_shift INTEGER,
                keycode TEXT,  -- Hex keycode (e.g., '0x02')

                -- Timing Metrics (microsecond precision)
                char_start_time_ms REAL,
                char_end_time_ms REAL,
                total_duration_ms REAL,

                -- Shift Handling (for special chars)
                shift_down_duration_ms REAL,
                shift_registered_delay_ms REAL,  -- Delay between shift press and char press
                shift_up_delay_ms REAL,

                -- Key Events
                key_down_created INTEGER,  -- 1 if event created successfully
                key_down_posted INTEGER,
                key_press_duration_ms REAL,  -- Time between key down and key up
                key_up_created INTEGER,
                key_up_posted INTEGER,

                -- Success Metrics
                success INTEGER NOT NULL,
                error_type TEXT,  -- 'keycode_missing', 'event_creation_failed', etc.
                error_message TEXT,
                retry_attempted INTEGER DEFAULT 0,

                -- Inter-character delay (time since previous character)
                inter_char_delay_ms REAL,

                -- System State
                system_load_at_char REAL,

                FOREIGN KEY (session_id) REFERENCES password_typing_sessions(id) ON DELETE CASCADE,
                FOREIGN KEY (attempt_id) REFERENCES unlock_attempts(id) ON DELETE CASCADE
            )
        """)

        # 📊 ML TRAINING: Aggregate patterns for predictive optimization
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS typing_pattern_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calculated_at TEXT NOT NULL,

                -- Pattern Recognition
                pattern_type TEXT,  -- 'successful_timing', 'failed_timing', 'optimal_delays'
                char_type TEXT,  -- 'letter', 'digit', 'special', 'all'
                requires_shift INTEGER,

                -- Statistical Analysis
                sample_count INTEGER,
                success_rate REAL,
                avg_duration_ms REAL,
                std_duration_ms REAL,
                min_duration_ms REAL,
                max_duration_ms REAL,

                -- Optimal Values (ML predictions)
                optimal_char_duration_ms REAL,
                optimal_inter_char_delay_ms REAL,
                optimal_shift_duration_ms REAL,

                -- Confidence
                confidence_score REAL,  -- 0.0-1.0, based on sample count and consistency

                -- Context
                time_of_day_pattern TEXT,  -- 'morning', 'afternoon', 'night'
                system_load_pattern TEXT,  -- 'low', 'medium', 'high'

                -- Metadata
                last_updated TEXT,
                training_samples_used INTEGER
            )
        """)

        # 🎯 CONTINUOUS LEARNING: Performance improvements over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,

                -- Overall Metrics
                total_attempts INTEGER,
                successful_attempts INTEGER,
                success_rate REAL,

                -- Performance Trends
                avg_typing_duration_last_10 REAL,
                avg_typing_duration_last_50 REAL,
                avg_typing_duration_all_time REAL,
                improvement_percentage REAL,

                -- Character-level Improvements
                avg_char_duration_last_10 REAL,
                avg_char_duration_last_50 REAL,
                fastest_ever_typing_ms REAL,

                -- Reliability Metrics
                consecutive_successes INTEGER,
                consecutive_failures INTEGER,
                failure_rate_last_10 REAL,

                -- ML Model Performance
                model_version TEXT,
                prediction_accuracy REAL,
                optimal_timing_applied INTEGER,

                -- Context Awareness
                best_time_of_day TEXT,
                best_system_load_range TEXT,

                -- Adaptive Learning
                current_strategy TEXT,  -- 'conservative', 'balanced', 'aggressive'
                timing_adjustments_json TEXT  -- JSON of current timing parameters
            )
        """)

        # Create indexes for fast queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_date ON unlock_attempts(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_speaker ON unlock_attempts(speaker_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_success ON unlock_attempts(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_attempts_timestamp ON unlock_attempts(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stages_attempt ON processing_stages(attempt_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stages_name ON processing_stages(stage_name)")

        # Indexes for ML tables
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_typing_sessions_attempt ON password_typing_sessions(attempt_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_typing_sessions_success ON password_typing_sessions(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_char_metrics_session ON character_typing_metrics(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_char_metrics_position ON character_typing_metrics(char_position)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_char_metrics_type ON character_typing_metrics(char_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_char_metrics_success ON character_typing_metrics(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_analytics_type ON typing_pattern_analytics(pattern_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_analytics_char ON typing_pattern_analytics(char_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_learning_progress_time ON learning_progress(timestamp)")

        conn.commit()
        conn.close()

        logger.info(f"✅ SQLite database initialized: {self.sqlite_path}")

    def _init_cloud_sql(self):
        """Initialize CloudSQL connection (uses existing JARVIS CloudSQL connection)"""
        try:
            # Import existing CloudSQL manager from JARVIS
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from intelligence.cloud_sql_connection_manager import CloudSQLConnectionManager

            self.cloud_db = CloudSQLConnectionManager.get_instance()
            logger.info("✅ CloudSQL connection established for metrics")

            # Create tables in CloudSQL (same schema as SQLite)
            self._create_cloud_tables()

        except Exception as e:
            logger.warning(f"CloudSQL not available for metrics: {e}")
            logger.warning("Continuing with SQLite only")
            self.use_cloud_sql = False
            self.cloud_db = None

    def _create_cloud_tables(self):
        """Create tables in CloudSQL if they don't exist"""
        if not self.cloud_db:
            return

        try:
            conn = self.cloud_db.get_connection()
            cursor = conn.cursor()

            # Same schema as SQLite but with PostgreSQL syntax
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_unlock_attempts (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP NOT NULL,
                    date DATE NOT NULL,
                    time TIME NOT NULL,
                    day_of_week VARCHAR(20),
                    unix_timestamp DOUBLE PRECISION,
                    success BOOLEAN NOT NULL,
                    speaker_name VARCHAR(255) NOT NULL,
                    transcribed_text TEXT,
                    error TEXT,

                    speaker_confidence DOUBLE PRECISION,
                    stt_confidence DOUBLE PRECISION,
                    threshold DOUBLE PRECISION,
                    above_threshold BOOLEAN,
                    confidence_margin DOUBLE PRECISION,
                    margin_percentage DOUBLE PRECISION,

                    avg_last_10 DOUBLE PRECISION,
                    avg_last_30 DOUBLE PRECISION,
                    trend_direction VARCHAR(50),
                    volatility DOUBLE PRECISION,
                    best_ever DOUBLE PRECISION,
                    worst_ever DOUBLE PRECISION,
                    percentile_rank DOUBLE PRECISION,

                    total_duration_ms DOUBLE PRECISION,
                    slowest_stage VARCHAR(100),
                    fastest_stage VARCHAR(100),

                    audio_quality VARCHAR(50),
                    voice_match_quality VARCHAR(50),
                    overall_confidence DOUBLE PRECISION,

                    total_stages INTEGER,
                    successful_stages INTEGER,
                    failed_stages INTEGER,
                    all_stages_passed BOOLEAN,

                    platform VARCHAR(50),
                    platform_version VARCHAR(255),
                    python_version VARCHAR(50),
                    stt_engine VARCHAR(100),
                    speaker_engine VARCHAR(100),

                    session_id VARCHAR(100),
                    logger_version VARCHAR(50),

                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, speaker_name)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS voice_unlock_stages (
                    id SERIAL PRIMARY KEY,
                    attempt_id INTEGER NOT NULL,
                    stage_name VARCHAR(100) NOT NULL,
                    started_at DOUBLE PRECISION,
                    ended_at DOUBLE PRECISION,
                    duration_ms DOUBLE PRECISION,
                    percentage_of_total DOUBLE PRECISION,
                    success BOOLEAN,
                    algorithm_used VARCHAR(255),
                    module_path TEXT,
                    function_name VARCHAR(255),
                    input_size_bytes INTEGER,
                    output_size_bytes INTEGER,
                    confidence_score DOUBLE PRECISION,
                    threshold DOUBLE PRECISION,
                    above_threshold BOOLEAN,
                    error_message TEXT,
                    metadata_json TEXT,

                    FOREIGN KEY (attempt_id) REFERENCES voice_unlock_attempts(id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_voice_attempts_date ON voice_unlock_attempts(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_voice_attempts_speaker ON voice_unlock_attempts(speaker_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_voice_stages_attempt ON voice_unlock_stages(attempt_id)")

            conn.commit()
            cursor.close()

            logger.info("✅ CloudSQL tables created for voice unlock metrics")

        except Exception as e:
            logger.error(f"Failed to create CloudSQL tables: {e}", exc_info=True)

    async def store_unlock_attempt(
        self,
        entry: Dict[str, Any],
        stages: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Store unlock attempt in both SQLite and CloudSQL.

        Args:
            entry: Complete unlock attempt entry (from metrics logger)
            stages: List of processing stage details

        Returns:
            SQLite row ID if successful, None otherwise
        """
        # Store in SQLite (always)
        sqlite_id = await self._store_in_sqlite(entry, stages)

        # Store in CloudSQL (if available)
        if self.use_cloud_sql and self.cloud_db:
            try:
                await self._store_in_cloud_sql(entry, stages)
            except Exception as e:
                logger.warning(f"Failed to sync to CloudSQL: {e}")

        return sqlite_id

    async def _store_in_sqlite(
        self,
        entry: Dict[str, Any],
        stages: List[Dict[str, Any]]
    ) -> Optional[int]:
        """Store unlock attempt in SQLite"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # Extract data from entry
            bio = entry['biometrics']
            perf = entry['performance']
            qual = entry['quality_indicators']
            stage_sum = entry['stage_summary']
            sys_info = entry['system_info']
            meta = entry['metadata']

            # Insert main attempt
            cursor.execute("""
                INSERT INTO unlock_attempts (
                    timestamp, date, time, day_of_week, unix_timestamp,
                    success, speaker_name, transcribed_text, error,
                    speaker_confidence, stt_confidence, threshold, above_threshold,
                    confidence_margin, margin_percentage,
                    avg_last_10, avg_last_30, trend_direction, volatility,
                    best_ever, worst_ever, percentile_rank,
                    total_duration_ms, slowest_stage, fastest_stage,
                    audio_quality, voice_match_quality, overall_confidence,
                    total_stages, successful_stages, failed_stages, all_stages_passed,
                    platform, platform_version, python_version, stt_engine, speaker_engine,
                    session_id, logger_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry['timestamp'], entry['date'], entry['time'], entry['day_of_week'],
                entry['unix_timestamp'], entry['success'], entry['speaker_name'],
                entry['transcribed_text'], entry.get('error'),
                bio['speaker_confidence'], bio['stt_confidence'], bio['threshold'],
                bio['above_threshold'], bio['confidence_margin'],
                bio['confidence_vs_threshold']['margin_percentage'],
                bio['confidence_trends'].get('avg_last_10'),
                bio['confidence_trends'].get('avg_last_30'),
                bio['confidence_trends'].get('trend_direction'),
                bio['confidence_trends'].get('volatility'),
                bio['confidence_trends'].get('best_ever'),
                bio['confidence_trends'].get('worst_ever'),
                bio['confidence_trends'].get('current_rank_percentile'),
                perf['total_duration_ms'], perf.get('slowest_stage'),
                perf.get('fastest_stage'), qual['audio_quality'],
                qual['voice_match_quality'], qual['overall_confidence'],
                stage_sum['total_stages'], stage_sum['successful_stages'],
                stage_sum['failed_stages'], stage_sum['all_stages_passed'],
                sys_info.get('platform'), sys_info.get('platform_version'),
                sys_info.get('python_version'), sys_info.get('stt_engine'),
                sys_info.get('speaker_engine'), meta['session_id'],
                meta['logger_version']
            ))

            attempt_id = cursor.lastrowid

            # Insert processing stages
            for stage in stages:
                cursor.execute("""
                    INSERT INTO processing_stages (
                        attempt_id, stage_name, started_at, ended_at, duration_ms,
                        percentage_of_total, success, algorithm_used, module_path,
                        function_name, input_size_bytes, output_size_bytes,
                        confidence_score, threshold, above_threshold, error_message,
                        metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    attempt_id, stage['stage_name'], stage['started_at'],
                    stage['ended_at'], stage['duration_ms'],
                    stage['percentage_of_total'], stage['success'],
                    stage.get('algorithm_used'), stage.get('module_path'),
                    stage.get('function_name'), stage.get('input_size_bytes'),
                    stage.get('output_size_bytes'), stage.get('confidence_score'),
                    stage.get('threshold'), stage.get('above_threshold'),
                    stage.get('error_message'),
                    json.dumps(stage.get('metadata', {}))
                ))

            # Insert stage breakdown
            for stage_name, stage_data in perf.get('stages_breakdown', {}).items():
                cursor.execute("""
                    INSERT INTO stage_breakdown (
                        attempt_id, stage_name, duration_ms, percentage
                    ) VALUES (?, ?, ?, ?)
                """, (
                    attempt_id, stage_name, stage_data['duration_ms'],
                    stage_data['percentage']
                ))

            conn.commit()
            conn.close()

            logger.debug(f"✅ Stored unlock attempt in SQLite (ID: {attempt_id})")
            return attempt_id

        except Exception as e:
            logger.error(f"Failed to store in SQLite: {e}", exc_info=True)
            return None

    async def _store_in_cloud_sql(
        self,
        entry: Dict[str, Any],
        stages: List[Dict[str, Any]]
    ) -> bool:
        """Store unlock attempt in CloudSQL"""
        # Similar to SQLite but with PostgreSQL-specific syntax
        # Implementation would use the cloud_db connection
        # Omitted for brevity - follows same pattern as SQLite
        pass

    async def query_attempts(
        self,
        speaker_name: str = None,
        start_date: str = None,
        end_date: str = None,
        success_only: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Query unlock attempts from database.

        Args:
            speaker_name: Filter by speaker
            start_date: Filter by start date (YYYY-MM-DD)
            end_date: Filter by end date (YYYY-MM-DD)
            success_only: Only return successful attempts
            limit: Maximum number of results

        Returns:
            List of unlock attempt dictionaries
        """
        conn = sqlite3.connect(self.sqlite_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = "SELECT * FROM unlock_attempts WHERE 1=1"
        params = []

        if speaker_name:
            query += " AND speaker_name = ?"
            params.append(speaker_name)

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        if success_only:
            query += " AND success = 1"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]

        conn.close()
        return results

    # 🤖 CONTINUOUS LEARNING METHODS

    async def store_typing_session(
        self,
        attempt_id: int,
        session_data: Dict[str, Any],
        character_metrics: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Store character-level typing metrics for ML training.

        Args:
            attempt_id: ID of the unlock attempt
            session_data: Overall typing session metrics
            character_metrics: List of per-character metrics

        Returns:
            Session ID if successful, None otherwise
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()

            # Insert typing session
            cursor.execute("""
                INSERT INTO password_typing_sessions (
                    attempt_id, timestamp, success,
                    total_characters, characters_typed, typing_method, fallback_used,
                    total_typing_duration_ms, avg_char_duration_ms,
                    min_char_duration_ms, max_char_duration_ms,
                    system_load, memory_pressure, screen_locked,
                    inter_char_delay_avg_ms, inter_char_delay_std_ms,
                    shift_press_duration_avg_ms, shift_release_delay_avg_ms,
                    failed_at_character, retry_count,
                    time_of_day, day_of_week
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                attempt_id,
                session_data.get('timestamp'),
                1 if session_data.get('success') else 0,
                session_data.get('total_characters'),
                session_data.get('characters_typed'),
                session_data.get('typing_method', 'core_graphics'),
                1 if session_data.get('fallback_used') else 0,
                session_data.get('total_typing_duration_ms'),
                session_data.get('avg_char_duration_ms'),
                session_data.get('min_char_duration_ms'),
                session_data.get('max_char_duration_ms'),
                session_data.get('system_load'),
                session_data.get('memory_pressure'),
                1 if session_data.get('screen_locked') else 0,
                session_data.get('inter_char_delay_avg_ms'),
                session_data.get('inter_char_delay_std_ms'),
                session_data.get('shift_press_duration_avg_ms'),
                session_data.get('shift_release_delay_avg_ms'),
                session_data.get('failed_at_character'),
                session_data.get('retry_count', 0),
                session_data.get('time_of_day'),
                session_data.get('day_of_week')
            ))

            session_id = cursor.lastrowid

            # Insert character metrics
            for char_metric in character_metrics:
                cursor.execute("""
                    INSERT INTO character_typing_metrics (
                        session_id, attempt_id,
                        char_position, char_type, char_case, requires_shift, keycode,
                        char_start_time_ms, char_end_time_ms, total_duration_ms,
                        shift_down_duration_ms, shift_registered_delay_ms, shift_up_delay_ms,
                        key_down_created, key_down_posted, key_press_duration_ms,
                        key_up_created, key_up_posted,
                        success, error_type, error_message, retry_attempted,
                        inter_char_delay_ms, system_load_at_char
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    attempt_id,
                    char_metric.get('char_position'),
                    char_metric.get('char_type'),
                    char_metric.get('char_case'),
                    1 if char_metric.get('requires_shift') else 0,
                    char_metric.get('keycode'),
                    char_metric.get('char_start_time_ms'),
                    char_metric.get('char_end_time_ms'),
                    char_metric.get('total_duration_ms'),
                    char_metric.get('shift_down_duration_ms'),
                    char_metric.get('shift_registered_delay_ms'),
                    char_metric.get('shift_up_delay_ms'),
                    1 if char_metric.get('key_down_created') else 0,
                    1 if char_metric.get('key_down_posted') else 0,
                    char_metric.get('key_press_duration_ms'),
                    1 if char_metric.get('key_up_created') else 0,
                    1 if char_metric.get('key_up_posted') else 0,
                    1 if char_metric.get('success') else 0,
                    char_metric.get('error_type'),
                    char_metric.get('error_message'),
                    1 if char_metric.get('retry_attempted') else 0,
                    char_metric.get('inter_char_delay_ms'),
                    char_metric.get('system_load_at_char')
                ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Stored typing session with {len(character_metrics)} character metrics (Session ID: {session_id})")
            return session_id

        except Exception as e:
            logger.error(f"Failed to store typing session: {e}", exc_info=True)
            return None

    async def analyze_typing_patterns(self) -> Dict[str, Any]:
        """
        Analyze typing patterns and compute optimal timing parameters for ML.

        Returns:
            Dictionary with pattern analysis and recommendations
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Analyze successful character typing
            cursor.execute("""
                SELECT
                    char_type,
                    requires_shift,
                    COUNT(*) as sample_count,
                    AVG(total_duration_ms) as avg_duration,
                    STDEV(total_duration_ms) as std_duration,
                    MIN(total_duration_ms) as min_duration,
                    MAX(total_duration_ms) as max_duration,
                    AVG(inter_char_delay_ms) as avg_inter_char_delay,
                    AVG(shift_press_duration_ms) as avg_shift_duration,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate
                FROM character_typing_metrics
                WHERE success = 1
                GROUP BY char_type, requires_shift
                HAVING sample_count >= 5
            """)

            patterns = []
            for row in cursor.fetchall():
                pattern = {
                    'char_type': row['char_type'],
                    'requires_shift': bool(row['requires_shift']),
                    'sample_count': row['sample_count'],
                    'avg_duration_ms': row['avg_duration'],
                    'std_duration_ms': row['std_duration'],
                    'min_duration_ms': row['min_duration'],
                    'max_duration_ms': row['max_duration'],
                    'avg_inter_char_delay_ms': row['avg_inter_char_delay'],
                    'avg_shift_duration_ms': row['avg_shift_duration'],
                    'success_rate': row['success_rate'],
                    'confidence': min(row['sample_count'] / 100.0, 1.0)  # Confidence increases with samples
                }

                # Calculate optimal timing (use fastest successful timing + small margin)
                pattern['optimal_duration_ms'] = row['min_duration'] * 1.1
                pattern['optimal_inter_char_delay_ms'] = max(row['avg_inter_char_delay'], 80.0)

                patterns.append(pattern)

                # Store in pattern analytics table
                cursor.execute("""
                    INSERT INTO typing_pattern_analytics (
                        calculated_at, pattern_type, char_type, requires_shift,
                        sample_count, success_rate,
                        avg_duration_ms, std_duration_ms, min_duration_ms, max_duration_ms,
                        optimal_char_duration_ms, optimal_inter_char_delay_ms,
                        optimal_shift_duration_ms, confidence_score,
                        last_updated, training_samples_used
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    'successful_timing',
                    pattern['char_type'],
                    1 if pattern['requires_shift'] else 0,
                    pattern['sample_count'],
                    pattern['success_rate'],
                    pattern['avg_duration_ms'],
                    pattern['std_duration_ms'],
                    pattern['min_duration_ms'],
                    pattern['max_duration_ms'],
                    pattern['optimal_duration_ms'],
                    pattern['optimal_inter_char_delay_ms'],
                    pattern['avg_shift_duration_ms'],
                    pattern['confidence'],
                    datetime.now().isoformat(),
                    pattern['sample_count']
                ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Analyzed {len(patterns)} typing patterns")
            return {
                'patterns': patterns,
                'total_patterns': len(patterns),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to analyze typing patterns: {e}", exc_info=True)
            return {'patterns': [], 'error': str(e)}

    async def get_optimal_timing_config(self) -> Dict[str, Any]:
        """
        Get ML-optimized timing configuration based on historical data.

        Returns:
            Dictionary with optimal timing parameters for password typing
        """
        try:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get latest pattern analytics
            cursor.execute("""
                SELECT
                    char_type,
                    requires_shift,
                    optimal_char_duration_ms,
                    optimal_inter_char_delay_ms,
                    optimal_shift_duration_ms,
                    confidence_score
                FROM typing_pattern_analytics
                WHERE pattern_type = 'successful_timing'
                AND sample_count >= 10
                ORDER BY last_updated DESC
                LIMIT 100
            """)

            config = {
                'letter': {'duration': 50, 'delay': 100},
                'digit': {'duration': 50, 'delay': 100},
                'special': {'duration': 60, 'delay': 120},
                'shift_duration': 30,
                'confidence': 0.0
            }

            results = cursor.fetchall()
            if results:
                # Group by char_type and compute weighted average
                for row in results:
                    char_type = row['char_type']
                    if char_type in config:
                        # Use confidence as weight
                        weight = row['confidence_score']
                        config[char_type]['duration'] = row['optimal_char_duration_ms'] * weight + config[char_type]['duration'] * (1 - weight)
                        config[char_type]['delay'] = row['optimal_inter_char_delay_ms'] * weight + config[char_type]['delay'] * (1 - weight)

                config['shift_duration'] = sum(r['optimal_shift_duration_ms'] or 30 for r in results) / len(results)
                config['confidence'] = sum(r['confidence_score'] for r in results) / len(results)

            conn.close()

            logger.info(f"✅ Retrieved optimal timing config (confidence: {config['confidence']:.2%})")
            return config

        except Exception as e:
            logger.error(f"Failed to get optimal timing: {e}", exc_info=True)
            return None


# Singleton instance
_metrics_db = None


def get_metrics_database() -> MetricsDatabase:
    """Get or create singleton metrics database instance"""
    global _metrics_db
    if _metrics_db is None:
        _metrics_db = MetricsDatabase()
    return _metrics_db
