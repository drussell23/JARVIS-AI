#!/usr/bin/env python3
"""
Advanced Async Unlock Metrics Logger
=====================================
Ultra-detailed, dynamic biometric and performance metrics logger for voice unlock.

Features:
- Fully async I/O for non-blocking operation
- Dynamic stage tracking with no hardcoding
- Per-stage timing, success/failure, and algorithm tracking
- Confidence evolution tracking over time
- File/module tracking for each stage
- Statistical analysis and trend detection
- Historical confidence comparison
- Automatic anomaly detection
"""

import json
import logging
import time
import inspect
import asyncio
import aiofiles
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from collections import defaultdict
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single processing stage"""
    stage_name: str
    started_at: float
    ended_at: Optional[float] = None
    duration_ms: Optional[float] = None
    success: Optional[bool] = None
    algorithm_used: Optional[str] = None
    module_path: Optional[str] = None
    function_name: Optional[str] = None
    input_size_bytes: Optional[int] = None
    output_size_bytes: Optional[int] = None
    confidence_score: Optional[float] = None
    threshold: Optional[float] = None
    above_threshold: Optional[bool] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def complete(self, success: bool, **kwargs):
        """Mark stage as complete and calculate duration"""
        self.ended_at = time.time()
        self.duration_ms = (self.ended_at - self.started_at) * 1000
        self.success = success
        for key, value in kwargs.items():
            setattr(self, key, value)


class UnlockMetricsLogger:
    """
    Advanced async metrics logger with dynamic stage tracking.

    This logger automatically tracks:
    - All processing stages with precise timing
    - Algorithms and files used in each stage
    - Confidence scores vs thresholds over time
    - Success/failure patterns
    - Performance trends
    """

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the advanced metrics logger.

        Args:
            log_dir: Directory to store metrics logs. Defaults to ~/.jarvis/logs/unlock_metrics/
        """
        if log_dir is None:
            home_dir = Path.home()
            self.log_dir = home_dir / ".jarvis" / "logs" / "unlock_metrics"
        else:
            self.log_dir = Path(log_dir)

        # Create directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create daily log file
        self.log_file = self._get_daily_log_file()

        # Stats file for aggregated metrics
        self.stats_file = self.log_dir / "unlock_stats.json" # Store stats across sessions and speakers

        # Historical trends file
        self.trends_file = self.log_dir / "confidence_trends.json" # Store confidence trends

        logger.info(f"📊 Advanced Async UnlockMetricsLogger initialized")
        logger.info(f"   └─ Log file: {self.log_file}")

    def _get_daily_log_file(self) -> Path:
        """Get the log file for today's date."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"unlock_metrics_{today}.json"

    def create_stage(self, stage_name: str, **metadata) -> StageMetrics:
        """
        Create and start tracking a new processing stage.

        Args:
            stage_name: Name of the stage (e.g., "audio_preparation", "transcription")
            **metadata: Additional metadata for the stage

        Returns:
            StageMetrics object that can be completed later
        """
        # Auto-detect caller information
        frame = inspect.currentframe().f_back
        module_path = frame.f_code.co_filename
        function_name = frame.f_code.co_name

        stage = StageMetrics(
            stage_name=stage_name,
            started_at=time.time(),
            module_path=module_path,
            function_name=function_name,
            metadata=metadata
        )

        logger.debug(f"📊 Stage started: {stage_name} in {function_name}")
        return stage

    async def log_unlock_attempt(
        self,
        success: bool,
        speaker_name: str,
        transcribed_text: str,
        stages: List[StageMetrics],
        biometrics: Dict[str, Any],
        performance: Dict[str, Any],
        quality_indicators: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Async log a complete voice unlock attempt with all stages and metrics.

        Args:
            success: Whether unlock succeeded
            speaker_name: Identified speaker name
            transcribed_text: What was transcribed
            stages: List of StageMetrics objects for each processing stage
            biometrics: Biometric confidence metrics
            performance: Performance timing metrics
            quality_indicators: Audio/voice quality metrics
            system_info: System/environment information
            error: Error message if failed
        """
        try:
            # Calculate total duration from stages
            total_duration_ms = sum(s.duration_ms for s in stages if s.duration_ms)

            # Build stage summary with detailed metrics
            stage_details = []
            for stage in stages:
                stage_data = asdict(stage)
                # Calculate stage-specific stats
                stage_data['percentage_of_total'] = (
                    (stage.duration_ms / total_duration_ms * 100)
                    if stage.duration_ms and total_duration_ms > 0
                    else 0
                )
                stage_details.append(stage_data)

            # Get confidence history for trend analysis
            confidence_history = await self._get_confidence_history(speaker_name) # List of past confidence scores for this speaker across attempts
            current_confidence = biometrics.get('speaker_confidence', 0)
            threshold = biometrics.get('threshold', 0.35)

            # Calculate confidence trends
            confidence_trend = self._calculate_confidence_trend(
                confidence_history,
                current_confidence
            )

            # Create comprehensive log entry
            entry = {
                # Timestamp information
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],  # millisecond precision
                "day_of_week": datetime.now().strftime("%A"),
                "unix_timestamp": time.time(),

                # Attempt outcome
                "success": success,
                "speaker_name": speaker_name,
                "transcribed_text": transcribed_text,
                "error": error,

                # Biometric metrics with trend analysis
                "biometrics": {
                    **biometrics,
                    "confidence_vs_threshold": {
                        "current_confidence": current_confidence,
                        "threshold": threshold,
                        "margin": current_confidence - threshold,
                        "margin_percentage": ((current_confidence - threshold) / threshold * 100) if threshold > 0 else 0,
                        "above_threshold": current_confidence >= threshold,
                    },
                    "confidence_trends": {
                        "avg_last_10": confidence_trend.get('avg_last_10'),
                        "avg_last_30": confidence_trend.get('avg_last_30'),
                        "trend_direction": confidence_trend.get('direction'),  # "improving", "declining", "stable"
                        "volatility": confidence_trend.get('volatility'),  # standard deviation
                        "best_ever": confidence_trend.get('best'),
                        "worst_ever": confidence_trend.get('worst'),
                        "current_rank_percentile": confidence_trend.get('percentile'),
                    }
                },

                # Performance metrics
                "performance": {
                    **performance,
                    "total_duration_ms": total_duration_ms,
                    "stages_breakdown": {
                        stage['stage_name']: {
                            "duration_ms": stage['duration_ms'],
                            "percentage": stage['percentage_of_total']
                        }
                        for stage in stage_details
                        if stage['duration_ms']
                    },
                    "slowest_stage": max(stage_details, key=lambda s: s['duration_ms'] or 0)['stage_name'] if stage_details else None,
                    "fastest_stage": min(stage_details, key=lambda s: s['duration_ms'] or float('inf'))['stage_name'] if stage_details else None,
                },

                # Quality indicators
                "quality_indicators": quality_indicators,

                # Detailed stage information
                "processing_stages": stage_details,

                # Stage success summary
                "stage_summary": {
                    "total_stages": len(stages),
                    "successful_stages": sum(1 for s in stages if s.success),
                    "failed_stages": sum(1 for s in stages if s.success is False),
                    "stages_above_threshold": sum(1 for s in stages if s.above_threshold),
                    "all_stages_passed": all(s.success for s in stages if s.success is not None),
                },

                # System information
                "system_info": system_info or {},

                # Metadata
                "metadata": {
                    "total_attempts_today": await self._count_todays_attempts() + 1, # Including this attempt being logged now
                    "session_id": self._get_session_id(),
                    "logger_version": "2.0.0-advanced-async", # Version of this advanced logger module (advanced-async)
                }
            }

            # Load existing entries
            entries = await self._load_entries() # List of all unlock attempts so far today

            # Add new entry
            entries.append(entry)

            # Save back to file
            await self._save_entries(entries) # Overwrite today's log file with updated entries 

            # Update confidence trends
            await self._update_confidence_trends(speaker_name, current_confidence, success)

            # Update aggregated stats
            await self._update_stats(entry) # Update overall stats file with this new entry 

            logger.info(f"✅ Logged unlock attempt: success={success}, speaker={speaker_name}")
            logger.info(f"   └─ Confidence: {current_confidence:.2%} (threshold: {threshold:.2%}, margin: {(current_confidence - threshold):.2%})")
            logger.info(f"   └─ Total duration: {total_duration_ms:.0f}ms across {len(stages)} stages")

            # Log detailed stage breakdown
            for stage in stage_details: # Each stage with detailed metrics 
                if stage['duration_ms']: # Only log stages with recorded duration
                    status_emoji = "✅" if stage['success'] else "❌" # Success or failure emoji 
                    logger.info(f"   └─ {status_emoji} {stage['stage_name']}: {stage['duration_ms']:.0f}ms ({stage['percentage_of_total']:.1f}%)") # Log each stage's duration and percentage of total duration 

        except Exception as e:
            logger.error(f"Failed to log unlock metrics: {e}", exc_info=True)

    # Helper methods for confidence trends and stats 
    async def _get_confidence_history(self, speaker_name: str) -> List[float]:
        """Get historical confidence scores for a speaker"""
        entries = await self._load_entries() # Load today's entries 
        history = [
            e['biometrics']['speaker_confidence']
            for e in entries
            if e.get('speaker_name') == speaker_name
            and e.get('biometrics', {}).get('speaker_confidence') is not None
        ]
        return history

    def _calculate_confidence_trend(
        self,
        history: List[float],
        current: float
    ) -> Dict[str, Any]:
        """Calculate confidence trends and statistics"""
        if not history:
            return {
                'avg_last_10': current,
                'avg_last_30': current,
                'direction': 'new',
                'volatility': 0,
                'best': current,
                'worst': current,
                'percentile': 100,
            }

        all_scores = history + [current]
        last_10 = all_scores[-10:]
        last_30 = all_scores[-30:]

        # Calculate averages
        avg_last_10 = sum(last_10) / len(last_10) if last_10 else 0
        avg_last_30 = sum(last_30) / len(last_30) if last_30 else 0

        # Determine trend direction
        if len(last_10) >= 5:
            recent_avg = sum(last_10[-5:]) / 5
            older_avg = sum(last_10[:5]) / 5
            if recent_avg > older_avg * 1.05:
                direction = "improving"
            elif recent_avg < older_avg * 0.95:
                direction = "declining"
            else:
                direction = "stable"
        else:
            direction = "insufficient_data"

        # Calculate volatility (standard deviation)
        if len(all_scores) > 1:
            mean = sum(all_scores) / len(all_scores)
            variance = sum((x - mean) ** 2 for x in all_scores) / len(all_scores)
            volatility = variance ** 0.5
        else:
            volatility = 0

        # Best/worst
        best = max(all_scores)
        worst = min(all_scores)

        # Percentile ranking
        better_count = sum(1 for score in history if score <= current)
        percentile = (better_count / len(history) * 100) if history else 100

        return {
            'avg_last_10': avg_last_10,
            'avg_last_30': avg_last_30,
            'direction': direction,
            'volatility': volatility,
            'best': best,
            'worst': worst,
            'percentile': percentile,
        }

    async def _update_confidence_trends(
        self,
        speaker_name: str,
        confidence: float,
        success: bool
    ) -> None:
        """Update the confidence trends file asynchronously"""
        try:
            if self.trends_file.exists():
                async with aiofiles.open(self.trends_file, 'r') as f: # Read existing trends file 
                    content = await f.read() # Read file content 
                    trends = json.loads(content) # Parse JSON content 
            else:
                trends = {}

            if speaker_name not in trends:
                trends[speaker_name] = {
                    'confidence_history': [],
                    'success_history': [],
                    'timestamps': [],
                }

            trends[speaker_name]['confidence_history'].append(confidence)
            trends[speaker_name]['success_history'].append(success)
            trends[speaker_name]['timestamps'].append(datetime.now().isoformat())

            # Keep only last 1000 entries per speaker
            for key in ['confidence_history', 'success_history', 'timestamps']:
                trends[speaker_name][key] = trends[speaker_name][key][-1000:]

            async with aiofiles.open(self.trends_file, 'w') as f: # Write updated trends file 
                await f.write(json.dumps(trends, indent=2)) # Write JSON content 

        except Exception as e:
            logger.warning(f"Failed to update confidence trends: {e}")

    async def _update_stats(self, entry: Dict[str, Any]) -> None:
        """Update aggregated statistics asynchronously"""
        try:
            if self.stats_file.exists(): # Check if stats file exists 
                async with aiofiles.open(self.stats_file, 'r') as f: # Read existing stats file 
                    content = await f.read() # Read file content 
                    stats = json.loads(content) # Parse JSON content 
            else: # If stats file doesn't exist, initialize new stats 
                stats = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'failed_attempts': 0,
                    'speakers': {},
                    'last_updated': None,
                }

            # Update totals
            stats['total_attempts'] += 1
            if entry['success']:
                stats['successful_attempts'] += 1
            else:
                stats['failed_attempts'] += 1

            # Update per-speaker stats
            speaker = entry['speaker_name']
            if speaker not in stats['speakers']:
                stats['speakers'][speaker] = {
                    'total_attempts': 0,
                    'successful_attempts': 0,
                    'avg_confidence': 0,
                    'best_confidence': 0,
                    'worst_confidence': 1.0,
                    'avg_duration_ms': 0,
                }

            speaker_stats = stats['speakers'][speaker]
            speaker_stats['total_attempts'] += 1
            if entry['success']:
                speaker_stats['successful_attempts'] += 1

            confidence = entry['biometrics']['speaker_confidence']
            speaker_stats['best_confidence'] = max(speaker_stats['best_confidence'], confidence)
            speaker_stats['worst_confidence'] = min(speaker_stats['worst_confidence'], confidence)

            # Rolling average
            n = speaker_stats['total_attempts']
            old_avg = speaker_stats['avg_confidence']
            speaker_stats['avg_confidence'] = (old_avg * (n - 1) + confidence) / n

            duration = entry['performance']['total_duration_ms']
            old_dur_avg = speaker_stats['avg_duration_ms']
            speaker_stats['avg_duration_ms'] = (old_dur_avg * (n - 1) + duration) / n

            stats['last_updated'] = datetime.now().isoformat()

            async with aiofiles.open(self.stats_file, 'w') as f:
                await f.write(json.dumps(stats, indent=2))

        except Exception as e:
            logger.warning(f"Failed to update stats: {e}")

    async def _count_todays_attempts(self) -> int:
        """Count attempts made today"""
        entries = await self._load_entries()
        return len(entries)

    def _get_session_id(self) -> str:
        """Get or create session ID for this run"""
        if not hasattr(self, '_session_id'):
            self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self._session_id

    async def _load_entries(self) -> list:
        """Load existing entries from today's log file asynchronously."""
        if self.log_file.exists():
            try:
                async with aiofiles.open(self.log_file, 'r') as f:
                    content = await f.read()
                    return json.loads(content)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {self.log_file}, starting fresh")
                return []
        return []

    async def _save_entries(self, entries: list) -> None:
        """Save entries to today's log file asynchronously."""
        async with aiofiles.open(self.log_file, 'w') as f:
            await f.write(json.dumps(entries, indent=2, default=str))

    async def get_today_stats(self) -> Dict[str, Any]:
        """Get statistics for today's unlock attempts."""
        entries = await self._load_entries()

        total_attempts = len(entries)
        successful = sum(1 for e in entries if e["success"])
        failed = total_attempts - successful

        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_attempts": total_attempts,
            "successful": successful,
            "failed": failed,
            "success_rate": (successful / total_attempts * 100) if total_attempts > 0 else 0,
            "avg_confidence": self._calculate_avg_confidence(entries),
            "avg_latency": self._calculate_avg_latency(entries),
            "confidence_trend": self._get_todays_trend(entries),
        }

    def _calculate_avg_confidence(self, entries: list) -> float:
        """Calculate average speaker confidence."""
        confidences = [
            e["biometrics"]["speaker_confidence"]
            for e in entries
            if e.get("biometrics", {}).get("speaker_confidence") is not None
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def _calculate_avg_latency(self, entries: list) -> float:
        """Calculate average total latency."""
        latencies = [
            e["performance"]["total_duration_ms"]
            for e in entries
            if e.get("performance", {}).get("total_duration_ms") is not None
        ]
        return sum(latencies) / len(latencies) if latencies else 0.0

    def _get_todays_trend(self, entries: list) -> str:
        """Determine if confidence is improving or declining today"""
        if len(entries) < 5:
            return "insufficient_data"

        confidences = [
            e["biometrics"]["speaker_confidence"]
            for e in entries
            if e.get("biometrics", {}).get("speaker_confidence") is not None
        ]

        if len(confidences) < 5:
            return "insufficient_data"

        first_half = confidences[:len(confidences)//2]
        second_half = confidences[len(confidences)//2:]

        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_second > avg_first * 1.05:
            return "improving"
        elif avg_second < avg_first * 0.95:
            return "declining"
        else:
            return "stable"


# Singleton instance
_metrics_logger = None


def get_metrics_logger() -> UnlockMetricsLogger:
    """Get or create singleton metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = UnlockMetricsLogger()
    return _metrics_logger
