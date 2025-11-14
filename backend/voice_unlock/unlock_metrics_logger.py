#!/usr/bin/env python3
"""
Unlock Metrics Logger
=====================
Logs detailed biometric and performance metrics for voice unlock attempts
to a JSON file for developer analysis.

Features:
- Timestamped entries with full date/time
- Detailed biometric confidence scores
- Processing stage metrics
- Performance timing
- Success/failure tracking
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)


class UnlockMetricsLogger:
    """Logs voice unlock metrics to JSON file for analysis"""

    def __init__(self, log_dir: Optional[str] = None):
        """
        Initialize the metrics logger.

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

        logger.info(f"📊 UnlockMetricsLogger initialized: {self.log_file}")

    def _get_daily_log_file(self) -> Path:
        """Get the log file for today's date."""
        today = datetime.now().strftime("%Y-%m-%d")
        return self.log_dir / f"unlock_metrics_{today}.json"

    def log_unlock_attempt(
        self,
        success: bool,
        speaker_name: str,
        transcribed_text: str,
        biometrics: Dict[str, Any],
        performance: Dict[str, Any],
        quality_indicators: Dict[str, Any],
        processing_stages: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log a voice unlock attempt with full metrics.

        Args:
            success: Whether unlock succeeded
            speaker_name: Identified speaker name
            transcribed_text: What was transcribed
            biometrics: Biometric confidence metrics
            performance: Performance timing metrics
            quality_indicators: Audio/voice quality metrics
            processing_stages: Detailed stage-by-stage metrics
            error: Error message if failed
        """
        try:
            # Create log entry
            entry = {
                "timestamp": datetime.now().isoformat(),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "time": datetime.now().strftime("%H:%M:%S"),
                "day_of_week": datetime.now().strftime("%A"),
                "success": success,
                "speaker_name": speaker_name,
                "transcribed_text": transcribed_text,
                "biometrics": biometrics,
                "performance": performance,
                "quality_indicators": quality_indicators,
                "processing_stages": processing_stages or {},
                "error": error
            }

            # Load existing entries
            entries = self._load_entries()

            # Add new entry
            entries.append(entry)

            # Save back to file
            self._save_entries(entries)

            logger.info(f"✅ Logged unlock attempt: success={success}, speaker={speaker_name}")

        except Exception as e:
            logger.error(f"Failed to log unlock metrics: {e}", exc_info=True)

    def _load_entries(self) -> list:
        """Load existing entries from today's log file."""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse {self.log_file}, starting fresh")
                return []
        return []

    def _save_entries(self, entries: list) -> None:
        """Save entries to today's log file."""
        with open(self.log_file, 'w') as f:
            json.dump(entries, f, indent=2, default=str)

    def get_today_stats(self) -> Dict[str, Any]:
        """Get statistics for today's unlock attempts."""
        entries = self._load_entries()

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
            "avg_latency": self._calculate_avg_latency(entries)
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
            e["performance"]["total_latency_ms"]
            for e in entries
            if e.get("performance", {}).get("total_latency_ms") is not None
        ]
        return sum(latencies) / len(latencies) if latencies else 0.0


# Singleton instance
_metrics_logger = None


def get_metrics_logger() -> UnlockMetricsLogger:
    """Get or create singleton metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = UnlockMetricsLogger()
    return _metrics_logger
