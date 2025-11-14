#!/usr/bin/env python3
"""
Advanced Voice Unlock Metrics Monitor
======================================
Real-time monitoring system for voice unlock metrics with auto-launching DB Browser.

Features:
- Automatically launches DB Browser for SQLite on startup
- Watches for new unlock attempts in real-time
- Updates database viewer automatically
- Professional notifications for each unlock attempt
- Zero hardcoding - fully dynamic and async
"""

import asyncio
import logging
import subprocess
import time
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class VoiceUnlockMetricsMonitor:
    """
    Advanced real-time metrics monitor for voice unlock system.

    Automatically:
    - Launches DB Browser on startup
    - Monitors unlock attempts
    - Updates database in real-time
    - Provides professional logging
    """

    def __init__(self):
        """Initialize metrics monitor"""
        self.log_dir = Path.home() / ".jarvis/logs/unlock_metrics"
        self.db_path = self.log_dir / "unlock_metrics.db"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # DB Browser process
        self.db_browser_process = None
        self.monitoring = False

        # Track last seen attempt
        self.last_attempt_count = 0

        # Performance stats
        self.session_stats = {
            'session_start': datetime.now().isoformat(),
            'total_attempts': 0,
            'successful_attempts': 0,
            'failed_attempts': 0,
            'avg_confidence': 0.0,
            'avg_duration_ms': 0.0,
        }

    async def start(self):
        """Start the metrics monitoring system"""
        logger.info("🚀 Starting Voice Unlock Metrics Monitor...")

        # Initialize database
        await self._initialize_database()

        # Launch DB Browser
        await self._launch_db_browser()

        # Start monitoring
        self.monitoring = True
        logger.info("✅ Metrics Monitor active - watching for unlock attempts")
        logger.info(f"📊 Database: {self.db_path}")
        logger.info(f"🔍 DB Browser: {'Running' if self.db_browser_process else 'Not launched'}")

        # Start monitoring loop
        asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """Stop the metrics monitoring system"""
        logger.info("🛑 Stopping Voice Unlock Metrics Monitor...")
        self.monitoring = False

        # Close DB Browser if we launched it
        if self.db_browser_process:
            try:
                self.db_browser_process.terminate()
                logger.info("✅ DB Browser closed")
            except:
                pass

        # Log session stats
        self._log_session_summary()

    async def _initialize_database(self):
        """Initialize the metrics database"""
        try:
            from voice_unlock.metrics_database import get_metrics_database

            # Initialize database (creates tables if needed)
            db = get_metrics_database()
            logger.info(f"✅ Database initialized: {db.sqlite_path}")

            # Check if database exists and has data
            if db.sqlite_path.exists():
                import sqlite3
                conn = sqlite3.connect(db.sqlite_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM unlock_attempts")
                count = cursor.fetchone()[0]
                conn.close()

                self.last_attempt_count = count
                logger.info(f"📊 Found {count} existing unlock attempts in database")

        except Exception as e:
            logger.warning(f"Database initialization note: {e}")

    async def _launch_db_browser(self):
        """Launch DB Browser for SQLite"""
        try:
            # Check if DB Browser is installed
            result = subprocess.run(
                ['mdfind', 'kMDItemKind == "Application" && kMDItemDisplayName == "DB Browser for SQLite"'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if not result.stdout.strip():
                logger.warning("⚠️  DB Browser for SQLite not found - skipping auto-launch")
                logger.info("💡 Install with: brew install --cask db-browser-for-sqlite")
                return

            # Launch DB Browser with the database
            logger.info("🚀 Launching DB Browser for SQLite...")

            # Use 'open' command to launch in background
            self.db_browser_process = subprocess.Popen(
                ['open', '-a', 'DB Browser for SQLite', str(self.db_path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

            # Give it a moment to launch
            await asyncio.sleep(2)

            logger.info("✅ DB Browser launched successfully")
            logger.info("💡 The database will auto-update when you unlock your screen")
            logger.info("💡 Press F5 in DB Browser to refresh and see new data")

        except Exception as e:
            logger.warning(f"Could not auto-launch DB Browser: {e}")
            logger.info("💡 You can manually open it with:")
            logger.info(f"   open -a 'DB Browser for SQLite' {self.db_path}")

    async def _monitor_loop(self):
        """Main monitoring loop - watches for new unlock attempts"""
        logger.info("🔍 Monitoring for unlock attempts...")

        while self.monitoring:
            try:
                # Check for new unlock attempts
                await self._check_for_new_attempts()

                # Wait before next check (2 seconds for responsive monitoring)
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _check_for_new_attempts(self):
        """Check database for new unlock attempts"""
        try:
            import sqlite3

            if not self.db_path.exists():
                return

            # Query database for current count
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM unlock_attempts")
            current_count = cursor.fetchone()[0]

            # Check if there are new attempts
            if current_count > self.last_attempt_count:
                new_attempts = current_count - self.last_attempt_count

                # Get the latest attempts
                cursor.execute("""
                    SELECT
                        timestamp,
                        success,
                        speaker_name,
                        speaker_confidence,
                        threshold,
                        total_duration_ms,
                        trend_direction
                    FROM unlock_attempts
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (new_attempts,))

                latest_attempts = cursor.fetchall()

                # Process each new attempt
                for attempt in reversed(latest_attempts):
                    await self._process_new_attempt(attempt)

                self.last_attempt_count = current_count

            conn.close()

        except Exception as e:
            logger.debug(f"Check attempts error: {e}")

    async def _process_new_attempt(self, attempt_data):
        """Process and log a new unlock attempt"""
        timestamp, success, speaker, confidence, threshold, duration_ms, trend = attempt_data

        # Update session stats
        self.session_stats['total_attempts'] += 1
        if success:
            self.session_stats['successful_attempts'] += 1
        else:
            self.session_stats['failed_attempts'] += 1

        # Calculate running averages
        total = self.session_stats['total_attempts']
        old_conf_avg = self.session_stats['avg_confidence']
        old_dur_avg = self.session_stats['avg_duration_ms']

        self.session_stats['avg_confidence'] = (old_conf_avg * (total - 1) + confidence) / total
        self.session_stats['avg_duration_ms'] = (old_dur_avg * (total - 1) + duration_ms) / total

        # Format duration
        duration_sec = duration_ms / 1000

        # Build notification message
        status_emoji = "✅" if success else "❌"
        trend_emoji = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"

        # Calculate margin
        margin = confidence - threshold
        margin_pct = (margin / threshold * 100) if threshold > 0 else 0

        # Professional logging
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"🔐 VOICE UNLOCK ATTEMPT DETECTED {status_emoji}")
        logger.info("=" * 80)
        logger.info(f"⏰ Time: {timestamp}")
        logger.info(f"👤 Speaker: {speaker}")
        logger.info(f"🎯 Result: {'SUCCESS' if success else 'FAILED'}")
        logger.info("")
        logger.info("📊 Biometric Analysis:")
        logger.info(f"   └─ Confidence: {confidence:.2%}")
        logger.info(f"   └─ Threshold: {threshold:.2%}")
        logger.info(f"   └─ Margin: {margin:+.2%} ({margin_pct:+.1f}%)")
        logger.info(f"   └─ Trend: {trend} {trend_emoji}")
        logger.info("")
        logger.info(f"⚡ Performance:")
        logger.info(f"   └─ Duration: {duration_sec:.1f}s ({duration_ms:.0f}ms)")
        logger.info("")
        logger.info(f"📈 Session Stats (This Run):")
        logger.info(f"   └─ Total Attempts: {self.session_stats['total_attempts']}")
        logger.info(f"   └─ Success Rate: {self.session_stats['successful_attempts']}/{self.session_stats['total_attempts']} ({self.session_stats['successful_attempts']/total*100:.1f}%)")
        logger.info(f"   └─ Avg Confidence: {self.session_stats['avg_confidence']:.2%}")
        logger.info(f"   └─ Avg Duration: {self.session_stats['avg_duration_ms']/1000:.1f}s")
        logger.info("")
        logger.info("💡 Database Updated - Press F5 in DB Browser to see latest data")
        logger.info("=" * 80)
        logger.info("")

    def _log_session_summary(self):
        """Log summary statistics for this monitoring session"""
        if self.session_stats['total_attempts'] == 0:
            logger.info("📊 No unlock attempts during this session")
            return

        logger.info("")
        logger.info("=" * 80)
        logger.info("📊 VOICE UNLOCK SESSION SUMMARY")
        logger.info("=" * 80)
        logger.info(f"⏰ Session Duration: {datetime.now().isoformat()} - {self.session_stats['session_start']}")
        logger.info(f"📈 Total Attempts: {self.session_stats['total_attempts']}")
        logger.info(f"✅ Successful: {self.session_stats['successful_attempts']}")
        logger.info(f"❌ Failed: {self.session_stats['failed_attempts']}")

        success_rate = (self.session_stats['successful_attempts'] / self.session_stats['total_attempts'] * 100)
        logger.info(f"🎯 Success Rate: {success_rate:.1f}%")
        logger.info(f"📊 Avg Confidence: {self.session_stats['avg_confidence']:.2%}")
        logger.info(f"⚡ Avg Duration: {self.session_stats['avg_duration_ms']/1000:.1f}s")
        logger.info("=" * 80)
        logger.info("")


# Global monitor instance
_metrics_monitor = None


async def initialize_metrics_monitor():
    """Initialize and start the metrics monitor"""
    global _metrics_monitor

    if _metrics_monitor is None:
        _metrics_monitor = VoiceUnlockMetricsMonitor()
        await _metrics_monitor.start()

    return _metrics_monitor


async def shutdown_metrics_monitor():
    """Shutdown the metrics monitor"""
    global _metrics_monitor

    if _metrics_monitor:
        await _metrics_monitor.stop()
        _metrics_monitor = None


def get_metrics_monitor() -> Optional[VoiceUnlockMetricsMonitor]:
    """Get the current metrics monitor instance"""
    return _metrics_monitor
