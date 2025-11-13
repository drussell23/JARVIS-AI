#!/usr/bin/env python3
"""
Voice Unlock Monitoring Service
Real-time monitoring and detailed logging for voice unlock operations
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class UnlockAttempt:
    """Single unlock attempt record"""
    timestamp: str
    success: bool
    confidence: float
    transcription: str
    speaker_name: Optional[str]
    error_message: Optional[str]
    duration_ms: float
    password_typed: bool
    password_success: Optional[bool]


@dataclass
class PasswordTypingMetrics:
    """Password typing performance metrics"""
    timestamp: str
    characters_typed: int
    keystrokes_sent: int
    typing_time_ms: float
    wake_time_ms: float
    submit_time_ms: float
    total_duration_ms: float
    retries: int
    fallback_used: bool
    success: bool
    error_message: Optional[str]


class VoiceUnlockMonitor:
    """
    Real-time monitoring service for voice unlock operations

    Features:
    - Tracks unlock attempts with detailed metrics
    - Monitors password typing performance
    - Provides real-time status and diagnostics
    - Maintains rolling history for analysis
    - Exports metrics for display
    """

    def __init__(self, history_size: int = 100):
        """
        Initialize voice unlock monitor

        Args:
            history_size: Number of attempts to keep in rolling history
        """
        self.history_size = history_size

        # Unlock attempt tracking
        self.unlock_attempts: deque = deque(maxlen=history_size)
        self.password_attempts: deque = deque(maxlen=history_size)

        # Real-time metrics
        self.total_attempts = 0
        self.successful_unlocks = 0
        self.failed_unlocks = 0

        # Current attempt tracking
        self.current_attempt_start: Optional[float] = None
        self.current_attempt_data: Dict = {}

        # Performance tracking
        self.avg_unlock_time_ms = 0.0
        self.avg_confidence = 0.0
        self.avg_typing_time_ms = 0.0

        # Monitoring task
        self.monitoring_task: Optional[asyncio.Task] = None
        self.running = False
        self.monitoring_interval = 10  # seconds

        logger.info("🔐 Voice Unlock Monitor initialized")

    async def start(self):
        """Start monitoring service"""
        if self.running:
            logger.warning("Voice unlock monitor already running")
            return

        self.running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("🚀 Voice unlock monitoring started")
        logger.info(f"   Monitoring interval: {self.monitoring_interval}s")
        logger.info(f"   History size: {self.history_size} attempts")

    async def stop(self):
        """Stop monitoring service"""
        self.running = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Voice unlock monitoring stopped")

    async def _monitoring_loop(self):
        """Continuous monitoring loop"""
        while self.running:
            try:
                # Log periodic summary
                if self.total_attempts > 0:
                    success_rate = (self.successful_unlocks / self.total_attempts) * 100

                    logger.info("📊 Voice Unlock Status:")
                    logger.info(f"   Total Attempts: {self.total_attempts}")
                    logger.info(f"   Success Rate: {success_rate:.1f}%")
                    logger.info(f"   Avg Confidence: {self.avg_confidence:.1f}%")
                    logger.info(f"   Avg Unlock Time: {self.avg_unlock_time_ms:.0f}ms")
                    logger.info(f"   Avg Typing Time: {self.avg_typing_time_ms:.0f}ms")

                    # Recent failures
                    recent_failures = [a for a in list(self.unlock_attempts)[-10:] if not a.success]
                    if recent_failures:
                        logger.warning(f"   Recent failures: {len(recent_failures)}/10")
                        for failure in recent_failures[-3:]:
                            logger.warning(f"      - {failure.timestamp}: {failure.error_message}")

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")

            await asyncio.sleep(self.monitoring_interval)

    def start_unlock_attempt(self):
        """Mark the start of an unlock attempt"""
        self.current_attempt_start = time.time()
        self.current_attempt_data = {
            'start_time': self.current_attempt_start,
            'timestamp': datetime.now().isoformat()
        }
        logger.debug("🔐 Unlock attempt started")

    def record_unlock_attempt(
        self,
        success: bool,
        confidence: float,
        transcription: str,
        speaker_name: Optional[str] = None,
        error_message: Optional[str] = None,
        password_typed: bool = False,
        password_success: Optional[bool] = None
    ):
        """
        Record an unlock attempt

        Args:
            success: Whether unlock succeeded
            confidence: Voice verification confidence
            transcription: Transcribed command
            speaker_name: Identified speaker name
            error_message: Error message if failed
            password_typed: Whether password typing was attempted
            password_success: Whether password entry succeeded
        """
        duration_ms = 0.0
        if self.current_attempt_start:
            duration_ms = (time.time() - self.current_attempt_start) * 1000

        attempt = UnlockAttempt(
            timestamp=datetime.now().isoformat(),
            success=success,
            confidence=confidence,
            transcription=transcription,
            speaker_name=speaker_name,
            error_message=error_message,
            duration_ms=duration_ms,
            password_typed=password_typed,
            password_success=password_success
        )

        self.unlock_attempts.append(attempt)
        self.total_attempts += 1

        if success:
            self.successful_unlocks += 1
            logger.info(f"✅ Unlock SUCCESS: {speaker_name} ({confidence:.1f}% confidence, {duration_ms:.0f}ms)")
        else:
            self.failed_unlocks += 1
            logger.warning(f"❌ Unlock FAILED: {error_message} ({confidence:.1f}% confidence, {duration_ms:.0f}ms)")
            logger.warning(f"   Transcription: '{transcription}'")
            if password_typed and password_success is False:
                logger.error(f"   Password typing FAILED")

        # Update averages
        self._update_averages()

        # Reset current attempt
        self.current_attempt_start = None
        self.current_attempt_data = {}

    def record_password_typing(
        self,
        characters_typed: int,
        keystrokes_sent: int,
        typing_time_ms: float,
        wake_time_ms: float,
        submit_time_ms: float,
        total_duration_ms: float,
        retries: int,
        fallback_used: bool,
        success: bool,
        error_message: Optional[str] = None
    ):
        """
        Record password typing metrics

        Args:
            characters_typed: Number of characters in password
            keystrokes_sent: Total keystrokes sent (including shift, etc)
            typing_time_ms: Time spent typing password
            wake_time_ms: Time spent waking screen
            submit_time_ms: Time spent submitting
            total_duration_ms: Total typing operation duration
            retries: Number of retries attempted
            fallback_used: Whether AppleScript fallback was used
            success: Whether typing succeeded
            error_message: Error message if failed
        """
        metrics = PasswordTypingMetrics(
            timestamp=datetime.now().isoformat(),
            characters_typed=characters_typed,
            keystrokes_sent=keystrokes_sent,
            typing_time_ms=typing_time_ms,
            wake_time_ms=wake_time_ms,
            submit_time_ms=submit_time_ms,
            total_duration_ms=total_duration_ms,
            retries=retries,
            fallback_used=fallback_used,
            success=success,
            error_message=error_message
        )

        self.password_attempts.append(metrics)

        if success:
            logger.info(f"🔐 Password typed: {characters_typed} chars, {keystrokes_sent} keystrokes, {total_duration_ms:.0f}ms")
            logger.info(f"   Breakdown: wake={wake_time_ms:.0f}ms, type={typing_time_ms:.0f}ms, submit={submit_time_ms:.0f}ms")
            if retries > 0:
                logger.info(f"   Retries: {retries}")
            if fallback_used:
                logger.info(f"   Used AppleScript fallback")
        else:
            logger.error(f"❌ Password typing FAILED: {error_message}")
            logger.error(f"   Attempted {characters_typed} chars, {keystrokes_sent} keystrokes, {retries} retries")
            logger.error(f"   Duration: {total_duration_ms:.0f}ms")

        # Update averages
        self._update_averages()

    def _update_averages(self):
        """Update running averages"""
        if len(self.unlock_attempts) > 0:
            attempts = list(self.unlock_attempts)
            self.avg_confidence = sum(a.confidence for a in attempts) / len(attempts)
            self.avg_unlock_time_ms = sum(a.duration_ms for a in attempts) / len(attempts)

        if len(self.password_attempts) > 0:
            passwords = list(self.password_attempts)
            self.avg_typing_time_ms = sum(p.typing_time_ms for p in passwords) / len(passwords)

    def get_status(self) -> Dict:
        """
        Get current monitoring status

        Returns:
            Dictionary with current status and metrics
        """
        return {
            'running': self.running,
            'total_attempts': self.total_attempts,
            'successful_unlocks': self.successful_unlocks,
            'failed_unlocks': self.failed_unlocks,
            'success_rate': (self.successful_unlocks / self.total_attempts * 100) if self.total_attempts > 0 else 0.0,
            'avg_confidence': self.avg_confidence,
            'avg_unlock_time_ms': self.avg_unlock_time_ms,
            'avg_typing_time_ms': self.avg_typing_time_ms,
            'recent_attempts': [asdict(a) for a in list(self.unlock_attempts)[-10:]],
            'recent_password_attempts': [asdict(p) for p in list(self.password_attempts)[-10:]]
        }

    def get_recent_failures(self, count: int = 10) -> List[Dict]:
        """
        Get recent failure details

        Args:
            count: Number of recent failures to return

        Returns:
            List of recent failure dictionaries
        """
        failures = [a for a in self.unlock_attempts if not a.success]
        return [asdict(f) for f in list(failures)[-count:]]


# Global monitor instance
_voice_unlock_monitor: Optional[VoiceUnlockMonitor] = None


def get_voice_unlock_monitor() -> VoiceUnlockMonitor:
    """Get or create global voice unlock monitor instance"""
    global _voice_unlock_monitor
    if _voice_unlock_monitor is None:
        _voice_unlock_monitor = VoiceUnlockMonitor()
    return _voice_unlock_monitor
