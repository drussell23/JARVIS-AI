#!/usr/bin/env python3
"""
WebSocket Streaming Safeguard
==============================

Detects commands during audio streaming and immediately closes the stream
to prevent audio accumulation (60+ second pileups).

Features:
- Real-time command detection during streaming
- Fuzzy command matching (handles variations like "unlock", "UNLOCK", "unlock screen")
- Configurable command triggers (wake phrases, unlock commands, etc.)
- Async, non-blocking detection
- WebSocket-aware (can close streams immediately)
- Event callbacks for command detection
- Metrics and monitoring

Architecture:
- Monitor transcription results in real-time
- When target command detected → immediately signal stream closure
- WebSocket handler respects closure signal and stops accepting audio
- Prevents 60s audio accumulation problem

Usage:
    # Create safeguard
    safeguard = StreamingSafeguard(
        target_commands=['unlock', 'lock', 'jarvis'],
        fuzzy_threshold=0.8
    )

    # Register callback
    safeguard.on_command_detected(lambda cmd: websocket.close())

    # Monitor transcription results
    await safeguard.check_transcription(transcription_text)

    # Check if stream should be closed
    if safeguard.should_close_stream():
        await websocket.close()
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class CommandMatchStrategy(Enum):
    """Strategy for matching commands in transcription text"""
    EXACT = "exact"  # Exact string match (case-insensitive)
    FUZZY = "fuzzy"  # Fuzzy match with similarity threshold
    REGEX = "regex"  # Regular expression matching
    CONTAINS = "contains"  # Check if command appears anywhere in text
    WORD_BOUNDARY = "word_boundary"  # Match as whole word only


@dataclass
class CommandDetectionConfig:
    """Configuration for command detection"""
    # Target commands to detect
    target_commands: List[str] = field(default_factory=lambda: [
        "unlock",
        "lock",
        "jarvis",
        "hey jarvis",
        "unlock my screen",
        "lock my screen",
    ])

    # Matching strategy
    match_strategy: CommandMatchStrategy = CommandMatchStrategy.WORD_BOUNDARY

    # Fuzzy matching threshold (0.0-1.0, used with FUZZY strategy)
    fuzzy_threshold: float = float(os.getenv('COMMAND_FUZZY_THRESHOLD', '0.8'))

    # Case sensitivity
    case_sensitive: bool = False

    # Strip punctuation before matching
    strip_punctuation: bool = True

    # Minimum confidence required for transcription (0.0-1.0)
    min_transcription_confidence: float = float(os.getenv('MIN_TRANSCRIPTION_CONFIDENCE', '0.5'))

    # Cooldown period between detections (seconds)
    detection_cooldown: float = float(os.getenv('COMMAND_DETECTION_COOLDOWN', '1.0'))

    # Enable command detection logging
    enable_logging: bool = os.getenv('ENABLE_COMMAND_DETECTION_LOGGING', 'true').lower() == 'true'


@dataclass
class DetectionEvent:
    """Command detection event details"""
    command: str  # Detected command
    matched_text: str  # Actual text that matched
    transcription: str  # Full transcription text
    confidence: float  # Match confidence (0.0-1.0)
    timestamp: float  # Detection timestamp
    strategy: CommandMatchStrategy  # Strategy used for matching
    metadata: Dict[str, Any] = field(default_factory=dict)


class StreamingSafeguard:
    """
    WebSocket Streaming Safeguard

    Monitors transcription results in real-time and signals stream closure
    when target commands are detected.

    This prevents audio accumulation by cutting off the stream immediately
    after a command (like "unlock") is recognized.
    """

    def __init__(self, config: Optional[CommandDetectionConfig] = None):
        """
        Initialize streaming safeguard

        Args:
            config: Detection configuration (uses defaults if None)
        """
        self.config = config or CommandDetectionConfig()

        # Detection state
        self._stream_should_close = False
        self._last_detection_time: Optional[float] = None
        self._detection_events: List[DetectionEvent] = []

        # Callbacks
        self._on_command_callbacks: List[Callable[[DetectionEvent], None]] = []
        self._on_stream_close_callbacks: List[Callable[[], None]] = []

        # Metrics
        self.metrics = {
            "total_checks": 0,
            "total_detections": 0,
            "commands_detected": {},  # command -> count
            "false_positives": 0,
            "avg_detection_time_ms": 0.0,
            "last_detection": None,
        }

        # Compile regex patterns if needed
        self._regex_patterns: Dict[str, re.Pattern] = {}
        if self.config.match_strategy == CommandMatchStrategy.REGEX:
            self._compile_regex_patterns()

        logger.info(
            f"🛡️ Streaming Safeguard initialized | "
            f"Strategy: {self.config.match_strategy.value} | "
            f"Commands: {len(self.config.target_commands)} | "
            f"Fuzzy threshold: {self.config.fuzzy_threshold}"
        )

    def _compile_regex_patterns(self):
        """Compile regex patterns for target commands"""
        for cmd in self.config.target_commands:
            try:
                flags = 0 if self.config.case_sensitive else re.IGNORECASE
                self._regex_patterns[cmd] = re.compile(cmd, flags)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{cmd}': {e}")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for matching

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        if not self.config.case_sensitive:
            text = text.lower()

        if self.config.strip_punctuation:
            # Remove punctuation but keep spaces
            text = re.sub(r'[^\w\s]', '', text)

        # Collapse multiple spaces
        text = ' '.join(text.split())

        return text

    def _fuzzy_match(self, text: str, command: str) -> float:
        """
        Calculate fuzzy match score using Levenshtein distance

        Args:
            text: Text to match against
            command: Command to find

        Returns:
            Similarity score (0.0-1.0)
        """
        # Simple character-based similarity
        # For production, consider using python-Levenshtein or rapidfuzz
        text = text.lower()
        command = command.lower()

        if command in text:
            return 1.0

        # Calculate simple similarity based on common characters
        common_chars = set(text) & set(command)
        max_len = max(len(text), len(command))

        if max_len == 0:
            return 0.0

        return len(common_chars) / max_len

    def _exact_match(self, text: str, command: str) -> bool:
        """Check for exact match"""
        return self._normalize_text(text) == self._normalize_text(command)

    def _contains_match(self, text: str, command: str) -> bool:
        """Check if command appears anywhere in text"""
        normalized_text = self._normalize_text(text)
        normalized_cmd = self._normalize_text(command)
        return normalized_cmd in normalized_text

    def _word_boundary_match(self, text: str, command: str) -> bool:
        """Check if command appears as whole word(s)"""
        normalized_text = self._normalize_text(text)
        normalized_cmd = self._normalize_text(command)

        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(normalized_cmd) + r'\b'
        return re.search(pattern, normalized_text) is not None

    def _regex_match(self, text: str, command: str) -> bool:
        """Check if regex pattern matches"""
        pattern = self._regex_patterns.get(command)
        if not pattern:
            return False

        return pattern.search(text) is not None

    def _check_match(self, text: str, command: str) -> Optional[float]:
        """
        Check if command matches text using configured strategy

        Args:
            text: Transcription text
            command: Command to match

        Returns:
            Match confidence (0.0-1.0) if matched, None otherwise
        """
        strategy = self.config.match_strategy

        if strategy == CommandMatchStrategy.EXACT:
            return 1.0 if self._exact_match(text, command) else None

        elif strategy == CommandMatchStrategy.FUZZY:
            score = self._fuzzy_match(text, command)
            return score if score >= self.config.fuzzy_threshold else None

        elif strategy == CommandMatchStrategy.REGEX:
            return 1.0 if self._regex_match(text, command) else None

        elif strategy == CommandMatchStrategy.CONTAINS:
            return 1.0 if self._contains_match(text, command) else None

        elif strategy == CommandMatchStrategy.WORD_BOUNDARY:
            return 1.0 if self._word_boundary_match(text, command) else None

        else:
            logger.warning(f"Unknown match strategy: {strategy}")
            return None

    async def check_transcription(
        self,
        transcription: str,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[DetectionEvent]:
        """
        Check transcription for target commands

        Args:
            transcription: Transcription text to check
            confidence: Transcription confidence (0.0-1.0)
            metadata: Additional metadata

        Returns:
            DetectionEvent if command detected, None otherwise
        """
        start_time = time.time()
        self.metrics["total_checks"] += 1

        # Validate confidence threshold
        if confidence is not None and confidence < self.config.min_transcription_confidence:
            logger.debug(
                f"Transcription confidence {confidence:.2f} below threshold "
                f"{self.config.min_transcription_confidence:.2f}, skipping"
            )
            return None

        # Check cooldown
        if self._last_detection_time is not None:
            time_since_last = time.time() - self._last_detection_time
            if time_since_last < self.config.detection_cooldown:
                logger.debug(
                    f"Detection cooldown active ({time_since_last:.2f}s < "
                    f"{self.config.detection_cooldown:.2f}s)"
                )
                return None

        # Check each target command
        for command in self.config.target_commands:
            match_confidence = self._check_match(transcription, command)

            if match_confidence is not None:
                # Command detected!
                detection_time_ms = (time.time() - start_time) * 1000

                event = DetectionEvent(
                    command=command,
                    matched_text=transcription,
                    transcription=transcription,
                    confidence=match_confidence,
                    timestamp=time.time(),
                    strategy=self.config.match_strategy,
                    metadata=metadata or {}
                )

                # Update state
                self._stream_should_close = True
                self._last_detection_time = time.time()
                self._detection_events.append(event)

                # Update metrics
                self.metrics["total_detections"] += 1
                self.metrics["commands_detected"][command] = (
                    self.metrics["commands_detected"].get(command, 0) + 1
                )
                self.metrics["last_detection"] = {
                    "command": command,
                    "timestamp": datetime.now().isoformat(),
                    "transcription": transcription,
                    "confidence": match_confidence,
                }

                # Update average detection time
                prev_avg = self.metrics["avg_detection_time_ms"]
                total = self.metrics["total_detections"]
                self.metrics["avg_detection_time_ms"] = (
                    (prev_avg * (total - 1) + detection_time_ms) / total
                )

                if self.config.enable_logging:
                    logger.info(
                        f"🎯 COMMAND DETECTED: '{command}' in '{transcription}' | "
                        f"Confidence: {match_confidence:.2f} | "
                        f"Strategy: {self.config.match_strategy.value} | "
                        f"Detection time: {detection_time_ms:.1f}ms | "
                        f"🚨 CLOSING STREAM"
                    )

                # Notify callbacks
                await self._notify_command_callbacks(event)
                await self._notify_stream_close_callbacks()

                return event

        return None

    async def _notify_command_callbacks(self, event: DetectionEvent):
        """Notify command detection callbacks"""
        for callback in self._on_command_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Command callback error: {e}")

    async def _notify_stream_close_callbacks(self):
        """Notify stream closure callbacks"""
        for callback in self._on_stream_close_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Stream close callback error: {e}")

    def should_close_stream(self) -> bool:
        """
        Check if stream should be closed

        Returns:
            True if stream should be closed
        """
        return self._stream_should_close

    def reset(self):
        """Reset safeguard state (for new stream session)"""
        self._stream_should_close = False
        self._last_detection_time = None

        logger.debug("Streaming safeguard reset for new session")

    def on_command_detected(self, callback: Callable[[DetectionEvent], None]):
        """
        Register callback for command detection

        Args:
            callback: Async or sync function called when command detected
        """
        self._on_command_callbacks.append(callback)

    def on_stream_close(self, callback: Callable[[], None]):
        """
        Register callback for stream closure

        Args:
            callback: Async or sync function called when stream should close
        """
        self._on_stream_close_callbacks.append(callback)

    def get_detection_events(self, limit: Optional[int] = None) -> List[DetectionEvent]:
        """
        Get recent detection events

        Args:
            limit: Maximum number of events to return

        Returns:
            List of detection events (most recent first)
        """
        events = sorted(self._detection_events, key=lambda e: e.timestamp, reverse=True)

        if limit is not None:
            events = events[:limit]

        return events

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get safeguard metrics

        Returns:
            Metrics dictionary
        """
        return {
            **self.metrics,
            "config": {
                "strategy": self.config.match_strategy.value,
                "target_commands": self.config.target_commands,
                "fuzzy_threshold": self.config.fuzzy_threshold,
                "cooldown_seconds": self.config.detection_cooldown,
            },
            "state": {
                "should_close_stream": self._stream_should_close,
                "last_detection_time": self._last_detection_time,
                "total_events": len(self._detection_events),
            }
        }

    def add_command(self, command: str):
        """
        Add target command to detection list

        Args:
            command: Command to add
        """
        if command not in self.config.target_commands:
            self.config.target_commands.append(command)

            # Recompile regex if needed
            if self.config.match_strategy == CommandMatchStrategy.REGEX:
                try:
                    flags = 0 if self.config.case_sensitive else re.IGNORECASE
                    self._regex_patterns[command] = re.compile(command, flags)
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{command}': {e}")

            logger.info(f"Added command to safeguard: '{command}'")

    def remove_command(self, command: str):
        """
        Remove target command from detection list

        Args:
            command: Command to remove
        """
        if command in self.config.target_commands:
            self.config.target_commands.remove(command)

            if command in self._regex_patterns:
                del self._regex_patterns[command]

            logger.info(f"Removed command from safeguard: '{command}'")


# Global safeguard instance
_global_safeguard: Optional[StreamingSafeguard] = None


def get_streaming_safeguard(
    config: Optional[CommandDetectionConfig] = None
) -> StreamingSafeguard:
    """
    Get or create global streaming safeguard instance

    Args:
        config: Optional configuration (used on first call)

    Returns:
        StreamingSafeguard instance
    """
    global _global_safeguard

    if _global_safeguard is None:
        _global_safeguard = StreamingSafeguard(config)

    return _global_safeguard


def reset_streaming_safeguard():
    """Reset global safeguard instance"""
    global _global_safeguard

    if _global_safeguard is not None:
        _global_safeguard.reset()
