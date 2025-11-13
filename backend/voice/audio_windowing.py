#!/usr/bin/env python3
"""
Audio Windowing and Truncation
Hard limits on audio duration to prevent hallucinations and timeouts
"""

import logging
from typing import Optional
from dataclasses import dataclass
import numpy as np
import os

logger = logging.getLogger(__name__)


@dataclass
class WindowConfig:
    """Configuration for audio windowing"""
    # Global hard limit (seconds) - applies to ALL transcription
    max_audio_seconds: float = float(os.getenv('MAX_AUDIO_SECONDS', '5.0'))

    # Unlock-specific window (seconds) - stricter limit for unlock flow
    unlock_window_seconds: float = float(os.getenv('UNLOCK_WINDOW_SECONDS', '2.0'))

    # Command-specific window (seconds) - for command detection
    command_window_seconds: float = float(os.getenv('COMMAND_WINDOW_SECONDS', '3.0'))

    # Sample rate for calculating window sizes
    sample_rate: int = 16000

    # Keep strategy: 'last' (most recent) or 'first' (beginning)
    keep_strategy: str = 'last'  # 'last' is recommended for real-time

    def __post_init__(self):
        """Validate configuration"""
        if self.max_audio_seconds <= 0:
            raise ValueError(f"max_audio_seconds must be positive, got {self.max_audio_seconds}")

        if self.unlock_window_seconds > self.max_audio_seconds:
            logger.warning(
                f"unlock_window_seconds ({self.unlock_window_seconds}s) > "
                f"max_audio_seconds ({self.max_audio_seconds}s), clamping to max"
            )
            self.unlock_window_seconds = self.max_audio_seconds

        if self.command_window_seconds > self.max_audio_seconds:
            logger.warning(
                f"command_window_seconds ({self.command_window_seconds}s) > "
                f"max_audio_seconds ({self.max_audio_seconds}s), clamping to max"
            )
            self.command_window_seconds = self.max_audio_seconds


class AudioWindowManager:
    """
    Manages audio window truncation to prevent:
    1. Audio accumulation (60+ second pileups)
    2. Whisper hallucinations on long audio
    3. Timeout errors from processing too much audio
    4. Poor transcription quality on stale audio

    Strategy:
    - Global hard limit: 5 seconds max for any transcription
    - Unlock flow: Even stricter 2-second window for speed
    - Keep most recent audio (discard old) for real-time use
    """

    def __init__(self, config: Optional[WindowConfig] = None):
        """
        Initialize audio window manager

        Args:
            config: Window configuration (uses defaults if None)
        """
        self.config = config or WindowConfig()

        # Calculate window sizes in samples
        self.max_samples = int(self.config.max_audio_seconds * self.config.sample_rate)
        self.unlock_samples = int(self.config.unlock_window_seconds * self.config.sample_rate)
        self.command_samples = int(self.config.command_window_seconds * self.config.sample_rate)

        logger.info(f"🪟 Audio Window Manager initialized:")
        logger.info(f"   Global limit: {self.config.max_audio_seconds}s ({self.max_samples} samples)")
        logger.info(f"   Unlock window: {self.config.unlock_window_seconds}s ({self.unlock_samples} samples)")
        logger.info(f"   Command window: {self.config.command_window_seconds}s ({self.command_samples} samples)")
        logger.info(f"   Strategy: Keep {self.config.keep_strategy}")

    def truncate_to_max(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply global hard limit - keep only last N seconds

        This is the PRIMARY defense against audio accumulation.
        Call this BEFORE any Whisper transcription.

        Args:
            audio: Audio array (float32)

        Returns:
            Truncated audio (max self.max_samples)
        """
        if len(audio) <= self.max_samples:
            return audio

        # Keep most recent audio
        if self.config.keep_strategy == 'last':
            truncated = audio[-self.max_samples:]
            logger.info(
                f"⚠️ TRUNCATED: {len(audio)} → {len(truncated)} samples "
                f"({len(audio) / self.config.sample_rate:.2f}s → "
                f"{self.config.max_audio_seconds:.2f}s) [KEEPING LAST {self.config.max_audio_seconds}s]"
            )
        else:
            # Keep beginning (not recommended for real-time)
            truncated = audio[:self.max_samples]
            logger.info(
                f"⚠️ TRUNCATED: {len(audio)} → {len(truncated)} samples "
                f"({len(audio) / self.config.sample_rate:.2f}s → "
                f"{self.config.max_audio_seconds:.2f}s) [KEEPING FIRST {self.config.max_audio_seconds}s]"
            )

        return truncated

    def truncate_for_unlock(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply unlock-specific window - ultra-short for speed

        Call this in the unlock flow for minimum latency.
        This applies AFTER global truncation (so it's <= global max).

        Args:
            audio: Audio array (float32)

        Returns:
            Truncated audio (max self.unlock_samples)
        """
        if len(audio) <= self.unlock_samples:
            return audio

        # Keep most recent audio for unlock
        truncated = audio[-self.unlock_samples:]

        logger.info(
            f"🔐 UNLOCK WINDOW: {len(audio)} → {len(truncated)} samples "
            f"({len(audio) / self.config.sample_rate:.2f}s → "
            f"{self.config.unlock_window_seconds:.2f}s) [ULTRA-FAST MODE]"
        )

        return truncated

    def truncate_for_command(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply command-specific window

        Call this for command detection (slightly longer than unlock).

        Args:
            audio: Audio array (float32)

        Returns:
            Truncated audio (max self.command_samples)
        """
        if len(audio) <= self.command_samples:
            return audio

        truncated = audio[-self.command_samples:]

        logger.info(
            f"🎯 COMMAND WINDOW: {len(audio)} → {len(truncated)} samples "
            f"({len(audio) / self.config.sample_rate:.2f}s → "
            f"{self.config.command_window_seconds:.2f}s)"
        )

        return truncated

    def get_duration(self, audio: np.ndarray) -> float:
        """
        Get audio duration in seconds

        Args:
            audio: Audio array

        Returns:
            Duration in seconds
        """
        return len(audio) / self.config.sample_rate

    def is_within_limit(self, audio: np.ndarray, limit_type: str = 'max') -> bool:
        """
        Check if audio is within specified limit

        Args:
            audio: Audio array
            limit_type: 'max', 'unlock', or 'command'

        Returns:
            True if within limit
        """
        if limit_type == 'max':
            return len(audio) <= self.max_samples
        elif limit_type == 'unlock':
            return len(audio) <= self.unlock_samples
        elif limit_type == 'command':
            return len(audio) <= self.command_samples
        else:
            logger.warning(f"Unknown limit_type: {limit_type}, defaulting to 'max'")
            return len(audio) <= self.max_samples

    def prepare_for_transcription(
        self,
        audio: np.ndarray,
        mode: str = 'general'
    ) -> np.ndarray:
        """
        Prepare audio for transcription with appropriate windowing

        This is the main entry point for windowing logic.

        Args:
            audio: Raw audio array (float32)
            mode: 'general', 'unlock', or 'command'

        Returns:
            Windowed audio ready for Whisper
        """
        if len(audio) == 0:
            logger.warning("Empty audio provided to prepare_for_transcription")
            return audio

        original_duration = self.get_duration(audio)

        # Step 1: Always apply global hard limit first
        audio = self.truncate_to_max(audio)

        # Step 2: Apply mode-specific windowing
        if mode == 'unlock':
            audio = self.truncate_for_unlock(audio)
        elif mode == 'command':
            audio = self.truncate_for_command(audio)
        # 'general' mode uses only global truncation

        final_duration = self.get_duration(audio)

        if original_duration != final_duration:
            logger.info(
                f"📊 Windowing summary ({mode} mode): "
                f"{original_duration:.2f}s → {final_duration:.2f}s "
                f"({final_duration / original_duration * 100:.1f}% retained)"
            )

        return audio


# Global window manager instance
_window_manager: Optional[AudioWindowManager] = None


def get_window_manager(config: Optional[WindowConfig] = None) -> AudioWindowManager:
    """
    Get or create global window manager instance

    Args:
        config: Optional configuration (used on first call)

    Returns:
        AudioWindowManager instance
    """
    global _window_manager

    if _window_manager is None:
        _window_manager = AudioWindowManager(config)

    return _window_manager


def truncate_audio(
    audio: np.ndarray,
    mode: str = 'general',
    config: Optional[WindowConfig] = None
) -> np.ndarray:
    """
    Convenience function to truncate audio

    Args:
        audio: Audio array
        mode: 'general', 'unlock', or 'command'
        config: Optional window configuration

    Returns:
        Truncated audio
    """
    manager = get_window_manager(config)
    return manager.prepare_for_transcription(audio, mode=mode)
