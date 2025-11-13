#!/usr/bin/env python3
"""
Voice Pipeline Configuration
=============================

Centralized configuration for the voice processing pipeline including:
- VAD (Voice Activity Detection)
- Audio Windowing & Truncation
- Streaming Safeguard (Command Detection)
- Whisper STT
- Hybrid STT Router

All settings are configurable via environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VADConfig:
    """Voice Activity Detection configuration"""
    # Enable VAD preprocessing
    enabled: bool = os.getenv('ENABLE_VAD', 'true').lower() == 'true'

    # Primary VAD engine ('webrtc' or 'silero')
    primary_vad: str = os.getenv('PRIMARY_VAD', 'webrtc')

    # Use secondary VAD for refinement
    use_secondary_vad: bool = os.getenv('USE_SECONDARY_VAD', 'true').lower() == 'true'

    # Secondary VAD engine ('silero' or 'none')
    secondary_vad: str = os.getenv('SECONDARY_VAD', 'silero')

    # Combination strategy ('sequential', 'parallel', 'voting')
    combination_strategy: str = os.getenv('VAD_COMBINATION_STRATEGY', 'sequential')

    # Sample rate for VAD processing
    sample_rate: int = int(os.getenv('VAD_SAMPLE_RATE', '16000'))

    # Frame duration in milliseconds (10, 20, or 30)
    frame_duration_ms: int = int(os.getenv('VAD_FRAME_DURATION_MS', '30'))

    # WebRTC VAD aggressiveness (0-3, higher = more aggressive)
    webrtc_aggressiveness: int = int(os.getenv('WEBRTC_AGGRESSIVENESS', '2'))

    # Silero VAD speech threshold (0.0-1.0)
    silero_threshold: float = float(os.getenv('SILERO_THRESHOLD', '0.5'))

    # Minimum speech duration to keep (milliseconds)
    min_speech_duration_ms: int = int(os.getenv('MIN_SPEECH_DURATION_MS', '300'))

    # Maximum silence duration to keep (milliseconds)
    max_silence_duration_ms: int = int(os.getenv('MAX_SILENCE_DURATION_MS', '300'))

    # Padding duration around speech segments (milliseconds)
    padding_duration_ms: int = int(os.getenv('VAD_PADDING_DURATION_MS', '200'))


@dataclass
class WindowingConfig:
    """Audio windowing and truncation configuration"""
    # Enable audio windowing
    enabled: bool = os.getenv('ENABLE_WINDOWING', 'true').lower() == 'true'

    # Global hard limit (seconds) - applies to ALL transcription
    max_audio_seconds: float = float(os.getenv('MAX_AUDIO_SECONDS', '5.0'))

    # Unlock-specific window (seconds) - stricter limit for unlock flow
    unlock_window_seconds: float = float(os.getenv('UNLOCK_WINDOW_SECONDS', '2.0'))

    # Command-specific window (seconds) - for command detection
    command_window_seconds: float = float(os.getenv('COMMAND_WINDOW_SECONDS', '3.0'))

    # Sample rate for calculating window sizes
    sample_rate: int = 16000

    # Keep strategy: 'last' (most recent) or 'first' (beginning)
    keep_strategy: str = os.getenv('WINDOWING_KEEP_STRATEGY', 'last')


@dataclass
class StreamingSafeguardConfig:
    """Streaming safeguard (command detection) configuration"""
    # Enable streaming safeguard
    enabled: bool = os.getenv('ENABLE_STREAMING_SAFEGUARD', 'true').lower() == 'true'

    # Target commands to detect (will close stream when detected)
    target_commands: List[str] = field(default_factory=lambda: [
        'unlock',
        'lock',
        'jarvis',
        'hey jarvis',
        'unlock my screen',
        'lock my screen',
        'unlock the screen',
        'lock the screen',
    ])

    # Matching strategy ('exact', 'fuzzy', 'regex', 'contains', 'word_boundary')
    match_strategy: str = os.getenv('COMMAND_MATCH_STRATEGY', 'word_boundary')

    # Fuzzy matching threshold (0.0-1.0, used with 'fuzzy' strategy)
    fuzzy_threshold: float = float(os.getenv('COMMAND_FUZZY_THRESHOLD', '0.8'))

    # Case sensitivity for matching
    case_sensitive: bool = os.getenv('COMMAND_CASE_SENSITIVE', 'false').lower() == 'true'

    # Strip punctuation before matching
    strip_punctuation: bool = os.getenv('COMMAND_STRIP_PUNCTUATION', 'true').lower() == 'true'

    # Minimum confidence required for transcription (0.0-1.0)
    min_transcription_confidence: float = float(os.getenv('MIN_TRANSCRIPTION_CONFIDENCE', '0.5'))

    # Cooldown period between detections (seconds)
    detection_cooldown: float = float(os.getenv('COMMAND_DETECTION_COOLDOWN', '1.0'))

    # Enable command detection logging
    enable_logging: bool = os.getenv('ENABLE_COMMAND_DETECTION_LOGGING', 'true').lower() == 'true'


@dataclass
class WhisperConfig:
    """Whisper STT configuration"""
    # Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
    model_size: str = os.getenv('WHISPER_MODEL_SIZE', 'base')

    # Device for Whisper inference ('cpu', 'cuda', 'mps')
    device: str = os.getenv('WHISPER_DEVICE', 'cpu')

    # Compute type ('int8', 'float16', 'float32')
    compute_type: str = os.getenv('WHISPER_COMPUTE_TYPE', 'int8')

    # Number of threads for CPU inference
    cpu_threads: int = int(os.getenv('WHISPER_CPU_THREADS', '4'))

    # Beam size for decoding
    beam_size: int = int(os.getenv('WHISPER_BEAM_SIZE', '5'))

    # Language (None = auto-detect)
    language: Optional[str] = os.getenv('WHISPER_LANGUAGE', 'en')

    # Initial prompt for Whisper
    initial_prompt: Optional[str] = os.getenv('WHISPER_INITIAL_PROMPT')

    # Enable VAD filter in Whisper
    vad_filter: bool = os.getenv('WHISPER_VAD_FILTER', 'false').lower() == 'true'


@dataclass
class HybridSTTConfig:
    """Hybrid STT Router configuration"""
    # Enable hybrid routing
    enabled: bool = os.getenv('ENABLE_HYBRID_STT', 'true').lower() == 'true'

    # Primary STT engine ('whisper', 'google', 'azure', 'deepgram')
    primary_engine: str = os.getenv('PRIMARY_STT_ENGINE', 'whisper')

    # Fallback STT engine
    fallback_engine: str = os.getenv('FALLBACK_STT_ENGINE', 'whisper')

    # Routing strategy ('speed', 'accuracy', 'balanced', 'cost')
    routing_strategy: str = os.getenv('STT_ROUTING_STRATEGY', 'balanced')

    # Confidence threshold for accepting results (0.0-1.0)
    confidence_threshold: float = float(os.getenv('STT_CONFIDENCE_THRESHOLD', '0.7'))


@dataclass
class VoicePipelineConfig:
    """
    Master configuration for the entire voice processing pipeline

    Environment Variables:
    ----------------------
    # VAD Configuration
    ENABLE_VAD=true
    PRIMARY_VAD=webrtc
    USE_SECONDARY_VAD=true
    SECONDARY_VAD=silero
    VAD_COMBINATION_STRATEGY=sequential
    WEBRTC_AGGRESSIVENESS=2
    SILERO_THRESHOLD=0.5

    # Windowing Configuration
    ENABLE_WINDOWING=true
    MAX_AUDIO_SECONDS=5.0
    UNLOCK_WINDOW_SECONDS=2.0
    COMMAND_WINDOW_SECONDS=3.0
    WINDOWING_KEEP_STRATEGY=last

    # Streaming Safeguard Configuration
    ENABLE_STREAMING_SAFEGUARD=true
    COMMAND_MATCH_STRATEGY=word_boundary
    COMMAND_FUZZY_THRESHOLD=0.8
    MIN_TRANSCRIPTION_CONFIDENCE=0.5
    COMMAND_DETECTION_COOLDOWN=1.0

    # Whisper Configuration
    WHISPER_MODEL_SIZE=base
    WHISPER_DEVICE=cpu
    WHISPER_COMPUTE_TYPE=int8
    WHISPER_CPU_THREADS=4
    WHISPER_LANGUAGE=en

    # Hybrid STT Configuration
    ENABLE_HYBRID_STT=true
    PRIMARY_STT_ENGINE=whisper
    STT_ROUTING_STRATEGY=balanced
    STT_CONFIDENCE_THRESHOLD=0.7
    """
    vad: VADConfig = field(default_factory=VADConfig)
    windowing: WindowingConfig = field(default_factory=WindowingConfig)
    safeguard: StreamingSafeguardConfig = field(default_factory=StreamingSafeguardConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    hybrid_stt: HybridSTTConfig = field(default_factory=HybridSTTConfig)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            "vad": self.vad.__dict__,
            "windowing": self.windowing.__dict__,
            "safeguard": self.safeguard.__dict__,
            "whisper": self.whisper.__dict__,
            "hybrid_stt": self.hybrid_stt.__dict__,
        }

    def __str__(self) -> str:
        """Pretty string representation"""
        return (
            f"Voice Pipeline Configuration:\n"
            f"  VAD: {'✅' if self.vad.enabled else '❌'} "
            f"({self.vad.primary_vad} → {self.vad.secondary_vad if self.vad.use_secondary_vad else 'none'})\n"
            f"  Windowing: {'✅' if self.windowing.enabled else '❌'} "
            f"({self.windowing.max_audio_seconds}s global, "
            f"{self.windowing.unlock_window_seconds}s unlock)\n"
            f"  Safeguard: {'✅' if self.safeguard.enabled else '❌'} "
            f"({len(self.safeguard.target_commands)} commands, "
            f"strategy={self.safeguard.match_strategy})\n"
            f"  Whisper: {self.whisper.model_size} on {self.whisper.device}\n"
            f"  Hybrid STT: {'✅' if self.hybrid_stt.enabled else '❌'} "
            f"({self.hybrid_stt.routing_strategy})"
        )


# Global configuration instance
_global_config: Optional[VoicePipelineConfig] = None


def get_voice_pipeline_config() -> VoicePipelineConfig:
    """
    Get or create global voice pipeline configuration

    Returns:
        VoicePipelineConfig instance
    """
    global _global_config

    if _global_config is None:
        _global_config = VoicePipelineConfig()

    return _global_config


def reload_config() -> VoicePipelineConfig:
    """
    Reload configuration from environment variables

    Returns:
        New VoicePipelineConfig instance
    """
    global _global_config
    _global_config = VoicePipelineConfig()
    return _global_config


# Export convenience function
def get_config() -> VoicePipelineConfig:
    """Alias for get_voice_pipeline_config()"""
    return get_voice_pipeline_config()
