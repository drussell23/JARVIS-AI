"""
Wake Word Detection Configuration
=================================

Central configuration for the wake word detection system with environment variable support.
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass, field
from enum import Enum


class WakeWordEngine(str, Enum):
    """Available wake word detection engines"""
    PORCUPINE = "porcupine"
    VOSK = "vosk"
    WEBRTC_VAD = "webrtc_vad"
    CUSTOM_CNN = "custom_cnn"
    HYBRID = "hybrid"


class SensitivityLevel(str, Enum):
    """Wake word detection sensitivity levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = int(os.getenv('WAKE_WORD_SAMPLE_RATE', '16000'))
    channels: int = int(os.getenv('WAKE_WORD_CHANNELS', '1'))
    chunk_size: int = int(os.getenv('WAKE_WORD_CHUNK_SIZE', '512'))
    format: str = os.getenv('WAKE_WORD_AUDIO_FORMAT', 'int16')
    
    # Audio preprocessing
    noise_reduction: bool = os.getenv('WAKE_WORD_NOISE_REDUCTION', 'true').lower() == 'true'
    auto_gain_control: bool = os.getenv('WAKE_WORD_AGC', 'true').lower() == 'true'
    echo_cancellation: bool = os.getenv('WAKE_WORD_ECHO_CANCEL', 'false').lower() == 'true'
    
    # Buffer settings
    buffer_seconds: float = float(os.getenv('WAKE_WORD_BUFFER_SECONDS', '2.0'))
    pre_recording_seconds: float = float(os.getenv('WAKE_WORD_PRE_RECORDING', '0.5'))
    post_recording_seconds: float = float(os.getenv('WAKE_WORD_POST_RECORDING', '1.0'))


@dataclass
class DetectionConfig:
    """Wake word detection configuration"""
    # Primary wake words (can be multiple)
    wake_words: List[str] = field(default_factory=lambda: 
        os.getenv('WAKE_WORDS', 'hey jarvis,jarvis,ok jarvis').split(','))
    
    # Detection engine
    engine: WakeWordEngine = WakeWordEngine(os.getenv('WAKE_WORD_ENGINE', 'hybrid'))
    
    # Sensitivity settings
    sensitivity: SensitivityLevel = SensitivityLevel(os.getenv('WAKE_WORD_SENSITIVITY', 'medium'))
    threshold: float = float(os.getenv('WAKE_WORD_THRESHOLD', '0.5'))
    
    # Multiple wake word support
    enable_multiple_wake_words: bool = os.getenv('ENABLE_MULTIPLE_WAKE_WORDS', 'true').lower() == 'true'
    wake_word_timeout: float = float(os.getenv('WAKE_WORD_TIMEOUT', '5.0'))
    
    # Advanced detection
    require_silence_before: bool = os.getenv('REQUIRE_SILENCE_BEFORE', 'true').lower() == 'true'
    silence_threshold_db: float = float(os.getenv('SILENCE_THRESHOLD_DB', '-40.0'))
    
    # Context awareness
    context_aware: bool = os.getenv('WAKE_WORD_CONTEXT_AWARE', 'true').lower() == 'true'
    max_false_positive_rate: float = float(os.getenv('MAX_FALSE_POSITIVE_RATE', '0.1'))


@dataclass
class ResponseConfig:
    """Configuration for JARVIS responses"""
    # Response phrases (randomly selected)
    activation_responses: List[str] = field(default_factory=lambda:
        os.getenv('JARVIS_ACTIVATION_RESPONSES', 
                  'I\'m online Sir. Waiting for your command.,'
                  'Yes Sir. How may I assist you?,'
                  'At your service Sir.,'
                  'Ready for your instructions Sir.,'
                  'How can I help you today Sir?').split(','))
    
    # Response behavior
    use_voice_response: bool = os.getenv('USE_VOICE_RESPONSE', 'true').lower() == 'true'
    response_delay_ms: int = int(os.getenv('RESPONSE_DELAY_MS', '200'))
    
    # Visual feedback
    show_visual_indicator: bool = os.getenv('SHOW_VISUAL_INDICATOR', 'true').lower() == 'true'
    play_activation_sound: bool = os.getenv('PLAY_ACTIVATION_SOUND', 'true').lower() == 'true'
    
    # Personalization
    use_time_based_greetings: bool = os.getenv('TIME_BASED_GREETINGS', 'true').lower() == 'true'
    learn_user_patterns: bool = os.getenv('LEARN_USER_PATTERNS', 'true').lower() == 'true'


@dataclass
class PerformanceConfig:
    """Performance and optimization settings"""
    # CPU usage
    max_cpu_percent: int = int(os.getenv('WAKE_WORD_MAX_CPU', '20'))
    use_gpu_acceleration: bool = os.getenv('USE_GPU_ACCELERATION', 'false').lower() == 'true'
    
    # Memory management
    max_memory_mb: int = int(os.getenv('WAKE_WORD_MAX_MEMORY_MB', '100'))
    enable_memory_optimization: bool = os.getenv('ENABLE_MEMORY_OPT', 'true').lower() == 'true'
    
    # Threading
    num_detection_threads: int = int(os.getenv('NUM_DETECTION_THREADS', '2'))
    detection_queue_size: int = int(os.getenv('DETECTION_QUEUE_SIZE', '10'))
    
    # Power management
    low_power_mode: bool = os.getenv('LOW_POWER_MODE', 'false').lower() == 'true'
    adaptive_processing: bool = os.getenv('ADAPTIVE_PROCESSING', 'true').lower() == 'true'


@dataclass
class PrivacyConfig:
    """Privacy and security settings"""
    # Local processing
    process_locally_only: bool = os.getenv('PROCESS_LOCALLY_ONLY', 'true').lower() == 'true'
    
    # Audio retention
    save_wake_word_audio: bool = os.getenv('SAVE_WAKE_WORD_AUDIO', 'false').lower() == 'true'
    audio_retention_days: int = int(os.getenv('AUDIO_RETENTION_DAYS', '7'))
    
    # Privacy indicators
    show_listening_indicator: bool = os.getenv('SHOW_LISTENING_INDICATOR', 'true').lower() == 'true'
    require_user_consent: bool = os.getenv('REQUIRE_USER_CONSENT', 'true').lower() == 'true'
    
    # Data collection
    collect_analytics: bool = os.getenv('COLLECT_ANALYTICS', 'false').lower() == 'true'
    anonymize_data: bool = os.getenv('ANONYMIZE_DATA', 'true').lower() == 'true'


@dataclass
class IntegrationConfig:
    """Integration with JARVIS system"""
    # WebSocket settings
    websocket_enabled: bool = os.getenv('WAKE_WORD_WEBSOCKET', 'true').lower() == 'true'
    websocket_port: int = int(os.getenv('WAKE_WORD_WS_PORT', '8765'))
    
    # API settings
    enable_rest_api: bool = os.getenv('WAKE_WORD_REST_API', 'true').lower() == 'true'
    api_endpoint: str = os.getenv('WAKE_WORD_API_ENDPOINT', '/api/wake-word')
    
    # Event system
    emit_events: bool = os.getenv('EMIT_WAKE_WORD_EVENTS', 'true').lower() == 'true'
    event_queue_size: int = int(os.getenv('EVENT_QUEUE_SIZE', '100'))
    
    # Fallback options
    enable_button_fallback: bool = os.getenv('ENABLE_BUTTON_FALLBACK', 'true').lower() == 'true'
    enable_keyboard_shortcut: bool = os.getenv('ENABLE_KEYBOARD_SHORTCUT', 'true').lower() == 'true'
    keyboard_shortcut: str = os.getenv('KEYBOARD_SHORTCUT', 'cmd+shift+j')


@dataclass
class WakeWordConfig:
    """Main wake word configuration"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    response: ResponseConfig = field(default_factory=ResponseConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    
    # Feature flags
    enabled: bool = os.getenv('WAKE_WORD_ENABLED', 'true').lower() == 'true'
    debug_mode: bool = os.getenv('WAKE_WORD_DEBUG', 'false').lower() == 'true'
    
    def get_sensitivity_value(self) -> float:
        """Convert sensitivity level to numeric value"""
        sensitivity_map = {
            SensitivityLevel.VERY_LOW: 0.1,
            SensitivityLevel.LOW: 0.3,
            SensitivityLevel.MEDIUM: 0.5,
            SensitivityLevel.HIGH: 0.7,
            SensitivityLevel.VERY_HIGH: 0.9
        }
        return sensitivity_map.get(self.detection.sensitivity, 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'channels': self.audio.channels,
                'chunk_size': self.audio.chunk_size,
                'noise_reduction': self.audio.noise_reduction,
                'auto_gain_control': self.audio.auto_gain_control
            },
            'detection': {
                'wake_words': self.detection.wake_words,
                'engine': self.detection.engine,
                'sensitivity': self.detection.sensitivity,
                'threshold': self.detection.threshold
            },
            'response': {
                'activation_responses': self.response.activation_responses,
                'use_voice_response': self.response.use_voice_response,
                'show_visual_indicator': self.response.show_visual_indicator
            },
            'enabled': self.enabled
        }


# Global config instance
_config: WakeWordConfig = None


def get_config() -> WakeWordConfig:
    """Get wake word configuration singleton"""
    global _config
    if _config is None:
        _config = WakeWordConfig()
    return _config


def reload_config():
    """Reload configuration from environment"""
    global _config
    _config = WakeWordConfig()
    return _config