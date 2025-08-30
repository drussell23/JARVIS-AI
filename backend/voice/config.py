"""
Voice System Configuration
All configurable parameters in one place
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class VoiceConfig:
    """Central configuration for voice system"""
    
    # Wake word detection thresholds
    wake_word_threshold_default: float = 0.55  # Lowered from 0.85
    wake_word_threshold_min: float = 0.5
    wake_word_threshold_max: float = 0.95
    confidence_threshold: float = 0.6  # Lowered from 0.7
    
    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    noise_floor_quiet: float = 20.0  # dB - quiet environment
    noise_floor_moderate: float = 10.0  # dB - moderate noise
    noise_floor_noisy: float = 0.0  # dB - very noisy
    
    # VAD (Voice Activity Detection) settings
    enable_vad: bool = True
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive
    vad_frame_duration_ms: int = 30  # milliseconds per frame
    vad_padding_frames: int = 10  # frames to pad before/after speech
    
    # Audio buffer settings
    audio_buffer_duration: float = 3.0  # seconds
    audio_sample_rate: int = 16000  # Hz
    audio_chunk_size: int = 1024  # samples per chunk
    
    # Wake word buffer
    enable_wake_word_buffer: bool = True
    wake_word_buffer_pre: float = 0.5  # seconds before detection
    wake_word_buffer_post: float = 2.5  # seconds after start
    
    # Streaming settings
    enable_streaming: bool = True
    stream_chunk_duration_ms: int = 100  # milliseconds
    max_memory_mb: int = 200  # Max memory for audio processing
    
    # Model selection
    use_picovoice: bool = field(default_factory=lambda: os.getenv("USE_PICOVOICE", "true").lower() == "true")
    picovoice_access_key: Optional[str] = field(default_factory=lambda: os.getenv("PICOVOICE_ACCESS_KEY"))
    wake_words: list = field(default_factory=lambda: ["jarvis", "hey jarvis"])
    
    # Performance tuning
    max_cpu_percent: float = 30.0  # Max CPU usage
    enable_gpu: bool = False  # Use GPU if available
    model_cache_size: int = 5  # Number of models to keep in memory
    
    # Environmental adaptation
    adaptation_rate: float = 0.1  # How quickly to adapt (0-1)
    noise_estimation_window: float = 0.5  # seconds
    snr_threshold_quiet: float = 20.0  # dB
    snr_threshold_moderate: float = 10.0  # dB
    
    # Personalization
    enable_personalization: bool = True
    learning_rate: float = 0.01
    false_positive_threshold: int = 20  # Retrain after N false positives
    save_interval: int = 100  # Save models every N detections
    
    # Logging and debugging
    log_audio_features: bool = False
    save_false_positives: bool = True
    debug_mode: bool = False
    
    @classmethod
    def from_env(cls) -> 'VoiceConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Override from environment
        if val := os.getenv("WAKE_WORD_THRESHOLD"):
            config.wake_word_threshold_default = float(val)
        if val := os.getenv("CONFIDENCE_THRESHOLD"):
            config.confidence_threshold = float(val)
        if val := os.getenv("ENABLE_VAD"):
            config.enable_vad = val.lower() == "true"
        if val := os.getenv("ENABLE_STREAMING"):
            config.enable_streaming = val.lower() == "true"
        if val := os.getenv("DEBUG_MODE"):
            config.debug_mode = val.lower() == "true"
            
        return config
    
    def get_adaptive_threshold(self, snr: float) -> float:
        """Get threshold adapted to current noise level"""
        if not self.enable_adaptive_thresholds:
            return self.wake_word_threshold_default
            
        if snr >= self.snr_threshold_quiet:
            # Quiet environment - use default threshold
            return self.wake_word_threshold_default
        elif snr >= self.snr_threshold_moderate:
            # Moderate noise - slightly lower threshold
            adaptation = 0.95
        else:
            # Noisy environment - lower threshold more
            adaptation = 0.9
            
        adapted = self.wake_word_threshold_default * adaptation
        return max(self.wake_word_threshold_min, 
                  min(self.wake_word_threshold_max, adapted))

# Global config instance
VOICE_CONFIG = VoiceConfig.from_env()