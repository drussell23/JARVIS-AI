"""
Voice Unlock Configuration
=========================

Dynamic configuration system with environment-based overrides.
No hardcoded values - everything is configurable.
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _get_system_ram_gb() -> int:
    """Get system RAM in GB"""
    if PSUTIL_AVAILABLE:
        try:
            return int(psutil.virtual_memory().total / (1024**3))
        except:
            pass
    return 16  # Default assumption


@dataclass
class AudioSettings:
    """Audio capture and processing settings"""
    sample_rate: int = int(os.getenv('VOICE_UNLOCK_SAMPLE_RATE', '16000'))
    channels: int = int(os.getenv('VOICE_UNLOCK_CHANNELS', '1'))
    chunk_size: int = int(os.getenv('VOICE_UNLOCK_CHUNK_SIZE', '1024'))
    format: str = os.getenv('VOICE_UNLOCK_AUDIO_FORMAT', 'int16')
    
    # Voice Activity Detection
    vad_enabled: bool = os.getenv('VOICE_UNLOCK_VAD_ENABLED', 'true').lower() == 'true'
    vad_energy_threshold: float = float(os.getenv('VOICE_UNLOCK_VAD_THRESHOLD', '0.02'))
    vad_silence_duration: float = float(os.getenv('VOICE_UNLOCK_VAD_SILENCE', '1.5'))
    
    # Quality requirements
    min_duration: float = float(os.getenv('VOICE_UNLOCK_MIN_DURATION', '1.0'))
    max_duration: float = float(os.getenv('VOICE_UNLOCK_MAX_DURATION', '10.0'))
    min_snr_db: float = float(os.getenv('VOICE_UNLOCK_MIN_SNR', '10.0'))


@dataclass
class EnrollmentSettings:
    """Enrollment process configuration"""
    min_samples: int = int(os.getenv('VOICE_UNLOCK_MIN_SAMPLES', '3'))
    max_samples: int = int(os.getenv('VOICE_UNLOCK_MAX_SAMPLES', '5'))
    
    # Phrase configuration
    use_custom_phrases: bool = os.getenv('VOICE_UNLOCK_CUSTOM_PHRASES', 'true').lower() == 'true'
    default_phrases: List[str] = field(default_factory=lambda: json.loads(
        os.getenv('VOICE_UNLOCK_PHRASES', '["Hello JARVIS, unlock my Mac", "JARVIS, this is {user}", "Open sesame, JARVIS"]')
    ))
    allow_user_phrases: bool = os.getenv('VOICE_UNLOCK_USER_PHRASES', 'true').lower() == 'true'
    
    # Quality thresholds
    min_quality_score: float = float(os.getenv('VOICE_UNLOCK_MIN_QUALITY', '0.7'))
    consistency_threshold: float = float(os.getenv('VOICE_UNLOCK_CONSISTENCY', '0.6'))
    
    # Retry configuration
    max_retries_per_sample: int = int(os.getenv('VOICE_UNLOCK_MAX_RETRIES', '3'))
    retry_delay: float = float(os.getenv('VOICE_UNLOCK_RETRY_DELAY', '1.0'))


@dataclass
class AuthenticationSettings:
    """Authentication and verification settings"""
    # Threshold configuration
    base_threshold: float = float(os.getenv('VOICE_UNLOCK_BASE_THRESHOLD', '0.75'))
    high_quality_threshold: float = float(os.getenv('VOICE_UNLOCK_HQ_THRESHOLD', '0.85'))
    low_quality_threshold: float = float(os.getenv('VOICE_UNLOCK_LQ_THRESHOLD', '0.65'))
    
    # Adaptive thresholds
    adaptive_thresholds: bool = os.getenv('VOICE_UNLOCK_ADAPTIVE', 'true').lower() == 'true'
    threshold_learning_rate: float = float(os.getenv('VOICE_UNLOCK_LEARNING_RATE', '0.1'))
    
    # Security settings
    max_attempts: int = int(os.getenv('VOICE_UNLOCK_MAX_ATTEMPTS', '3'))
    lockout_duration: int = int(os.getenv('VOICE_UNLOCK_LOCKOUT_DURATION', '300'))  # seconds
    require_liveness: bool = os.getenv('VOICE_UNLOCK_REQUIRE_LIVENESS', 'true').lower() == 'true'
    
    # Challenge-response
    use_challenges: bool = os.getenv('VOICE_UNLOCK_USE_CHALLENGES', 'false').lower() == 'true'
    challenge_types: List[str] = field(default_factory=lambda: json.loads(
        os.getenv('VOICE_UNLOCK_CHALLENGE_TYPES', '["repeat", "math", "random_words"]')
    ))


@dataclass
class SecuritySettings:
    """Security and privacy settings"""
    # Encryption
    encrypt_voiceprints: bool = os.getenv('VOICE_UNLOCK_ENCRYPT', 'true').lower() == 'true'
    encryption_algorithm: str = os.getenv('VOICE_UNLOCK_ENCRYPTION', 'AES-256-GCM')
    
    # Storage
    storage_backend: str = os.getenv('VOICE_UNLOCK_STORAGE', 'keychain')  # keychain, file, memory
    storage_path: str = os.getenv('VOICE_UNLOCK_STORAGE_PATH', '~/.jarvis/voice_unlock')
    
    # Anti-spoofing
    anti_spoofing_level: str = os.getenv('VOICE_UNLOCK_ANTI_SPOOFING', 'high')  # low, medium, high
    ultrasonic_markers: bool = os.getenv('VOICE_UNLOCK_ULTRASONIC', 'false').lower() == 'true'
    
    # Audit logging
    audit_enabled: bool = os.getenv('VOICE_UNLOCK_AUDIT', 'true').lower() == 'true'
    audit_path: str = os.getenv('VOICE_UNLOCK_AUDIT_PATH', '~/.jarvis/voice_unlock/audit.log')
    
    # Privacy
    delete_audio_after_processing: bool = os.getenv('VOICE_UNLOCK_DELETE_AUDIO', 'true').lower() == 'true'
    anonymize_logs: bool = os.getenv('VOICE_UNLOCK_ANONYMIZE_LOGS', 'false').lower() == 'true'


@dataclass
class SystemIntegrationSettings:
    """macOS system integration settings"""
    # Integration mode
    integration_mode: str = os.getenv('VOICE_UNLOCK_MODE', 'screensaver')  # screensaver, pam, both
    
    # Screensaver settings
    screensaver_timeout: int = int(os.getenv('VOICE_UNLOCK_SCREENSAVER_TIMEOUT', '5'))
    unlock_animation: bool = os.getenv('VOICE_UNLOCK_ANIMATION', 'true').lower() == 'true'
    
    # PAM settings
    pam_service_name: str = os.getenv('VOICE_UNLOCK_PAM_SERVICE', 'jarvis-voice-auth')
    pam_fallback: bool = os.getenv('VOICE_UNLOCK_PAM_FALLBACK', 'true').lower() == 'true'
    
    # Notifications
    show_notifications: bool = os.getenv('VOICE_UNLOCK_NOTIFICATIONS', 'true').lower() == 'true'
    notification_sound: bool = os.getenv('VOICE_UNLOCK_NOTIFICATION_SOUND', 'true').lower() == 'true'
    
    # JARVIS integration
    jarvis_responses: bool = os.getenv('VOICE_UNLOCK_JARVIS_RESPONSES', 'true').lower() == 'true'
    custom_responses: Dict[str, str] = field(default_factory=lambda: json.loads(
        os.getenv('VOICE_UNLOCK_RESPONSES', '{"success": "Welcome back, Sir", "failure": "Voice not recognized, Sir", "lockout": "Security lockout activated, Sir"}')
    ))
    
    # Apple Watch integration
    enable_apple_watch: bool = os.getenv('VOICE_UNLOCK_APPLE_WATCH', 'true').lower() == 'true'
    auto_lock_on_distance: bool = os.getenv('VOICE_UNLOCK_AUTO_LOCK', 'true').lower() == 'true'
    unlock_distance: float = float(os.getenv('VOICE_UNLOCK_UNLOCK_DISTANCE', '3.0'))  # meters
    lock_distance: float = float(os.getenv('VOICE_UNLOCK_LOCK_DISTANCE', '10.0'))  # meters


@dataclass
class PerformanceSettings:
    """Performance and optimization settings"""
    # Auto-detect system RAM and apply appropriate limits
    _total_ram_gb: int = field(default_factory=lambda: _get_system_ram_gb())
    
    # Processing
    use_gpu: bool = os.getenv('VOICE_UNLOCK_USE_GPU', 'false').lower() == 'true'
    num_threads: int = int(os.getenv('VOICE_UNLOCK_THREADS', '0'))  # 0 = auto
    
    # Caching - reduced for 30% memory target
    cache_enabled: bool = os.getenv('VOICE_UNLOCK_CACHE', 'true').lower() == 'true'
    cache_size_mb: int = field(default_factory=lambda: int(os.getenv('VOICE_UNLOCK_CACHE_SIZE', 
        '100' if _get_system_ram_gb() <= 16 else '200')))  # Reduced from 150 to 100MB
    
    # Background processing
    background_monitoring: bool = os.getenv('VOICE_UNLOCK_BACKGROUND', 'true').lower() == 'true'
    monitoring_interval: float = float(os.getenv('VOICE_UNLOCK_MONITOR_INTERVAL', '0.1'))
    
    # Resource limits - ultra-aggressive for 30% target on 16GB
    max_cpu_percent: int = int(os.getenv('VOICE_UNLOCK_MAX_CPU', '20'))  # Reduced from 25
    max_memory_mb: int = field(default_factory=lambda: int(os.getenv('VOICE_UNLOCK_MAX_MEMORY', 
        '300' if _get_system_ram_gb() <= 16 else '500')))  # Reduced from 400 to 300MB
    
    # ML Optimization settings (especially important for 16GB systems)
    enable_quantization: bool = field(default_factory=lambda: os.getenv('VOICE_UNLOCK_QUANTIZATION', 
        'true' if _get_system_ram_gb() <= 16 else 'false').lower() == 'true')
    enable_compression: bool = field(default_factory=lambda: os.getenv('VOICE_UNLOCK_COMPRESSION', 
        'true' if _get_system_ram_gb() <= 16 else 'false').lower() == 'true')
    enable_mmap: bool = os.getenv('VOICE_UNLOCK_MMAP', 'true').lower() == 'true'
    enable_lazy_loading: bool = os.getenv('VOICE_UNLOCK_LAZY_LOADING', 'true').lower() == 'true'
    enable_predictive_loading: bool = os.getenv('VOICE_UNLOCK_PREDICTIVE_LOADING', 'true').lower() == 'true'
    
    # Model management - ultra-strict for 30% target on 16GB systems
    model_unload_timeout: int = field(default_factory=lambda: int(os.getenv('VOICE_UNLOCK_UNLOAD_TIMEOUT', 
        '30' if _get_system_ram_gb() <= 16 else '300')))  # Reduced from 60 to 30
    aggressive_unload_timeout: int = field(default_factory=lambda: int(os.getenv('VOICE_UNLOCK_AGGRESSIVE_TIMEOUT', 
        '15' if _get_system_ram_gb() <= 16 else '60')))  # Reduced from 30 to 15
    model_cache_size: int = field(default_factory=lambda: int(os.getenv('VOICE_UNLOCK_MODEL_CACHE_SIZE', 
        '5' if _get_system_ram_gb() <= 16 else '10')))


class VoiceUnlockConfig:
    """Main configuration manager with dynamic loading"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else self._get_default_config_path()
        
        # Initialize all settings
        self.audio = AudioSettings()
        self.enrollment = EnrollmentSettings()
        self.authentication = AuthenticationSettings()
        self.security = SecuritySettings()
        self.system = SystemIntegrationSettings()
        self.performance = PerformanceSettings()
        
        # Load from file if exists
        if self.config_path.exists():
            self.load_from_file()
            
        # Create config directory if needed
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Apply memory optimizations based on system RAM
        self.apply_memory_optimization()
        
        # Validate configuration
        self.validate()
        
    def _get_default_config_path(self) -> Path:
        """Get default config path based on environment"""
        if os.getenv('VOICE_UNLOCK_CONFIG'):
            return Path(os.getenv('VOICE_UNLOCK_CONFIG'))
            
        # Check standard locations
        locations = [
            Path.home() / '.jarvis' / 'voice_unlock' / 'config.json',
            Path('/etc/jarvis/voice_unlock/config.json'),
            Path('./voice_unlock_config.json')
        ]
        
        for loc in locations:
            if loc.exists():
                return loc
                
        # Default to user home
        return locations[0]
        
    def load_from_file(self):
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                data = json.load(f)
                
            # Update settings from file
            for section, settings in data.items():
                if hasattr(self, section) and isinstance(settings, dict):
                    section_obj = getattr(self, section)
                    for key, value in settings.items():
                        if hasattr(section_obj, key):
                            setattr(section_obj, key, value)
                            
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            
    def save_to_file(self):
        """Save current configuration to file"""
        data = {
            'audio': asdict(self.audio),
            'enrollment': asdict(self.enrollment),
            'authentication': asdict(self.authentication),
            'security': asdict(self.security),
            'system': asdict(self.system),
            'performance': asdict(self.performance)
        }
        
        try:
            with open(self.config_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")
            
    def validate(self):
        """Validate configuration values"""
        # Audio validation
        assert 8000 <= self.audio.sample_rate <= 48000, "Invalid sample rate"
        assert self.audio.channels in [1, 2], "Invalid channel count"
        assert 0 < self.audio.vad_energy_threshold < 1, "Invalid VAD threshold"
        
        # Enrollment validation
        assert self.enrollment.min_samples <= self.enrollment.max_samples, "Invalid sample counts"
        assert 0 < self.enrollment.min_quality_score <= 1, "Invalid quality score"
        
        # Authentication validation
        assert 0 < self.authentication.base_threshold <= 1, "Invalid base threshold"
        assert self.authentication.max_attempts > 0, "Invalid max attempts"
        
        # Performance validation
        assert 0 <= self.performance.max_cpu_percent <= 100, "Invalid CPU limit"
        assert self.performance.max_memory_mb > 0, "Invalid memory limit"
        
        logger.info("Configuration validated successfully")
        
    def get_feature_extraction_params(self) -> Dict[str, Any]:
        """Get parameters for feature extraction"""
        return {
            'sample_rate': self.audio.sample_rate,
            'n_mfcc': 13,  # Could be made configurable
            'n_mels': 128,
            'hop_length': 512,
            'n_fft': 2048
        }
        
    def get_anti_spoofing_params(self) -> Dict[str, Any]:
        """Get parameters for anti-spoofing based on security level"""
        levels = {
            'low': {
                'checks': ['replay'],
                'thresholds': {'replay': 0.6}
            },
            'medium': {
                'checks': ['replay', 'synthetic'],
                'thresholds': {'replay': 0.7, 'synthetic': 0.7}
            },
            'high': {
                'checks': ['replay', 'synthetic', 'liveness', 'environment'],
                'thresholds': {'replay': 0.8, 'synthetic': 0.8, 'liveness': 0.7, 'environment': 0.6}
            }
        }
        
        return levels.get(self.security.anti_spoofing_level, levels['high'])
        
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, settings in updates.items():
            if hasattr(self, section) and isinstance(settings, dict):
                section_obj = getattr(self, section)
                for key, value in settings.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
                        
        self.validate()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'audio': asdict(self.audio),
            'enrollment': asdict(self.enrollment),
            'authentication': asdict(self.authentication),
            'security': asdict(self.security),
            'system': asdict(self.system),
            'performance': asdict(self.performance)
        }
        
    def apply_memory_optimization(self):
        """Apply memory optimization based on system RAM"""
        ram_gb = _get_system_ram_gb()
        logger.info(f"System RAM: {ram_gb}GB - applying optimizations")
        
        if ram_gb <= 16:
            # Apply 16GB optimizations
            self.performance.enable_quantization = True
            self.performance.enable_compression = True
            self.performance.max_memory_mb = min(self.performance.max_memory_mb, 400)
            self.performance.cache_size_mb = min(self.performance.cache_size_mb, 150)
            self.performance.model_unload_timeout = min(self.performance.model_unload_timeout, 60)
            self.performance.aggressive_unload_timeout = min(self.performance.aggressive_unload_timeout, 30)
            
            # Reduce quality for better performance
            self.enrollment.max_samples = min(self.enrollment.max_samples, 50)
            
            logger.info("Applied 16GB RAM optimizations")
            
        elif ram_gb <= 32:
            # Apply 32GB optimizations
            self.performance.max_memory_mb = min(self.performance.max_memory_mb, 800)
            self.performance.cache_size_mb = min(self.performance.cache_size_mb, 300)
            logger.info("Applied 32GB RAM optimizations")
            
        else:
            # 64GB+ systems - use full capabilities
            logger.info("High-memory system detected - using full capabilities")
            
    def get_memory_budget(self) -> Dict[str, int]:
        """Get memory budget allocation for different components"""
        total_mb = self.performance.max_memory_mb
        
        return {
            'ml_models': int(total_mb * 0.5),      # 50% for ML models
            'cache': self.performance.cache_size_mb,  # Fixed cache size
            'audio_buffer': int(total_mb * 0.125),    # 12.5% for audio
            'misc': int(total_mb * 0.125)            # 12.5% for misc
        }
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and recommendations"""
        info = {
            'ram_gb': _get_system_ram_gb(),
            'ram_available_gb': 0,
            'cpu_count': os.cpu_count() or 4,
            'optimizations_applied': [],
            'recommendations': []
        }
        
        if PSUTIL_AVAILABLE:
            vm = psutil.virtual_memory()
            info['ram_available_gb'] = round(vm.available / (1024**3), 1)
            
        # Check optimization status
        if self.performance.enable_quantization:
            info['optimizations_applied'].append('Model quantization')
        if self.performance.enable_compression:
            info['optimizations_applied'].append('Model compression')
        if self.performance.enable_lazy_loading:
            info['optimizations_applied'].append('Lazy loading')
        if self.performance.enable_predictive_loading:
            info['optimizations_applied'].append('Predictive loading')
            
        # Recommendations
        if info['ram_gb'] <= 16:
            if not self.performance.enable_quantization:
                info['recommendations'].append('Enable quantization for better memory usage')
            if self.performance.max_memory_mb > 400:
                info['recommendations'].append('Reduce max_memory_mb to 400 or less')
                
        if info['ram_available_gb'] < 2:
            info['recommendations'].append('Close some applications to free up memory')
            
        return info


# Global configuration instance
_config = None


def get_config() -> VoiceUnlockConfig:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = VoiceUnlockConfig()
    return _config


def reset_config():
    """Reset global configuration"""
    global _config
    _config = None