"""
Context Intelligence Configuration
==================================

Configuration settings for the context intelligence system
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import os


class ContextMode(Enum):
    """Operating modes for context intelligence"""
    STANDARD = "standard"          # Normal operation
    ENHANCED = "enhanced"          # Full context awareness
    MINIMAL = "minimal"            # Minimal context checking
    PROACTIVE = "proactive"        # Proactive assistance
    DEBUG = "debug"                # Debug mode with verbose logging


@dataclass
class ContextConfig:
    """Configuration for context intelligence system"""
    
    # Screen lock detection
    screen_lock_check_interval: float = 1.0  # How often to check screen lock state
    screen_unlock_wait_time: float = 2.0     # Time to wait after unlock before proceeding
    
    # Command patterns that require screen access
    screen_required_patterns: List[str] = None
    
    # System monitoring
    monitor_refresh_interval: float = 5.0    # How often to refresh system state
    monitor_cache_ttl: float = 10.0         # Cache TTL for system states
    
    # Voice unlock integration
    voice_unlock_timeout: float = 30.0      # Timeout for unlock operations
    voice_unlock_retry_count: int = 1       # Number of retries for unlock
    
    # Context awareness features
    enable_screen_lock_detection: bool = True
    enable_app_context: bool = True
    enable_network_context: bool = True
    enable_window_context: bool = True
    
    # Logging and debugging
    verbose_logging: bool = False
    log_execution_steps: bool = True
    
    def __post_init__(self):
        """Initialize default patterns if not provided"""
        if self.screen_required_patterns is None:
            self.screen_required_patterns = [
                # Browser operations
                'open safari', 'open chrome', 'open firefox', 'open browser',
                'search for', 'google', 'look up', 'find online',
                'go to', 'navigate to', 'visit',
                
                # Application operations
                'open', 'launch', 'start', 'run',
                'switch to', 'show me', 'display',
                
                # File operations
                'create', 'edit', 'save', 'close',
                'find file', 'open file', 'open document',
                
                # System operations that need UI
                'take screenshot', 'show desktop', 'minimize',
                'maximize', 'resize', 'move window'
            ]
    
    @classmethod
    def from_env(cls) -> "ContextConfig":
        """Create config from environment variables"""
        config = cls()
        
        # Override from environment if set
        if os.getenv("JARVIS_CONTEXT_VERBOSE"):
            config.verbose_logging = os.getenv("JARVIS_CONTEXT_VERBOSE", "").lower() == "true"
            
        if os.getenv("JARVIS_SCREEN_LOCK_CHECK_INTERVAL"):
            config.screen_lock_check_interval = float(os.getenv("JARVIS_SCREEN_LOCK_CHECK_INTERVAL"))
            
        if os.getenv("JARVIS_DISABLE_SCREEN_LOCK_DETECTION"):
            config.enable_screen_lock_detection = False
            
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "screen_lock_check_interval": self.screen_lock_check_interval,
            "screen_unlock_wait_time": self.screen_unlock_wait_time,
            "screen_required_patterns": self.screen_required_patterns,
            "monitor_refresh_interval": self.monitor_refresh_interval,
            "monitor_cache_ttl": self.monitor_cache_ttl,
            "voice_unlock_timeout": self.voice_unlock_timeout,
            "voice_unlock_retry_count": self.voice_unlock_retry_count,
            "enable_screen_lock_detection": self.enable_screen_lock_detection,
            "enable_app_context": self.enable_app_context,
            "enable_network_context": self.enable_network_context,
            "enable_window_context": self.enable_window_context,
            "verbose_logging": self.verbose_logging,
            "log_execution_steps": self.log_execution_steps
        }


# Global instances
_config = None
_config_manager = None


def get_context_config() -> ContextConfig:
    """Get or create context configuration"""
    global _config
    if _config is None:
        _config = ContextConfig.from_env()
    return _config


def get_config() -> ContextConfig:
    """Alias for get_context_config for compatibility"""
    return get_context_config()


class ConfigManager:
    """Configuration manager for context intelligence"""
    
    def __init__(self):
        self.config = get_context_config()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return getattr(self.config, key, default)
        
    def set(self, key: str, value: Any):
        """Set a configuration value"""
        setattr(self.config, key, value)
        
    def reload(self):
        """Reload configuration from environment"""
        global _config
        _config = ContextConfig.from_env()
        self.config = _config


def get_config_manager() -> ConfigManager:
    """Get or create configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager