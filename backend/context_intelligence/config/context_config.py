"""
Context Intelligence Configuration
==================================

Configuration settings for the context intelligence system.

This module provides configuration classes and utilities for managing context
intelligence settings, including screen lock detection, system monitoring,
voice unlock integration, and various context awareness features.

The configuration system supports environment variable overrides and provides
both singleton access patterns and manager-based configuration handling.

Example:
    >>> config = get_context_config()
    >>> config.enable_screen_lock_detection
    True
    >>> manager = get_config_manager()
    >>> manager.set('verbose_logging', True)
"""

from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import os


class ContextMode(Enum):
    """Operating modes for context intelligence.
    
    Defines different operational modes that control the behavior and
    level of context awareness in the system.
    
    Attributes:
        STANDARD: Normal operation with standard context checking
        ENHANCED: Full context awareness with all features enabled
        MINIMAL: Minimal context checking for performance
        PROACTIVE: Proactive assistance with predictive features
        DEBUG: Debug mode with verbose logging and detailed output
    """
    STANDARD = "standard"          # Normal operation
    ENHANCED = "enhanced"          # Full context awareness
    MINIMAL = "minimal"            # Minimal context checking
    PROACTIVE = "proactive"        # Proactive assistance
    DEBUG = "debug"                # Debug mode with verbose logging


@dataclass
class ContextConfig:
    """Configuration for context intelligence system.
    
    Contains all configuration parameters for the context intelligence system,
    including timing settings, feature toggles, and operational parameters.
    
    Attributes:
        screen_lock_check_interval: Interval in seconds for checking screen lock state
        screen_unlock_wait_time: Time to wait after unlock before proceeding
        screen_required_patterns: List of command patterns requiring screen access
        monitor_refresh_interval: Interval for refreshing system state monitoring
        monitor_cache_ttl: Time-to-live for cached system states
        voice_unlock_timeout: Timeout for voice unlock operations
        voice_unlock_retry_count: Number of retries for unlock operations
        enable_screen_lock_detection: Whether to enable screen lock detection
        enable_app_context: Whether to enable application context awareness
        enable_network_context: Whether to enable network context awareness
        enable_window_context: Whether to enable window context awareness
        verbose_logging: Whether to enable verbose logging output
        log_execution_steps: Whether to log detailed execution steps
    """
    
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
    
    def __post_init__(self) -> None:
        """Initialize default patterns if not provided.
        
        Sets up default screen-required command patterns if none were
        provided during initialization. These patterns help identify
        commands that require screen access.
        """
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
        """Create config from environment variables.
        
        Creates a ContextConfig instance with default values, then overrides
        specific settings based on environment variables if they are set.
        
        Returns:
            ContextConfig: Configuration instance with environment overrides applied
            
        Example:
            >>> import os
            >>> os.environ['JARVIS_CONTEXT_VERBOSE'] = 'true'
            >>> config = ContextConfig.from_env()
            >>> config.verbose_logging
            True
        """
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
        """Convert config to dictionary.
        
        Converts the configuration object to a dictionary representation
        for serialization, logging, or API responses.
        
        Returns:
            Dict[str, Any]: Dictionary containing all configuration values
            
        Example:
            >>> config = ContextConfig()
            >>> config_dict = config.to_dict()
            >>> config_dict['enable_screen_lock_detection']
            True
        """
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
    """Get or create context configuration.
    
    Returns the global context configuration instance, creating it from
    environment variables if it doesn't exist. Uses singleton pattern
    to ensure consistent configuration across the application.
    
    Returns:
        ContextConfig: The global context configuration instance
        
    Example:
        >>> config = get_context_config()
        >>> config.screen_lock_check_interval
        1.0
    """
    global _config
    if _config is None:
        _config = ContextConfig.from_env()
    return _config


def get_config() -> ContextConfig:
    """Alias for get_context_config for compatibility.
    
    Provides a shorter alias for accessing the context configuration.
    This function exists for backward compatibility and convenience.
    
    Returns:
        ContextConfig: The global context configuration instance
    """
    return get_context_config()


class ConfigManager:
    """Configuration manager for context intelligence.
    
    Provides a management interface for context configuration with
    methods to get, set, and reload configuration values dynamically.
    
    Attributes:
        config: The managed ContextConfig instance
    """
    
    def __init__(self) -> None:
        """Initialize the configuration manager.
        
        Creates a new ConfigManager instance with the current global
        context configuration.
        """
        self.config = get_context_config()
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.
        
        Retrieves a configuration value by key name, returning a default
        value if the key doesn't exist.
        
        Args:
            key: The configuration key to retrieve
            default: Default value to return if key doesn't exist
            
        Returns:
            Any: The configuration value or default if not found
            
        Example:
            >>> manager = ConfigManager()
            >>> manager.get('verbose_logging', False)
            False
        """
        return getattr(self.config, key, default)
        
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value.
        
        Updates a configuration value by key name. The key must be a valid
        attribute of the ContextConfig class.
        
        Args:
            key: The configuration key to set
            value: The value to set for the key
            
        Raises:
            AttributeError: If the key is not a valid configuration attribute
            
        Example:
            >>> manager = ConfigManager()
            >>> manager.set('verbose_logging', True)
        """
        setattr(self.config, key, value)
        
    def reload(self) -> None:
        """Reload configuration from environment.
        
        Reloads the configuration from environment variables, updating
        both the global configuration instance and this manager's config.
        This is useful for picking up configuration changes without
        restarting the application.
        
        Example:
            >>> manager = ConfigManager()
            >>> manager.reload()  # Picks up new environment variables
        """
        global _config
        _config = ContextConfig.from_env()
        self.config = _config


def get_config_manager() -> ConfigManager:
    """Get or create configuration manager.
    
    Returns the global configuration manager instance, creating it if
    it doesn't exist. Uses singleton pattern to ensure consistent
    configuration management across the application.
    
    Returns:
        ConfigManager: The global configuration manager instance
        
    Example:
        >>> manager = get_config_manager()
        >>> manager.get('enable_screen_lock_detection')
        True
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager