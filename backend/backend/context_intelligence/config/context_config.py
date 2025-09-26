"""
Context Intelligence Configuration
=================================
"""

from enum import Enum
from typing import Dict, Any, Optional


class ContextMode(Enum):
    """Context intelligence operation modes"""
    MINIMAL = "minimal"
    STANDARD = "standard"
    ADVANCED = "advanced"


class ContextConfig:
    """Configuration for context intelligence"""
    
    def __init__(self):
        self.mode = ContextMode.STANDARD
        self.monitoring_enabled = True
        self.proactive_enabled = False
        self.config = {
            "monitoring.enabled": True,
            "monitoring.poll_interval": 0.5,
            "proactive.enabled": False,
            "screen_lock.auto_unlock": True,
            "screen_lock.confirm_before_unlock": True
        }
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
        
    def set(self, key: str, value: Any, persist: bool = True):
        """Set configuration value"""
        self.config[key] = value
        
    def get_mode_config(self) -> Dict[str, Any]:
        """Get mode-specific configuration"""
        if self.mode == ContextMode.MINIMAL:
            return {
                "monitoring.enabled": False,
                "proactive.enabled": False
            }
        elif self.mode == ContextMode.ADVANCED:
            return {
                "monitoring.enabled": True,
                "proactive.enabled": True,
                "monitoring.poll_interval": 0.3
            }
        else:  # STANDARD
            return {
                "monitoring.enabled": True,
                "proactive.enabled": False,
                "monitoring.poll_interval": 0.5
            }


# Global configuration instance
_config_manager = None


def get_config_manager() -> ContextConfig:
    """Get or create configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ContextConfig()
    return _config_manager


def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return get_config_manager().get(key, default)
