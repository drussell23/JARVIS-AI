"""
TV Monitor Configuration
========================

All configurable parameters for TV/Display monitoring.
No hardcoding - everything is configurable!

Author: Derek Russell
Date: 2025-10-15
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TVMonitorConfig:
    """Configuration for TV/Display monitoring"""
    
    # Display to monitor
    display_name: str = field(default_factory=lambda: os.getenv("TV_DISPLAY_NAME", "Living Room TV"))
    
    # Monitoring settings
    check_interval_seconds: float = field(default_factory=lambda: float(os.getenv("TV_CHECK_INTERVAL", "10.0")))
    enabled: bool = field(default_factory=lambda: os.getenv("TV_MONITOR_ENABLED", "true").lower() == "true")
    
    # Connection settings
    default_connection_mode: str = field(default_factory=lambda: os.getenv("TV_CONNECTION_MODE", "extend"))  # "extend" or "mirror"
    auto_prompt: bool = field(default_factory=lambda: os.getenv("TV_AUTO_PROMPT", "true").lower() == "true")
    
    # Voice settings
    enable_voice_prompts: bool = field(default_factory=lambda: os.getenv("TV_VOICE_PROMPTS", "true").lower() == "true")
    voice_name: str = field(default_factory=lambda: os.getenv("TV_VOICE_NAME", "Samantha"))  # macOS voice
    voice_rate: int = field(default_factory=lambda: int(os.getenv("TV_VOICE_RATE", "180")))  # Words per minute
    
    # Prompt settings
    prompt_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("TV_PROMPT_TIMEOUT", "30.0")))
    prompt_cooldown_minutes: int = field(default_factory=lambda: int(os.getenv("TV_PROMPT_COOLDOWN", "60")))  # Don't ask again for N minutes
    
    # Detection method preferences
    use_applescript_detection: bool = field(default_factory=lambda: os.getenv("TV_USE_APPLESCRIPT", "true").lower() == "true")
    use_core_graphics_fallback: bool = field(default_factory=lambda: os.getenv("TV_USE_CORE_GRAPHICS", "true").lower() == "true")
    applescript_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("TV_APPLESCRIPT_TIMEOUT", "5.0")))
    
    # Permissions and security
    require_permissions_check: bool = field(default_factory=lambda: os.getenv("TV_REQUIRE_PERMISSIONS", "true").lower() == "true")
    
    # Logging and debugging
    debug_mode: bool = field(default_factory=lambda: os.getenv("TV_DEBUG", "false").lower() == "true")
    log_detection_attempts: bool = field(default_factory=lambda: os.getenv("TV_LOG_DETECTION", "false").lower() == "true")
    
    # Multiple displays support
    additional_displays: List[str] = field(default_factory=list)
    
    @classmethod
    def from_env(cls) -> 'TVMonitorConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Load additional displays from environment
        if additional := os.getenv("TV_ADDITIONAL_DISPLAYS"):
            config.additional_displays = [d.strip() for d in additional.split(",")]
        
        return config
    
    def get_all_monitored_displays(self) -> List[str]:
        """Get all displays to monitor (primary + additional)"""
        displays = [self.display_name]
        displays.extend(self.additional_displays)
        return displays
    
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            "display_name": self.display_name,
            "check_interval_seconds": self.check_interval_seconds,
            "enabled": self.enabled,
            "default_connection_mode": self.default_connection_mode,
            "auto_prompt": self.auto_prompt,
            "enable_voice_prompts": self.enable_voice_prompts,
            "voice_name": self.voice_name,
            "voice_rate": self.voice_rate,
            "prompt_timeout_seconds": self.prompt_timeout_seconds,
            "prompt_cooldown_minutes": self.prompt_cooldown_minutes,
            "use_applescript_detection": self.use_applescript_detection,
            "use_core_graphics_fallback": self.use_core_graphics_fallback,
            "applescript_timeout_seconds": self.applescript_timeout_seconds,
            "require_permissions_check": self.require_permissions_check,
            "debug_mode": self.debug_mode,
            "log_detection_attempts": self.log_detection_attempts,
            "additional_displays": self.additional_displays,
            "all_monitored_displays": self.get_all_monitored_displays()
        }


# Global config instance
TV_CONFIG = TVMonitorConfig.from_env()
