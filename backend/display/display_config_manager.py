#!/usr/bin/env python3
"""
Display Configuration Manager for JARVIS
=========================================

Manages display monitoring configuration with:
- Dynamic updates
- Validation
- Persistence
- Migration support
- Configuration presets

Author: Derek Russell
Date: 2025-10-15
Version: 1.0
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import copy

logger = logging.getLogger(__name__)


@dataclass
class DisplayConfigVersion:
    """Configuration version information"""
    version: str
    updated_at: str
    updated_by: str = "system"


class DisplayConfigManager:
    """
    Configuration manager for display monitoring

    Features:
    - Load/save configuration
    - Validation
    - Dynamic updates
    - Configuration presets
    - Migration support
    """

    # Configuration schema version
    CURRENT_VERSION = "2.0"

    # Default configuration
    DEFAULT_CONFIG = {
        "display_monitoring": {
            "enabled": True,
            "check_interval_seconds": 10.0,
            "startup_delay_seconds": 2.0,
            "detection_methods": ["applescript", "coregraphics"],
            "preferred_detection_method": "applescript"
        },
        "displays": {
            "monitored_displays": [],
            "ignored_displays": ["Built-in Retina Display"]
        },
        "voice_integration": {
            "enabled": True,
            "voice_engine": "edge_tts",
            "voice_name": "en-US-GuyNeural",
            "prompt_template": "Sir, I see your {display_name} is now available. Would you like to extend your display to it?",
            "connection_success_message": "Connected to {display_name}, sir.",
            "connection_failure_message": "Unable to connect to {display_name}. {error_detail}",
            "speak_on_detection": True,
            "speak_on_connection": True,
            "speak_on_disconnection": False
        },
        "applescript": {
            "enabled": True,
            "timeout_seconds": 5.0,
            "retry_attempts": 3,
            "retry_delay_seconds": 0.5,
            "control_center_process": "ControlCenter",
            "screen_mirroring_menu_item": "Screen Mirroring",
            "filter_system_items": ["Turn Display Mirroring Off", ""]
        },
        "coregraphics": {
            "enabled": True,
            "max_displays": 32,
            "detect_airplay": True,
            "detect_external": True,
            "exclude_builtin": True
        },
        "yabai": {
            "enabled": False,
            "command_timeout": 3.0
        },
        "caching": {
            "enabled": True,
            "screenshot_ttl_seconds": 30,
            "ocr_result_ttl_seconds": 300,
            "display_list_ttl_seconds": 5,
            "max_cache_size_mb": 100
        },
        "performance": {
            "parallel_detection": True,
            "max_concurrent_operations": 4,
            "graceful_degradation": True,
            "fallback_on_error": True
        },
        "notifications": {
            "enabled": True,
            "notification_types": ["display_connected", "display_disconnected", "detection_error"],
            "use_system_notifications": False,
            "use_voice_notifications": True
        },
        "logging": {
            "level": "INFO",
            "log_detection_events": True,
            "log_applescript_commands": False,
            "log_performance_metrics": True
        },
        "security": {
            "require_user_consent_first_time": True,
            "remember_consent": True,
            "auto_connect_only_known_displays": True
        },
        "advanced": {
            "multi_monitor_support": True,
            "temporal_tracking": False,
            "predictive_detection": False,
            "learn_user_patterns": False
        }
    }

    # Configuration presets
    PRESETS = {
        "minimal": {
            "display_monitoring": {
                "enabled": True,
                "check_interval_seconds": 30.0,
                "detection_methods": ["applescript"]
            },
            "voice_integration": {
                "enabled": False
            },
            "caching": {
                "enabled": False
            }
        },
        "performance": {
            "display_monitoring": {
                "check_interval_seconds": 5.0
            },
            "performance": {
                "parallel_detection": True,
                "max_concurrent_operations": 8
            },
            "caching": {
                "enabled": True,
                "display_list_ttl_seconds": 2
            }
        },
        "voice_focused": {
            "voice_integration": {
                "enabled": True,
                "speak_on_detection": True,
                "speak_on_connection": True,
                "speak_on_disconnection": True
            },
            "notifications": {
                "use_voice_notifications": True
            }
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to configuration file
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            # Default to backend/config/display_monitor_config.json
            self.config_path = Path(__file__).parent.parent / 'config' / 'display_monitor_config.json'

        self.config: Dict = {}
        self.config_version: Optional[DisplayConfigVersion] = None

        # Load or create config
        if self.config_path.exists():
            self.load()
        else:
            self.config = copy.deepcopy(self.DEFAULT_CONFIG)
            self._update_timestamp()
            self.save()

    def load(self) -> Dict:
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            # Validate and migrate if needed
            self._validate_and_migrate()

            logger.info(f"[CONFIG] Loaded configuration from {self.config_path}")
            return self.config

        except json.JSONDecodeError as e:
            logger.error(f"[CONFIG] Invalid JSON in {self.config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"[CONFIG] Error loading config: {e}")
            raise

    def save(self) -> bool:
        """Save configuration to file"""
        try:
            # Ensure directory exists
            self.config_path.parent.mkdir(parents=True, exist_ok=True)

            # Update timestamp
            self._update_timestamp()

            # Save to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"[CONFIG] Saved configuration to {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"[CONFIG] Error saving config: {e}")
            return False

    def _validate_and_migrate(self):
        """Validate configuration and migrate if needed"""
        # Check if migration is needed
        config_version = self.config.get('version', '1.0')

        if config_version != self.CURRENT_VERSION:
            logger.info(f"[CONFIG] Migrating from v{config_version} to v{self.CURRENT_VERSION}")
            self._migrate_config(config_version)

        # Validate required keys
        self._validate_config()

    def _migrate_config(self, from_version: str):
        """Migrate configuration from old version"""
        # Merge with defaults to add any new keys
        self.config = self._deep_merge(self.DEFAULT_CONFIG, self.config)

        # Version-specific migrations
        if from_version == "1.0":
            # Example: Add new fields introduced in 2.0
            pass

        self.config['version'] = self.CURRENT_VERSION

    def _deep_merge(self, default: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries"""
        result = copy.deepcopy(default)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _validate_config(self):
        """Validate configuration structure"""
        required_sections = [
            'display_monitoring',
            'displays',
            'voice_integration',
            'applescript',
            'coregraphics'
        ]

        for section in required_sections:
            if section not in self.config:
                logger.warning(f"[CONFIG] Missing section: {section}, using default")
                self.config[section] = self.DEFAULT_CONFIG.get(section, {})

    def _update_timestamp(self):
        """Update last_updated timestamp"""
        self.config['last_updated'] = datetime.now().isoformat()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key

        Args:
            key: Dot-separated key path (e.g., 'voice_integration.enabled')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """
        Set configuration value

        Args:
            key: Dot-separated key path
            value: Value to set
            save: Whether to save immediately

        Returns:
            Success status
        """
        keys = key.split('.')
        config = self.config

        # Navigate to parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set value
        config[keys[-1]] = value

        logger.info(f"[CONFIG] Set {key} = {value}")

        if save:
            return self.save()

        return True

    def add_display(self, display_config: Dict, save: bool = True) -> bool:
        """
        Add a monitored display to configuration

        Args:
            display_config: Display configuration dictionary
            save: Whether to save immediately

        Returns:
            Success status
        """
        # Validate required fields
        required_fields = ['id', 'name', 'display_type']
        for field in required_fields:
            if field not in display_config:
                logger.error(f"[CONFIG] Missing required field: {field}")
                return False

        # Add defaults
        defaults = {
            'aliases': [],
            'auto_connect': False,
            'auto_prompt': True,
            'connection_mode': 'extend',
            'priority': 1,
            'enabled': True
        }

        for key, default_value in defaults.items():
            if key not in display_config:
                display_config[key] = default_value

        # Add to monitored displays
        if 'monitored_displays' not in self.config['displays']:
            self.config['displays']['monitored_displays'] = []

        # Check if display already exists
        existing = next(
            (d for d in self.config['displays']['monitored_displays'] if d['id'] == display_config['id']),
            None
        )

        if existing:
            logger.warning(f"[CONFIG] Display {display_config['id']} already exists, updating")
            self.config['displays']['monitored_displays'].remove(existing)

        self.config['displays']['monitored_displays'].append(display_config)
        logger.info(f"[CONFIG] Added display: {display_config['name']}")

        if save:
            return self.save()

        return True

    def remove_display(self, display_id: str, save: bool = True) -> bool:
        """
        Remove a monitored display

        Args:
            display_id: Display ID to remove
            save: Whether to save immediately

        Returns:
            Success status
        """
        if 'monitored_displays' not in self.config['displays']:
            return False

        original_count = len(self.config['displays']['monitored_displays'])

        self.config['displays']['monitored_displays'] = [
            d for d in self.config['displays']['monitored_displays']
            if d['id'] != display_id
        ]

        if len(self.config['displays']['monitored_displays']) < original_count:
            logger.info(f"[CONFIG] Removed display: {display_id}")
            if save:
                return self.save()
            return True

        logger.warning(f"[CONFIG] Display {display_id} not found")
        return False

    def get_monitored_displays(self) -> List[Dict]:
        """Get list of monitored displays"""
        return self.config.get('displays', {}).get('monitored_displays', [])

    def apply_preset(self, preset_name: str, save: bool = True) -> bool:
        """
        Apply configuration preset

        Args:
            preset_name: Name of preset (minimal, performance, voice_focused)
            save: Whether to save immediately

        Returns:
            Success status
        """
        if preset_name not in self.PRESETS:
            logger.error(f"[CONFIG] Unknown preset: {preset_name}")
            return False

        preset = self.PRESETS[preset_name]

        # Deep merge preset with current config
        self.config = self._deep_merge(self.config, preset)

        logger.info(f"[CONFIG] Applied preset: {preset_name}")

        if save:
            return self.save()

        return True

    def reset_to_defaults(self, save: bool = True) -> bool:
        """
        Reset configuration to defaults

        Args:
            save: Whether to save immediately

        Returns:
            Success status
        """
        # Keep monitored displays
        monitored_displays = self.get_monitored_displays()

        # Reset to defaults
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)

        # Restore monitored displays
        self.config['displays']['monitored_displays'] = monitored_displays

        logger.info("[CONFIG] Reset to defaults")

        if save:
            return self.save()

        return True

    def export_config(self, output_path: str) -> bool:
        """
        Export configuration to file

        Args:
            output_path: Path to export file

        Returns:
            Success status
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"[CONFIG] Exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"[CONFIG] Export error: {e}")
            return False

    def import_config(self, input_path: str, merge: bool = False, save: bool = True) -> bool:
        """
        Import configuration from file

        Args:
            input_path: Path to import file
            merge: Whether to merge with current config or replace
            save: Whether to save immediately

        Returns:
            Success status
        """
        try:
            with open(input_path, 'r') as f:
                imported_config = json.load(f)

            if merge:
                self.config = self._deep_merge(self.config, imported_config)
            else:
                self.config = imported_config

            # Validate
            self._validate_and_migrate()

            logger.info(f"[CONFIG] Imported from {input_path}")

            if save:
                return self.save()

            return True

        except Exception as e:
            logger.error(f"[CONFIG] Import error: {e}")
            return False

    def get_full_config(self) -> Dict:
        """Get full configuration"""
        return copy.deepcopy(self.config)

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            "version": self.config.get('version', self.CURRENT_VERSION),
            "last_updated": self.config.get('last_updated', 'unknown'),
            "monitoring_enabled": self.get('display_monitoring.enabled', False),
            "monitored_displays_count": len(self.get_monitored_displays()),
            "voice_enabled": self.get('voice_integration.enabled', False),
            "detection_methods": self.get('display_monitoring.detection_methods', []),
            "caching_enabled": self.get('caching.enabled', False),
            "config_path": str(self.config_path)
        }


# Singleton instance
_config_manager: Optional[DisplayConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> DisplayConfigManager:
    """Get singleton configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = DisplayConfigManager(config_path)
    return _config_manager


if __name__ == "__main__":
    # Test the config manager
    logging.basicConfig(level=logging.INFO)

    manager = get_config_manager()

    print("Configuration Summary:")
    print(json.dumps(manager.get_summary(), indent=2))

    print("\nMonitored Displays:")
    for display in manager.get_monitored_displays():
        print(f"  - {display['name']} ({display['id']})")
