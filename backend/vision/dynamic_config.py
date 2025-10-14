#!/usr/bin/env python3
"""
Dynamic Configuration System for JARVIS Vision
==============================================

Provides dynamic, non-hardcoded configuration for the intelligent orchestration system.
Supports runtime configuration updates, performance optimization, and adaptive behavior.

Features:
- Dynamic capture priorities
- Adaptive performance tuning
- User preference learning
- Cost optimization
- Pattern recognition thresholds
- Fallback strategies
"""

import json
import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ConfigSource(Enum):
    """Configuration source types"""
    DEFAULT = "default"
    USER_PREFERENCE = "user_preference"
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    ADAPTIVE = "adaptive"

@dataclass
class CaptureConfig:
    """Dynamic capture configuration"""
    max_targets: int = 3
    min_value_threshold: float = 0.3
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "error_keywords": 0.4,
        "app_relevance": 0.3,
        "current_space": 0.2,
        "user_history": 0.1
    })
    fallback_strategies: List[str] = field(default_factory=lambda: [
        "cg_windows_api",
        "multi_space_capture",
        "current_space_capture"
    ])

@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    target_response_time: float = 3.0  # seconds
    max_capture_time: float = 1.0  # seconds
    parallel_capture: bool = True
    cache_ttl: int = 30  # seconds
    performance_monitoring: bool = True
    adaptive_tuning: bool = True

@dataclass
class CostConfig:
    """Cost optimization configuration"""
    max_claude_calls_per_session: int = 5
    prefer_metadata_over_vision: bool = True
    batch_analysis: bool = True
    cost_threshold_per_query: float = 0.05  # USD
    optimize_for_cost: bool = True

@dataclass
class QualityConfig:
    """Quality optimization configuration"""
    min_resolution: tuple = (800, 600)
    preferred_resolution: tuple = (1920, 1080)
    quality_threshold: float = 0.8
    detail_level: str = "high"
    analysis_depth: str = "comprehensive"

class DynamicConfigManager:
    """
    Dynamic configuration manager for intelligent orchestration.
    
    Provides adaptive configuration based on:
    - User preferences and history
    - Performance metrics
    - Cost optimization
    - Quality requirements
    - System capabilities
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._lock = threading.Lock()
        self._config_history: deque = deque(maxlen=100)
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self._user_preferences: Dict[str, Any] = {}
        self._system_capabilities: Dict[str, Any] = {}
        
        # Initialize default configurations
        self._default_configs = {
            ConfigSource.DEFAULT: {
                "capture": CaptureConfig(),
                "performance": PerformanceConfig(),
                "cost": CostConfig(),
                "quality": QualityConfig()
            }
        }
        
        # Current active configuration
        self._active_config = self._default_configs[ConfigSource.DEFAULT].copy()
        self._config_source = ConfigSource.DEFAULT
        
        # Load user preferences and system capabilities
        self._load_user_preferences()
        self._detect_system_capabilities()
        
        # Generate optimized configurations
        self._generate_optimized_configs()
    
    def _load_user_preferences(self):
        """Load user preferences from various sources"""
        try:
            # Load from environment variables
            env_prefs = {}
            for key, value in os.environ.items():
                if key.startswith("JARVIS_"):
                    env_prefs[key[7:].lower()] = value
            
            # Load from config file if exists
            config_file = os.path.expanduser("~/.jarvis/config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    file_prefs = json.load(f)
                    env_prefs.update(file_prefs)
            
            self._user_preferences = env_prefs
            self.logger.info(f"Loaded user preferences: {len(self._user_preferences)} items")
            
        except Exception as e:
            self.logger.warning(f"Failed to load user preferences: {e}")
            self._user_preferences = {}
    
    def _detect_system_capabilities(self):
        """Detect system capabilities for adaptive configuration"""
        try:
            import platform
            import psutil
            
            # System information
            self._system_capabilities = {
                "os": platform.system(),
                "architecture": platform.machine(),
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                "disk_space_gb": psutil.disk_usage('/').free / (1024**3)
            }
            
            # Detect available capture methods
            capture_methods = []
            try:
                import Quartz
                capture_methods.append("cg_windows_api")
            except ImportError:
                pass
            
            try:
                import subprocess
                result = subprocess.run(['yabai', '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    capture_methods.append("yabai")
            except FileNotFoundError:
                pass
            
            self._system_capabilities["capture_methods"] = capture_methods
            
            self.logger.info(f"Detected system capabilities: {self._system_capabilities}")
            
        except Exception as e:
            self.logger.warning(f"Failed to detect system capabilities: {e}")
            self._system_capabilities = {}
    
    def _generate_optimized_configs(self):
        """Generate optimized configurations based on system capabilities and user preferences"""
        
        # Performance optimized configuration
        perf_config = self._default_configs[ConfigSource.DEFAULT].copy()
        perf_config["capture"].max_targets = 2
        perf_config["capture"].min_value_threshold = 0.5
        perf_config["performance"].target_response_time = 2.0
        perf_config["performance"].max_capture_time = 0.5
        self._default_configs[ConfigSource.PERFORMANCE_OPTIMIZED] = perf_config
        
        # Cost optimized configuration
        cost_config = self._default_configs[ConfigSource.DEFAULT].copy()
        cost_config["capture"].max_targets = 1
        cost_config["capture"].min_value_threshold = 0.7
        cost_config["cost"].prefer_metadata_over_vision = True
        cost_config["cost"].max_claude_calls_per_session = 3
        self._default_configs[ConfigSource.COST_OPTIMIZED] = cost_config
        
        # Quality optimized configuration
        quality_config = self._default_configs[ConfigSource.DEFAULT].copy()
        quality_config["capture"].max_targets = 5
        quality_config["capture"].min_value_threshold = 0.2
        quality_config["quality"].detail_level = "maximum"
        quality_config["quality"].analysis_depth = "deep"
        self._default_configs[ConfigSource.QUALITY_OPTIMIZED] = quality_config
        
        # Adaptive configuration based on system capabilities
        adaptive_config = self._default_configs[ConfigSource.DEFAULT].copy()
        
        # Adjust based on available memory
        available_memory = self._system_capabilities.get("available_memory_gb", 8)
        if available_memory < 4:
            adaptive_config["capture"].max_targets = 1
            adaptive_config["performance"].target_response_time = 5.0
        elif available_memory > 16:
            adaptive_config["capture"].max_targets = 4
            adaptive_config["performance"].target_response_time = 2.0
        
        # Adjust based on available capture methods
        capture_methods = self._system_capabilities.get("capture_methods", [])
        if "cg_windows_api" not in capture_methods:
            adaptive_config["capture"].fallback_strategies = [
                "multi_space_capture",
                "current_space_capture"
            ]
        
        self._default_configs[ConfigSource.ADAPTIVE] = adaptive_config
    
    def get_config(self, source: ConfigSource = None) -> Dict[str, Any]:
        """Get configuration for specified source"""
        if source is None:
            source = self._config_source
        
        with self._lock:
            return self._default_configs.get(source, self._default_configs[ConfigSource.DEFAULT]).copy()
    
    def update_config(self, source: ConfigSource, updates: Dict[str, Any]):
        """Update configuration for specified source"""
        with self._lock:
            if source not in self._default_configs:
                self._default_configs[source] = self._default_configs[ConfigSource.DEFAULT].copy()
            
            # Deep update configuration
            self._deep_update_config(self._default_configs[source], updates)
            
            # Record configuration change
            self._config_history.append({
                "timestamp": datetime.now(),
                "source": source.value,
                "updates": updates
            })
            
            self.logger.info(f"Updated configuration for {source.value}: {updates}")
    
    def _deep_update_config(self, config: Dict[str, Any], updates: Dict[str, Any]):
        """Deep update configuration dictionary"""
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._deep_update_config(config[key], value)
            else:
                config[key] = value
    
    def set_active_config(self, source: ConfigSource):
        """Set active configuration source"""
        with self._lock:
            if source in self._default_configs:
                self._active_config = self._default_configs[source].copy()
                self._config_source = source
                self.logger.info(f"Active configuration set to {source.value}")
            else:
                self.logger.warning(f"Configuration source {source.value} not found")
    
    def get_active_config(self) -> Dict[str, Any]:
        """Get currently active configuration"""
        with self._lock:
            return self._active_config.copy()
    
    def optimize_for_performance(self):
        """Optimize configuration for performance"""
        self.set_active_config(ConfigSource.PERFORMANCE_OPTIMIZED)
        
        # Additional performance optimizations based on metrics
        if self._performance_metrics.get("total_time"):
            avg_time = sum(self._performance_metrics["total_time"]) / len(self._performance_metrics["total_time"])
            if avg_time > 3.0:
                # Performance is slow, reduce targets
                config = self.get_active_config()
                config["capture"].max_targets = max(1, config["capture"].max_targets - 1)
                config["capture"].min_value_threshold = min(0.8, config["capture"].min_value_threshold + 0.1)
                self.update_config(ConfigSource.PERFORMANCE_OPTIMIZED, config)
    
    def optimize_for_cost(self):
        """Optimize configuration for cost"""
        self.set_active_config(ConfigSource.COST_OPTIMIZED)
    
    def optimize_for_quality(self):
        """Optimize configuration for quality"""
        self.set_active_config(ConfigSource.QUALITY_OPTIMIZED)
    
    def optimize_adaptively(self):
        """Optimize configuration adaptively based on system state"""
        self.set_active_config(ConfigSource.ADAPTIVE)
        
        # Additional adaptive optimizations
        config = self.get_active_config()
        
        # Adjust based on recent performance
        if self._performance_metrics.get("total_time"):
            recent_times = self._performance_metrics["total_time"][-10:]
            if recent_times:
                avg_recent_time = sum(recent_times) / len(recent_times)
                if avg_recent_time > 4.0:
                    # Recent performance is poor, reduce complexity
                    config["capture"].max_targets = max(1, config["capture"].max_targets - 1)
                    config["capture"].min_value_threshold = min(0.8, config["capture"].min_value_threshold + 0.1)
        
        # Adjust based on system resources
        available_memory = self._system_capabilities.get("available_memory_gb", 8)
        if available_memory < 2:
            config["capture"].max_targets = 1
            config["performance"].target_response_time = 6.0
        
        self.update_config(ConfigSource.ADAPTIVE, config)
    
    def record_performance_metric(self, metric_name: str, value: float):
        """Record performance metric for adaptive optimization"""
        with self._lock:
            self._performance_metrics[metric_name].append(value)
            
            # Keep only recent metrics
            if len(self._performance_metrics[metric_name]) > 50:
                self._performance_metrics[metric_name] = self._performance_metrics[metric_name][-50:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for optimization decisions"""
        with self._lock:
            summary = {}
            for metric_name, values in self._performance_metrics.items():
                if values:
                    summary[metric_name] = {
                        "average": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "count": len(values),
                        "trend": "improving" if len(values) > 1 and values[-1] < values[0] else "stable"
                    }
            return summary
    
    def get_recommendation(self) -> ConfigSource:
        """Get recommended configuration source based on current state"""
        
        # Analyze performance metrics
        perf_summary = self.get_performance_summary()
        
        # Check system resources
        available_memory = self._system_capabilities.get("available_memory_gb", 8)
        
        # Check user preferences
        prefer_performance = self._user_preferences.get("prefer_performance", "false").lower() == "true"
        prefer_cost = self._user_preferences.get("prefer_cost", "false").lower() == "true"
        prefer_quality = self._user_preferences.get("prefer_quality", "false").lower() == "true"
        
        # Make recommendation
        if prefer_performance or (perf_summary.get("total_time", {}).get("average", 0) > 4.0):
            return ConfigSource.PERFORMANCE_OPTIMIZED
        elif prefer_cost:
            return ConfigSource.COST_OPTIMIZED
        elif prefer_quality or available_memory > 16:
            return ConfigSource.QUALITY_OPTIMIZED
        else:
            return ConfigSource.ADAPTIVE
    
    def auto_optimize(self):
        """Automatically optimize configuration based on current state"""
        recommendation = self.get_recommendation()
        
        if recommendation == ConfigSource.PERFORMANCE_OPTIMIZED:
            self.optimize_for_performance()
        elif recommendation == ConfigSource.COST_OPTIMIZED:
            self.optimize_for_cost()
        elif recommendation == ConfigSource.QUALITY_OPTIMIZED:
            self.optimize_for_quality()
        else:
            self.optimize_adaptively()
        
        return recommendation
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of all configurations"""
        with self._lock:
            return {
                "active_source": self._config_source.value,
                "available_sources": [source.value for source in self._default_configs.keys()],
                "system_capabilities": self._system_capabilities,
                "user_preferences": self._user_preferences,
                "performance_summary": self.get_performance_summary(),
                "recommendation": self.get_recommendation().value
            }


# Global instance
_config_manager_instance = None

def get_dynamic_config_manager() -> DynamicConfigManager:
    """Get or create the global configuration manager instance"""
    global _config_manager_instance
    
    if _config_manager_instance is None:
        _config_manager_instance = DynamicConfigManager()
    
    return _config_manager_instance