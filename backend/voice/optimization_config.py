"""
Optimization Configuration for 16GB MacBook Pro
Dynamic resource management and performance tuning
"""

import os
import psutil
import platform
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from enum import Enum

class OptimizationLevel(Enum):
    """Performance vs Memory trade-off levels"""
    MAXIMUM_PERFORMANCE = "max_performance"  # Use all available resources
    BALANCED = "balanced"                    # Default, good performance with memory limits
    MEMORY_SAVER = "memory_saver"           # Minimize memory usage
    ULTRA_LIGHT = "ultra_light"             # Extreme memory savings

class ModelPriority(Enum):
    """Model loading priorities"""
    ESSENTIAL = 1      # Always keep loaded
    HIGH = 2          # Keep loaded if possible
    MEDIUM = 3        # Load on demand, unload when not used
    LOW = 4           # Load only when needed, unload immediately

@dataclass
class MemoryConfig:
    """Memory management configuration"""
    # System limits
    total_system_memory_gb: float = field(default_factory=lambda: psutil.virtual_memory().total / (1024**3))
    max_memory_usage_percent: float = 30.0  # Max % of system RAM
    max_memory_usage_mb: int = field(default_factory=lambda: int(psutil.virtual_memory().total * 0.3 / (1024**2)))
    
    # Model memory limits
    max_models_in_memory: int = 3
    model_cache_timeout_seconds: int = 300  # 5 minutes
    
    # Buffer limits
    audio_buffer_max_mb: float = 50.0
    feature_cache_max_mb: float = 100.0
    
    # Thresholds for action
    memory_warning_threshold: float = 80.0  # Start unloading at 80% of limit
    memory_critical_threshold: float = 90.0  # Aggressive unloading at 90%
    
    # Garbage collection
    gc_interval_seconds: float = 60.0
    force_gc_on_model_unload: bool = True

@dataclass
class StreamingConfig:
    """Audio streaming configuration"""
    # Chunk processing
    chunk_size_samples: int = 1024
    chunk_overlap_samples: int = 256
    max_chunks_in_flight: int = 3
    
    # Buffering
    input_buffer_chunks: int = 10
    output_buffer_chunks: int = 5
    
    # Performance
    use_multiprocessing: bool = False  # Use threads instead for lower overhead
    worker_threads: int = 2
    
    # Latency vs throughput
    low_latency_mode: bool = True
    batch_timeout_ms: float = 50.0

@dataclass 
class MacOSAcceleration:
    """macOS-specific acceleration settings"""
    # Core ML
    use_coreml: bool = field(default_factory=lambda: platform.system() == "Darwin")
    coreml_compute_units: str = "ALL"  # "CPU_AND_GPU", "CPU_ONLY", "GPU_ONLY"
    
    # Metal Performance Shaders
    use_metal: bool = field(default_factory=lambda: platform.system() == "Darwin")
    metal_device_id: int = 0
    
    # Accelerate framework
    use_accelerate: bool = field(default_factory=lambda: platform.system() == "Darwin")
    
    # Neural Engine (M1/M2)
    use_neural_engine: bool = field(default_factory=lambda: platform.system() == "Darwin" and platform.processor() == "arm")

@dataclass
class ModelSwapConfig:
    """Model swapping configuration"""
    # Model priorities
    model_priorities: Dict[str, ModelPriority] = field(default_factory=lambda: {
        "wake_word_detector": ModelPriority.ESSENTIAL,
        "vad": ModelPriority.ESSENTIAL,
        "feature_extractor": ModelPriority.HIGH,
        "neural_network": ModelPriority.MEDIUM,
        "svm_classifier": ModelPriority.MEDIUM,
        "anomaly_detector": ModelPriority.LOW,
        "conversation_model": ModelPriority.LOW
    })
    
    # Swap settings
    swap_to_disk: bool = True
    swap_directory: str = field(default_factory=lambda: os.path.join(os.path.expanduser("~"), ".jarvis", "model_swap"))
    compress_on_swap: bool = True
    
    # Preloading
    preload_essential: bool = True
    lazy_load_all_others: bool = True

@dataclass
class OptimizationConfig:
    """Master optimization configuration"""
    # Optimization level
    level: OptimizationLevel = OptimizationLevel.BALANCED
    
    # Sub-configurations  
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    streaming: StreamingConfig = field(default_factory=StreamingConfig)
    macos: MacOSAcceleration = field(default_factory=MacOSAcceleration)
    model_swap: ModelSwapConfig = field(default_factory=ModelSwapConfig)
    
    # Performance monitoring
    enable_profiling: bool = False
    profile_interval_seconds: float = 10.0
    log_performance_metrics: bool = True
    
    # Adaptive optimization
    enable_adaptive_optimization: bool = True
    adaptation_interval_seconds: float = 60.0
    
    @classmethod
    def from_env(cls) -> 'OptimizationConfig':
        """Create config from environment variables"""
        config = cls()
        
        # Optimization level
        if level := os.getenv("JARVIS_OPTIMIZATION_LEVEL"):
            try:
                config.level = OptimizationLevel(level)
            except ValueError:
                pass
        
        # Memory settings
        if val := os.getenv("JARVIS_MAX_MEMORY_PERCENT"):
            config.memory.max_memory_usage_percent = float(val)
        if val := os.getenv("JARVIS_MAX_MODELS_IN_MEMORY"):
            config.memory.max_models_in_memory = int(val)
            
        # Streaming settings
        if val := os.getenv("JARVIS_CHUNK_SIZE"):
            config.streaming.chunk_size_samples = int(val)
        if val := os.getenv("JARVIS_LOW_LATENCY"):
            config.streaming.low_latency_mode = val.lower() == "true"
            
        # macOS settings
        if val := os.getenv("JARVIS_USE_COREML"):
            config.macos.use_coreml = val.lower() == "true"
        if val := os.getenv("JARVIS_USE_METAL"):
            config.macos.use_metal = val.lower() == "true"
            
        # Apply optimization level presets
        config._apply_optimization_level()
        
        return config
    
    def _apply_optimization_level(self):
        """Apply preset configurations based on optimization level"""
        if self.level == OptimizationLevel.MAXIMUM_PERFORMANCE:
            self.memory.max_memory_usage_percent = 50.0
            self.memory.max_models_in_memory = 10
            self.streaming.low_latency_mode = True
            self.streaming.worker_threads = 4
            self.model_swap.lazy_load_all_others = False
            
        elif self.level == OptimizationLevel.MEMORY_SAVER:
            self.memory.max_memory_usage_percent = 20.0
            self.memory.max_models_in_memory = 2
            self.streaming.chunk_size_samples = 512
            self.streaming.worker_threads = 1
            self.model_swap.compress_on_swap = True
            
        elif self.level == OptimizationLevel.ULTRA_LIGHT:
            self.memory.max_memory_usage_percent = 15.0
            self.memory.max_models_in_memory = 1
            self.memory.audio_buffer_max_mb = 25.0
            self.streaming.use_multiprocessing = False
            self.streaming.worker_threads = 1
            self.macos.coreml_compute_units = "CPU_ONLY"
    
    def get_memory_limit_bytes(self) -> int:
        """Get memory limit in bytes"""
        total_memory = psutil.virtual_memory().total
        return int(total_memory * self.memory.max_memory_usage_percent / 100)
    
    def should_use_acceleration(self) -> bool:
        """Check if any acceleration is available and enabled"""
        return (self.macos.use_coreml or 
                self.macos.use_metal or 
                self.macos.use_neural_engine)

# Global config instance
OPTIMIZATION_CONFIG = OptimizationConfig.from_env()

# Convenience presets
PRESETS = {
    "16gb_macbook_pro": OptimizationConfig(
        level=OptimizationLevel.BALANCED,
        memory=MemoryConfig(
            max_memory_usage_percent=25.0,  # ~4GB for JARVIS
            max_models_in_memory=3,
            audio_buffer_max_mb=50.0
        ),
        streaming=StreamingConfig(
            chunk_size_samples=1024,
            low_latency_mode=True,
            worker_threads=2
        )
    ),
    "8gb_mac": OptimizationConfig(
        level=OptimizationLevel.MEMORY_SAVER,
        memory=MemoryConfig(
            max_memory_usage_percent=20.0,  # ~1.6GB
            max_models_in_memory=2
        )
    ),
    "32gb_mac_studio": OptimizationConfig(
        level=OptimizationLevel.MAXIMUM_PERFORMANCE,
        memory=MemoryConfig(
            max_memory_usage_percent=40.0,  # ~12GB
            max_models_in_memory=10
        ),
        streaming=StreamingConfig(
            worker_threads=6
        )
    )
}

def get_preset(name: str) -> OptimizationConfig:
    """Get a configuration preset"""
    return PRESETS.get(name, OptimizationConfig())