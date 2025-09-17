"""
Claude Vision Analyzer - Advanced screen understanding using Claude's vision capabilities
Fully dynamic and configurable - NO HARDCODING
Optimized for macOS with 16GB RAM - includes caching, compression, and memory management
Integrates all enhanced memory-optimized components:
- continuous_screen_analyzer.py (MemoryAwareScreenAnalyzer)
- window_analysis.py (MemoryAwareWindowAnalyzer)
- window_relationship_detector.py (ConfigurableWindowRelationshipDetector)
- swift_vision_integration.py (MemoryAwareSwiftVisionIntegration)
- memory_efficient_vision_analyzer.py (MemoryEfficientVisionAnalyzer)
- vision_system_claude_only.py (SimplifiedVisionSystem)
"""

import base64
import io
import hashlib
import asyncio
import time
import gc
import os
import sys
import re
import subprocess
import platform
from typing import Dict, List, Optional, Any, Tuple, Union, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps
import numpy as np
from anthropic import Anthropic
import json
import logging
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)

# Import Vision Intelligence System
try:
    from .intelligence import (
        VisionIntelligenceBridge,
        get_vision_intelligence_bridge,
        VisualStateManagementSystem
    )
    from .intelligence.vsms_core import get_vsms, StateCategory
    from .intelligence.state_intelligence import get_state_intelligence
    VISION_INTELLIGENCE_AVAILABLE = True
    VSMS_CORE_AVAILABLE = True
except ImportError as e:
    VISION_INTELLIGENCE_AVAILABLE = False
    VSMS_CORE_AVAILABLE = False
    VisionIntelligenceBridge = None
    get_vsms = None
    logger.warning(f"Vision Intelligence System not fully available - install with ./intelligence/build.sh: {e}")

@dataclass
class VisionConfig:
    """Dynamic configuration for vision analyzer with memory safety"""
    # Image processing
    max_image_dimension: int = field(default_factory=lambda: int(os.getenv('VISION_MAX_IMAGE_DIM', '1536')))
    jpeg_quality: int = field(default_factory=lambda: int(os.getenv('VISION_JPEG_QUALITY', '85')))
    compression_enabled: bool = field(default_factory=lambda: os.getenv('VISION_COMPRESSION', 'true').lower() == 'true')
    
    # API settings
    max_tokens: int = field(default_factory=lambda: int(os.getenv('VISION_MAX_TOKENS', '1500')))
    model_name: str = field(default_factory=lambda: os.getenv('VISION_MODEL', 'claude-3-5-sonnet-20241022'))
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv('VISION_MAX_CONCURRENT', '10')))  # Safe default
    api_timeout: int = field(default_factory=lambda: int(os.getenv('VISION_API_TIMEOUT', '30')))
    
    # Cache settings
    cache_enabled: bool = field(default_factory=lambda: os.getenv('VISION_CACHE_ENABLED', 'true').lower() == 'true')
    cache_size_mb: int = field(default_factory=lambda: int(os.getenv('VISION_CACHE_SIZE_MB', '100')))
    cache_max_entries: int = field(default_factory=lambda: int(os.getenv('VISION_CACHE_ENTRIES', '50')))
    cache_ttl_minutes: int = field(default_factory=lambda: int(os.getenv('VISION_CACHE_TTL_MIN', '30')))
    
    # Performance settings
    memory_threshold_percent: float = field(default_factory=lambda: float(os.getenv('VISION_MEMORY_THRESHOLD', '60')))  # More aggressive
    cpu_threshold_percent: float = field(default_factory=lambda: float(os.getenv('VISION_CPU_THRESHOLD', '70')))
    thread_pool_size: int = field(default_factory=lambda: int(os.getenv('VISION_THREAD_POOL', '2')))
    
    # Memory safety settings - dynamic based on system
    process_memory_limit_mb: int = field(default_factory=lambda: VisionConfig._calculate_process_memory_limit())
    memory_warning_threshold_mb: int = field(default_factory=lambda: VisionConfig._calculate_memory_warning_threshold())
    min_system_available_gb: float = field(default_factory=lambda: VisionConfig._calculate_min_system_available())
    enable_memory_safety: bool = field(default_factory=lambda: os.getenv('VISION_MEMORY_SAFETY', 'true').lower() == 'true')
    reject_on_memory_pressure: bool = field(default_factory=lambda: os.getenv('VISION_REJECT_ON_MEMORY', 'true').lower() == 'true')
    
    # Feature flags
    enable_metrics: bool = field(default_factory=lambda: os.getenv('VISION_METRICS', 'true').lower() == 'true')
    enable_entity_extraction: bool = field(default_factory=lambda: os.getenv('VISION_EXTRACT_ENTITIES', 'true').lower() == 'true')
    enable_action_detection: bool = field(default_factory=lambda: os.getenv('VISION_DETECT_ACTIONS', 'true').lower() == 'true')
    enable_screen_sharing: bool = field(default_factory=lambda: os.getenv('VISION_SCREEN_SHARING', 'true').lower() == 'true')
    enable_continuous_monitoring: bool = field(default_factory=lambda: os.getenv('VISION_CONTINUOUS_ENABLED', 'true').lower() == 'true')
    enable_video_streaming: bool = field(default_factory=lambda: True)  # Always enable video streaming
    prefer_video_over_screenshots: bool = field(default_factory=lambda: os.getenv('VISION_PREFER_VIDEO', 'true').lower() == 'true')
    
    # Vision Intelligence enhancement
    vision_intelligence_enabled: bool = field(default_factory=lambda: os.getenv('VISION_INTELLIGENCE_ENABLED', 'false').lower() == 'true')
    vision_intelligence_confidence_threshold: float = field(default_factory=lambda: float(os.getenv('VISION_INTELLIGENCE_CONFIDENCE', '0.7')))
    
    @staticmethod
    def _calculate_process_memory_limit() -> int:
        """Calculate dynamic process memory limit based on system RAM"""
        try:
            import psutil
            vm = psutil.virtual_memory()
            total_gb = vm.total / (1024 * 1024 * 1024)
            available_mb = vm.available / (1024 * 1024)
            
            # Use 25% of total system RAM as process limit
            dynamic_limit = int(total_gb * 1024 * 0.25)
            
            # But don't exceed 50% of currently available
            available_limit = int(available_mb * 0.5)
            
            # Apply reasonable bounds
            final_limit = min(dynamic_limit, available_limit, 4096)  # Cap at 4GB
            final_limit = max(final_limit, 512)  # At least 512MB
            
            logger.info(f"Vision process memory limit: {final_limit}MB (25% of {total_gb:.1f}GB total)")
            return final_limit
        except:
            return 2048  # Default 2GB
    
    @staticmethod
    def _calculate_memory_warning_threshold() -> int:
        """Calculate memory warning threshold"""
        limit = VisionConfig._calculate_process_memory_limit()
        # Warning at 75% of limit
        return int(limit * 0.75)
    
    @staticmethod
    def _calculate_min_system_available() -> float:
        """Calculate minimum system available RAM in GB"""
        try:
            import psutil
            total_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
            # Keep at least 15% of total RAM available
            min_available = total_gb * 0.15
            # But at least 1GB and no more than 4GB
            return max(1.0, min(min_available, 4.0))
        except:
            return 2.0  # Default 2GB
    
    # VSMS Core configuration
    vsms_core_enabled: bool = field(default_factory=lambda: os.getenv('VSMS_CORE_ENABLED', 'false').lower() == 'true')
    vsms_state_learning_enabled: bool = field(default_factory=lambda: os.getenv('VSMS_STATE_LEARNING', 'true').lower() == 'true')
    vsms_pattern_detection_enabled: bool = field(default_factory=lambda: os.getenv('VSMS_PATTERN_DETECTION', 'true').lower() == 'true')
    
    # Semantic Scene Graph configuration
    scene_graph_enabled: bool = field(default_factory=lambda: os.getenv('SCENE_GRAPH_ENABLED', 'true').lower() == 'true')
    scene_graph_element_detection: bool = field(default_factory=lambda: os.getenv('SCENE_GRAPH_ELEMENTS', 'true').lower() == 'true')
    scene_graph_relationship_discovery: bool = field(default_factory=lambda: os.getenv('SCENE_GRAPH_RELATIONSHIPS', 'true').lower() == 'true')
    
    # Temporal Context Engine configuration
    temporal_context_enabled: bool = field(default_factory=lambda: os.getenv('TEMPORAL_CONTEXT_ENABLED', 'true').lower() == 'true')
    temporal_pattern_extraction: bool = field(default_factory=lambda: os.getenv('TEMPORAL_PATTERNS', 'true').lower() == 'true')
    temporal_prediction_enabled: bool = field(default_factory=lambda: os.getenv('TEMPORAL_PREDICTIONS', 'true').lower() == 'true')
    
    # Activity Recognition Engine configuration
    activity_recognition_enabled: bool = field(default_factory=lambda: os.getenv('ACTIVITY_RECOGNITION_ENABLED', 'true').lower() == 'true')
    task_inference_enabled: bool = field(default_factory=lambda: os.getenv('TASK_INFERENCE', 'true').lower() == 'true')
    progress_monitoring_enabled: bool = field(default_factory=lambda: os.getenv('PROGRESS_MONITORING', 'true').lower() == 'true')
    
    vision_intelligence_consensus: bool = field(default_factory=lambda: os.getenv('VISION_INTELLIGENCE_CONSENSUS', 'true').lower() == 'true')
    state_persistence_enabled: bool = field(default_factory=lambda: os.getenv('VISION_STATE_PERSISTENCE', 'true').lower() == 'true')
    
    # VSMS Core settings
    enable_vsms_core: bool = field(default_factory=lambda: os.getenv('VSMS_CORE_ENABLED', 'true').lower() == 'true')
    vsms_track_workflows: bool = field(default_factory=lambda: os.getenv('VSMS_TRACK_WORKFLOWS', 'true').lower() == 'true')
    vsms_detect_anomalies: bool = field(default_factory=lambda: os.getenv('VSMS_DETECT_ANOMALIES', 'true').lower() == 'true')
    vsms_personalization: bool = field(default_factory=lambda: os.getenv('VSMS_PERSONALIZATION', 'true').lower() == 'true')
    vsms_stuck_threshold_minutes: int = field(default_factory=lambda: int(os.getenv('VSMS_STUCK_THRESHOLD', '5')))
    
    @classmethod
    def from_file(cls, config_path: str) -> 'VisionConfig':
        """Load configuration from JSON file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                return cls(**config_data)
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'max_image_dimension': self.max_image_dimension,
            'jpeg_quality': self.jpeg_quality,
            'compression_enabled': self.compression_enabled,
            'max_tokens': self.max_tokens,
            'model_name': self.model_name,
            'max_concurrent_requests': self.max_concurrent_requests,
            'cache_enabled': self.cache_enabled,
            'cache_size_mb': self.cache_size_mb,
            'cache_ttl_minutes': self.cache_ttl_minutes,
            'memory_threshold_percent': self.memory_threshold_percent,
            'enable_metrics': self.enable_metrics,
            # Memory safety settings
            'process_memory_limit_mb': self.process_memory_limit_mb,
            'memory_warning_threshold_mb': self.memory_warning_threshold_mb,
            'min_system_available_gb': self.min_system_available_gb,
            'enable_memory_safety': self.enable_memory_safety,
            'reject_on_memory_pressure': self.reject_on_memory_pressure
        }

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    result: Dict[str, Any]
    timestamp: datetime
    prompt_hash: str
    image_hash: str
    access_count: int = 0
    size_bytes: int = 0

@dataclass
class AnalysisMetrics:
    """Performance metrics for analysis"""
    preprocessing_time: float = 0.0
    api_call_time: float = 0.0
    parsing_time: float = 0.0
    total_time: float = 0.0
    cache_hit: bool = False
    image_size_original: int = 0
    image_size_compressed: int = 0
    compression_ratio: float = 0.0
    vision_intelligence_time: float = 0.0  # Time for Vision Intelligence analysis
    vsms_core_time: float = 0.0  # Time for VSMS Core analysis
    quadtree_time: float = 0.0  # Time for Quadtree spatial analysis
    predictive_hit: bool = False  # Whether result came from predictive engine
    # Region-based optimization metrics
    regions_extracted: int = 0  # Number of regions extracted from image
    coverage_ratio: float = 0.0  # Percentage of screen covered by regions
    processing_strategy: str = ""  # Strategy used: 'full_standard', 'full_compressed', 'region_composite'
    orchestrator_time: float = 0.0  # Time for orchestrator processing
    system_mode: str = "normal"  # System mode during processing
    orchestrator_cache_hits: int = 0  # Cache hits from orchestrator
    orchestrator_api_saved: int = 0  # API calls saved by orchestrator

class DynamicEntityExtractor:
    """Dynamic entity extraction without hardcoded patterns"""
    
    def __init__(self):
        self.discovered_apps: Set[str] = set()
        self.discovered_patterns: Dict[str, Set[str]] = {
            'applications': set(),
            'file_extensions': set(),
            'ui_elements': set(),
            'actions': set()
        }
        self._load_discovered_patterns()
    
    def _load_discovered_patterns(self):
        """Load previously discovered patterns from disk"""
        patterns_file = Path.home() / '.jarvis' / 'vision_patterns.json'
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r') as f:
                    data = json.load(f)
                    for key, values in data.items():
                        if key in self.discovered_patterns:
                            self.discovered_patterns[key] = set(values)
            except Exception as e:
                logger.debug(f"Could not load patterns: {e}")
    
    def _save_discovered_patterns(self):
        """Save discovered patterns to disk"""
        patterns_file = Path.home() / '.jarvis' / 'vision_patterns.json'
        patterns_file.parent.mkdir(exist_ok=True)
        
        try:
            data = {k: list(v) for k, v in self.discovered_patterns.items()}
            with open(patterns_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.debug(f"Could not save patterns: {e}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities dynamically from text"""
        entities = {
            'applications': [],
            'files': [],
            'urls': [],
            'ui_elements': []
        }
        
        # Dynamic application detection
        # Look for patterns like "in <App>" or "<App> window" or "<App> is open"
        app_patterns = [
            r'(?:in|using|running|open|closed?)\s+(\w+(?:\s+\w+)?)\s*(?:app|application|window)?',
            r'(\w+(?:\s+\w+)?)\s+(?:is|was|are|were)\s+(?:open|running|active)',
            r'(?:launch|start|open|close|quit)\s+(\w+(?:\s+\w+)?)'
        ]
        
        for pattern in app_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                app = match.group(1).strip()
                # Filter out common words that aren't apps
                if len(app) > 2 and not app.lower() in ['the', 'and', 'this', 'that', 'with']:
                    entities['applications'].append(app.title())
                    self.discovered_patterns['applications'].add(app.title())
        
        # Dynamic file detection - learn extensions from text
        file_pattern = r'([\w\-]+\.[\w]{2,5})'
        files = re.findall(file_pattern, text)
        for file in files:
            entities['files'].append(file)
            # Learn the extension
            ext = file.split('.')[-1].lower()
            if 2 <= len(ext) <= 5:  # Reasonable extension length
                self.discovered_patterns['file_extensions'].add(ext)
        
        # Dynamic URL detection
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            r'www\.[^\s<>"{}|\\^`\[\]]+',
            r'[a-zA-Z0-9\-]+\.(com|org|net|io|dev|app|edu|gov)[^\s]*'
        ]
        
        for pattern in url_patterns:
            urls = re.findall(pattern, text, re.IGNORECASE)
            entities['urls'].extend(urls[:5])  # Limit URLs
        
        # Dynamic UI element detection
        ui_pattern = r'(\w+(?:\s+\w+)?)\s*(?:button|menu|dialog|window|tab|panel|bar|field|box)'
        ui_matches = re.finditer(ui_pattern, text, re.IGNORECASE)
        for match in ui_matches:
            element = match.group(0).strip()
            if element and len(element) < 50:  # Reasonable length
                entities['ui_elements'].append(element)
                self.discovered_patterns['ui_elements'].add(element.lower())
        
        # Save newly discovered patterns
        self._save_discovered_patterns()
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities

class MemoryAwareCache:
    """LRU cache with memory awareness - fully configurable"""
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.max_size_bytes = config.cache_size_mb * 1024 * 1024
        self.max_entries = config.cache_max_entries
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self._lock = asyncio.Lock()
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get item from cache with LRU update"""
        async with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                entry = self.cache.pop(key)
                entry.access_count += 1
                self.cache[key] = entry
                return entry
            return None
    
    async def put(self, key: str, entry: CacheEntry):
        """Add item to cache with memory management"""
        async with self._lock:
            # Calculate entry size
            entry.size_bytes = len(json.dumps(entry.result).encode())
            
            # Check if we need to evict entries
            while (self.current_size_bytes + entry.size_bytes > self.max_size_bytes or 
                   len(self.cache) >= self.max_entries) and self.cache:
                # Remove least recently used
                oldest_key, oldest_entry = self.cache.popitem(last=False)
                self.current_size_bytes -= oldest_entry.size_bytes
                logger.debug(f"Evicted cache entry: {oldest_key[:8]}...")
            
            # Add new entry
            self.cache[key] = entry
            self.current_size_bytes += entry.size_bytes
            
            # Force garbage collection if memory usage is high
            if self._get_memory_usage_percent() > self.config.memory_threshold_percent:
                gc.collect()
    
    def _get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage"""
        return psutil.Process().memory_percent()
    
    async def clear(self):
        """Clear the cache"""
        async with self._lock:
            self.cache.clear()
            self.current_size_bytes = 0
            gc.collect()


@dataclass
class MemorySafetyStatus:
    """Status of memory safety checks"""
    is_safe: bool
    process_mb: float
    process_limit_mb: float
    system_available_gb: float
    system_min_gb: float
    warnings: List[str] = field(default_factory=list)
    rejected_count: int = 0
    last_check: datetime = field(default_factory=datetime.now)


class MemorySafetyMonitor:
    """Monitor and enforce memory safety to prevent crashes"""
    
    # Memory estimates by image size (MB)
    MEMORY_ESTIMATES = {
        'tiny': 10,      # < 0.5MP
        'small': 20,     # 0.5-1MP  
        'medium': 100,   # 1-2MP
        'large': 300,    # 2MP+
        'concurrent_overhead': 10
    }
    
    def __init__(self, config: VisionConfig):
        self.config = config
        self.rejected_requests = 0
        self.warnings_issued = 0
        self._last_gc_time = time.time()
        self._emergency_mode = False
        
    def check_memory_safety(self) -> MemorySafetyStatus:
        """Check current memory status"""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            process_mb = process.memory_info().rss / (1024**2)
            system_available_gb = memory.available / (1024**3)
            
            warnings = []
            
            # Check process memory
            if process_mb > self.config.memory_warning_threshold_mb:
                warnings.append(f"Process memory high: {process_mb:.0f}MB")
            
            # Check system memory
            if system_available_gb < self.config.min_system_available_gb * 1.5:
                warnings.append(f"System memory low: {system_available_gb:.1f}GB available")
            
            # Determine if safe
            is_safe = (
                process_mb < self.config.process_memory_limit_mb and
                system_available_gb > self.config.min_system_available_gb
            )
            
            # Enter emergency mode if critically low
            if not is_safe and not self._emergency_mode:
                self._emergency_mode = True
                logger.warning("Entering emergency memory mode - aggressive limits enabled")
            elif is_safe and self._emergency_mode and system_available_gb > self.config.min_system_available_gb * 2:
                self._emergency_mode = False
                logger.info("Exiting emergency memory mode")
            
            return MemorySafetyStatus(
                is_safe=is_safe,
                process_mb=process_mb,
                process_limit_mb=self.config.process_memory_limit_mb,
                system_available_gb=system_available_gb,
                system_min_gb=self.config.min_system_available_gb,
                warnings=warnings,
                rejected_count=self.rejected_requests
            )
        except Exception as e:
            logger.error(f"Memory safety check failed: {e}")
            # Fail safe - assume unsafe
            return MemorySafetyStatus(
                is_safe=False,
                process_mb=0,
                process_limit_mb=self.config.process_memory_limit_mb,
                system_available_gb=0,
                system_min_gb=self.config.min_system_available_gb,
                warnings=[f"Memory check error: {e}"]
            )
    
    def estimate_memory_usage(self, width: int, height: int, 
                            concurrent_count: int = 1) -> Dict[str, Any]:
        """Estimate memory required for operation"""
        pixels = width * height
        megapixels = pixels / 1_000_000
        
        # Determine size category
        if megapixels < 0.5:
            category = 'tiny'
        elif megapixels < 1:
            category = 'small'
        elif megapixels < 2:
            category = 'medium'
        else:
            category = 'large'
        
        # Calculate memory
        base_memory = self.MEMORY_ESTIMATES[category]
        concurrent_memory = (concurrent_count - 1) * self.MEMORY_ESTIMATES['concurrent_overhead']
        
        # Add overhead for emergency mode
        if self._emergency_mode:
            base_memory *= 1.5
        
        total_estimate = base_memory + concurrent_memory
        
        return {
            'category': category,
            'megapixels': megapixels,
            'base_memory_mb': base_memory,
            'concurrent_memory_mb': concurrent_memory,
            'total_estimate_mb': total_estimate,
            'emergency_mode': self._emergency_mode
        }
    
    async def ensure_memory_available(self, width: int, height: int) -> bool:
        """Ensure enough memory is available for operation"""
        if not self.config.enable_memory_safety:
            return True
        
        # Check current status
        status = self.check_memory_safety()
        
        # Log warnings
        for warning in status.warnings:
            if self.warnings_issued < 10:  # Limit warning spam
                logger.warning(warning)
                self.warnings_issued += 1
        
        # Reject if already unsafe
        if not status.is_safe:
            self.rejected_requests += 1
            if self.config.reject_on_memory_pressure:
                return False
        
        # Estimate required memory
        estimate = self.estimate_memory_usage(width, height)
        projected_memory = status.process_mb + estimate['total_estimate_mb']
        
        # Check if operation would exceed limits
        if projected_memory > self.config.process_memory_limit_mb:
            self.rejected_requests += 1
            logger.warning(
                f"Rejecting {estimate['category']} image ({estimate['megapixels']:.1f}MP). "
                f"Would exceed memory limit: {projected_memory:.0f}MB > {self.config.process_memory_limit_mb}MB"
            )
            if self.config.reject_on_memory_pressure:
                return False
        
        # Force GC if needed
        if time.time() - self._last_gc_time > 30 and status.process_mb > self.config.memory_warning_threshold_mb:
            gc.collect()
            self._last_gc_time = time.time()
        
        return True
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get current status as dictionary"""
        status = self.check_memory_safety()
        return {
            'is_safe': status.is_safe,
            'process_mb': status.process_mb,
            'process_limit_mb': status.process_limit_mb,
            'system_available_gb': status.system_available_gb,
            'warnings': status.warnings,
            'rejected_requests': self.rejected_requests,
            'emergency_mode': self._emergency_mode
        }


class ClaudeVisionAnalyzer:
    """Enhanced Claude vision analyzer - fully dynamic and configurable"""
    
    def __init__(self, api_key: str, config: Optional[VisionConfig] = None,
                 config_path: Optional[str] = None, enable_realtime: bool = True):
        """Initialize enhanced Claude vision analyzer
        
        Args:
            api_key: Anthropic API key
            config: VisionConfig instance (optional)
            config_path: Path to JSON config file (optional)
            enable_realtime: Enable real-time monitoring capabilities (default: True)
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = VisionConfig.from_file(config_path)
        else:
            self.config = VisionConfig()
        
        # Initialize real-time capabilities
        self.enable_realtime = enable_realtime
        self._realtime_callbacks = []
        
        # Initialize API client
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not provided - vision analysis will not work")
            self.client = None
        else:
            try:
                self.client = Anthropic(api_key=api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic client: {str(e)}")
                self.client = None
        
        # Initialize memory safety monitor
        self.memory_monitor = MemorySafetyMonitor(self.config)
        
        # Initialize components based on config
        self.cache = MemoryAwareCache(self.config) if self.config.cache_enabled else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        self.api_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Metrics storage
        self.recent_metrics: List[AnalysisMetrics] = []
        
        # Initialize context dictionary
        self._context = {
            'app_id': 'unknown',
            'workflow': 'unknown'
        }
        
        # Dynamic entity extractor
        self.entity_extractor = DynamicEntityExtractor()
        
        # Dynamic prompt templates loaded from file or defaults
        self.prompt_templates = self._load_prompt_templates()
        
        # Initialize enhanced memory-optimized components
        self._init_enhanced_components()
        
        logger.info(f"Initialized ClaudeVisionAnalyzer with config: {self.config.to_dict()}")
    
    def _init_enhanced_components(self):
        """Initialize all enhanced memory-optimized components"""
        # Initialize continuous screen analyzer (lazy loading)
        self.continuous_analyzer = None
        self._continuous_analyzer_config = {
            'enabled': os.getenv('VISION_CONTINUOUS_ENABLED', 'true').lower() == 'true',
            'update_interval': float(os.getenv('VISION_MONITOR_INTERVAL', '3.0')),
            'enable_proactive': os.getenv('VISION_PROACTIVE_ENABLED', 'true').lower() == 'true',
            'proactive_confidence': float(os.getenv('VISION_PROACTIVE_CONFIDENCE', '0.75')),
            'enable_voice': os.getenv('VISION_PROACTIVE_VOICE', 'true').lower() == 'true'
        }
        
        # Initialize window analyzer (lazy loading)
        self.window_analyzer = None
        self._window_analyzer_config = {
            'enabled': os.getenv('VISION_WINDOW_ANALYSIS_ENABLED', 'true').lower() == 'true'
        }
        
        # Initialize window relationship detector (lazy loading)
        self.relationship_detector = None
        self._relationship_detector_config = {
            'enabled': os.getenv('VISION_RELATIONSHIP_ENABLED', 'true').lower() == 'true'
        }
        
        # Initialize Swift vision integration (lazy loading)
        self.swift_vision = None
        self._swift_vision_config = {
            'enabled': os.getenv('VISION_SWIFT_ENABLED', 'true').lower() == 'true'
        }
        
        # Initialize memory-efficient analyzer (lazy loading)
        self.memory_efficient_analyzer = None
        self._memory_efficient_config = {
            'enabled': os.getenv('VISION_MEMORY_EFFICIENT_ENABLED', 'true').lower() == 'true'
        }
        
        # Initialize simplified vision system (lazy loading)
        self.simplified_vision = None
        self._simplified_vision_config = {
            'enabled': os.getenv('VISION_SIMPLIFIED_ENABLED', 'true').lower() == 'true'
        }
        
        # Initialize screen sharing module (lazy loading)
        self.screen_sharing = None
        
        # Initialize Vision Intelligence System (lazy loading)
        self.vision_intelligence = None
        self._vision_intelligence_config = {
            'enabled': self.config.vision_intelligence_enabled and VISION_INTELLIGENCE_AVAILABLE,
            'learning': getattr(self.config, 'vision_intelligence_learning', True),
            'consensus': self.config.vision_intelligence_consensus,
            'persistence': self.config.state_persistence_enabled
        }
        
        if self._vision_intelligence_config['enabled']:
            logger.info("Vision Intelligence System available and enabled")
        
        # Initialize VSMS Core (lazy loading)
        self.vsms_core = None
        self._vsms_core_config = {
            'enabled': self.config.enable_vsms_core and VSMS_CORE_AVAILABLE,
            'track_workflows': self.config.vsms_track_workflows,
            'detect_anomalies': self.config.vsms_detect_anomalies,
            'personalization': self.config.vsms_personalization,
            'stuck_threshold_minutes': self.config.vsms_stuck_threshold_minutes
        }
        
        # Initialize Workflow Pattern configuration
        self._workflow_pattern_config = {
            'enabled': os.getenv('WORKFLOW_PATTERN_ENABLED', 'true').lower() == 'true',
            'min_support': float(os.getenv('WORKFLOW_MIN_SUPPORT', '0.2')),
            'automation_enabled': os.getenv('WORKFLOW_AUTOMATION_ENABLED', 'true').lower() == 'true',
            'clustering_method': os.getenv('WORKFLOW_CLUSTERING_METHOD', 'hybrid'),
            'max_pattern_length': int(os.getenv('WORKFLOW_MAX_PATTERN_LENGTH', '20')),
            'use_rust_mining': os.getenv('WORKFLOW_USE_RUST', 'true').lower() == 'true',
            'neural_predictions': os.getenv('WORKFLOW_NEURAL_PREDICTIONS', 'true').lower() == 'true'
        }
        
        # Initialize workflow pattern engine (lazy loading)
        self.workflow_engine = None
        self.enhanced_workflow_engine = None
        
        # Initialize Scene Graph configuration
        self._scene_graph_config = {
            'enabled': self.config.scene_graph_enabled,
            'element_detection': self.config.scene_graph_element_detection,
            'relationship_discovery': self.config.scene_graph_relationship_discovery
        }
        
        # Initialize Temporal Context configuration
        self._temporal_context_config = {
            'enabled': self.config.temporal_context_enabled,
            'pattern_extraction': self.config.temporal_pattern_extraction,
            'predictions': self.config.temporal_prediction_enabled
        }
        
        # Initialize Activity Recognition configuration
        self._activity_recognition_config = {
            'enabled': self.config.activity_recognition_enabled,
            'task_inference': self.config.task_inference_enabled,
            'progress_monitoring': self.config.progress_monitoring_enabled
        }
        
        # Initialize Anomaly Detection Framework configuration
        self._anomaly_detection_config = {
            'enabled': os.getenv('ANOMALY_DETECTION_ENABLED', 'true').lower() == 'true',
            'ml_enabled': os.getenv('ANOMALY_ML_ENABLED', 'true').lower() == 'true',
            'baseline_samples': int(os.getenv('ANOMALY_BASELINE_SAMPLES', '50')),
            'detection_cooldown': int(os.getenv('ANOMALY_COOLDOWN_SECONDS', '30'))
        }
        
        # Initialize anomaly detector (lazy loading)
        self.anomaly_detector = None
        
        # Initialize Intervention Decision Engine configuration
        self._intervention_config = {
            'enabled': os.getenv('INTERVENTION_ENABLED', 'true').lower() == 'true',
            'min_confidence': float(os.getenv('INTERVENTION_MIN_CONFIDENCE', '0.7')),
            'cooldown_minutes': int(os.getenv('INTERVENTION_COOLDOWN_MINUTES', '5')),
            'learning_enabled': os.getenv('INTERVENTION_LEARNING_ENABLED', 'true').lower() == 'true'
        }
        
        # Initialize intervention engine (lazy loading)
        self.intervention_engine = None
        
        # Initialize Solution Memory Bank configuration
        self._solution_memory_config = {
            'enabled': os.getenv('SOLUTION_MEMORY_ENABLED', 'true').lower() == 'true',
            'auto_capture': os.getenv('SOLUTION_AUTO_CAPTURE', 'true').lower() == 'true',
            'min_confidence': float(os.getenv('SOLUTION_MIN_CONFIDENCE', '0.6')),
            'auto_apply_threshold': float(os.getenv('SOLUTION_AUTO_APPLY_THRESHOLD', '0.9')),
            'capture_screenshots': os.getenv('SOLUTION_CAPTURE_SCREENSHOTS', 'true').lower() == 'true'
        }
        
        # Initialize solution memory bank (lazy loading)
        self.solution_memory_bank = None
        
        # Initialize Quadtree Spatial Intelligence configuration
        self._quadtree_spatial_config = {
            'enabled': os.getenv('QUADTREE_SPATIAL_ENABLED', 'true').lower() == 'true',
            'max_depth': int(os.getenv('QUADTREE_MAX_DEPTH', '6')),
            'min_node_size': int(os.getenv('QUADTREE_MIN_NODE_SIZE', '50')),
            'importance_threshold': float(os.getenv('QUADTREE_IMPORTANCE_THRESHOLD', '0.7')),
            'enable_caching': os.getenv('QUADTREE_ENABLE_CACHING', 'true').lower() == 'true',
            'cache_duration_minutes': int(os.getenv('QUADTREE_CACHE_DURATION', '5')),
            'use_rust_acceleration': os.getenv('QUADTREE_USE_RUST', 'true').lower() == 'true',
            'use_swift_detection': os.getenv('QUADTREE_USE_SWIFT', 'true').lower() == 'true',
            'optimize_api_calls': os.getenv('QUADTREE_OPTIMIZE_API', 'true').lower() == 'true',
            'max_regions_per_analysis': int(os.getenv('QUADTREE_MAX_REGIONS', '10'))
        }
        
        # Initialize quadtree spatial intelligence (lazy loading)
        self.quadtree_spatial = None
        
        # Initialize Semantic Cache with LSH configuration
        self._semantic_cache_config = {
            'enabled': os.getenv('SEMANTIC_CACHE_ENABLED', 'true').lower() == 'true',
            'use_lsh': os.getenv('SEMANTIC_CACHE_USE_LSH', 'true').lower() == 'true',
            'l1_enabled': os.getenv('SEMANTIC_CACHE_L1_ENABLED', 'true').lower() == 'true',
            'l2_enabled': os.getenv('SEMANTIC_CACHE_L2_ENABLED', 'true').lower() == 'true',
            'l3_enabled': os.getenv('SEMANTIC_CACHE_L3_ENABLED', 'true').lower() == 'true',
            'l4_enabled': os.getenv('SEMANTIC_CACHE_L4_ENABLED', 'true').lower() == 'true',
            'similarity_threshold': float(os.getenv('SEMANTIC_CACHE_SIMILARITY', '0.85')),
            'use_rust_acceleration': os.getenv('SEMANTIC_CACHE_USE_RUST', 'true').lower() == 'true',
            'use_swift_cache': os.getenv('SEMANTIC_CACHE_USE_SWIFT', 'true').lower() == 'true'
        }
        
        # Initialize semantic cache (lazy loading)
        self.semantic_cache = None
        
        # Initialize Predictive Pre-computation Engine configuration
        self._predictive_engine_config = {
            'enabled': os.getenv('PREDICTIVE_ENGINE_ENABLED', 'true').lower() == 'true',
            'confidence_threshold': float(os.getenv('PREDICTIVE_CONFIDENCE_THRESHOLD', '0.7')),
            'max_predictions': int(os.getenv('PREDICTIVE_MAX_PREDICTIONS', '5')),
            'enable_speculative': os.getenv('PREDICTIVE_ENABLE_SPECULATIVE', 'true').lower() == 'true',
            'cache_ttl_seconds': int(os.getenv('PREDICTIVE_CACHE_TTL', '300')),
            'use_rust_engine': os.getenv('PREDICTIVE_USE_RUST', 'true').lower() == 'true',
            'use_swift_tracker': os.getenv('PREDICTIVE_USE_SWIFT', 'true').lower() == 'true'
        }
        
        # Initialize predictive engine (lazy loading)
        self.predictive_engine = None
        
        # Initialize Bloom Filter Network configuration
        self._bloom_filter_config = {
            'enabled': os.getenv('BLOOM_FILTER_ENABLED', 'true').lower() == 'true',
            'global_size_mb': float(os.getenv('BLOOM_GLOBAL_SIZE_MB', '4.0')),
            'regional_size_mb': float(os.getenv('BLOOM_REGIONAL_SIZE_MB', '1.0')),
            'element_size_mb': float(os.getenv('BLOOM_ELEMENT_SIZE_MB', '2.0')),
            'hierarchical_checking': os.getenv('BLOOM_HIERARCHICAL', 'true').lower() == 'true',
            'use_rust_hashing': os.getenv('BLOOM_USE_RUST', 'true').lower() == 'true',
            'use_swift_tracking': os.getenv('BLOOM_USE_SWIFT', 'true').lower() == 'true'
        }
        
        # Initialize bloom filter (lazy loading)
        self.bloom_filter = None
        
        # Initialize Integration Orchestrator configuration
        self._orchestrator_config = {
            'enabled': os.getenv('INTEGRATION_ORCHESTRATOR_ENABLED', 'true').lower() == 'true',
            'total_memory_mb': int(os.getenv('ORCHESTRATOR_MEMORY_MB', '1200')),
            'intelligence_memory_mb': int(os.getenv('INTELLIGENCE_MEMORY_MB', '600')),
            'optimization_memory_mb': int(os.getenv('OPTIMIZATION_MEMORY_MB', '460')),
            'buffer_memory_mb': int(os.getenv('BUFFER_MEMORY_MB', '140')),
            'use_rust_pipeline': os.getenv('USE_RUST_PIPELINE', 'true').lower() == 'true',
            'use_swift_coordinator': os.getenv('USE_SWIFT_COORDINATOR', 'true').lower() == 'true',
            'adaptive_quality': os.getenv('ORCHESTRATOR_ADAPTIVE_QUALITY', 'true').lower() == 'true',
            'batch_processing': os.getenv('ORCHESTRATOR_BATCH_MODE', 'true').lower() == 'true',
        }
        
        # Initialize orchestrator (lazy loading)
        self.orchestrator = None
        self._last_intelligence_context = None  # Store last intelligence insights
        
        if self._vsms_core_config['enabled']:
            logger.info("VSMS Core available and enabled")
        self._screen_sharing_config = {
            'enabled': self.config.enable_screen_sharing
        }
        
        # Initialize video streaming module (lazy loading)
        self.video_streaming = None
        self._video_streaming_config = {
            'enabled': self.config.enable_video_streaming
        }
        logger.info(f"[VISION ANALYZER INIT] Instance {id(self)} created with video streaming enabled: {self.config.enable_video_streaming}")
        
        # Flag to track if analyzer is busy (for screen sharing priority)
        self.is_analyzing = False
        
        logger.info("Enhanced components configured for lazy initialization")
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates from file or use defaults"""
        templates_file = Path.home() / '.jarvis' / 'vision_prompts.json'
        
        # Default templates
        default_templates = {
            "general": "Analyze this screenshot and provide a concise description. Focus on: {focus}",
            "json": "Analyze this screenshot and respond ONLY with valid JSON in this format: {format}",
            "action": "Identify actionable items in this screenshot. List specific actions the user can take.",
            "error": "Check this screenshot for any errors, warnings, or issues that need attention.",
            "workspace": "Describe the user's current workspace and what they appear to be working on.",
            "quick": "Briefly describe what's on screen in 2-3 sentences.",
            "detailed": "Provide a detailed analysis of everything visible on screen, including all text, UI elements, and their relationships."
        }
        
        # Try to load custom templates
        if templates_file.exists():
            try:
                with open(templates_file, 'r') as f:
                    custom_templates = json.load(f)
                    default_templates.update(custom_templates)
                    logger.info(f"Loaded {len(custom_templates)} custom prompt templates")
            except Exception as e:
                logger.debug(f"Could not load custom templates: {e}")
        
        return default_templates
    
    def update_config(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
    
    async def get_continuous_analyzer(self):
        """Get continuous screen analyzer with lazy loading"""
        if self.continuous_analyzer is None and self._continuous_analyzer_config['enabled']:
            try:
                from .continuous_screen_analyzer import MemoryAwareScreenAnalyzer
                self.continuous_analyzer = MemoryAwareScreenAnalyzer(
                    vision_handler=self,
                    update_interval=self._continuous_analyzer_config['update_interval']
                )
                logger.info("Initialized continuous screen analyzer for proactive monitoring")
            except ImportError as e:
                logger.warning(f"Could not import continuous screen analyzer: {e}")
        return self.continuous_analyzer
    
    def get_proactive_config(self) -> Dict[str, Any]:
        """Get proactive monitoring configuration"""
        return {
            'proactive_enabled': self._continuous_analyzer_config.get('enable_proactive', True),
            'confidence_threshold': self._continuous_analyzer_config.get('proactive_confidence', 0.75),
            'voice_enabled': self._continuous_analyzer_config.get('enable_voice', True),
            'continuous_enabled': self._continuous_analyzer_config.get('enabled', True)
        }
    
    async def get_window_analyzer(self):
        """Get window analyzer with lazy loading"""
        if self.window_analyzer is None and self._window_analyzer_config['enabled']:
            try:
                from .window_analysis import MemoryAwareWindowAnalyzer
                self.window_analyzer = MemoryAwareWindowAnalyzer()
                logger.info("Initialized window analyzer")
            except ImportError as e:
                logger.warning(f"Could not import window analyzer: {e}")
        return self.window_analyzer
    
    async def get_relationship_detector(self):
        """Get window relationship detector with lazy loading"""
        if self.relationship_detector is None and self._relationship_detector_config['enabled']:
            try:
                from .window_relationship_detector import ConfigurableWindowRelationshipDetector
                self.relationship_detector = ConfigurableWindowRelationshipDetector()
                logger.info("Initialized window relationship detector")
            except ImportError as e:
                logger.warning(f"Could not import relationship detector: {e}")
        return self.relationship_detector
    
    async def get_swift_vision(self):
        """Get Swift vision integration with lazy loading"""
        if self.swift_vision is None and self._swift_vision_config['enabled']:
            try:
                from .swift_vision_integration import get_swift_vision_integration
                self.swift_vision = get_swift_vision_integration()
                logger.info("Initialized Swift vision integration")
            except ImportError as e:
                logger.warning(f"Could not import Swift vision: {e}")
        return self.swift_vision
    
    async def get_memory_efficient_analyzer(self):
        """Get memory-efficient analyzer with lazy loading"""
        if self.memory_efficient_analyzer is None and self._memory_efficient_config['enabled']:
            try:
                from .memory_efficient_vision_analyzer import MemoryEfficientVisionAnalyzer
                # Use same API key
                api_key = os.getenv('ANTHROPIC_API_KEY', 'dummy')
                self.memory_efficient_analyzer = MemoryEfficientVisionAnalyzer(api_key)
                logger.info("Initialized memory-efficient analyzer")
            except ImportError as e:
                logger.warning(f"Could not import memory-efficient analyzer: {e}")
        return self.memory_efficient_analyzer
    
    async def get_vision_intelligence(self):
        """Get Vision Intelligence System with lazy loading"""
        if self.vision_intelligence is None and self._vision_intelligence_config['enabled']:
            try:
                self.vision_intelligence = get_vision_intelligence_bridge()
                
                # Configure learning mode
                if hasattr(self.vision_intelligence, 'vsms') and self.vision_intelligence.vsms:
                    self.vision_intelligence.vsms.enable_learning(self._vision_intelligence_config['learning'])
                
                logger.info("Initialized Vision Intelligence System")
            except Exception as e:
                logger.warning(f"Could not initialize Vision Intelligence: {e}")
                self._vision_intelligence_config['enabled'] = False
        
        return self.vision_intelligence
    
    async def get_vsms_core(self):
        """Get VSMS Core with lazy loading"""
        if self.vsms_core is None and self._vsms_core_config['enabled']:
            try:
                self.vsms_core = get_vsms()
                
                # Configure VSMS settings
                if self.vsms_core:
                    # Set stuck threshold
                    if hasattr(self.vsms_core, 'state_intelligence'):
                        from datetime import timedelta
                        self.vsms_core.state_intelligence.stuck_threshold = timedelta(
                            minutes=self._vsms_core_config['stuck_threshold_minutes']
                        )
                
                logger.info("Initialized VSMS Core")
            except Exception as e:
                logger.warning(f"Could not initialize VSMS Core: {e}")
                self._vsms_core_config['enabled'] = False
        
        return self.vsms_core
    
    async def get_workflow_engine(self):
        """Get Workflow Pattern Engine with lazy loading"""
        if self.workflow_engine is None and self._workflow_pattern_config['enabled']:
            try:
                from .intelligence.workflow_pattern_engine import (
                    get_workflow_pattern_engine, WorkflowEvent
                )
                self.workflow_engine = get_workflow_pattern_engine()
                
                # Use enhanced engine if neural predictions enabled
                if self._workflow_pattern_config['neural_predictions']:
                    try:
                        from .intelligence.enhanced_workflow_engine import get_enhanced_workflow_engine
                        self.enhanced_workflow_engine = get_enhanced_workflow_engine()
                        logger.info("Initialized enhanced workflow engine with neural predictions")
                    except Exception as e:
                        logger.warning(f"Could not initialize enhanced workflow engine: {e}")
                
                logger.info("Initialized Workflow Pattern Engine")
            except Exception as e:
                logger.warning(f"Could not initialize Workflow Pattern Engine: {e}")
                self._workflow_pattern_config['enabled'] = False
        
        # Return enhanced engine if available, otherwise basic engine
        return self.enhanced_workflow_engine or self.workflow_engine
    
    async def get_workflow_patterns(self, pattern_type: Optional[str] = None):
        """Get discovered workflow patterns"""
        engine = await self.get_workflow_engine()
        if not engine:
            return []
        
        if pattern_type:
            return engine.get_patterns_by_type(pattern_type)
        return list(engine.patterns.values())
    
    async def predict_workflow(self, current_sequence: List[str], top_k: int = 5):
        """Predict next actions in workflow"""
        engine = await self.get_workflow_engine()
        if not engine:
            return []
        
        return await engine.predict_next_actions(current_sequence, top_k)
    
    async def get_automation_suggestions(self, context: Dict[str, Any] = None):
        """Get workflow automation suggestions"""
        engine = await self.get_workflow_engine()
        if not engine or not hasattr(engine, 'suggest_automation'):
            return []
        
        # Build context from current state
        if context is None:
            context = {
                'time_of_day': datetime.now().strftime('%H:%M'),
                'recent_actions': []
            }
            
            # Add VSMS context if available
            if self.vsms_core:
                try:
                    current_state = self.vsms_core.get_current_state()
                    if current_state:
                        context['active_app'] = current_state.active_application
                        context['goal'] = current_state.inferred_goal
                except:
                    pass
        
        return engine.suggest_automation(context)
    
    async def get_anomaly_detector(self):
        """Get Anomaly Detection Framework with lazy loading"""
        if self.anomaly_detector is None and self._anomaly_detection_config['enabled']:
            try:
                from .intelligence.anomaly_detection_framework import (
                    get_anomaly_detection_framework, AnomalyType
                )
                self.anomaly_detector = get_anomaly_detection_framework()
                logger.info("Initialized Anomaly Detection Framework")
            except Exception as e:
                logger.warning(f"Could not initialize Anomaly Detection Framework: {e}")
                self._anomaly_detection_config['enabled'] = False
        return self.anomaly_detector
    
    async def get_intervention_engine(self):
        """Get Intervention Decision Engine with lazy loading"""
        if self.intervention_engine is None and self._intervention_config['enabled']:
            try:
                from .intelligence.intervention_decision_engine import (
                    get_intervention_decision_engine, UserStateSignal, InterventionType
                )
                self.intervention_engine = get_intervention_decision_engine()
                logger.info("Initialized Intervention Decision Engine")
            except Exception as e:
                logger.warning(f"Could not initialize Intervention Decision Engine: {e}")
                self._intervention_config['enabled'] = False
        return self.intervention_engine
    
    async def get_solution_memory_bank(self):
        """Get Solution Memory Bank with lazy loading"""
        if self.solution_memory_bank is None and self._solution_memory_config['enabled']:
            try:
                from .intelligence.solution_memory_bank import (
                    get_solution_memory_bank, ProblemSignature, ProblemType
                )
                self.solution_memory_bank = get_solution_memory_bank()
                logger.info("Initialized Solution Memory Bank")
            except Exception as e:
                logger.warning(f"Could not initialize Solution Memory Bank: {e}")
                self._solution_memory_config['enabled'] = False
        return self.solution_memory_bank
    
    async def get_quadtree_spatial(self):
        """Get Quadtree Spatial Intelligence with lazy loading"""
        if self.quadtree_spatial is None and self._quadtree_spatial_config['enabled']:
            try:
                from .intelligence.quadtree_spatial_intelligence import (
                    get_quadtree_spatial_intelligence, RegionImportance, QueryResult
                )
                self.quadtree_spatial = get_quadtree_spatial_intelligence()
                logger.info("Initialized Quadtree Spatial Intelligence")
            except Exception as e:
                logger.warning(f"Could not initialize Quadtree Spatial Intelligence: {e}")
                self._quadtree_spatial_config['enabled'] = False
        return self.quadtree_spatial
    
    async def get_semantic_cache(self):
        """Get Semantic Cache with LSH with lazy loading"""
        if self.semantic_cache is None and self._semantic_cache_config['enabled']:
            try:
                from .intelligence.semantic_cache_lsh import (
                    get_semantic_cache, SemanticCacheWithLSH, CacheLevel
                )
                self.semantic_cache = await get_semantic_cache()
                
                # Set integration points
                if hasattr(self, 'goal_system') and self.goal_system:
                    self.semantic_cache.set_integration_points(
                        goal_system=self.goal_system,
                        anomaly_detector=await self.get_anomaly_detector()
                    )
                
                logger.info("Initialized Semantic Cache with LSH")
            except Exception as e:
                logger.warning(f"Could not initialize Semantic Cache: {e}")
                self._semantic_cache_config['enabled'] = False
        return self.semantic_cache
    
    async def get_predictive_engine(self):
        """Get Predictive Pre-computation Engine with lazy loading"""
        if self.predictive_engine is None and self._predictive_engine_config['enabled']:
            try:
                from .intelligence.predictive_precomputation_engine import (
                    get_predictive_engine, PredictivePrecomputationEngine, StateVector
                )
                self.predictive_engine = await get_predictive_engine()
                
                # Set integration points
                self.predictive_engine.set_integration_points(
                    temporal_engine=await self.get_temporal_context_engine() if hasattr(self, 'get_temporal_context_engine') else None,
                    workflow_engine=await self.get_workflow_pattern_engine() if hasattr(self, 'get_workflow_pattern_engine') else None,
                    semantic_cache=await self.get_semantic_cache()
                )
                
                logger.info("Initialized Predictive Pre-computation Engine")
            except Exception as e:
                logger.warning(f"Could not initialize Predictive Engine: {e}")
                self._predictive_engine_config['enabled'] = False
        return self.predictive_engine
    
    async def get_bloom_filter(self):
        """Get Bloom Filter Network with lazy loading"""
        if self.bloom_filter is None and self._bloom_filter_config['enabled']:
            try:
                from .bloom_filter_network import (
                    get_bloom_filter_network, BloomFilterLevel, VisionBloomFilterIntegration
                )
                bloom_network = get_bloom_filter_network()
                self.bloom_filter = VisionBloomFilterIntegration(bloom_network)
                logger.info("Initialized Bloom Filter Network")
            except Exception as e:
                logger.warning(f"Could not initialize Bloom Filter Network: {e}")
                self._bloom_filter_config['enabled'] = False
        return self.bloom_filter
    
    async def get_simplified_vision(self):
        """Get enhanced simplified vision system with lazy loading and dynamic configuration
        
        The simplified vision system provides a streamlined interface for basic vision tasks
        while maintaining integration with all advanced components when available.
        
        Returns:
            SimplifiedVisionSystem: Configured vision system instance or None if disabled
        """
        if self.simplified_vision is None and self._simplified_vision_config['enabled']:
            try:
                # Import with detailed error tracking
                try:
                    from .vision_system_claude_only import SimplifiedVisionSystem
                except ImportError as e:
                    logger.error(f"Failed to import SimplifiedVisionSystem: {e}")
                    # Try alternative import paths
                    try:
                        from vision_system_claude_only import SimplifiedVisionSystem
                    except ImportError:
                        logger.warning("Could not import simplified vision from any path")
                        self._simplified_vision_config['enabled'] = False
                        return None
                
                # Prepare dynamic configuration
                simplified_config = {
                    # Core settings from environment
                    'enable_caching': os.getenv('SIMPLIFIED_VISION_CACHE', 'true').lower() == 'true',
                    'enable_compression': os.getenv('SIMPLIFIED_VISION_COMPRESS', 'true').lower() == 'true',
                    'enable_metrics': os.getenv('SIMPLIFIED_VISION_METRICS', 'true').lower() == 'true',
                    'max_retries': int(os.getenv('SIMPLIFIED_VISION_RETRIES', '3')),
                    'timeout_seconds': int(os.getenv('SIMPLIFIED_VISION_TIMEOUT', '30')),
                    
                    # Quality settings
                    'default_quality': os.getenv('SIMPLIFIED_VISION_QUALITY', 'balanced'),
                    'auto_quality_adjust': os.getenv('SIMPLIFIED_VISION_AUTO_QUALITY', 'true').lower() == 'true',
                    'min_quality_threshold': float(os.getenv('SIMPLIFIED_VISION_MIN_QUALITY', '0.6')),
                    
                    # Integration settings
                    'use_advanced_components': os.getenv('SIMPLIFIED_USE_ADVANCED', 'true').lower() == 'true',
                    'fallback_on_error': os.getenv('SIMPLIFIED_FALLBACK', 'true').lower() == 'true',
                    'share_cache': os.getenv('SIMPLIFIED_SHARE_CACHE', 'true').lower() == 'true',
                    
                    # Performance settings
                    'batch_size': int(os.getenv('SIMPLIFIED_BATCH_SIZE', '5')),
                    'parallel_requests': int(os.getenv('SIMPLIFIED_PARALLEL', '2')),
                    'memory_limit_mb': int(os.getenv('SIMPLIFIED_MEMORY_LIMIT', '500')),
                }
                
                # Initialize with enhanced configuration
                self.simplified_vision = SimplifiedVisionSystem(
                    claude_analyzer=self,
                    config=simplified_config
                )
                
                # Set up integration points if advanced components are available
                if simplified_config['use_advanced_components']:
                    integration_count = 0
                    
                    # Integrate with semantic cache if available
                    if simplified_config['share_cache'] and hasattr(self, 'semantic_cache') and self.semantic_cache:
                        try:
                            self.simplified_vision.set_cache(self.semantic_cache)
                            integration_count += 1
                            logger.debug("Simplified vision integrated with semantic cache")
                        except Exception as e:
                            logger.debug(f"Could not integrate semantic cache: {e}")
                    
                    # Integrate with bloom filter if available
                    if hasattr(self, 'bloom_filter') and self.bloom_filter:
                        try:
                            self.simplified_vision.set_bloom_filter(self.bloom_filter)
                            integration_count += 1
                            logger.debug("Simplified vision integrated with bloom filter")
                        except Exception as e:
                            logger.debug(f"Could not integrate bloom filter: {e}")
                    
                    # Integrate with predictive engine if available
                    if hasattr(self, 'predictive_engine') and self.predictive_engine:
                        try:
                            self.simplified_vision.set_predictive_engine(self.predictive_engine)
                            integration_count += 1
                            logger.debug("Simplified vision integrated with predictive engine")
                        except Exception as e:
                            logger.debug(f"Could not integrate predictive engine: {e}")
                    
                    # Integrate with VSMS if available
                    if hasattr(self, 'vsms_core') and self.vsms_core:
                        try:
                            self.simplified_vision.set_vsms_core(self.vsms_core)
                            integration_count += 1
                            logger.debug("Simplified vision integrated with VSMS core")
                        except Exception as e:
                            logger.debug(f"Could not integrate VSMS core: {e}")
                    
                    logger.info(f"Simplified vision initialized with {integration_count} integrations")
                
                # Configure callbacks for coordination
                if hasattr(self.simplified_vision, 'register_callback'):
                    # Register memory warning callback
                    self.simplified_vision.register_callback(
                        'memory_warning',
                        self._handle_simplified_memory_warning
                    )
                    
                    # Register quality change callback
                    self.simplified_vision.register_callback(
                        'quality_changed',
                        self._handle_simplified_quality_change
                    )
                    
                    # Register error callback
                    self.simplified_vision.register_callback(
                        'error',
                        self._handle_simplified_error
                    )
                
                # Perform initialization checks
                if hasattr(self.simplified_vision, 'health_check'):
                    health_status = await self.simplified_vision.health_check()
                    if not health_status.get('healthy', False):
                        logger.warning(f"Simplified vision health check failed: {health_status}")
                        if not simplified_config['fallback_on_error']:
                            self.simplified_vision = None
                            self._simplified_vision_config['enabled'] = False
                            return None
                
                # Set up performance monitoring
                if simplified_config['enable_metrics'] and hasattr(self.simplified_vision, 'enable_metrics'):
                    self.simplified_vision.enable_metrics()
                
                # Configure auto-quality adjustment
                if simplified_config['auto_quality_adjust'] and hasattr(self.simplified_vision, 'configure_auto_quality'):
                    self.simplified_vision.configure_auto_quality({
                        'min_threshold': simplified_config['min_quality_threshold'],
                        'adjustment_interval': 60,  # seconds
                        'sample_size': 10
                    })
                
                # Log successful initialization
                logger.info(
                    f"Initialized simplified vision system: "
                    f"cache={simplified_config['enable_caching']}, "
                    f"compression={simplified_config['enable_compression']}, "
                    f"quality={simplified_config['default_quality']}, "
                    f"integrations={'enabled' if integration_count > 0 else 'disabled'}"
                )
                
                # Store configuration for runtime adjustments
                self._simplified_vision_config.update(simplified_config)
                
            except Exception as e:
                logger.error(f"Failed to initialize simplified vision system: {e}", exc_info=True)
                self._simplified_vision_config['enabled'] = False
                
                # Clean up partial initialization
                if hasattr(self, 'simplified_vision') and self.simplified_vision:
                    try:
                        if hasattr(self.simplified_vision, 'cleanup'):
                            await self.simplified_vision.cleanup()
                    except:
                        pass
                    self.simplified_vision = None
                
                return None
        
        return self.simplified_vision
    
    async def _handle_simplified_memory_warning(self, data: Dict[str, Any]):
        """Handle memory warnings from simplified vision system"""
        logger.warning(f"Simplified vision memory warning: {data}")
        
        # Reduce quality if auto-adjust is enabled
        if self._simplified_vision_config.get('auto_quality_adjust', True):
            if hasattr(self.simplified_vision, 'reduce_quality'):
                await self.simplified_vision.reduce_quality()
        
        # Trigger garbage collection
        import gc
        gc.collect()
    
    async def _handle_simplified_quality_change(self, data: Dict[str, Any]):
        """Handle quality changes in simplified vision"""
        logger.info(f"Simplified vision quality changed: {data}")
        
        # Update configuration
        if 'new_quality' in data:
            self._simplified_vision_config['current_quality'] = data['new_quality']
    
    async def _handle_simplified_error(self, data: Dict[str, Any]):
        """Handle errors from simplified vision system"""
        logger.error(f"Simplified vision error: {data}")
        
        # Increment error counter
        if 'error_count' not in self._simplified_vision_config:
            self._simplified_vision_config['error_count'] = 0
        self._simplified_vision_config['error_count'] += 1
        
        # Disable if too many errors
        max_errors = int(os.getenv('SIMPLIFIED_MAX_ERRORS', '10'))
        if self._simplified_vision_config['error_count'] >= max_errors:
            logger.error(f"Simplified vision disabled due to {max_errors} errors")
            self._simplified_vision_config['enabled'] = False
    
    async def get_orchestrator(self):
        """Get Integration Orchestrator with lazy loading and full component integration
        
        The orchestrator manages the complete vision processing pipeline with:
        - Dynamic memory allocation (1.2GB total budget)
        - Intelligent component coordination
        - Adaptive resource management
        - Cross-language optimization (Python/Rust/Swift)
        
        Returns:
            IntegrationOrchestrator: Configured orchestrator instance or None if disabled
        """
        if self.orchestrator is None and self._orchestrator_config['enabled']:
            try:
                # Import orchestrator
                from .intelligence.integration_orchestrator import (
                    get_integration_orchestrator, IntegrationOrchestrator, SystemMode
                )
                
                # Create orchestrator with custom config
                orchestrator_config = {
                    'total_memory_mb': self._orchestrator_config['total_memory_mb'],
                    'intelligence_memory_mb': self._orchestrator_config['intelligence_memory_mb'],
                    'optimization_memory_mb': self._orchestrator_config['optimization_memory_mb'],
                    'buffer_memory_mb': self._orchestrator_config['buffer_memory_mb'],
                    'enable_all_components': True,
                    'adaptive_quality': self._orchestrator_config['adaptive_quality'],
                    'aggressive_caching': True,
                }
                
                self.orchestrator = get_integration_orchestrator(orchestrator_config)
                logger.info(f"Initialized Integration Orchestrator with {orchestrator_config['total_memory_mb']}MB budget")
                
                # Pass component references to orchestrator
                await self._integrate_orchestrator_components()
                
                # Setup Rust pipeline if enabled
                if self._orchestrator_config['use_rust_pipeline']:
                    try:
                        from .jarvis_rust_core.vision import IntegrationPipeline
                        rust_pipeline = IntegrationPipeline(self._orchestrator_config['total_memory_mb'])
                        self.orchestrator._rust_pipeline = rust_pipeline
                        logger.info("Integrated Rust pipeline for high-performance processing")
                    except Exception as e:
                        logger.warning(f"Could not integrate Rust pipeline: {e}")
                
                # Setup Swift coordinator if on macOS
                if self._orchestrator_config['use_swift_coordinator'] and sys.platform == 'darwin':
                    try:
                        import subprocess
                        # Compile Swift coordinator if needed
                        swift_path = os.path.join(os.path.dirname(__file__), 'integration_coordinator_macos.swift')
                        if os.path.exists(swift_path):
                            # The Swift code would be integrated via PyObjC or as a separate process
                            logger.info("Swift coordinator available for macOS optimization")
                    except Exception as e:
                        logger.warning(f"Could not setup Swift coordinator: {e}")
                
                # Configure memory monitoring callback
                if hasattr(self.orchestrator, 'set_memory_callback'):
                    self.orchestrator.set_memory_callback(self._handle_orchestrator_memory_event)
                
                # Get initial status
                status = await self.orchestrator.get_system_status()
                logger.info(f"Orchestrator status: mode={status['system_mode']}, components={len(status['components'])}")
                
            except Exception as e:
                logger.error(f"Failed to initialize Integration Orchestrator: {e}", exc_info=True)
                self._orchestrator_config['enabled'] = False
                self.orchestrator = None
                
        return self.orchestrator
    
    async def _integrate_orchestrator_components(self):
        """Integrate existing components with the orchestrator"""
        if not self.orchestrator:
            return
        
        component_count = 0
        
        # Pass existing component references
        components_map = {
            'vsms': ('vsms_core', getattr(self, 'vsms_core', None)),
            'scene_graph': ('scene_graph', None),  # Would get from intelligence bridge
            'temporal_context': ('temporal_context_engine', None),
            'activity_recognition': ('activity_recognizer', None),
            'goal_inference': ('goal_system', getattr(self, 'goal_system', None)),
            'workflow_patterns': ('pattern_miner', getattr(self, 'pattern_miner', None)),
            'anomaly_detection': ('anomaly_detector', None),
            'intervention_engine': ('intervention_engine', getattr(self, 'intervention_engine', None)),
            'solution_bank': ('solution_memory_bank', getattr(self, 'solution_memory_bank', None)),
            'quadtree': ('quadtree_spatial', getattr(self, 'quadtree_spatial', None)),
            'semantic_cache': ('semantic_cache', getattr(self, 'semantic_cache', None)),
            'predictive_engine': ('predictive_engine', getattr(self, 'predictive_engine', None)),
            'bloom_filter': ('bloom_filter', getattr(self, 'bloom_filter', None)),
        }
        
        for orchestrator_name, (attr_name, component) in components_map.items():
            if component is not None:
                self.orchestrator.components[orchestrator_name] = component
                component_count += 1
            elif hasattr(self, attr_name) and getattr(self, attr_name) is not None:
                self.orchestrator.components[orchestrator_name] = getattr(self, attr_name)
                component_count += 1
        
        logger.info(f"Integrated {component_count} components with orchestrator")
    
    async def _handle_orchestrator_memory_event(self, event: Dict[str, Any]):
        """Handle memory events from orchestrator"""
        mode = event.get('mode', 'unknown')
        logger.warning(f"Orchestrator memory event: mode={mode}, pressure={event.get('pressure', 0):.2%}")
        
        # Notify other components of memory pressure
        if mode in ['critical', 'emergency']:
            # Reduce quality across the board
            if hasattr(self.config, 'jpeg_quality'):
                self.config.jpeg_quality = max(60, self.config.jpeg_quality - 20)
            if hasattr(self.config, 'max_image_dimension'):
                self.config.max_image_dimension = min(1024, self.config.max_image_dimension)
    
    async def get_screen_sharing(self):
        """Get screen sharing manager with lazy loading"""
        if self.screen_sharing is None and self._screen_sharing_config['enabled']:
            try:
                from .screen_sharing_module import ScreenSharingManager
                self.screen_sharing = ScreenSharingManager(vision_analyzer=self)
                logger.info("Initialized screen sharing manager")
                
                # Register callbacks to coordinate with continuous analyzer
                if self.config.enable_continuous_monitoring:
                    continuous = await self.get_continuous_analyzer()
                    if continuous:
                        # Share memory warnings
                        continuous.register_callback('memory_warning', 
                                                   self._handle_memory_warning_for_sharing)
                        
                # Register screen sharing callbacks
                self.screen_sharing.register_callback('memory_warning',
                                                    self._handle_sharing_memory_warning)
                self.screen_sharing.register_callback('quality_changed',
                                                    self._handle_sharing_quality_change)
                
            except ImportError as e:
                logger.warning(f"Could not import screen sharing: {e}")
        return self.screen_sharing
    
    async def _handle_memory_warning_for_sharing(self, data: Dict[str, Any]):
        """Handle memory warning from continuous analyzer for screen sharing"""
        if self.screen_sharing and self.screen_sharing.is_sharing:
            # Reduce screen sharing quality when memory is low
            logger.warning(f"Memory warning received: {data}")
            # Screen sharing will automatically adjust based on its own monitoring
    
    async def _handle_sharing_memory_warning(self, data: Dict[str, Any]):
        """Handle memory warning from screen sharing"""
        logger.warning(f"Screen sharing memory warning: {data}")
        # Could trigger cleanup in other components if needed
    
    async def _handle_sharing_quality_change(self, data: Dict[str, Any]):
        """Log screen sharing quality changes"""
        logger.info(f"Screen sharing quality adjusted: {data}")
    
    async def get_video_streaming(self):
        """Get video streaming manager with lazy loading"""
        logger.info(f"[VISION ANALYZER] get_video_streaming called, current: {self.video_streaming is not None}")
        logger.info(f"[VISION ANALYZER] Video streaming config: {self._video_streaming_config}")
        
        # Always try to initialize video streaming if not already done
        if self.video_streaming is None:
            try:
                logger.info("[VISION ANALYZER] Importing VideoStreamCapture...")
                from .video_stream_capture import VideoStreamCapture
                
                logger.info("[VISION ANALYZER] Creating VideoStreamCapture instance...")
                self.video_streaming = VideoStreamCapture(vision_analyzer=self)
                logger.info("Initialized video streaming capture")
                
                # Register callbacks for analysis
                self.video_streaming.register_callback('frame_analyzed', 
                                                     self._handle_video_frame_analyzed)
                self.video_streaming.register_callback('motion_detected',
                                                     self._handle_video_motion_detected)
                self.video_streaming.register_callback('memory_warning',
                                                     self._handle_video_memory_warning)
                
            except ImportError as e:
                logger.warning(f"Could not import video streaming: {e}")
        return self.video_streaming
    
    async def _handle_video_frame_analyzed(self, data: Dict[str, Any]):
        """Handle analyzed video frame"""
        logger.debug(f"Video frame {data['frame_number']} analyzed")
        # Could trigger additional actions based on analysis
    
    async def _handle_video_motion_detected(self, data: Dict[str, Any]):
        """Handle motion detection in video"""
        logger.info(f"Motion detected: score={data['motion_score']:.2f}")
        # Could trigger more detailed analysis on motion
    
    async def _handle_video_memory_warning(self, data: Dict[str, Any]):
        """Handle memory warning from video streaming"""
        logger.warning(f"Video streaming memory warning: {data}")
        # Could trigger quality reduction or cleanup
    
    async def analyze_screenshot(self, image: Any, prompt: str, 
                               use_cache: Optional[bool] = None,
                               priority: str = "normal",
                               custom_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], AnalysisMetrics]:
        """Enhanced screenshot analysis with full configurability
        
        Args:
            image: Screenshot as PIL Image or numpy array
            prompt: What to analyze in the image
            use_cache: Override cache setting for this request
            priority: "high" for urgent requests, "normal" for standard
            custom_config: Override config values for this request
            
        Returns:
            Tuple of (analysis results, performance metrics)
        """
        metrics = AnalysisMetrics()
        start_time = time.time()
        
        # Apply custom config for this request if provided
        if custom_config:
            original_config = {}
            for key, value in custom_config.items():
                if hasattr(self.config, key):
                    original_config[key] = getattr(self.config, key)
                    setattr(self.config, key, value)
        
        # Set analyzing flag for screen sharing coordination
        self.is_analyzing = True
        
        try:
            # Memory safety check
            if self.config.enable_memory_safety:
                # Get image dimensions
                if isinstance(image, np.ndarray):
                    height, width = image.shape[:2]
                elif isinstance(image, Image.Image):
                    width, height = image.size
                else:
                    # Default conservative estimate
                    width, height = 1920, 1080
                
                # Check if safe to proceed
                memory_safe = await self.memory_monitor.ensure_memory_available(width, height)
                if not memory_safe:
                    metrics.total_time = time.time() - start_time
                    error_msg = (
                        f"Analysis rejected due to memory constraints. "
                        f"Process: {self.memory_monitor.check_memory_safety().process_mb:.0f}MB, "
                        f"Available: {self.memory_monitor.check_memory_safety().system_available_gb:.1f}GB"
                    )
                    logger.error(error_msg)
                    raise MemoryError(error_msg)
            
            # Determine if we should use cache
            should_use_cache = use_cache if use_cache is not None else self.config.cache_enabled
            
            # Try Swift vision acceleration first if available
            swift_vision = await self.get_swift_vision()
            if swift_vision and swift_vision.enabled:
                try:
                    # Use Swift for preprocessing if memory allows
                    swift_result = await swift_vision.process_screenshot(image if isinstance(image, Image.Image) else Image.fromarray(image))
                    if swift_result.method == "swift":
                        logger.debug(f"Used Swift vision acceleration (processing time: {swift_result.processing_time:.2f}s)")
                        # Update metrics from Swift
                        metrics.preprocessing_time = swift_result.processing_time
                        metrics.image_size_compressed = swift_result.compressed_size
                except Exception as e:
                    logger.debug(f"Swift vision fallback: {e}")
            
            # Convert and preprocess image
            preprocessing_start = time.time()
            pil_image, image_hash = await self._preprocess_image(image)
            if not hasattr(metrics, 'preprocessing_time') or metrics.preprocessing_time == 0:
                metrics.preprocessing_time = time.time() - preprocessing_start
            metrics.image_size_original = self._estimate_image_size(pil_image)
            
            # Integration Orchestrator Processing (Part 3: Integration Architecture)
            orchestrator_result = None
            if self._orchestrator_config.get('enabled', False):
                try:
                    orchestrator = await self.get_orchestrator()
                    if orchestrator:
                        # Convert PIL image to numpy array for orchestrator
                        np_image = np.array(pil_image)
                        
                        # Build context for orchestrator
                        orchestrator_context = {
                            'image_hash': image_hash,
                            'prompt': prompt,
                            'priority': priority,
                            'user_action': custom_config.get('user_action') if custom_config else 'analyze',
                            'workflow_phase': custom_config.get('workflow_phase') if custom_config else None,
                            'timestamp': time.time(),
                            'app_id': self._context.get('app_id', 'unknown'),
                            'quality_level': self.quality_monitor.current_level if hasattr(self, 'quality_monitor') else 'high'
                        }
                        
                        # Process through orchestrator's 9-stage pipeline
                        orchestrator_start = time.time()
                        orchestrator_result = await orchestrator.process_frame(
                            frame=np_image,
                            context=orchestrator_context
                        )
                        
                        # Extract metrics from orchestrator
                        if '_metrics' in orchestrator_result:
                            metrics.orchestrator_time = time.time() - orchestrator_start
                            metrics.system_mode = orchestrator_result['_metrics'].get('system_mode', 'normal')
                            metrics.orchestrator_cache_hits = orchestrator_result['_metrics'].get('cache_hits', 0)
                            metrics.orchestrator_api_saved = orchestrator_result['_metrics'].get('api_calls_saved', 0)
                            
                            # Log orchestrator performance
                            logger.info(f"Orchestrator processed in {metrics.orchestrator_time:.2f}s "
                                      f"(mode: {metrics.system_mode}, cache hits: {metrics.orchestrator_cache_hits})")
                        
                        # Check if orchestrator found a cached/predicted result
                        if orchestrator_result.get('cached', False) or orchestrator_result.get('predicted', False):
                            # Orchestrator found a result, skip remaining processing
                            metrics.cache_hit = True
                            metrics.total_time = time.time() - start_time
                            
                            # Extract the actual result from orchestrator
                            result = {
                                'analysis': orchestrator_result.get('analysis', ''),
                                'elements': orchestrator_result.get('elements', []),
                                'suggestions': orchestrator_result.get('suggestions', []),
                                'spatial_analysis': orchestrator_result.get('spatial_analysis'),
                                'intelligence': orchestrator_result.get('intelligence'),
                                '_orchestrator_optimized': True
                            }
                            
                            logger.info(f"Orchestrator optimization: Returned cached/predicted result, "
                                      f"saved API call (total time: {metrics.total_time:.2f}s)")
                            return result, metrics
                        
                        # Extract intelligence insights for enhanced processing
                        if 'intelligence' in orchestrator_result:
                            self._last_intelligence_context = orchestrator_result['intelligence']
                
                except Exception as e:
                    logger.warning(f"Integration orchestrator error: {e}")
                    # Continue with normal processing if orchestrator fails
            
            # Update predictive engine state if enabled
            predictive_result = None
            if self._predictive_engine_config['enabled']:
                try:
                    predictive_engine = await self.get_predictive_engine()
                    if predictive_engine:
                        # Create current state vector
                        from .intelligence.predictive_precomputation_engine import StateVector
                        
                        current_state = StateVector(
                            app_id=self._context.get('app_id', 'unknown'),
                            app_state='analyzing_screenshot',
                            user_action=custom_config.get('user_action') if custom_config else 'analyze',
                            time_context=self._get_time_context(),
                            goal_context=prompt[:50] if len(prompt) > 50 else prompt,
                            workflow_phase=custom_config.get('workflow_phase') if custom_config else None,
                            metadata={
                                'image_hash': image_hash,
                                'prompt': prompt,
                                'timestamp': datetime.now().isoformat()
                            }
                        )
                        
                        # Update state for learning
                        await predictive_engine.update_state(current_state)
                        
                        # Check for pre-computed results
                        if should_use_cache and self._predictive_engine_config['enable_speculative']:
                            predictions = predictive_engine.transition_matrix.get_predictions(
                                current_state, 
                                top_k=self._predictive_engine_config['max_predictions']
                            )
                            
                            for next_state, probability, confidence in predictions:
                                if confidence >= self._predictive_engine_config['confidence_threshold']:
                                    # Check if we have pre-computed result
                                    cached_prediction = await predictive_engine.get_prediction(
                                        current_state, next_state
                                    )
                                    if cached_prediction:
                                        predictive_result = cached_prediction
                                        logger.info(f"Using pre-computed prediction (confidence: {confidence:.2f})")
                                        break
                        
                except Exception as e:
                    logger.warning(f"Predictive engine error: {e}")
            
            # Check bloom filter for duplicate requests
            bloom_filter = None
            if self._bloom_filter_config['enabled']:
                try:
                    bloom_filter = await self.get_bloom_filter()
                    if bloom_filter:
                        # Create request signature
                        request_context = {
                            'prompt_hash': prompt_hash,
                            'image_hash': image_hash,
                            'timestamp': time.time(),
                            'image_size': f"{pil_image.size[0]}x{pil_image.size[1]}"
                        }
                        
                        # Check if this exact request was recently processed
                        from .bloom_filter_network import BloomFilterLevel
                        if bloom_filter.is_image_duplicate(image_hash, request_context):
                            logger.info("Bloom filter: Duplicate screenshot detected")
                            # Still check cache for the actual result
                        else:
                            logger.debug("Bloom filter: New screenshot")
                except Exception as e:
                    logger.debug(f"Bloom filter check failed: {e}")
            
            # Check semantic cache if enabled (replaces basic cache)
            semantic_cache_result = None
            if should_use_cache and self._semantic_cache_config['enabled'] and not predictive_result:
                try:
                    semantic_cache = await self.get_semantic_cache()
                    if semantic_cache:
                        # Generate context for cache
                        cache_context = {
                            'image_hash': image_hash,
                            'image_size': f"{pil_image.size[0]}x{pil_image.size[1]}",
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        # Add VSMS context if available
                        if self.vsms_core:
                            try:
                                current_state = self.vsms_core.get_current_state()
                                if current_state:
                                    cache_context['app'] = current_state.active_application
                                    cache_context['goal'] = current_state.inferred_goal
                            except:
                                pass
                        
                        # Generate embedding for semantic search
                        embedding = None
                        if self._semantic_cache_config['l2_enabled']:
                            # Use prompt as basis for embedding
                            # In production, this would use a proper embedding model
                            embedding = await self._generate_prompt_embedding(prompt)
                        
                        # Check if should bypass cache
                        bypass_cache = semantic_cache.should_bypass_cache(cache_context)
                        
                        if not bypass_cache:
                            # Query semantic cache
                            cache_result = await semantic_cache.get(
                                key=prompt,
                                context=cache_context,
                                embedding=embedding
                            )
                            
                            if cache_result:
                                value, cache_level, similarity = cache_result
                                metrics.cache_hit = True
                                metrics.total_time = time.time() - start_time
                                logger.debug(f"Semantic cache hit: {cache_level.value} "
                                           f"(similarity: {similarity:.2f})")
                                
                                # Record cache hit in pattern predictor
                                semantic_cache.pattern_predictor.record_access(prompt)
                                
                                return value, metrics
                
                except Exception as e:
                    logger.warning(f"Semantic cache check failed: {e}")
            
            # Fallback to basic cache if semantic cache not available
            elif should_use_cache and self.cache:
                cache_key = self._generate_cache_key(image_hash, prompt)
                cached_entry = await self.cache.get(cache_key)
                if cached_entry and self._is_cache_valid(cached_entry):
                    metrics.cache_hit = True
                    metrics.total_time = time.time() - start_time
                    logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                    return cached_entry.result, metrics
            
            # Apply Quadtree Spatial Intelligence if enabled
            quadtree_regions = None
            if self._quadtree_spatial_config['enabled'] and self._quadtree_spatial_config['optimize_api_calls']:
                quadtree_start = time.time()
                try:
                    quadtree = await self.get_quadtree_spatial()
                    if quadtree:
                        # Convert PIL image to numpy for quadtree analysis
                        np_image = np.array(pil_image)
                        
                        # Build quadtree for image
                        tree_id = f"img_{image_hash[:8]}"
                        await quadtree.build_quadtree(np_image, tree_id)
                        
                        # Query important regions based on configuration
                        query_result = await quadtree.query_regions(
                            tree_id,
                            importance_threshold=self._quadtree_spatial_config['importance_threshold'],
                            max_regions=self._quadtree_spatial_config['max_regions_per_analysis']
                        )
                        
                        if query_result.nodes:
                            quadtree_regions = []
                            for node in query_result.nodes:
                                quadtree_regions.append({
                                    'x': node.x,
                                    'y': node.y,
                                    'width': node.width,
                                    'height': node.height,
                                    'importance': node.importance,
                                    'complexity': node.complexity
                                })
                            
                            # Get processing recommendations
                            recommendations = quadtree.get_processing_recommendations(
                                tree_id,
                                available_api_calls=self._quadtree_spatial_config['max_regions_per_analysis']
                            )
                            
                            logger.info(f"Quadtree identified {len(quadtree_regions)} important regions "
                                      f"(coverage: {query_result.coverage_ratio:.1%})")
                            
                            if recommendations:
                                logger.debug(f"Quadtree recommendations: {recommendations[0]['reason']}")
                        
                        # Add quadtree time to metrics
                        metrics.quadtree_time = time.time() - quadtree_start
                            
                except Exception as e:
                    logger.warning(f"Quadtree spatial analysis failed: {e}")
            
            # Check if we have a pre-computed result from predictive engine
            if predictive_result:
                metrics.cache_hit = True
                metrics.predictive_hit = True
                metrics.total_time = time.time() - start_time
                logger.info("Using pre-computed predictive result")
                return predictive_result, metrics
            
            # Intelligent region-based processing with Quadtree optimization
            if quadtree_regions and len(quadtree_regions) > 0:
                # Determine strategy based on regions and system mode
                num_regions = len(quadtree_regions)
                system_mode = metrics.system_mode if hasattr(metrics, 'system_mode') else 'normal'
                
                # Dynamic thresholds based on system mode
                region_limits = {
                    'normal': 5,      # Process up to 5 regions in normal mode
                    'pressure': 3,    # Reduce to 3 in pressure mode
                    'critical': 2,    # Only 2 most important in critical mode
                    'emergency': 1    # Single most important region in emergency
                }
                max_regions = region_limits.get(system_mode, 5)
                
                # Calculate total coverage of important regions
                total_area = pil_image.width * pil_image.height
                important_area = sum(r['width'] * r['height'] for r in quadtree_regions[:max_regions])
                coverage_ratio = important_area / total_area
                
                # Decide strategy based on coverage and importance
                if coverage_ratio > 0.7 or num_regions == 1:
                    # High coverage or single region - use full image but compressed more
                    logger.info(f"Using full image strategy (coverage: {coverage_ratio:.1%})")
                    compression_quality = self.config.jpeg_quality - 10  # More compression
                    image_to_send = pil_image
                    strategy = "full_compressed"
                else:
                    # Low coverage - create composite of important regions
                    logger.info(f"Using region composite strategy ({num_regions} regions, "
                              f"coverage: {coverage_ratio:.1%})")
                    
                    # Sort regions by importance and take top N
                    sorted_regions = sorted(quadtree_regions, 
                                         key=lambda r: r['importance'] * r['width'] * r['height'], 
                                         reverse=True)[:max_regions]
                    
                    # Create composite image with regions
                    composite_image = await self._create_region_composite(
                        pil_image, sorted_regions, system_mode
                    )
                    image_to_send = composite_image
                    compression_quality = self.config.jpeg_quality
                    strategy = "region_composite"
                    
                    # Update metrics with region extraction info
                    metrics.regions_extracted = len(sorted_regions)
                    metrics.coverage_ratio = coverage_ratio
            else:
                # No quadtree regions - use standard full image
                logger.debug("No quadtree regions available, using full image")
                image_to_send = pil_image
                compression_quality = self.config.jpeg_quality
                strategy = "full_standard"
            
            # Compress the selected image/composite
            if self.config.compression_enabled:
                compression_start = time.time()
                # Use dynamic quality based on strategy
                original_quality = self.config.jpeg_quality
                self.config.jpeg_quality = compression_quality
                try:
                    image_base64, compressed_size = await self._compress_and_encode(image_to_send)
                finally:
                    self.config.jpeg_quality = original_quality  # Restore
                
                metrics.image_size_compressed = compressed_size
                metrics.compression_ratio = 1 - (compressed_size / metrics.image_size_original)
                metrics.processing_strategy = strategy
                logger.debug(f"Compressed {strategy} image from {metrics.image_size_original} to "
                           f"{compressed_size} bytes ({metrics.compression_ratio:.1%} reduction)")
            else:
                image_base64 = self._encode_image(image_to_send)
                metrics.image_size_compressed = len(base64.b64decode(image_base64))
            
            # Make API call with rate limiting
            api_start = time.time()
            async with self.api_semaphore:
                # Create enhanced prompt based on strategy
                enhanced_prompt = prompt
                if quadtree_regions:
                    if strategy == "region_composite":
                        region_info = f"\n\n[Spatial Optimization: Showing {metrics.regions_extracted} most important regions "
                        region_info += f"covering {coverage_ratio:.1%} of screen. "
                        region_info += "Each region is labeled with its location and importance. "
                        region_info += "Please focus analysis on these specific areas.]"
                    elif strategy == "full_compressed":
                        region_info = f"\n\n[Note: {len(quadtree_regions)} important regions detected. "
                        region_info += "Key areas: "
                        for i, region in enumerate(quadtree_regions[:3]):  # Top 3 for context
                            region_info += f"({region['x']},{region['y']}) "
                        region_info += "Please prioritize these areas in your analysis.]"
                    else:
                        region_info = ""
                    enhanced_prompt = prompt + region_info
                
                if priority == "high":
                    # High priority requests get processed immediately
                    result = await self._call_claude_api(image_base64, enhanced_prompt)
                else:
                    # Normal priority may be delayed if system is busy
                    if self._get_system_load() > (self.config.cpu_threshold_percent / 100):
                        await asyncio.sleep(0.5)  # Brief delay to reduce load
                    result = await self._call_claude_api(image_base64, enhanced_prompt)
            metrics.api_call_time = time.time() - api_start
            
            # Parse response
            parsing_start = time.time()
            parsed_result = self._parse_claude_response(result)
            metrics.parsing_time = time.time() - parsing_start
            
            # Add quadtree spatial information to result if available
            if quadtree_regions:
                parsed_result['spatial_analysis'] = {
                    'regions_detected': len(quadtree_regions),
                    'important_regions': quadtree_regions,
                    'optimization_applied': True,
                    'coverage_ratio': query_result.coverage_ratio if 'query_result' in locals() else 0.0
                }
                logger.debug(f"Added {len(quadtree_regions)} spatial regions to result")
            
            # Enhance with Vision Intelligence if enabled
            if self._vision_intelligence_config['enabled']:
                vi_start = time.time()
                try:
                    vision_intelligence = await self.get_vision_intelligence()
                    if vision_intelligence:
                        # Extract application context from prompt or result
                        app_id = self._extract_app_context(prompt, parsed_result)
                        
                        # Analyze visual state
                        vi_result = await vision_intelligence.analyze_visual_state(
                            screenshot=image if isinstance(image, np.ndarray) else np.array(pil_image),
                            app_id=app_id,
                            metadata={
                                'prompt': prompt,
                                'claude_result': parsed_result
                            }
                        )
                        
                        # Merge Vision Intelligence insights
                        if vi_result and 'final_state' in vi_result:
                            parsed_result['_vision_intelligence'] = {
                                'state': vi_result['final_state'],
                                'app_id': app_id,
                                'components_used': vi_result.get('components_used', []),
                                'confidence': vi_result['final_state'].get('confidence', 0.0)
                            }
                            
                            # Add state insights to entities if available
                            if 'entities' not in parsed_result:
                                parsed_result['entities'] = {}
                            
                            parsed_result['entities']['application_state'] = {
                                'state_id': vi_result['final_state'].get('state_id', 'unknown'),
                                'consensus': vi_result['final_state'].get('consensus', False),
                                'sources': vi_result['final_state'].get('sources', [])
                            }
                        
                        metrics.vision_intelligence_time = time.time() - vi_start
                        logger.debug(f"Vision Intelligence analysis completed in {metrics.vision_intelligence_time:.2f}s")
                        
                        # Save learned states periodically
                        if hasattr(vision_intelligence, 'save_learned_states'):
                            # Save every 100 analyses (configurable)
                            if hasattr(self, '_vi_analysis_count'):
                                self._vi_analysis_count += 1
                            else:
                                self._vi_analysis_count = 1
                            
                            if self._vi_analysis_count % 100 == 0:
                                vision_intelligence.save_learned_states()
                                logger.info("Saved Vision Intelligence learned states")
                                
                except Exception as e:
                    logger.warning(f"Vision Intelligence enhancement failed: {e}")
                    metrics.vision_intelligence_time = time.time() - vi_start
            
            # Enhance with VSMS Core if enabled
            if self._vsms_core_config['enabled']:
                vsms_start = time.time()
                try:
                    vsms_core = await self.get_vsms_core()
                    if vsms_core:
                        # Use same app_id if available
                        if 'app_id' not in locals():
                            app_id = self._extract_app_context(prompt, parsed_result)
                        
                        # Process through VSMS Core
                        vsms_result = await vsms_core.process_visual_observation(
                            screenshot=image if isinstance(image, np.ndarray) else np.array(pil_image),
                            app_id=app_id
                        )
                        
                        # Add VSMS Core insights
                        if vsms_result:
                            parsed_result['vsms_core'] = {
                                'detected_state': vsms_result.get('detected_state'),
                                'confidence': vsms_result.get('confidence', 0.0),
                                'app_identity': vsms_result.get('app_identity'),
                                'scene_graph': vsms_result.get('scene_graph'),
                                'scene_context': vsms_result.get('scene_context'),
                                'content': vsms_result.get('content'),
                                'predictions': vsms_result.get('predictions', []),
                                'recommendations': vsms_result.get('recommendations', {})
                            }
                            
                            # Add activity recognition data if enabled and available
                            if self._activity_recognition_config['enabled'] and 'activity' in vsms_result:
                                parsed_result['vsms_core']['activity'] = vsms_result['activity']
                            
                            # Add temporal context if enabled
                            if self._temporal_context_config['enabled']:
                                temporal_context = await self.get_temporal_context(app_id)
                                if temporal_context.get('enabled'):
                                    parsed_result['temporal_context'] = temporal_context
                            
                            # Add workflow insights if detected
                            if vsms_result.get('recommendations', {}).get('workflow_hint'):
                                workflow = vsms_result['recommendations']['workflow_hint']
                                if 'entities' not in parsed_result:
                                    parsed_result['entities'] = {}
                                parsed_result['entities']['workflow'] = {
                                    'detected': workflow['detected_workflow'],
                                    'next_step': workflow.get('next_in_workflow'),
                                    'steps_remaining': workflow.get('completion_steps', 0)
                                }
                            
                            # Add warnings if any
                            if vsms_result.get('recommendations', {}).get('warnings'):
                                if 'warnings' not in parsed_result:
                                    parsed_result['warnings'] = []
                                for warning in vsms_result['recommendations']['warnings']:
                                    parsed_result['warnings'].append({
                                        'type': warning['type'],
                                        'message': warning['message'],
                                        'source': 'vsms_core'
                                    })
                        
                        metrics.vsms_core_time = time.time() - vsms_start
                        logger.debug(f"VSMS Core analysis completed in {metrics.vsms_core_time:.2f}s")
                        
                except Exception as e:
                    logger.warning(f"VSMS Core enhancement failed: {e}")
                    metrics.vsms_core_time = time.time() - vsms_start
            
            # Record workflow event if pattern engine enabled
            if self._workflow_pattern_config['enabled']:
                try:
                    workflow_engine = await self.get_workflow_engine()
                    if workflow_engine:
                        # Extract action from analysis
                        action = None
                        if 'actions' in parsed_result and parsed_result['actions']:
                            action = parsed_result['actions'][0]  # Use first detected action
                        elif 'entities' in parsed_result and 'action' in parsed_result['entities']:
                            action = parsed_result['entities']['action']
                        elif 'vsms_core' in parsed_result and 'detected_state' in parsed_result['vsms_core']:
                            action = parsed_result['vsms_core']['detected_state']
                        
                        # Record event if action detected
                        if action:
                            from .intelligence.workflow_pattern_engine import WorkflowEvent
                            
                            event = WorkflowEvent(
                                timestamp=datetime.now(),
                                event_type='visual_observation',
                                source_system='claude_vision_analyzer',
                                event_data={
                                    'action': action,
                                    'app_id': app_id if 'app_id' in locals() else 'unknown',
                                    'prompt': prompt[:100],  # First 100 chars
                                    'confidence': parsed_result.get('confidence', 0.5)
                                }
                            )
                            
                            await workflow_engine.record_event(event)
                            
                            # Mine patterns periodically
                            if hasattr(self, '_workflow_event_count'):
                                self._workflow_event_count += 1
                            else:
                                self._workflow_event_count = 1
                            
                            # Mine patterns every 50 events
                            if self._workflow_event_count % 50 == 0:
                                await workflow_engine.mine_patterns(
                                    min_support=self._workflow_pattern_config['min_support']
                                )
                                logger.info("Mined workflow patterns after 50 events")
                        
                        # Add workflow predictions to result if available
                        if action and hasattr(workflow_engine, 'predict_next_actions'):
                            predictions = await workflow_engine.predict_next_actions([action], top_k=3)
                            if predictions:
                                parsed_result['workflow_predictions'] = [
                                    {'action': act, 'confidence': conf} 
                                    for act, conf in predictions
                                ]
                
                except Exception as e:
                    logger.warning(f"Workflow pattern recording failed: {e}")
            
            # Detect anomalies if enabled
            if self._anomaly_detection_config['enabled']:
                try:
                    anomaly_detector = await self.get_anomaly_detector()
                    if anomaly_detector:
                        from .intelligence.anomaly_detection_framework import Observation, AnomalyType
                        
                        # Extract app_id dynamically
                        current_app_id = self._context.get('app_id', 'unknown')
                        if 'app_id' in locals():
                            current_app_id = app_id
                        elif 'entities' in parsed_result and 'application' in parsed_result['entities']:
                            current_app_id = parsed_result['entities']['application'].get('name', 'unknown')
                        
                        # Build observation data dynamically
                        observation_data = {
                            'prompt': prompt[:200] if len(prompt) > 200 else prompt,
                            'entities': parsed_result.get('entities', {}),
                            'actions': parsed_result.get('actions', []),
                            'app_id': current_app_id,
                            'analysis_text': parsed_result.get('analysis', ''),
                            'elements_detected': len(parsed_result.get('elements', [])),
                            'suggestions_count': len(parsed_result.get('suggestions', []))
                        }
                        
                        # Add any additional analysis results
                        if 'spatial_analysis' in parsed_result:
                            observation_data['spatial_data'] = parsed_result['spatial_analysis']
                        if 'temporal_context' in parsed_result:
                            observation_data['temporal_data'] = parsed_result['temporal_context']
                        if 'workflow' in parsed_result:
                            observation_data['workflow_state'] = parsed_result['workflow']
                        
                        # Determine confidence dynamically
                        confidence = parsed_result.get('confidence', 0.5)
                        if '_metrics' in parsed_result:
                            # Use API confidence if available
                            confidence = max(confidence, parsed_result['_metrics'].get('api_confidence', 0.5))
                        
                        # Check for errors/warnings dynamically
                        has_error = False
                        has_warning = False
                        
                        # Check in various fields for error indicators
                        error_indicators = ['error', 'failed', 'failure', 'exception', 'crash']
                        warning_indicators = ['warning', 'warn', 'caution', 'alert']
                        
                        check_fields = [
                            str(parsed_result.get('analysis', '')),
                            str(parsed_result.get('description', '')),
                            ' '.join(str(s) for s in parsed_result.get('suggestions', [])),
                            ' '.join(parsed_result.get('warnings', []))
                        ]
                        
                        combined_text = ' '.join(check_fields).lower()
                        has_error = any(indicator in combined_text for indicator in error_indicators)
                        has_warning = any(indicator in combined_text for indicator in warning_indicators)
                        
                        observation = Observation(
                            timestamp=datetime.now(),
                            observation_type='screenshot_analysis',
                            data=observation_data,
                            source='claude_vision_analyzer',
                            metadata={
                                'has_error': has_error,
                                'has_warning': has_warning,
                                'confidence': confidence,
                                'image_size': metrics.image_size_original if hasattr(metrics, 'image_size_original') else 0,
                                'processing_time': metrics.total_time if hasattr(metrics, 'total_time') else 0,
                                'cache_hit': metrics.cache_hit if hasattr(metrics, 'cache_hit') else False,
                                'priority': priority,
                                'custom_config_applied': bool(custom_config)
                            }
                        )
                        
                        # Detect anomalies
                        anomaly = await anomaly_detector.detect_anomaly(observation)
                        
                        if anomaly:
                            # Add anomaly to result
                            parsed_result['anomaly_detected'] = {
                                'type': anomaly.anomaly_type.value,
                                'severity': anomaly.severity.value,
                                'confidence': anomaly.confidence,
                                'description': anomaly.description,
                                'requires_intervention': anomaly.severity.value in ['HIGH', 'CRITICAL']
                            }
                            
                            # Process intervention if enabled and needed
                            if (self._intervention_config['enabled'] and 
                                anomaly.severity.value in ['HIGH', 'CRITICAL']):
                                
                                intervention_engine = await self.get_intervention_engine()
                                if intervention_engine:
                                    # Create user state signal from anomaly
                                    from .intelligence.intervention_decision_engine import UserStateSignal
                                    
                                    signal = UserStateSignal(
                                        signal_type='error_rate' if anomaly.anomaly_type == AnomalyType.VISUAL else 'anomaly_detected',
                                        value=0.7 if anomaly.severity.value == 'HIGH' else 0.9,
                                        confidence=anomaly.confidence,
                                        timestamp=datetime.now(),
                                        source='anomaly_detector',
                                        metadata={
                                            'anomaly_type': anomaly.anomaly_type.value,
                                            'anomaly_id': anomaly.anomaly_id
                                        }
                                    )
                                    
                                    await intervention_engine.process_user_signal(signal)
                                    
                                    # Assess situation
                                    situation_data = {
                                        'has_error': True,
                                        'error_type': 'anomaly',
                                        'anomaly_type': anomaly.anomaly_type.value,
                                        'severity': anomaly.severity.value,
                                        'context_type': 'visual_analysis',
                                        'app_context': app_id if 'app_id' in locals() else 'unknown'
                                    }
                                    
                                    await intervention_engine.assess_situation(situation_data)
                                    
                                    # Decide on intervention
                                    opportunity = await intervention_engine.decide_intervention()
                                    if opportunity:
                                        parsed_result['intervention_suggested'] = {
                                            'type': opportunity.intervention_type.value,
                                            'timing': opportunity.timing_strategy.value,
                                            'confidence': opportunity.confidence_score,
                                            'urgency': opportunity.urgency_score,
                                            'rationale': opportunity.rationale
                                        }
                                        
                                        logger.info(f"Intervention suggested: {opportunity.intervention_type.value} "
                                                  f"with timing {opportunity.timing_strategy.value}")
                
                except Exception as e:
                    logger.warning(f"Anomaly detection/intervention failed: {e}")
            
            # Solution Memory Bank integration
            if self._solution_memory_config['enabled']:
                try:
                    solution_memory = await self.get_solution_memory_bank()
                    if solution_memory:
                        # Check if this looks like a problem
                        has_problem = (
                            'error' in parsed_result or 
                            'warning' in parsed_result or
                            'anomaly_detected' in parsed_result or
                            (parsed_result.get('entities', {}).get('errors', []))
                        )
                        
                        if has_problem:
                            # Create problem signature from analysis
                            from .intelligence.solution_memory_bank import ProblemSignature, ProblemType
                            
                            # Determine problem type
                            problem_type = ProblemType.UNKNOWN
                            if 'error' in parsed_result:
                                problem_type = ProblemType.ERROR
                            elif parsed_result.get('performance_issue'):
                                problem_type = ProblemType.PERFORMANCE
                            elif parsed_result.get('configuration_issue'):
                                problem_type = ProblemType.CONFIGURATION
                            
                            # Extract error messages and symptoms
                            error_messages = []
                            symptoms = []
                            
                            if isinstance(parsed_result.get('error'), str):
                                error_messages.append(parsed_result['error'])
                            if isinstance(parsed_result.get('description'), str):
                                # Extract error-like text from description
                                desc = parsed_result['description']
                                if 'error' in desc.lower() or 'fail' in desc.lower():
                                    symptoms.append(desc[:200])  # First 200 chars
                            
                            # Add entities as symptoms
                            if 'entities' in parsed_result:
                                for entity_type, values in parsed_result['entities'].items():
                                    if 'error' in entity_type.lower() or 'warning' in entity_type.lower():
                                        symptoms.extend(values[:3])  # First 3
                            
                            # Create problem signature
                            problem = ProblemSignature(
                                visual_pattern={
                                    'app': app_id if 'app_id' in locals() else 'unknown',
                                    'screenshot_hash': image_hash[:8],
                                    'ui_state': parsed_result.get('ui_state', 'unknown')
                                },
                                error_messages=error_messages,
                                context_state={
                                    'prompt': prompt[:100],
                                    'timestamp': datetime.now().isoformat(),
                                    'app_id': app_id if 'app_id' in locals() else 'unknown'
                                },
                                symptoms=symptoms,
                                problem_type=problem_type
                            )
                            
                            # Find similar solutions
                            similar_solutions = await solution_memory.find_similar_solutions(
                                problem, 
                                threshold=self._solution_memory_config['min_confidence']
                            )
                            
                            if similar_solutions:
                                # Get recommendations
                                context = {
                                    'app': app_id if 'app_id' in locals() else 'unknown',
                                    'os': 'macOS',  # Could be detected dynamically
                                    'urgency': 'normal'
                                }
                                
                                recommendations = await solution_memory.get_solution_recommendations(
                                    problem, context
                                )
                                
                                if recommendations:
                                    # Add to result
                                    parsed_result['solution_recommendations'] = []
                                    for rec in recommendations[:3]:  # Top 3
                                        parsed_result['solution_recommendations'].append({
                                            'solution_id': rec['solution_id'],
                                            'score': rec['score'],
                                            'success_rate': rec['success_rate'],
                                            'auto_applicable': rec['auto_applicable'],
                                            'estimated_time': rec['estimated_time']
                                        })
                                    
                                    # Auto-apply if confidence is high enough
                                    best_rec = recommendations[0]
                                    if (best_rec['auto_applicable'] and 
                                        self._solution_memory_config['auto_capture']):
                                        
                                        logger.info(f"Found high-confidence solution: {best_rec['solution_id']}")
                                        parsed_result['solution_available'] = True
                                        parsed_result['solution_auto_applicable'] = True
                        
                        # Capture solution if enabled and problem was resolved
                        elif (self._solution_memory_config['auto_capture'] and 
                              'solution_steps' in parsed_result):
                            # This means Claude detected and provided a solution
                            # We should capture it for future use
                            
                            # Extract problem that was solved
                            if hasattr(self, '_last_problem_signature'):
                                solution_steps = parsed_result.get('solution_steps', [])
                                if solution_steps:
                                    execution_time = metrics.total_time if metrics else 30.0
                                    
                                    captured_solution = await solution_memory.capture_solution(
                                        problem=self._last_problem_signature,
                                        solution_steps=solution_steps,
                                        success=True,  # Assume success if Claude provided solution
                                        execution_time=execution_time,
                                        context={
                                            'app': app_id if 'app_id' in locals() else 'unknown',
                                            'source': 'claude_vision'
                                        }
                                    )
                                    
                                    logger.info(f"Captured solution: {captured_solution.solution_details.solution_id}")
                                    parsed_result['solution_captured'] = True
                
                except Exception as e:
                    logger.warning(f"Solution Memory Bank integration failed: {e}")
            
            # Cache result if enabled
            if should_use_cache:
                # Generate cache key for both cache types
                cache_key = self._generate_cache_key(image_hash, prompt)
                
                if self._semantic_cache_config['enabled'] and self.semantic_cache:
                    # Store in semantic cache with embedding and context
                    prompt_embedding = await self._generate_prompt_embedding(prompt)
                    
                    cache_context = {
                        'prompt': prompt,
                        'image_hash': image_hash,
                        'timestamp': datetime.now().isoformat(),
                        'app_id': self._context.get('app_id', 'unknown'),
                        'workflow': self._context.get('workflow', 'unknown')
                    }
                    
                    # Import CacheLevel if not already imported
                    from vision.intelligence.semantic_cache_lsh import CacheLevel
                    
                    await self.semantic_cache.put(
                        key=cache_key,
                        value=parsed_result,
                        context=cache_context,
                        embedding=prompt_embedding,
                        cache_levels=[CacheLevel.L1_EXACT, CacheLevel.L2_SEMANTIC]
                    )
                    
                    logger.info(f"Cached result in semantic cache for prompt: {prompt[:50]}...")
                elif self.cache:
                    # Fallback to basic cache
                    cache_entry = CacheEntry(
                        result=parsed_result,
                        timestamp=datetime.now(),
                        prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                        image_hash=image_hash
                    )
                    await self.cache.put(cache_key, cache_entry)
            
            # Register the analysis in bloom filter if enabled
            if self._bloom_filter_config['enabled'] and bloom_filter:
                try:
                    # Register at appropriate level based on context
                    bloom_context = {
                        'prompt_hash': prompt_hash,
                        'image_hash': image_hash,
                        'timestamp': time.time(),
                        'app_id': self._context.get('app_id', 'unknown')
                    }
                    
                    # Register as global since it's a complete analysis
                    bloom_filter.network.check_and_add(
                        element_data={'request_id': f"{prompt_hash}_{image_hash}"},
                        context=bloom_context,
                        check_level=BloomFilterLevel.GLOBAL
                    )
                    logger.debug("Registered analysis in bloom filter")
                except Exception as e:
                    logger.debug(f"Failed to register in bloom filter: {e}")
            
            # Add bloom filter stats to result if enabled
            if self._bloom_filter_config['enabled'] and bloom_filter:
                try:
                    bloom_stats = bloom_filter.network.get_network_stats()
                    parsed_result['bloom_filter_stats'] = {
                        'total_checks': bloom_stats['network_metrics']['total_checks'],
                        'hit_rate': bloom_stats['efficiency_stats']['hit_rate'],
                        'memory_usage_mb': bloom_stats['total_memory_mb'],
                        'hierarchical_efficiency': bloom_stats['efficiency_stats']['hierarchical_efficiency']
                    }
                except Exception as e:
                    logger.debug(f"Could not get bloom filter stats: {e}")
            
            # Track metrics
            metrics.total_time = time.time() - start_time
            if self.config.enable_metrics:
                self._track_metrics(metrics)
            
            return parsed_result, metrics
            
        except Exception as e:
            logger.error(f"Error in analyze_screenshot: {type(e).__name__}: {e}", exc_info=True)
            metrics.total_time = time.time() - start_time
            
            # Log additional context for debugging
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"API client exists: {hasattr(self, 'client') and self.client is not None}")
            logger.error(f"Config model: {self.config.model_name if hasattr(self, 'config') else 'No config'}")
            logger.error(f"API key present in env: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
            
            # Provide more specific error messages and ensure consistent response
            error_message = str(e)
            error_type = type(e).__name__
            
            # Determine user-friendly error description
            if "ANTHROPIC_API_KEY" in error_message or "api_key" in error_message.lower():
                description = "I need the Claude Vision API key to analyze your screen. Please set ANTHROPIC_API_KEY in your environment."
            elif "rate_limit" in error_message.lower():
                description = "I'm being rate limited. Please try again in a moment."
            elif "timeout" in error_message.lower():
                description = "The analysis timed out. Please try again."
            elif "network" in error_message.lower() or "connection" in error_message.lower():
                description = "I'm having trouble connecting to the vision API. Please check your internet connection."
            elif isinstance(e, MemoryError):
                description = "Not enough memory available to analyze the image. Please close some applications and try again."
            elif "permission" in error_message.lower():
                description = "I don't have permission to capture your screen. Please grant screen recording permissions in System Preferences."
            else:
                description = f"I encountered an error analyzing your screen: {error_type}. Please try again."
            
            # Always return a consistent structure
            return {
                "error": error_message,
                "error_type": error_type,
                "description": description,
                "success": False,
                "timestamp": datetime.now().isoformat()
            }, metrics
        finally:
            # Clear analyzing flag
            self.is_analyzing = False
            
            # Restore original config if it was modified
            if custom_config and 'original_config' in locals():
                for key, value in original_config.items():
                    setattr(self.config, key, value)
    
    async def _preprocess_image(self, image: Any) -> Tuple[Image.Image, str]:
        """Preprocess image for optimal performance"""
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            if image.dtype == object:
                raise ValueError("Invalid numpy array dtype. Expected uint8 array.")
            pil_image = await asyncio.get_event_loop().run_in_executor(
                self.executor, Image.fromarray, image.astype(np.uint8)
            )
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize if too large (based on config)
        if max(pil_image.size) > self.config.max_image_dimension:
            pil_image = await asyncio.get_event_loop().run_in_executor(
                self.executor, self._resize_image, pil_image
            )
        
        # Generate hash for caching
        image_bytes = io.BytesIO()
        pil_image.save(image_bytes, format="PNG")
        image_hash = hashlib.md5(image_bytes.getvalue()).hexdigest()
        
        return pil_image, image_hash
    
    def _resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image maintaining aspect ratio"""
        image.thumbnail(
            (self.config.max_image_dimension, self.config.max_image_dimension), 
            Image.Resampling.LANCZOS
        )
        return image
    
    async def _compress_and_encode(self, image: Image.Image) -> Tuple[str, int]:
        """Compress and encode image for API transmission"""
        # Try Swift compression first if available
        swift_vision = await self.get_swift_vision()
        if swift_vision and swift_vision.enabled:
            try:
                compressed_bytes = swift_vision.compress_image(image)
                encoded = base64.b64encode(compressed_bytes).decode()
                return encoded, len(compressed_bytes)
            except Exception as e:
                logger.debug(f"Swift compression fallback: {e}")
        
        # Fallback to standard compression
        buffer = io.BytesIO()
        
        # Convert RGBA to RGB if necessary (saves space)
        if image.mode == 'RGBA':
            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
            rgb_image.paste(image, mask=image.split()[3])
            image = rgb_image
        
        # Save as JPEG with configured quality
        # Use lambda to properly pass keyword arguments
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: image.save(buffer, "JPEG", quality=self.config.jpeg_quality, optimize=True)
        )
        
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        return encoded, buffer.tell()
    
    def _encode_image(self, image: Image.Image) -> str:
        """Simple image encoding without compression"""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    async def _create_region_composite(self, original_image: Image.Image, 
                                     regions: List[Dict[str, Any]], 
                                     system_mode: str) -> Image.Image:
        """Create a composite image from important regions
        
        Args:
            original_image: The full screenshot
            regions: List of region dictionaries with x, y, width, height, importance
            system_mode: Current system mode (affects layout)
            
        Returns:
            Composite image containing the important regions
        """
        # Sort regions by importance
        regions = sorted(regions, key=lambda r: r['importance'], reverse=True)
        
        # Extract regions from original image
        cropped_regions = []
        for region in regions:
            try:
                # Get region bounds
                x, y = region['x'], region['y']
                width, height = region['width'], region['height']
                
                # Add small padding (2% of dimensions) for context
                padding_x = int(width * 0.02)
                padding_y = int(height * 0.02)
                
                # Calculate padded bounds (ensure within image bounds)
                x1 = max(0, x - padding_x)
                y1 = max(0, y - padding_y)
                x2 = min(original_image.width, x + width + padding_x)
                y2 = min(original_image.height, y + height + padding_y)
                
                # Crop region
                cropped = original_image.crop((x1, y1, x2, y2))
                
                # Add metadata for labeling
                cropped_regions.append({
                    'image': cropped,
                    'original_x': x,
                    'original_y': y,
                    'importance': region['importance'],
                    'width': x2 - x1,
                    'height': y2 - y1
                })
            except Exception as e:
                logger.warning(f"Failed to crop region: {e}")
                continue
        
        if not cropped_regions:
            logger.warning("No regions could be cropped, returning original image")
            return original_image
        
        # Calculate composite layout based on system mode
        if system_mode in ['critical', 'emergency']:
            # Stack vertically in critical/emergency modes (simpler layout)
            layout = 'vertical'
            gap = 5  # Minimal gap between regions
        else:
            # Try optimal grid layout in normal/pressure modes
            layout = 'grid'
            gap = 10  # Larger gap for clarity
        
        # Create composite image
        if layout == 'vertical':
            # Stack regions vertically
            composite_width = max(r['width'] for r in cropped_regions)
            composite_height = sum(r['height'] for r in cropped_regions) + gap * (len(cropped_regions) - 1)
            
            composite = Image.new('RGB', (composite_width, composite_height), color=(30, 30, 30))
            
            # Place regions
            y_offset = 0
            for i, region_data in enumerate(cropped_regions):
                cropped = region_data['image']
                composite.paste(cropped, (0, y_offset))
                
                # Add label
                self._add_region_label(composite, 0, y_offset, region_data, i + 1)
                
                y_offset += region_data['height'] + gap
        
        else:  # grid layout
            # Calculate grid dimensions (aim for roughly square layout)
            num_regions = len(cropped_regions)
            cols = max(1, int(np.sqrt(num_regions)))
            rows = (num_regions + cols - 1) // cols
            
            # Calculate cell dimensions based on largest regions
            cell_width = max(r['width'] for r in cropped_regions) + gap
            cell_height = max(r['height'] for r in cropped_regions) + gap
            
            composite_width = cols * cell_width
            composite_height = rows * cell_height
            
            composite = Image.new('RGB', (composite_width, composite_height), color=(30, 30, 30))
            
            # Place regions in grid
            for i, region_data in enumerate(cropped_regions):
                row = i // cols
                col = i % cols
                x_offset = col * cell_width
                y_offset = row * cell_height
                
                cropped = region_data['image']
                composite.paste(cropped, (x_offset, y_offset))
                
                # Add label
                self._add_region_label(composite, x_offset, y_offset, region_data, i + 1)
        
        logger.info(f"Created composite image: {composite.width}x{composite.height}, "
                   f"{len(cropped_regions)} regions, layout: {layout}")
        
        return composite
    
    def _add_region_label(self, image: Image.Image, x: int, y: int, 
                         region_data: Dict[str, Any], region_num: int):
        """Add a small label to a region in the composite
        
        Note: This is a simple labeling method. In production, you might want
        to use PIL.ImageDraw for better text rendering.
        """
        try:
            from PIL import ImageDraw, ImageFont
            
            draw = ImageDraw.Draw(image)
            
            # Create label text
            label = f"R{region_num}: ({region_data['original_x']},{region_data['original_y']}) "
            label += f"imp={region_data['importance']:.2f}"
            
            # Try to use a small font (fallback to default if not available)
            try:
                font = ImageFont.truetype("Arial", 10)
            except:
                font = ImageFont.load_default()
            
            # Add semi-transparent background for label
            bbox = draw.textbbox((x, y), label, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0, 180))
            
            # Draw text
            draw.text((x + 2, y + 2), label, fill=(255, 255, 0), font=font)
            
        except Exception as e:
            # Labeling is optional - don't fail the whole process
            logger.debug(f"Could not add label to region: {e}")
    
    def _enhance_prompt_for_ui_elements(self, prompt: str) -> str:
        """Enhance prompt to get specific UI element details from Claude"""
        prompt_lower = prompt.lower()
        
        # Define UI element patterns and their specific prompts
        ui_prompts = {
            'battery': (
                "Look at the status bar/menu bar (usually top-right on macOS, bottom-right on Windows). "
                "Find and report the EXACT battery percentage number. Don't say 'the battery icon shows...' - "
                "give me the specific percentage like '67%' or 'Battery: 82%'. If you see a charging icon, mention that too."
            ),
            'time': (
                "Look at the clock in the status bar/menu bar. Report the EXACT time displayed, "
                "including AM/PM if shown. For example: '2:34 PM' or '14:34'. Don't describe where the clock is, "
                "just tell me the exact time you see."
            ),
            'notifications': (
                "Check for notification badges, counts, or alerts. Look for: "
                "1) Red circles with numbers on dock/taskbar icons "
                "2) Notification banners or pop-ups "
                "3) Any unread counts visible "
                "Report the specific numbers and which apps have them."
            ),
            'wifi': (
                "Find the WiFi/network icon in the status bar. Report: "
                "1) Connection status (connected/disconnected) "
                "2) Signal strength if visible (full bars, half bars, etc.) "
                "3) Network name if displayed"
            ),
            'status_bar': (
                "Examine the entire status bar/menu bar carefully. List EVERY element you see with specific details: "
                "Time (exact), Battery (percentage), WiFi status, Bluetooth status, Volume level, "
                "Any app icons or indicators. Give me actual values, not descriptions."
            ),
            'screen': (
                "Analyze what's currently on screen. Be specific about: "
                "1) Active application names and window titles "
                "2) Any specific UI elements visible (buttons, text fields, etc.) "
                "3) Content being displayed "
                "4) Status bar details (time, battery %, etc.) "
                "Provide concrete details, not vague descriptions."
            )
        }
        
        # Check which UI element is being asked about
        enhanced_prompt = prompt
        prompt_matched = False
        
        for element, enhancement in ui_prompts.items():
            if element in prompt_lower or (
                element == 'battery' and any(word in prompt_lower for word in ['power', 'charge']) or
                element == 'time' and any(word in prompt_lower for word in ['clock', 'hour']) or
                element == 'notifications' and any(word in prompt_lower for word in ['alert', 'badge', 'unread']) or
                element == 'wifi' and any(word in prompt_lower for word in ['network', 'internet', 'connection']) or
                element == 'screen' and any(phrase in prompt_lower for phrase in ['what do you see', "what's on", 'describe'])
            ):
                enhanced_prompt = f"You are JARVIS, Tony Stark's AI assistant. {enhancement}\n\nUser asked: {prompt}"
                prompt_matched = True
                break
        
        # If no specific pattern matched but user is asking about seeing/screen
        if not prompt_matched and any(word in prompt_lower for word in ['see', 'screen', 'display', 'showing']):
            enhanced_prompt = (
                "You are JARVIS, Tony Stark's AI assistant. The user is asking about what's on their screen. "
                "Provide specific, detailed information about what you observe. Include: "
                "1) Exact values for time, battery percentage, etc. from the status bar "
                "2) Application names and window titles "
                "3) Any specific content or UI elements visible "
                "Don't give generic responses - be as specific as JARVIS would be.\n\n"
                f"User asked: {prompt}"
            )
        
        logger.debug(f"Enhanced prompt: {enhanced_prompt[:200]}...")
        return enhanced_prompt

    async def _call_claude_api(self, image_base64: str, prompt: str) -> str:
        """Make API call to Claude with timeout"""
        try:
            # Enhance the prompt for better UI element detection
            enhanced_prompt = self._enhance_prompt_for_ui_elements(prompt)
            
            logger.info(f"[CLAUDE API] Making API call to Claude with prompt: {enhanced_prompt[:100]}...")
            logger.info(f"[CLAUDE API] Image size: {len(image_base64)} chars")
            logger.info(f"[CLAUDE API] Model: {self.config.model_name}, Max tokens: {self.config.max_tokens}")
            
            # Ensure we have a client
            if not hasattr(self, 'client') or self.client is None:
                raise Exception("ANTHROPIC_API_KEY not configured. Please set the API key to use vision analysis.")
            
            # Create a future for the API call
            api_future = asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg" if self.config.compression_enabled else "image/png",
                                    "data": image_base64
                                }
                            },
                            {
                                "type": "text",
                                "text": enhanced_prompt
                            }
                        ]
                    }]
                )
            )
            
            # Wait with timeout (use configured timeout or default to 20 seconds)
            timeout = getattr(self.config, 'api_timeout', 20)
            message = await asyncio.wait_for(api_future, timeout=timeout)
            
            logger.info("[CLAUDE API] API call successful")
            return message.content[0].text
            
        except asyncio.TimeoutError:
            logger.error(f"Claude API call timed out after {timeout} seconds")
            raise Exception("Claude API timeout - the vision analysis is taking too long")
        except Exception as e:
            # Log the full error for debugging
            logger.error(f"Claude API call failed: {e}", exc_info=True)
            # Re-raise with more context
            raise Exception(f"Claude API error: {str(e)}")
    
    def _generate_prompt_embedding_sync(self, prompt: str) -> Optional[np.ndarray]:
        """Generate embedding for prompt (simplified for now)"""
        # In production, use a proper sentence transformer
        # For now, use a simple hash-based approach
        prompt_hash = hashlib.sha384(prompt.encode()).digest()
        # Convert to float array
        embedding = np.frombuffer(prompt_hash, dtype=np.uint8).astype(np.float32)
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        # Pad or truncate to standard dimension (384)
        if len(embedding) < 384:
            embedding = np.pad(embedding, (0, 384 - len(embedding)))
        else:
            embedding = embedding[:384]
        return embedding
    
    async def _generate_prompt_embedding(self, prompt: str) -> Optional[np.ndarray]:
        """Async version of prompt embedding generation"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor, self._generate_prompt_embedding_sync, prompt
        )
    
    def _get_time_context(self) -> str:
        """Get current time context"""
        hour = datetime.now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"
    
    def _extract_app_context(self, prompt: str, result: Dict[str, Any]) -> str:
        """Extract application context from prompt or analysis result"""
        # Try to extract from entities
        if 'entities' in result:
            if 'application' in result['entities']:
                return result['entities']['application']
            if 'app_name' in result['entities']:
                return result['entities']['app_name']
        
        # Try to extract from prompt
        app_patterns = [
            r'in (\w+)\s*app',
            r'(\w+)\s*application',
            r'using (\w+)',
            r'on (\w+)',
        ]
        
        for pattern in app_patterns:
            match = re.search(pattern, prompt, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        # Try to extract from window title in result
        if 'description' in result:
            desc_lower = result['description'].lower()
            # Common app names
            common_apps = ['chrome', 'safari', 'firefox', 'vscode', 'code', 'terminal', 
                          'finder', 'slack', 'discord', 'zoom', 'teams', 'outlook',
                          'mail', 'messages', 'whatsapp', 'telegram', 'notion',
                          'obsidian', 'spotify', 'music', 'preview', 'photoshop',
                          'illustrator', 'figma', 'sketch', 'xcode', 'intellij']
            
            for app in common_apps:
                if app in desc_lower:
                    return app
        
        # Default to generic identifier
        return 'unknown_app'
    
    def _parse_claude_response(self, response: str) -> Dict[str, Any]:
        """Enhanced response parsing with dynamic extraction"""
        # Try to extract JSON if present
        try:
            # Look for JSON blocks in various formats
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # Markdown code block
                r'\{[^{}]*\}',  # Simple JSON object
                r'\{.*\}',  # Complex JSON object
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response, re.DOTALL | re.MULTILINE)
                if match:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    parsed_json = json.loads(json_str)
                    # Add metadata
                    parsed_json['_metadata'] = {
                        'timestamp': datetime.now().isoformat(),
                        'response_type': 'json',
                        'confidence': 0.95
                    }
                    return parsed_json
        except Exception as e:
            logger.debug(f"JSON parsing failed: {e}")
        
        # Enhanced fallback parsing
        result = {
            "description": response,
            "timestamp": datetime.now().isoformat(),
            "confidence": self._estimate_confidence(response)
        }
        
        # Add optional extractions based on config
        if self.config.enable_entity_extraction:
            result["entities"] = self.entity_extractor.extract_entities(response)
        
        if self.config.enable_action_detection:
            result["actions"] = self._extract_actions_dynamic(response)
        
        # Generate summary
        result["summary"] = self._generate_summary(response)
        
        return result
    
    def _estimate_confidence(self, response: str) -> float:
        """Estimate confidence based on response characteristics"""
        confidence = 0.5  # Base confidence
        
        # Dynamic confidence indicators
        confidence_boosters = {
            'high': ['clearly', 'definitely', 'certainly', 'exactly', 'precisely'],
            'medium': ['appears', 'seems', 'likely', 'probably'],
            'low': ['might', 'possibly', 'unclear', 'uncertain']
        }
        
        response_lower = response.lower()
        
        # Check confidence indicators
        for level, words in confidence_boosters.items():
            for word in words:
                if word in response_lower:
                    if level == 'high':
                        confidence += 0.2
                    elif level == 'medium':
                        confidence += 0.1
                    else:
                        confidence -= 0.1
        
        # Length-based confidence
        if len(response) > 200:
            confidence += 0.1
        elif len(response) < 50:
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)
    
    def _extract_actions_dynamic(self, text: str) -> List[Dict[str, str]]:
        """Extract actionable items dynamically without hardcoded patterns"""
        actions = []
        
        # Dynamic action detection using linguistic patterns
        # Look for imperative verbs or action-suggesting phrases
        action_patterns = [
            # Imperative patterns
            r'(?:you should|please|need to|must|have to|ought to)\s+(\w+)',
            # Direct action verbs
            r'(?:^|\. )([A-Z]\w+)\s+(?:the|a|an|your)',
            # Action recommendations
            r'(?:recommend|suggest|advise)\s+(?:to\s+)?(\w+)',
            # Modal verbs suggesting actions
            r'(?:can|could|would|should)\s+(\w+)\s+(?:the|a|an|your)'
        ]
        
        sentences = text.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check each pattern
            for pattern in action_patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                for match in matches:
                    verb = match.group(1) if match.lastindex else match.group(0)
                    
                    # Create action entry
                    action = {
                        "type": "detected_action",
                        "verb": verb.lower(),
                        "description": sentence,
                        "priority": self._estimate_action_priority_dynamic(sentence),
                        "confidence": self._estimate_confidence(sentence)
                    }
                    
                    # Avoid duplicates
                    if not any(a['description'] == action['description'] for a in actions):
                        actions.append(action)
        
        # Sort by priority and confidence
        actions.sort(key=lambda x: (x['priority'] == 'high', x['confidence']), reverse=True)
        
        return actions[:10]  # Limit to top 10 actions
    
    def _estimate_action_priority_dynamic(self, text: str) -> str:
        """Dynamically estimate action priority"""
        text_lower = text.lower()
        
        # Priority indicators (learned from context)
        high_priority_indicators = [
            'critical', 'urgent', 'immediately', 'security', 'error', 
            'fail', 'crash', 'warning', 'alert', 'important'
        ]
        
        medium_priority_indicators = [
            'should', 'recommend', 'suggest', 'update', 'improve',
            'optimize', 'enhance', 'consider'
        ]
        
        # Check indicators
        if any(indicator in text_lower for indicator in high_priority_indicators):
            return "high"
        elif any(indicator in text_lower for indicator in medium_priority_indicators):
            return "medium"
        else:
            return "low"
    
    def _generate_summary(self, text: str) -> str:
        """Generate a brief summary of the response"""
        # Dynamic summary generation
        sentences = text.split('.')
        
        # Filter out very short sentences
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if not meaningful_sentences:
            return text[:150].strip() + '...' if len(text) > 150 else text
        
        # Take first 1-2 meaningful sentences
        if len(meaningful_sentences) >= 2:
            return '. '.join(meaningful_sentences[:2]) + '.'
        else:
            return meaningful_sentences[0] + '.'
    
    def _generate_cache_key(self, image_hash: str, prompt: str) -> str:
        """Generate cache key from image and prompt"""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return f"{image_hash[:16]}_{prompt_hash[:16]}"
    
    def _is_cache_valid(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        age = datetime.now() - entry.timestamp
        return age < timedelta(minutes=self.config.cache_ttl_minutes)
    
    def _estimate_image_size(self, image: Image.Image) -> int:
        """Estimate image size in bytes"""
        # Rough estimate: width * height * channels * bytes_per_channel
        channels = len(image.getbands())
        return image.width * image.height * channels
    
    def _get_system_load(self) -> float:
        """Get current system load (0.0 to 1.0)"""
        return psutil.cpu_percent(interval=0.1) / 100.0
    
    def _track_metrics(self, metrics: AnalysisMetrics):
        """Track performance metrics"""
        self.recent_metrics.append(metrics)
        # Keep only last N metrics (configurable)
        max_metrics = int(os.getenv('VISION_MAX_METRICS', '100'))
        if len(self.recent_metrics) > max_metrics:
            self.recent_metrics.pop(0)
    
    # Dynamic analysis methods
    
    async def analyze_with_template(self, screenshot: np.ndarray, 
                                  template_name: str, 
                                  **template_vars) -> Dict[str, Any]:
        """Analyze using a named template with variables"""
        if template_name not in self.prompt_templates:
            raise ValueError(f"Unknown template: {template_name}")
        
        prompt = self.prompt_templates[template_name]
        
        # Format template with provided variables
        if template_vars:
            prompt = prompt.format(**template_vars)
        
        result, metrics = await self.analyze_screenshot(screenshot, prompt)
        return result
    
    async def analyze_workspace(self, screenshot: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze workspace - wrapper for comprehensive analysis"""
        return await self.analyze_workspace_comprehensive(screenshot)
    
    async def read_text_from_area(self, screenshot: np.ndarray, area: Dict[str, int]) -> Dict[str, Any]:
        """Read text from a specific area of the screenshot"""
        try:
            # Crop the image to the specified area
            x, y, width, height = area['x'], area['y'], area['width'], area['height']
            
            # Ensure coordinates are within bounds
            h, w = screenshot.shape[:2]
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            width = min(width, w - x)
            height = min(height, h - y)
            
            # Crop the area
            cropped = screenshot[y:y+height, x:x+width]
            
            # Analyze the cropped area with text strategy
            result = await self.analyze_with_compression_strategy(
                cropped,
                "Read all text in this area",
                "text"
            )
            
            return {
                'success': True,
                'text': result.get('description', ''),
                'area': area
            }
            
        except Exception as e:
            logger.error(f"Error reading text from area: {e}")
            return {
                'success': False,
                'error': str(e),
                'area': area
            }
    
    async def analyze_workspace_comprehensive(self, screenshot: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Comprehensive workspace analysis using all enhanced components"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'components_used': [],
            'memory_stats': {}
        }
        
        # Get window analyzer
        window_analyzer = await self.get_window_analyzer()
        if window_analyzer:
            try:
                workspace_analysis = await window_analyzer.analyze_workspace()
                results['workspace'] = workspace_analysis
                results['components_used'].append('window_analyzer')
                results['memory_stats']['window_analyzer'] = window_analyzer.get_memory_stats()
            except Exception as e:
                logger.error(f"Window analysis failed: {e}")
                results['workspace'] = {'error': str(e)}
        
        # Get relationship detector
        relationship_detector = await self.get_relationship_detector()
        if relationship_detector and 'workspace' in results:
            try:
                windows = results['workspace'].get('windows', [])
                if windows:
                    relationships = relationship_detector.detect_relationships(windows)
                    groups = relationship_detector.group_windows(windows, relationships)
                    results['relationships'] = {
                        'detected': len(relationships),
                        'groups': len(groups),
                        'details': relationships[:5]  # Top 5 relationships
                    }
                    results['components_used'].append('relationship_detector')
                    results['memory_stats']['relationship_detector'] = relationship_detector.get_stats()
            except Exception as e:
                logger.error(f"Relationship detection failed: {e}")
                results['relationships'] = {'error': str(e)}
        
        # Analyze screenshot if provided
        if screenshot is not None:
            try:
                # Use smart analysis
                vision_result = await self.smart_analyze(
                    screenshot,
                    "Analyze the workspace and identify what the user is working on"
                )
                results['vision_analysis'] = vision_result
                results['components_used'].append('vision_analyzer')
            except Exception as e:
                logger.error(f"Vision analysis failed: {e}")
                results['vision_analysis'] = {'error': str(e)}
        
        # Get Swift vision stats if used
        swift_vision = await self.get_swift_vision()
        if swift_vision:
            results['memory_stats']['swift_vision'] = swift_vision.get_memory_stats()
        
        # Get memory-efficient analyzer stats
        mem_analyzer = await self.get_memory_efficient_analyzer()
        if mem_analyzer:
            results['memory_stats']['memory_efficient_analyzer'] = mem_analyzer.get_metrics()
            results['components_used'].append('memory_efficient_analyzer')
        
        # Get simplified vision stats
        simplified = await self.get_simplified_vision()
        if simplified:
            results['memory_stats']['simplified_vision'] = simplified.get_performance_stats()
            results['components_used'].append('simplified_vision')
        
        # Overall memory stats
        results['memory_stats']['system'] = {
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'used_percent': psutil.virtual_memory().percent,
            'process_mb': psutil.Process().memory_info().rss / 1024 / 1024
        }
        
        return results
    
    async def start_continuous_monitoring(self, event_callbacks: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """Start continuous screen monitoring with memory management"""
        analyzer = await self.get_continuous_analyzer()
        if analyzer:
            # Register event callbacks if provided
            if event_callbacks:
                for event_type, callback in event_callbacks.items():
                    analyzer.register_callback(event_type, callback)
            
            await analyzer.start_monitoring()
            
            # Update config to reflect monitoring is active
            self.config.enable_continuous_monitoring = True
            
            return {'success': True}
        return {'success': False}
    
    async def stop_continuous_monitoring(self) -> bool:
        """Stop continuous screen monitoring"""
        if self.continuous_analyzer:
            await self.continuous_analyzer.stop_monitoring()
            return True
        return False
    
    async def get_current_screen_context(self) -> Dict[str, Any]:
        """Get current screen context from continuous analyzer"""
        analyzer = await self.get_continuous_analyzer()
        if analyzer and analyzer.is_monitoring:
            return await analyzer.get_current_screen_context()
        else:
            # Fallback to quick analysis
            return await self.quick_analysis(None, detail_level="brief")
    
    async def query_screen_for_weather(self) -> Optional[str]:
        """Query screen for weather information using continuous analyzer"""
        analyzer = await self.get_continuous_analyzer()
        if analyzer:
            return await analyzer.query_screen_for_weather()
        
        # Fallback to simplified vision
        simplified = await self.get_simplified_vision()
        if simplified:
            result = await simplified.check_weather_visible()
            if result['success']:
                return result['analysis']
        
        # Final fallback - direct weather analysis
        return await self.analyze_weather_directly()
    
    async def analyze_weather_directly(self) -> Optional[str]:
        """Direct weather analysis by capturing and reading Weather app"""
        try:
            # Capture current screen
            screenshot = await self.capture_screen()
            if screenshot is None:
                return None
            
            # Convert to numpy array if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            # Analyze for weather specifically
            result = await self.smart_analyze(
                screenshot,
                "Focus ONLY on the Weather app if visible. Extract: 1) Current temperature (exact number), 2) Current conditions (sunny/cloudy/etc), 3) Today's high/low temperatures, 4) Location city name, 5) Any precipitation chance. Be very specific with the numbers you see."
            )
            
            # Extract description from result
            if isinstance(result, dict):
                description = result.get('description', result.get('summary', ''))
                if description and any(word in description.lower() for word in ['temperature', 'degrees', '°', 'weather']):
                    return description
            
            return None
            
        except Exception as e:
            logger.error(f"Direct weather analysis failed: {e}")
            return None
    
    async def analyze_weather_fast(self, screenshot: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Fast single-region weather analysis optimized for speed"""
        try:
            # Capture screen if not provided
            if screenshot is None:
                screenshot = await self.capture_screen()
                if screenshot is None:
                    return {"success": False, "error": "Failed to capture screen"}
            
            # Convert PIL to numpy if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            # Use direct API call with optimized settings
            prompt = """You are looking at the macOS Weather app. Focus on the MAIN WEATHER DISPLAY (center/right part of screen), NOT the sidebar.
Extract from the MAIN display only:
1. Location name (shown at top of main display)
2. Current temperature (the large number in center)
3. Current condition (Clear, Cloudy, Rain, etc.)
4. Today's high/low if visible
5. Any precipitation percentage shown

IMPORTANT: Read the currently selected location's weather, not locations from the sidebar.
Be concise. Format: Location: X, Temp: X°F, Condition: X, High/Low: X/X"""

            # Prepare image for API
            from PIL import Image as PILImage
            import io
            import base64
            
            # Convert and resize
            img = PILImage.fromarray(screenshot)
            
            # Convert RGBA to RGB if necessary
            if img.mode == 'RGBA':
                rgb_img = PILImage.new('RGB', img.size, (255, 255, 255))
                rgb_img.paste(img, mask=img.split()[3] if img.mode == 'RGBA' else None)
                img = rgb_img
            
            # Resize if needed
            max_dim = 1280
            if img.width > max_dim or img.height > max_dim:
                img.thumbnail((max_dim, max_dim), PILImage.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            base64_image = base64.b64encode(buffer.getvalue()).decode()
            
            # Direct API call using httpx
            import httpx
            
            # Get API key from client
            api_key = self.client.api_key if hasattr(self.client, 'api_key') else os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                return {"success": False, "error": "No API key available"}
                
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 200,
                "temperature": 0.3,
                "messages": [{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            }
            
            # Make the API call
            logger.info("[WEATHER] Making direct API call to Claude")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=data,
                    timeout=10.0
                )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('content', [{}])[0].get('text', '')
                logger.info(f"[WEATHER] Got response: {content[:100]}...")
                return {
                    "success": True,
                    "analysis": content,
                    "method": "direct_api_call"
                }
            else:
                logger.error(f"[WEATHER] API error: {response.status_code}")
                return {
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Fast weather analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def quick_analysis(self, screenshot: np.ndarray, 
                           detail_level: str = "brief") -> Dict[str, Any]:
        """Configurable quick analysis"""
        # Adjust config temporarily for quick analysis
        custom_config = {
            'max_image_dimension': 1024 if detail_level == "brief" else 1280,
            'jpeg_quality': 70 if detail_level == "brief" else 85,
            'max_tokens': 500 if detail_level == "brief" else 1000
        }
        
        prompt = self.prompt_templates.get(detail_level, self.prompt_templates['quick'])
        result, metrics = await self.analyze_screenshot(
            screenshot, prompt, 
            priority="high",
            custom_config=custom_config
        )
        return result
    
    async def batch_analyze(self, screenshots: List[np.ndarray], 
                          prompts: List[str],
                          batch_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Analyze multiple screenshots with optional batch configuration"""
        tasks = []
        for screenshot, prompt in zip(screenshots, prompts):
            task = self.analyze_screenshot(
                screenshot, prompt, 
                custom_config=batch_config
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return [result for result, _ in results]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.recent_metrics:
            return {"message": "No metrics available"}
        
        total_times = [m.total_time for m in self.recent_metrics]
        cache_hits = sum(1 for m in self.recent_metrics if m.cache_hit)
        
        stats = {
            "performance": {
                "avg_total_time": sum(total_times) / len(total_times),
                "min_total_time": min(total_times),
                "max_total_time": max(total_times),
                "p95_time": sorted(total_times)[int(len(total_times) * 0.95)] if len(total_times) > 20 else max(total_times)
            },
            "cache": {
                "hit_rate": cache_hits / len(self.recent_metrics),
                "total_hits": cache_hits,
                "total_requests": len(self.recent_metrics)
            },
            "compression": {
                "avg_ratio": sum(m.compression_ratio for m in self.recent_metrics) / len(self.recent_metrics),
                "total_saved_bytes": sum(m.image_size_original - m.image_size_compressed for m in self.recent_metrics)
            },
            "api": {
                "total_calls": len([m for m in self.recent_metrics if not m.cache_hit]),
                "avg_api_time": sum(m.api_call_time for m in self.recent_metrics if not m.cache_hit) / max(1, len([m for m in self.recent_metrics if not m.cache_hit]))
            },
            "config": self.config.to_dict()
        }
        
        return stats
    
    async def get_workflow_stats(self) -> Dict[str, Any]:
        """Get workflow pattern statistics"""
        stats = {
            'enabled': self._workflow_pattern_config['enabled'],
            'patterns': {},
            'predictions': {},
            'automation_potential': {}
        }
        
        if not self._workflow_pattern_config['enabled']:
            return stats
        
        try:
            engine = await self.get_workflow_engine()
            if not engine:
                return stats
            
            # Get pattern statistics
            if hasattr(engine, 'get_pattern_statistics'):
                pattern_stats = engine.get_pattern_statistics()
                stats['patterns'] = pattern_stats
            
            # Get memory usage
            if hasattr(engine, 'get_memory_usage'):
                stats['memory_usage'] = engine.get_memory_usage()
            
            # Get pattern counts by type
            patterns = await self.get_workflow_patterns()
            stats['pattern_counts'] = {}
            for pattern in patterns:
                ptype = pattern.pattern_type.value if hasattr(pattern.pattern_type, 'value') else str(pattern.pattern_type)
                stats['pattern_counts'][ptype] = stats['pattern_counts'].get(ptype, 0) + 1
            
            # Get automation suggestions count
            suggestions = await self.get_automation_suggestions()
            stats['automation_potential'] = {
                'total_suggestions': len(suggestions),
                'high_benefit': len([s for s in suggestions if s.get('benefit_score', 0) > 0.8]),
                'estimated_time_savings': sum(s.get('estimated_time_saved', 0) for s in suggestions)
            }
            
            # Add event count if available
            if hasattr(self, '_workflow_event_count'):
                stats['events_recorded'] = self._workflow_event_count
            
        except Exception as e:
            logger.error(f"Error getting workflow stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    async def clear_cache(self):
        """Clear the response cache"""
        if self.cache:
            await self.cache.clear()
            logger.info("Cache cleared")
    
    def save_config(self, path: str):
        """Save current configuration to file"""
        with open(path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")
    
    # Sliding Window Integration
    
    async def analyze_with_sliding_window(self, screenshot: np.ndarray, 
                                         query: str,
                                         window_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analyze screenshot using sliding window approach for memory efficiency
        
        Args:
            screenshot: Screenshot as numpy array
            query: Analysis query
            window_config: Optional window configuration override
            
        Returns:
            Combined analysis results from all windows
        """
        # Default sliding window configuration
        default_window_config = {
            'window_width': int(os.getenv('VISION_WINDOW_WIDTH', '400')),
            'window_height': int(os.getenv('VISION_WINDOW_HEIGHT', '300')),
            'overlap': float(os.getenv('VISION_WINDOW_OVERLAP', '0.3')),
            'max_windows': int(os.getenv('VISION_MAX_WINDOWS', '4')),
            'prioritize_center': os.getenv('VISION_PRIORITIZE_CENTER', 'true').lower() == 'true',
            'adaptive_sizing': os.getenv('VISION_ADAPTIVE_SIZING', 'true').lower() == 'true'
        }
        
        # Merge with provided config
        if window_config:
            default_window_config.update(window_config)
        
        # Generate sliding windows
        windows = self._generate_sliding_windows(screenshot, default_window_config)
        
        # Analyze each window
        window_results = []
        for i, window in enumerate(windows):
            window_prompt = self._create_window_prompt(query, i, len(windows), window['bounds'])
            
            # Extract window region
            x, y, w, h = window['bounds']
            window_image = screenshot[y:y+h, x:x+w]
            
            # Analyze window with caching
            result, metrics = await self.analyze_screenshot(
                window_image, 
                window_prompt,
                custom_config={'max_tokens': 300}  # Smaller tokens for windows
            )
            
            window_results.append({
                'bounds': window['bounds'],
                'priority': window['priority'],
                'result': result,
                'metrics': metrics
            })
        
        # Combine results
        combined_result = self._combine_window_results(window_results, query)
        combined_result['metadata'] = {
            'analysis_method': 'sliding_window',
            'windows_analyzed': len(windows),
            'window_config': default_window_config
        }
        
        return combined_result
    
    def _generate_sliding_windows(self, image: np.ndarray, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sliding windows for the image"""
        height, width = image.shape[:2]
        window_width = config['window_width']
        window_height = config['window_height']
        
        # Adaptive sizing based on available memory
        if config['adaptive_sizing']:
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            if available_mb < 2000:  # Less than 2GB available
                window_width = int(window_width * 0.75)
                window_height = int(window_height * 0.75)
        
        # Calculate step sizes with overlap
        overlap = config['overlap']
        step_x = int(window_width * (1 - overlap))
        step_y = int(window_height * (1 - overlap))
        
        windows = []
        
        # Generate windows
        for y in range(0, height - window_height + 1, step_y):
            for x in range(0, width - window_width + 1, step_x):
                # Calculate priority (center gets higher priority)
                priority = 1.0
                if config['prioritize_center']:
                    center_x = x + window_width // 2
                    center_y = y + window_height // 2
                    image_center_x = width // 2
                    image_center_y = height // 2
                    
                    # Distance from center (normalized)
                    dx = (center_x - image_center_x) / width
                    dy = (center_y - image_center_y) / height
                    distance = np.sqrt(dx**2 + dy**2)
                    priority = 1.0 - min(distance, 1.0)
                
                windows.append({
                    'bounds': (x, y, window_width, window_height),
                    'priority': priority
                })
        
        # Sort by priority and limit to max_windows
        windows.sort(key=lambda w: w['priority'], reverse=True)
        return windows[:config['max_windows']]
    
    def _create_window_prompt(self, base_query: str, window_index: int, 
                            total_windows: int, bounds: Tuple[int, int, int, int]) -> str:
        """Create analysis prompt for a specific window"""
        x, y, w, h = bounds
        
        return f"""Analyze this portion of the screen (region {window_index+1}/{total_windows} at position x:{x}, y:{y}).
This is part of a larger screen being analyzed in sections.

{base_query}

Focus on what's visible in this specific region. Be concise but thorough."""
    
    def _combine_window_results(self, window_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Combine results from multiple windows into a unified response"""
        # Aggregate all findings
        all_descriptions = []
        all_entities = {'applications': [], 'files': [], 'urls': [], 'ui_elements': []}
        all_actions = []
        total_confidence = 0
        high_priority_regions = []
        
        for window_result in window_results:
            result = window_result['result']
            bounds = window_result['bounds']
            priority = window_result['priority']
            
            # Collect descriptions
            if 'description' in result:
                all_descriptions.append(f"Region ({bounds[0]}, {bounds[1]}): {result['description']}")
            
            # Aggregate entities
            if 'entities' in result:
                for key in all_entities:
                    if key in result['entities']:
                        all_entities[key].extend(result['entities'][key])
            
            # Collect actions
            if 'actions' in result:
                all_actions.extend(result['actions'])
            
            # Track confidence
            confidence = result.get('confidence', 0.5)
            total_confidence += confidence * priority  # Weight by priority
            
            # Track high priority regions
            if priority > 0.7 and confidence > 0.7:
                high_priority_regions.append({
                    'bounds': bounds,
                    'summary': result.get('summary', result.get('description', ''))[:100],
                    'confidence': confidence
                })
        
        # Remove duplicates from entities
        for key in all_entities:
            all_entities[key] = list(dict.fromkeys(all_entities[key]))
        
        # Generate combined summary
        if high_priority_regions:
            summary = f"Found {len(high_priority_regions)} important regions. "
            summary += high_priority_regions[0]['summary']
        else:
            summary = f"Analyzed {len(window_results)} screen regions. "
            if all_descriptions:
                summary += all_descriptions[0].split(': ', 1)[1] if ': ' in all_descriptions[0] else all_descriptions[0]
        
        # Calculate average confidence
        avg_confidence = total_confidence / len(window_results) if window_results else 0
        
        return {
            'summary': summary,
            'description': '\n'.join(all_descriptions[:5]),  # Top 5 descriptions
            'entities': all_entities,
            'actions': all_actions[:10],  # Top 10 actions
            'confidence': avg_confidence,
            'important_regions': high_priority_regions[:3],  # Top 3 regions
            'total_regions_analyzed': len(window_results)
        }
    
    async def smart_analyze(self, screenshot: np.ndarray, query: str, 
                          force_method: Optional[str] = None) -> Dict[str, Any]:
        """
        Smart analysis that automatically chooses between full or sliding window
        
        Args:
            screenshot: Screenshot to analyze
            query: Analysis query
            force_method: Force specific method ('full' or 'sliding_window')
            
        Returns:
            Analysis results with metadata about method used
        """
        height, width = screenshot.shape[:2]
        total_pixels = height * width
        available_mb = psutil.virtual_memory().available / 1024 / 1024
        
        # Decision thresholds (configurable)
        pixel_threshold = int(os.getenv('VISION_SLIDING_THRESHOLD_PX', '800000'))  # 800k pixels
        memory_threshold = float(os.getenv('VISION_SLIDING_MEMORY_MB', '2000'))  # 2GB
        
        # Determine method
        if force_method:
            use_sliding = (force_method == 'sliding_window')
        else:
            # Use sliding window if:
            # 1. Image is large
            # 2. Low memory available
            # 3. Query suggests detailed search
            use_sliding = (
                total_pixels > pixel_threshold or
                available_mb < memory_threshold or
                any(word in query.lower() for word in ['find', 'locate', 'search', 'where', 'all'])
            )
        
        # Log decision
        logger.info(f"Smart analyze: {'sliding window' if use_sliding else 'full'} "
                   f"(pixels: {total_pixels}, memory: {available_mb:.0f}MB)")
        
        # Perform analysis
        if use_sliding:
            result = await self.analyze_with_sliding_window(screenshot, query)
        else:
            result, metrics = await self.analyze_screenshot(screenshot, query)
            result['metadata'] = {
                'analysis_method': 'full',
                'metrics': metrics.__dict__
            }
        
        return result
    
    async def analyze_screenshot_async(self, screenshot: np.ndarray, query: str,
                                     quick_mode: bool = False,
                                     use_sliding_window: Optional[bool] = None) -> Dict[str, Any]:
        """
        Convenience method for backward compatibility and easy async usage
        
        Args:
            screenshot: Screenshot to analyze
            query: Analysis query
            quick_mode: Use quick analysis mode
            use_sliding_window: Force sliding window mode
            
        Returns:
            Analysis results
        """
        if quick_mode:
            return await self.quick_analysis(screenshot, detail_level="brief")
        elif use_sliding_window is not None:
            method = 'sliding_window' if use_sliding_window else 'full'
            return await self.smart_analyze(screenshot, query, force_method=method)
        else:
            return await self.smart_analyze(screenshot, query)
    
    # New methods for memory-efficient analysis
    
    async def analyze_with_compression_strategy(self, screenshot: np.ndarray, prompt: str,
                                              strategy: str = "ui") -> Dict[str, Any]:
        """Analyze using memory-efficient compression strategy"""
        mem_analyzer = await self.get_memory_efficient_analyzer()
        if mem_analyzer:
            return await mem_analyzer.analyze_screenshot(screenshot, prompt, strategy)
        else:
            # Fallback to standard analysis
            result, metrics = await self.analyze_screenshot(screenshot, prompt)
            return result
    
    async def batch_analyze_regions(self, screenshot: np.ndarray, 
                                  regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch analyze multiple regions efficiently"""
        mem_analyzer = await self.get_memory_efficient_analyzer()
        if mem_analyzer:
            return await mem_analyzer.batch_analyze_regions(screenshot, regions)
        else:
            # Fallback to sequential analysis
            results = []
            for region in regions:
                x, y, w, h = region.get('x', 0), region.get('y', 0), \
                           region.get('width', 100), region.get('height', 100)
                region_img = screenshot[y:y+h, x:x+w]
                prompt = region.get('prompt', 'Analyze this region')
                result, _ = await self.analyze_screenshot(region_img, prompt)
                result['region'] = region
                results.append(result)
            return results
    
    async def analyze_with_change_detection(self, current: np.ndarray, 
                                          previous: Optional[np.ndarray],
                                          prompt: str) -> Dict[str, Any]:
        """Analyze only if significant changes detected"""
        mem_analyzer = await self.get_memory_efficient_analyzer()
        if mem_analyzer:
            return await mem_analyzer.analyze_with_change_detection(current, previous, prompt)
        else:
            # Simple fallback - always analyze
            result, _ = await self.analyze_screenshot(current, prompt)
            return result
    
    # New methods for simplified vision queries
    
    async def check_for_notifications(self) -> Dict[str, Any]:
        """Check for notifications using simplified vision"""
        simplified = await self.get_simplified_vision()
        if simplified:
            return await simplified.check_for_notifications()
        else:
            # Fallback to standard analysis
            return await self.analyze_with_template(None, 'action')
    
    async def check_for_errors(self) -> Dict[str, Any]:
        """Check for errors on screen"""
        simplified = await self.get_simplified_vision()
        if simplified:
            return await simplified.check_for_errors()
        else:
            # Fallback to standard analysis
            return await self.analyze_with_template(None, 'error')
    
    async def find_ui_element(self, element_description: str) -> Dict[str, Any]:
        """Find specific UI element"""
        simplified = await self.get_simplified_vision()
        if simplified:
            return await simplified.find_element(element_description)
        else:
            # Fallback to standard analysis
            prompt = f"Locate the following element: {element_description}"
            result, _ = await self.analyze_screenshot(None, prompt)
            return {'success': True, 'analysis': result.get('description', '')}
    
    async def cleanup_all_components(self):
        """Cleanup all enhanced components"""
        cleanup_tasks = []
        
        # Stop screen sharing if active
        if self.screen_sharing and self.screen_sharing.is_sharing:
            cleanup_tasks.append(self.screen_sharing.stop_sharing())
        
        # Stop video streaming if active
        if hasattr(self, 'video_streaming') and self.video_streaming and self.video_streaming.is_capturing:
            cleanup_tasks.append(self.video_streaming.stop_streaming())
        
        # Cleanup continuous analyzer
        if self.continuous_analyzer:
            cleanup_tasks.append(self.continuous_analyzer.stop_monitoring())
        
        # Cleanup window analyzer
        if self.window_analyzer:
            cleanup_tasks.append(self.window_analyzer.cleanup())
        
        # Cleanup Swift vision
        if self.swift_vision:
            cleanup_tasks.append(self.swift_vision.cleanup())
        
        # Cleanup memory-efficient analyzer
        if self.memory_efficient_analyzer:
            cleanup_count = self.memory_efficient_analyzer.cleanup_old_cache()
            logger.info(f"Cleaned up {cleanup_count} cache entries from memory-efficient analyzer")
        
        # Save VSMS states
        if hasattr(self, '_vsms_core') and self._vsms_core:
            self.save_vsms_states()
        
        # Save Vision Intelligence states
        if hasattr(self, '_vision_intelligence') and self._vision_intelligence:
            self._vision_intelligence.save_learned_states()
        
        # Cleanup cache
        if self.cache:
            cleanup_tasks.append(self.cache.clear())
        
        # Wait for all cleanup tasks
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Force garbage collection
        gc.collect()
        
        logger.info("All enhanced components cleaned up")
    
    async def capture_screen(self) -> Any:
        """Capture screen using the best available method with robust error handling"""
        logger.info("[CAPTURE SCREEN] Starting screen capture")
        capture_methods = []
        
        # Method 1: Video streaming (if available)
        if self.config.prefer_video_over_screenshots and hasattr(self, 'video_streaming'):
            capture_methods.append(("video_streaming", self._capture_from_video_stream))
        
        # Method 2: macOS screencapture command
        if platform.system() == 'Darwin':
            capture_methods.append(("macos_screencapture", self._capture_macos_screencapture))
        
        # Method 3: PIL ImageGrab
        capture_methods.append(("pil_imagegrab", self._capture_pil_imagegrab))
        
        # Try each method in order
        for method_name, method_func in capture_methods:
            try:
                logger.info(f"[CAPTURE SCREEN] Trying method: {method_name}")
                result = await method_func()
                if result is not None:
                    logger.info(f"[CAPTURE SCREEN] Success with {method_name}")
                    # Validate the captured image
                    if self._validate_captured_image(result):
                        return result
                    else:
                        logger.warning(f"[CAPTURE SCREEN] {method_name} returned invalid image")
            except Exception as e:
                logger.warning(f"[CAPTURE SCREEN] {method_name} failed: {type(e).__name__}: {e}")
                continue
        
        # All methods failed
        logger.error("[CAPTURE SCREEN] All capture methods failed")
        return None
    
    async def _capture_from_video_stream(self) -> Optional[Image.Image]:
        """Capture from video stream if available"""
        if not (self.video_streaming and hasattr(self.video_streaming, 'is_capturing') and self.video_streaming.is_capturing):
            return None
            
        if not hasattr(self.video_streaming, 'frame_buffer'):
            return None
            
        frame_data = self.video_streaming.frame_buffer.get_latest_frame()
        if frame_data and 'data' in frame_data:
            frame = frame_data['data']
            if isinstance(frame, np.ndarray) and frame.size > 0:
                return Image.fromarray(frame)
        return None
    
    async def _capture_macos_screencapture(self) -> Optional[Image.Image]:
        """Capture using macOS screencapture command"""
        import tempfile
        import subprocess
        
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Run screencapture with timeout
            result = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    'screencapture', '-C', '-x', '-t', 'png', tmp_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                ),
                timeout=5.0
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0 and os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                image = Image.open(tmp_path)
                return image
            else:
                if stderr:
                    logger.error(f"screencapture stderr: {stderr.decode()}")
                return None
                
        except asyncio.TimeoutError:
            logger.error("screencapture timed out after 5 seconds")
            return None
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
    async def _capture_pil_imagegrab(self) -> Optional[Image.Image]:
        """Capture using PIL ImageGrab"""
        try:
            from PIL import ImageGrab
            # Run in thread to avoid blocking
            return await asyncio.to_thread(ImageGrab.grab)
        except Exception as e:
            logger.error(f"PIL ImageGrab failed: {e}")
            return None
    
    def _validate_captured_image(self, image: Any) -> bool:
        """Validate that captured image is valid"""
        if image is None:
            return False
            
        if isinstance(image, Image.Image):
            # Check if image has valid size
            width, height = image.size
            if width <= 0 or height <= 0:
                return False
            # Check if image has content (not all black/white)
            try:
                extrema = image.convert('L').getextrema()
                if extrema[0] == extrema[1]:  # All pixels same value
                    return False
            except:
                pass
            return True
            
        return False
    
    async def describe_screen(self, params: Dict[str, Any]) -> Any:
        """Describe screen for continuous analyzer compatibility"""
        # Extract query from params
        query = params.get('query', 'Describe what you see on screen')
        
        # Check if screenshot provided, otherwise capture
        screenshot = params.get('screenshot')
        if screenshot is None:
            # Capture current screen
            logger.info("No screenshot provided, capturing current screen")
            screenshot = await self.capture_screen()
            if screenshot is None:
                return {'success': False, 'description': 'Unable to capture screen'}
            
            # Convert to numpy array if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
        
        result = await self.smart_analyze(screenshot, query)
        return {
            'success': True,
            'description': result.get('description', result.get('summary', '')),
            'data': result
        }
    
    def get_all_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics from all components"""
        stats = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Get stats from each component
        if self.continuous_analyzer:
            stats['components']['continuous_analyzer'] = self.continuous_analyzer.get_memory_stats()
        
        if self.window_analyzer:
            stats['components']['window_analyzer'] = self.window_analyzer.get_memory_stats()
        
        if self.relationship_detector:
            stats['components']['relationship_detector'] = self.relationship_detector.get_stats()
        
        if self.swift_vision:
            stats['components']['swift_vision'] = self.swift_vision.get_memory_stats()
        
        if self.memory_efficient_analyzer:
            stats['components']['memory_efficient_analyzer'] = self.memory_efficient_analyzer.get_metrics()
        
        if self.simplified_vision:
            stats['components']['simplified_vision'] = self.simplified_vision.get_performance_stats()
        
        if self.screen_sharing:
            stats['components']['screen_sharing'] = self.screen_sharing.get_metrics()
        
        if hasattr(self, 'video_streaming') and self.video_streaming:
            stats['components']['video_streaming'] = self.video_streaming.get_metrics()
        
        # Overall stats
        stats['system'] = {
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'used_percent': psutil.virtual_memory().percent,
            'process_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'cpu_percent': psutil.cpu_percent(interval=0.1)
        }
        
        # Memory safety status
        stats['memory_safety'] = self.memory_monitor.get_status_dict()
        
        return stats
    
    async def check_memory_health(self) -> Dict[str, Any]:
        """Check memory health and provide recommendations"""
        status = self.memory_monitor.check_memory_safety()
        health = {
            'healthy': status.is_safe,
            'process_mb': status.process_mb,
            'system_available_gb': status.system_available_gb,
            'warnings': status.warnings,
            'rejected_requests': self.memory_monitor.rejected_requests,
            'emergency_mode': self.memory_monitor._emergency_mode,
            'recommendations': []
        }
        
        # Add recommendations based on status
        if not status.is_safe:
            health['recommendations'].append("System memory critically low - consider reducing concurrent requests")
        
        if status.process_mb > self.config.memory_warning_threshold_mb:
            health['recommendations'].append(f"Process memory high ({status.process_mb:.0f}MB) - consider clearing cache")
        
        if self.memory_monitor._emergency_mode:
            health['recommendations'].append("Emergency mode active - only critical requests will be processed")
        
        if self.memory_monitor.rejected_requests > 10:
            health['recommendations'].append(f"High rejection rate ({self.memory_monitor.rejected_requests} requests) - consider scaling limits")
        
        return health
    
    def get_memory_safe_config(self) -> Dict[str, Any]:
        """Get recommended configuration for current memory conditions"""
        status = self.memory_monitor.check_memory_safety()
        
        # Adjust limits based on available memory
        if status.system_available_gb < 3:
            # Very low memory - aggressive limits
            max_concurrent = 5
            cache_items = 25
            max_dimension = 1024
        elif status.system_available_gb < 5:
            # Low memory - conservative limits
            max_concurrent = 10
            cache_items = 50
            max_dimension = 1536
        else:
            # Normal memory - standard limits
            max_concurrent = self.config.max_concurrent_requests
            cache_items = self.config.cache_max_entries
            max_dimension = self.config.max_image_dimension
        
        return {
            'max_concurrent_requests': max_concurrent,
            'cache_max_entries': cache_items,
            'max_image_dimension': max_dimension,
            'compression_enabled': True,
            'memory_threshold_percent': 60,
            'current_memory_status': {
                'process_mb': status.process_mb,
                'system_available_gb': status.system_available_gb
            }
        }
    
    # Screen sharing integration methods
    
    async def start_screen_sharing(self, peer_id: Optional[str] = None) -> Dict[str, Any]:
        """Start screen sharing with memory safety checks"""
        # Check memory before starting
        memory_status = self.memory_monitor.check_memory_safety()
        if not memory_status.is_safe:
            return {
                'success': False,
                'error': 'Insufficient memory for screen sharing',
                'memory_status': memory_status.__dict__
            }
        
        # Get screen sharing manager
        screen_sharing = await self.get_screen_sharing()
        if not screen_sharing:
            return {
                'success': False,
                'error': 'Screen sharing not available'
            }
        
        # Start continuous monitoring if not already active
        if not self.continuous_analyzer or not self.continuous_analyzer.is_monitoring:
            await self.start_continuous_monitoring()
        
        # Start screen sharing
        success = await screen_sharing.start_sharing(peer_id)
        
        if success:
            url = screen_sharing.get_sharing_url()
            return {
                'success': True,
                'sharing_url': url,
                'metrics': screen_sharing.get_metrics()
            }
        else:
            return {
                'success': False,
                'error': 'Failed to start screen sharing'
            }
    
    async def stop_screen_sharing(self) -> Dict[str, Any]:
        """Stop screen sharing"""
        if self.screen_sharing:
            await self.screen_sharing.stop_sharing()
            return {
                'success': True,
                'message': 'Screen sharing stopped'
            }
        return {
            'success': False,
            'error': 'Screen sharing not active'
        }
    
    async def get_screen_sharing_status(self) -> Dict[str, Any]:
        """Get current screen sharing status"""
        if self.screen_sharing:
            return {
                'active': self.screen_sharing.is_sharing,
                'metrics': self.screen_sharing.get_metrics()
            }
        return {
            'active': False,
            'metrics': None
        }
    
    async def add_screen_sharing_peer(self, peer_id: str, offer: Optional[Dict] = None) -> Dict[str, Any]:
        """Add a peer to screen sharing session"""
        if not self.screen_sharing or not self.screen_sharing.is_sharing:
            return {
                'success': False,
                'error': 'Screen sharing not active'
            }
        
        success = await self.screen_sharing.add_peer(peer_id, offer)
        return {
            'success': success,
            'peer_id': peer_id
        }
    
    async def remove_screen_sharing_peer(self, peer_id: str) -> Dict[str, Any]:
        """Remove a peer from screen sharing session"""
        if self.screen_sharing:
            await self.screen_sharing.remove_peer(peer_id)
            return {
                'success': True,
                'peer_id': peer_id
            }
        return {
            'success': False,
            'error': 'Screen sharing not active'
        }
    
    # Video streaming integration methods
    
    async def start_video_streaming(self) -> Dict[str, Any]:
        """Start video streaming capture with memory safety"""
        logger.info("[VISION ANALYZER] start_video_streaming called")
        
        # Check memory before starting
        memory_status = self.memory_monitor.check_memory_safety()
        logger.info(f"[VISION ANALYZER] Memory status: {memory_status.is_safe}")
        
        if not memory_status.is_safe:
            logger.error(f"[VISION ANALYZER] Memory not safe: {memory_status.warnings}")
            return {
                'success': False,
                'error': 'Insufficient memory for video streaming',
                'memory_status': memory_status.__dict__
            }
        
        # Get video streaming manager
        logger.info("[VISION ANALYZER] Getting video streaming manager...")
        video_streaming = await self.get_video_streaming()
        
        if not video_streaming:
            logger.error("[VISION ANALYZER] Video streaming manager not available")
            # Try to initialize it now
            logger.info("[VISION ANALYZER] Attempting to initialize video streaming...")
            try:
                from .video_stream_capture import VideoStreamCapture
                self.video_streaming = VideoStreamCapture(vision_analyzer=self)
                logger.info("[VISION ANALYZER] Video streaming initialized successfully")
                video_streaming = self.video_streaming
            except Exception as e:
                logger.error(f"[VISION ANALYZER] Failed to initialize video streaming: {e}")
                return {
                    'success': False,
                    'error': f'Failed to initialize video streaming: {str(e)}'
                }
        
        if not video_streaming:
            return {
                'success': False,
                'error': 'Video streaming not available after initialization attempt'
            }
        
        # Start video streaming
        logger.info("[VISION ANALYZER] Calling video_streaming.start_streaming()")
        success = await video_streaming.start_streaming()
        logger.info(f"[VISION ANALYZER] start_streaming returned: {success}")
        
        if success:
            # Update continuous monitoring to use video frames
            if self.continuous_analyzer and self.continuous_analyzer.is_monitoring:
                logger.info("Switching continuous analyzer to use video frames")
            
            return {
                'success': True,
                'message': 'Video streaming started - macOS will show screen recording indicator',
                'metrics': video_streaming.get_metrics()
            }
        else:
            # Try to get more specific error information
            error_msg = 'Failed to start video streaming'
            try:
                # Check if there are existing Swift processes blocking
                existing_pids = subprocess.check_output(
                    ["pgrep", "-f", "(persistent_capture|infinite_purple_capture).swift"],
                    text=True
                ).strip()
                if existing_pids:
                    error_msg = 'Failed to start video streaming - existing capture process detected. Please try again.'
            except:
                pass
                
            return {
                'success': False,
                'error': error_msg
            }
    
    async def stop_video_streaming(self) -> Dict[str, Any]:
        """Stop video streaming"""
        if hasattr(self, 'video_streaming') and self.video_streaming:
            await self.video_streaming.stop_streaming()
            return {
                'success': True,
                'message': 'Video streaming stopped'
            }
        return {
            'success': False,
            'error': 'Video streaming not active'
        }
    
    async def get_video_streaming_status(self) -> Dict[str, Any]:
        """Get current video streaming status"""
        if hasattr(self, 'video_streaming') and self.video_streaming:
            metrics = self.video_streaming.get_metrics()
            return {
                'active': self.video_streaming.is_capturing,
                'metrics': metrics,
                'capture_method': metrics.get('capture_method', 'unknown')
            }
        return {
            'active': False,
            'metrics': None
        }
    
    async def analyze_video_stream(self, query: str, duration_seconds: float = 5.0) -> Dict[str, Any]:
        """Analyze video stream for a specific duration"""
        if not hasattr(self, 'video_streaming') or not self.video_streaming or not self.video_streaming.is_capturing:
            # Start video streaming if not active
            start_result = await self.start_video_streaming()
            if not start_result['success']:
                return start_result
        
        # Collect analysis results
        results = []
        start_time = time.time()
        
        def on_frame_analyzed(data):
            results.append({
                'timestamp': time.time() - start_time,
                'frame_number': data['frame_number'],
                'analysis': data['results']
            })
        
        # Register callback
        self.video_streaming.register_callback('frame_analyzed', on_frame_analyzed)
        
        # Wait for duration
        await asyncio.sleep(duration_seconds)
        
        # Remove callback
        self.video_streaming.event_callbacks['frame_analyzed'].discard(on_frame_analyzed)
        
        return {
            'success': True,
            'duration': duration_seconds,
            'frames_analyzed': len(results),
            'results': results,
            'query': query
        }
    
    async def switch_to_video_mode(self) -> Dict[str, Any]:
        """Switch from screenshot mode to video streaming mode"""
        # Start video streaming
        video_result = await self.start_video_streaming()
        
        if video_result['success']:
            # Update config to prefer video
            self.config.prefer_video_over_screenshots = True
            
            # Stop continuous screenshot monitoring if active
            if self.continuous_analyzer and self.continuous_analyzer.is_monitoring:
                await self.continuous_analyzer.stop_monitoring()
                logger.info("Stopped screenshot-based monitoring in favor of video streaming")
            
            return {
                'success': True,
                'mode': 'video_streaming',
                'message': 'Switched to video streaming mode'
            }
        
        return video_result
    
    async def switch_to_screenshot_mode(self) -> Dict[str, Any]:
        """Switch from video streaming to screenshot mode"""
        # Stop video streaming if active
        if hasattr(self, 'video_streaming') and self.video_streaming and self.video_streaming.is_capturing:
            await self.video_streaming.stop_streaming()
        
        # Update config
        self.config.prefer_video_over_screenshots = False
        
        # Restart continuous monitoring if it was active
        if self.config.enable_continuous_monitoring:
            await self.start_continuous_monitoring()
        
        return {
            'success': True,
            'mode': 'screenshot',
            'message': 'Switched to screenshot mode'
        }
    
    # Real-time monitoring capabilities
    
    async def start_real_time_monitoring(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start real-time screen monitoring for JARVIS to see continuously"""
        try:
            # Check memory before starting
            memory_status = self.memory_monitor.check_memory_safety()
            if not memory_status.is_safe:
                return {
                    'success': False,
                    'error': 'Insufficient memory for real-time monitoring',
                    'memory_status': memory_status.__dict__
                }
            
            # Use video streaming for real-time if available
            if self._video_streaming_config and self._video_streaming_config.get('enabled', False):
                result = await self.start_video_streaming()
                if result.get('success', False):
                    logger.info("Real-time monitoring using video streaming")
                    
                    # Set up real-time analysis callback
                    if callback:
                        self.video_streaming.register_callback('frame_analyzed', callback)
                    
                    return {
                        'success': True,
                        'mode': 'video_streaming',
                        'message': 'Real-time monitoring active via video streaming'
                    }
            
            # Fall back to continuous screenshot monitoring
            result = await self.start_continuous_monitoring()
            if result.get('success', False):
                logger.info("Real-time monitoring using screenshot capture")
                
                # Set up analysis callback
                if callback and self.continuous_analyzer:
                    self.continuous_analyzer.set_callback('analysis_complete', callback)
                
                return {
                    'success': True,
                    'mode': 'screenshot',
                    'message': 'Real-time monitoring active via screenshots'
                }
            
            return {
                'success': False,
                'error': 'Failed to start real-time monitoring'
            }
            
        except Exception as e:
            logger.error(f"Real-time monitoring error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_real_time_context(self) -> Dict[str, Any]:
        """Get current screen context in real-time (what JARVIS sees right now)"""
        try:
            # Capture current screen
            screenshot = await self.capture_screen()
            if screenshot is None:
                return {'error': 'Unable to capture screen'}
            
            # Convert to numpy array if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            # Analyze with context-aware prompt
            result_tuple = await self.analyze_screenshot(
                screenshot,
                "What's currently visible on the screen? Describe the active application, any notifications, dialogs, or important content. Be specific about UI elements and text you can see."
            )
            
            # Extract result from tuple
            if isinstance(result_tuple, tuple):
                result = result_tuple[0]
            else:
                result = result_tuple
            
            # Add real-time metadata
            result['timestamp'] = datetime.now().isoformat()
            result['capture_mode'] = 'video_streaming' if hasattr(self, 'video_streaming') and self.video_streaming and self.video_streaming.is_capturing else 'screenshot'
            result['is_real_time'] = True
            
            # Add autonomous behavior insights
            behavior_insights = await self._analyze_for_behaviors(result)
            if behavior_insights:
                result['behavior_insights'] = behavior_insights
            
            return result
            
        except Exception as e:
            logger.error(f"Real-time context error: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def _analyze_for_behaviors(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze screen content for autonomous behavior triggers"""
        insights = {
            'detected_patterns': [],
            'suggested_actions': []
        }
        
        description = analysis_result.get('description', '').lower()
        
        # Check for various patterns
        patterns = {
            'notification': ['notification', 'alert', 'message', 'unread', 'new message'],
            'error': ['error', 'failed', 'exception', 'problem', 'warning', 'issue'],
            'dialog': ['dialog', 'popup', 'confirm', 'save', 'cancel', 'ok'],
            'loading': ['loading', 'processing', 'waiting', 'spinner'],
            'update': ['update available', 'new version', 'install', 'upgrade']
        }
        
        for pattern_type, keywords in patterns.items():
            if any(keyword in description for keyword in keywords):
                insights['detected_patterns'].append(pattern_type)
                
                # Suggest actions based on pattern
                if pattern_type == 'notification':
                    insights['suggested_actions'].append({
                        'type': 'read_notification',
                        'description': 'Read or dismiss the notification'
                    })
                elif pattern_type == 'error':
                    insights['suggested_actions'].append({
                        'type': 'handle_error',
                        'description': 'Investigate or dismiss the error'
                    })
                elif pattern_type == 'dialog':
                    insights['suggested_actions'].append({
                        'type': 'handle_dialog',
                        'description': 'Respond to the dialog box'
                    })
        
        # Check for specific applications
        apps = analysis_result.get('entities', {}).get('applications', [])
        for app in apps:
            app_lower = app.lower()
            if any(msg_app in app_lower for msg_app in ['slack', 'teams', 'discord', 'messages']):
                insights['detected_patterns'].append('messaging_app')
                insights['suggested_actions'].append({
                    'type': 'check_messages',
                    'description': f'Check for new messages in {app}'
                })
        
        return insights if insights['detected_patterns'] else None
    
    async def watch_for_changes(self, duration: float = 60.0, callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Watch screen for changes over a duration and collect insights"""
        changes = []
        start_time = time.time()
        last_description = ""
        
        while time.time() - start_time < duration:
            try:
                # Get current context
                context = await self.get_real_time_context()
                
                if 'error' not in context:
                    current_description = context.get('description', '')
                    
                    # Check if significant change occurred
                    if current_description != last_description:
                        change_event = {
                            'timestamp': context['timestamp'],
                            'description': current_description,
                            'insights': context.get('behavior_insights', {}),
                            'change_detected': True
                        }
                        
                        changes.append(change_event)
                        last_description = current_description
                        
                        # Trigger callback if provided
                        if callback:
                            await callback(change_event)
                
                # Wait before next check
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception as e:
                logger.error(f"Change detection error: {e}")
        
        return changes
    
    async def handle_autonomous_behavior(self, behavior_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle specific autonomous behaviors based on screen content"""
        try:
            if behavior_type == 'check_messages':
                # Analyze for message content
                screenshot = await self.capture_screen()
                if screenshot:
                    result, _ = await self.analyze_screenshot(
                        np.array(screenshot) if isinstance(screenshot, Image.Image) else screenshot,
                        "Extract any message content, sender information, and timestamps visible on screen."
                    )
                    return {
                        'success': True,
                        'behavior': behavior_type,
                        'content': result
                    }
            
            elif behavior_type == 'handle_error':
                # Analyze error details
                screenshot = await self.capture_screen()
                if screenshot:
                    result, _ = await self.analyze_screenshot(
                        np.array(screenshot) if isinstance(screenshot, Image.Image) else screenshot,
                        "Extract the error message, error code, and any suggested solutions visible."
                    )
                    return {
                        'success': True,
                        'behavior': behavior_type,
                        'error_details': result
                    }
            
            elif behavior_type == 'handle_dialog':
                # Analyze dialog options
                screenshot = await self.capture_screen()
                if screenshot:
                    result, _ = await self.analyze_screenshot(
                        np.array(screenshot) if isinstance(screenshot, Image.Image) else screenshot,
                        "What dialog or popup is shown? List all buttons and options available."
                    )
                    return {
                        'success': True,
                        'behavior': behavior_type,
                        'dialog_info': result
                    }
            
            return {
                'success': False,
                'behavior': behavior_type,
                'error': 'Unknown behavior type'
            }
            
        except Exception as e:
            logger.error(f"Autonomous behavior error: {e}")
            return {
                'success': False,
                'behavior': behavior_type,
                'error': str(e)
            }
    
    # JARVIS Integration Methods (from wrapper)
    
    async def analyze_screenshot_clean(self, image_array, prompt, **kwargs):
        """
        Analyze a screenshot and return just the result dictionary (wrapper compatibility)
        This method provides a clean interface that returns only the result dict
        
        Returns:
            dict: Analysis result with 'description', 'entities', 'actions', etc.
        """
        try:
            # Call the main analyze_screenshot method which returns (result, metrics) tuple
            raw_result = await self.analyze_screenshot(image_array, prompt, **kwargs)
            
            # Handle different return formats
            if isinstance(raw_result, tuple) and len(raw_result) >= 2:
                # Extract just the result dict from the tuple
                result = raw_result[0]
                logger.debug(f"Extracted result from tuple: {type(result)}")
                return result
            elif isinstance(raw_result, dict):
                # Already in correct format
                return raw_result
            else:
                logger.warning(f"Unexpected result format: {type(raw_result)}")
                return raw_result
                
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            # Return a safe default
            return {
                'description': f'Analysis failed: {str(e)}',
                'entities': {},
                'actions': [],
                'error': str(e)
            }
    
    async def get_screen_context(self):
        """Get current screen context with real-time awareness (wrapper compatibility)"""
        # Use the real-time context method
        return await self.get_real_time_context()
    
    async def start_jarvis_vision(self, callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Start JARVIS real-time vision - see everything happening on screen"""
        logger.info("🤖 Starting JARVIS real-time vision...")
        
        # Define internal callback to handle vision events
        async def vision_callback(event):
            logger.debug(f"Vision event: {event.get('description', 'Unknown')[:100]}...")
            
            # Trigger user callbacks
            for cb in self._realtime_callbacks:
                try:
                    await cb(event)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
        
        # Add user callback if provided
        if callback:
            self._realtime_callbacks.append(callback)
        
        # Start real-time monitoring
        result = await self.start_real_time_monitoring(vision_callback)
        
        if result['success']:
            logger.info(f"✅ JARVIS vision active in {result['mode']} mode")
        else:
            logger.error(f"❌ Failed to start vision: {result.get('error', 'Unknown error')}")
        
        return result
    
    async def stop_jarvis_vision(self) -> Dict[str, Any]:
        """Stop JARVIS real-time vision"""
        logger.info("🛑 Stopping JARVIS vision...")
        
        # Stop video streaming if active
        if hasattr(self, 'video_streaming') and self.video_streaming and self.video_streaming.is_capturing:
            await self.stop_video_streaming()
        
        # Stop continuous monitoring
        if hasattr(self, 'continuous_analyzer') and self.continuous_analyzer and self.continuous_analyzer.is_monitoring:
            await self.stop_continuous_monitoring()
        
        # Clear callbacks
        self._realtime_callbacks.clear()
        
        return {
            'success': True,
            'message': 'JARVIS vision stopped'
        }
    
    async def see_and_respond(self, user_command: str) -> Dict[str, Any]:
        """
        JARVIS sees the screen and responds to user commands with visual context
        This is the main method for vision-aware command handling
        """
        try:
            # Get current screen context
            context = await self.get_real_time_context()
            
            if 'error' in context:
                return {
                    'success': False,
                    'error': context['error'],
                    'response': "I'm having trouble seeing the screen right now."
                }
            
            # Analyze command in context of what's visible
            screenshot = await self.capture_screen()
            if screenshot:
                # Convert to numpy array if needed
                if isinstance(screenshot, Image.Image):
                    screenshot = np.array(screenshot)
                
                # Analyze with command context
                result = await self.analyze_screenshot_clean(
                    screenshot,
                    f"The user said: '{user_command}'. Based on what you see on screen, how should I help them? Be specific about what actions to take."
                )
                
                # Check for autonomous behaviors
                if context.get('behavior_insights'):
                    result['suggested_behaviors'] = context['behavior_insights']['suggested_actions']
                
                return {
                    'success': True,
                    'visual_context': context,
                    'command_analysis': result,
                    'response': result.get('description', 'I can see the screen and am ready to help.')
                }
            
            return {
                'success': False,
                'error': 'Unable to capture screen',
                'response': "I need to see the screen to help with that."
            }
            
        except Exception as e:
            logger.error(f"See and respond error: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I encountered an error while trying to see the screen."
            }
    
    async def monitor_for_notifications(self, duration: float = 300.0) -> List[Dict[str, Any]]:
        """Monitor screen for notifications and important events"""
        notifications = []
        
        async def notification_callback(event):
            insights = event.get('insights', {})
            if 'notification' in insights.get('detected_patterns', []):
                notifications.append(event)
                logger.info(f"📬 Notification detected: {event['description'][:100]}...")
        
        # Start monitoring with callback
        await self.start_jarvis_vision(notification_callback)
        
        # Watch for changes
        changes = await self.watch_for_changes(duration, notification_callback)
        
        # Stop monitoring
        await self.stop_jarvis_vision()
        
        return notifications
    
    def add_realtime_callback(self, callback: Callable):
        """Add a callback for real-time vision events"""
        if callback not in self._realtime_callbacks:
            self._realtime_callbacks.append(callback)
    
    def remove_realtime_callback(self, callback: Callable):
        """Remove a real-time vision callback"""
        if callback in self._realtime_callbacks:
            self._realtime_callbacks.remove(callback)
    
    def __del__(self):
        """Cleanup on deletion"""
        self.executor.shutdown(wait=False)
        # Schedule async cleanup
        try:
            asyncio.create_task(self.cleanup_all_components())
        except RuntimeError:
            # Event loop might be closed
            pass
    
    # Vision Intelligence Methods
    
    async def get_vision_intelligence_insights(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        """Get insights from Vision Intelligence System"""
        try:
            vision_intelligence = await self.get_vision_intelligence()
            if not vision_intelligence:
                return {"enabled": False, "message": "Vision Intelligence not available"}
            
            insights = vision_intelligence.get_system_insights()
            
            # Add specific app insights if requested
            if app_id and 'app_insights' in insights and app_id in insights['app_insights']:
                insights['requested_app'] = insights['app_insights'][app_id]
            
            return insights
        except Exception as e:
            logger.error(f"Failed to get Vision Intelligence insights: {e}")
            return {"error": str(e)}
    
    async def train_vision_intelligence(self, 
                                      screenshot: Union[np.ndarray, Image.Image],
                                      app_id: str,
                                      state_id: str,
                                      state_type: Optional[str] = None) -> Dict[str, Any]:
        """Train Vision Intelligence on a labeled example"""
        try:
            vision_intelligence = await self.get_vision_intelligence()
            if not vision_intelligence:
                return {"success": False, "message": "Vision Intelligence not available"}
            
            # Convert image to numpy array if needed
            if isinstance(screenshot, Image.Image):
                screenshot = np.array(screenshot)
            
            result = await vision_intelligence.train_on_labeled_state(
                screenshot=screenshot,
                app_id=app_id,
                state_id=state_id,
                state_type=state_type
            )
            
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Failed to train Vision Intelligence: {e}")
            return {"success": False, "error": str(e)}
    
    def save_vision_intelligence_states(self):
        """Save all learned states from Vision Intelligence"""
        try:
            if (self._vision_intelligence_config.get('enabled') and 
                self._vision_intelligence_config.get('persistence') and
                self.vision_intelligence):
                self.vision_intelligence.save_learned_states()
                logger.info("Saved Vision Intelligence learned states")
                return True
        except Exception as e:
            logger.error(f"Failed to save Vision Intelligence states: {e}")
        return False
    
    async def analyze_with_state_tracking(self, 
                                        image: Any,
                                        prompt: str,
                                        app_id: Optional[str] = None,
                                        **kwargs) -> Tuple[Dict[str, Any], AnalysisMetrics]:
        """Analyze screenshot with automatic state tracking"""
        # Perform normal analysis
        result, metrics = await self.analyze_screenshot(image, prompt, **kwargs)
        
        # If app_id not provided, try to extract it
        if not app_id and '_vision_intelligence' in result:
            app_id = result['_vision_intelligence'].get('app_id')
        
        # Get state insights for this app if available
        if app_id:
            insights = await self.get_vision_intelligence_insights(app_id)
            if 'requested_app' in insights:
                result['state_insights'] = insights['requested_app']
        
        return result, metrics
    
    # VSMS Core Methods
    
    async def get_vsms_insights(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        """Get insights from VSMS Core"""
        try:
            vsms = await self.get_vsms_core()
            if not vsms:
                return {"enabled": False, "message": "VSMS Core not available"}
            
            # Get general insights
            insights = vsms.get_insights()
            
            # Add app-specific insights if requested
            if app_id:
                app_insights = vsms.get_application_insights(app_id)
                insights['app_specific'] = app_insights
                
                # Get productivity insights from state intelligence
                if hasattr(vsms, 'state_intelligence'):
                    productivity = vsms.state_intelligence.get_productivity_insights()
                    insights['productivity'] = productivity
            
            return insights
        except Exception as e:
            logger.error(f"Failed to get VSMS insights: {e}")
            return {"error": str(e)}
    
    async def get_state_recommendations(self, app_id: str, current_state: Optional[str] = None) -> Dict[str, Any]:
        """Get intelligent state recommendations from VSMS"""
        try:
            vsms = await self.get_vsms_core()
            if not vsms or not hasattr(vsms, 'state_intelligence'):
                return {"enabled": False, "message": "VSMS state intelligence not available"}
            
            # If current state not provided, get from VSMS
            if not current_state:
                current_state = vsms.current_states.get(app_id)
            
            if not current_state:
                return {"error": "No current state detected for application"}
            
            # Get recommendations
            recommendations = vsms.state_intelligence.get_state_recommendations(current_state)
            
            # Add personalization score
            recommendations['personalization_score'] = vsms.state_intelligence.user_preference.personalization_score
            
            return recommendations
        except Exception as e:
            logger.error(f"Failed to get state recommendations: {e}")
            return {"error": str(e)}
    
    async def get_temporal_context(self, app_id: Optional[str] = None) -> Dict[str, Any]:
        """Get temporal context for an application"""
        if not self._temporal_context_config['enabled'] or not self.vsms_core:
            return {'enabled': False, 'reason': 'Temporal Context not enabled or VSMS Core not available'}
        
        try:
            # Get temporal context from VSMS Core
            context = await self.vsms_core.get_temporal_context(app_id)
            
            return {
                'enabled': True,
                **context
            }
        except Exception as e:
            logger.error(f"Failed to get temporal context: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }
    
    async def get_temporal_predictions(self, lookahead_seconds: int = 60) -> List[Dict[str, Any]]:
        """Get predictions for next likely events"""
        if not self._temporal_context_config['enabled'] or not self._temporal_context_config['predictions']:
            return []
        
        try:
            if hasattr(self.vsms_core, 'temporal_engine'):
                return await self.vsms_core.temporal_engine.predict_next_events(lookahead_seconds)
            return []
        except Exception as e:
            logger.error(f"Failed to get temporal predictions: {e}")
            return []
    
    async def get_scene_graph_insights(self) -> Dict[str, Any]:
        """Get insights from the Semantic Scene Graph"""
        if not self._scene_graph_config['enabled'] or not self.vsms_core:
            return {'enabled': False, 'reason': 'Scene Graph not enabled or VSMS Core not available'}
        
        try:
            # Get current scene graph from VSMS Core
            if hasattr(self.vsms_core, 'scene_graph') and self.vsms_core.scene_graph:
                scene_graph = self.vsms_core.scene_graph
                
                # Get current graph metrics
                if scene_graph.current_graph:
                    analysis = scene_graph.intelligence.analyze_graph(scene_graph.current_graph)
                    
                    return {
                        'enabled': True,
                        'graph_metrics': analysis.get('graph_metrics', {}),
                        'key_nodes': analysis.get('key_nodes', [])[:5],
                        'information_flow': len(analysis.get('information_flow', [])),
                        'interaction_patterns': analysis.get('interaction_patterns', {}),
                        'anomalies': analysis.get('anomalies', []),
                        'memory_usage_mb': sum(scene_graph.memory_usage.values()) / 1024 / 1024
                    }
                else:
                    return {
                        'enabled': True,
                        'status': 'No scene graph built yet',
                        'memory_usage_mb': 0
                    }
            else:
                return {
                    'enabled': True,
                    'status': 'Scene Graph not initialized',
                    'memory_usage_mb': 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get Scene Graph insights: {e}")
            return {
                'enabled': True,
                'error': str(e)
            }
    
    def save_vsms_states(self):
        """Save all VSMS learned states"""
        try:
            saved_components = []
            
            # Save Vision Intelligence states
            if self.vision_intelligence:
                self.vision_intelligence.save_learned_states()
                saved_components.append("vision_intelligence")
            
            # Save VSMS Core states
            if self.vsms_core:
                self.vsms_core.state_history.save_to_disk()
                self.vsms_core._save_state_definitions()
                if hasattr(self.vsms_core, 'state_intelligence'):
                    self.vsms_core.state_intelligence._save_intelligence_data()
                saved_components.append("vsms_core")
            
            logger.info(f"Saved states for: {', '.join(saved_components)}")
            return True
        except Exception as e:
            logger.error(f"Failed to save VSMS states: {e}")
            return False
    
    async def create_state_definition(self, 
                                    app_id: str,
                                    state_id: str,
                                    state_name: str,
                                    category: str = "CUSTOM",
                                    visual_signatures: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Create a new state definition in VSMS"""
        try:
            vsms = await self.get_vsms_core()
            if not vsms:
                return {"success": False, "message": "VSMS Core not available"}
            
            # Map category string to enum
            from .intelligence.vsms_core import StateCategory
            category_enum = getattr(StateCategory, category.upper(), StateCategory.CUSTOM)
            
            # Create state definition
            state = vsms.create_state_definition(
                app_id=app_id,
                state_id=state_id,
                category=category_enum,
                name=state_name,
                visual_signatures=visual_signatures or []
            )
            
            return {
                "success": True,
                "state": {
                    "state_id": state.state_id,
                    "name": state.name,
                    "category": state.category.name,
                    "detection_count": state.detection_count
                }
            }
        except Exception as e:
            logger.error(f"Failed to create state definition: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_current_activities(self) -> List[Dict[str, Any]]:
        """Get current active tasks/activities from Activity Recognition Engine
        
        Returns:
            List of current activities with detailed insights
        """
        if not self._activity_recognition_config['enabled']:
            return []
        
        try:
            vsms = await self.get_vsms_core()
            if not vsms:
                return []
            
            # Get current activities from VSMS
            activities = vsms.get_current_activities()
            
            # Transform for API response
            return [{
                'task_id': activity.get('task_id'),
                'task_name': activity.get('task_name'),
                'primary_activity': activity.get('primary_activity'),
                'status': activity.get('status'),
                'completion_percentage': activity.get('completion_percentage', 0),
                'elapsed_time': activity.get('elapsed_time'),
                'estimated_remaining': activity.get('estimated_remaining'),
                'is_stuck': activity.get('is_stuck', False),
                'applications': activity.get('applications', []),
                'key_insights': activity.get('key_insights', [])
            } for activity in activities]
            
        except Exception as e:
            logger.error(f"Failed to get current activities: {e}")
            return []
    
    async def get_activity_summary(self) -> Dict[str, Any]:
        """Get summary of all activities from Activity Recognition Engine
        
        Returns:
            Summary with statistics and insights about all activities
        """
        if not self._activity_recognition_config['enabled']:
            return {
                'enabled': False,
                'message': 'Activity Recognition Engine not enabled'
            }
        
        try:
            vsms = await self.get_vsms_core()
            if not vsms:
                return {
                    'enabled': False,
                    'message': 'VSMS Core not available'
                }
            
            # Get activity summary from VSMS
            summary = vsms.get_activity_summary()
            
            return {
                'enabled': True,
                'total_tasks': summary.get('total_tasks', 0),
                'active_tasks': summary.get('active_tasks', 0),
                'completed_tasks': summary.get('completed_tasks', 0),
                'stuck_tasks': summary.get('stuck_tasks', 0),
                'primary_activities': summary.get('primary_activities', {}),
                'productivity_score': summary.get('productivity_score', 0.0),
                'time_distribution': summary.get('time_distribution', {}),
                'workflow_patterns': summary.get('workflow_patterns', []),
                'recommendations': summary.get('recommendations', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get activity summary: {e}")
            return {
                'enabled': False,
                'error': str(e)
            }
    
    async def get_activity_insights(self, task_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific activity/task
        
        Args:
            task_id: The ID of the task to get insights for
            
        Returns:
            Detailed insights about the task including progress, blockers, and suggestions
        """
        if not self._activity_recognition_config['enabled']:
            return {
                'enabled': False,
                'message': 'Activity Recognition Engine not enabled'
            }
        
        try:
            vsms = await self.get_vsms_core()
            if not vsms or not hasattr(vsms, 'activity_engine'):
                return {
                    'enabled': False,
                    'message': 'Activity Recognition not available'
                }
            
            # Get task insights from activity engine
            insights = vsms.activity_engine.get_task_insights(task_id)
            
            if not insights:
                return {
                    'enabled': True,
                    'found': False,
                    'task_id': task_id
                }
            
            return {
                'enabled': True,
                'found': True,
                **insights
            }
            
        except Exception as e:
            logger.error(f"Failed to get activity insights for {task_id}: {e}")
            return {
                'enabled': False,
                'error': str(e)
            }
    
    async def get_inferred_goals(self) -> Dict[str, Any]:
        """Get currently inferred user goals from Goal Inference System
        
        Returns:
            Dictionary with goals organized by level (high, intermediate, immediate)
        """
        try:
            vsms = await self.get_vsms_core()
            if not vsms or not hasattr(vsms, 'goal_inference'):
                return {
                    'enabled': False,
                    'message': 'Goal Inference System not available'
                }
            
            # Get active goals summary
            summary = vsms.goal_inference.get_active_goals_summary()
            
            return {
                'enabled': True,
                'total_active': summary.get('total_active', 0),
                'by_level': summary.get('by_level', {}),
                'high_confidence_goals': summary.get('high_confidence', []),
                'recently_updated': summary.get('recently_updated', []),
                'near_completion': summary.get('near_completion', [])
            }
            
        except Exception as e:
            logger.error(f"Failed to get inferred goals: {e}")
            return {
                'enabled': False,
                'error': str(e)
            }
    
    async def get_goal_insights(self, goal_id: str) -> Dict[str, Any]:
        """Get detailed insights for a specific goal
        
        Args:
            goal_id: The ID of the goal to get insights for
            
        Returns:
            Detailed insights about the goal including progress and relationships
        """
        try:
            vsms = await self.get_vsms_core()
            if not vsms or not hasattr(vsms, 'goal_inference'):
                return {
                    'enabled': False,
                    'message': 'Goal Inference System not available'
                }
            
            # Get goal insights
            insights = vsms.goal_inference.get_goal_insights(goal_id)
            
            if not insights:
                return {
                    'enabled': True,
                    'found': False,
                    'goal_id': goal_id
                }
            
            return {
                'enabled': True,
                'found': True,
                **insights
            }
            
        except Exception as e:
            logger.error(f"Failed to get goal insights for {goal_id}: {e}")
            return {
                'enabled': False,
                'error': str(e)
            }
    
    async def track_goal_progress(self, goal_id: str, progress_delta: float) -> Dict[str, Any]:
        """Update goal progress
        
        Args:
            goal_id: The ID of the goal to update
            progress_delta: Progress increment (0.0 to 1.0)
            
        Returns:
            Success status and updated goal info
        """
        try:
            vsms = await self.get_vsms_core()
            if not vsms or not hasattr(vsms, 'goal_inference'):
                return {
                    'success': False,
                    'message': 'Goal Inference System not available'
                }
            
            # Update progress
            vsms.goal_inference.track_goal_progress(goal_id, progress_delta)
            
            # Get updated insights
            insights = vsms.goal_inference.get_goal_insights(goal_id)
            
            return {
                'success': True,
                'goal_id': goal_id,
                'current_progress': insights.get('progress', 0.0) if insights else 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to track goal progress: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def process_intervention_signal(self, signal_type: str, value: float, 
                                        confidence: float = 0.8, metadata: Dict[str, Any] = None):
        """Process user state signal for intervention decision"""
        intervention_engine = await self.get_intervention_engine()
        if not intervention_engine:
            return {"error": "Intervention engine not available"}
        
        from .intelligence.intervention_decision_engine import UserStateSignal
        
        signal = UserStateSignal(
            signal_type=signal_type,
            value=value,
            confidence=confidence,
            timestamp=datetime.now(),
            source='manual_input',
            metadata=metadata or {}
        )
        
        await intervention_engine.process_user_signal(signal)
        
        return {
            "success": True,
            "current_state": intervention_engine.current_user_state.value if intervention_engine.current_user_state else "unknown",
            "state_confidence": intervention_engine.state_confidence
        }
    
    async def check_intervention_opportunity(self, situation_data: Dict[str, Any] = None):
        """Check if intervention is recommended based on current state"""
        intervention_engine = await self.get_intervention_engine()
        if not intervention_engine:
            return {"error": "Intervention engine not available"}
        
        # Assess situation if data provided
        if situation_data:
            await intervention_engine.assess_situation(situation_data)
        
        # Get intervention decision
        opportunity = await intervention_engine.decide_intervention()
        
        if opportunity:
            return {
                "intervention_recommended": True,
                "type": opportunity.intervention_type.value,
                "timing": opportunity.timing_strategy.value,
                "confidence": opportunity.confidence_score,
                "urgency": opportunity.urgency_score,
                "rationale": opportunity.rationale,
                "content": opportunity.content
            }
        else:
            return {
                "intervention_recommended": False,
                "current_state": intervention_engine.current_user_state.value if intervention_engine.current_user_state else "unknown",
                "reason": "No intervention needed at this time"
            }
    
    async def get_intervention_stats(self) -> Dict[str, Any]:
        """Get intervention engine statistics"""
        intervention_engine = await self.get_intervention_engine()
        if not intervention_engine:
            return {"error": "Intervention engine not available"}
        
        return intervention_engine.get_statistics()
    
    async def detect_anomalies_in_screenshot(self, screenshot: np.ndarray, 
                                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect anomalies in a screenshot"""
        anomaly_detector = await self.get_anomaly_detector()
        if not anomaly_detector:
            return {"error": "Anomaly detector not available"}
        
        # Build dynamic analysis prompt based on context
        analysis_prompt = "Analyze this screenshot for any unusual or anomalous patterns"
        if context:
            # Add context-specific analysis requests
            if 'expected_state' in context:
                analysis_prompt += f". Expected state: {context['expected_state']}"
            if 'previous_action' in context:
                analysis_prompt += f". Previous action: {context['previous_action']}"
            if 'focus_areas' in context:
                areas = ', '.join(context['focus_areas'])
                analysis_prompt += f". Pay special attention to: {areas}"
        
        # Analyze the screenshot with custom config for anomaly detection
        anomaly_config = {
            'max_tokens': 1000,  # More tokens for detailed analysis
            'enable_entity_extraction': True,
            'enable_spatial_analysis': True
        }
        
        result, metrics = await self.analyze_screenshot(
            screenshot, 
            analysis_prompt,
            priority='high',  # High priority for anomaly detection
            custom_config=anomaly_config
        )
        
        from .intelligence.anomaly_detection_framework import Observation
        
        # Extract detailed information from result
        analysis_text = ''
        detected_issues = []
        confidence_score = 0.5
        
        if isinstance(result, dict):
            analysis_text = result.get('analysis', result.get('description', str(result)))
            detected_issues.extend(result.get('warnings', []))
            detected_issues.extend(result.get('errors', []))
            confidence_score = result.get('confidence', 0.5)
            
            # Check for anomaly indicators in various fields
            if 'elements' in result:
                for element in result.get('elements', []):
                    if element.get('is_error') or element.get('is_warning'):
                        detected_issues.append(f"Anomalous element: {element.get('text', 'Unknown')}")
        else:
            analysis_text = str(result)
        
        # Build comprehensive observation data
        observation_data = {
            'analysis': analysis_text,
            'context': context or {},
            'screenshot_shape': screenshot.shape,
            'detected_issues': detected_issues,
            'metrics': {
                'processing_time': metrics.total_time if hasattr(metrics, 'total_time') else 0,
                'api_call_time': metrics.api_call_time if hasattr(metrics, 'api_call_time') else 0,
                'cache_hit': metrics.cache_hit if hasattr(metrics, 'cache_hit') else False
            }
        }
        
        # Add result details if available
        if isinstance(result, dict):
            observation_data.update({
                'entities': result.get('entities', {}),
                'suggestions': result.get('suggestions', []),
                'spatial_analysis': result.get('spatial_analysis'),
                'elements_count': len(result.get('elements', [])),
                'app_id': result.get('app_id', 'unknown')
            })
        
        # Determine error and warning presence dynamically
        combined_text = f"{analysis_text} {' '.join(detected_issues)}".lower()
        has_errors = any(term in combined_text for term in ['error', 'fail', 'crash', 'exception', 'critical'])
        has_warnings = any(term in combined_text for term in ['warning', 'warn', 'alert', 'issue', 'problem'])
        
        # Add contextual metadata
        metadata = {
            'has_errors': has_errors,
            'has_warnings': has_warnings,
            'confidence': confidence_score,
            'issue_count': len(detected_issues),
            'context_provided': bool(context),
            'screenshot_analyzed': True
        }
        
        # Add context-specific metadata
        if context:
            metadata.update({
                'has_expected_state': 'expected_state' in context,
                'has_previous_action': 'previous_action' in context,
                'focus_area_count': len(context.get('focus_areas', []))
            })
        
        observation = Observation(
            timestamp=datetime.now(),
            observation_type='manual_screenshot',
            data=observation_data,
            source='detect_anomalies_in_screenshot',
            metadata=metadata
        )
        
        # Detect anomaly
        anomaly = await anomaly_detector.detect_anomaly(observation)
        
        if anomaly:
            return {
                "anomaly_detected": True,
                "type": anomaly.anomaly_type.value,
                "severity": anomaly.severity.value,
                "confidence": anomaly.confidence,
                "description": anomaly.description,
                "detection_method": anomaly.detection_method,
                "requires_intervention": anomaly.severity.value in ['HIGH', 'CRITICAL']
            }
        else:
            # Return detailed analysis even when no anomaly detected
            return {
                "anomaly_detected": False,
                "message": "No anomalies detected in screenshot",
                "analysis_summary": {
                    "issues_found": len(detected_issues),
                    "has_errors": has_errors,
                    "has_warnings": has_warnings,
                    "confidence": confidence_score,
                    "elements_analyzed": observation_data.get('elements_count', 0),
                    "processing_time_ms": observation_data['metrics']['processing_time'] * 1000 if observation_data['metrics']['processing_time'] else 0
                },
                "context_used": bool(context),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_anomaly_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent anomaly detection history"""
        anomaly_detector = await self.get_anomaly_detector()
        if not anomaly_detector:
            return []
        
        history = anomaly_detector.get_anomaly_history(limit)
        return [
            {
                "anomaly_id": a.anomaly_id,
                "type": a.anomaly_type.value,
                "severity": a.severity.value,
                "timestamp": a.timestamp.isoformat(),
                "confidence": a.confidence,
                "description": a.description
            }
            for a in history
        ]
    
    async def test_intervention_system(self, scenario: str = "frustrated_user") -> Dict[str, Any]:
        """Test the intervention system with predefined scenarios"""
        intervention_engine = await self.get_intervention_engine()
        if not intervention_engine:
            return {"error": "Intervention engine not available"}
        
        from .intelligence.intervention_decision_engine import UserStateSignal
        
        scenarios = {
            "frustrated_user": [
                UserStateSignal("error_rate", 0.7, 0.9, datetime.now(), "test"),
                UserStateSignal("repeated_actions", 0.8, 0.85, datetime.now(), "test"),
                UserStateSignal("mouse_movement", 0.9, 0.8, datetime.now(), "test", 
                              {"velocity": 2.5, "acceleration": 1.8})
            ],
            "productive_user": [
                UserStateSignal("typing_pattern", 0.8, 0.9, datetime.now(), "test",
                              {"wpm": 65, "accuracy": 0.95}),
                UserStateSignal("task_completion", 0.9, 0.9, datetime.now(), "test"),
                UserStateSignal("focus_duration", 0.85, 0.9, datetime.now(), "test")
            ],
            "struggling_user": [
                UserStateSignal("help_searches", 0.7, 0.9, datetime.now(), "test"),
                UserStateSignal("documentation_views", 0.8, 0.8, datetime.now(), "test"),
                UserStateSignal("pause_duration", 0.7, 0.85, datetime.now(), "test")
            ]
        }
        
        if scenario not in scenarios:
            return {"error": f"Unknown scenario. Available: {list(scenarios.keys())}"}
        
        # Process signals
        for signal in scenarios[scenario]:
            await intervention_engine.process_user_signal(signal)
        
        # Force state update
        await intervention_engine._update_user_state()
        
        # Create situation based on scenario
        situation_data = {
            "frustrated_user": {
                'has_error': True,
                'error_type': 'repeated_failures',
                'failure_count': 5,
                'context_type': 'debugging'
            },
            "productive_user": {
                'has_error': False,
                'context_type': 'coding',
                'task_completion_rate': 0.9
            },
            "struggling_user": {
                'has_error': False,
                'context_type': 'learning',
                'documentation_available': True
            }
        }
        
        await intervention_engine.assess_situation(situation_data[scenario])
        
        # Get intervention decision
        opportunity = await intervention_engine.decide_intervention()
        
        result = {
            "scenario": scenario,
            "detected_state": intervention_engine.current_user_state.value,
            "state_confidence": intervention_engine.state_confidence,
            "intervention_recommended": opportunity is not None
        }
        
        if opportunity:
            result.update({
                "intervention_type": opportunity.intervention_type.value,
                "timing_strategy": opportunity.timing_strategy.value,
                "confidence": opportunity.confidence_score,
                "rationale": opportunity.rationale
            })
        
        return result
    
    async def capture_problem_solution(self, problem_description: str, 
                                     solution_steps: List[Dict[str, Any]],
                                     success: bool = True,
                                     execution_time: Optional[float] = None) -> Dict[str, Any]:
        """Manually capture a problem and its solution"""
        solution_memory = await self.get_solution_memory_bank()
        if not solution_memory:
            return {"error": "Solution Memory Bank not available"}
        
        from .intelligence.solution_memory_bank import ProblemSignature, ProblemType
        
        # Create problem signature
        problem = ProblemSignature(
            visual_pattern={'manual_entry': True},
            error_messages=[problem_description] if problem_description else [],
            symptoms=['user_reported'],
            problem_type=ProblemType.UNKNOWN
        )
        
        # Capture solution
        solution = await solution_memory.capture_solution(
            problem=problem,
            solution_steps=solution_steps,
            success=success,
            execution_time=execution_time or 60.0,
            context={'source': 'manual_capture'}
        )
        
        return {
            "success": True,
            "solution_id": solution.solution_details.solution_id,
            "status": solution.status.value
        }
    
    async def find_solutions_for_screenshot(self, screenshot: np.ndarray,
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Find solutions for problems in a screenshot"""
        solution_memory = await self.get_solution_memory_bank()
        if not solution_memory:
            return {"error": "Solution Memory Bank not available"}
        
        # First analyze the screenshot
        result, _ = await self.analyze_screenshot(
            screenshot,
            "Identify any errors, problems, or issues in this screenshot"
        )
        
        # Check if problems were found
        if 'solution_recommendations' in result:
            return {
                "found_solutions": True,
                "recommendations": result['solution_recommendations'],
                "analysis": result.get('description', '')
            }
        else:
            return {
                "found_solutions": False,
                "message": "No problems detected requiring solutions",
                "analysis": result.get('description', '')
            }
    
    async def apply_recommended_solution(self, solution_id: str,
                                       execute_callback=None) -> Dict[str, Any]:
        """Apply a recommended solution"""
        solution_memory = await self.get_solution_memory_bank()
        if not solution_memory:
            return {"error": "Solution Memory Bank not available"}
        
        # Get current context
        context = {
            'timestamp': datetime.now().isoformat(),
            'source': 'vision_analyzer'
        }
        
        # Apply solution
        result = await solution_memory.apply_solution(
            solution_id,
            context,
            execute_callback=execute_callback
        )
        
        return result
    
    async def refine_solution_with_feedback(self, solution_id: str,
                                          feedback: str,
                                          rating: Optional[float] = None,
                                          refinements: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Provide feedback to improve a solution"""
        solution_memory = await self.get_solution_memory_bank()
        if not solution_memory:
            return {"error": "Solution Memory Bank not available"}
        
        await solution_memory.refine_solution(
            solution_id,
            refinements or {},
            user_feedback=feedback,
            rating=rating
        )
        
        return {
            "success": True,
            "message": "Solution refined with feedback"
        }
    
    async def get_solution_stats(self) -> Dict[str, Any]:
        """Get Solution Memory Bank statistics"""
        solution_memory = await self.get_solution_memory_bank()
        if not solution_memory:
            return {"error": "Solution Memory Bank not available"}
        
        stats = solution_memory.get_statistics()
        memory = solution_memory.get_memory_usage()
        
        return {
            "statistics": stats,
            "memory_usage": {
                "database_mb": memory['solution_database'] / 1024 / 1024,
                "indices_mb": memory['index_structures'] / 1024 / 1024,
                "total_mb": memory['total'] / 1024 / 1024
            }
        }
    
    async def optimize_regions_with_quadtree(self, screenshot: np.ndarray,
                                           importance_threshold: float = 0.6) -> Dict[str, Any]:
        """Use Quadtree to find and optimize important regions in screenshot"""
        quadtree = await self.get_quadtree_spatial()
        if not quadtree:
            return {"error": "Quadtree spatial intelligence not available"}
        
        # Build quadtree
        tree_id = f"optimize_{hashlib.md5(screenshot.tobytes()).hexdigest()[:8]}"
        await quadtree.build_quadtree(screenshot, tree_id)
        
        # Query regions
        query_result = await quadtree.query_regions(
            tree_id,
            importance_threshold=importance_threshold,
            max_regions=20
        )
        
        # Get recommendations
        recommendations = quadtree.get_processing_recommendations(
            tree_id,
            available_api_calls=10
        )
        
        return {
            "regions_found": len(query_result.nodes),
            "coverage_ratio": query_result.coverage_ratio,
            "from_cache": query_result.from_cache,
            "regions": [
                {
                    "x": node.x,
                    "y": node.y,
                    "width": node.width,
                    "height": node.height,
                    "importance": node.importance,
                    "complexity": node.complexity,
                    "level": node.level
                }
                for node in query_result.nodes[:10]
            ],
            "recommendations": recommendations,
            "stats": quadtree.get_statistics()
        }
    
    async def analyze_with_spatial_focus(self, screenshot: np.ndarray, prompt: str,
                                       focus_regions: List[Dict[str, int]] = None) -> Dict[str, Any]:
        """Analyze screenshot with spatial focus on specific regions"""
        quadtree = await self.get_quadtree_spatial()
        if not quadtree:
            # Fallback to standard analysis
            result, _ = await self.analyze_screenshot(screenshot, prompt)
            return result
        
        # Convert focus regions to tuples
        focus_tuples = None
        if focus_regions:
            focus_tuples = [
                (r['x'], r['y'], r['x'] + r['width'], r['y'] + r['height'])
                for r in focus_regions
            ]
        
        # Build quadtree with focus regions
        tree_id = f"focus_{hashlib.md5(screenshot.tobytes()).hexdigest()[:8]}"
        await quadtree.build_quadtree(screenshot, tree_id, focus_regions=focus_tuples)
        
        # Query with higher threshold for focused analysis
        query_result = await quadtree.query_regions(
            tree_id,
            importance_threshold=0.5,
            max_regions=15
        )
        
        # Create enhanced prompt with spatial information
        spatial_prompt = prompt
        if query_result.nodes:
            spatial_prompt += f"\n\nPay special attention to {len(query_result.nodes)} important regions identified."
            if focus_regions:
                spatial_prompt += " User has indicated specific areas of interest."
        
        # Analyze with spatial context
        result, metrics = await self.analyze_screenshot(screenshot, spatial_prompt)
        
        # Add detailed spatial information
        result['spatial_focus'] = {
            'method': 'quadtree_focused',
            'focus_regions': focus_regions,
            'detected_regions': len(query_result.nodes),
            'coverage': query_result.coverage_ratio
        }
        
        return result
    
    async def get_quadtree_stats(self) -> Dict[str, Any]:
        """Get Quadtree spatial intelligence statistics"""
        quadtree = await self.get_quadtree_spatial()
        if not quadtree:
            return {"error": "Quadtree spatial intelligence not available"}
        
        return quadtree.get_statistics()
    
    async def search_solutions_by_error(self, error_message: str) -> List[Dict[str, Any]]:
        """Search for solutions by error message"""
        solution_memory = await self.get_solution_memory_bank()
        if not solution_memory:
            return []
        
        from .intelligence.solution_memory_bank import ProblemSignature, ProblemType
        
        # Create problem signature from error
        problem = ProblemSignature(
            error_messages=[error_message],
            symptoms=['error_search'],
            problem_type=ProblemType.ERROR
        )
        
        # Find similar solutions
        similar = await solution_memory.find_similar_solutions(problem, threshold=0.5)
        
        results = []
        for solution_id, similarity in similar[:5]:
            solution = solution_memory.solutions.get(solution_id)
            if solution:
                results.append({
                    "solution_id": solution_id,
                    "similarity": similarity,
                    "success_rate": solution.solution_details.success_rate,
                    "usage_count": solution.learning_metadata.usage_count,
                    "status": solution.status.value
                })
        
        return results