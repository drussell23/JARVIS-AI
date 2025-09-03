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
import re
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
    
    # Memory safety settings
    process_memory_limit_mb: int = field(default_factory=lambda: int(os.getenv('VISION_PROCESS_LIMIT_MB', '2048')))  # 2GB
    memory_warning_threshold_mb: int = field(default_factory=lambda: int(os.getenv('VISION_MEMORY_WARNING_MB', '1536')))  # 1.5GB
    min_system_available_gb: float = field(default_factory=lambda: float(os.getenv('VISION_MIN_SYSTEM_RAM_GB', '2.0')))  # 2GB
    enable_memory_safety: bool = field(default_factory=lambda: os.getenv('VISION_MEMORY_SAFETY', 'true').lower() == 'true')
    reject_on_memory_pressure: bool = field(default_factory=lambda: os.getenv('VISION_REJECT_ON_MEMORY', 'true').lower() == 'true')
    
    # Feature flags
    enable_metrics: bool = field(default_factory=lambda: os.getenv('VISION_METRICS', 'true').lower() == 'true')
    enable_entity_extraction: bool = field(default_factory=lambda: os.getenv('VISION_EXTRACT_ENTITIES', 'true').lower() == 'true')
    enable_action_detection: bool = field(default_factory=lambda: os.getenv('VISION_DETECT_ACTIONS', 'true').lower() == 'true')
    enable_screen_sharing: bool = field(default_factory=lambda: os.getenv('VISION_SCREEN_SHARING', 'true').lower() == 'true')
    enable_continuous_monitoring: bool = field(default_factory=lambda: os.getenv('VISION_CONTINUOUS_ENABLED', 'true').lower() == 'true')
    enable_video_streaming: bool = field(default_factory=lambda: os.getenv('VISION_VIDEO_STREAMING', 'true').lower() == 'true')
    prefer_video_over_screenshots: bool = field(default_factory=lambda: os.getenv('VISION_PREFER_VIDEO', 'true').lower() == 'true')
    
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
        self.client = Anthropic(api_key=api_key)
        
        # Initialize memory safety monitor
        self.memory_monitor = MemorySafetyMonitor(self.config)
        
        # Initialize components based on config
        self.cache = MemoryAwareCache(self.config) if self.config.cache_enabled else None
        self.executor = ThreadPoolExecutor(max_workers=self.config.thread_pool_size)
        self.api_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        # Metrics storage
        self.recent_metrics: List[AnalysisMetrics] = []
        
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
            'update_interval': float(os.getenv('VISION_MONITOR_INTERVAL', '3.0'))
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
        self._screen_sharing_config = {
            'enabled': self.config.enable_screen_sharing
        }
        
        # Initialize video streaming module (lazy loading)
        self.video_streaming = None
        self._video_streaming_config = {
            'enabled': self.config.enable_video_streaming
        }
        
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
                logger.info("Initialized continuous screen analyzer")
            except ImportError as e:
                logger.warning(f"Could not import continuous screen analyzer: {e}")
        return self.continuous_analyzer
    
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
    
    async def get_simplified_vision(self):
        """Get simplified vision system with lazy loading"""
        if self.simplified_vision is None and self._simplified_vision_config['enabled']:
            try:
                from .vision_system_claude_only import SimplifiedVisionSystem
                # Pass self as the claude_analyzer
                self.simplified_vision = SimplifiedVisionSystem(claude_analyzer=self)
                logger.info("Initialized simplified vision system")
            except ImportError as e:
                logger.warning(f"Could not import simplified vision: {e}")
        return self.simplified_vision
    
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
        if self.video_streaming is None and self._video_streaming_config['enabled']:
            try:
                from .video_stream_capture import VideoStreamCapture
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
            
            # Check cache if enabled
            if should_use_cache and self.cache:
                cache_key = self._generate_cache_key(image_hash, prompt)
                cached_entry = await self.cache.get(cache_key)
                if cached_entry and self._is_cache_valid(cached_entry):
                    metrics.cache_hit = True
                    metrics.total_time = time.time() - start_time
                    logger.debug(f"Cache hit for prompt: {prompt[:50]}...")
                    return cached_entry.result, metrics
            
            # Compress image if enabled
            if self.config.compression_enabled:
                compression_start = time.time()
                image_base64, compressed_size = await self._compress_and_encode(pil_image)
                metrics.image_size_compressed = compressed_size
                metrics.compression_ratio = 1 - (compressed_size / metrics.image_size_original)
                logger.debug(f"Compressed image from {metrics.image_size_original} to "
                           f"{compressed_size} bytes ({metrics.compression_ratio:.1%} reduction)")
            else:
                image_base64 = self._encode_image(pil_image)
                metrics.image_size_compressed = len(base64.b64decode(image_base64))
            
            # Make API call with rate limiting
            api_start = time.time()
            async with self.api_semaphore:
                if priority == "high":
                    # High priority requests get processed immediately
                    result = await self._call_claude_api(image_base64, prompt)
                else:
                    # Normal priority may be delayed if system is busy
                    if self._get_system_load() > (self.config.cpu_threshold_percent / 100):
                        await asyncio.sleep(0.5)  # Brief delay to reduce load
                    result = await self._call_claude_api(image_base64, prompt)
            metrics.api_call_time = time.time() - api_start
            
            # Parse response
            parsing_start = time.time()
            parsed_result = self._parse_claude_response(result)
            metrics.parsing_time = time.time() - parsing_start
            
            # Cache result if enabled
            if should_use_cache and self.cache:
                cache_entry = CacheEntry(
                    result=parsed_result,
                    timestamp=datetime.now(),
                    prompt_hash=hashlib.md5(prompt.encode()).hexdigest(),
                    image_hash=image_hash
                )
                await self.cache.put(cache_key, cache_entry)
            
            # Track metrics
            metrics.total_time = time.time() - start_time
            if self.config.enable_metrics:
                self._track_metrics(metrics)
            
            return parsed_result, metrics
            
        except Exception as e:
            logger.error(f"Error in analyze_screenshot: {e}")
            metrics.total_time = time.time() - start_time
            return {"error": str(e), "description": "Analysis failed"}, metrics
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
    
    async def _call_claude_api(self, image_base64: str, prompt: str) -> str:
        """Make API call to Claude"""
        # Use lambda to properly pass arguments to messages.create
        message = await asyncio.get_event_loop().run_in_executor(
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
                            "text": prompt
                        }
                    ]
                }]
            )
        )
        return message.content[0].text
    
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
        
        return None
    
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
        if self.video_streaming and self.video_streaming.is_capturing:
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
        """Capture screen using the best available method"""
        try:
            # If video streaming is enabled and running, get frame from there
            if self.config.prefer_video_over_screenshots and self.video_streaming and self.video_streaming.is_capturing:
                frame_data = self.video_streaming.frame_buffer.get_latest_frame()
                if frame_data:
                    # Convert numpy array to PIL Image
                    frame = frame_data['data']
                    return Image.fromarray(frame)
            
            # Otherwise use traditional screenshot method
            import subprocess
            from PIL import ImageGrab
            import platform
            
            if platform.system() == 'Darwin':
                # Use macOS screencapture command for better performance
                import tempfile
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                    tmp_path = tmp.name
                
                # Capture screen with reduced quality for memory efficiency
                result = subprocess.run(
                    ['screencapture', '-C', '-x', '-t', 'png', tmp_path],
                    capture_output=True
                )
                
                if result.returncode == 0:
                    # Load and return the image
                    image = Image.open(tmp_path)
                    # Clean up temp file
                    os.unlink(tmp_path)
                    return image
            else:
                # Fallback to PIL ImageGrab
                return ImageGrab.grab()
                
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    async def describe_screen(self, params: Dict[str, Any]) -> Any:
        """Describe screen for continuous analyzer compatibility"""
        # Extract query from params
        query = params.get('query', 'Describe what you see on screen')
        
        # Use smart analyze for the description
        # In real use, this would capture the current screen
        # For now, we expect the screenshot to be provided
        screenshot = params.get('screenshot')
        if screenshot is None:
            return {'success': False, 'description': 'No screenshot provided'}
        
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
        
        if self.video_streaming:
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
        # Check memory before starting
        memory_status = self.memory_monitor.check_memory_safety()
        if not memory_status.is_safe:
            return {
                'success': False,
                'error': 'Insufficient memory for video streaming',
                'memory_status': memory_status.__dict__
            }
        
        # Get video streaming manager
        video_streaming = await self.get_video_streaming()
        if not video_streaming:
            return {
                'success': False,
                'error': 'Video streaming not available'
            }
        
        # Start video streaming
        success = await video_streaming.start_streaming()
        
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
            return {
                'success': False,
                'error': 'Failed to start video streaming'
            }
    
    async def stop_video_streaming(self) -> Dict[str, Any]:
        """Stop video streaming"""
        if self.video_streaming:
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
        if self.video_streaming:
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
        if not self.video_streaming or not self.video_streaming.is_capturing:
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
        if self.video_streaming and self.video_streaming.is_capturing:
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
            result['capture_mode'] = 'video_streaming' if self.video_streaming and self.video_streaming.is_capturing else 'screenshot'
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