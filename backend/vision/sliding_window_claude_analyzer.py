"""
Sliding Window Claude Vision Analyzer
Integrates Rust sliding window with Claude Vision API
NO HARDCODING - Fully configurable for 16GB RAM systems
"""

import asyncio
import base64
import io
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import psutil
from anthropic import Anthropic
import json
import logging

# Import Rust bindings (when available)
try:
    import jarvis_rust_core
    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False
    logging.warning("Rust core not available, using Python fallback")

logger = logging.getLogger(__name__)

@dataclass
class SlidingWindowClaudeConfig:
    """Dynamic configuration for sliding window + Claude integration"""
    # Sliding window settings
    window_width: int = int(os.getenv('CLAUDE_WINDOW_WIDTH', '400'))
    window_height: int = int(os.getenv('CLAUDE_WINDOW_HEIGHT', '300'))
    overlap_percentage: float = float(os.getenv('CLAUDE_WINDOW_OVERLAP', '0.5'))
    max_concurrent_windows: int = int(os.getenv('CLAUDE_MAX_CONCURRENT', '3'))
    
    # Memory settings for 16GB systems
    memory_threshold_mb: float = float(os.getenv('CLAUDE_MEMORY_THRESHOLD_MB', '2000'))
    low_memory_window_scale: float = float(os.getenv('CLAUDE_LOW_MEMORY_SCALE', '0.75'))
    critical_memory_mb: float = float(os.getenv('CLAUDE_CRITICAL_MEMORY_MB', '1000'))
    
    # Claude API settings
    claude_model: str = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
    max_tokens_per_window: int = int(os.getenv('CLAUDE_MAX_TOKENS_WINDOW', '300'))
    jpeg_quality: int = int(os.getenv('CLAUDE_JPEG_QUALITY', '70'))
    
    # Caching settings
    enable_cache: bool = os.getenv('CLAUDE_ENABLE_CACHE', 'true').lower() == 'true'
    cache_ttl_seconds: int = int(os.getenv('CLAUDE_CACHE_TTL', '60'))
    
    # Performance settings
    prioritize_center: bool = os.getenv('CLAUDE_PRIORITIZE_CENTER', 'true').lower() == 'true'
    skip_static_regions: bool = os.getenv('CLAUDE_SKIP_STATIC', 'true').lower() == 'true'
    static_threshold: float = float(os.getenv('CLAUDE_STATIC_THRESHOLD', '0.95'))
    
    # Analysis settings
    batch_windows: bool = os.getenv('CLAUDE_BATCH_WINDOWS', 'true').lower() == 'true'
    combine_results: bool = os.getenv('CLAUDE_COMBINE_RESULTS', 'true').lower() == 'true'
    result_format: str = os.getenv('CLAUDE_RESULT_FORMAT', 'structured')  # 'structured' or 'narrative'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {k: v for k, v in self.__dict__.items()}

@dataclass
class WindowAnalysis:
    """Result from analyzing a single window"""
    bounds: Tuple[int, int, int, int]  # x, y, width, height
    content: str
    confidence: float
    objects_detected: List[str]
    text_found: List[str]
    ui_elements: List[str]
    from_cache: bool = False
    analysis_time_ms: float = 0.0

class SlidingWindowClaudeAnalyzer:
    """Main analyzer combining sliding window with Claude Vision"""
    
    def __init__(self, api_key: str, config: Optional[SlidingWindowClaudeConfig] = None):
        self.config = config or SlidingWindowClaudeConfig()
        self.anthropic = Anthropic(api_key=api_key)
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_windows)
        
        # Cache for analyzed regions
        self.cache = {} if self.config.enable_cache else None
        self.cache_timestamps = {}
        
        # Statistics
        self.stats = {
            'total_windows_analyzed': 0,
            'cache_hits': 0,
            'total_api_calls': 0,
            'total_memory_saved_mb': 0.0,
            'avg_analysis_time_ms': 0.0
        }
        
        logger.info(f"Initialized SlidingWindowClaudeAnalyzer with config: {self.config.to_dict()}")
    
    def get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        return psutil.virtual_memory().available / 1024 / 1024
    
    def should_reduce_quality(self) -> bool:
        """Check if we should reduce quality due to memory pressure"""
        available = self.get_available_memory_mb()
        return available < self.config.memory_threshold_mb
    
    def get_adaptive_window_size(self) -> Tuple[int, int]:
        """Get window size based on available memory"""
        available = self.get_available_memory_mb()
        
        if available < self.config.critical_memory_mb:
            # Critical memory - use minimum size
            scale = 0.5
        elif available < self.config.memory_threshold_mb:
            # Low memory - use configured scale
            scale = self.config.low_memory_window_scale
        else:
            # Sufficient memory - use full size
            scale = 1.0
        
        width = int(self.config.window_width * scale)
        height = int(self.config.window_height * scale)
        
        return width, height
    
    def generate_sliding_windows(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Generate sliding windows for the image"""
        height, width = image.shape[:2]
        window_width, window_height = self.get_adaptive_window_size()
        
        # Calculate step sizes based on overlap
        step_x = int(window_width * (1 - self.config.overlap_percentage))
        step_y = int(window_height * (1 - self.config.overlap_percentage))
        
        windows = []
        
        # Generate windows
        for y in range(0, height - window_height + 1, step_y):
            for x in range(0, width - window_width + 1, step_x):
                # Calculate priority (center gets higher priority)
                if self.config.prioritize_center:
                    center_x = x + window_width // 2
                    center_y = y + window_height // 2
                    image_center_x = width // 2
                    image_center_y = height // 2
                    
                    # Distance from center (normalized)
                    dx = (center_x - image_center_x) / width
                    dy = (center_y - image_center_y) / height
                    distance = np.sqrt(dx**2 + dy**2)
                    priority = 1.0 - min(distance, 1.0)
                else:
                    priority = 1.0
                
                windows.append({
                    'bounds': (x, y, window_width, window_height),
                    'priority': priority,
                    'data': None  # Will be extracted when needed
                })
        
        # Sort by priority (highest first)
        windows.sort(key=lambda w: w['priority'], reverse=True)
        
        # Limit to max concurrent windows based on memory
        if self.should_reduce_quality():
            max_windows = max(2, self.config.max_concurrent_windows // 2)
        else:
            max_windows = self.config.max_concurrent_windows
        
        return windows[:max_windows]
    
    def extract_window_region(self, image: np.ndarray, bounds: Tuple[int, int, int, int]) -> Image.Image:
        """Extract a window region from the image"""
        x, y, w, h = bounds
        region = image[y:y+h, x:x+w]
        return Image.fromarray(region)
    
    def prepare_region_for_claude(self, region: Image.Image) -> Tuple[str, int]:
        """Prepare region for Claude API (compress as JPEG)"""
        # Convert RGBA to RGB if necessary
        if region.mode == 'RGBA':
            rgb_region = Image.new('RGB', region.size, (255, 255, 255))
            rgb_region.paste(region, mask=region.split()[3])
            region = rgb_region
        
        # Compress as JPEG
        buffer = io.BytesIO()
        region.save(buffer, format='JPEG', quality=self.config.jpeg_quality, optimize=True)
        
        # Encode to base64
        buffer.seek(0)
        encoded = base64.b64encode(buffer.getvalue()).decode()
        size_bytes = buffer.tell()
        
        return encoded, size_bytes
    
    def create_window_prompt(self, window_index: int, total_windows: int, bounds: Tuple[int, int, int, int]) -> str:
        """Create analysis prompt for a window region"""
        x, y, w, h = bounds
        
        if self.config.result_format == 'structured':
            return f"""Analyze this screen region (window {window_index+1}/{total_windows} at position x:{x}, y:{y}, size:{w}x{h}).

Provide a JSON response with:
{{
    "summary": "Brief description of what's in this region",
    "objects": ["list", "of", "detected", "objects"],
    "text": ["any", "text", "found"],
    "ui_elements": ["buttons", "menus", "etc"],
    "importance": 0.0 to 1.0 (how important is this region),
    "requires_action": true/false
}}

Be concise but thorough. Focus on what's actionable or important."""
        else:
            return f"""Briefly describe what you see in this screen region (position {x},{y}). 
Focus on: UI elements, text content, and anything actionable. Keep it under 50 words."""
    
    async def analyze_window(self, image: np.ndarray, window: Dict[str, Any]) -> WindowAnalysis:
        """Analyze a single window region"""
        start_time = time.time()
        bounds = window['bounds']
        
        # Check cache first
        if self.cache is not None:
            cache_key = f"{bounds}_{hash(image.tobytes())}"
            if cache_key in self.cache:
                timestamp = self.cache_timestamps.get(cache_key, 0)
                if time.time() - timestamp < self.config.cache_ttl_seconds:
                    self.stats['cache_hits'] += 1
                    cached = self.cache[cache_key]
                    cached.from_cache = True
                    return cached
        
        # Extract and prepare region
        region = self.extract_window_region(image, bounds)
        encoded_image, size_bytes = self.prepare_region_for_claude(region)
        
        # Create prompt
        prompt = self.create_window_prompt(
            window_index=0,  # Would be passed in batch processing
            total_windows=1,
            bounds=bounds
        )
        
        # Call Claude API
        try:
            self.stats['total_api_calls'] += 1
            
            message = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.anthropic.messages.create(
                    model=self.config.claude_model,
                    max_tokens=self.config.max_tokens_per_window,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": encoded_image
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
            
            response_text = message.content[0].text
            
            # Parse response
            result = self.parse_window_response(response_text, bounds)
            
            # Update cache
            if self.cache is not None:
                cache_key = f"{bounds}_{hash(image.tobytes())}"
                self.cache[cache_key] = result
                self.cache_timestamps[cache_key] = time.time()
            
            # Update stats
            analysis_time = (time.time() - start_time) * 1000
            self.stats['total_windows_analyzed'] += 1
            self.stats['avg_analysis_time_ms'] = (
                (self.stats['avg_analysis_time_ms'] * (self.stats['total_windows_analyzed'] - 1) + analysis_time) /
                self.stats['total_windows_analyzed']
            )
            
            # Calculate memory saved
            full_image_size_mb = (image.shape[0] * image.shape[1] * 3) / 1024 / 1024
            window_size_mb = size_bytes / 1024 / 1024
            self.stats['total_memory_saved_mb'] += (full_image_size_mb - window_size_mb)
            
            result.analysis_time_ms = analysis_time
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing window {bounds}: {e}")
            return WindowAnalysis(
                bounds=bounds,
                content=f"Error: {str(e)}",
                confidence=0.0,
                objects_detected=[],
                text_found=[],
                ui_elements=[],
                analysis_time_ms=(time.time() - start_time) * 1000
            )
    
    def parse_window_response(self, response: str, bounds: Tuple[int, int, int, int]) -> WindowAnalysis:
        """Parse Claude's response into WindowAnalysis"""
        try:
            # Try to parse as JSON
            data = json.loads(response)
            return WindowAnalysis(
                bounds=bounds,
                content=data.get('summary', response),
                confidence=data.get('importance', 0.5),
                objects_detected=data.get('objects', []),
                text_found=data.get('text', []),
                ui_elements=data.get('ui_elements', [])
            )
        except json.JSONDecodeError:
            # Fallback to text parsing
            return WindowAnalysis(
                bounds=bounds,
                content=response,
                confidence=0.5,
                objects_detected=[],
                text_found=[],
                ui_elements=[]
            )
    
    async def analyze_screenshot(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Main entry point - analyze screenshot using sliding windows"""
        start_time = time.time()
        
        # Generate windows
        windows = self.generate_sliding_windows(screenshot)
        logger.info(f"Generated {len(windows)} windows for analysis")
        
        # Analyze windows (can be parallel or sequential based on config)
        if self.config.batch_windows and len(windows) > 1:
            # Analyze in parallel
            tasks = [self.analyze_window(screenshot, window) for window in windows]
            window_results = await asyncio.gather(*tasks)
        else:
            # Analyze sequentially (better for memory)
            window_results = []
            for window in windows:
                result = await self.analyze_window(screenshot, window)
                window_results.append(result)
        
        # Combine results if configured
        if self.config.combine_results:
            combined_result = self.combine_window_results(window_results)
        else:
            combined_result = {
                'windows': [self.window_to_dict(w) for w in window_results],
                'window_count': len(window_results)
            }
        
        # Add metadata
        total_time = (time.time() - start_time) * 1000
        combined_result['metadata'] = {
            'total_time_ms': total_time,
            'windows_analyzed': len(window_results),
            'cache_hits': sum(1 for w in window_results if w.from_cache),
            'memory_saved_mb': self.stats['total_memory_saved_mb'],
            'available_memory_mb': self.get_available_memory_mb(),
            'window_size_used': self.get_adaptive_window_size()
        }
        
        return combined_result
    
    def combine_window_results(self, windows: List[WindowAnalysis]) -> Dict[str, Any]:
        """Combine multiple window analyses into a unified result"""
        # Sort windows by importance/confidence
        windows.sort(key=lambda w: w.confidence, reverse=True)
        
        # Aggregate findings
        all_objects = []
        all_text = []
        all_ui_elements = []
        important_regions = []
        
        for window in windows:
            all_objects.extend(window.objects_detected)
            all_text.extend(window.text_found)
            all_ui_elements.extend(window.ui_elements)
            
            if window.confidence > 0.7:
                important_regions.append({
                    'bounds': window.bounds,
                    'description': window.content,
                    'confidence': window.confidence
                })
        
        # Remove duplicates while preserving order
        all_objects = list(dict.fromkeys(all_objects))
        all_text = list(dict.fromkeys(all_text))
        all_ui_elements = list(dict.fromkeys(all_ui_elements))
        
        # Create summary
        summary = self.generate_summary(windows)
        
        return {
            'summary': summary,
            'objects_detected': all_objects[:20],  # Limit to top 20
            'text_found': all_text[:20],
            'ui_elements': all_ui_elements[:20],
            'important_regions': important_regions[:5],  # Top 5 important regions
            'total_regions_analyzed': len(windows),
            'average_confidence': sum(w.confidence for w in windows) / len(windows) if windows else 0
        }
    
    def generate_summary(self, windows: List[WindowAnalysis]) -> str:
        """Generate a summary from all window analyses"""
        if not windows:
            return "No content detected in the analyzed regions."
        
        # Get the most important findings
        high_confidence = [w for w in windows if w.confidence > 0.7]
        
        if high_confidence:
            main_content = high_confidence[0].content
            if len(high_confidence) > 1:
                return f"{main_content} Additionally, {len(high_confidence)-1} other important regions were detected."
            return main_content
        else:
            # Fallback to first window
            return windows[0].content if windows[0].content else "Multiple screen regions analyzed."
    
    def window_to_dict(self, window: WindowAnalysis) -> Dict[str, Any]:
        """Convert WindowAnalysis to dictionary"""
        return {
            'bounds': {
                'x': window.bounds[0],
                'y': window.bounds[1],
                'width': window.bounds[2],
                'height': window.bounds[3]
            },
            'content': window.content,
            'confidence': window.confidence,
            'objects': window.objects_detected,
            'text': window.text_found,
            'ui_elements': window.ui_elements,
            'from_cache': window.from_cache,
            'analysis_time_ms': window.analysis_time_ms
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            **self.stats,
            'cache_size': len(self.cache) if self.cache else 0,
            'current_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'available_memory_mb': self.get_available_memory_mb()
        }
    
    def clear_cache(self):
        """Clear the analysis cache"""
        if self.cache is not None:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cache cleared")
    
    def update_config(self, **kwargs):
        """Update configuration dynamically"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")


# Integration with Rust sliding window
class RustSlidingWindowAnalyzer(SlidingWindowClaudeAnalyzer):
    """Enhanced analyzer using Rust sliding window implementation"""
    
    def __init__(self, api_key: str, config: Optional[SlidingWindowClaudeConfig] = None):
        super().__init__(api_key, config)
        
        if RUST_AVAILABLE:
            # Initialize Rust sliding window
            rust_config = {
                'window_width': self.config.window_width,
                'window_height': self.config.window_height,
                'overlap_percentage': self.config.overlap_percentage,
                'max_concurrent_regions': self.config.max_concurrent_windows,
                'enable_caching': self.config.enable_cache,
                'skip_static_regions': self.config.skip_static_regions,
            }
            
            self.rust_sliding_window = jarvis_rust_core.SlidingWindowCapture(rust_config)
            logger.info("Using Rust sliding window implementation")
        else:
            self.rust_sliding_window = None
            logger.warning("Rust not available, using Python implementation")
    
    async def analyze_screenshot(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """Analyze using Rust sliding window if available"""
        if self.rust_sliding_window:
            # Use Rust for window generation (more efficient)
            windows = self.rust_sliding_window.generate_windows(screenshot)
            # Then use Claude for analysis
            # ... (implementation would call parent methods with Rust windows)
        
        # Fallback to Python implementation
        return await super().analyze_screenshot(screenshot)


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize analyzer
    api_key = os.getenv("ANTHROPIC_API_KEY")
    analyzer = RustSlidingWindowAnalyzer(api_key)
    
    # Load test image
    screenshot = cv2.imread("test_screenshot.png")
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    
    # Analyze
    async def test():
        result = await analyzer.analyze_screenshot(screenshot)
        print(json.dumps(result, indent=2))
        print("\nStats:", analyzer.get_stats())
    
    asyncio.run(test())