"""
JARVIS Vision System - Sliding Window Integration Example
Shows how to use sliding window in the JARVIS vision pipeline
Optimized for 16GB RAM macOS systems
"""

import asyncio
import os
import cv2
import numpy as np
from typing import Dict, Any, Optional
import json
import time
from dataclasses import dataclass
import logging

# Import JARVIS vision components
from claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JarvisVisionCommand:
    """Vision command from JARVIS"""
    command_type: str  # 'analyze_screen', 'find_element', 'monitor_region'
    query: str
    region: Optional[Dict[str, int]] = None  # x, y, width, height
    priority: str = 'normal'  # 'high', 'normal', 'low'

class JarvisSlidingWindowVision:
    """
    JARVIS Vision System with Sliding Window Support
    Intelligently switches between full-screen and sliding window analysis
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Initialize the integrated analyzer
        self.analyzer = ClaudeVisionAnalyzer(api_key)
        
        # Decision thresholds (from environment)
        self.use_sliding_threshold_px = int(os.getenv('JARVIS_SLIDING_THRESHOLD_PX', '800000'))  # 800k pixels
        self.memory_threshold_mb = float(os.getenv('JARVIS_MEMORY_THRESHOLD_MB', '2000'))
        self.complexity_threshold = float(os.getenv('JARVIS_COMPLEXITY_THRESHOLD', '0.7'))
        
        logger.info(f"JarvisVisionSystem initialized with sliding threshold: {self.use_sliding_threshold_px} pixels")
    
    async def process_vision_command(self, command: JarvisVisionCommand, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Process a vision command from JARVIS
        Automatically decides whether to use full or sliding window analysis
        """
        start_time = time.time()
        
        # Extract region if specified
        if command.region:
            screenshot = self._extract_region(screenshot, command.region)
        
        # Decide analysis method
        use_sliding = self._should_use_sliding_window(screenshot, command)
        
        logger.info(f"Processing {command.command_type} with {'sliding window' if use_sliding else 'full'} analysis")
        
        # Perform analysis
        if use_sliding:
            result = await self._sliding_window_analysis(screenshot, command)
        else:
            result = await self._full_analysis(screenshot, command)
        
        # Add metadata
        result['metadata'] = {
            'command_type': command.command_type,
            'analysis_method': 'sliding_window' if use_sliding else 'full',
            'processing_time_ms': (time.time() - start_time) * 1000,
            'image_size': f"{screenshot.shape[1]}x{screenshot.shape[0]}",
            'query': command.query
        }
        
        return result
    
    def _should_use_sliding_window(self, screenshot: np.ndarray, command: JarvisVisionCommand) -> bool:
        """Decide whether to use sliding window based on multiple factors"""
        height, width = screenshot.shape[:2]
        total_pixels = height * width
        
        # Factor 1: Image size
        if total_pixels > self.use_sliding_threshold_px:
            logger.info(f"Using sliding window due to large image size: {total_pixels} pixels")
            return True
        
        # Factor 2: Available memory
        available_mb = self._get_available_memory_mb()
        if available_mb < self.memory_threshold_mb:
            logger.info(f"Using sliding window due to low memory: {available_mb:.1f} MB")
            return True
        
        # Factor 3: Command type
        if command.command_type in ['find_element', 'monitor_region']:
            # These commands benefit from focused analysis
            logger.info(f"Using sliding window for {command.command_type} command")
            return True
        
        # Factor 4: Query complexity
        if self._estimate_query_complexity(command.query) > self.complexity_threshold:
            logger.info("Using sliding window due to complex query")
            return True
        
        return False
    
    def _estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity (0.0 to 1.0)"""
        # Simple heuristic based on query characteristics
        complexity = 0.0
        
        # Long queries are more complex
        if len(query) > 100:
            complexity += 0.3
        
        # Multiple requirements increase complexity
        if any(word in query.lower() for word in ['and', 'also', 'with', 'including']):
            complexity += 0.2
        
        # Specific element searches are complex
        if any(word in query.lower() for word in ['find', 'locate', 'search', 'where']):
            complexity += 0.3
        
        # Counting or listing increases complexity
        if any(word in query.lower() for word in ['count', 'list', 'all', 'every']):
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    async def _sliding_window_analysis(self, screenshot: np.ndarray, command: JarvisVisionCommand) -> Dict[str, Any]:
        """Perform sliding window analysis"""
        # Configure based on command priority
        window_config = {}
        if command.priority == 'high':
            # Use higher quality for high priority
            window_config = {
                'window_width': 500,
                'window_height': 400,
                'max_windows': 5
            }
        elif command.priority == 'low':
            # Use lower quality for low priority
            window_config = {
                'window_width': 300,
                'window_height': 250,
                'max_windows': 2
            }
        
        # Perform analysis
        result = await self.analyzer.analyze_with_sliding_window(
            screenshot, 
            command.query,
            window_config=window_config
        )
        
        # Post-process based on command type
        if command.command_type == 'find_element':
            result = self._filter_for_element_search(result, command.query)
        elif command.command_type == 'monitor_region':
            result = self._enhance_for_monitoring(result)
        
        return result
    
    async def _full_analysis(self, screenshot: np.ndarray, command: JarvisVisionCommand) -> Dict[str, Any]:
        """Perform full image analysis"""
        # Use the enhanced Claude analyzer
        result = await self.analyzer.analyze_screenshot_async(
            screenshot=screenshot,
            query=command.query,
            quick_mode=(command.priority == 'low')
        )
        
        # Convert to consistent format
        return {
            'summary': result.get('description', ''),
            'objects_detected': result.get('entities', {}).get('applications', []),
            'text_found': result.get('entities', {}).get('text', []),
            'ui_elements': result.get('entities', {}).get('ui_elements', []),
            'confidence': result.get('confidence', 0.5)
        }
    
    def _filter_for_element_search(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Filter results for element search commands"""
        # Find regions that likely contain the searched element
        important_regions = result.get('important_regions', [])
        
        # Score each region based on query relevance
        scored_regions = []
        for region in important_regions:
            score = self._calculate_relevance_score(region['description'], query)
            if score > 0.5:
                scored_regions.append({
                    **region,
                    'relevance_score': score
                })
        
        # Sort by relevance
        scored_regions.sort(key=lambda r: r['relevance_score'], reverse=True)
        
        # Update result
        result['found_elements'] = scored_regions[:3]  # Top 3 matches
        if scored_regions:
            result['summary'] = f"Found {len(scored_regions)} potential matches for '{query}'. " \
                               f"Best match at ({scored_regions[0]['bounds']['x']}, {scored_regions[0]['bounds']['y']})"
        
        return result
    
    def _enhance_for_monitoring(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance results for monitoring commands"""
        # Add change detection info
        result['changes_detected'] = []
        result['monitoring_summary'] = "Region monitored successfully"
        
        # Identify potential changes or important events
        for region in result.get('important_regions', []):
            if region['confidence'] > 0.8:
                result['changes_detected'].append({
                    'location': region['bounds'],
                    'description': region['description'],
                    'importance': 'high' if region['confidence'] > 0.9 else 'medium'
                })
        
        return result
    
    def _calculate_relevance_score(self, description: str, query: str) -> float:
        """Calculate relevance score between description and query"""
        # Simple keyword matching (in production, use better NLP)
        query_words = query.lower().split()
        description_words = description.lower().split()
        
        matches = sum(1 for word in query_words if word in description_words)
        return matches / len(query_words) if query_words else 0.0
    
    def _extract_region(self, image: np.ndarray, region: Dict[str, int]) -> np.ndarray:
        """Extract a specific region from the image"""
        x, y = region.get('x', 0), region.get('y', 0)
        w, h = region.get('width', image.shape[1]), region.get('height', image.shape[0])
        
        # Ensure bounds are valid
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        return image[y:y+h, x:x+w]
    
    def _get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        try:
            import psutil
            return psutil.virtual_memory().available / 1024 / 1024
        except:
            return 2000.0  # Default to 2GB if can't determine
    
    async def benchmark_methods(self, screenshot: np.ndarray, query: str) -> Dict[str, Any]:
        """Benchmark sliding window vs full analysis"""
        results = {}
        
        # Test full analysis
        start = time.time()
        full_result = await self._full_analysis(
            screenshot,
            JarvisVisionCommand('analyze_screen', query)
        )
        full_time = time.time() - start
        
        # Test sliding window
        start = time.time()
        sliding_result = await self._sliding_window_analysis(
            screenshot,
            JarvisVisionCommand('analyze_screen', query)
        )
        sliding_time = time.time() - start
        
        # Memory usage comparison
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            'full_analysis': {
                'time_seconds': full_time,
                'summary': full_result.get('summary', '')[:100] + '...'
            },
            'sliding_window': {
                'time_seconds': sliding_time,
                'windows_analyzed': sliding_result.get('metadata', {}).get('windows_analyzed', 0),
                'memory_saved_mb': sliding_result.get('metadata', {}).get('memory_saved_mb', 0)
            },
            'comparison': {
                'speedup': full_time / sliding_time if sliding_time > 0 else 1,
                'memory_usage_mb': memory_mb
            }
        }

# Example usage scenarios
async def example_use_cases():
    """Demonstrate various JARVIS vision use cases"""
    
    # Initialize JARVIS vision
    api_key = os.getenv("ANTHROPIC_API_KEY", "mock_api_key")
    jarvis_vision = JarvisSlidingWindowVision(api_key)
    
    # Create test screenshot
    screenshot = create_test_screenshot()
    
    print("=" * 60)
    print("JARVIS VISION - SLIDING WINDOW EXAMPLES")
    print("=" * 60)
    
    # Use Case 1: Find specific UI element
    print("\n1. Finding WhatsApp close button:")
    command = JarvisVisionCommand(
        command_type='find_element',
        query='close button for WhatsApp window',
        priority='high'
    )
    result = await jarvis_vision.process_vision_command(command, screenshot)
    print(f"Result: {result['summary']}")
    if 'found_elements' in result and result['found_elements']:
        print(f"Found at: {result['found_elements'][0]['bounds']}")
    
    # Use Case 2: Monitor a region for changes
    print("\n2. Monitoring notification area:")
    command = JarvisVisionCommand(
        command_type='monitor_region',
        query='check for new notifications or alerts',
        region={'x': 1500, 'y': 0, 'width': 420, 'height': 100},
        priority='normal'
    )
    result = await jarvis_vision.process_vision_command(command, screenshot)
    print(f"Result: {result.get('monitoring_summary', 'No summary')}")
    print(f"Changes detected: {len(result.get('changes_detected', []))}")
    
    # Use Case 3: General screen analysis
    print("\n3. General screen analysis:")
    command = JarvisVisionCommand(
        command_type='analyze_screen',
        query='What applications are currently open and what is the user doing?',
        priority='normal'
    )
    result = await jarvis_vision.process_vision_command(command, screenshot)
    print(f"Result: {result['summary'][:200]}...")
    print(f"Analysis method: {result['metadata']['analysis_method']}")
    
    # Use Case 4: Quick low-priority check
    print("\n4. Quick status check:")
    command = JarvisVisionCommand(
        command_type='analyze_screen',
        query='Is there any error message on screen?',
        priority='low'
    )
    result = await jarvis_vision.process_vision_command(command, screenshot)
    print(f"Result: {result['summary']}")
    print(f"Processing time: {result['metadata']['processing_time_ms']:.1f}ms")
    
    # Benchmark comparison
    print("\n5. Benchmark comparison:")
    benchmark = await jarvis_vision.benchmark_methods(screenshot, "Analyze the entire screen content")
    print(f"Full analysis time: {benchmark['full_analysis']['time_seconds']:.2f}s")
    print(f"Sliding window time: {benchmark['sliding_window']['time_seconds']:.2f}s")
    print(f"Speedup: {benchmark['comparison']['speedup']:.1f}x")
    print(f"Memory saved: {benchmark['sliding_window']['memory_saved_mb']:.1f}MB")

def create_test_screenshot():
    """Create a realistic test screenshot"""
    # Create 1920x1080 screenshot
    screenshot = np.ones((1080, 1920, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add WhatsApp window mockup
    cv2.rectangle(screenshot, (100, 100), (800, 700), (255, 255, 255), -1)
    cv2.rectangle(screenshot, (100, 100), (800, 140), (0, 128, 105), -1)  # Green header
    cv2.putText(screenshot, "WhatsApp", (120, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    # Add close button
    cv2.circle(screenshot, (770, 120), 10, (255, 255, 255), -1)
    cv2.putText(screenshot, "X", (765, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 105), 2)
    
    # Add notification area
    cv2.rectangle(screenshot, (1500, 0), (1920, 100), (50, 50, 50), -1)
    cv2.putText(screenshot, "2 new messages", (1520, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Add some other UI elements
    cv2.rectangle(screenshot, (850, 200), (1400, 600), (255, 255, 255), -1)
    cv2.putText(screenshot, "Main Application", (870, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return screenshot

if __name__ == "__main__":
    # Run examples
    asyncio.run(example_use_cases())