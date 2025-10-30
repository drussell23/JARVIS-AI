"""
JARVIS Vision System - Sliding Window Integration Example

This module demonstrates how to integrate sliding window analysis into the JARVIS vision pipeline.
It provides intelligent switching between full-screen and sliding window analysis based on
image size, memory constraints, and query complexity. Optimized for 16GB RAM macOS systems.

The module includes:
- JarvisVisionCommand dataclass for structured vision commands
- JarvisSlidingWindowVision class for intelligent analysis switching
- Example use cases and benchmarking functionality
- Test screenshot generation for demonstration

Example:
    >>> import asyncio
    >>> from jarvis_sliding_window_example import JarvisSlidingWindowVision, JarvisVisionCommand
    >>> 
    >>> async def main():
    ...     jarvis_vision = JarvisSlidingWindowVision("your_api_key")
    ...     command = JarvisVisionCommand('find_element', 'close button')
    ...     result = await jarvis_vision.process_vision_command(command, screenshot)
    ...     print(result['summary'])
    >>> 
    >>> asyncio.run(main())
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
    """Vision command structure for JARVIS system.
    
    Represents a structured command that can be sent to the JARVIS vision system
    for various types of visual analysis tasks.
    
    Attributes:
        command_type: Type of vision command ('analyze_screen', 'find_element', 'monitor_region')
        query: Natural language description of what to analyze or find
        region: Optional region of interest with x, y, width, height coordinates
        priority: Command priority level ('high', 'normal', 'low')
    
    Example:
        >>> command = JarvisVisionCommand(
        ...     command_type='find_element',
        ...     query='close button for WhatsApp window',
        ...     priority='high'
        ... )
    """
    command_type: str  # 'analyze_screen', 'find_element', 'monitor_region'
    query: str
    region: Optional[Dict[str, int]] = None  # x, y, width, height
    priority: str = 'normal'  # 'high', 'normal', 'low'

class JarvisSlidingWindowVision:
    """
    JARVIS Vision System with intelligent sliding window support.
    
    This class provides an intelligent vision analysis system that automatically
    switches between full-screen and sliding window analysis based on multiple
    factors including image size, available memory, query complexity, and command type.
    
    The system is optimized for macOS systems with 16GB RAM and provides efficient
    processing of various vision tasks while maintaining high accuracy.
    
    Attributes:
        api_key: Anthropic API key for Claude vision analysis
        analyzer: ClaudeVisionAnalyzer instance for performing analysis
        use_sliding_threshold_px: Pixel threshold for switching to sliding window
        memory_threshold_mb: Memory threshold in MB for sliding window decision
        complexity_threshold: Query complexity threshold (0.0-1.0)
    
    Example:
        >>> jarvis_vision = JarvisSlidingWindowVision("your_api_key")
        >>> command = JarvisVisionCommand('analyze_screen', 'What is on the screen?')
        >>> result = await jarvis_vision.process_vision_command(command, screenshot)
    """
    
    def __init__(self, api_key: str):
        """Initialize the JARVIS Vision System.
        
        Args:
            api_key: Anthropic API key for Claude vision analysis
            
        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError("API key cannot be empty or None")
            
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
        Process a vision command from JARVIS with intelligent analysis method selection.
        
        Automatically decides whether to use full-screen or sliding window analysis
        based on image characteristics, system resources, and command requirements.
        
        Args:
            command: JarvisVisionCommand containing the analysis request
            screenshot: Input image as numpy array (H, W, C)
            
        Returns:
            Dictionary containing analysis results with the following structure:
            {
                'summary': str,  # Main analysis summary
                'objects_detected': List[Dict],  # Detected objects/applications
                'text_found': List[str],  # Extracted text elements
                'ui_elements': List[Dict],  # UI elements found
                'confidence': float,  # Overall confidence score
                'metadata': Dict  # Processing metadata
            }
            
        Raises:
            ValueError: If screenshot is invalid or command is malformed
            RuntimeError: If analysis fails due to system constraints
            
        Example:
            >>> command = JarvisVisionCommand('find_element', 'close button')
            >>> result = await jarvis_vision.process_vision_command(command, screenshot)
            >>> print(f"Found element: {result['summary']}")
        """
        if screenshot is None or screenshot.size == 0:
            raise ValueError("Screenshot cannot be empty or None")
            
        start_time = time.time()
        
        # Extract region if specified
        if command.region:
            screenshot = self._extract_region(screenshot, command.region)
        
        # Decide analysis method
        use_sliding = self._should_use_sliding_window(screenshot, command)
        
        logger.info(f"Processing {command.command_type} with {'sliding window' if use_sliding else 'full'} analysis")
        
        try:
            # Perform analysis
            if use_sliding:
                result = await self._sliding_window_analysis(screenshot, command)
            else:
                result = await self._full_analysis(screenshot, command)
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise RuntimeError(f"Vision analysis failed: {str(e)}")
        
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
        """Decide whether to use sliding window based on multiple factors.
        
        Evaluates image size, available memory, command type, and query complexity
        to make an intelligent decision about analysis method.
        
        Args:
            screenshot: Input image as numpy array
            command: Vision command to be processed
            
        Returns:
            True if sliding window should be used, False for full analysis
        """
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
        """Estimate query complexity using heuristic analysis.
        
        Analyzes query characteristics to estimate processing complexity,
        helping decide between analysis methods.
        
        Args:
            query: Natural language query string
            
        Returns:
            Complexity score from 0.0 (simple) to 1.0 (very complex)
            
        Example:
            >>> complexity = jarvis_vision._estimate_query_complexity("Find all red buttons")
            >>> print(f"Query complexity: {complexity}")
        """
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
        """Perform sliding window analysis with command-specific optimization.
        
        Configures sliding window parameters based on command priority and type,
        then performs the analysis and post-processes results.
        
        Args:
            screenshot: Input image as numpy array
            command: Vision command with analysis parameters
            
        Returns:
            Analysis results dictionary with sliding window specific enhancements
            
        Raises:
            RuntimeError: If sliding window analysis fails
        """
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
        
        try:
            # Perform analysis
            result = await self.analyzer.analyze_with_sliding_window(
                screenshot, 
                command.query,
                window_config=window_config
            )
        except Exception as e:
            raise RuntimeError(f"Sliding window analysis failed: {str(e)}")
        
        # Post-process based on command type
        if command.command_type == 'find_element':
            result = self._filter_for_element_search(result, command.query)
        elif command.command_type == 'monitor_region':
            result = self._enhance_for_monitoring(result)
        
        return result
    
    async def _full_analysis(self, screenshot: np.ndarray, command: JarvisVisionCommand) -> Dict[str, Any]:
        """Perform full image analysis using the enhanced Claude analyzer.
        
        Processes the entire image at once using the Claude vision API,
        with optimization based on command priority.
        
        Args:
            screenshot: Input image as numpy array
            command: Vision command with analysis parameters
            
        Returns:
            Analysis results in standardized format
            
        Raises:
            RuntimeError: If full analysis fails
        """
        try:
            # Use the enhanced Claude analyzer
            result = await self.analyzer.analyze_screenshot_async(
                screenshot=screenshot,
                query=command.query,
                quick_mode=(command.priority == 'low')
            )
        except Exception as e:
            raise RuntimeError(f"Full analysis failed: {str(e)}")
        
        # Convert to consistent format
        return {
            'summary': result.get('description', ''),
            'objects_detected': result.get('entities', {}).get('applications', []),
            'text_found': result.get('entities', {}).get('text', []),
            'ui_elements': result.get('entities', {}).get('ui_elements', []),
            'confidence': result.get('confidence', 0.5)
        }
    
    def _filter_for_element_search(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Filter and enhance results for element search commands.
        
        Processes sliding window results to find and rank regions that likely
        contain the searched UI element based on query relevance.
        
        Args:
            result: Raw sliding window analysis results
            query: Original search query
            
        Returns:
            Enhanced results with found_elements and relevance scoring
        """
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
        """Enhance results for monitoring commands.
        
        Adds change detection information and monitoring-specific metadata
        to help track region changes over time.
        
        Args:
            result: Raw analysis results
            
        Returns:
            Enhanced results with monitoring-specific information
        """
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
        """Calculate relevance score between description and query.
        
        Uses simple keyword matching to determine how well a region description
        matches the search query. In production, this could be enhanced with
        more sophisticated NLP techniques.
        
        Args:
            description: Region description from analysis
            query: Original search query
            
        Returns:
            Relevance score from 0.0 to 1.0
        """
        # Simple keyword matching (in production, use better NLP)
        query_words = query.lower().split()
        description_words = description.lower().split()
        
        matches = sum(1 for word in query_words if word in description_words)
        return matches / len(query_words) if query_words else 0.0
    
    def _extract_region(self, image: np.ndarray, region: Dict[str, int]) -> np.ndarray:
        """Extract a specific region from the image with bounds validation.
        
        Safely extracts a rectangular region from the input image,
        ensuring all coordinates are within valid bounds.
        
        Args:
            image: Source image as numpy array
            region: Dictionary with x, y, width, height keys
            
        Returns:
            Extracted image region as numpy array
            
        Raises:
            ValueError: If region parameters are invalid
        """
        if not isinstance(region, dict):
            raise ValueError("Region must be a dictionary")
            
        x, y = region.get('x', 0), region.get('y', 0)
        w, h = region.get('width', image.shape[1]), region.get('height', image.shape[0])
        
        # Ensure bounds are valid
        x = max(0, min(x, image.shape[1] - 1))
        y = max(0, min(y, image.shape[0] - 1))
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            raise ValueError("Invalid region dimensions")
        
        return image[y:y+h, x:x+w]
    
    def _get_available_memory_mb(self) -> float:
        """Get available system memory in megabytes.
        
        Attempts to determine available system memory using psutil.
        Falls back to a default value if psutil is not available.
        
        Returns:
            Available memory in megabytes
        """
        try:
            import psutil
            return psutil.virtual_memory().available / 1024 / 1024
        except ImportError:
            logger.warning("psutil not available, using default memory estimate")
            return 2000.0  # Default to 2GB if can't determine
        except Exception as e:
            logger.warning(f"Error getting memory info: {e}")
            return 2000.0
    
    async def benchmark_methods(self, screenshot: np.ndarray, query: str) -> Dict[str, Any]:
        """Benchmark sliding window vs full analysis performance.
        
        Compares the performance, speed, and memory usage of both analysis
        methods to help optimize system configuration.
        
        Args:
            screenshot: Test image for benchmarking
            query: Test query for analysis
            
        Returns:
            Comprehensive benchmark results including timing and memory usage
            
        Example:
            >>> benchmark = await jarvis_vision.benchmark_methods(screenshot, "Analyze screen")
            >>> print(f"Speedup: {benchmark['comparison']['speedup']:.1f}x")
        """
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
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_mb = 0.0
        
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
    """Demonstrate various JARVIS vision use cases.
    
    Provides comprehensive examples of different vision commands and analysis
    scenarios, including element finding, region monitoring, and benchmarking.
    
    This function serves as both documentation and testing for the JARVIS
    vision system capabilities.
    
    Raises:
        RuntimeError: If examples fail to execute properly
    """
    
    # Initialize JARVIS vision
    api_key = os.getenv("ANTHROPIC_API_KEY", "mock_api_key")
    jarvis_vision = JarvisSlidingWindowVision(api_key)
    
    # Create test screenshot
    screenshot = create_test_screenshot()
    
    print("=" * 60)
    print("JARVIS VISION - SLIDING WINDOW EXAMPLES")
    print("=" * 60)
    
    try:
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
        
    except Exception as e:
        logger.error(f"Example execution failed: {str(e)}")
        raise RuntimeError(f"Failed to run examples: {str(e)}")

def create_test_screenshot() -> np.ndarray:
    """Create a realistic test screenshot for demonstration purposes.
    
    Generates a synthetic screenshot with typical desktop elements including
    application windows, notification areas, and UI components for testing
    the vision analysis system.
    
    Returns:
        Synthetic screenshot as numpy array (1080, 1920, 3)
        
    Example:
        >>> screenshot = create_test_screenshot()
        >>> print(f"Screenshot shape: {screenshot.shape}")
        Screenshot shape: (1080, 1920, 3)
    """
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
    """Main execution block for running demonstration examples."""
    # Run examples
    asyncio.run(example_use_cases())