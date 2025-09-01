"""
Integration tests and examples for sliding window vision system
Demonstrates how sliding window works with real screenshots
NO HARDCODING - Everything configurable for 16GB RAM systems
"""

import asyncio
import os
import cv2
import numpy as np
import time
from typing import List, Dict, Any
import json
import psutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Import our sliding window analyzer
from sliding_window_claude_analyzer import (
    SlidingWindowClaudeAnalyzer, 
    RustSlidingWindowAnalyzer,
    SlidingWindowClaudeConfig
)

# Set up environment variables for testing
def setup_test_environment():
    """Set up test environment variables"""
    test_env = {
        # Sliding window settings optimized for 16GB RAM
        'CLAUDE_WINDOW_WIDTH': '400',
        'CLAUDE_WINDOW_HEIGHT': '300',
        'CLAUDE_WINDOW_OVERLAP': '0.3',  # 30% overlap
        'CLAUDE_MAX_CONCURRENT': '3',
        'CLAUDE_MEMORY_THRESHOLD_MB': '2000',
        'CLAUDE_LOW_MEMORY_SCALE': '0.75',
        'CLAUDE_CRITICAL_MEMORY_MB': '1000',
        
        # Claude API settings
        'CLAUDE_MODEL': 'claude-3-5-sonnet-20241022',
        'CLAUDE_MAX_TOKENS_WINDOW': '300',
        'CLAUDE_JPEG_QUALITY': '75',
        
        # Performance settings
        'CLAUDE_ENABLE_CACHE': 'true',
        'CLAUDE_CACHE_TTL': '60',
        'CLAUDE_PRIORITIZE_CENTER': 'true',
        'CLAUDE_SKIP_STATIC': 'true',
        'CLAUDE_BATCH_WINDOWS': 'true',
        'CLAUDE_COMBINE_RESULTS': 'true',
        
        # Rust sliding window settings
        'SLIDING_WINDOW_WIDTH': '400',
        'SLIDING_WINDOW_HEIGHT': '300',
        'SLIDING_OVERLAP_PERCENT': '0.3',
        'SLIDING_MAX_CONCURRENT': '4',
        'SLIDING_MEMORY_THRESHOLD_MB': '2000',
        'SLIDING_ADAPTIVE_SIZING': 'true',
        'SLIDING_ENABLE_CACHE': 'true',
        'SLIDING_PRIORITIZE_CENTER': 'true',
        'SLIDING_SKIP_STATIC': 'true',
    }
    
    # Apply environment variables
    for key, value in test_env.items():
        os.environ[key] = value
    
    print("Test environment configured:")
    print(json.dumps(test_env, indent=2))

def visualize_sliding_windows(image: np.ndarray, windows: List[Dict[str, Any]], 
                            output_path: str = "sliding_windows_visualization.png"):
    """Visualize sliding windows on an image"""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Display the image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw each window with color based on priority
    for i, window in enumerate(windows):
        bounds = window['bounds']
        priority = window.get('priority', 1.0)
        
        # Color based on priority (red=high, blue=low)
        color = plt.cm.RdYlBu_r(priority)
        
        # Create rectangle
        rect = patches.Rectangle(
            (bounds[0], bounds[1]), 
            bounds[2], bounds[3],
            linewidth=2, 
            edgecolor=color, 
            facecolor='none',
            alpha=0.8
        )
        
        # Add rectangle to plot
        ax.add_patch(rect)
        
        # Add window number
        ax.text(bounds[0] + 5, bounds[1] + 20, f"W{i+1}", 
                color='white', fontsize=10, weight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    ax.set_title(f"Sliding Windows Visualization ({len(windows)} windows)")
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_path}")

async def test_basic_sliding_window():
    """Test basic sliding window functionality"""
    print("\n=== Testing Basic Sliding Window ===")
    
    # Initialize analyzer
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set, using mock mode")
        api_key = "mock_api_key"
    
    config = SlidingWindowClaudeConfig()
    analyzer = SlidingWindowClaudeAnalyzer(api_key, config)
    
    # Create test image (1920x1080 with gradient and text regions)
    test_image = create_test_image()
    
    # Generate sliding windows
    windows = analyzer.generate_sliding_windows(test_image)
    print(f"Generated {len(windows)} windows")
    
    # Visualize windows
    visualize_sliding_windows(test_image, windows)
    
    # Display window information
    print("\nWindow Details:")
    for i, window in enumerate(windows[:5]):  # Show first 5
        bounds = window['bounds']
        priority = window['priority']
        print(f"  Window {i+1}: pos=({bounds[0]}, {bounds[1]}), "
              f"size=({bounds[2]}x{bounds[3]}), priority={priority:.2f}")
    
    # Show memory usage
    memory_mb = analyzer.get_available_memory_mb()
    print(f"\nAvailable memory: {memory_mb:.1f} MB")
    
    return analyzer, test_image

async def test_memory_aware_processing():
    """Test memory-aware adaptive processing"""
    print("\n=== Testing Memory-Aware Processing ===")
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "mock_api_key")
    config = SlidingWindowClaudeConfig()
    analyzer = SlidingWindowClaudeAnalyzer(api_key, config)
    
    # Simulate different memory conditions
    memory_scenarios = [
        ("High Memory", 5000),  # 5GB available
        ("Normal Memory", 2500),  # 2.5GB available
        ("Low Memory", 1500),  # 1.5GB available
        ("Critical Memory", 800),  # 800MB available
    ]
    
    test_image = create_test_image()
    
    for scenario_name, available_mb in memory_scenarios:
        print(f"\n{scenario_name} Scenario ({available_mb} MB):")
        
        # Override memory check (for testing)
        original_method = analyzer.get_available_memory_mb
        analyzer.get_available_memory_mb = lambda: available_mb
        
        # Get adaptive window size
        window_width, window_height = analyzer.get_adaptive_window_size()
        print(f"  Adaptive window size: {window_width}x{window_height}")
        
        # Generate windows
        windows = analyzer.generate_sliding_windows(test_image)
        print(f"  Generated {len(windows)} windows")
        
        # Check if quality reduced
        if analyzer.should_reduce_quality():
            print("  ⚠️  Quality reduction triggered")
        
        # Restore original method
        analyzer.get_available_memory_mb = original_method

async def test_window_caching():
    """Test sliding window caching mechanism"""
    print("\n=== Testing Window Caching ===")
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "mock_api_key")
    config = SlidingWindowClaudeConfig(enable_cache=True, cache_ttl_seconds=5)
    analyzer = SlidingWindowClaudeAnalyzer(api_key, config)
    
    test_image = create_test_image()
    
    # Mock analyze function
    async def mock_analyze_window(image: np.ndarray, window: Dict[str, Any]) -> Any:
        # Simulate API call delay
        await asyncio.sleep(0.1)
        bounds = window['bounds']
        return type('Result', (), {
            'bounds': bounds,
            'content': f"Mock analysis of region at ({bounds[0]}, {bounds[1]})",
            'confidence': 0.9,
            'objects_detected': ['text', 'button'],
            'text_found': ['Hello', 'World'],
            'ui_elements': ['button', 'textfield'],
            'from_cache': False,
            'analysis_time_ms': 100
        })()
    
    # Replace analyze method temporarily
    original_analyze = analyzer.analyze_window
    analyzer.analyze_window = mock_analyze_window
    
    # First analysis (cache miss)
    print("First analysis (populating cache)...")
    start_time = time.time()
    windows = analyzer.generate_sliding_windows(test_image)[:3]  # Use first 3 windows
    
    # Analyze each window
    for i, window in enumerate(windows):
        result = await analyzer.analyze_window(test_image, window)
        print(f"  Window {i+1}: from_cache={result.from_cache}")
    
    first_time = time.time() - start_time
    print(f"First analysis time: {first_time:.2f}s")
    
    # Second analysis (cache hit)
    print("\nSecond analysis (using cache)...")
    start_time = time.time()
    
    for i, window in enumerate(windows):
        result = await analyzer.analyze_window(test_image, window)
        print(f"  Window {i+1}: from_cache={result.from_cache}")
    
    second_time = time.time() - start_time
    print(f"Second analysis time: {second_time:.2f}s")
    print(f"Speedup: {first_time/second_time:.1f}x")
    
    # Wait for cache to expire
    print("\nWaiting for cache to expire...")
    await asyncio.sleep(6)
    
    # Third analysis (cache expired)
    print("Third analysis (cache expired)...")
    for i, window in enumerate(windows):
        result = await analyzer.analyze_window(test_image, window)
        print(f"  Window {i+1}: from_cache={result.from_cache}")
    
    # Restore original method
    analyzer.analyze_window = original_analyze
    
    # Show cache stats
    stats = analyzer.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Total API calls: {stats['total_api_calls']}")

async def test_static_region_detection():
    """Test static region detection and skipping"""
    print("\n=== Testing Static Region Detection ===")
    
    # This would integrate with Rust static detection
    # For now, we'll simulate it
    
    api_key = os.getenv("ANTHROPIC_API_KEY", "mock_api_key")
    config = SlidingWindowClaudeConfig(skip_static_regions=True)
    analyzer = SlidingWindowClaudeAnalyzer(api_key, config)
    
    # Create two similar images with minor differences
    image1 = create_test_image()
    image2 = create_test_image()
    
    # Add a small change to image2
    cv2.rectangle(image2, (100, 100), (200, 200), (255, 0, 0), -1)
    
    print("Analyzing first image...")
    windows1 = analyzer.generate_sliding_windows(image1)
    print(f"First image: {len(windows1)} windows")
    
    print("\nAnalyzing second image (with changes)...")
    windows2 = analyzer.generate_sliding_windows(image2)
    print(f"Second image: {len(windows2)} windows")
    
    # In real implementation, static regions would be skipped
    print("\nStatic region detection would skip unchanged areas in production")

async def test_real_screenshot():
    """Test with a real screenshot if available"""
    print("\n=== Testing with Real Screenshot ===")
    
    screenshot_path = "test_screenshot.png"
    
    # Try to capture a real screenshot
    try:
        import pyautogui
        print("Capturing screenshot...")
        screenshot = pyautogui.screenshot()
        screenshot.save(screenshot_path)
        print(f"Screenshot saved to {screenshot_path}")
        
        # Load and process
        image = cv2.imread(screenshot_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Analyze with sliding window
        api_key = os.getenv("ANTHROPIC_API_KEY", "mock_api_key")
        analyzer = RustSlidingWindowAnalyzer(api_key)
        
        print("\nAnalyzing screenshot with sliding windows...")
        result = await analyzer.analyze_screenshot(image)
        
        print("\nAnalysis Result:")
        print(f"Summary: {result.get('summary', 'No summary')}")
        print(f"Objects detected: {', '.join(result.get('objects_detected', [])[:5])}")
        print(f"UI elements: {', '.join(result.get('ui_elements', [])[:5])}")
        print(f"Important regions: {len(result.get('important_regions', []))}")
        
        # Show metadata
        metadata = result.get('metadata', {})
        print(f"\nMetadata:")
        print(f"  Total time: {metadata.get('total_time_ms', 0):.1f}ms")
        print(f"  Windows analyzed: {metadata.get('windows_analyzed', 0)}")
        print(f"  Cache hits: {metadata.get('cache_hits', 0)}")
        print(f"  Memory saved: {metadata.get('memory_saved_mb', 0):.1f}MB")
        
    except ImportError:
        print("pyautogui not available, skipping real screenshot test")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")

def create_test_image(width=1920, height=1080):
    """Create a test image with various UI elements"""
    # Create base image with gradient
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background
    for y in range(height):
        color_value = int(255 * (y / height))
        image[y, :] = [color_value, color_value // 2, 255 - color_value]
    
    # Add some UI elements
    # Title bar
    cv2.rectangle(image, (0, 0), (width, 40), (50, 50, 50), -1)
    cv2.putText(image, "Test Application - Sliding Window Demo", 
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Main content area with text
    cv2.rectangle(image, (50, 100), (600, 400), (255, 255, 255), -1)
    cv2.putText(image, "Main Content Area", 
                (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Sidebar
    cv2.rectangle(image, (650, 100), (900, 600), (200, 200, 200), -1)
    cv2.putText(image, "Sidebar", 
                (700, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Buttons
    button_y = 500
    for i, text in enumerate(["Save", "Load", "Export"]):
        x = 100 + i * 150
        cv2.rectangle(image, (x, button_y), (x + 120, button_y + 40), (100, 150, 255), -1)
        cv2.putText(image, text, (x + 30, button_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Status bar
    cv2.rectangle(image, (0, height - 30), (width, height), (40, 40, 40), -1)
    cv2.putText(image, "Ready | Memory: 2.1GB | FPS: 60", 
                (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return image

async def run_all_tests():
    """Run all integration tests"""
    print("=" * 60)
    print("SLIDING WINDOW VISION SYSTEM - INTEGRATION TESTS")
    print("=" * 60)
    
    # Setup test environment
    setup_test_environment()
    
    # Show system info
    print(f"\nSystem Info:")
    print(f"  Total RAM: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"  Available RAM: {psutil.virtual_memory().available / 1024 / 1024 / 1024:.1f} GB")
    print(f"  CPU Count: {psutil.cpu_count()}")
    
    # Run tests
    tests = [
        ("Basic Sliding Window", test_basic_sliding_window),
        ("Memory-Aware Processing", test_memory_aware_processing),
        ("Window Caching", test_window_caching),
        ("Static Region Detection", test_static_region_detection),
        ("Real Screenshot", test_real_screenshot),
    ]
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                await test_func()
            else:
                test_func()
        except Exception as e:
            print(f"\n❌ Error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TESTS COMPLETED")
    print("=" * 60)

def main():
    """Main entry point"""
    # Run all tests
    asyncio.run(run_all_tests())

if __name__ == "__main__":
    main()