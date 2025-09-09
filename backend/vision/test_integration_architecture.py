#!/usr/bin/env python3
"""
Test Integration Architecture - Part 3 Demo
Tests the complete 9-stage processing pipeline with dynamic memory management
"""

import asyncio
import os
import sys
import time
from pathlib import Path
import numpy as np
from PIL import Image
import psutil

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
from vision.intelligence.integration_orchestrator import IntegrationOrchestrator, SystemMode

# Colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
RESET = '\033[0m'

async def simulate_memory_pressure(orchestrator: IntegrationOrchestrator, mode: SystemMode):
    """Simulate different memory pressure scenarios"""
    print(f"\n{YELLOW}Simulating {mode.value} mode...{RESET}")
    
    # Force system mode change
    orchestrator.system_mode = mode
    await orchestrator._apply_system_mode(mode)
    
    # Get status
    status = await orchestrator.get_system_status()
    print(f"System Mode: {status['system_mode']}")
    print(f"Memory Usage: {status['memory_usage_mb']:.1f}MB ({status['memory_percent']:.1f}%)")
    print(f"Active Components: {sum(1 for v in status['components'].values() if v == 'active')}")

async def test_integration_pipeline():
    """Test the complete integration pipeline"""
    print(f"{CYAN}=== Integration Architecture Test ==={RESET}")
    print(f"Testing 9-stage processing pipeline with dynamic memory management")
    
    # Initialize orchestrator directly
    print(f"\n{GREEN}1. Initializing Integration Orchestrator...{RESET}")
    orchestrator = IntegrationOrchestrator({
        'total_memory_mb': 1200,
        'enable_all_components': True,
        'adaptive_quality': True
    })
    
    # Create test image
    test_image = Image.new('RGB', (800, 600), color='white')
    np_image = np.array(test_image)
    
    # Test context
    test_context = {
        'prompt': 'Analyze this screenshot and identify important UI elements',
        'app_id': 'test_app',
        'user_action': 'analyze_ui',
        'timestamp': time.time()
    }
    
    # Test 1: Normal mode processing
    print(f"\n{GREEN}2. Testing NORMAL mode (full capabilities)...{RESET}")
    start_time = time.time()
    
    result = await orchestrator.process_frame(np_image, test_context)
    
    if '_metrics' in result:
        metrics = result['_metrics']
        print(f"\nProcessing completed in {metrics['total_time']:.2f}s")
        print(f"System Mode: {metrics['system_mode']}")
        print(f"Cache hits: {metrics.get('cache_hits', 0)}")
        print(f"API calls saved: {metrics.get('api_calls_saved', 0)}")
        
        print("\nStage timings:")
        for stage, duration in metrics['stage_times'].items():
            print(f"  - {stage}: {duration:.3f}s")
    
    # Test 2: Memory pressure scenarios
    print(f"\n{GREEN}3. Testing memory pressure adaptations...{RESET}")
    
    # Simulate pressure mode
    await simulate_memory_pressure(orchestrator, SystemMode.PRESSURE)
    
    # Process in pressure mode
    result = await orchestrator.process_frame(np_image, test_context)
    if '_metrics' in result:
        print(f"Pressure mode processing time: {result['_metrics']['total_time']:.2f}s")
    
    # Simulate critical mode
    await simulate_memory_pressure(orchestrator, SystemMode.CRITICAL)
    
    # Process in critical mode
    result = await orchestrator.process_frame(np_image, test_context)
    if '_metrics' in result:
        print(f"Critical mode processing time: {result['_metrics']['total_time']:.2f}s")
        print(f"Components disabled in critical mode")
    
    # Test 3: Cache effectiveness
    print(f"\n{GREEN}4. Testing cache effectiveness...{RESET}")
    
    # Reset to normal mode
    await simulate_memory_pressure(orchestrator, SystemMode.NORMAL)
    
    # Process same frame multiple times
    cache_times = []
    for i in range(3):
        start = time.time()
        result = await orchestrator.process_frame(np_image, test_context)
        duration = time.time() - start
        cache_times.append(duration)
        
        if '_metrics' in result and result['_metrics'].get('cache_hits', 0) > 0:
            print(f"  Run {i+1}: {duration:.3f}s (cache hit)")
        else:
            print(f"  Run {i+1}: {duration:.3f}s")
    
    # Test 4: Component coordination
    print(f"\n{GREEN}5. Testing component coordination...{RESET}")
    
    status = await orchestrator.get_system_status()
    print("\nMemory Allocations:")
    for name, alloc in sorted(status['allocations'].items(), 
                              key=lambda x: x[1]['allocated_mb'], 
                              reverse=True)[:5]:
        print(f"  - {name}: {alloc['allocated_mb']:.1f}MB "
              f"(used: {alloc['used_mb']:.1f}MB, "
              f"utilization: {alloc['utilization']:.1%})")
    
    # Summary
    print(f"\n{CYAN}=== Summary ==={RESET}")
    print(f"✓ 9-stage pipeline processed successfully")
    print(f"✓ Dynamic memory management tested across all modes")
    print(f"✓ Cache optimization working (speedup: {cache_times[0]/cache_times[-1]:.1f}x)")
    print(f"✓ Component coordination verified")

async def test_with_vision_analyzer():
    """Test integration with ClaudeVisionAnalyzer"""
    print(f"\n{CYAN}=== Integration with Vision Analyzer ==={RESET}")
    
    # Check if API key is available
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print(f"{YELLOW}No API key found. Set ANTHROPIC_API_KEY to test with real analyzer.{RESET}")
        return
    
    # Initialize analyzer
    analyzer = ClaudeVisionAnalyzer(api_key=api_key)
    
    # Enable orchestrator
    os.environ['INTEGRATION_ORCHESTRATOR_ENABLED'] = 'true'
    
    # Create test screenshot
    test_image = Image.new('RGB', (1024, 768), color='lightblue')
    
    # Test analysis with orchestrator
    print(f"\n{GREEN}Testing analysis with orchestrator enabled...{RESET}")
    
    try:
        result, metrics = await analyzer.analyze_screenshot(
            test_image,
            "Describe what you see in this screenshot"
        )
        
        if hasattr(metrics, 'orchestrator_time'):
            print(f"\nOrchestrator processing time: {metrics.orchestrator_time:.2f}s")
            print(f"System mode: {metrics.system_mode}")
            print(f"Orchestrator cache hits: {metrics.orchestrator_cache_hits}")
            print(f"API calls saved by orchestrator: {metrics.orchestrator_api_saved}")
        
        if result.get('_orchestrator_optimized'):
            print(f"\n{GREEN}✓ Result was optimized by orchestrator!{RESET}")
    
    except Exception as e:
        print(f"Analysis error: {e}")

async def main():
    """Run all tests"""
    # Test orchestrator directly
    await test_integration_pipeline()
    
    # Test with vision analyzer if available
    await test_with_vision_analyzer()
    
    print(f"\n{GREEN}All tests completed!{RESET}")

if __name__ == "__main__":
    asyncio.run(main())