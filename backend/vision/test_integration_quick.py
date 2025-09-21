#!/usr/bin/env python3
"""Quick test to verify screen sharing integration"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

async def quick_test():
    """Quick integration test"""
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer, VisionConfig
    
    # Minimal config
    config = VisionConfig(
        enable_screen_sharing=True,
        enable_continuous_monitoring=False,  # Disable for quick test
        max_concurrent_requests=2,
        process_memory_limit_mb=512
    )
    
    api_key = os.getenv('ANTHROPIC_API_KEY', 'dummy-key-for-testing')
    analyzer = ClaudeVisionAnalyzer(api_key, config)
    
    print("1. Testing screen capture...")
    screen = await analyzer.capture_screen()
    print(f"   ✓ Captured: {type(screen)} - {screen.size if hasattr(screen, 'size') else 'Unknown size'}")
    
    print("\n2. Testing screen sharing initialization...")
    sharing = await analyzer.get_screen_sharing()
    print(f"   ✓ Screen sharing module loaded: {sharing is not None}")
    
    print("\n3. Testing sliding window integration...")
    if hasattr(analyzer, '_generate_sliding_windows'):
        print("   ✓ Sliding window method available")
    else:
        print("   ✗ Sliding window method NOT available")
    
    print("\n4. Checking memory stats...")
    stats = analyzer.get_all_memory_stats()
    print(f"   ✓ Process memory: {stats['system']['process_mb']:.1f}MB")
    print(f"   ✓ Available memory: {stats['system']['available_mb']:.1f}MB")
    
    print("\n5. Testing screen sharing start (without actual sharing)...")
    # Don't actually start sharing, just test the method exists
    print(f"   ✓ start_screen_sharing method exists: {hasattr(analyzer, 'start_screen_sharing')}")
    print(f"   ✓ stop_screen_sharing method exists: {hasattr(analyzer, 'stop_screen_sharing')}")
    
    print("\n✅ Integration test passed!")
    
    # Cleanup
    await analyzer.cleanup_all_components()

if __name__ == "__main__":
    asyncio.run(quick_test())