#!/usr/bin/env python3
"""
Test Vision-Guided UI Navigation
=================================

Test script to verify vision navigator can connect to displays.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def test_vision_navigation():
    """Test vision navigation system"""
    print("\n" + "="*70)
    print("🎯 Vision-Guided UI Navigation Test")
    print("="*70)
    
    # Step 1: Initialize navigator
    print("\n📋 Step 1: Initializing Vision Navigator...")
    from display.vision_ui_navigator import get_vision_navigator
    
    navigator = get_vision_navigator()
    status = navigator.get_status()
    
    print(f"   ✅ Navigator initialized")
    print(f"   Vision connected: {status['vision_connected']}")
    print(f"   Config loaded: {status['config_loaded']}")
    
    # Step 2: Connect vision analyzer
    if not navigator.vision_analyzer:
        print("\n📋 Step 2: Connecting Claude Vision analyzer...")
        try:
            import os
            from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                analyzer = ClaudeVisionAnalyzer(api_key)
                navigator.set_vision_analyzer(analyzer)
                print(f"   ✅ Vision analyzer connected")
            else:
                print(f"   ⚠️  No ANTHROPIC_API_KEY found")
        except Exception as e:
            print(f"   ⚠️  Could not connect vision analyzer: {e}")
    else:
        print("\n📋 Step 2: Vision analyzer already connected ✅")
    
    # Step 3: Test connection
    print("\n📋 Step 3: Testing connection to Living Room TV...")
    print("   ⚠️  This will attempt to ACTUALLY connect your TV!")
    print("   Make sure your TV is ON and available.")
    print()
    
    input("Press ENTER to continue or Ctrl+C to cancel...")
    
    print("\n🚀 Starting vision-guided navigation...")
    result = await navigator.connect_to_display("Living Room TV")
    
    print("\n📊 Result:")
    print(f"   Success: {'✅' if result.success else '❌'} {result.success}")
    print(f"   Message: {result.message}")
    print(f"   Duration: {result.duration:.2f}s")
    print(f"   Steps completed: {result.steps_completed}")
    
    if result.error_details:
        print(f"\n❌ Error details:")
        for key, value in result.error_details.items():
            print(f"   {key}: {value}")
    
    # Step 4: Get statistics
    print("\n📊 Navigation Statistics:")
    stats = navigator.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\n" + "="*70)
    print("✅ Test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(test_vision_navigation())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
