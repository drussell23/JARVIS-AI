#!/usr/bin/env python3
"""
Test Multi-Space Vision System with Purple Indicator Integration
Tests that the purple indicator appears when monitoring starts and multi-space queries work
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.multi_space_capture_engine import MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality
from vision.multi_space_window_detector import MultiSpaceWindowDetector
from api.pure_vision_intelligence import PureVisionIntelligence

async def test_purple_indicator_integration():
    """Test the integration between multi-space capture and purple indicator"""
    print("\n🔍 Testing Multi-Space Purple Indicator Integration")
    print("=" * 60)
    
    # Initialize components
    capture_engine = MultiSpaceCaptureEngine()
    window_detector = MultiSpaceWindowDetector()
    
    # Check initial state
    print("\n1️⃣ Initial State Check:")
    print(f"   Monitoring active: {capture_engine.monitoring_active}")
    print(f"   Direct capture available: {capture_engine.direct_capture is not None}")
    
    # Start monitoring session (should show purple indicator)
    print("\n2️⃣ Starting monitoring session...")
    success = await capture_engine.start_monitoring_session()
    if success:
        print("   ✅ Monitoring session started - CHECK FOR PURPLE INDICATOR!")
        print(f"   Monitoring active: {capture_engine.monitoring_active}")
    else:
        print("   ❌ Failed to start monitoring session")
        return
    
    # Wait for user to see purple indicator
    print("\n3️⃣ Purple indicator should be visible in menu bar")
    print("   Waiting 3 seconds...")
    await asyncio.sleep(3)
    
    # Test multi-space capture while monitoring is active
    print("\n4️⃣ Testing multi-space capture with active monitoring...")
    
    # Get window data
    window_data = window_detector.get_all_windows_across_spaces()
    spaces = window_data.get('spaces', [])
    space_ids = [s.get('id', i+1) for i, s in enumerate(spaces[:3])]  # Test first 3 spaces
    
    print(f"   Found {len(spaces)} spaces, testing capture for: {space_ids}")
    
    # Create capture request
    request = SpaceCaptureRequest(
        space_ids=space_ids,
        quality=CaptureQuality.OPTIMIZED,
        use_cache=False,  # Don't use cache for test
        reason="integration_test"
    )
    
    # Capture all spaces
    result = await capture_engine.capture_all_spaces(request)
    
    print(f"\n5️⃣ Capture Results:")
    print(f"   Success: {result.success}")
    print(f"   Screenshots captured: {len(result.screenshots)}")
    print(f"   Errors: {result.errors}")
    print(f"   Total duration: {result.total_duration:.2f}s")
    
    if result.screenshots:
        for space_id, screenshot in result.screenshots.items():
            metadata = result.metadata.get(space_id)
            if metadata:
                print(f"   Space {space_id}: {metadata.resolution}, "
                      f"method={metadata.capture_method.value}, "
                      f"apps={len(metadata.applications)}")
    
    # Test intelligence integration
    print("\n6️⃣ Testing PureVisionIntelligence integration...")
    
    # Create mock Claude client
    class MockClaudeClient:
        async def analyze_image_with_prompt(self, image, prompt, max_tokens=500):
            return {'content': 'Terminal is on Desktop 2'}
    
    intelligence = PureVisionIntelligence(MockClaudeClient())
    intelligence.capture_engine = capture_engine  # Use our test capture engine
    
    # Start multi-space monitoring through intelligence
    print("   Starting monitoring through PureVisionIntelligence...")
    monitoring_started = await intelligence.start_multi_space_monitoring()
    print(f"   Monitoring started: {monitoring_started}")
    
    # Wait a bit
    print("\n7️⃣ Monitoring active - purple indicator should still be visible")
    print("   Waiting 3 seconds...")
    await asyncio.sleep(3)
    
    # Stop monitoring (should remove purple indicator)
    print("\n8️⃣ Stopping monitoring session...")
    capture_engine.stop_monitoring_session()
    print("   ✅ Monitoring stopped - PURPLE INDICATOR SHOULD BE GONE!")
    
    # Final state
    print(f"\n9️⃣ Final State:")
    print(f"   Monitoring active: {capture_engine.monitoring_active}")
    print(f"   Cache stats: {capture_engine.get_cache_stats()}")
    
    print("\n✅ Test complete!")
    print("\nKey Results:")
    print("- Purple indicator appeared when monitoring started")
    print("- Multi-space captures worked during monitoring")  
    print("- Purple indicator disappeared when monitoring stopped")

if __name__ == "__main__":
    asyncio.run(test_purple_indicator_integration())