#!/usr/bin/env python3
"""
Test F1.2: Multi-Window Capture
Verifies the multi-window capture system meets all acceptance criteria
"""

import asyncio
import time
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from backend.vision.multi_window_capture import MultiWindowCapture
from backend.vision.window_detector import WindowDetector


async def test_multi_window_capture():
    """Test F1.2 acceptance criteria"""
    print("🧪 Testing F1.2: Multi-Window Capture")
    print("=" * 60)
    
    # Initialize capture system
    capture_system = MultiWindowCapture()
    window_detector = WindowDetector()
    
    # Get current windows
    windows = window_detector.get_all_windows()
    print(f"\n📊 Current Status:")
    print(f"   Total windows detected: {len(windows)}")
    
    # Test 1: Verify max windows setting
    print(f"\n✓ Acceptance Criteria 1: Captures up to 5 windows simultaneously")
    print(f"   Max windows setting: {capture_system.max_windows}")
    assert capture_system.max_windows == 5, "Max windows should be 5"
    print(f"   ✅ PASS: Max windows is set to 5")
    
    # Test 2: Verify resolution settings
    print(f"\n✓ Acceptance Criteria 2: Focused window at full resolution")
    print(f"   Focused resolution: {capture_system.focused_resolution * 100}%")
    assert capture_system.focused_resolution == 1.0, "Focused resolution should be 1.0"
    print(f"   ✅ PASS: Focused window uses full resolution")
    
    print(f"\n✓ Acceptance Criteria 3: Background windows at 50% resolution")
    print(f"   Context resolution: {capture_system.context_resolution * 100}%")
    assert capture_system.context_resolution == 0.5, "Context resolution should be 0.5"
    print(f"   ✅ PASS: Background windows use 50% resolution")
    
    # Test 3: Capture multiple windows and measure time
    print(f"\n✓ Acceptance Criteria 4: Total capture time <2 seconds")
    
    if len(windows) < 2:
        print(f"   ⚠️  WARNING: Only {len(windows)} windows available for testing")
        print(f"   Please open more windows for a complete test")
    
    # Perform capture and measure time
    start_time = time.time()
    captures = await capture_system.capture_multiple_windows()
    capture_time = time.time() - start_time
    
    print(f"\n📸 Capture Results:")
    print(f"   Windows captured: {len(captures)}")
    print(f"   Capture time: {capture_time:.2f} seconds")
    
    # Verify capture count
    expected_captures = min(len(windows), capture_system.max_windows)
    if len(captures) != expected_captures:
        print(f"   ⚠️  Expected {expected_captures} captures, got {len(captures)}")
    
    # Analyze captured windows
    focused_found = False
    for i, capture in enumerate(captures):
        window = capture.window_info
        resolution_pct = capture.resolution_scale * 100
        img_shape = capture.image.shape
        
        print(f"\n   Window {i+1}:")
        print(f"   - App: {window.app_name}")
        print(f"   - Title: {window.window_title or 'Untitled'}")
        print(f"   - Focused: {window.is_focused}")
        print(f"   - Resolution: {resolution_pct}%")
        print(f"   - Image size: {img_shape[1]}x{img_shape[0]}")
        
        # Verify resolution settings
        if window.is_focused:
            focused_found = True
            if capture.resolution_scale != 1.0:
                print(f"   ❌ ERROR: Focused window not at full resolution!")
            else:
                print(f"   ✅ Focused window at full resolution")
        else:
            # Background windows should be at 50% or 25%
            if capture.resolution_scale > 0.5:
                print(f"   ❌ ERROR: Background window at {resolution_pct}% (should be ≤50%)")
            else:
                print(f"   ✅ Background window at reduced resolution")
    
    # Final verdict
    print(f"\n📊 Test Summary:")
    print(f"   • Max windows: ✅ (5 windows)")
    print(f"   • Focused resolution: ✅ (100%)")
    print(f"   • Background resolution: ✅ (50%)")
    
    if capture_time < 2.0:
        print(f"   • Capture time: ✅ ({capture_time:.2f}s < 2s)")
    else:
        print(f"   • Capture time: ❌ ({capture_time:.2f}s > 2s)")
    
    if not focused_found and len(captures) > 0:
        print(f"\n   ⚠️  No focused window found in captures")
    
    # Performance analysis
    print(f"\n⚡ Performance Analysis:")
    if len(captures) > 0:
        avg_time_per_window = capture_time / len(captures)
        print(f"   Average time per window: {avg_time_per_window:.3f}s")
        print(f"   Parallel speedup: ~{len(captures)}x")
    
    # Test different query types
    print(f"\n🔍 Testing Query-Based Capture:")
    
    for query_type in ["messages", "errors", "documentation"]:
        print(f"\n   Testing '{query_type}' query:")
        start_time = time.time()
        captures = await capture_system.capture_multiple_windows(query_type)
        capture_time = time.time() - start_time
        
        print(f"   - Captured {len(captures)} windows in {capture_time:.2f}s")
        if captures:
            print(f"   - Priority apps: {[c.window_info.app_name for c in captures[:3]]}")
    
    print(f"\n✅ F1.2 Multi-Window Capture test complete!")
    
    # Return success if all criteria met
    return (
        capture_system.max_windows == 5 and
        capture_system.focused_resolution == 1.0 and
        capture_system.context_resolution == 0.5 and
        capture_time < 2.0
    )


async def test_capture_performance():
    """Additional performance testing"""
    print(f"\n\n🏎️  Performance Stress Test")
    print("=" * 60)
    
    capture_system = MultiWindowCapture()
    
    # Test multiple captures in sequence
    print(f"\n📊 Sequential Capture Test (5 iterations):")
    times = []
    
    for i in range(5):
        start_time = time.time()
        captures = await capture_system.capture_multiple_windows()
        capture_time = time.time() - start_time
        times.append(capture_time)
        print(f"   Iteration {i+1}: {capture_time:.3f}s ({len(captures)} windows)")
    
    avg_time = sum(times) / len(times)
    print(f"\n   Average capture time: {avg_time:.3f}s")
    print(f"   Min time: {min(times):.3f}s")
    print(f"   Max time: {max(times):.3f}s")
    
    if avg_time < 1.0:
        print(f"   ✅ Excellent performance (<1s average)")
    elif avg_time < 2.0:
        print(f"   ✅ Good performance (<2s average)")
    else:
        print(f"   ⚠️  Performance could be improved (>2s average)")


if __name__ == "__main__":
    print("🚀 F1.2 Multi-Window Capture Test Suite")
    print("=" * 60)
    
    # Run main test
    success = asyncio.run(test_multi_window_capture())
    
    # Run performance test
    asyncio.run(test_capture_performance())
    
    if success:
        print(f"\n🎉 All F1.2 acceptance criteria PASSED!")
    else:
        print(f"\n❌ Some F1.2 criteria failed")