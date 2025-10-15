#!/usr/bin/env python3
"""
Test script to verify Terminal capture from other spaces
"""

import asyncio
import logging
import numpy as np
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_terminal_capture():
    """Test that Terminal is properly captured from Space 2"""

    print("\n=== Testing Multi-Space Terminal Capture ===\n")

    # Import the necessary modules
    from backend.vision.multi_space_window_detector import MultiSpaceWindowDetector
    from backend.vision.cg_window_capture import CGWindowCapture
    from backend.vision.multi_space_capture_engine import MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality

    # 1. Detect all windows across spaces
    print("1. Detecting windows across all spaces...")
    detector = MultiSpaceWindowDetector()
    window_data = detector.get_all_windows_across_spaces()

    current_space = window_data.get("current_space", {}).get("id", 1)
    print(f"   Current space: {current_space}")
    print(f"   Total spaces: {len(window_data.get('spaces', []))}")

    # 2. Find Terminal in other spaces
    print("\n2. Looking for Terminal in other spaces...")
    terminal_found = False
    terminal_space = None
    terminal_window_id = None

    for window in window_data.get("windows", []):
        if hasattr(window, "app_name") and "terminal" in window.app_name.lower():
            space_id = window.space_id if hasattr(window, "space_id") else window.get("space", 1)
            print(f"   Found Terminal in space {space_id}: {window.window_title if hasattr(window, 'window_title') else window.get('title', 'Unknown')}")
            if space_id != current_space:
                terminal_found = True
                terminal_space = space_id
                # Get window ID for CG capture
                terminal_window_id = CGWindowCapture.find_window_by_name("Terminal")
                print(f"   Terminal window ID: {terminal_window_id}")
                break

    if not terminal_found:
        print("   ❌ No Terminal found in other spaces")
        return

    print(f"   ✅ Terminal found in space {terminal_space} (not current space)")

    # 3. Test CG capture directly
    print("\n3. Testing Core Graphics capture...")
    if terminal_window_id:
        screenshot = CGWindowCapture.capture_window_by_id(terminal_window_id)
        if screenshot is not None:
            print(f"   ✅ CG capture successful: {screenshot.shape}")
            # Save for inspection
            Image.fromarray(screenshot).save("/tmp/terminal_cg_test.png")
            print("   Saved to /tmp/terminal_cg_test.png")
        else:
            print("   ❌ CG capture failed")

    # 4. Test multi-space capture engine
    print("\n4. Testing Multi-Space Capture Engine...")
    engine = MultiSpaceCaptureEngine()

    request = SpaceCaptureRequest(
        space_ids=[terminal_space],
        quality=CaptureQuality.OPTIMIZED,
        use_cache=False,  # Don't use cache for testing
        reason="Testing Terminal capture"
    )

    result = await engine.capture_all_spaces(request)

    if result.success and terminal_space in result.screenshots:
        screenshot = result.screenshots[terminal_space]
        print(f"   ✅ Multi-space capture successful: {screenshot.shape}")
        # Save for inspection
        Image.fromarray(screenshot).save("/tmp/terminal_multispace_test.png")
        print("   Saved to /tmp/terminal_multispace_test.png")
    else:
        print(f"   ❌ Multi-space capture failed: {result.errors}")

    # 5. Test with Vision Intelligence
    print("\n5. Testing with Vision Intelligence...")
    from backend.api.pure_vision_intelligence import PureVisionIntelligence

    vision = PureVisionIntelligence()

    # Test query that should trigger multi-space capture
    query = "Can you see my terminal in the other window space?"
    print(f"   Query: '{query}'")

    # Mock screenshot for current view (not used in multi-space)
    mock_screenshot = np.zeros((100, 100, 3), dtype=np.uint8)

    response = await vision.understand_and_respond(mock_screenshot, query)
    print(f"\n   Response: {response[:500]}...")

    # Check if Terminal was mentioned
    if "terminal" in response.lower():
        if "browser" in response.lower() or "console" in response.lower() or "developer" in response.lower():
            print("\n   ⚠️ ISSUE: Response mentions browser/console instead of Terminal app")
        else:
            print("\n   ✅ Response correctly identifies Terminal")
    else:
        print("\n   ❌ Response doesn't mention Terminal")

    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_terminal_capture())