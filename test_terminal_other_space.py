#!/usr/bin/env python3
"""
Test that asking about Terminal in "other window" only captures the other space
"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_other_space_capture():
    from backend.vision.multi_space_window_detector import MultiSpaceWindowDetector
    from backend.vision.multi_space_intelligence import MultiSpaceIntelligenceExtension
    from backend.vision.multi_space_capture_engine import MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality

    print("\n=== Testing 'Other Window' Query Handling ===\n")

    # 1. Detect current state
    detector = MultiSpaceWindowDetector()
    window_data = detector.get_all_windows_across_spaces()
    current_space = window_data.get("current_space", {}).get("id", 1)

    print(f"Current space: {current_space}")
    print(f"Total spaces: {len(window_data.get('spaces', []))}")

    # 2. Test query analysis
    msi = MultiSpaceIntelligenceExtension()
    test_query = "Can you see my terminal in the other window space?"

    query_analysis = msi.process_multi_space_query(test_query, window_data)
    intent = query_analysis.get("intent")

    print(f"\nQuery: '{test_query}'")
    if intent:
        print(f"Detected target app: {intent.target_app if hasattr(intent, 'target_app') else 'None'}")
        print(f"Target space: {intent.target_space if hasattr(intent, 'target_space') else 'None'}")

    # 3. Test space determination (simulating what Vision Intelligence does)
    from backend.api.pure_vision_intelligence import PureVisionIntelligence

    # Create a mock vision instance just for testing the logic
    class MockVision:
        def _determine_spaces_to_capture(self, query_analysis, window_data):
            # Copy the exact logic from PureVisionIntelligence
            intent = query_analysis.get("intent")
            spaces = set()
            current_space = window_data.get("current_space", {}).get("id", 1)

            # Check if query is specifically asking about "other" spaces
            query = query_analysis.get("query", "").lower()
            is_other_space_query = any(phrase in query for phrase in [
                "other window", "other space", "another window", "another space",
                "different window", "different space"
            ])

            # Add spaces with relevant apps
            if intent and hasattr(intent, "target_app") and intent.target_app:
                for window in window_data.get("windows", []):
                    if (hasattr(window, "app_name") and
                        intent.target_app.lower() in window.app_name.lower() and
                        hasattr(window, "space_id")):
                        spaces.add(window.space_id)
                        print(f"  Found {intent.target_app} in space {window.space_id}")

            # Only include current space based on new logic
            if not spaces:
                spaces.add(current_space)
                print(f"  No specific spaces found, using current space {current_space}")
            elif not is_other_space_query:
                if intent and hasattr(intent, "target_app") and intent.target_app:
                    for window in window_data.get("windows", []):
                        if (hasattr(window, "app_name") and
                            intent.target_app.lower() in window.app_name.lower() and
                            hasattr(window, "space_id") and
                            window.space_id == current_space):
                            spaces.add(current_space)
                            print(f"  {intent.target_app} also found in current space {current_space}")
                            break
                else:
                    spaces.add(current_space)
            else:
                print(f"  Query is about OTHER spaces, excluding current space {current_space}")

            return sorted(list(spaces))

    mock_vision = MockVision()
    spaces_to_capture = mock_vision._determine_spaces_to_capture(query_analysis, window_data)

    print(f"\nSpaces to capture: {spaces_to_capture}")

    if current_space in spaces_to_capture and len(spaces_to_capture) > 1:
        print("⚠️ WARNING: Current space is being captured along with other spaces!")
        print("   This may cause Claude to describe the wrong window")
    elif current_space not in spaces_to_capture:
        print("✅ SUCCESS: Current space is NOT being captured")
        print("   Only capturing the other space(s) with Terminal")
    else:
        print("ℹ️ Only capturing current space (Terminal might be in current space)")

    # 4. Test actual capture
    print("\n=== Testing Actual Capture ===")

    engine = MultiSpaceCaptureEngine()
    request = SpaceCaptureRequest(
        space_ids=spaces_to_capture,
        quality=CaptureQuality.OPTIMIZED,
        use_cache=False,
        reason=test_query[:50]
    )

    result = await engine.capture_all_spaces(request)

    if result.success:
        print(f"Captured {len(result.screenshots)} screenshot(s)")
        for space_id in result.screenshots:
            print(f"  - Space {space_id} {'(current)' if space_id == current_space else '(other)'}")
    else:
        print(f"Capture failed: {result.errors}")

    print("\n=== Test Complete ===\n")

if __name__ == "__main__":
    asyncio.run(test_other_space_capture())