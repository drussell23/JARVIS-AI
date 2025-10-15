"""
Live Testing Script for Feedback Learning & Command Safety
Run this while JARVIS is running to test the systems in real-time.

Usage:
    python -m backend.examples.live_feedback_test
"""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def test_feedback_loop_basic():
    """Test 1: Basic feedback loop functionality."""
    print("\n" + "="*80)
    print("TEST 1: Basic Feedback Loop")
    print("="*80 + "\n")

    from backend.core.learning.feedback_loop import (
        FeedbackLearningLoop,
        NotificationPattern,
        UserResponse,
    )

    # Create temporary feedback loop
    loop = FeedbackLearningLoop(storage_path=Path("/tmp/jarvis_test_feedback.json"))

    print("✓ Feedback loop initialized")
    print(f"  Storage: /tmp/jarvis_test_feedback.json")
    print(f"  Events in history: {len(loop.feedback_history)}")
    print()

    # Test recording engagement
    print("→ Recording 3 ENGAGED responses for TERMINAL_ERROR...")
    for i in range(3):
        await loop.record_feedback(
            pattern=NotificationPattern.TERMINAL_ERROR,
            response=UserResponse.ENGAGED,
            notification_text=f"Test terminal error #{i+1}",
            context={"window_type": "terminal"},
            time_to_respond=2.0,
        )
        print(f"  ✓ Event {i+1} recorded")

    print()
    print("→ Checking pattern stats...")
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
    print(f"  ✓ Total shown: {insights['total_shown']}")
    print(f"  ✓ Engagement rate: {insights['engagement_rate']:.0%}")
    print(f"  ✓ Is valued: {insights['is_valued']}")

    # Test should_show
    print()
    print("→ Testing notification filtering...")
    should_show, adjusted = loop.should_show_notification(
        pattern=NotificationPattern.TERMINAL_ERROR,
        base_importance=0.7,
    )
    print(f"  ✓ Should show: {should_show}")
    print(f"  ✓ Base importance: 0.70")
    print(f"  ✓ Adjusted importance: {adjusted:.2f}")

    if adjusted > 0.7:
        print(f"  ✓ SUCCESS: Importance boosted due to engagement!")

    print()
    return True


async def test_command_safety_basic():
    """Test 2: Basic command safety classification."""
    print("\n" + "="*80)
    print("TEST 2: Command Safety Classification")
    print("="*80 + "\n")

    from backend.system_control.command_safety import get_command_classifier, SafetyTier

    classifier = get_command_classifier()
    print("✓ Command classifier initialized")
    print()

    # Test commands
    test_cases = [
        ("ls -la", SafetyTier.GREEN, "Should be GREEN - read only"),
        ("git status", SafetyTier.GREEN, "Should be GREEN - read only"),
        ("npm install express", SafetyTier.YELLOW, "Should be YELLOW - installs package"),
        ("rm -rf /tmp/test", SafetyTier.RED, "Should be RED - destructive"),
        ("git push --force", SafetyTier.RED, "Should be RED - force flag"),
    ]

    all_passed = True
    for command, expected_tier, description in test_cases:
        result = classifier.classify(command)
        passed = result.tier == expected_tier

        symbol = "✓" if passed else "✗"
        color = "GREEN" if passed else "RED"

        print(f"{symbol} {command}")
        print(f"  Expected: {expected_tier.value.upper()}")
        print(f"  Got: {result.tier.value.upper()}")
        print(f"  {description}")

        if not passed:
            all_passed = False
            print(f"  ⚠️  FAILED!")

        print()

    if all_passed:
        print("✓ SUCCESS: All command classifications correct!")
    else:
        print("✗ FAILURE: Some classifications were wrong")

    print()
    return all_passed


async def test_terminal_intelligence():
    """Test 3: Terminal command intelligence."""
    print("\n" + "="*80)
    print("TEST 3: Terminal Command Intelligence")
    print("="*80 + "\n")

    from backend.vision.handlers.terminal_command_intelligence import (
        get_terminal_intelligence,
    )

    intel = get_terminal_intelligence()
    print("✓ Terminal intelligence initialized")
    print()

    # Simulate terminal with error
    terminal_ocr = """
    user@host:~/project $ python app.py
    Traceback (most recent call last):
      File "app.py", line 5, in <module>
        import requests
    ModuleNotFoundError: No module named 'requests'
    user@host:~/project $
    """

    print("→ Analyzing simulated terminal output...")
    print("  (Contains ModuleNotFoundError)")
    print()

    context = await intel.analyze_terminal_context(terminal_ocr)

    print("✓ Context extracted:")
    print(f"  Last command: {context.last_command}")
    print(f"  Errors found: {len(context.errors)}")
    for err in context.errors:
        print(f"    • {err[:60]}...")
    print()

    # Get suggestions
    print("→ Getting fix suggestions...")
    suggestions = await intel.suggest_fix_commands(context)

    print(f"✓ Found {len(suggestions)} suggestion(s)")

    if suggestions:
        for i, suggestion in enumerate(suggestions, 1):
            print(f"\n  Suggestion #{i}:")
            print(f"    Command: {suggestion.command}")
            print(f"    Purpose: {suggestion.purpose}")
            print(f"    Safety tier: {suggestion.safety_tier}")
            print(f"    Requires confirmation: {suggestion.requires_confirmation}")

        # Check if pip install requests is suggested
        has_pip_install = any('pip install requests' in s.command for s in suggestions)
        if has_pip_install:
            print("\n✓ SUCCESS: Correctly suggests 'pip install requests'!")
            return True
        else:
            print("\n✗ FAILURE: Did not suggest expected fix")
            return False
    else:
        print("\n✗ FAILURE: No suggestions generated")
        return False


async def test_persistence():
    """Test 4: Data persistence."""
    print("\n" + "="*80)
    print("TEST 4: Data Persistence")
    print("="*80 + "\n")

    from backend.core.learning.feedback_loop import (
        FeedbackLearningLoop,
        NotificationPattern,
        UserResponse,
    )

    storage_path = Path("/tmp/jarvis_persistence_test.json")

    # Delete existing file
    if storage_path.exists():
        storage_path.unlink()
        print("✓ Cleaned up old test data")

    print()
    print("→ Creating loop and recording feedback...")
    loop1 = FeedbackLearningLoop(storage_path=storage_path)

    await loop1.record_feedback(
        pattern=NotificationPattern.TERMINAL_ERROR,
        response=UserResponse.ENGAGED,
        notification_text="Persistence test",
        context={},
    )
    print("  ✓ Recorded 1 event")

    # Force save
    await loop1._save_to_disk()
    print("  ✓ Saved to disk")

    # Verify file exists
    if storage_path.exists():
        print(f"  ✓ File created: {storage_path}")
        file_size = storage_path.stat().st_size
        print(f"  ✓ File size: {file_size} bytes")
    else:
        print("  ✗ File not created!")
        return False

    print()
    print("→ Creating new loop instance (should load data)...")
    loop2 = FeedbackLearningLoop(storage_path=storage_path)

    if len(loop2.feedback_history) > 0:
        print(f"  ✓ Loaded {len(loop2.feedback_history)} event(s) from disk")
        print("  ✓ SUCCESS: Data persists!")
        return True
    else:
        print("  ✗ FAILURE: Data not loaded")
        return False


async def test_integration_with_vision():
    """Test 5: Integration with vision system."""
    print("\n" + "="*80)
    print("TEST 5: Integration Test (Simulated)")
    print("="*80 + "\n")

    from backend.core.learning.feedback_loop import (
        FeedbackLearningLoop,
        NotificationPattern,
        UserResponse,
    )
    from backend.vision.handlers.terminal_command_intelligence import (
        get_terminal_intelligence,
    )

    print("→ Simulating complete workflow...")
    print()

    # Initialize components
    loop = FeedbackLearningLoop(storage_path=Path("/tmp/jarvis_integration_test.json"))
    intel = get_terminal_intelligence()

    print("Step 1: Terminal error detected")
    terminal_ocr = """
    $ npm start
    Error: Cannot find module 'express'
    """

    context = await intel.analyze_terminal_context(terminal_ocr)
    print(f"  ✓ Extracted {len(context.errors)} error(s)")

    print()
    print("Step 2: Generate fix suggestions")
    suggestions = await intel.suggest_fix_commands(context)
    print(f"  ✓ Generated {len(suggestions)} suggestion(s)")

    if suggestions:
        suggestion = suggestions[0]
        print(f"  → Suggestion: {suggestion.command}")
        print(f"  → Safety tier: {suggestion.safety_tier}")

    print()
    print("Step 3: Check if should show (based on learned patterns)")
    should_show, adjusted = loop.should_show_notification(
        pattern=NotificationPattern.TERMINAL_ERROR,
        base_importance=0.7,
    )
    print(f"  ✓ Should show: {should_show}")
    print(f"  ✓ Adjusted importance: {adjusted:.2f}")

    print()
    print("Step 4: Simulate user engagement")
    await loop.record_feedback(
        pattern=NotificationPattern.TERMINAL_ERROR,
        response=UserResponse.ENGAGED,
        notification_text="npm install error",
        context={"window_type": "terminal"},
        time_to_respond=1.5,
    )
    print("  ✓ Recorded user engagement")

    print()
    print("Step 5: Verify learning occurred")
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
    if insights.get('has_data'):
        print(f"  ✓ Pattern has data: {insights['total_shown']} events")
        print(f"  ✓ Engagement rate: {insights['engagement_rate']:.0%}")
        print()
        print("✓ SUCCESS: Complete workflow works!")
        return True
    else:
        print("  ✗ FAILURE: No learning data")
        return False


async def test_live_jarvis_integration():
    """Test 6: Live JARVIS integration (if running)."""
    print("\n" + "="*80)
    print("TEST 6: Live JARVIS Integration Check")
    print("="*80 + "\n")

    print("→ Checking if JARVIS is running...")

    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=2)

        if response.status_code == 200:
            print("  ✓ JARVIS is running on port 8000")
            print()

            # Try to access learning endpoint (if available)
            print("→ Checking for learning endpoints...")
            try:
                # This might not exist yet, just checking
                learning_response = requests.get("http://localhost:8000/api/learning/stats", timeout=2)
                if learning_response.status_code == 200:
                    print("  ✓ Learning stats endpoint available")
                    data = learning_response.json()
                    print(f"  → Total events: {data.get('total_events', 'N/A')}")
                else:
                    print("  ℹ  Learning endpoints not yet implemented")
            except:
                print("  ℹ  Learning endpoints not yet implemented (this is OK)")

            print()
            print("✓ JARVIS is accessible")
            print()
            print("🎯 Next steps to test with live JARVIS:")
            print("  1. Trigger a terminal error (e.g., run 'python nonexistent.py')")
            print("  2. Wait for JARVIS to detect it")
            print("  3. Respond 'yes' or 'no' to the notification")
            print("  4. Check: cat ~/.jarvis/learning/feedback.json")
            print()
            return True

        else:
            print("  ✗ JARVIS responded but with error")
            return False

    except requests.exceptions.ConnectionError:
        print("  ℹ  JARVIS not running (this is OK for unit tests)")
        print()
        print("To test with live JARVIS:")
        print("  1. Start JARVIS: python backend/main.py")
        print("  2. Re-run this test script")
        print()
        return None  # Not a failure, just not running
    except Exception as e:
        print(f"  ✗ Error checking JARVIS: {e}")
        return False


async def check_file_locations():
    """Helper: Check that files exist."""
    print("\n" + "="*80)
    print("FILE LOCATION CHECK")
    print("="*80 + "\n")

    files_to_check = [
        ("Feedback Loop", "backend/core/learning/feedback_loop.py"),
        ("Command Safety", "backend/system_control/command_safety.py"),
        ("Terminal Intelligence", "backend/vision/handlers/terminal_command_intelligence.py"),
        ("Feedback-Aware Vision", "backend/vision/intelligence/feedback_aware_vision.py"),
        ("Tests", "backend/tests/test_feedback_learning_and_safety.py"),
        ("Demo", "backend/examples/demo_feedback_and_safety.py"),
    ]

    all_exist = True
    for name, filepath in files_to_check:
        path = Path(filepath)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            print(f"✓ {name}")
            print(f"  → {filepath} ({size_kb:.1f} KB)")
        else:
            print(f"✗ {name}")
            print(f"  → {filepath} (NOT FOUND)")
            all_exist = False

    print()
    return all_exist


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("  JARVIS Feedback Learning & Command Safety - Live Test Suite")
    print("="*80)

    # Check files exist
    files_ok = await check_file_locations()
    if not files_ok:
        print("⚠️  Some files are missing. Did the implementation complete?")
        return

    # Run tests
    results = {}

    try:
        results['feedback_basic'] = await test_feedback_loop_basic()
        results['command_safety'] = await test_command_safety_basic()
        results['terminal_intel'] = await test_terminal_intelligence()
        results['persistence'] = await test_persistence()
        results['integration'] = await test_integration_with_vision()
        results['live_jarvis'] = await test_live_jarvis_integration()

    except Exception as e:
        print(f"\n✗ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80 + "\n")

    for test_name, result in results.items():
        if result is True:
            symbol = "✓"
            status = "PASSED"
        elif result is False:
            symbol = "✗"
            status = "FAILED"
        else:
            symbol = "ℹ"
            status = "SKIPPED"

        print(f"{symbol} {test_name.replace('_', ' ').title()}: {status}")

    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    total = len(results)

    print()
    print(f"Results: {passed}/{total} passed, {failed} failed, {skipped} skipped")

    if failed == 0 and passed > 0:
        print()
        print("✨ SUCCESS! All tests passed!")
        print()
        print("🎯 The systems are working correctly!")
        print()
        print("Next steps:")
        print("  1. Run the full demo: python -m backend.examples.demo_feedback_and_safety")
        print("  2. Start JARVIS and test with real terminal errors")
        print("  3. Check learned data: cat ~/.jarvis/learning/feedback.json")
    elif failed > 0:
        print()
        print("⚠️  Some tests failed. Check the output above for details.")

    print()


if __name__ == "__main__":
    asyncio.run(main())
