"""
Simplified Live Test - Avoids circular import issues
Tests our new systems without importing the whole backend
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


async def test_feedback_loop():
    """Test feedback loop works."""
    print("\n" + "="*60)
    print("TEST 1: Feedback Learning Loop")
    print("="*60 + "\n")

    from backend.core.learning.feedback_loop import (
        FeedbackLearningLoop,
        NotificationPattern,
        UserResponse,
    )

    loop = FeedbackLearningLoop(storage_path=Path("/tmp/simple_test.json"))

    # Record engagement
    for i in range(3):
        await loop.record_feedback(
            pattern=NotificationPattern.TERMINAL_ERROR,
            response=UserResponse.ENGAGED,
            notification_text=f"Test {i+1}",
            context={},
        )

    # Check stats
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)

    print(f"✓ Events recorded: {insights['total_shown']}")
    print(f"✓ Engagement rate: {insights['engagement_rate']:.0%}")
    print(f"✓ Is valued: {insights['is_valued']}")

    # Test filtering
    should_show, adjusted = loop.should_show_notification(
        pattern=NotificationPattern.TERMINAL_ERROR,
        base_importance=0.7,
    )

    print(f"✓ Should show: {should_show}")
    print(f"✓ Adjusted importance: {adjusted:.2f}")

    if insights['total_shown'] == 3 and insights['engagement_rate'] == 1.0:
        print("\n✅ PASSED")
        return True
    else:
        print("\n❌ FAILED")
        return False


async def test_command_safety():
    """Test command safety classification."""
    print("\n" + "="*60)
    print("TEST 2: Command Safety Classification")
    print("="*60 + "\n")

    # Import ONLY command_safety, not the whole system_control package
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "command_safety",
        "backend/system_control/command_safety.py"
    )
    command_safety = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command_safety)

    classifier = command_safety.CommandSafetyClassifier()

    tests = [
        ("ls -la", "green"),
        ("npm install express", "yellow"),
        ("rm -rf /tmp/test", "red"),
    ]

    passed = 0
    for cmd, expected in tests:
        result = classifier.classify(cmd)
        actual = result.tier.value

        if actual == expected:
            print(f"✓ {cmd}: {actual} (expected {expected})")
            passed += 1
        else:
            print(f"✗ {cmd}: {actual} (expected {expected})")

    if passed == len(tests):
        print(f"\n✅ PASSED ({passed}/{len(tests)})")
        return True
    else:
        print(f"\n❌ FAILED ({passed}/{len(tests)})")
        return False


async def test_terminal_intelligence():
    """Test terminal intelligence."""
    print("\n" + "="*60)
    print("TEST 3: Terminal Intelligence")
    print("="*60 + "\n")

    # Direct import
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "terminal_command_intelligence",
        "backend/vision/handlers/terminal_command_intelligence.py"
    )
    terminal_intel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(terminal_intel)

    intel = terminal_intel.TerminalCommandIntelligence()

    # Test OCR analysis
    ocr_text = """
    $ python app.py
    ModuleNotFoundError: No module named 'requests'
    """

    context = await intel.analyze_terminal_context(ocr_text)

    print(f"✓ Errors found: {len(context.errors)}")

    if context.errors:
        print(f"  → {context.errors[0][:60]}...")

    # Test suggestions
    suggestions = await intel.suggest_fix_commands(context)

    print(f"✓ Suggestions: {len(suggestions)}")

    if suggestions:
        sug = suggestions[0]
        print(f"  → Command: {sug.command}")
        print(f"  → Safety tier: {sug.safety_tier}")

    if len(context.errors) > 0 and len(suggestions) > 0:
        print("\n✅ PASSED")
        return True
    else:
        print("\n❌ FAILED")
        return False


async def main():
    """Run tests."""
    print("\n" + "="*60)
    print(" JARVIS Feedback & Safety - Simple Test")
    print("="*60)

    results = {}

    try:
        results['feedback'] = await test_feedback_loop()
        results['safety'] = await test_command_safety()
        results['terminal'] = await test_terminal_intelligence()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60 + "\n")

    for name, passed in results.items():
        symbol = "✅" if passed else "❌"
        print(f"{symbol} {name.title()}")

    passed_count = sum(results.values())
    total = len(results)

    print(f"\nResults: {passed_count}/{total} passed")

    if passed_count == total:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed")


if __name__ == "__main__":
    asyncio.run(main())
