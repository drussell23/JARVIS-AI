"""
Demo: Feedback Learning Loop & Command Safety Classification

This example demonstrates the "Invisible Assistant" UX philosophy:
- Learning from user feedback
- Adapting notification importance
- Safe command classification
- Terminal intelligence with safety warnings

Run this to see how JARVIS learns and improves over time.
"""
import asyncio
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_feedback_learning():
    """Demonstrate feedback learning loop."""
    from backend.core.learning.feedback_loop import (
        FeedbackLearningLoop,
        NotificationPattern,
        UserResponse,
    )

    print("\n" + "=" * 80)
    print("DEMO 1: Feedback Learning Loop")
    print("=" * 80 + "\n")

    # Create feedback loop
    loop = FeedbackLearningLoop(storage_path=Path("/tmp/jarvis_feedback_demo.json"))

    print("📚 Scenario: Teaching JARVIS what notifications you care about\n")

    # Simulate user engaging with terminal errors
    print("→ User engages with terminal errors (5 times)...")
    for i in range(5):
        await loop.record_feedback(
            pattern=NotificationPattern.TERMINAL_ERROR,
            response=UserResponse.ENGAGED,
            notification_text=f"ModuleNotFoundError in terminal (#{i+1})",
            context={"window_type": "terminal", "error_type": "ModuleNotFoundError"},
            time_to_respond=2.0,
        )

    # Check if pattern is valued
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
    print(f"\n✅ Terminal Error Pattern:")
    print(f"   - Shown: {insights['total_shown']} times")
    print(f"   - Engagement rate: {insights['engagement_rate']:.0%}")
    print(f"   - Recommendation: {insights['recommendation']}")
    print(f"   - Is valued: {insights['is_valued']}")

    # Simulate user dismissing workflow suggestions
    print("\n→ User dismisses workflow suggestions (8 times)...")
    for i in range(8):
        await loop.record_feedback(
            pattern=NotificationPattern.WORKFLOW_SUGGESTION,
            response=UserResponse.DISMISSED,
            notification_text=f"Workflow optimization suggestion (#{i+1})",
            context={"window_type": "code"},
            time_to_respond=0.5,
        )

    # Check if pattern is suppressed
    insights = loop.get_pattern_insights(NotificationPattern.WORKFLOW_SUGGESTION)
    print(f"\n❌ Workflow Suggestion Pattern:")
    print(f"   - Shown: {insights['total_shown']} times")
    print(f"   - Dismissal rate: {insights['dismissal_rate']:.0%}")
    print(f"   - Recommendation: {insights['recommendation']}")
    print(f"   - Is ignored: {insights['is_ignored']}")

    # Test notification filtering
    print("\n🧠 Testing Adaptive Filtering:")

    should_show_error, importance_error = loop.should_show_notification(
        pattern=NotificationPattern.TERMINAL_ERROR,
        base_importance=0.7,
        context={"window_type": "terminal"},
    )

    should_show_workflow, importance_workflow = loop.should_show_notification(
        pattern=NotificationPattern.WORKFLOW_SUGGESTION,
        base_importance=0.7,
        context={"window_type": "code"},
    )

    print(f"\n   Terminal Error (base=0.7):")
    print(f"      → Should show: {should_show_error}")
    print(f"      → Adjusted importance: {importance_error:.2f}")
    print(f"      → Reason: User consistently engages")

    print(f"\n   Workflow Suggestion (base=0.7):")
    print(f"      → Should show: {should_show_workflow}")
    print(f"      → Adjusted importance: {importance_workflow:.2f}")
    print(f"      → Reason: User consistently dismisses")

    # Export learned data
    print("\n📊 Full Learning Summary:")
    exported = loop.export_learned_data()
    print(f"   - Total feedback events: {exported['total_feedback_events']}")
    print(f"   - Pattern stats tracked: {len(exported['pattern_stats'])}")
    print(f"   - Suppressed patterns: {len(exported['suppressed_patterns'])}")


async def demo_command_safety():
    """Demonstrate command safety classification."""
    from backend.system_control.command_safety import get_command_classifier, SafetyTier

    print("\n" + "=" * 80)
    print("DEMO 2: Command Safety Classification")
    print("=" * 80 + "\n")

    classifier = get_command_classifier()

    test_commands = [
        # GREEN tier - safe
        ("ls -la", "List files"),
        ("git status", "Check git status"),
        ("cat README.md", "Read file"),

        # YELLOW tier - caution
        ("npm install express", "Install package"),
        ("git add .", "Stage changes"),
        ("mkdir new_folder", "Create directory"),

        # RED tier - dangerous
        ("rm -rf /tmp/important", "Delete directory"),
        ("git push --force", "Force push"),
        ("sudo rm -f /etc/config", "Delete system file"),
        ("DROP TABLE users;", "Drop database table"),
    ]

    print("🔒 Classifying Commands by Safety Tier:\n")

    for command, description in test_commands:
        result = classifier.classify(command)

        # Color coding
        tier_symbols = {
            SafetyTier.GREEN: "✅",
            SafetyTier.YELLOW: "⚠️",
            SafetyTier.RED: "🛑",
            SafetyTier.UNKNOWN: "❓",
        }
        symbol = tier_symbols[result.tier]

        print(f"{symbol} {result.tier.value.upper()}: {command}")
        print(f"   Description: {description}")
        print(f"   Requires confirmation: {result.requires_confirmation}")
        print(f"   Is destructive: {result.is_destructive}")
        print(f"   Is reversible: {result.is_reversible}")
        print(f"   Risk categories: {[r.value for r in result.risk_categories]}")

        if result.suggested_alternative:
            print(f"   💡 Safer alternative: {result.suggested_alternative}")

        if result.dry_run_available:
            print(f"   🔬 Dry-run available: Yes")

        print(f"   Reasoning: {result.reasoning}")
        print()


async def demo_terminal_intelligence():
    """Demonstrate terminal command intelligence."""
    from backend.vision.handlers.terminal_command_intelligence import (
        get_terminal_intelligence,
        TerminalCommandContext,
    )

    print("\n" + "=" * 80)
    print("DEMO 3: Terminal Command Intelligence")
    print("=" * 80 + "\n")

    intel = get_terminal_intelligence()

    # Simulate terminal OCR output with error
    terminal_ocr = """
    user@host:~/project $ python app.py
    Traceback (most recent call last):
      File "app.py", line 5, in <module>
        import requests
    ModuleNotFoundError: No module named 'requests'
    user@host:~/project $
    """

    print("🖥️  Analyzing Terminal Output:\n")
    print("```")
    print(terminal_ocr.strip())
    print("```\n")

    # Analyze context
    context = await intel.analyze_terminal_context(terminal_ocr)

    print("📊 Extracted Context:")
    print(f"   - Last command: {context.last_command}")
    print(f"   - Errors found: {len(context.errors)}")
    for err in context.errors:
        print(f"      • {err}")
    print(f"   - Shell type: {context.shell_type}")
    print()

    # Get fix suggestions
    suggestions = await intel.suggest_fix_commands(context)

    print("💡 Intelligent Fix Suggestions:\n")

    for i, suggestion in enumerate(suggestions, 1):
        formatted = await intel.format_suggestion_for_user(
            suggestion,
            include_safety_warning=True,
        )
        print(f"Suggestion #{i}:")
        print(formatted)
        print()

    # Demonstrate safety classification for suggested commands
    if suggestions:
        print("🔒 Safety Analysis of Suggested Commands:\n")
        for suggestion in suggestions:
            classification = await intel.classify_command(suggestion.command)
            print(f"Command: {suggestion.command}")
            print(f"   Tier: {classification['tier_color']} {classification['tier'].upper()}")
            print(f"   Requires confirmation: {classification['requires_confirmation']}")
            print(f"   Estimated impact: {suggestion.estimated_impact}")
            print()


async def demo_integrated_workflow():
    """Demonstrate complete integrated workflow."""
    from backend.core.learning.feedback_loop import (
        FeedbackLearningLoop,
        NotificationPattern,
        UserResponse,
    )
    from backend.vision.handlers.terminal_command_intelligence import (
        get_terminal_intelligence,
    )

    print("\n" + "=" * 80)
    print("DEMO 4: Integrated Workflow - Learn & Adapt")
    print("=" * 80 + "\n")

    loop = FeedbackLearningLoop(storage_path=Path("/tmp/jarvis_integrated_demo.json"))
    intel = get_terminal_intelligence()

    print("🎯 Scenario: User fixes terminal error multiple times\n")

    # Simulate 3 iterations of the same error
    for iteration in range(3):
        print(f"--- Iteration {iteration + 1} ---\n")

        # Analyze terminal error
        context_data = {
            "errors": ["ModuleNotFoundError: No module named 'pandas'"],
        }
        from backend.vision.handlers.terminal_command_intelligence import TerminalCommandContext
        context = TerminalCommandContext(**context_data)

        suggestions = await intel.suggest_fix_commands(context)

        if suggestions:
            suggestion = suggestions[0]
            base_importance = 0.8

            # Check if should show (based on learned patterns)
            should_show, adjusted_importance = loop.should_show_notification(
                pattern=NotificationPattern.TERMINAL_ERROR,
                base_importance=base_importance,
                context={"window_type": "terminal", "error_type": "ModuleNotFoundError"},
            )

            print(f"📬 Notification Decision:")
            print(f"   - Should show: {should_show}")
            print(f"   - Base importance: {base_importance:.2f}")
            print(f"   - Adjusted importance: {adjusted_importance:.2f}")

            if should_show:
                print(f"\n💬 JARVIS: {suggestion.purpose}")
                print(f"   Suggested command: {suggestion.command}")
                print(f"   Safety tier: {suggestion.safety_tier}")

                # Simulate user response (engaged for first 2, dismissed on 3rd)
                if iteration < 2:
                    user_response = UserResponse.ENGAGED
                    print(f"\n👤 User: Yes, run that")
                else:
                    user_response = UserResponse.DISMISSED
                    print(f"\n👤 User: Not now (dismisses)")

                # Record feedback
                await loop.record_feedback(
                    pattern=NotificationPattern.TERMINAL_ERROR,
                    response=user_response,
                    notification_text=suggestion.purpose,
                    context={"window_type": "terminal", "error_type": "ModuleNotFoundError"},
                    time_to_respond=1.5,
                )

        print()

    # Show final learned state
    print("📈 Learning Results:")
    insights = loop.get_pattern_insights(NotificationPattern.TERMINAL_ERROR)
    print(f"   - Total shown: {insights['total_shown']}")
    print(f"   - Engagement rate: {insights['engagement_rate']:.0%}")
    print(f"   - Recommendation: {insights['recommendation']}")


async def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("  JARVIS Feedback Learning & Command Safety - Interactive Demo")
    print("=" * 80)

    try:
        await demo_feedback_learning()
        await demo_command_safety()
        await demo_terminal_intelligence()
        await demo_integrated_workflow()

        print("\n" + "=" * 80)
        print("✨ Demo Complete!")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. JARVIS learns from your engagement/dismissal patterns")
        print("2. Notifications adapt in real-time to your preferences")
        print("3. Commands are classified by safety tier (GREEN/YELLOW/RED)")
        print("4. Terminal intelligence provides context-aware suggestions")
        print("5. The system gets smarter without being intrusive\n")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
