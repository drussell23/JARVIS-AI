#!/usr/bin/env python3
"""
Test JARVIS with Goal Inference + Autonomous Decision Integration
"""

import asyncio
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from backend.api.unified_command_processor import UnifiedCommandProcessor


async def test_jarvis_integration():
    """Test the integrated JARVIS system"""

    print("=" * 70)
    print("🚀 TESTING JARVIS WITH GOAL INFERENCE + AUTONOMOUS INTEGRATION")
    print("=" * 70)

    # Initialize the unified command processor
    processor = UnifiedCommandProcessor()

    # Test scenarios
    test_scenarios = [
        {
            "name": "Simple TV Connection",
            "command": "living room tv",
            "context": "Normal usage"
        },
        {
            "name": "Display Connection with Context",
            "command": "connect to display",
            "context": "With Keynote open"
        },
        {
            "name": "Ambiguous Display Request",
            "command": "display",
            "context": "Should use Goal Inference"
        },
        {
            "name": "Full Command",
            "command": "connect to living room tv",
            "context": "Explicit command"
        }
    ]

    for scenario in test_scenarios:
        print(f"\n{'='*60}")
        print(f"📝 TEST: {scenario['name']}")
        print(f"   Command: '{scenario['command']}'")
        print(f"   Context: {scenario['context']}")
        print(f"{'='*60}")

        try:
            # Process the command
            result = await processor.process_command(scenario['command'])

            # Display results
            print(f"\n✅ Result:")
            print(f"   Success: {result.get('success', False)}")
            print(f"   Response: {result.get('response', 'No response')}")

            # Check for Goal Inference indicators
            if result.get('goal_inference_active'):
                print(f"   🎯 GOAL INFERENCE: Active!")
                print(f"   ⚡ Execution Time: {result.get('execution_time', 'N/A')}")

            # Check command type
            print(f"   Command Type: {result.get('command_type', 'unknown')}")

            # If there were proactive suggestions
            if 'proactive_suggestion' in str(result):
                print(f"   💡 Proactive Suggestion Detected!")

        except Exception as e:
            print(f"\n❌ Error: {e}")

    print("\n" + "=" * 70)
    print("📊 INTEGRATION STATUS CHECK")
    print("=" * 70)

    # Check if Goal Inference is loaded
    if hasattr(processor, 'goal_autonomous_integration') and processor.goal_autonomous_integration:
        print("✅ Goal Inference + Autonomous Engine: LOADED")

        # Get metrics
        metrics = processor.goal_autonomous_integration.get_metrics()
        print(f"   Goals Inferred: {metrics['goals_inferred']}")
        print(f"   Decisions Made: {metrics['decisions_made']}")
        print(f"   Display Connections: {metrics['display_connections']}")
        print(f"   Prediction Accuracy: {metrics.get('prediction_accuracy', 0):.0%}")
    else:
        print("❌ Goal Inference + Autonomous Engine: NOT LOADED")

    # Check other components
    print("\n📦 Component Status:")
    components = [
        ('Display Reference Handler', 'display_ref_handler'),
        ('Advanced Display Monitor', 'display_monitor'),
        ('UAE Engine', 'uae_engine'),
        ('Query Complexity Manager', 'query_complexity_manager'),
        ('Context-Aware Manager', 'context_aware_manager')
    ]

    for name, attr in components:
        if hasattr(processor, attr) and getattr(processor, attr):
            print(f"   ✅ {name}: Loaded")
        else:
            print(f"   ❌ {name}: Not loaded")

    print("\n" + "=" * 70)
    print("✨ TEST COMPLETE")
    print("=" * 70)


async def test_proactive_scenario():
    """Test a scenario that should trigger proactive suggestions"""

    print("\n" + "=" * 70)
    print("🎬 PROACTIVE SUGGESTION TEST")
    print("=" * 70)

    processor = UnifiedCommandProcessor()

    # Simulate a meeting preparation scenario
    print("\n📅 Simulating: User preparing for presentation")
    print("   Context: Keynote open, meeting in 10 minutes")

    # First, establish context with some commands
    context_commands = [
        "what's on my calendar",
        "open keynote",
        "check the time"
    ]

    print("\n🔄 Building context...")
    for cmd in context_commands:
        print(f"   → {cmd}")
        await processor.process_command(cmd)
        await asyncio.sleep(0.5)

    print("\n⏰ Waiting for Goal Inference to analyze...")
    await asyncio.sleep(2)

    # Now try a display command
    print("\n💬 User: 'display'")
    result = await processor.process_command("display")

    print("\n📊 Result:")
    print(f"   Success: {result.get('success', False)}")
    print(f"   Response: {result.get('response', 'No response')}")

    if result.get('goal_inference_active'):
        print("\n🎯 GOAL INFERENCE ACTIVATED!")
        print("   → System understood context and made intelligent decision")
    else:
        print("\n💡 Goal Inference not activated (may need more context)")

    print("\n✅ Proactive test complete")


if __name__ == "__main__":
    print("\n🤖 JARVIS GOAL INFERENCE INTEGRATION TEST\n")

    # Run basic integration test
    asyncio.run(test_jarvis_integration())

    # Run proactive suggestion test
    asyncio.run(test_proactive_scenario())

    print("\n" + "=" * 70)
    print("🎓 WHAT TO LOOK FOR:")
    print("=" * 70)
    print("""
1. ✅ Goal Inference Loaded - Integration is active
2. 🎯 Goal Inference Active - Commands are being optimized
3. ⚡ Faster execution times - <0.5s means optimization worked
4. 💡 Proactive suggestions - System anticipates needs
5. 📊 Metrics showing inferred goals - System is learning

If you see these indicators, the integration is WORKING!
""")