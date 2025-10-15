"""
Multi-Space Context Graph - Interactive Demo
=============================================

This demo shows how JARVIS uses the multi-space context graph to:
1. Track activity across multiple spaces
2. Detect cross-space relationships
3. Answer "what does it say?" queries
4. Provide intelligent workspace summaries

Run this to see the system in action!
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.context.multi_space_context_graph import MultiSpaceContextGraph
from backend.core.context.context_integration_bridge import initialize_integration_bridge


# ============================================================================
# SCENARIO 1: Debugging Workflow
# ============================================================================

async def demo_debugging_workflow():
    """
    Scenario: Developer debugging an error across 3 spaces
    - Space 1: Terminal with test failure
    - Space 2: Browser researching JWT error
    - Space 3: VS Code editing the fix
    """
    print("\n" + "="*80)
    print(" SCENARIO 1: Multi-Space Debugging Workflow")
    print("="*80 + "\n")

    graph = MultiSpaceContextGraph(enable_cross_space_correlation=True)
    await graph.start()

    print("Simulating debugging workflow...\n")

    # Step 1: Terminal error in Space 1
    print("📍 Space 1 - Terminal")
    print("  $ npm test")
    print("  ❌ FAIL src/auth.test.js")
    print("     TypeError: Cannot read property 'verify' of undefined\n")

    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command="npm test",
        output="FAIL src/auth.test.js\n  TypeError: Cannot read property 'verify' of undefined",
        errors=["TypeError: Cannot read property 'verify' of undefined"],
        exit_code=1,
        working_dir="/Users/developer/my-project"
    )

    await asyncio.sleep(1)

    # Step 2: User switches to Space 3 to research
    print("📍 Space 3 - Chrome (Research)")
    print("  🔍 Searching: jwt verify undefined")
    print("  📄 Reading: Stack Overflow - JWT verification error\n")

    graph.set_active_space(3)
    graph.update_browser_context(
        space_id=3,
        app_name="Chrome",
        url="https://stackoverflow.com/questions/jwt-verify-undefined",
        title="JWT verify undefined - Stack Overflow",
        extracted_text="Stack Overflow Q&A about JWT documentation verify method TypeError undefined import jsonwebtoken",
        search_query="jwt verify undefined"
    )

    await asyncio.sleep(1)

    # Step 3: User switches to Space 2 to edit code
    print("📍 Space 2 - VS Code")
    print("  📝 Editing: auth.test.js")
    print("  📂 Files: auth.test.js, auth.js\n")

    graph.set_active_space(2)
    graph.update_ide_context(
        space_id=2,
        app_name="VS Code",
        active_file="auth.test.js",
        open_files=["auth.test.js", "auth.js", "package.json"]
    )

    await asyncio.sleep(1)

    # Give correlation time to detect pattern
    print("🧠 JARVIS is analyzing cross-space activity...\n")
    await asyncio.sleep(2)

    # Check what JARVIS detected
    if graph.correlator and graph.correlator.relationships:
        print("✨ JARVIS Detected Cross-Space Relationships:\n")
        for rel in graph.correlator.relationships.values():
            print(f"   Type: {rel.relationship_type}")
            print(f"   Confidence: {rel.confidence:.0%}")
            print(f"   Involved Spaces: {rel.involved_spaces}")
            print(f"   Description: {rel.description}\n")
    else:
        print("   (Correlation analysis in progress...)\n")

    # Demo: User asks "what's the error?"
    print("💬 User: 'what's the error?'\n")
    error_context = graph.find_most_recent_error()
    if error_context:
        space_id, app_name, details = error_context
        print(f"🤖 JARVIS: The error in {app_name} (Space {space_id}) is:")
        print(f"   {details.get('errors', ['Unknown'])[0]}")
        print(f"\n   This happened when you ran: `{details.get('command', 'unknown')}`")
        print(f"   Working directory: {graph.spaces[space_id].applications[app_name].terminal_context.working_directory}\n")

    # Demo: Cross-space summary
    print("💬 User: 'what am I working on?'\n")
    summary = graph.get_cross_space_summary()
    print(f"🤖 JARVIS: {summary}\n")

    await graph.stop()
    print("="*80 + "\n")


# ============================================================================
# SCENARIO 2: "What Does It Say?" Query
# ============================================================================

async def demo_what_does_it_say():
    """
    Scenario: User switches spaces and forgets error details
    Tests the "what does it say?" implicit reference resolution
    """
    print("\n" + "="*80)
    print(" SCENARIO 2: 'What Does It Say?' - Context Preservation")
    print("="*80 + "\n")

    bridge = await initialize_integration_bridge(auto_start=False)
    graph = bridge.context_graph
    await graph.start()

    print("Scenario: You saw an error 2 minutes ago, then switched spaces...\n")

    # 2 minutes ago: Error in terminal
    print("📍 2:47 PM - Space 1 - Terminal")
    print("  $ python app.py")
    print("  ModuleNotFoundError: No module named 'requests'\n")

    await bridge.process_ocr_update(
        space_id=1,
        app_name="Terminal",
        ocr_text="""
        $ python app.py
        Traceback (most recent call last):
          File "app.py", line 5, in <module>
            import requests
        ModuleNotFoundError: No module named 'requests'
        $
        """
    )

    await asyncio.sleep(0.5)

    # You switch to browser
    print("📍 2:48 PM - You switch to Space 3 - Chrome")
    print("  (Reading documentation...)\n")

    graph.set_active_space(3)
    await asyncio.sleep(0.5)

    # 3 minutes later: You switch back to terminal
    print("📍 2:51 PM - You switch back to Space 1 - Terminal")
    print("  (Error has scrolled off screen...)\n")

    graph.set_active_space(1)
    await asyncio.sleep(0.5)

    # Now you ask: "what does it say?"
    print("💬 User: 'what does it say?'\n")

    response = await bridge.answer_query("what does it say?")
    print(f"🤖 JARVIS:\n{response}\n")

    # Alternative queries
    print("💬 User: 'what was the error?'\n")
    response2 = await bridge.answer_query("what was the error?")
    print(f"🤖 JARVIS:\n{response2}\n")

    await graph.stop()
    print("="*80 + "\n")


# ============================================================================
# SCENARIO 3: Workspace Health Summary
# ============================================================================

async def demo_workspace_health():
    """
    Scenario: User has many spaces open
    JARVIS provides intelligent summary and recommendations
    """
    print("\n" + "="*80)
    print(" SCENARIO 3: Workspace Health & Organization")
    print("="*80 + "\n")

    graph = MultiSpaceContextGraph(enable_cross_space_correlation=True)
    await graph.start()

    print("Simulating a busy workspace with 4 spaces...\n")

    # Space 1: Development
    print("📍 Space 1 - Development")
    print("  • Terminal (running dev server)")
    print("  • VS Code (editing code)\n")

    graph.update_terminal_context(1, "Terminal", command="npm run dev", working_dir="/Users/dev/project")
    graph.update_ide_context(1, "VS Code", active_file="app.js", open_files=["app.js", "server.js"])

    # Space 2: Testing
    print("📍 Space 2 - Testing")
    print("  • Terminal (running tests)")
    print("  • Chrome (viewing test results)\n")

    graph.update_terminal_context(2, "Terminal", command="npm test", working_dir="/Users/dev/project")
    graph.update_browser_context(2, "Chrome", url="localhost:3000/tests")

    # Space 3: Research
    print("📍 Space 3 - Research")
    print("  • Safari (reading documentation)\n")

    graph.update_browser_context(3, "Safari", url="https://docs.example.com", extracted_text="Documentation guide tutorial")

    # Space 4: Communication (idle)
    print("📍 Space 4 - Communication (Idle)")
    print("  • Slack (open but idle)\n")

    graph.get_or_create_space(4)
    space4 = graph.spaces[4]
    space4.add_application("Slack", ContextType.COMMUNICATION)

    await asyncio.sleep(1)

    # Infer tags for all spaces
    for space in graph.spaces.values():
        space.infer_tags()

    # Get workspace summary
    print("🧠 JARVIS Workspace Analysis:\n")
    summary = graph.get_summary()

    print(f"   Total Spaces: {summary['total_spaces']}")
    print(f"   Currently Active: Space {summary['current_space_id']}\n")

    for space_id, space_data in summary['spaces'].items():
        tags = ', '.join(space_data['tags']) if space_data['tags'] else 'none'
        apps = ', '.join(space_data['applications'].keys())
        print(f"   Space {space_id}:")
        print(f"     Tags: {tags}")
        print(f"     Apps: {apps}")
        print(f"     Last Activity: {space_data['last_activity'][:19]}\n")

    # Cross-space relationships
    if summary['cross_space_relationships']:
        print("   🔗 Cross-Space Relationships Detected:")
        for rel in summary['cross_space_relationships']:
            print(f"     • {rel['description']}")
        print()

    await graph.stop()
    print("="*80 + "\n")


# ============================================================================
# SCENARIO 4: Real-Time Context Updates
# ============================================================================

async def demo_realtime_updates():
    """
    Scenario: Simulating real-time updates as user works
    Shows how context updates in real-time
    """
    print("\n" + "="*80)
    print(" SCENARIO 4: Real-Time Context Tracking")
    print("="*80 + "\n")

    bridge = await initialize_integration_bridge(auto_start=True)
    graph = bridge.context_graph

    print("Watching real-time activity in Space 1...\n")

    # Simulate a series of activities
    activities = [
        ("$ cd ~/project", None, None),
        ("$ git status", None, 0),
        ("$ npm install", "Installing packages...", 0),
        ("$ npm test", "FAIL: 3 tests failed", 1),
        ("$ vim test.js", None, 0),
    ]

    for i, (command, output, exit_code) in enumerate(activities, 1):
        print(f"{i}. Terminal: {command}")

        errors = []
        if exit_code == 1 and output:
            errors = [output]

        graph.update_terminal_context(
            space_id=1,
            app_name="Terminal",
            command=command,
            output=output,
            errors=errors,
            exit_code=exit_code
        )

        # Show context state after each command
        space1 = graph.spaces[1]
        terminal = space1.applications["Terminal"]

        recent_events = space1.get_recent_events(within_seconds=10)
        print(f"   Recent Events: {len(recent_events)}")
        print(f"   Activity Count: {terminal.activity_count}")
        print(f"   Significance: {terminal.significance.value}")

        if errors:
            print(f"   ⚠️  Error Detected: {errors[0][:60]}...")

        print()
        await asyncio.sleep(0.5)

    print("✅ Context tracking complete!\n")
    await bridge.stop()
    print("="*80 + "\n")


# ============================================================================
# MAIN DEMO
# ============================================================================

async def main():
    """Run all demo scenarios"""
    print("\n" + "="*80)
    print(" 🤖 JARVIS Multi-Space Context Graph - Interactive Demo")
    print("="*80)
    print("\nThis demo shows how JARVIS tracks your workspace across")
    print("multiple macOS Spaces and understands your workflow.\n")

    scenarios = [
        ("Debugging Workflow", demo_debugging_workflow),
        ("What Does It Say?", demo_what_does_it_say),
        ("Workspace Health", demo_workspace_health),
        ("Real-Time Updates", demo_realtime_updates),
    ]

    for i, (name, demo_func) in enumerate(scenarios, 1):
        try:
            await demo_func()

            if i < len(scenarios):
                print(f"\nPress Enter to continue to Scenario {i+1}...")
                input()
        except KeyboardInterrupt:
            print("\n\nDemo interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Error in scenario '{name}': {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*80)
    print(" Demo Complete!")
    print("="*80)
    print("\n✨ This is how JARVIS tracks your workspace and answers")
    print("   'what does it say?' queries!\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nDemo stopped.")
