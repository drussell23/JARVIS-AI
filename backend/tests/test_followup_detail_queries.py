"""
Test Dynamic Follow-Up Detail Queries
======================================

Tests the conversational follow-up system where users can ask for
detailed explanations after an initial query.

Example flow:
1. User: "can you see my terminal in the other window?"
2. JARVIS: "Yes, I can see Terminal in Space 2..."
3. User: "explain what's happening in detail"
4. JARVIS: [Dynamic detailed explanation based on terminal context]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
from backend.core.context.context_integration_bridge import ContextIntegrationBridge
from backend.core.context.multi_space_context_graph import MultiSpaceContextGraph, TerminalContext
from backend.vision.handlers.terminal_command_intelligence import TerminalCommandIntelligence


async def test_followup_detail_query():
    """Test that follow-up detail queries work dynamically"""
    print("\n" + "="*80)
    print(" Dynamic Follow-Up Detail Query Test")
    print("="*80 + "\n")

    # Create context graph and bridge
    graph = MultiSpaceContextGraph()
    terminal_intel = TerminalCommandIntelligence()
    bridge = ContextIntegrationBridge(
        context_graph=graph,
        terminal_intelligence=terminal_intel
    )

    await graph.start()

    # Simulate a terminal with an error in Space 2
    print("📝 Setting up test scenario...")
    print("   • Terminal in Space 2 with ModuleNotFoundError\n")

    terminal_ctx = TerminalContext(
        last_command="python app.py",
        errors=["ModuleNotFoundError: No module named 'requests'"],
        working_directory="/Users/test/project",
        shell_type="zsh"
    )

    graph.update_terminal_context(
        space_id=2,
        app_name="Terminal",
        command=terminal_ctx.last_command,
        output="",
        errors=terminal_ctx.errors,
        working_dir=terminal_ctx.working_directory
    )

    # Step 1: Initial visibility query
    print("👤 User: \"can you see my terminal in the other window?\"\n")
    response1 = await bridge.answer_query(
        "can you see my terminal in the other window?",
        current_space_id=1
    )
    print(f"🤖 JARVIS:\n{response1}\n")
    print("-" * 80 + "\n")

    # Step 2: Follow-up detail query
    print("👤 User: \"explain what's happening in detail\"\n")
    response2 = await bridge.answer_query(
        "explain what's happening in detail",
        current_space_id=1
    )
    print(f"🤖 JARVIS:\n{response2}\n")
    print("-" * 80 + "\n")

    # Verify the response is dynamic and contains expected information
    checks = {
        "Contains app name": "Terminal" in response2,
        "Contains space ID": "Space 2" in response2,
        "Contains error": "ModuleNotFoundError" in response2,
        "Contains command": "python app.py" in response2,
        "Contains working directory": "/Users/test/project" in response2 or "Working directory" in response2,
        "NOT hardcoded": "Would you like me to" not in response2  # Should be detailed, not asking
    }

    print("✅ Validation Checks:")
    all_passed = True
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {check_name}")
        if not passed:
            all_passed = False

    await graph.stop()

    print("\n" + "="*80)
    if all_passed:
        print("🎉 All checks passed! Follow-up system is working dynamically.")
    else:
        print("⚠️  Some checks failed. Review output above.")
    print("="*80 + "\n")

    return all_passed


async def test_followup_timeout():
    """Test that follow-up queries time out after 2 minutes"""
    print("\n" + "="*80)
    print(" Follow-Up Query Timeout Test")
    print("="*80 + "\n")

    graph = MultiSpaceContextGraph()
    bridge = ContextIntegrationBridge(context_graph=graph)
    await graph.start()

    # Simulate terminal context
    terminal_ctx = TerminalContext(
        last_command="ls",
        working_directory="/tmp"
    )
    graph.update_terminal_context(
        space_id=1,
        app_name="Terminal",
        command=terminal_ctx.last_command,
        output="file1.txt file2.txt",
        working_dir=terminal_ctx.working_directory
    )

    # Initial query
    print("👤 User: \"can you see my terminal?\"\n")
    response1 = await bridge.answer_query("can you see my terminal?")
    print(f"🤖 JARVIS: {response1[:100]}...\n")

    # Manually expire the conversation timestamp
    from datetime import datetime, timedelta
    bridge._conversation_timestamp = datetime.now() - timedelta(minutes=3)

    # Follow-up should NOT work (too old)
    print("👤 User (3 minutes later): \"explain in detail\"\n")
    response2 = await bridge.answer_query("explain in detail")
    print(f"🤖 JARVIS: {response2}\n")

    # Should fall back to generic response (not detailed explanation)
    # The key is it should NOT provide detailed terminal context
    has_detailed_info = "working directory" in response2.lower() or "last command" in response2.lower()
    is_fallback = not has_detailed_info

    await graph.stop()

    print("-" * 80)
    if is_fallback:
        print("✓ PASS: Conversation context properly times out after 2 minutes")
    else:
        print("✗ FAIL: Conversation context should timeout")
    print("="*80 + "\n")

    return is_fallback


async def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("🧪 Testing Dynamic Follow-Up Detail Query System")
    print("="*80)

    test1_passed = await test_followup_detail_query()
    test2_passed = await test_followup_timeout()

    print("\n" + "="*80)
    print("📊 Test Results Summary")
    print("="*80)
    print(f"  • Dynamic follow-up query: {'✓ PASS' if test1_passed else '✗ FAIL'}")
    print(f"  • Conversation timeout: {'✓ PASS' if test2_passed else '✗ FAIL'}")
    print("="*80 + "\n")

    all_passed = test1_passed and test2_passed
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
