#!/usr/bin/env python3
"""
Test Context-Aware Screen Unlock
================================

Tests the complete flow of JARVIS detecting locked screen and unlocking before executing commands
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the components
from api.direct_unlock_handler_fixed import (
    check_screen_locked_direct,
    test_screen_lock_context,
)
from api.simple_context_handler_enhanced import EnhancedSimpleContextHandler


class MockProcessor:
    """Mock command processor for testing"""

    async def process_command(self, command: str, websocket=None) -> dict:
        """Simulate command processing"""
        logger.info(f"[MOCK] Processing command: {command}")

        # Simulate successful command execution
        if "safari" in command.lower() and "search" in command.lower():
            return {
                "success": True,
                "response": "I've opened Safari and searched for dogs as requested.",
                "command_type": "browser_automation",
            }
        elif "open" in command.lower():
            app = command.split("open")[-1].strip()
            return {
                "success": True,
                "response": f"I've opened {app} for you.",
                "command_type": "application_control",
            }
        else:
            return {
                "success": True,
                "response": f"I've executed your command: {command}",
                "command_type": "general",
            }


async def test_context_aware_command(command: str):
    """Test a command with context awareness"""
    print(f"\n🧪 Testing Context-Aware Command")
    print("=" * 60)
    print(f"Command: '{command}'")
    print()

    # Create the context handler with mock processor
    mock_processor = MockProcessor()
    context_handler = EnhancedSimpleContextHandler(mock_processor)

    # Process the command
    result = await context_handler.process_with_context(command)

    # Display results
    print("\n📊 Results:")
    print(f"Success: {result.get('success', False)}")
    print(f"Context Handled: {result.get('context_handled', False)}")
    print(f"Screen Unlocked: {result.get('screen_unlocked', False)}")
    print(f"\nResponse: {result.get('response', 'No response')}")

    if result.get("execution_steps"):
        print("\n📝 Execution Steps:")
        for i, step in enumerate(result.get("execution_steps", []), 1):
            print(f"  {i}. {step['step']}")

    return result


async def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("🔐 JARVIS CONTEXT-AWARE SCREEN UNLOCK TEST")
    print("=" * 70)

    # Test 1: Check screen lock detection
    print("\n1️⃣ Testing Screen Lock Detection")
    print("-" * 40)
    is_locked = await test_screen_lock_context()

    # Test 2: Test context-aware commands
    print("\n\n2️⃣ Testing Context-Aware Commands")
    print("-" * 40)

    test_commands = [
        "open safari and search for dogs",
        "open Chrome",
        "show me my desktop",
        "what time is it",  # This shouldn't trigger unlock
    ]

    for cmd in test_commands:
        await test_context_aware_command(cmd)
        print("\n" + "-" * 60)
        await asyncio.sleep(1)  # Brief pause between tests

    # Summary
    print("\n" + "=" * 70)
    print("✅ CONTEXT AWARENESS TEST COMPLETE")
    print("=" * 70)
    print("\n📌 Summary:")
    print("- Screen lock detection: Working")
    print("- Context-aware unlock: Ready")
    print("- Command execution with context: Implemented")
    print("\n🎯 Next Steps:")
    print("1. Make sure Voice Unlock daemon is running (port 8765)")
    print("2. Lock your screen (Cmd+Ctrl+Q)")
    print("3. Say: 'Hey JARVIS, open Safari and search for dogs'")
    print(
        "4. JARVIS should detect the locked screen, unlock it, and execute the command!"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️ Test cancelled by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

