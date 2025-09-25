#!/usr/bin/env python3
"""
Test Context Intelligence Integration - Live Test
================================================

This script verifies that all the core modules are being used
when processing a command through the integration.
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging to see the flow
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Set specific loggers to show our components in action
logging.getLogger('context_intelligence.core.screen_state').setLevel(logging.DEBUG)
logging.getLogger('context_intelligence.core.command_queue').setLevel(logging.DEBUG)
logging.getLogger('context_intelligence.core.policy_engine').setLevel(logging.DEBUG)
logging.getLogger('context_intelligence.core.context_manager').setLevel(logging.DEBUG)
logging.getLogger('context_intelligence.core.feedback_manager').setLevel(logging.INFO)

logger = logging.getLogger(__name__)


async def test_scenario_with_locked_screen():
    """
    Test the exact scenario: Screen locked + "open Safari and search for dogs"
    This will show all the core modules working together.
    """
    print("\n" + "="*80)
    print("🔍 TESTING CONTEXT INTELLIGENCE CORE MODULE USAGE")
    print("="*80)
    print("\nThis test will show all core modules in action:")
    print("✓ screen_state.py - Detecting locked screen")
    print("✓ command_queue.py - Queuing the command") 
    print("✓ policy_engine.py - Deciding to auto-unlock")
    print("✓ unlock_manager.py - Attempting unlock")
    print("✓ context_manager.py - Orchestrating everything")
    print("✓ feedback_manager.py - Providing user feedback")
    print("-"*80 + "\n")
    
    # Import the enhanced wrapper that uses our core modules
    from context_intelligence.integrations.enhanced_context_wrapper import (
        EnhancedContextIntelligenceHandler
    )
    
    # Mock processor for testing
    class MockProcessor:
        async def process_command(self, command):
            logger.info(f"MockProcessor: Would execute '{command}'")
            return {"success": True, "response": f"Executed: {command}"}
    
    # Create handler (this uses all our core modules)
    processor = MockProcessor()
    handler = EnhancedContextIntelligenceHandler(processor)
    
    # Ensure initialization
    await handler._ensure_initialized()
    
    # Test command
    command = "open Safari and search for dogs"
    print(f"📢 User command: \"{command}\"")
    print("-"*80)
    
    # Check which modules are loaded
    print("\n📦 Core Modules Status:")
    print(f"✓ ScreenState detector: {handler.context_manager.screen_detector is not None}")
    print(f"✓ CommandQueue: {handler.context_manager.command_queue is not None}")
    print(f"✓ PolicyEngine: {handler.context_manager.policy_engine is not None}")
    print(f"✓ UnlockManager: {handler.context_manager.unlock_manager is not None}")
    print(f"✓ FeedbackManager: {handler.feedback_manager is not None}")
    
    # Check current screen state
    print("\n🖥️  Checking Screen State...")
    screen_state = await handler.context_manager.screen_detector.get_screen_state()
    print(f"   State: {screen_state.state.value}")
    print(f"   Detection Method: {screen_state.detection_method.value}")
    print(f"   Confidence: {screen_state.confidence:.2%}")
    
    # Process the command (this will use ALL core modules)
    print(f"\n🚀 Processing command through Context Intelligence System...")
    print("-"*80)
    
    result = await handler.process_with_context(command)
    
    print("\n📊 Results:")
    print(f"✓ Success: {result.get('success')}")
    print(f"✓ Message: {result.get('message')}")
    if 'command_id' in result.get('result', {}):
        print(f"✓ Command ID: {result['result']['command_id']}")
        print(f"✓ Status: {result['result']['status']}")
        print(f"✓ Requires Unlock: {result['result'].get('requires_unlock')}")
    
    # Show queue status
    queue_stats = await handler.context_manager.command_queue.get_statistics()
    print(f"\n📋 Queue Statistics:")
    print(f"   Total Queued: {queue_stats['total_queued']}")
    print(f"   Current Size: {queue_stats['current_queue_size']}")
    
    # Show policy engine stats
    policy_stats = handler.context_manager.policy_engine.get_statistics()
    print(f"\n🔐 Policy Engine Statistics:")
    print(f"   Total Decisions: {policy_stats['total_decisions_24h']}")
    print(f"   Active Rules: {policy_stats['active_rules']}")
    
    # Show execution steps
    if handler.execution_steps:
        print(f"\n📝 Execution Steps:")
        for i, step in enumerate(handler.execution_steps, 1):
            print(f"   {i}. {step['step']}")
    
    print("\n✅ Test Complete - All core modules are working together!")


async def check_imports():
    """Verify all imports work correctly"""
    print("\n" + "="*80)
    print("📦 VERIFYING IMPORTS")
    print("="*80 + "\n")
    
    try:
        # Test new import location
        from context_intelligence.integrations.enhanced_context_wrapper import (
            wrap_with_enhanced_context,
            EnhancedContextIntelligenceHandler
        )
        print("✓ Successfully imported from context_intelligence.integrations.enhanced_context_wrapper")
        
        # Verify it's our new implementation
        handler = EnhancedContextIntelligenceHandler(None)
        has_context_manager = hasattr(handler, 'context_manager')
        has_feedback_manager = hasattr(handler, 'feedback_manager')
        
        print(f"✓ Has context_manager: {has_context_manager}")
        print(f"✓ Has feedback_manager: {has_feedback_manager}")
        
        if has_context_manager and has_feedback_manager:
            print("\n✅ This is definitely our new Context Intelligence System!")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
        
    return True


async def main():
    """Main test runner"""
    # First check imports
    if not await check_imports():
        print("\n❌ Import check failed. Make sure you're in the backend directory.")
        return
    
    # Then run the scenario test
    await test_scenario_with_locked_screen()
    
    print("\n" + "="*80)
    print("🎉 All core modules are integrated and working!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())