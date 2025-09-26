#!/usr/bin/env python3
"""
Test Full Context Intelligence Flow
==================================

Demonstrates the complete flow for the example scenario:
- Mac is locked
- User says "JARVIS, open Safari and search for dogs"
- System handles it with full context awareness
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

# Reduce noise from other loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class ScenarioSimulator:
    """Simulates the complete scenario with all components"""
    
    def __init__(self):
        self.screen_locked = True
        self.unlock_attempts = 0
        self.commands_executed = []
        self.feedback_messages = []
        
    async def simulate_screen_state(self, locked: bool):
        """Simulate screen lock state"""
        self.screen_locked = locked
        logger.info(f"🖥️  Screen state: {'LOCKED' if locked else 'UNLOCKED'}")
        
    async def simulate_unlock(self):
        """Simulate screen unlock process"""
        self.unlock_attempts += 1
        logger.info(f"🔓 Attempting to unlock screen (attempt #{self.unlock_attempts})")
        
        # Simulate unlock delay
        await asyncio.sleep(1.5)
        
        # Succeed on first attempt
        if self.unlock_attempts <= 1:
            self.screen_locked = False
            logger.info("✅ Screen unlocked successfully!")
            return True
        else:
            logger.info("❌ Unlock failed")
            return False
            
    async def simulate_command_execution(self, command: str):
        """Simulate actual command execution"""
        self.commands_executed.append(command)
        logger.info(f"⚙️  Executing: {command}")
        
        # Simulate execution time
        await asyncio.sleep(1.0)
        
        logger.info("✅ Command executed successfully")
        return True


async def run_full_scenario():
    """Run the complete scenario with mocked components"""
    print("\n" + "="*70)
    print("🤖 JARVIS CONTEXT INTELLIGENCE - FULL SCENARIO TEST")
    print("="*70)
    print("\nScenario: User wants to search while screen is locked")
    print("-"*70 + "\n")
    
    simulator = ScenarioSimulator()
    
    # Mock screen state detection
    with patch('context_intelligence.core.screen_state.ScreenStateDetector._detect_via_quartz') as mock_screen:
        # Initially locked
        from context_intelligence.core.screen_state import ScreenState
        mock_screen.return_value = (ScreenState.LOCKED, 0.95, {})
        
        # Mock unlock manager
        with patch('context_intelligence.core.unlock_manager.UnlockManager._unlock_via_applescript') as mock_unlock:
            mock_unlock.return_value = (True, None)
            
            # Import our system
            from context_intelligence.integrations.jarvis_integration import get_jarvis_integration
            from context_intelligence.core.feedback_manager import get_feedback_manager, FeedbackChannel
            
            integration = get_jarvis_integration()
            feedback_manager = get_feedback_manager()
            
            # Capture feedback
            captured_feedback = []
            
            async def capture_voice_feedback(feedback):
                captured_feedback.append(feedback.content)
                print(f"\n🗣️  JARVIS: \"{feedback.content}\"")
                
            feedback_manager.register_channel_handler(
                FeedbackChannel.VOICE,
                capture_voice_feedback
            )
            
            # Initialize
            await integration.initialize()
            
            # Step 1: User gives command
            command = "open Safari and search for dogs"
            print(f"👤 User: \"JARVIS, {command}\"")
            
            # Step 2: Process command
            result = await integration.process_voice_command(
                command=command,
                voice_context={
                    "source": "voice",
                    "urgency": "normal"
                }
            )
            
            # Show initial response
            print(f"\n📋 Initial Response:")
            print(f"   - Status: {result.get('status')}")
            print(f"   - Requires Unlock: {result.get('requires_unlock')}")
            print(f"   - Command ID: {result.get('command_id')}")
            
            # Step 3: Monitor execution
            if result.get('status') == 'queued':
                print(f"\n⏳ Command queued due to locked screen...")
                
                # Wait a moment for queue processing
                await asyncio.sleep(2)
                
                # Now change screen state to simulate unlock
                mock_screen.return_value = (ScreenState.UNLOCKED, 0.95, {})
                
                # Give system time to process
                await asyncio.sleep(3)
                
            # Step 4: Check final status
            from context_intelligence.core.command_queue import get_command_queue
            queue = get_command_queue()
            
            if result.get('command_id'):
                final_status = await queue.get_command_status(result['command_id'])
                if final_status:
                    print(f"\n📊 Final Status: {final_status.status.value}")
                    
            # Show all feedback messages
            print(f"\n💬 Feedback Summary:")
            for i, msg in enumerate(captured_feedback, 1):
                print(f"   {i}. {msg}")
                
            # Verify expected flow
            print(f"\n✅ Expected Flow Verification:")
            expected_steps = [
                ("JARVIS detects screen is locked", "requires_unlock" in result and result["requires_unlock"]),
                ("Queues request", result.get("status") == "queued"),
                ("Provides unlock feedback", any("locked" in msg.lower() for msg in captured_feedback)),
                ("Executes command after unlock", len(captured_feedback) > 2)
            ]
            
            for step, success in expected_steps:
                print(f"   {'✓' if success else '✗'} {step}")


async def run_simple_test():
    """Run a simpler test without mocks"""
    print("\n" + "="*70)
    print("🧪 SIMPLE INTEGRATION TEST")
    print("="*70 + "\n")
    
    try:
        # Test basic imports
        from context_intelligence.core.context_manager import get_context_manager
        from context_intelligence.core.screen_state import get_screen_state_detector
        from context_intelligence.core.command_queue import get_command_queue
        from context_intelligence.core.policy_engine import get_policy_engine
        
        print("✅ All core imports successful")
        
        # Test component creation
        context_manager = get_context_manager()
        screen_detector = get_screen_state_detector()
        command_queue = get_command_queue()
        policy_engine = get_policy_engine()
        
        print("✅ All components created successfully")
        
        # Test screen state detection
        screen_state = await screen_detector.get_screen_state()
        print(f"\n📱 Current screen state: {screen_state.state.value}")
        print(f"   Detection method: {screen_state.detection_method.value}")
        print(f"   Confidence: {screen_state.confidence:.2%}")
        
        # Test queue
        stats = await command_queue.get_statistics()
        print(f"\n📊 Queue statistics:")
        print(f"   Total queued: {stats['total_queued']}")
        print(f"   Queue size: {stats['current_queue_size']}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test runner"""
    # Run simple test first
    await run_simple_test()
    
    # Then run full scenario
    print("\n" + "="*70)
    print("Press Enter to run full scenario test...")
    input()
    
    await run_full_scenario()
    
    print("\n🎉 Tests completed!")


if __name__ == "__main__":
    asyncio.run(main())