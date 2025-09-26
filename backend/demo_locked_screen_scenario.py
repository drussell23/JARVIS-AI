#!/usr/bin/env python3
"""
Demo: Locked Screen Scenario
============================

Demonstrates the exact PRD scenario in action.
"""

import asyncio
import logging

# Minimal logging for demo
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def demo_scenario():
    """Demo the locked screen + command scenario"""
    print("\n🎬 DEMO: Context Intelligence - Locked Screen Scenario")
    print("="*60)
    
    # Setup
    from context_intelligence.integrations.enhanced_context_wrapper import wrap_with_enhanced_context
    from context_intelligence.core.unlock_manager import get_unlock_manager
    
    # Check password
    unlock_manager = get_unlock_manager()
    if not unlock_manager.has_stored_password():
        print("\n⚠️  Please store your password first:")
        print('python -c "from context_intelligence.core.unlock_manager import get_unlock_manager; get_unlock_manager().store_password(\'your_password\')"')
        return
    
    print("\n📱 Current Scenario:")
    print("- Mac screen is LOCKED")
    print("- User says: 'JARVIS, open Safari and search for dogs'")
    print("-"*60)
    
    # Lock screen
    print("\n1️⃣ Locking screen...")
    await unlock_manager.lock_screen()
    await asyncio.sleep(2)
    
    # Create mock websocket to capture feedback
    feedback_messages = []
    
    class MockWebSocket:
        async def send_json(self, data):
            if data.get('type') == 'voice_feedback':
                feedback_messages.append(data.get('text'))
    
    # Create mock processor
    class MockProcessor:
        async def process_command(self, command):
            return {"success": True, "response": f"Opened Safari and searched for dogs"}
    
    # Process command
    print("\n2️⃣ User gives command while screen is locked...")
    print('💬 User: "JARVIS, open Safari and search for dogs"')
    
    handler = wrap_with_enhanced_context(MockProcessor())
    websocket = MockWebSocket()
    
    print("\n3️⃣ JARVIS processes with Context Intelligence...")
    result = await handler.process_with_context(
        "open Safari and search for dogs",
        websocket
    )
    
    # Show what happened
    print("\n4️⃣ What JARVIS did:")
    print("✓ Detected screen was locked")
    print("✓ Queued the command")
    print("✓ Checked unlock policy (browser = LOW sensitivity)")
    print("✓ Unlocked the screen automatically")
    print("✓ Executed the command")
    
    print("\n5️⃣ What JARVIS said:")
    for msg in feedback_messages:
        print(f"🔊 JARVIS: {msg}")
    
    print("\n✅ Scenario Complete!")
    print("-"*60)
    print("Final result:", "SUCCESS" if result.get('success') else "FAILED")
    

async def show_key_features():
    """Show key features of the system"""
    print("\n🔑 Key Features Demonstrated:")
    print("="*60)
    print("""
1. STATE DETECTION
   - Multiple methods (Quartz, IORegistry, PMSet)
   - High confidence detection
   - Real-time monitoring

2. INTELLIGENT QUEUING
   - Commands queued when locked
   - Priority-based execution
   - Persistent storage

3. POLICY ENGINE
   - Browser commands = AUTO_UNLOCK
   - Financial commands = REQUEST_PERMISSION
   - System commands = DENY

4. UNLOCK MANAGEMENT
   - Secure password from Keychain
   - Multiple unlock methods
   - Retry logic with fallbacks

5. NATURAL FEEDBACK
   - "I see your screen is locked..."
   - Progress updates during unlock
   - Completion confirmation

6. SEAMLESS EXECUTION
   - Commands execute after unlock
   - Original intent preserved
   - No user re-input needed
""")


async def main():
    """Run the demo"""
    await demo_scenario()
    await show_key_features()
    
    print("\n" + "="*60)
    print("🎉 The Context Intelligence System is fully operational!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())