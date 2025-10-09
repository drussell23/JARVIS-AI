#!/usr/bin/env python3
"""
Test Lock Command Without Voice Unlock Daemon
============================================

Tests that lock commands work through Context Intelligence
when Voice Unlock daemon is not available.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_lock_without_daemon():
    """Test lock command with daemon unavailable"""
    # Import and clear the connector to simulate no daemon
    import api.voice_unlock_integration as vui
    
    print("\n🔐 Testing Lock Command (Without Voice Unlock Daemon)")
    print("="*60)
    
    # Simulate daemon not connected
    original_connector = vui.voice_unlock_connector
    vui.voice_unlock_connector = None
    
    try:
        # Test lock command
        command = "lock my screen"
        print(f"\nCommand: '{command}'")
        print("Voice Unlock Daemon: NOT CONNECTED (simulated)")
        
        result = await vui.handle_voice_unlock_in_jarvis(command)
        
        print(f"\nResult:")
        print(f"- Success: {result.get('success')}")
        print(f"- Response: {result.get('response')}")
        print(f"- Method: {result.get('method', 'not specified')}")
        
        if result.get('success'):
            print("\n✅ Lock command working via Context Intelligence!")
        else:
            print("\n❌ Lock command failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
                
    finally:
        # Restore original
        vui.voice_unlock_connector = original_connector


async def test_actual_lock():
    """Actually lock the screen to verify it works"""
    print("\n🔒 Actually Locking Screen...")
    print("-"*60)
    
    from context_intelligence.core.unlock_manager import get_unlock_manager
    unlock_manager = get_unlock_manager()
    
    success, message = await unlock_manager.lock_screen("Testing Context Intelligence lock")
    
    if success:
        print(f"✅ Screen locked: {message}")
    else:
        print(f"❌ Lock failed: {message}")


async def main():
    """Run tests"""
    # Test without daemon
    await test_lock_without_daemon()
    
    # Ask before actually locking
    print("\n" + "="*60)
    print("Would you like to actually lock your screen to test?")
    print("(The test will lock your screen using Context Intelligence)")
    # Since we can't get input in this context, just show what would happen
    print("\nTo test actual locking, run:")
    print('python -c "from context_intelligence.core.unlock_manager import get_unlock_manager; import asyncio; asyncio.run(get_unlock_manager().lock_screen())"')


if __name__ == "__main__":
    asyncio.run(main())