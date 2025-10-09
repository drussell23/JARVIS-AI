#!/usr/bin/env python3
"""
Test Lock Screen Integration
===========================

Tests the lock screen functionality in the Context Intelligence System.
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_lock_screen():
    """Test locking the screen through Context Intelligence"""
    from context_intelligence.core.unlock_manager import get_unlock_manager
    
    logger.info("Testing screen lock functionality...")
    
    unlock_manager = get_unlock_manager()
    
    # Test locking the screen
    logger.info("Attempting to lock screen...")
    success, message = await unlock_manager.lock_screen("Testing lock functionality")
    
    if success:
        logger.info(f"✅ Screen locked successfully: {message}")
    else:
        logger.error(f"❌ Failed to lock screen: {message}")
    
    return success


async def test_lock_command_execution():
    """Test lock command through the command executor"""
    from context_intelligence.executors.unified_command_executor import get_command_executor
    
    logger.info("\nTesting lock command through executor...")
    
    executor = get_command_executor()
    
    # Test with different lock commands
    test_commands = [
        "lock my screen",
        "lock the mac",
        "lock computer"
    ]
    
    for command in test_commands:
        logger.info(f"\nTesting command: '{command}'")
        
        result = await executor.execute_command(
            command_text=command,
            intent={"action": "lock", "target": "screen"},
            context={}
        )
        
        if result.get("success"):
            logger.info(f"✅ Command executed successfully: {result.get('result', {}).get('message', 'No message')}")
        else:
            logger.error(f"❌ Command failed: {result.get('error', 'Unknown error')}")
        
        # Wait a bit before next test
        await asyncio.sleep(1)


async def test_voice_unlock_integration():
    """Test if Voice Unlock connector is available"""
    try:
        from api.voice_unlock_integration import initialize_voice_unlock, voice_unlock_connector
        
        logger.info("\nTesting Voice Unlock integration...")
        
        # Try to initialize
        await initialize_voice_unlock()
        
        if voice_unlock_connector and voice_unlock_connector.connected:
            logger.info("✅ Voice Unlock connector is available")
            
            # Get status
            status = await voice_unlock_connector.get_status()
            logger.info(f"Voice Unlock status: {status}")
        else:
            logger.info("ℹ️ Voice Unlock connector not available (this is OK - using fallback methods)")
            
    except Exception as e:
        logger.info(f"ℹ️ Voice Unlock not available: {e} (this is OK - using fallback methods)")


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TESTING LOCK SCREEN INTEGRATION")
    print("="*60)
    
    # Test 1: Direct lock via unlock manager
    await test_lock_screen()
    
    # Test 2: Lock via command executor
    await test_lock_command_execution()
    
    # Test 3: Check Voice Unlock integration
    await test_voice_unlock_integration()
    
    print("\n" + "="*60)
    print("TESTS COMPLETE")
    print("="*60)
    print("\nThe lock screen functionality is now integrated into the Context Intelligence System.")
    print("It will work with or without Voice Unlock connector:")
    print("1. If Voice Unlock is available, it uses that")
    print("2. Otherwise, it uses AppleScript (Cmd+Ctrl+Q)")
    print("3. As a fallback, it uses pmset to sleep the display")


if __name__ == "__main__":
    asyncio.run(main())