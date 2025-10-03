#!/usr/bin/env python3
"""
Final test for lock/unlock screen command routing fix
Ensures "lock my screen" and "unlock my screen" go to system commands, not vision/monitoring
"""

import asyncio
import sys
import os
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_lock_unlock_routing():
    """Test that lock/unlock commands are properly routed"""
    
    print("\n" + "="*70)
    print("🔐 TESTING LOCK/UNLOCK SCREEN COMMAND ROUTING FIX")
    print("="*70)
    
    success_count = 0
    total_tests = 0
    
    try:
        # Test 1: Check vision command detection
        print("\n1️⃣ Testing Vision Command Detection...")
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        
        chatbot = ClaudeVisionChatbot()
        
        # These should NOT be vision commands
        test_commands = [
            ("lock my screen", False, "Should NOT be vision"),
            ("unlock my screen", False, "Should NOT be vision"),
            ("lock the screen", False, "Should NOT be vision"),
            ("please lock screen", False, "Should NOT be vision"),
            ("what's on my screen", True, "Should be vision"),
            ("show me the screen", True, "Should be vision"),
            ("monitor my screen", True, "Should be vision/monitoring")
        ]
        
        for cmd, should_be_vision, desc in test_commands:
            total_tests += 1
            is_vision = chatbot.is_vision_command(cmd)
            
            if is_vision == should_be_vision:
                print(f"   ✅ '{cmd}': {desc} - CORRECT")
                success_count += 1
            else:
                print(f"   ❌ '{cmd}': {desc} - WRONG (got {is_vision})")
        
        # Test 2: Check monitoring command detection
        print("\n2️⃣ Testing Monitoring Command Detection...")
        
        monitoring_tests = [
            ("lock my screen", False, "Should NOT be monitoring"),
            ("unlock my screen", False, "Should NOT be monitoring"),
            ("start monitoring", True, "Should be monitoring"),
            ("monitor my screen", True, "Should be monitoring"),
            ("watch my screen", True, "Should be monitoring")
        ]
        
        for cmd, should_be_monitoring, desc in monitoring_tests:
            total_tests += 1
            is_monitoring = await chatbot._is_monitoring_command(cmd)
            
            if is_monitoring == should_be_monitoring:
                print(f"   ✅ '{cmd}': {desc} - CORRECT")
                success_count += 1
            else:
                print(f"   ❌ '{cmd}': {desc} - WRONG (got {is_monitoring})")
        
        # Test 3: Check unified command processor routing
        print("\n3️⃣ Testing Unified Command Processor Routing...")
        from api.unified_command_processor import UnifiedCommandProcessor
        
        processor = UnifiedCommandProcessor()
        
        # Test lock command
        print("\n   Testing 'lock my screen'...")
        total_tests += 1
        result = await processor.process_command("lock my screen")
        
        command_type = result.get('command_type', 'unknown')
        response = result.get('response', '')
        
        # Check if it's routed to system (not vision)
        if command_type in ['system', 'screen_lock'] and 'monitoring' not in response.lower():
            print(f"   ✅ Correctly routed to {command_type}")
            print(f"   Response: {response[:100]}...")
            success_count += 1
        else:
            print(f"   ❌ WRONG! Routed to: {command_type}")
            print(f"   Response: {response[:200]}...")
            if 'monitoring' in response.lower() or 'purple' in response.lower():
                print("   ⚠️  ERROR: Still being treated as monitoring command!")
        
        # Test unlock command
        print("\n   Testing 'unlock my screen'...")
        total_tests += 1
        result = await processor.process_command("unlock my screen")
        
        command_type = result.get('command_type', 'unknown')
        response = result.get('response', '')
        
        if command_type in ['system', 'screen_unlock'] and 'monitoring' not in response.lower():
            print(f"   ✅ Correctly routed to {command_type}")
            print(f"   Response: {response[:100]}...")
            success_count += 1
        else:
            print(f"   ❌ WRONG! Routed to: {command_type}")
            print(f"   Response: {response[:200]}...")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return False
    
    # Summary
    print("\n" + "="*70)
    print(f"📊 RESULTS: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ ALL TESTS PASSED! Lock/unlock should now work correctly!")
        return True
    else:
        print(f"❌ {total_tests - success_count} tests failed. Please review.")
        return False

async def test_daemon_lock_unlock():
    """Test actual lock/unlock functionality"""
    
    print("\n" + "="*70)
    print("🔒 TESTING ACTUAL LOCK/UNLOCK FUNCTIONALITY")
    print("="*70)
    
    print("\n⚠️  This will actually lock your screen!")
    print("Press Enter to continue or Ctrl+C to cancel...")
    input()
    
    try:
        from api.simple_unlock_handler import handle_unlock_command
        
        # Test lock
        print("\n🔒 Attempting to lock screen...")
        result = await handle_unlock_command("lock my screen")
        
        if result.get('success'):
            print(f"   ✅ Lock command successful!")
            print(f"   Response: {result.get('response')}")
        else:
            print(f"   ❌ Lock failed: {result.get('response')}")
            
    except Exception as e:
        logger.error(f"Error testing lock/unlock: {e}")

async def main():
    """Main test function"""
    
    print("\n🔧 JARVIS Lock/Unlock Screen Fix - Final Test")
    print("This ensures 'lock screen' commands work correctly")
    print("and are not confused with vision monitoring.")
    
    # Test routing
    routing_success = await test_lock_unlock_routing()
    
    if routing_success:
        print("\n✅ Routing tests passed!")
        print("\nWould you like to test actual lock/unlock? (y/n): ", end='')
        
        choice = input().strip().lower()
        if choice == 'y':
            await test_daemon_lock_unlock()
    else:
        print("\n❌ Fix routing issues before testing actual lock/unlock")

if __name__ == "__main__":
    asyncio.run(main())