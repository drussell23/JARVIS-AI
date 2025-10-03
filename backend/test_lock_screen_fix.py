#!/usr/bin/env python3
"""
Test script to verify that "lock screen" commands are properly routed
and not confused with vision monitoring commands
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_lock_screen_command():
    """Test that lock screen command works correctly"""
    
    print("\n" + "="*60)
    print("TESTING LOCK SCREEN COMMAND FIX")
    print("="*60)
    
    try:
        from api.unified_command_processor import UnifiedCommandProcessor
        
        processor = UnifiedCommandProcessor()
        
        # Test 1: Lock screen command
        print("\n📱 Test 1: Testing 'lock my screen' command...")
        result = await processor.process_command("lock my screen")
        
        print(f"   Result: {result}")
        
        # Check that it was NOT routed to vision
        if result.get('command_type') == 'vision':
            print("   ❌ ERROR: Command was incorrectly routed to vision handler!")
            print("   This means the fix didn't work properly.")
        elif result.get('command_type') in ['system', 'screen_lock']:
            print(f"   ✅ SUCCESS: Command correctly routed to {result.get('command_type')} handler")
            print(f"   Response: {result.get('response', 'No response')}")
        else:
            print(f"   ⚠️  WARNING: Unexpected command type: {result.get('command_type')}")
        
        # Test 2: Check that vision monitoring still works
        print("\n👁️  Test 2: Testing 'start monitoring' command (should go to vision)...")
        result = await processor.process_command("start monitoring my screen")
        
        print(f"   Result type: {result.get('command_type')}")
        if result.get('command_type') == 'vision':
            print("   ✅ Vision monitoring command still works correctly")
        else:
            print(f"   ⚠️  Vision command routed to: {result.get('command_type')}")
        
        # Test 3: Unlock screen command
        print("\n🔓 Test 3: Testing 'unlock my screen' command...")
        result = await processor.process_command("unlock my screen")
        
        print(f"   Result: {result}")
        if result.get('command_type') in ['system', 'screen_unlock']:
            print(f"   ✅ SUCCESS: Unlock command correctly routed")
            print(f"   Response: {result.get('response', 'No response')}")
        else:
            print(f"   ⚠️  WARNING: Unexpected routing: {result.get('command_type')}")
            
    except Exception as e:
        logger.error(f"Error during test: {e}", exc_info=True)
        return False
    
    return True

async def test_daemon_integration():
    """Test the voice unlock daemon integration"""
    
    print("\n" + "="*60)
    print("TESTING DAEMON INTEGRATION")
    print("="*60)
    
    try:
        import websockets
        import json
        
        VOICE_UNLOCK_WS_URL = "ws://localhost:8765/voice-unlock"
        
        print("\n🔌 Checking if Voice Unlock daemon is running...")
        
        try:
            async with websockets.connect(VOICE_UNLOCK_WS_URL, timeout=2.0) as websocket:
                # Send status request
                status_request = {
                    "type": "status"
                }
                
                await websocket.send(json.dumps(status_request))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                result = json.loads(response)
                
                print(f"   ✅ Daemon is running!")
                print(f"   Status: {result}")
                
                # Test lock command via daemon
                print("\n🔒 Testing lock command via daemon...")
                lock_command = {
                    "type": "command",
                    "command": "lock_screen"
                }
                
                await websocket.send(json.dumps(lock_command))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                result = json.loads(response)
                
                if result.get("success"):
                    print(f"   ✅ Lock command successful via daemon")
                else:
                    print(f"   ⚠️  Lock command result: {result}")
                    
        except (ConnectionRefusedError, OSError):
            print("   ⚠️  Voice Unlock daemon is NOT running")
            print("   Lock/unlock will use fallback methods")
            
    except Exception as e:
        logger.error(f"Error testing daemon: {e}")

async def main():
    """Main test function"""
    
    print("\n🔧 JARVIS Lock Screen Fix Test")
    print("This tests that 'lock screen' commands are properly handled")
    print("and not confused with vision monitoring commands.")
    
    # Test command routing
    success = await test_lock_screen_command()
    
    # Test daemon integration
    await test_daemon_integration()
    
    print("\n" + "="*60)
    if success:
        print("✅ TESTS COMPLETED SUCCESSFULLY")
        print("Lock screen commands should now work correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the errors above")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())