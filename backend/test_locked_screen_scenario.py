#!/usr/bin/env python3
"""
Test Locked Screen Scenario with Enhanced Context
================================================

Simulates the PRD scenario where screen is locked and user asks to open Safari
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime

# Test imports
from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor

class TestWebSocket:
    """Mock WebSocket that captures messages"""
    def __init__(self):
        self.messages = []
    
    async def send_json(self, data):
        self.messages.append(data)
        print(f"\n📡 WebSocket Message:")
        print(f"   Type: {data.get('type')}")
        if data.get('message'):
            print(f"   Message: {data.get('message')}")
        if data.get('status'):
            print(f"   Status: {data.get('status')}")
        if data.get('steps'):
            print(f"   Steps: {len(data.get('steps', []))} recorded")

async def simulate_locked_screen_scenario():
    """Simulate the exact PRD scenario"""
    print("\n" + "="*80)
    print("🔐 Simulating Locked Screen Scenario")
    print("="*80)
    
    # Step 1: Lock the screen
    print("\n1️⃣ Locking the screen...")
    lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
    result = subprocess.run(lock_cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("   ✅ Screen locked successfully")
    else:
        print("   ⚠️  Could not lock screen (may need permissions)")
        print("   Continuing test anyway...")
    
    # Wait a moment for lock to complete
    await asyncio.sleep(2)
    
    # Step 2: Create test components
    print("\n2️⃣ Setting up test components...")
    processor = UnifiedCommandProcessor()
    context_handler = wrap_with_enhanced_context(processor)
    websocket = TestWebSocket()
    
    # Step 3: Send the command
    test_command = "JARVIS, open Safari and search for dogs"
    print(f"\n3️⃣ Sending command: '{test_command}'")
    print("   Expected flow:")
    print("   - JARVIS detects locked screen")
    print("   - Sends feedback: 'I see your screen is locked. I'll unlock it now...'")
    print("   - Attempts to unlock screen")
    print("   - Executes command after unlock")
    
    # Process command
    try:
        result = await context_handler.process_with_context(
            test_command, 
            websocket
        )
        
        print("\n4️⃣ Command processed!")
        
        # Check WebSocket messages
        print("\n📨 WebSocket Messages Received:")
        for i, msg in enumerate(websocket.messages, 1):
            print(f"\n   Message {i}:")
            print(f"   - Type: {msg.get('type')}")
            print(f"   - Message: {msg.get('message', 'N/A')}")
            print(f"   - Status: {msg.get('status', 'N/A')}")
        
        # Check result
        print("\n📋 Final Result:")
        print(f"   Success: {result.get('success')}")
        print(f"   Response: {result.get('response')}")
        
        if result.get('execution_steps'):
            print("\n📊 Execution Steps:")
            for i, step in enumerate(result['execution_steps'], 1):
                print(f"   {i}. {step['step']}")
                
        # Verify expected behavior
        print("\n✅ Verification:")
        
        # Check if context message was sent
        context_messages = [m for m in websocket.messages if m.get('type') == 'context_update']
        if context_messages:
            first_msg = context_messages[0]['message']
            if "I see your screen is locked" in first_msg and "I'll unlock it now" in first_msg:
                print("   ✅ Correct pre-unlock feedback message sent")
            else:
                print(f"   ❌ Unexpected feedback: {first_msg}")
        else:
            print("   ❌ No context update messages found")
            
        # Check final response
        if "unlocked" in result.get('response', '').lower():
            print("   ✅ Final response mentions unlock action")
        else:
            print("   ⚠️  Final response doesn't mention unlock")
            
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

async def test_screen_detection():
    """Test screen lock detection"""
    print("\n" + "="*80)
    print("🔍 Testing Screen Lock Detection")
    print("="*80)
    
    from api.direct_unlock_handler_fixed import check_screen_locked_direct
    
    # Check current state
    is_locked = await check_screen_locked_direct()
    print(f"\nScreen is currently: {'LOCKED' if is_locked else 'UNLOCKED'}")
    
    return is_locked

if __name__ == "__main__":
    print("🚀 Starting Enhanced Context Locked Screen Test")
    print("\n⚠️  Note: This test will lock your screen!")
    print("Press Ctrl+C to cancel, or wait 3 seconds to continue...")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Test cancelled")
        exit(0)
    
    # Run tests
    asyncio.run(simulate_locked_screen_scenario())
    
    # Also test detection separately
    asyncio.run(test_screen_detection())
    
    print("\n✅ All tests completed!")