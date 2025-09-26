#!/usr/bin/env python3
"""
Test JARVIS Complete Locked Screen Flow
=======================================

Tests the full scenario from the PRD:
1. Screen is locked
2. User says "JARVIS, open Safari and search for dogs"
3. JARVIS detects lock, provides feedback, unlocks, then executes
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime

# Import JARVIS components
from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor

class FlowTestWebSocket:
    """WebSocket that captures the complete flow"""
    def __init__(self):
        self.messages = []
        self.flow_steps = []
        
    async def send_json(self, data):
        self.messages.append(data)
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Log the flow
        msg_type = data.get('type', '')
        message = data.get('message', '')
        
        if msg_type == 'context_update':
            self.flow_steps.append(f"[{timestamp}] CONTEXT: {message}")
            if data.get('status'):
                self.flow_steps.append(f"[{timestamp}] STATUS: {data['status']}")
        elif msg_type == 'response':
            self.flow_steps.append(f"[{timestamp}] RESPONSE: {data.get('text', message)}")

async def test_complete_flow():
    """Test the complete locked screen flow"""
    print("\n" + "="*80)
    print("🚀 Testing Complete JARVIS Locked Screen Flow")
    print("="*80)
    
    print("\n📋 Scenario: Mac is locked, user says 'JARVIS, open Safari and search for dogs'")
    print("   Expected flow:")
    print("   1. JARVIS detects locked state")
    print("   2. Says: 'I see your screen is locked. I'll unlock it now...'") 
    print("   3. Attempts unlock")
    print("   4. Executes command")
    print("   5. Provides final feedback")
    
    # Step 1: Lock the screen
    print("\n1️⃣ Locking the screen...")
    lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
    subprocess.run(lock_cmd, shell=True)
    
    # Wait for lock
    print("   Waiting 3 seconds for lock to complete...")
    await asyncio.sleep(3)
    
    # Step 2: Create JARVIS components
    print("\n2️⃣ Setting up JARVIS with enhanced context...")
    processor = UnifiedCommandProcessor()
    context_handler = wrap_with_enhanced_context(processor)
    websocket = FlowTestWebSocket()
    
    # Step 3: Send the command
    command = "JARVIS, open Safari and search for dogs"
    print(f"\n3️⃣ Sending command: '{command}'")
    
    try:
        # Process command
        result = await context_handler.process_with_context(command, websocket)
        
        print("\n4️⃣ Command completed!")
        
        # Show the complete flow
        print("\n📝 Complete Flow Timeline:")
        for step in websocket.flow_steps:
            print(f"   {step}")
        
        # Analyze results
        print("\n📊 Analysis:")
        
        # Check for lock detection message
        lock_detected = any("I see your screen is locked" in msg.get('message', '') 
                           for msg in websocket.messages 
                           if msg.get('type') == 'context_update')
        
        print(f"   ✅ Lock detected and announced: {lock_detected}")
        
        # Check for unlock intent
        unlock_intent = any("I'll unlock it now" in msg.get('message', '')
                           for msg in websocket.messages)
        print(f"   ✅ Unlock intent communicated: {unlock_intent}")
        
        # Check execution steps
        if result.get('execution_steps'):
            print(f"   ✅ Execution steps tracked: {len(result['execution_steps'])} steps")
            for i, step in enumerate(result['execution_steps'], 1):
                print(f"      {i}. {step['step']}")
        
        # Final response
        print(f"\n5️⃣ Final Response:")
        print(f"   '{result.get('response', 'No response')}'")
        
        # Overall success
        if lock_detected and unlock_intent:
            print("\n✅ SUCCESS! JARVIS properly handled the locked screen scenario!")
        else:
            print("\n❌ ISSUE: JARVIS did not follow the expected flow")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def test_quick_check():
    """Quick check of screen detection"""
    print("\n" + "="*40)
    print("Quick Screen Detection Check")
    print("="*40)
    
    from api.direct_unlock_handler_fixed import check_screen_locked_direct
    
    is_locked = await check_screen_locked_direct()
    print(f"\nScreen is currently: {'LOCKED' if is_locked else 'UNLOCKED'}")

if __name__ == "__main__":
    print("🎯 JARVIS Locked Screen Flow Test")
    print("\n⚠️  This test will lock your screen!")
    print("Press Ctrl+C to cancel, or wait 3 seconds...")
    
    try:
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n❌ Cancelled")
        exit(0)
    
    # Run quick check first
    asyncio.run(test_quick_check())
    
    # Run complete flow test
    asyncio.run(test_complete_flow())