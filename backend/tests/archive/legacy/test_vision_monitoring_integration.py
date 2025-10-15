#!/usr/bin/env python3
"""
Test Vision Monitoring Integration
Tests that "enable screen monitoring" triggers purple indicator and vision status update
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.vision_command_handler import vision_command_handler

async def test_enable_monitoring_command():
    """Test that 'enable screen monitoring capabilities' works correctly"""
    print("\n🧪 TESTING VISION MONITORING INTEGRATION")
    print("=" * 60)
    
    # Initialize vision command handler
    print("\n1️⃣ Initializing vision command handler...")
    await vision_command_handler.initialize_intelligence()
    
    # Test the command
    command = "enable screen monitoring capabilities"
    print(f"\n2️⃣ Testing command: '{command}'")
    
    result = await vision_command_handler.handle_command(command)
    
    print("\n📊 Result:")
    print(f"   Handled: {result.get('handled', False)}")
    print(f"   Response: {result.get('response', 'No response')[:200]}...")
    print(f"   Monitoring Active: {result.get('monitoring_active', False)}")
    print(f"   Pure Intelligence: {result.get('pure_intelligence', False)}")
    
    # Check if monitoring is active
    if vision_command_handler.monitoring_active:
        print("\n✅ Monitoring is now ACTIVE!")
        print("   - Purple indicator should be visible")
        print("   - Vision status should show 'connected'")
        
        # Wait a bit to see the monitoring in action
        print("\n⏳ Monitoring active for 10 seconds...")
        await asyncio.sleep(10)
        
        # Stop monitoring
        print("\n3️⃣ Stopping monitoring...")
        stop_result = await vision_command_handler.handle_command("stop monitoring")
        print(f"   Stop result: {stop_result.get('response', 'No response')[:100]}...")
        
    else:
        print("\n❌ Monitoring failed to activate")
    
    print("\n✅ Test completed!")

async def test_monitoring_patterns():
    """Test various monitoring command patterns"""
    print("\n🔍 Testing Various Monitoring Commands")
    print("=" * 60)
    
    test_commands = [
        "enable screen monitoring capabilities",
        "start monitoring my screen",
        "turn on monitoring",
        "activate vision monitoring",
        "enable monitoring",
        "begin monitoring my desktop"
    ]
    
    # Initialize once
    await vision_command_handler.initialize_intelligence()
    
    for command in test_commands:
        print(f"\n📝 Testing: '{command}'")
        
        # Quick pattern check
        is_monitoring = any(phrase in command.lower() for phrase in [
            'start monitoring', 'enable monitoring', 'monitor my screen',
            'enable screen monitoring', 'monitoring capabilities',
            'turn on monitoring', 'activate monitoring', 'begin monitoring'
        ])
        
        print(f"   Pattern match: {'YES' if is_monitoring else 'NO'}")
        
        # Reset monitoring state
        vision_command_handler.monitoring_active = False
        
        # Test command
        result = await vision_command_handler.handle_command(command)
        print(f"   Handled: {result.get('handled', False)}")
        print(f"   Monitoring active: {vision_command_handler.monitoring_active}")
        
        # Stop if activated
        if vision_command_handler.monitoring_active:
            await vision_command_handler.handle_command("stop monitoring")
            await asyncio.sleep(1)

if __name__ == "__main__":
    print("Choose test:")
    print("1. Test 'enable screen monitoring capabilities' command")
    print("2. Test various monitoring command patterns")
    
    choice = input("\nEnter choice (1-2): ").strip()
    
    if choice == "1":
        asyncio.run(test_enable_monitoring_command())
    elif choice == "2":
        asyncio.run(test_monitoring_patterns())
    else:
        print("Invalid choice")