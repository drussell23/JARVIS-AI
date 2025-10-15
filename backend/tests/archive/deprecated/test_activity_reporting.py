#!/usr/bin/env python3
"""Test activity reporting commands"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.activity_reporting_commands import is_activity_reporting_command
from api.vision_command_handler import vision_command_handler

async def test_activity_detection():
    """Test if activity reporting commands are detected"""
    print("🔍 Testing Activity Reporting Command Detection")
    print("=" * 60)
    
    test_commands = [
        "yes report any specific changes or activities I observe",
        "report changes across my workspace",
        "tell me what changes",
        "monitor my desktop activity",
        "what's happening on my screen?",
        "workspace insights",
        "start monitoring my screen",
        "what do you see?"
    ]
    
    print("\n📋 Command Detection Results:")
    print("-" * 60)
    
    for command in test_commands:
        is_activity = is_activity_reporting_command(command)
        print(f"Command: '{command}'")
        print(f"  → Activity reporting: {'YES ✅' if is_activity else 'NO ❌'}")
        print()
    
    # Test with vision command handler
    print("\n🧠 Testing with Vision Command Handler:")
    print("-" * 60)
    
    # Initialize if needed
    if not vision_command_handler.intelligence:
        print("Initializing vision intelligence...")
        await vision_command_handler.initialize_intelligence()
    
    # Test a command
    test_command = "yes report any specific changes"
    print(f"\nTesting command: '{test_command}'")
    
    try:
        result = await vision_command_handler.handle_command(test_command)
        print(f"Handled: {result.get('handled', False)}")
        print(f"Response: {result.get('response', 'No response')[:200]}...")
        print(f"Monitoring active: {result.get('monitoring_active', False)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_activity_detection())