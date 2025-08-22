#!/usr/bin/env python3
"""
Test script to verify "close whatsapp" command works correctly
"""

import asyncio
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from voice.jarvis_agent_voice import JARVISAgentVoice


async def test_close_whatsapp():
    """Test the close WhatsApp command"""
    print("🧪 Testing JARVIS Close WhatsApp Command")
    print("=" * 50)
    
    # Initialize JARVIS voice agent
    jarvis = JARVISAgentVoice(user_name="Sir")
    
    # Test commands
    test_commands = [
        "close whatsapp",
        "close WhatsApp",
        "quit whatsapp",
        "can you close whatsapp",
        "please close whatsapp"
    ]
    
    for command in test_commands:
        print(f"\n📢 Testing: '{command}'")
        
        # Process the command
        response = await jarvis.process_voice_input(command)
        
        print(f"🤖 JARVIS: {response}")
        
        # Add a small delay between tests
        await asyncio.sleep(1)
    
    print("\n✅ Test complete!")


async def test_direct_controller():
    """Test the dynamic app controller directly"""
    print("\n🔧 Testing Dynamic App Controller Directly")
    print("=" * 50)
    
    from system_control.dynamic_app_controller import get_dynamic_app_controller
    
    controller = get_dynamic_app_controller()
    
    # Check if WhatsApp is running
    app_info = controller.find_app_by_fuzzy_name("whatsapp")
    
    if app_info:
        print(f"✅ Found WhatsApp: {app_info['name']} (PID: {app_info['pid']})")
        
        # Try to close it
        success, message = await controller.close_app_intelligently("whatsapp")
        print(f"{'✅' if success else '❌'} Result: {message}")
    else:
        print("❌ WhatsApp not found running")
        
        # List all running apps
        apps = controller.get_all_running_apps()
        visible_apps = [app for app in apps if app["visible"]]
        print(f"\n📱 Visible apps ({len(visible_apps)}):")
        for app in visible_apps[:10]:  # Show first 10
            print(f"  • {app['name']}")


if __name__ == "__main__":
    print("🚀 JARVIS Close WhatsApp Test")
    print("\nThis test will verify that JARVIS can properly close WhatsApp")
    print("Make sure WhatsApp is running before starting the test")
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  WARNING: ANTHROPIC_API_KEY not set")
        print("System control features require the API key")
    
    # Run tests
    asyncio.run(test_close_whatsapp())
    asyncio.run(test_direct_controller())