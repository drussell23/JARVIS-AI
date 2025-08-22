#!/usr/bin/env python3
"""
Test Dynamic App Control
Tests the enhanced app control that works with any macOS application
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_control.dynamic_app_controller import get_dynamic_app_controller
from system_control.claude_command_interpreter import ClaudeCommandInterpreter


async def test_dynamic_app_control():
    """Test dynamic app control features"""
    print("🧪 Testing Dynamic App Control")
    print("=" * 50)
    
    controller = get_dynamic_app_controller()
    
    # Test 1: List all running apps
    print("\n📱 Currently Running Applications:")
    apps = controller.get_all_running_apps()
    for app in apps:
        if app["visible"]:
            print(f"  - {app['name']} (PID: {app['pid']})")
    
    # Test 2: Find WhatsApp
    print("\n🔍 Testing App Detection:")
    test_apps = ["whatsapp", "WhatsApp", "what's app", "whats app"]
    
    for test_name in test_apps:
        found = controller.find_app_by_fuzzy_name(test_name)
        if found:
            print(f"  ✅ Found '{test_name}' → {found['name']}")
        else:
            print(f"  ❌ Not found: '{test_name}'")
    
    # Test 3: Close WhatsApp (if found)
    print("\n🔧 Testing App Control:")
    whatsapp = controller.find_app_by_fuzzy_name("whatsapp")
    
    if whatsapp:
        print(f"  Found WhatsApp as: {whatsapp['name']}")
        success, message = await controller.close_app_intelligently("whatsapp")
        print(f"  Close result: {message}")
    else:
        print("  WhatsApp not running - testing with another app")
        # Find any visible app to test
        for app in apps:
            if app["visible"] and app["name"] not in ["Finder", "JARVIS"]:
                print(f"  Testing with: {app['name']}")
                success, message = await controller.close_app_intelligently(app["name"])
                print(f"  Close result: {message}")
                break
    
    # Test 4: Command interpreter integration
    print("\n🤖 Testing Command Interpreter:")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        interpreter = ClaudeCommandInterpreter(api_key)
        
        # Test closing WhatsApp through interpreter
        intent = await interpreter.interpret_command("close whatsapp")
        print(f"  Intent: {intent.action} → {intent.target}")
        print(f"  Confidence: {intent.confidence}")
        
        if intent.confidence > 0.5:
            result = await interpreter.execute_intent(intent)
            print(f"  Execution: {result.message}")
    else:
        print("  ⚠️  ANTHROPIC_API_KEY not set, skipping interpreter test")
    
    # Test 5: App suggestions
    print("\n💡 Testing App Suggestions:")
    suggestions = controller.get_app_suggestions("what")
    if suggestions:
        print(f"  Suggestions for 'what': {', '.join(suggestions)}")
    else:
        print("  No suggestions found")
    
    print("\n✅ Dynamic app control test complete!")


async def test_specific_app(app_name: str):
    """Test control of a specific app"""
    print(f"\n🎯 Testing control of '{app_name}'")
    
    controller = get_dynamic_app_controller()
    
    # Try to close the app
    success, message = await controller.close_app_intelligently(app_name)
    print(f"Result: {message}")
    
    return success


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific app
        app_name = " ".join(sys.argv[1:])
        asyncio.run(test_specific_app(app_name))
    else:
        # Run full test suite
        asyncio.run(test_dynamic_app_control())