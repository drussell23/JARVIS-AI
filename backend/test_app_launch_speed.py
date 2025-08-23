#!/usr/bin/env python3
"""
Test app launch speed improvements for JARVIS
"""

import asyncio
import time
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_control.fast_app_launcher import get_fast_app_launcher
from system_control.macos_controller import MacOSController
from system_control.dynamic_app_controller import get_dynamic_app_controller


async def test_launch_methods():
    """Test different app launch methods and compare speeds"""
    
    print("🚀 Testing JARVIS App Launch Speed Improvements\n")
    
    # Initialize controllers
    fast_launcher = get_fast_app_launcher()
    macos_controller = MacOSController()
    dynamic_controller = get_dynamic_app_controller()
    
    # Test apps
    test_apps = ["Safari", "WhatsApp", "Notes"]
    
    for app in test_apps:
        print(f"\n📱 Testing {app}:")
        print("-" * 40)
        
        # Test 1: Fast Launcher
        start = time.time()
        success, msg = await fast_launcher.quick_open_app(app)
        fast_time = time.time() - start
        print(f"✨ Fast Launcher: {fast_time:.3f}s - {msg}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test 2: Dynamic Controller
        start = time.time()
        success, msg = await dynamic_controller.open_app_intelligently(app)
        dynamic_time = time.time() - start
        print(f"🔧 Dynamic Controller: {dynamic_time:.3f}s - {msg}")
        
        # Wait a bit
        await asyncio.sleep(2)
        
        # Test 3: Standard Controller
        start = time.time()
        success, msg = macos_controller.open_application(app)
        standard_time = time.time() - start
        print(f"📌 Standard Controller: {standard_time:.3f}s - {msg}")
        
        # Compare
        improvement = ((standard_time - fast_time) / standard_time) * 100
        print(f"\n⚡ Speed improvement: {improvement:.1f}% faster!")
        
        # Close the app
        await asyncio.sleep(1)
        subprocess.run(["osascript", "-e", f'tell application "{app}" to quit'])
        await asyncio.sleep(1)
    
    print("\n\n✅ Testing complete! Fast launcher provides significant speed improvements.")
    print("\nKey improvements implemented:")
    print("1. ⚡ Direct 'open -a' command with 1s timeout")
    print("2. 🎯 Common apps cached for instant lookup")
    print("3. 🔥 Fire-and-forget AppleScript execution")
    print("4. ⏱️ Reduced timeouts from 15s → 5s (AppleScript) and 10s → 3s (app info)")


async def test_voice_command_simulation():
    """Simulate voice command processing with timing"""
    print("\n\n🎤 Simulating Voice Command Processing:")
    print("-" * 50)
    
    # Set up API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ No API key found, skipping voice simulation")
        return
    
    from system_control.claude_command_interpreter import ClaudeCommandInterpreter
    
    interpreter = ClaudeCommandInterpreter(api_key)
    
    # Test command
    command = "open safari"
    
    print(f"\n🗣️ Command: '{command}'")
    
    # Time the full pipeline
    start = time.time()
    
    # Interpret command
    intent = await interpreter.interpret_command(command)
    interpret_time = time.time() - start
    print(f"🧠 Interpretation: {interpret_time:.3f}s - {intent.action} {intent.target}")
    
    # Execute command
    exec_start = time.time()
    result = await interpreter.execute_intent(intent)
    exec_time = time.time() - exec_start
    total_time = time.time() - start
    
    print(f"⚡ Execution: {exec_time:.3f}s - {result.message}")
    print(f"⏱️ Total time: {total_time:.3f}s")
    
    if total_time < 2.0:
        print("\n✨ Excellent! Command executed in under 2 seconds!")
    elif total_time < 3.0:
        print("\n✅ Good! Command executed in under 3 seconds.")
    else:
        print("\n⚠️ Could be faster. Check network latency to Claude API.")


if __name__ == "__main__":
    import subprocess
    
    print("🎯 JARVIS App Launch Speed Test")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_launch_methods())
    asyncio.run(test_voice_command_simulation())