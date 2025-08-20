#!/usr/bin/env python3
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


Test JARVIS Vision Commands
"""

import asyncio
import os
from backend.voice.jarvis_agent_voice import JARVISAgentVoice


async def test_vision_commands():
    """Test vision commands through JARVIS"""
    print("🤖 Testing JARVIS Vision Commands\n")
    
    # Set API key for testing
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No ANTHROPIC_API_KEY found. Some features may be limited.\n")
    
    # Initialize JARVIS
    jarvis = JARVISAgentVoice(user_name="Sir")
    
    # Test commands
    vision_commands = [
        "Hey JARVIS, what's on my screen?",
        "Check for software updates",
        "What applications can you see?",
        "Start monitoring for updates",
        "Are there any security alerts?",
        "Stop monitoring"
    ]
    
    for command in vision_commands:
        print(f"🎤 You: {command}")
        
        # Process command
        try:
            response = await jarvis._handle_vision_command(command)
            print(f"🤖 JARVIS: {response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        # Small delay between commands
        await asyncio.sleep(1)


async def test_system_integration():
    """Test vision integration with system commands"""
    print("\n🔄 Testing System Integration\n")
    
    jarvis = JARVISAgentVoice(user_name="Sir")
    
    # Test mixed commands
    mixed_commands = [
        "Take a screenshot and check for updates",
        "Show me what's on screen and open Chrome",
        "Monitor my screen for important notifications"
    ]
    
    for command in mixed_commands:
        print(f"🎤 You: {command}")
        
        try:
            # This would go through the full system command handler
            if jarvis.system_control_enabled:
                response = await jarvis._handle_system_command(command)
            else:
                response = "System control is not enabled. Please set ANTHROPIC_API_KEY."
            print(f"🤖 JARVIS: {response}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")
        
        await asyncio.sleep(1)


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🖥️  JARVIS Vision Command Test Suite")
    print("=" * 60)
    print()
    
    # Test vision commands
    await test_vision_commands()
    
    # Test system integration
    await test_system_integration()
    
    print("\n✅ Testing complete!")
    print("\nNote: For full functionality, ensure:")
    print("1. ANTHROPIC_API_KEY is set in backend/.env")
    print("2. Vision dependencies are installed (opencv-python, pytesseract, etc.)")
    print("3. Tesseract is installed: brew install tesseract")


if __name__ == "__main__":
    asyncio.run(main())