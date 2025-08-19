#!/usr/bin/env python3
"""
Test Enhanced Vision Commands
Verifies that JARVIS can understand various vision-related queries
"""

import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from voice.jarvis_agent_voice import JARVISAgentVoice

async def test_vision_commands():
    """Test various vision commands"""
    print("🧠 Testing Enhanced Vision Commands")
    print("=" * 60)
    
    # Initialize JARVIS
    jarvis = JARVISAgentVoice()
    
    # Test commands that should trigger vision
    test_commands = [
        "Hey JARVIS, what am I working on in cursor?",
        "What can you see on my screen?",
        "JARVIS, can you see my screen?",
        "Hey JARVIS, analyze what I'm doing",
        "What am I currently working on?",
        "JARVIS, describe what you see",
        "Tell me what's on my screen",
        "What applications do I have open?",
        "Hey JARVIS, what are you seeing?",
    ]
    
    print(f"Vision enabled: {jarvis.vision_enabled}")
    print(f"Intelligent vision: {jarvis.intelligent_vision_enabled}")
    print(f"API key available: {bool(os.getenv('ANTHROPIC_API_KEY'))}")
    print("-" * 60)
    
    for command in test_commands:
        print(f"\n🎤 User: {command}")
        
        # Test if command is recognized as vision command
        is_vision = jarvis._is_system_command(command)
        print(f"   Detected as system command: {is_vision}")
        
        # Process the command
        response = await jarvis.process_voice_input(command)
        print(f"🤖 JARVIS: {response}")
        print("-" * 40)
        
        # Small delay between tests
        await asyncio.sleep(0.5)

async def test_direct_vision_analysis():
    """Test direct vision analysis"""
    print("\n\n🔍 Testing Direct Vision Analysis")
    print("=" * 60)
    
    try:
        from vision.screen_capture_fallback import capture_with_intelligence
        
        # Test with specific query
        result = capture_with_intelligence(
            query="What is the user currently working on in their development environment?",
            use_claude=True
        )
        
        print(f"Success: {result.get('success')}")
        print(f"Intelligence used: {result.get('intelligence_used')}")
        
        if result.get('analysis'):
            print(f"\nAnalysis:\n{result['analysis']}")
        else:
            print(f"\nError: {result.get('error', 'No analysis available')}")
            
    except Exception as e:
        print(f"Error: {e}")

async def main():
    """Run all tests"""
    await test_vision_commands()
    await test_direct_vision_analysis()
    
    print("\n✅ Test complete!")
    print("\nKey findings:")
    print("1. Vision commands should now be recognized")
    print("2. 'What am I working on' should trigger intelligent analysis")
    print("3. Claude should provide contextual responses about your work")

if __name__ == "__main__":
    asyncio.run(main())