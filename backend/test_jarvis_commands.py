#!/usr/bin/env python3
"""Test JARVIS commands directly"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from voice.jarvis_agent_voice import JARVISAgentVoice

async def test_commands():
    """Test various JARVIS commands"""
    print("Initializing JARVIS Agent...")
    
    jarvis = JARVISAgentVoice()
    jarvis.running = True  # Activate JARVIS
    
    # Test commands
    test_cases = [
        "open chrome",
        "set volume to 50%",
        "take a screenshot",
        "close safari",
        "list open applications",
    ]
    
    for command in test_cases:
        print(f"\n{'='*50}")
        print(f"Command: {command}")
        try:
            response = await jarvis.process_voice_input(command)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_commands())