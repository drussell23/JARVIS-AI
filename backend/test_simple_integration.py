#!/usr/bin/env python3
"""
Simple test of JARVIS integration
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jarvis_integrated_assistant import JARVISIntegratedAssistant


async def main():
    print("🤖 Testing JARVIS Integrated Vision & Control")
    print("=" * 50)
    
    # Check API
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY")
        return
    
    # Initialize
    jarvis = JARVISIntegratedAssistant("User")
    
    # Test basic vision with proactive info
    print("\n📺 Asking JARVIS about the screen...")
    response = await jarvis.process_vision_command("What can you see?")
    
    print("\n🗣️ JARVIS says:")
    print("-" * 50)
    print(response.verbal_response)
    print("-" * 50)
    
    print(f"\n📊 Context:")
    print(f"• Windows detected: {response.visual_context.get('window_count', 0)}")
    print(f"• Notifications: {response.visual_context.get('notification_count', 0)}")
    print(f"• Has urgent items: {response.visual_context.get('has_urgent', False)}")
    
    print(f"\n⚡ Available Actions:")
    for action in response.available_actions[:5]:
        print(f"  • {action}")
    
    print("\n✅ Integration test complete!")


if __name__ == "__main__":
    asyncio.run(main())