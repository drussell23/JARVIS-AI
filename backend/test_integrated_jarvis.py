#!/usr/bin/env python3
"""
Test JARVIS Integrated Assistant
Demonstrates seamless vision-control integration with proactive notifications
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jarvis_integrated_assistant import JARVISIntegratedAssistant
from vision.window_detector import WindowDetector


async def simulate_user_workflow():
    """Simulate a user workflow with JARVIS integrated assistant"""
    print("🚀 JARVIS Integrated Assistant Demo")
    print("=" * 60)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Initialize JARVIS
    print("\n🤖 Initializing JARVIS...")
    jarvis = JARVISIntegratedAssistant("Derek")
    
    print("✅ JARVIS is ready!")
    print("\n" + "="*60)
    
    # Scenario 1: User asks about screen
    print("\n📱 Scenario 1: User asks 'What's on my screen?'")
    print("-" * 40)
    
    response = await jarvis.process_vision_command("What's on my screen?")
    
    print("\n🗣️ JARVIS responds:")
    print(response.verbal_response)
    
    if response.follow_up_needed:
        print("\n⚡ JARVIS offers these actions:")
        for i, action in enumerate(response.available_actions[:5], 1):
            print(f"  {i}. {action}")
    
    await asyncio.sleep(2)
    
    # Scenario 2: Check for notifications on other windows
    print("\n\n📱 Scenario 2: User asks about other windows")
    print("-" * 40)
    
    response2 = await jarvis.process_vision_command(
        "What's happening on my other screens and windows?"
    )
    
    print("\n🗣️ JARVIS responds:")
    print(response2.verbal_response)
    
    # Scenario 3: If notifications detected, handle them
    if jarvis.context.active_notifications:
        print("\n\n📱 Scenario 3: Handling notifications")
        print("-" * 40)
        
        notif = jarvis.context.active_notifications[0]
        print(f"\n🔔 JARVIS detects: {notif.app_name} notification")
        
        # Read notification
        print("\n👤 User: 'Read the message'")
        read_response = await jarvis.read_notification(notif)
        
        print("\n🗣️ JARVIS:")
        print(read_response.verbal_response)
        
        # Show reply options
        print("\n💬 Quick reply options:")
        for i, action in enumerate(read_response.available_actions[:5], 1):
            print(f"  {i}. {action}")
        
        # Compose reply
        print("\n👤 User: 'Help me reply'")
        reply_response = await jarvis.compose_reply(notif)
        
        print("\n🗣️ JARVIS:")
        print(reply_response.verbal_response)
    
    # Scenario 4: Describe specific screen area
    print("\n\n📱 Scenario 4: Describing specific screen area")
    print("-" * 40)
    
    print("\n👤 User: 'Describe the top right corner of my screen'")
    area_response = await jarvis.describe_screen_area("top right corner")
    
    print("\n🗣️ JARVIS:")
    print(area_response.verbal_response)
    
    # Show learning
    print("\n\n📊 JARVIS Learning Stats:")
    print("-" * 40)
    print(f"• Interactions recorded: {len(jarvis.context.interaction_history)}")
    print(f"• Notification patterns learned: {len(jarvis.notification_responses)}")
    print(f"• Contextual preferences: {len(jarvis.contextual_preferences)}")
    
    print("\n✅ Demo complete!")
    print("\n🎯 Key Features Demonstrated:")
    print("• Proactive screen analysis with notification detection")
    print("• Natural verbal communication about what's visible")
    print("• Intelligent notification handling with context")
    print("• Dynamic reply suggestions based on history")
    print("• Seamless vision-control integration")
    print("• Zero hardcoding - everything is dynamic!")


async def demonstrate_notification_monitoring():
    """Demonstrate proactive notification monitoring"""
    print("\n\n🔔 Bonus: Proactive Notification Monitoring Demo")
    print("=" * 60)
    
    jarvis = JARVISIntegratedAssistant("Derek")
    
    print("\n🚨 Starting notification monitoring...")
    print("JARVIS will now monitor for new notifications and alert you.")
    print("Press Ctrl+C to stop.\n")
    
    # Start monitoring in background
    monitor_task = asyncio.create_task(jarvis.start_notification_monitoring())
    
    try:
        # Keep running
        await asyncio.sleep(30)  # Monitor for 30 seconds
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping notification monitoring...")
    finally:
        jarvis.notification_monitor_active = False
        monitor_task.cancel()
    
    print("✅ Monitoring stopped")


async def main():
    """Run all demonstrations"""
    # Run main workflow demo
    await simulate_user_workflow()
    
    # Ask if user wants to see monitoring demo
    print("\n\nWould you like to see the proactive notification monitoring demo?")
    print("This will monitor your screen for new notifications in real-time.")
    print("(Press Enter to skip)")
    
    try:
        # Simple non-blocking check
        await asyncio.wait_for(
            demonstrate_notification_monitoring(),
            timeout=1
        )
    except asyncio.TimeoutError:
        print("\nSkipping monitoring demo.")
    except Exception:
        pass
    
    print("\n\n🎉 All demos complete!")
    print("JARVIS Integrated Assistant is ready for use!")


if __name__ == "__main__":
    asyncio.run(main())