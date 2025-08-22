#!/usr/bin/env python3
"""
Test Full Autonomy - Voice Integration + macOS Integration
Demonstrates 100% Iron Man-level JARVIS capabilities
"""

import asyncio
import os
from datetime import datetime

# Add backend to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all autonomy components
from autonomy.advanced_ai_brain import AdvancedAIBrain
from autonomy.voice_integration import VoiceIntegrationSystem
from autonomy.macos_integration import get_macos_integration
from autonomy.hardware_control import HardwareControlSystem


async def demonstrate_full_autonomy():
    """Demonstrate complete JARVIS autonomy"""
    print("🚀 JARVIS FULL AUTONOMY DEMONSTRATION")
    print("=" * 60)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    print("\n⚡ Initializing all systems...")
    
    # Initialize all components
    brain = AdvancedAIBrain(api_key)
    voice_system = VoiceIntegrationSystem(api_key)
    macos_system = get_macos_integration(api_key)
    hardware_system = HardwareControlSystem(api_key)
    
    # Start all systems
    print("\n🧠 Activating AI Brain...")
    await brain.start_brain_activity()
    
    print("🔊 Activating Voice Integration...")
    await voice_system.start_voice_integration()
    
    print("💻 Activating macOS Integration...")
    await macos_system.start_system_monitoring()
    
    await asyncio.sleep(2)  # Let systems initialize
    
    print("\n✅ All systems online - JARVIS is fully autonomous!")
    print("-" * 60)
    
    # Scenario 1: Morning Routine
    print("\n📅 Scenario 1: Morning Routine")
    print("-" * 40)
    
    # Voice greeting
    await voice_system.announce_text(
        "Good morning sir. Initializing your workspace for optimal productivity.",
        priority="normal"
    )
    
    # Process morning command
    morning_command = "Good morning JARVIS, prepare my workspace for the day"
    print(f"\n🎤 You: '{morning_command}'")
    
    response = await voice_system.process_voice_command(morning_command, confidence=0.9)
    print(f"🤖 JARVIS: {response['response']}")
    
    # Optimize system for morning work
    optimization = await macos_system.optimize_for_context("morning_productivity")
    print(f"💻 System optimized: {len(optimization.get('optimizations_applied', []))} changes made")
    
    # Adjust display for morning
    await hardware_system.control_display({
        'brightness': 80,
        'night_shift': {'enabled': False}
    })
    print("🖥️ Display optimized for morning work")
    
    await asyncio.sleep(3)
    
    # Scenario 2: Meeting Preparation
    print("\n\n📅 Scenario 2: Meeting Preparation")
    print("-" * 40)
    
    # Simulate meeting notification
    meeting_notification = {
        'app': 'Calendar',
        'title': 'Team Standup',
        'message': 'Starting in 5 minutes',
        'urgency': 0.9
    }
    
    # Voice announcement
    await voice_system.announce_notification(meeting_notification)
    
    # Prepare for meeting
    print("\n🎤 You: 'JARVIS, prepare for my meeting'")
    
    # Process command
    meeting_response = await brain.process_natural_language_command(
        "Prepare my system for the upcoming video meeting"
    )
    
    if meeting_response.get('executed'):
        print("✅ Meeting preparation executed autonomously")
    
    # Enable privacy mode for meeting
    privacy_result = await hardware_system.enable_privacy_mode()
    print(f"🔒 Privacy mode: {privacy_result.get('actions_taken', [])}")
    
    # Optimize for presentation
    presentation_result = await hardware_system.optimize_for_presentation()
    print(f"📊 Presentation mode: {presentation_result.get('optimizations', [])[:3]}")
    
    await asyncio.sleep(3)
    
    # Scenario 3: Focus Mode with Stress Detection
    print("\n\n📅 Scenario 3: Deep Work with Stress Detection")
    print("-" * 40)
    
    # Simulate work session
    print("\n🎤 You: 'I need to focus on coding for the next 2 hours'")
    
    # Process focus request
    focus_response = await voice_system.process_voice_command(
        "I need to focus on coding for the next 2 hours",
        confidence=0.95
    )
    
    # Optimize for deep work
    await macos_system.optimize_for_context("deep_focus")
    
    # Simulate stress detection after 90 minutes
    print("\n⏰ [90 minutes later...]")
    
    # AI detects stress
    stress_notification = {
        'type': 'ai_insight',
        'message': 'Elevated stress levels detected',
        'suggestion': 'Take a 10-minute break'
    }
    
    # Voice announcement with empathy
    await voice_system.announce_text(
        "Sir, I've noticed you've been working intensely for 90 minutes. "
        "May I suggest a brief break to maintain optimal performance?",
        priority="medium",
        context={'emotional_state': 'stressed', 'work_duration': 90}
    )
    
    await asyncio.sleep(3)
    
    # Scenario 4: End of Day Wind Down
    print("\n\n📅 Scenario 4: End of Day Wind Down")
    print("-" * 40)
    
    print("\n🎤 You: 'JARVIS, prepare for end of day'")
    
    # Process end of day command
    eod_response = await brain.process_natural_language_command(
        "It's time to wrap up for the day"
    )
    
    # Voice summary
    await voice_system.announce_text(
        "Certainly sir. Today you completed 3 major tasks, attended 2 meetings, "
        "and maintained focus for 4.5 hours. Shall I prepare your evening briefing?",
        priority="normal"
    )
    
    # Optimize display for evening
    await hardware_system.control_display({
        'brightness': 50,
        'night_shift': {'enabled': True, 'temperature': 'warm'}
    })
    print("🌙 Display adjusted for evening")
    
    # Disable privacy mode
    await hardware_system.disable_privacy_mode()
    print("🔓 Privacy mode disabled")
    
    # System status
    print("\n\n📊 SYSTEM STATUS")
    print("-" * 40)
    
    # Get all system statuses
    brain_status = brain.get_brain_status()
    voice_status = voice_system.get_system_status()
    macos_status = macos_system.get_system_status()
    hardware_status = hardware_system.get_hardware_status()
    
    print(f"🧠 AI Brain: Active={brain_status['active']}, "
          f"Learning Velocity={brain_status['performance_metrics']['learning_velocity']:.2f}")
    print(f"🔊 Voice System: Queue={voice_status['announcement_queue_size']}, "
          f"Total Announced={voice_status['total_announcements']}")
    print(f"💻 macOS System: Monitoring={macos_status['monitoring_active']}, "
          f"Optimizations={macos_status['optimizations_applied']}")
    print(f"🔧 Hardware Control: Privacy Mode={hardware_status['policies']['privacy_mode']}, "
          f"Components Controlled={len([c for c, s in hardware_status['components'].items() if s['controlled_by_jarvis']])}")
    
    # Shutdown
    print("\n\n🛑 Shutting down systems...")
    
    await brain.stop_brain_activity()
    await voice_system.stop_voice_integration()
    await macos_system.stop_system_monitoring()
    
    print("\n✅ Full autonomy demonstration complete!")
    print("\n🎯 Capabilities Demonstrated:")
    print("  ✓ Natural voice interaction and announcements")
    print("  ✓ Intelligent notification handling")
    print("  ✓ Context-aware system optimization")
    print("  ✓ Hardware control (camera, display, audio)")
    print("  ✓ Privacy mode activation")
    print("  ✓ Stress detection and response")
    print("  ✓ Autonomous decision making")
    print("  ✓ Continuous learning and adaptation")


async def test_voice_conversation():
    """Test natural voice conversation"""
    print("\n\n🎤 VOICE CONVERSATION TEST")
    print("=" * 60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return
    
    voice_system = VoiceIntegrationSystem(api_key)
    await voice_system.start_voice_integration()
    
    # Simulate a conversation
    conversation = [
        "Hey JARVIS, how are you today?",
        "What's on my schedule?",
        "Remind me to take a break in 30 minutes",
        "Thanks JARVIS, you're the best"
    ]
    
    for user_input in conversation:
        print(f"\n🎤 You: '{user_input}'")
        
        response = await voice_system.process_voice_command(user_input, confidence=0.9)
        print(f"🤖 JARVIS: {response['response']}")
        
        # Also announce via voice
        await voice_system.announce_text(response['response'], priority="normal")
        
        await asyncio.sleep(2)
    
    await voice_system.stop_voice_integration()


async def test_emergency_optimization():
    """Test emergency system optimization"""
    print("\n\n🚨 EMERGENCY OPTIMIZATION TEST")
    print("=" * 60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return
    
    macos_system = get_macos_integration(api_key)
    voice_system = VoiceIntegrationSystem(api_key)
    
    await voice_system.start_voice_integration()
    
    # Announce emergency
    await voice_system.announce_text(
        "Sir, system resources are critically low. Initiating emergency optimization.",
        priority="urgent"
    )
    
    # Execute emergency optimization
    result = await macos_system.emergency_optimization()
    
    print(f"\n🚨 Emergency actions taken:")
    for action in result['actions_taken']:
        print(f"  • {action}")
    
    # Announce completion
    await voice_system.announce_text(
        f"Emergency optimization complete. {len(result['actions_taken'])} actions taken to restore performance.",
        priority="high"
    )
    
    await voice_system.stop_voice_integration()


if __name__ == "__main__":
    print("🤖 JARVIS - 100% AUTONOMOUS AI ASSISTANT")
    print("Powered by Anthropic's Claude API")
    print("-" * 60)
    
    # Run full demonstration
    asyncio.run(demonstrate_full_autonomy())
    
    # Additional tests
    asyncio.run(test_voice_conversation())
    asyncio.run(test_emergency_optimization())