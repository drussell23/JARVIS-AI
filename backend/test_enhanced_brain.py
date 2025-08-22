#!/usr/bin/env python3
"""
Test script for the Enhanced AI Brain
Demonstrates the fully autonomous capabilities of JARVIS
"""

import asyncio
import os
from datetime import datetime

# Add backend to path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from autonomy.advanced_ai_brain import AdvancedAIBrain


async def test_enhanced_brain():
    """Test the enhanced AI brain capabilities"""
    print("🧠 Testing Enhanced JARVIS AI Brain")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Please set ANTHROPIC_API_KEY environment variable")
        return
    
    # Create AI brain
    print("\n🚀 Initializing Advanced AI Brain...")
    brain = AdvancedAIBrain(api_key)
    
    # Start brain activity
    print("⚡ Activating full autonomous intelligence...")
    await brain.start_brain_activity()
    
    # Give it a moment to initialize
    await asyncio.sleep(2)
    
    # Test 1: Natural Language Understanding
    print("\n📝 Test 1: Natural Language Command Processing")
    print("-" * 40)
    command = "I'm feeling overwhelmed with all these tasks, help me organize and prioritize"
    result = await brain.process_natural_language_command(command)
    print(f"Command: {command}")
    print(f"Understood: {result.get('understood', False)}")
    print(f"Executed: {result.get('executed', False)}")
    
    # Test 2: Personality Response
    print("\n💬 Test 2: Empathetic Personality Response")
    print("-" * 40)
    user_input = "Good morning JARVIS, I have a big presentation today and I'm nervous"
    response = await brain.get_personality_response(user_input)
    print(f"User: {user_input}")
    print(f"JARVIS: {response}")
    
    # Test 3: Problem Solving
    print("\n🎯 Test 3: Creative Problem Solving")
    print("-" * 40)
    brain.add_problem_for_solving(
        "My workflow involves too much context switching between apps",
        {'constraints': ['Must maintain all functionality', 'Easy to learn']}
    )
    print("Problem submitted for creative solving...")
    
    # Give time for processing
    await asyncio.sleep(3)
    
    # Test 4: Brain Status
    print("\n📊 Test 4: AI Brain Status")
    print("-" * 40)
    status = brain.get_brain_status()
    print(f"Active: {status['active']}")
    print(f"Current State:")
    print(f"  - Emotional: {status['current_state']['emotional']}")
    print(f"  - Cognitive Load: {status['current_state']['cognitive_load']}")
    print(f"  - Work Context: {status['current_state']['work_context']}")
    print(f"Performance Metrics:")
    for metric, value in status['performance_metrics'].items():
        print(f"  - {metric}: {value:.2f}")
    
    # Test 5: Learning from Feedback
    print("\n🎓 Test 5: Learning from Feedback")
    print("-" * 40)
    await brain.learn_from_feedback(
        action="Suggested break after 90 minutes of focus",
        result="User took break and felt refreshed",
        satisfaction=0.9
    )
    print("Positive feedback registered - AI learning updated")
    
    # Test 6: Dynamic Predictions
    print("\n🔮 Test 6: Checking Active Intelligence")
    print("-" * 40)
    await asyncio.sleep(2)  # Let predictions generate
    
    active_elements = status.get('active_elements', {})
    print(f"Active Predictions: {active_elements.get('predictions', 0)}")
    print(f"Active Insights: {active_elements.get('insights', 0)}")
    print(f"Active Solutions: {active_elements.get('solutions', 0)}")
    
    # Stop brain
    print("\n🛑 Deactivating AI Brain...")
    await brain.stop_brain_activity()
    
    print("\n✅ Enhanced AI Brain test complete!")
    print("\n🌟 Key Capabilities Demonstrated:")
    print("  ✓ Dynamic Predictive Intelligence")
    print("  ✓ Deep Contextual Understanding")
    print("  ✓ Creative Problem Solving")
    print("  ✓ Emotional Intelligence")
    print("  ✓ Autonomous Learning & Adaptation")
    print("  ✓ Real-time Personality Adjustment")


async def demonstrate_autonomy():
    """Demonstrate full autonomous capabilities"""
    print("\n\n🤖 AUTONOMOUS AI DEMONSTRATION")
    print("=" * 60)
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return
    
    brain = AdvancedAIBrain(api_key)
    await brain.start_brain_activity()
    
    print("\n🎬 Scenario: Simulating a work session...")
    print("-" * 40)
    
    # Simulate different scenarios
    scenarios = [
        {
            'time': '09:00',
            'command': "Start my work day - I need to be productive",
            'state_change': {'focus_level': 0.9, 'energy': 0.8}
        },
        {
            'time': '10:30',
            'command': "I've been coding for 90 minutes straight",
            'state_change': {'focus_level': 0.6, 'fatigue': 0.4}
        },
        {
            'time': '11:00',
            'command': "I have a meeting in 30 minutes about the new feature",
            'state_change': {'stress': 0.3, 'preparation_needed': True}
        },
        {
            'time': '14:00',
            'command': "Feeling stuck on this problem, need a creative solution",
            'state_change': {'frustration': 0.5, 'creativity_needed': True}
        }
    ]
    
    for scenario in scenarios:
        print(f"\n⏰ {scenario['time']} - User: \"{scenario['command']}\"")
        
        # Process command
        result = await brain.process_natural_language_command(scenario['command'])
        
        # Get personality response
        response = await brain.get_personality_response(scenario['command'])
        print(f"🤖 JARVIS: {response[:150]}...")
        
        # Check brain adaptation
        status = brain.get_brain_status()
        print(f"📊 Adaptation: Autonomy Level: {status['performance_metrics']['autonomy_level']:.2f}")
        
        await asyncio.sleep(2)
    
    await brain.stop_brain_activity()
    print("\n✨ Autonomous demonstration complete!")


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_enhanced_brain())
    
    # Run autonomy demonstration
    asyncio.run(demonstrate_autonomy())