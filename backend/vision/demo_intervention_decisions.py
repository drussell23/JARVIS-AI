#!/usr/bin/env python3
"""
Simple demonstration of the Intervention Decision Engine
Shows how JARVIS can detect user states and offer timely help
"""

import asyncio
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    print("🤖 JARVIS Intervention Decision Demo")
    print("=" * 50)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    analyzer = ClaudeVisionAnalyzer()
    
    # Scenario 1: User making errors
    print("\n📍 Scenario 1: User experiencing repeated errors")
    print("-" * 40)
    
    # Send error signals
    await analyzer.process_intervention_signal("error_rate", 0.7, 0.9)
    await analyzer.process_intervention_signal("repeated_actions", 0.8, 0.85)
    
    # Check if intervention needed
    result = await analyzer.check_intervention_opportunity({
        'has_error': True,
        'error_type': 'syntax_error',
        'failure_count': 3,
        'context_type': 'coding'
    })
    
    if result['intervention_recommended']:
        print(f"✅ JARVIS recommends: {result['type']}")
        print(f"   When: {result['timing']}")
        print(f"   Why: {result['rationale']}")
    
    # Scenario 2: User being productive
    print("\n📍 Scenario 2: User in productive flow")
    print("-" * 40)
    
    # Send productive signals
    await analyzer.process_intervention_signal("typing_pattern", 0.9, 0.9, 
                                              {"wpm": 70, "accuracy": 0.98})
    await analyzer.process_intervention_signal("task_completion", 1.0, 0.95)
    
    # Check intervention
    result = await analyzer.check_intervention_opportunity({
        'has_error': False,
        'context_type': 'coding',
        'task_progress': 0.8
    })
    
    if not result['intervention_recommended']:
        print(f"✅ No intervention needed - user is {result['current_state']}")
    
    # Show statistics
    print("\n📊 Intervention System Statistics")
    print("-" * 40)
    stats = await analyzer.get_intervention_stats()
    for key, value in stats.items():
        if not key.startswith('_'):
            print(f"   {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())