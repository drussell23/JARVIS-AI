#!/usr/bin/env python3
"""
Test script for Goal Inference System integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_goal_inference():
    """Test the Goal Inference System integration"""
    print("🎯 Testing Goal Inference System Integration")
    print("=" * 60)
    
    try:
        # Import components
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        import numpy as np
        
        # Initialize analyzer
        os.environ['ACTIVITY_RECOGNITION_ENABLED'] = 'true'
        os.environ['TASK_INFERENCE'] = 'true'
        os.environ['GOAL_INFERENCE_ENABLED'] = 'true'
        
        analyzer = ClaudeVisionAnalyzer(
            api_key=os.getenv('ANTHROPIC_API_KEY', 'dummy'),
            enable_realtime=False
        )
        
        print("\n✅ Initialized Claude Vision Analyzer")
        
        # Test 1: Get inferred goals (should be empty initially)
        print("\n1️⃣ Testing get_inferred_goals()...")
        goals = await analyzer.get_inferred_goals()
        if goals.get('enabled'):
            print(f"   Goal Inference: Enabled")
            print(f"   Total active goals: {goals.get('total_active', 0)}")
            print(f"   By level: {goals.get('by_level', {})}")
            
            if goals.get('high_confidence_goals'):
                print("\n   High confidence goals:")
                for goal in goals['high_confidence_goals'][:3]:
                    print(f"   - {goal['description']} ({goal['confidence']:.2f})")
        else:
            print(f"   Goal Inference: {goals.get('message', 'Not available')}")
        
        # Test 2: Simulate a screenshot analysis with goal inference
        print("\n2️⃣ Testing screenshot analysis with Goal Inference...")
        # Create a dummy screenshot
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        screenshot.fill(128)  # Gray screen
        
        # Analyze with a goal-focused prompt
        result, metrics = await analyzer.analyze_screenshot(
            screenshot,
            "Analyze what the user is trying to accomplish and their current goals"
        )
        
        # Check if goal data is included
        if 'vsms_core' in result and 'goals' in result['vsms_core']:
            goals_data = result['vsms_core']['goals']
            print(f"\n   ✅ Goal Inference detected:")
            
            # High-level goals
            if goals_data.get('high_level'):
                print("\n   High-level goals:")
                for goal in goals_data['high_level']:
                    print(f"   - {goal['description']} (confidence: {goal['confidence']:.2f})")
            
            # Intermediate goals
            if goals_data.get('intermediate'):
                print("\n   Intermediate goals:")
                for goal in goals_data['intermediate']:
                    print(f"   - {goal['description']} (confidence: {goal['confidence']:.2f})")
            
            # Immediate goals
            if goals_data.get('immediate'):
                print("\n   Immediate goals:")
                for goal in goals_data['immediate']:
                    print(f"   - {goal['description']} (confidence: {goal['confidence']:.2f})")
        else:
            print("\n   ⚠️  No goal data in result (may need VSMS Core enabled)")
            print(f"   VSMS Core present: {'vsms_core' in result}")
        
        # Test 3: Test goal tracking
        print("\n3️⃣ Testing goal progress tracking...")
        # First, get current goals
        goals_summary = await analyzer.get_inferred_goals()
        if goals_summary.get('high_confidence_goals'):
            first_goal = goals_summary['high_confidence_goals'][0]
            goal_id = first_goal['goal_id']
            
            print(f"   Tracking progress for goal: {first_goal['description']}")
            
            # Update progress
            update_result = await analyzer.track_goal_progress(goal_id, 0.25)
            if update_result.get('success'):
                print(f"   ✅ Progress updated to: {update_result.get('current_progress', 0):.1%}")
            else:
                print(f"   ❌ Failed to update progress: {update_result.get('message', 'Unknown error')}")
            
            # Get insights
            insights = await analyzer.get_goal_insights(goal_id)
            if insights.get('found'):
                print(f"\n   Goal insights:")
                print(f"   - Type: {insights.get('type')}")
                print(f"   - Level: {insights.get('level')}")
                print(f"   - Progress: {insights.get('progress', 0):.1%}")
                print(f"   - Duration: {insights.get('duration', 0):.0f} seconds")
                
                if insights.get('child_goals'):
                    print(f"   - Child goals: {len(insights['child_goals'])}")
                if insights.get('parent_goal'):
                    print(f"   - Has parent goal: Yes")
        else:
            print("   No goals available for tracking test")
        
        # Test 4: Test different contexts
        print("\n4️⃣ Testing different work contexts...")
        
        # Development context
        print("\n   Development context:")
        dev_result, _ = await analyzer.analyze_screenshot(
            screenshot,
            "User is debugging code in VSCode with terminal open"
        )
        if 'vsms_core' in dev_result and 'goals' in dev_result['vsms_core']:
            goals = dev_result['vsms_core']['goals']
            if goals.get('high_level'):
                print(f"   → Inferred: {goals['high_level'][0]['description']}")
        
        # Communication context
        print("\n   Communication context:")
        comm_result, _ = await analyzer.analyze_screenshot(
            screenshot,
            "User has Slack and email open, composing a message"
        )
        if 'vsms_core' in comm_result and 'goals' in comm_result['vsms_core']:
            goals = comm_result['vsms_core']['goals']
            if goals.get('high_level'):
                print(f"   → Inferred: {goals['high_level'][0]['description']}")
        
        # Research context
        print("\n   Research context:")
        research_result, _ = await analyzer.analyze_screenshot(
            screenshot,
            "User has multiple browser tabs open with documentation and taking notes"
        )
        if 'vsms_core' in research_result and 'goals' in research_result['vsms_core']:
            goals = research_result['vsms_core']['goals']
            if goals.get('high_level'):
                print(f"   → Inferred: {goals['high_level'][0]['description']}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_goal_inference())