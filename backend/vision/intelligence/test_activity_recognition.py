#!/usr/bin/env python3
"""
Test script for Activity Recognition Engine integration
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_activity_recognition():
    """Test the Activity Recognition Engine integration"""
    print("🧪 Testing Activity Recognition Engine Integration")
    print("=" * 60)
    
    try:
        # Import components
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        import numpy as np
        
        # Initialize analyzer with Activity Recognition enabled
        os.environ['ACTIVITY_RECOGNITION_ENABLED'] = 'true'
        os.environ['TASK_INFERENCE'] = 'true'
        os.environ['PROGRESS_MONITORING'] = 'true'
        
        analyzer = ClaudeVisionAnalyzer(
            api_key=os.getenv('ANTHROPIC_API_KEY', 'dummy'),
            enable_realtime=False
        )
        
        print("\n✅ Initialized Claude Vision Analyzer")
        
        # Test 1: Get current activities (should be empty initially)
        print("\n1️⃣ Testing get_current_activities()...")
        activities = await analyzer.get_current_activities()
        print(f"   Current activities: {len(activities)} found")
        if activities:
            for activity in activities:
                print(f"   - {activity['task_name']} ({activity['status']})")
        
        # Test 2: Get activity summary
        print("\n2️⃣ Testing get_activity_summary()...")
        summary = await analyzer.get_activity_summary()
        if summary.get('enabled'):
            print(f"   Activity Recognition: Enabled")
            print(f"   Total tasks: {summary.get('total_tasks', 0)}")
            print(f"   Active tasks: {summary.get('active_tasks', 0)}")
            print(f"   Completed tasks: {summary.get('completed_tasks', 0)}")
            print(f"   Productivity score: {summary.get('productivity_score', 0.0):.2f}")
        else:
            print(f"   Activity Recognition: {summary.get('message', 'Not available')}")
        
        # Test 3: Simulate a screenshot analysis with activity recognition
        print("\n3️⃣ Testing screenshot analysis with Activity Recognition...")
        # Create a dummy screenshot
        screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        screenshot.fill(128)  # Gray screen
        
        # Analyze with a development-focused prompt
        result, metrics = await analyzer.analyze_screenshot(
            screenshot,
            "Analyze this development environment and identify what task the user is working on"
        )
        
        # Check if activity data is included
        if 'vsms_core' in result and 'activity' in result['vsms_core']:
            activity = result['vsms_core']['activity']
            print(f"\n   ✅ Activity Recognition detected:")
            print(f"   Task: {activity.get('task_name', 'Unknown')}")
            print(f"   Primary Activity: {activity.get('primary_activity', 'Unknown')}")
            print(f"   Status: {activity.get('status', 'Unknown')}")
            print(f"   Completion: {activity.get('completion_percentage', 0)}%")
            print(f"   Is Stuck: {activity.get('is_stuck', False)}")
        else:
            print("\n   ⚠️  No activity data in result (may need VSMS Core enabled)")
            print(f"   VSMS Core present: {'vsms_core' in result}")
        
        # Test 4: Test activity insights for a specific task
        print("\n4️⃣ Testing get_activity_insights()...")
        test_task_id = "test_task_001"
        insights = await analyzer.get_activity_insights(test_task_id)
        if insights.get('enabled'):
            if insights.get('found'):
                print(f"   Found insights for task {test_task_id}")
            else:
                print(f"   No insights found for task {test_task_id} (expected for test)")
        else:
            print(f"   Activity insights: {insights.get('message', 'Not available')}")
        
        # Test 5: Check configuration
        print("\n5️⃣ Checking Activity Recognition configuration...")
        print(f"   Enabled: {analyzer._activity_recognition_config['enabled']}")
        print(f"   Task Inference: {analyzer._activity_recognition_config['task_inference']}")
        print(f"   Progress Monitoring: {analyzer._activity_recognition_config['progress_monitoring']}")
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_activity_recognition())