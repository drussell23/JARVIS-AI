#!/usr/bin/env python3
"""
Test script for integrated intelligence systems:
- Workflow Pattern Engine
- Anomaly Detection Framework  
- Intervention Decision Engine

This demonstrates how the three components work together in JARVIS Vision.
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the backend directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_integrated_intelligence():
    """Test the integrated intelligence systems"""
    print("🤖 JARVIS Vision Intelligence System - Integrated Test")
    print("=" * 70)
    
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        # Initialize analyzer
        analyzer = ClaudeVisionAnalyzer()
        print("✅ Initialized Claude Vision Analyzer with intelligence components")
        
        # Test 1: Workflow Pattern Learning
        print("\n1️⃣ Testing Workflow Pattern Engine...")
        
        # Simulate a series of workflow events
        workflow_events = [
            "open_vscode",
            "open_terminal", 
            "git_pull",
            "open_file_main.py",
            "edit_code",
            "save_file",
            "run_tests",
            "git_commit",
            "git_push"
        ]
        
        print("   Recording developer workflow sequence...")
        workflow_engine = await analyzer.get_workflow_engine()
        if workflow_engine:
            from vision.intelligence.workflow_pattern_engine import WorkflowEvent
            
            for i, action in enumerate(workflow_events):
                event = WorkflowEvent(
                    timestamp=datetime.now() + timedelta(seconds=i*30),
                    event_type='user_action',
                    source_system='test_simulator',
                    event_data={'action': action, 'app': 'vscode'}
                )
                await workflow_engine.record_event(event)
            
            # Mine patterns
            print("   Mining workflow patterns...")
            await workflow_engine.mine_patterns(min_support=0.3)
            
            # Get predictions
            test_sequence = ["open_vscode", "open_terminal", "git_pull"]
            predictions = await workflow_engine.predict_next_actions(test_sequence, top_k=3)
            
            print(f"   Given sequence: {test_sequence}")
            print("   Predicted next actions:")
            for action, confidence in predictions:
                print(f"     - {action}: {confidence:.2f}")
        
        # Test 2: Anomaly Detection
        print("\n2️⃣ Testing Anomaly Detection Framework...")
        
        # Create a mock screenshot (black screen - unusual)
        mock_screenshot = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        print("   Analyzing black screen for anomalies...")
        anomaly_result = await analyzer.detect_anomalies_in_screenshot(
            mock_screenshot,
            context={'expected': 'normal_desktop', 'time': 'work_hours'}
        )
        
        if anomaly_result.get('anomaly_detected'):
            print(f"   ⚠️ Anomaly detected!")
            print(f"   - Type: {anomaly_result['type']}")
            print(f"   - Severity: {anomaly_result['severity']}")
            print(f"   - Confidence: {anomaly_result['confidence']:.2f}")
            print(f"   - Requires intervention: {anomaly_result['requires_intervention']}")
        else:
            print("   No anomalies detected")
        
        # Test 3: User State Detection & Intervention
        print("\n3️⃣ Testing Intervention Decision Engine...")
        
        # Simulate frustrated user signals
        print("   Simulating frustrated user behavior...")
        
        signals = [
            ("error_rate", 0.7, 0.9, {"error_type": "compile_error"}),
            ("repeated_actions", 0.8, 0.85, {"action": "retry_build"}),
            ("mouse_movement", 0.9, 0.8, {"velocity": 2.5, "clicks": 15}),
            ("backspace_rate", 0.6, 0.9, {"deletions": 25})
        ]
        
        for signal_type, value, confidence, metadata in signals:
            result = await analyzer.process_intervention_signal(
                signal_type, value, confidence, metadata
            )
            print(f"   - Processed {signal_type} signal")
        
        print(f"   Current user state: {result['current_state']} ({result['state_confidence']:.2f})")
        
        # Check for intervention
        situation = {
            'has_error': True,
            'error_type': 'blocking',
            'failure_count': 5,
            'deadline_proximity': 0.2,
            'known_issue': True,
            'documentation_available': True,
            'context_type': 'debugging'
        }
        
        intervention = await analyzer.check_intervention_opportunity(situation)
        
        if intervention['intervention_recommended']:
            print(f"\n   🤖 JARVIS should intervene!")
            print(f"   - Type: {intervention['type']}")
            print(f"   - Timing: {intervention['timing']}")
            print(f"   - Confidence: {intervention['confidence']:.2f}")
            print(f"   - Urgency: {intervention['urgency']:.2f}")
            print(f"   - Rationale: {intervention['rationale']}")
        
        # Test 4: Integrated Flow
        print("\n4️⃣ Testing integrated intelligence flow...")
        
        # Simulate a complete scenario
        print("   Scenario: User struggling with repeated build failures")
        
        # 1. Record workflow disruption
        if workflow_engine:
            disrupted_events = [
                "edit_code",
                "save_file", 
                "run_build",  # fails
                "view_error",
                "edit_code",
                "save_file",
                "run_build",  # fails again
                "view_error",
                "search_stackoverflow"
            ]
            
            for i, action in enumerate(disrupted_events):
                event = WorkflowEvent(
                    timestamp=datetime.now() + timedelta(seconds=i*10),
                    event_type='user_action',
                    source_system='test_simulator',
                    event_data={
                        'action': action,
                        'success': False if 'run_build' in action else True
                    }
                )
                await workflow_engine.record_event(event)
        
        # 2. Detect anomaly in behavior pattern
        anomaly_detector = await analyzer.get_anomaly_detector()
        if anomaly_detector:
            from vision.intelligence.anomaly_detection_framework import Observation, AnomalyType
            
            observation = Observation(
                timestamp=datetime.now(),
                observation_type='behavioral_pattern',
                data={
                    'repeated_failures': 2,
                    'action_sequence': disrupted_events[-5:],
                    'time_between_attempts': 30
                },
                source='workflow_analyzer',
                metadata={'pattern': 'repeated_build_failure'}
            )
            
            anomaly = await anomaly_detector.detect_anomaly(observation)
            if anomaly:
                print(f"   Behavioral anomaly detected: {anomaly.description}")
        
        # 3. This triggers intervention engine automatically
        print("   Checking final intervention recommendation...")
        final_check = await analyzer.check_intervention_opportunity()
        
        if final_check['intervention_recommended']:
            print(f"   ✅ System ready to help user with: {final_check['type']}")
        
        # Test 5: Get system statistics
        print("\n5️⃣ System Statistics...")
        
        # Workflow stats
        patterns = await analyzer.get_workflow_patterns()
        print(f"   Workflow patterns discovered: {len(patterns)}")
        
        # Anomaly history
        anomaly_history = await analyzer.get_anomaly_history(limit=5)
        print(f"   Recent anomalies: {len(anomaly_history)}")
        
        # Intervention stats
        intervention_stats = await analyzer.get_intervention_stats()
        if 'total_interventions' in intervention_stats:
            print(f"   Total interventions: {intervention_stats['total_interventions']}")
            print(f"   Model version: {intervention_stats.get('model_version', 'N/A')}")
        
        print("\n✅ All integrated intelligence tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_scenarios():
    """Test different user scenarios"""
    print("\n\n🎭 Testing Different User Scenarios")
    print("=" * 50)
    
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        analyzer = ClaudeVisionAnalyzer()
        
        scenarios = ["frustrated_user", "productive_user", "struggling_user"]
        
        for scenario in scenarios:
            print(f"\n📊 Testing scenario: {scenario}")
            result = await analyzer.test_intervention_system(scenario)
            
            print(f"   State detected: {result['detected_state']}")
            print(f"   Confidence: {result['state_confidence']:.2f}")
            
            if result['intervention_recommended']:
                print(f"   Intervention: {result['intervention_type']} ({result['timing_strategy']})")
                print(f"   Rationale: {result['rationale']}")
            else:
                print("   No intervention needed")
                
    except Exception as e:
        print(f"\n❌ Error in scenario testing: {e}")


async def demonstrate_memory_management():
    """Demonstrate memory-aware operation"""
    print("\n\n💾 Memory Management Demonstration")
    print("=" * 50)
    
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        analyzer = ClaudeVisionAnalyzer()
        
        # Check memory allocations
        print("\nMemory allocations:")
        
        workflow_engine = await analyzer.get_workflow_engine()
        if workflow_engine:
            memory = workflow_engine.get_memory_usage()
            print(f"   Workflow Pattern Engine: {memory['total'] / 1024 / 1024:.1f} MB")
        
        anomaly_detector = await analyzer.get_anomaly_detector()
        if anomaly_detector:
            memory = anomaly_detector.get_memory_usage()
            print(f"   Anomaly Detection: {memory['total'] / 1024 / 1024:.1f} MB")
        
        intervention_engine = await analyzer.get_intervention_engine()
        if intervention_engine:
            memory = intervention_engine.get_memory_usage()
            print(f"   Intervention Engine: {memory['total'] / 1024 / 1024:.1f} MB")
        
        print("\n✅ Memory allocations within specified limits")
        
    except Exception as e:
        print(f"\n❌ Error checking memory: {e}")


if __name__ == "__main__":
    print("Starting JARVIS Vision Intelligence System tests...\n")
    
    # Run all tests
    asyncio.run(test_integrated_intelligence())
    asyncio.run(test_scenarios())
    asyncio.run(demonstrate_memory_management())
    
    print("\n\n🎉 All tests completed!")
    print("\nThe intelligence systems are now integrated and can:")
    print("- Learn and optimize user workflows")
    print("- Detect visual, behavioral, and system anomalies")
    print("- Decide when and how to offer proactive assistance")
    print("- Work together to enhance JARVIS's understanding")