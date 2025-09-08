#!/usr/bin/env python3
"""
Test script for Workflow Pattern Engine
Tests pattern mining, formation, and application
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

async def test_workflow_patterns():
    """Test the Workflow Pattern Engine"""
    print("🔄 Testing Workflow Pattern Engine")
    print("=" * 60)
    
    try:
        # Import components
        from intelligence.workflow_pattern_engine import (
            WorkflowPatternEngine, WorkflowEvent, PatternType,
            get_workflow_pattern_engine
        )
        from intelligence.enhanced_workflow_engine import (
            EnhancedWorkflowEngine, get_enhanced_workflow_engine
        )
        
        # Initialize engines
        basic_engine = get_workflow_pattern_engine()
        enhanced_engine = get_enhanced_workflow_engine()
        
        print("\n✅ Initialized Workflow Pattern Engines")
        
        # Test 1: Event Recording
        print("\n1️⃣ Testing event recording...")
        
        # Simulate a development workflow
        dev_events = [
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=30),
                event_type='app_launch',
                source_system='system',
                event_data={'app': 'vscode'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=29),
                event_type='file_open',
                source_system='vsms',
                event_data={'file': 'main.py'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=28),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'edit_code'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=25),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'save_file'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=24),
                event_type='app_switch',
                source_system='system',
                event_data={'app': 'terminal'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=23),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'run_test'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=20),
                event_type='app_switch',
                source_system='system',
                event_data={'app': 'vscode'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=19),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'fix_error'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=15),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'save_file'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=14),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'git_commit'}
            ),
        ]
        
        # Record events
        for event in dev_events:
            await basic_engine.record_event(event)
        
        print(f"   Recorded {len(dev_events)} workflow events")
        
        # Test 2: Pattern Mining
        print("\n2️⃣ Testing pattern mining...")
        
        # Mine patterns
        patterns = await basic_engine.mine_patterns(min_support=0.2)
        print(f"   Discovered {len(patterns)} patterns")
        
        for i, pattern in enumerate(patterns[:3]):
            print(f"\n   Pattern {i+1}:")
            print(f"   - Type: {pattern.pattern_type}")
            print(f"   - Actions: {' → '.join(pattern.action_sequence[:5])}")
            print(f"   - Frequency: {pattern.frequency}")
            print(f"   - Confidence: {pattern.confidence:.2f}")
        
        # Test 3: Advanced Pattern Learning
        print("\n3️⃣ Testing advanced pattern learning...")
        
        # Add more varied events for clustering
        communication_events = [
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=10),
                event_type='app_launch',
                source_system='system',
                event_data={'app': 'slack'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=9),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'read_message'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=8),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'compose_reply'}
            ),
            WorkflowEvent(
                timestamp=datetime.now() - timedelta(minutes=5),
                event_type='action',
                source_system='activity_recognition',
                event_data={'action': 'send_message'}
            ),
        ]
        
        for event in communication_events:
            await enhanced_engine.record_event(event)
        
        # Learn advanced patterns
        advanced_patterns = await enhanced_engine.learn_patterns_advanced()
        print(f"   Learned {len(advanced_patterns)} advanced patterns")
        
        # Test 4: Pattern Prediction
        print("\n4️⃣ Testing pattern prediction...")
        
        # Current sequence
        current_seq = ['app_launch:vscode', 'file_open', 'edit_code']
        predictions = await basic_engine.predict_next_actions(current_seq, top_k=3)
        
        print(f"   Current sequence: {' → '.join(current_seq)}")
        print("   Predicted next actions:")
        for action, prob in predictions:
            print(f"   - {action}: {prob:.2f}")
        
        # Test 5: Workflow Automation Suggestions
        print("\n5️⃣ Testing automation suggestions...")
        
        current_context = {
            'active_app': 'vscode',
            'recent_actions': ['edit_code', 'save_file'],
            'time_of_day': 'afternoon',
            'goal': 'bug_fixing'
        }
        
        suggestions = enhanced_engine.suggest_automation(current_context)
        print(f"\n   Found {len(suggestions)} automation suggestions:")
        
        for i, suggestion in enumerate(suggestions[:3]):
            print(f"\n   Suggestion {i+1}:")
            print(f"   - Description: {suggestion['description']}")
            print(f"   - Benefit Score: {suggestion['benefit_score']:.2f}")
            print(f"   - Est. Time Saved: {suggestion['estimated_time_saved']:.0f} seconds")
        
        # Test 6: Pattern Statistics
        print("\n6️⃣ Testing pattern statistics...")
        
        stats = basic_engine.get_pattern_statistics()
        print("\n   Pattern Engine Statistics:")
        print(f"   - Total patterns: {stats['total_patterns']}")
        print(f"   - Unique actions: {stats['unique_actions']}")
        print(f"   - Average pattern length: {stats['average_pattern_length']:.1f}")
        print(f"   - Most frequent pattern type: {stats['most_frequent_type']}")
        
        # Memory usage
        memory_usage = basic_engine.get_memory_usage()
        print(f"\n   Memory Usage:")
        print(f"   - Pattern Database: {memory_usage['pattern_database'] / 1024 / 1024:.1f} MB")
        print(f"   - Sequence Buffer: {memory_usage['sequence_buffer'] / 1024 / 1024:.1f} MB")
        print(f"   - Mining Engine: {memory_usage['mining_engine'] / 1024 / 1024:.1f} MB")
        
        # Test 7: Simulate Daily Routine Pattern
        print("\n7️⃣ Testing daily routine pattern detection...")
        
        # Simulate morning routine
        morning_routine = [
            ('system_wake', 'system'),
            ('app_launch:mail', 'system'),
            ('check_email', 'activity'),
            ('app_launch:slack', 'system'),
            ('check_messages', 'activity'),
            ('app_launch:calendar', 'system'),
            ('review_schedule', 'activity'),
            ('app_launch:vscode', 'system'),
            ('start_work', 'activity'),
        ]
        
        # Record morning routine events
        base_time = datetime.now().replace(hour=8, minute=0)
        for i, (action, source) in enumerate(morning_routine):
            event = WorkflowEvent(
                timestamp=base_time + timedelta(minutes=i*5),
                event_type='action',
                source_system=source,
                event_data={'action': action}
            )
            await basic_engine.record_event(event)
        
        # Check if routine pattern detected
        await basic_engine.detect_routines()
        routine_patterns = [p for p in basic_engine.patterns.values() 
                           if p.pattern_type == PatternType.ROUTINE_PATTERN]
        
        if routine_patterns:
            print(f"   ✅ Detected {len(routine_patterns)} routine patterns")
            routine = routine_patterns[0]
            print(f"   Morning routine: {' → '.join(routine.action_sequence[:5])}...")
        else:
            print("   ⚠️  No routine patterns detected yet (need more data)")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_rust_integration():
    """Test Rust pattern mining integration"""
    print("\n🦀 Testing Rust Pattern Mining Integration")
    print("=" * 40)
    
    try:
        # This would require the Rust library to be compiled and available
        # For now, we'll simulate the integration
        print("   ⚠️  Rust integration requires compiled library")
        print("   To compile: cd jarvis-rust-core && cargo build --release")
        
    except Exception as e:
        print(f"   Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_workflow_patterns())
    # asyncio.run(test_rust_integration())