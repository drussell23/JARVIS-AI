#!/usr/bin/env python3
"""
Test script for Intervention Decision Engine
Demonstrates user state detection, situation assessment, and intervention decisions
"""

import asyncio
import time
from datetime import datetime, timedelta
import random
from pathlib import Path

async def test_intervention_decision():
    """Test the Intervention Decision Engine"""
    print("🤖 Testing Intervention Decision Engine")
    print("=" * 60)
    
    try:
        from intervention_decision_engine import (
            InterventionDecisionEngine, UserState, InterventionType,
            UserStateSignal, SituationContext, TimingStrategy,
            get_intervention_decision_engine
        )
        
        # Initialize engine
        engine = get_intervention_decision_engine()
        print("\n✅ Initialized Intervention Decision Engine")
        
        # Test 1: User State Detection
        print("\n1️⃣ Testing user state detection...")
        
        # Simulate focused user signals
        print("\n   Simulating FOCUSED user...")
        focused_signals = [
            UserStateSignal(
                signal_type="typing_pattern",
                value=0.8,  # Consistent typing
                confidence=0.9,
                timestamp=datetime.now(),
                source="keyboard_monitor",
                metadata={"wpm": 65, "accuracy": 0.95}
            ),
            UserStateSignal(
                signal_type="mouse_movement",
                value=0.2,  # Steady mouse
                confidence=0.8,
                timestamp=datetime.now(),
                source="mouse_monitor",
                metadata={"velocity": 0.3, "clicks": 2}
            ),
            UserStateSignal(
                signal_type="task_switches",
                value=0.1,  # Few switches
                confidence=0.9,
                timestamp=datetime.now(),
                source="window_monitor"
            ),
            UserStateSignal(
                signal_type="error_rate",
                value=0.1,  # Low errors
                confidence=0.85,
                timestamp=datetime.now(),
                source="error_detector"
            )
        ]
        
        for signal in focused_signals:
            await engine.process_user_signal(signal)
        
        # Force state update
        await engine._update_user_state()
        print(f"   Detected state: {engine.current_user_state.value} (confidence: {engine.state_confidence:.2f})")
        
        # Simulate frustrated user signals
        print("\n   Simulating FRUSTRATED user...")
        frustrated_signals = [
            UserStateSignal(
                signal_type="error_rate",
                value=0.7,  # High errors
                confidence=0.9,
                timestamp=datetime.now(),
                source="error_detector"
            ),
            UserStateSignal(
                signal_type="repeated_actions",
                value=0.8,  # Many repetitions
                confidence=0.85,
                timestamp=datetime.now(),
                source="action_monitor"
            ),
            UserStateSignal(
                signal_type="mouse_movement",
                value=0.9,  # Erratic mouse
                confidence=0.8,
                timestamp=datetime.now(),
                source="mouse_monitor",
                metadata={"velocity": 2.5, "acceleration": 1.8}
            ),
            UserStateSignal(
                signal_type="backspace_rate",
                value=0.6,  # High backspace
                confidence=0.9,
                timestamp=datetime.now(),
                source="keyboard_monitor"
            )
        ]
        
        for signal in frustrated_signals:
            await engine.process_user_signal(signal)
        
        await engine._update_user_state()
        print(f"   Detected state: {engine.current_user_state.value} (confidence: {engine.state_confidence:.2f})")
        
        # Test 2: Situation Assessment
        print("\n2️⃣ Testing situation assessment...")
        
        # Critical error situation
        critical_situation_data = {
            'has_error': True,
            'error_type': 'critical',
            'failure_count': 5,
            'deadline_proximity': 0.15,  # 15% time remaining
            'user_waiting': True,
            'known_issue': True,
            'similar_solutions_count': 3,
            'documentation_available': True,
            'complexity_score': 0.6,
            'context_type': 'debugging',
            'active_task': 'fix_critical_bug'
        }
        
        situation = await engine.assess_situation(critical_situation_data)
        print(f"\n   Critical situation assessment:")
        print(f"   - Problem severity: {situation.problem_severity:.2f}")
        print(f"   - Time criticality: {situation.time_criticality:.2f}")
        print(f"   - Solution availability: {situation.solution_availability:.2f}")
        print(f"   - Success probability: {situation.success_probability:.2f}")
        
        # Test 3: Intervention Decision
        print("\n3️⃣ Testing intervention decisions...")
        
        # Make decision for critical situation
        opportunity = await engine.decide_intervention()
        if opportunity:
            print(f"\n   ✅ Intervention recommended:")
            print(f"   - Type: {opportunity.intervention_type.value}")
            print(f"   - Timing: {opportunity.timing_strategy.value}")
            print(f"   - Confidence: {opportunity.confidence_score:.2f}")
            print(f"   - Urgency: {opportunity.urgency_score:.2f}")
            print(f"   - Rationale: {opportunity.rationale}")
            
            # Execute intervention
            result = await engine.execute_intervention(opportunity)
            print(f"\n   Execution result:")
            print(f"   - Response: {result.user_response}")
            print(f"   - Effectiveness: {result.effectiveness_score:.2f}")
        
        # Test 4: Different User States and Interventions
        print("\n4️⃣ Testing various scenarios...")
        
        scenarios = [
            # Struggling with learning
            {
                'user_state': UserState.STRUGGLING,
                'signals': [
                    UserStateSignal("help_searches", 0.7, 0.9, datetime.now(), "browser"),
                    UserStateSignal("documentation_views", 0.8, 0.8, datetime.now(), "browser"),
                    UserStateSignal("pause_duration", 0.7, 0.85, datetime.now(), "activity")
                ],
                'situation': {
                    'context_type': 'learning',
                    'problem_severity': 0.4,
                    'solution_availability': 0.8,
                    'documentation_available': True
                }
            },
            # Productive coding
            {
                'user_state': UserState.PRODUCTIVE,
                'signals': [
                    UserStateSignal("task_completion", 0.9, 0.9, datetime.now(), "task_tracker"),
                    UserStateSignal("typing_speed", 0.8, 0.85, datetime.now(), "keyboard"),
                    UserStateSignal("focus_duration", 0.85, 0.9, datetime.now(), "activity")
                ],
                'situation': {
                    'context_type': 'coding',
                    'problem_severity': 0.1,
                    'solution_availability': 0.2
                }
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            print(f"\n   Scenario {i+1}: {scenario['user_state'].value} user")
            
            # Process signals
            for signal in scenario['signals']:
                await engine.process_user_signal(signal)
            
            # Assess situation
            await engine.assess_situation(scenario['situation'])
            
            # Get intervention decision
            opportunity = await engine.decide_intervention()
            if opportunity:
                print(f"   - Recommended: {opportunity.intervention_type.value}")
                print(f"   - Timing: {opportunity.timing_strategy.value}")
            else:
                print("   - No intervention recommended")
        
        # Test 5: Timing Optimization
        print("\n5️⃣ Testing timing optimization...")
        
        # Simulate natural break
        print("\n   Simulating natural break...")
        break_signals = [
            UserStateSignal("idle_time", 0.8, 0.9, datetime.now(), "activity_monitor"),
            UserStateSignal("task_completion", 1.0, 0.95, datetime.now(), "task_tracker"),
            UserStateSignal("context_switch", 0.9, 0.8, datetime.now(), "window_monitor")
        ]
        
        for signal in break_signals:
            await engine.process_user_signal(signal)
        
        features = engine._extract_timing_features()
        print(f"   Natural break score: {features['natural_break_score']:.2f}")
        print(f"   Task boundary score: {features['task_boundary_score']:.2f}")
        print(f"   Cognitive load: {features['cognitive_load']:.2f}")
        print(f"   Request likelihood: {features['request_likelihood']:.2f}")
        
        # Test 6: Learning System
        print("\n6️⃣ Testing learning system...")
        
        # Simulate multiple interventions
        print("\n   Simulating intervention history...")
        for i in range(10):
            # Random user state
            state = random.choice(list(UserState))
            engine.current_user_state = state
            
            # Random situation
            situation = SituationContext(
                problem_severity=random.random(),
                time_criticality=random.random(),
                solution_availability=random.random(),
                success_probability=random.random(),
                context_type=random.choice(['coding', 'debugging', 'learning'])
            )
            engine.current_situation = situation
            
            # Make decision
            opportunity = await engine.decide_intervention()
            if opportunity:
                # Simulate execution with random response
                response = random.choice(['accepted', 'rejected', 'ignored'])
                opportunity.situation.success_probability = random.random()
                
                # Mock execution result
                from intervention_decision_engine import InterventionResult
                result = InterventionResult(
                    intervention_id=f"test_{i}",
                    opportunity_id=opportunity.opportunity_id,
                    executed_at=datetime.now(),
                    user_response=response,
                    effectiveness_score=0.8 if response == 'accepted' else 0.2
                )
                
                engine.intervention_history.append(result)
                engine.effectiveness_scores[opportunity.intervention_type].append(
                    result.effectiveness_score
                )
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"\n   Intervention Statistics:")
        print(f"   - Total interventions: {stats['total_interventions']}")
        print(f"   - Model version: {stats['model_version']}")
        
        if stats['effectiveness_by_type']:
            print("\n   Effectiveness by type:")
            for itype, data in stats['effectiveness_by_type'].items():
                print(f"   - {itype}: {data['mean']:.2f} (n={data['count']})")
        
        if stats.get('state_distribution'):
            print("\n   State distribution:")
            for state, count in stats['state_distribution'].items():
                print(f"   - {state}: {count}")
        
        # Test 7: Memory Usage
        print("\n7️⃣ Testing memory usage...")
        
        memory = engine.get_memory_usage()
        print(f"   Decision models: {memory['decision_models'] / 1024:.1f} KB")
        print(f"   Intervention history: {memory['intervention_history'] / 1024:.1f} KB")
        print(f"   Learning data: {memory['learning_data'] / 1024:.1f} KB")
        print(f"   Total: {memory['total'] / 1024:.1f} KB")
        
        # Test 8: Model Persistence
        print("\n8️⃣ Testing model persistence...")
        
        # Save models
        model_path = "/tmp/intervention_models.pkl"
        await engine.save_models(model_path)
        print(f"   ✅ Models saved to {model_path}")
        
        # Create new engine and load models
        new_engine = InterventionDecisionEngine()
        await new_engine.load_models(model_path)
        print(f"   ✅ Models loaded (version {new_engine.model_version})")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_rust_integration():
    """Test Rust intervention engine integration"""
    print("\n🦀 Testing Rust Intervention Engine Integration")
    print("=" * 50)
    
    try:
        # This would require the Rust library to be compiled
        print("   ⚠️  Rust integration requires compiled library")
        print("   To compile: cd jarvis-rust-core && cargo build --release")
        
        # If available, test the Rust components
        # from jarvis_rust_core import PyUserStateDetector, PyTimingOptimizer
        # detector = PyUserStateDetector(1000)
        # ... test state detection
        
    except Exception as e:
        print(f"   Error: {e}")


async def test_macos_timing():
    """Test macOS native timing detection"""
    print("\n🍎 Testing macOS Native Timing Detection")
    print("=" * 50)
    
    try:
        # This requires Swift compilation
        print("   ⚠️  macOS timing requires Swift compilation")
        print("   To compile: swiftc -o timing_detector intervention_timing_macos.swift")
        
        # If available, test the timing detector
        # Could use subprocess to call Swift binary or create Python bindings
        
    except Exception as e:
        print(f"   Error: {e}")


async def simulate_real_world_scenario():
    """Simulate a real-world intervention scenario"""
    print("\n🌍 Real-World Scenario: Helping a Struggling Developer")
    print("=" * 60)
    
    try:
        from intervention_decision_engine import (
            get_intervention_decision_engine, UserStateSignal
        )
        
        engine = get_intervention_decision_engine()
        
        # Scenario: Developer debugging a complex issue
        print("\n📖 Scenario: Developer has been debugging for 30 minutes...")
        
        # Initial signals - focused debugging
        signals = [
            UserStateSignal("error_rate", 0.3, 0.8, datetime.now(), "ide"),
            UserStateSignal("documentation_views", 0.4, 0.7, datetime.now(), "browser"),
            UserStateSignal("typing_pattern", 0.6, 0.8, datetime.now(), "keyboard")
        ]
        
        for signal in signals:
            await engine.process_user_signal(signal)
        
        print("   Initial state: Focused on debugging")
        
        # After 10 minutes - frustration building
        await asyncio.sleep(0.5)  # Simulate time passing
        print("\n   10 minutes later - errors increasing...")
        
        signals = [
            UserStateSignal("error_rate", 0.6, 0.9, datetime.now(), "ide"),
            UserStateSignal("repeated_actions", 0.7, 0.85, datetime.now(), "action_monitor"),
            UserStateSignal("backspace_rate", 0.5, 0.8, datetime.now(), "keyboard")
        ]
        
        for signal in signals:
            await engine.process_user_signal(signal)
        
        # Assess situation
        situation_data = {
            'has_error': True,
            'error_type': 'blocking',
            'failure_count': 4,
            'context_type': 'debugging',
            'known_issue': True,
            'similar_solutions_count': 2,
            'documentation_available': True
        }
        
        await engine.assess_situation(situation_data)
        
        # Check for intervention
        opportunity = await engine.decide_intervention()
        if opportunity:
            print(f"\n🤖 JARVIS: I should intervene!")
            print(f"   Type: {opportunity.intervention_type.value}")
            print(f"   Timing: {opportunity.timing_strategy.value}")
            print(f"   Message: {opportunity.content.get('suggestion', {}).get('message', 'Help available')}")
        
        # User takes a break (natural timing)
        print("\n   Developer pauses and leans back...")
        await engine.process_user_signal(
            UserStateSignal("idle_time", 0.9, 0.95, datetime.now(), "activity")
        )
        
        opportunity = await engine.decide_intervention()
        if opportunity and opportunity.timing_strategy.value == "natural_break":
            print("\n🤖 JARVIS: Perfect timing to offer help!")
            print(f"   '{opportunity.content.get('suggestion', {}).get('message', '')}'")
        
    except Exception as e:
        print(f"\n❌ Error in scenario: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_intervention_decision())
    # asyncio.run(test_rust_integration())
    # asyncio.run(test_macos_timing())
    asyncio.run(simulate_real_world_scenario())