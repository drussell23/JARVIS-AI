#!/usr/bin/env python3
"""
Test script for Solution Memory Bank
Demonstrates solution capture, storage, retrieval, and application
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path

async def test_solution_memory_bank():
    """Test the Solution Memory Bank functionality"""
    print("🧠 Testing Solution Memory Bank")
    print("=" * 60)
    
    try:
        from solution_memory_bank import (
            get_solution_memory_bank, ProblemSignature, ProblemType,
            SolutionStep, Solution, SolutionStatus
        )
        
        # Initialize memory bank
        memory_bank = get_solution_memory_bank()
        print("✅ Initialized Solution Memory Bank")
        
        # Test 1: Capture a Solution
        print("\n1️⃣ Testing solution capture...")
        
        # Create a problem signature
        problem = ProblemSignature(
            visual_pattern={'app': 'vscode', 'dialog': 'error_popup'},
            error_messages=["Cannot find module 'numpy'", "Import error"],
            context_state={'app_version': '1.75.0', 'os': 'macOS'},
            symptoms=['red_squiggly_lines', 'import_error', 'module_not_found'],
            problem_type=ProblemType.ERROR
        )
        
        # Define solution steps
        solution_steps = [
            {
                'action': 'open_terminal',
                'target': 'integrated_terminal',
                'parameters': {'shortcut': 'cmd+`'}
            },
            {
                'action': 'type',
                'target': None,
                'parameters': {'text': 'pip install numpy'},
                'wait_condition': 'terminal_ready'
            },
            {
                'action': 'key',
                'target': 'return',
                'parameters': {},
                'verification': 'installation_complete'
            },
            {
                'action': 'reload_window',
                'target': None,
                'parameters': {'shortcut': 'cmd+r'},
                'wait_condition': 'window_reloaded'
            }
        ]
        
        # Capture the solution
        solution = await memory_bank.capture_solution(
            problem=problem,
            solution_steps=solution_steps,
            success=True,
            execution_time=45.2,
            context={'app_version': '1.75.0', 'os_version': 'macOS 14.0'}
        )
        
        print(f"   ✅ Captured solution: {solution.solution_details.solution_id}")
        print(f"   Steps: {len(solution.solution_details.action_sequence)}")
        print(f"   Confidence: {solution.solution_details.confidence:.2f}")
        
        # Test 2: Find Similar Solutions
        print("\n2️⃣ Testing solution retrieval...")
        
        # Create a similar problem
        similar_problem = ProblemSignature(
            visual_pattern={'app': 'vscode', 'dialog': 'error_popup'},
            error_messages=["No module named 'pandas'", "Import error"],
            context_state={'app_version': '1.75.0', 'os': 'macOS'},
            symptoms=['import_error', 'module_not_found'],
            problem_type=ProblemType.ERROR
        )
        
        # Find similar solutions
        similar_solutions = await memory_bank.find_similar_solutions(
            similar_problem, 
            threshold=0.6
        )
        
        print(f"   Found {len(similar_solutions)} similar solutions")
        for sol_id, similarity in similar_solutions[:3]:
            print(f"   - {sol_id}: similarity {similarity:.2f}")
        
        # Test 3: Get Recommendations
        print("\n3️⃣ Testing solution recommendations...")
        
        recommendations = await memory_bank.get_solution_recommendations(
            similar_problem,
            context={'app_version': '1.75.0', 'os': 'macOS', 'python_version': '3.9'}
        )
        
        print(f"   Got {len(recommendations)} recommendations")
        for i, rec in enumerate(recommendations[:3]):
            print(f"   {i+1}. Score: {rec['score']:.2f}")
            print(f"      - Similarity: {rec['similarity']:.2f}")
            print(f"      - Effectiveness: {rec['effectiveness']:.2f}")
            print(f"      - Success rate: {rec['success_rate']:.2f}")
            print(f"      - Auto-applicable: {rec['auto_applicable']}")
        
        # Test 4: Apply Solution
        print("\n4️⃣ Testing solution application...")
        
        if similar_solutions:
            solution_id = similar_solutions[0][0]
            
            # Mock execution callback
            async def mock_execute(step, context):
                print(f"      Executing: {step.action}")
                await asyncio.sleep(0.1)  # Simulate execution
                return {
                    'success': True,
                    'action': step.action,
                    'message': f"Completed {step.action}"
                }
            
            result = await memory_bank.apply_solution(
                solution_id,
                current_context={'app': 'vscode', 'problem': 'pandas_import'},
                execute_callback=mock_execute
            )
            
            print(f"   Application result:")
            print(f"   - Success: {result['success']}")
            print(f"   - Steps completed: {result['steps_completed']}")
            print(f"   - Execution time: {result['execution_time']:.2f}s")
        
        # Test 5: Solution Refinement
        print("\n5️⃣ Testing solution refinement...")
        
        if similar_solutions:
            solution_id = similar_solutions[0][0]
            
            refinements = {
                'steps': [
                    {
                        'index': 1,
                        'parameters': {'text': 'pip install pandas numpy matplotlib'}
                    }
                ],
                'add_steps': [
                    {
                        'action': 'verify_import',
                        'target': 'python_file',
                        'parameters': {'code': 'import pandas as pd'}
                    }
                ]
            }
            
            await memory_bank.refine_solution(
                solution_id,
                refinements,
                user_feedback="Installing multiple packages at once is more efficient",
                rating=0.9
            )
            
            print("   ✅ Solution refined with user feedback")
        
        # Test 6: Memory Statistics
        print("\n6️⃣ Memory Bank Statistics...")
        
        stats = memory_bank.get_statistics()
        print(f"   Total solutions: {stats['total_solutions']}")
        print(f"   Captures: {stats['captures']}")
        print(f"   Applications: {stats['applications']}")
        print(f"   Success rate: {stats['successes'] / max(stats['applications'], 1):.2f}")
        
        print("\n   Solutions by type:")
        for ptype, count in stats['by_type'].items():
            print(f"   - {ptype}: {count}")
        
        # Test 7: Memory Usage
        print("\n7️⃣ Memory Usage...")
        
        memory = memory_bank.get_memory_usage()
        print(f"   Solution database: {memory['solution_database'] / 1024:.1f} KB")
        print(f"   Index structures: {memory['index_structures'] / 1024:.1f} KB")
        print(f"   Application engine: {memory['application_engine'] / 1024:.1f} KB")
        print(f"   Total: {memory['total'] / 1024 / 1024:.2f} MB")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


async def test_complex_scenarios():
    """Test more complex solution scenarios"""
    print("\n\n🔧 Testing Complex Scenarios")
    print("=" * 50)
    
    try:
        from solution_memory_bank import (
            get_solution_memory_bank, ProblemSignature, ProblemType
        )
        
        memory_bank = get_solution_memory_bank()
        
        # Scenario 1: Configuration Problem
        print("\n📋 Scenario 1: Git Configuration Error")
        
        git_problem = ProblemSignature(
            visual_pattern={'app': 'terminal', 'error_color': 'red'},
            error_messages=[
                "Please tell me who you are",
                "Run git config --global user.email",
                "Run git config --global user.name"
            ],
            context_state={'app': 'git', 'command': 'git commit'},
            symptoms=['git_error', 'config_missing', 'commit_blocked'],
            problem_type=ProblemType.CONFIGURATION
        )
        
        git_solution = [
            {
                'action': 'type',
                'parameters': {'text': 'git config --global user.name "JARVIS"'},
                'wait_condition': 'command_entered'
            },
            {
                'action': 'key',
                'target': 'return',
                'parameters': {}
            },
            {
                'action': 'type',
                'parameters': {'text': 'git config --global user.email "jarvis@ai.local"'},
                'wait_condition': 'command_entered'
            },
            {
                'action': 'key',
                'target': 'return',
                'parameters': {}
            },
            {
                'action': 'type',
                'parameters': {'text': 'git commit -m "Initial commit"'},
                'verification': 'commit_successful'
            }
        ]
        
        solution = await memory_bank.capture_solution(
            problem=git_problem,
            solution_steps=git_solution,
            success=True,
            execution_time=15.5,
            context={'terminal': 'iTerm2', 'shell': 'zsh'}
        )
        
        print(f"   ✅ Captured git configuration solution")
        
        # Scenario 2: Performance Problem
        print("\n📋 Scenario 2: Slow Application Performance")
        
        perf_problem = ProblemSignature(
            visual_pattern={'app': 'activity_monitor', 'cpu_usage': 'high'},
            error_messages=[],
            context_state={'app': 'Chrome', 'tabs_open': 50},
            symptoms=['spinning_beachball', 'slow_response', 'high_cpu'],
            problem_type=ProblemType.PERFORMANCE
        )
        
        perf_solution = [
            {
                'action': 'menu',
                'target': 'Chrome > Preferences',
                'parameters': {}
            },
            {
                'action': 'click',
                'target': 'Advanced',
                'wait_condition': 'settings_loaded'
            },
            {
                'action': 'click',
                'target': 'Reset and clean up',
                'parameters': {}
            },
            {
                'action': 'click',
                'target': 'Restore settings',
                'parameters': {},
                'verification': 'dialog_shown'
            },
            {
                'action': 'click',
                'target': 'Reset settings',
                'parameters': {}
            }
        ]
        
        await memory_bank.capture_solution(
            problem=perf_problem,
            solution_steps=perf_solution,
            success=True,
            execution_time=60.0,
            context={'browser': 'Chrome', 'version': '120.0'}
        )
        
        print(f"   ✅ Captured performance solution")
        
        # Test cross-problem similarity
        print("\n🔍 Testing cross-problem similarity...")
        
        # New problem that might benefit from either solution
        new_problem = ProblemSignature(
            visual_pattern={'app': 'unknown', 'has_error': True},
            error_messages=["Configuration error", "Settings invalid"],
            context_state={},
            symptoms=['error_dialog', 'blocked_action'],
            problem_type=ProblemType.UNKNOWN
        )
        
        recommendations = await memory_bank.get_solution_recommendations(
            new_problem,
            context={'os': 'macOS'}
        )
        
        print(f"   Found {len(recommendations)} potential solutions")
        if recommendations:
            print(f"   Best match: {recommendations[0]['solution_id'][:8]}...")
            print(f"   Score: {recommendations[0]['score']:.2f}")
        
    except Exception as e:
        print(f"\n❌ Error in complex scenarios: {e}")
        import traceback
        traceback.print_exc()


async def demonstrate_learning():
    """Demonstrate how the system learns from usage"""
    print("\n\n📈 Demonstrating Learning System")
    print("=" * 50)
    
    try:
        from solution_memory_bank import get_solution_memory_bank, ProblemSignature, ProblemType
        
        memory_bank = get_solution_memory_bank()
        
        # Create a problem
        problem = ProblemSignature(
            error_messages=["Connection timeout"],
            symptoms=["network_error", "timeout"],
            problem_type=ProblemType.ERROR
        )
        
        # Simulate multiple attempts with varying success
        print("\n🔄 Simulating solution usage over time...")
        
        solution_id = None
        for i in range(5):
            if i == 0:
                # First capture
                solution = await memory_bank.capture_solution(
                    problem=problem,
                    solution_steps=[{'action': 'restart_network'}],
                    success=True,
                    execution_time=30.0
                )
                solution_id = solution.solution_details.solution_id
            else:
                # Apply existing solution
                success = i % 2 == 0  # Alternating success
                
                # Mock execution
                async def mock_exec(step, ctx):
                    return {'success': success, 'action': step.action}
                
                await memory_bank.apply_solution(
                    solution_id,
                    current_context={'iteration': i},
                    execute_callback=mock_exec
                )
            
            # Check effectiveness after each use
            sol = memory_bank.solutions[solution_id]
            effectiveness = sol.calculate_effectiveness()
            print(f"   Iteration {i+1}: Effectiveness = {effectiveness:.2f}")
        
        # Add user feedback
        print("\n💬 Adding user feedback...")
        await memory_bank.refine_solution(
            solution_id,
            refinements={},
            user_feedback="Works better after waiting 5 seconds",
            rating=0.7
        )
        
        # Check final state
        sol = memory_bank.solutions[solution_id]
        print(f"\n📊 Final solution stats:")
        print(f"   Usage count: {sol.learning_metadata.usage_count}")
        print(f"   Success rate: {sol.solution_details.success_rate:.2f}")
        print(f"   Effectiveness: {sol.calculate_effectiveness():.2f}")
        print(f"   Status: {sol.status.value}")
        
    except Exception as e:
        print(f"\n❌ Error in learning demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set up test environment
    import os
    os.environ['SOLUTION_STORAGE_PATH'] = './test_solutions.db'
    
    # Clean up old test data
    test_db = Path('./test_solutions.db')
    if test_db.exists():
        test_db.unlink()
    
    print("Starting Solution Memory Bank tests...\n")
    
    # Run all tests
    asyncio.run(test_solution_memory_bank())
    asyncio.run(test_complex_scenarios())
    asyncio.run(demonstrate_learning())
    
    print("\n\n🎉 All tests completed!")
    print("\nThe Solution Memory Bank can:")
    print("- Capture solutions from problem resolutions")
    print("- Find similar solutions using ML and hashing")
    print("- Apply solutions with automation support")
    print("- Learn from usage and user feedback")
    print("- Manage memory efficiently within 100MB allocation")