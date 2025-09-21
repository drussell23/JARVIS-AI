#!/usr/bin/env python3
"""
Demonstration of the Solution Memory Bank
Shows how JARVIS learns from past problem resolutions
"""

import asyncio
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    print("🧠 JARVIS Solution Memory Bank Demo")
    print("=" * 50)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    analyzer = ClaudeVisionAnalyzer()
    
    # Scenario 1: Capture a solution
    print("\n📝 Scenario 1: Capturing a Python import error solution")
    print("-" * 40)
    
    # Define the problem and solution
    problem_description = "ModuleNotFoundError: No module named 'requests'"
    
    solution_steps = [
        {
            'action': 'open_terminal',
            'target': 'integrated_terminal',
            'parameters': {'shortcut': 'cmd+`'}
        },
        {
            'action': 'type',
            'parameters': {'text': 'pip install requests'}
        },
        {
            'action': 'key',
            'target': 'return',
            'parameters': {}
        }
    ]
    
    # Capture the solution
    result = await analyzer.capture_problem_solution(
        problem_description,
        solution_steps,
        success=True,
        execution_time=15.5
    )
    
    print(f"✅ Solution captured: {result['solution_id'][:8]}...")
    print(f"   Status: {result['status']}")
    
    # Scenario 2: Find solution for similar problem
    print("\n🔍 Scenario 2: Finding solution for similar error")
    print("-" * 40)
    
    # Search for solution by error
    similar_error = "ImportError: No module named pandas"
    results = await analyzer.search_solutions_by_error(similar_error)
    
    print(f"Found {len(results)} similar solutions:")
    for i, sol in enumerate(results):
        print(f"  {i+1}. Solution {sol['solution_id'][:8]}...")
        print(f"     Similarity: {sol['similarity']:.2f}")
        print(f"     Success rate: {sol['success_rate']:.1%}")
    
    # Scenario 3: Apply solution with feedback
    print("\n🚀 Scenario 3: Applying solution and providing feedback")
    print("-" * 40)
    
    if results:
        # Mock execution callback
        async def execute_step(step, context):
            print(f"   Executing: {step.action}")
            return {'success': True, 'message': 'Step completed'}
        
        # Apply the first solution
        solution_id = results[0]['solution_id']
        apply_result = await analyzer.apply_recommended_solution(
            solution_id,
            execute_callback=execute_step
        )
        
        print(f"✅ Solution applied: {'Success' if apply_result['success'] else 'Failed'}")
        print(f"   Time: {apply_result['execution_time']:.1f}s")
        
        # Provide feedback
        feedback_result = await analyzer.refine_solution_with_feedback(
            solution_id,
            feedback="Works great! Also needed to restart the kernel",
            rating=0.9,
            refinements={
                'add_steps': [{
                    'action': 'restart_kernel',
                    'target': 'jupyter',
                    'parameters': {}
                }]
            }
        )
        
        print(f"✅ Feedback provided: {feedback_result['message']}")
    
    # Show statistics
    print("\n📊 Solution Memory Bank Statistics")
    print("-" * 40)
    
    stats = await analyzer.get_solution_stats()
    if 'statistics' in stats:
        print(f"Total solutions: {stats['statistics']['total_solutions']}")
        print(f"Captures: {stats['statistics']['captures']}")
        print(f"Applications: {stats['statistics']['applications']}")
        
        print("\nMemory usage:")
        print(f"  Database: {stats['memory_usage']['database_mb']:.1f} MB")
        print(f"  Indices: {stats['memory_usage']['indices_mb']:.1f} MB")
        print(f"  Total: {stats['memory_usage']['total_mb']:.1f} MB")
    
    # Demonstrate screenshot analysis
    print("\n🖼️ Scenario 4: Analyzing screenshot for problems")
    print("-" * 40)
    
    # Create a mock error screenshot (black screen)
    mock_screenshot = np.zeros((600, 800, 3), dtype=np.uint8)
    
    result = await analyzer.find_solutions_for_screenshot(mock_screenshot)
    
    if result['found_solutions']:
        print("✅ Found solutions for detected problems:")
        for rec in result['recommendations']:
            print(f"  - Solution {rec['solution_id'][:8]}... (score: {rec['score']:.2f})")
    else:
        print(f"ℹ️ {result['message']}")
    
    print("\n✨ Solution Memory Bank Features:")
    print("- Captures solutions from problem resolutions")
    print("- Finds similar solutions using ML similarity")
    print("- Applies solutions with automation support")
    print("- Learns from user feedback and ratings")
    print("- Integrates with screenshot analysis")

if __name__ == "__main__":
    asyncio.run(main())