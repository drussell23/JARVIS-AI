#!/usr/bin/env python3
"""
Test script for Predictive Pre-computation Engine
Demonstrates Markov chain prediction and speculative execution
"""

import asyncio
import time
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import random

# Import predictive engine components
from backend.vision.intelligence.predictive_precomputation_engine import (
    PredictivePrecomputationEngine,
    StateVector,
    StateType,
    get_predictive_engine
)
from backend.vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer


class WorkflowSimulator:
    """Simulate user workflows for testing"""
    
    def __init__(self):
        # Define common workflows
        self.workflows = {
            'email_workflow': [
                ('chrome', 'homepage', 'navigate', 'morning', 'check_email'),
                ('chrome', 'gmail', 'click_inbox', 'morning', 'check_email'),
                ('chrome', 'gmail_compose', 'click_compose', 'morning', 'write_email'),
                ('chrome', 'gmail_compose', 'type_recipient', 'morning', 'write_email'),
                ('chrome', 'gmail_compose', 'type_subject', 'morning', 'write_email'),
                ('chrome', 'gmail_compose', 'type_body', 'morning', 'write_email'),
                ('chrome', 'gmail_compose', 'click_send', 'morning', 'write_email'),
                ('chrome', 'gmail', 'email_sent', 'morning', 'check_email')
            ],
            'coding_workflow': [
                ('vscode', 'welcome', 'open_file', 'afternoon', 'coding'),
                ('vscode', 'editor', 'navigate_code', 'afternoon', 'coding'),
                ('vscode', 'editor', 'type_code', 'afternoon', 'coding'),
                ('vscode', 'editor', 'save_file', 'afternoon', 'coding'),
                ('vscode', 'terminal', 'run_test', 'afternoon', 'testing'),
                ('vscode', 'terminal', 'view_output', 'afternoon', 'testing'),
                ('vscode', 'editor', 'fix_error', 'afternoon', 'debugging'),
                ('vscode', 'editor', 'save_file', 'afternoon', 'coding')
            ],
            'research_workflow': [
                ('chrome', 'google', 'search', 'evening', 'research'),
                ('chrome', 'search_results', 'click_result', 'evening', 'research'),
                ('chrome', 'article', 'read', 'evening', 'research'),
                ('chrome', 'article', 'copy_text', 'evening', 'research'),
                ('notion', 'notes', 'paste_text', 'evening', 'note_taking'),
                ('notion', 'notes', 'format_text', 'evening', 'note_taking'),
                ('notion', 'notes', 'save', 'evening', 'note_taking')
            ]
        }
        
        self.current_workflow = None
        self.workflow_position = 0
    
    def start_workflow(self, workflow_name: str):
        """Start a specific workflow"""
        if workflow_name in self.workflows:
            self.current_workflow = workflow_name
            self.workflow_position = 0
            return True
        return False
    
    def get_next_state(self) -> Tuple[StateVector, bool]:
        """Get next state in workflow (state, is_complete)"""
        if not self.current_workflow:
            return None, True
        
        workflow = self.workflows[self.current_workflow]
        if self.workflow_position >= len(workflow):
            return None, True
        
        state_data = workflow[self.workflow_position]
        state = StateVector(
            app_id=state_data[0],
            app_state=state_data[1],
            user_action=state_data[2],
            time_context=state_data[3],
            goal_context=state_data[4],
            workflow_phase=f"{self.current_workflow}_step_{self.workflow_position}"
        )
        
        self.workflow_position += 1
        is_complete = self.workflow_position >= len(workflow)
        
        return state, is_complete
    
    def add_noise(self, state: StateVector, noise_level: float = 0.1) -> StateVector:
        """Add noise to simulate real-world variations"""
        if random.random() < noise_level:
            # Randomly modify one field
            modifications = [
                lambda s: StateVector(
                    app_id=s.app_id,
                    app_state=s.app_state + "_variant",
                    user_action=s.user_action,
                    time_context=s.time_context,
                    goal_context=s.goal_context,
                    workflow_phase=s.workflow_phase
                ),
                lambda s: StateVector(
                    app_id=s.app_id,
                    app_state=s.app_state,
                    user_action="unexpected_action",
                    time_context=s.time_context,
                    goal_context=s.goal_context,
                    workflow_phase=s.workflow_phase
                )
            ]
            
            modifier = random.choice(modifications)
            return modifier(state)
        
        return state


async def test_state_transitions():
    """Test basic state transition tracking"""
    print("\n=== Testing State Transition Tracking ===")
    
    engine = await get_predictive_engine()
    simulator = WorkflowSimulator()
    
    # Train with email workflow multiple times
    print("Training with email workflow...")
    for i in range(5):
        simulator.start_workflow('email_workflow')
        
        while True:
            state, is_complete = simulator.get_next_state()
            if state is None:
                break
            
            # Add some noise to make it realistic
            if i > 2:  # Add noise after initial training
                state = simulator.add_noise(state, 0.1)
            
            await engine.update_state(state)
            await asyncio.sleep(0.01)  # Small delay to simulate time
        
        print(f"  Completed training iteration {i+1}")
    
    # Test predictions
    print("\nTesting predictions...")
    simulator.start_workflow('email_workflow')
    
    predictions_correct = 0
    total_predictions = 0
    
    while True:
        current_state, _ = simulator.get_next_state()
        if current_state is None:
            break
        
        # Get predictions before updating state
        predictions = engine.transition_matrix.get_predictions(current_state, top_k=3)
        
        # Get actual next state
        next_state, is_complete = simulator.get_next_state()
        
        if next_state and predictions:
            total_predictions += 1
            
            # Check if correct state is in top predictions
            for predicted_state, prob, conf in predictions:
                if (predicted_state.app_state == next_state.app_state and
                    predicted_state.user_action == next_state.user_action):
                    predictions_correct += 1
                    print(f"  ✓ Correctly predicted: {next_state.app_state} -> {next_state.user_action} "
                          f"(prob: {prob:.3f}, conf: {conf:.3f})")
                    break
            else:
                print(f"  ✗ Failed to predict: {next_state.app_state} -> {next_state.user_action}")
        
        # Update engine with actual state
        await engine.update_state(current_state)
        if next_state:
            await engine.update_state(next_state)
        
        if is_complete:
            break
    
    accuracy = predictions_correct / max(1, total_predictions)
    print(f"\nPrediction accuracy: {accuracy:.2%} ({predictions_correct}/{total_predictions})")
    
    return accuracy


async def test_speculative_execution():
    """Test speculative execution of predictions"""
    print("\n=== Testing Speculative Execution ===")
    
    engine = await get_predictive_engine()
    simulator = WorkflowSimulator()
    
    # Train with coding workflow
    print("Training with coding workflow...")
    for i in range(3):
        simulator.start_workflow('coding_workflow')
        
        while True:
            state, is_complete = simulator.get_next_state()
            if state is None:
                break
            
            await engine.update_state(state)
            await asyncio.sleep(0.01)
        
        if is_complete:
            break
    
    # Monitor prediction queue
    print("\nMonitoring prediction queue...")
    
    # Start a new workflow instance
    simulator.start_workflow('coding_workflow')
    queue_stats = []
    execution_times = []
    
    for step in range(5):
        state, _ = simulator.get_next_state()
        if state is None:
            break
        
        # Update state (triggers predictions)
        start_time = time.time()
        await engine.update_state(state)
        
        # Wait for predictions to be queued
        await asyncio.sleep(0.1)
        
        # Check queue status
        queue_len, active_len = engine.prediction_queue.queue.lock(), len(engine.prediction_queue.active_tasks)
        queue_stats.append({
            'step': step,
            'queued': queue_len,
            'active': active_len,
            'total_predictions': engine.stats['predictions_made']
        })
        
        # Execute some predictions
        executed = 0
        while executed < 3:
            task = engine.prediction_queue.get_next_task()
            if task:
                exec_start = time.time()
                await engine.prediction_queue.execute_task(task)
                execution_times.append(time.time() - exec_start)
                executed += 1
            else:
                break
        
        print(f"  Step {step}: {executed} predictions executed, "
              f"avg time: {np.mean(execution_times[-executed:]) if executed > 0 else 0:.3f}s")
    
    # Display statistics
    print(f"\nTotal predictions made: {engine.stats['predictions_made']}")
    print(f"Total predictions executed: {engine.stats['predictions_executed']}")
    print(f"Average execution time: {np.mean(execution_times) if execution_times else 0:.3f}s")
    
    return queue_stats


async def test_cache_performance():
    """Test predictive cache hit rates"""
    print("\n=== Testing Predictive Cache Performance ===")
    
    engine = await get_predictive_engine()
    simulator = WorkflowSimulator()
    
    # Train with research workflow
    print("Training with research workflow...")
    for i in range(4):
        simulator.start_workflow('research_workflow')
        
        while True:
            state, is_complete = simulator.get_next_state()
            if state is None:
                break
            
            await engine.update_state(state)
        
        if is_complete:
            break
    
    # Test cache hits
    print("\nTesting cache performance...")
    
    cache_tests = []
    simulator.start_workflow('research_workflow')
    
    while True:
        current_state, _ = simulator.get_next_state()
        if current_state is None:
            break
        
        # Get predictions
        predictions = engine.transition_matrix.get_predictions(current_state, top_k=3)
        
        cache_hits = 0
        cache_misses = 0
        
        for next_state, prob, conf in predictions:
            # Check cache
            result = await engine.get_prediction(current_state, next_state)
            
            if result:
                cache_hits += 1
            else:
                cache_misses += 1
        
        cache_tests.append({
            'state': current_state.app_state,
            'hits': cache_hits,
            'misses': cache_misses,
            'hit_rate': cache_hits / max(1, cache_hits + cache_misses)
        })
        
        await engine.update_state(current_state)
    
    # Calculate overall cache performance
    total_hits = sum(test['hits'] for test in cache_tests)
    total_misses = sum(test['misses'] for test in cache_tests)
    overall_hit_rate = total_hits / max(1, total_hits + total_misses)
    
    print(f"\nCache Performance:")
    print(f"  Total hits: {total_hits}")
    print(f"  Total misses: {total_misses}")
    print(f"  Overall hit rate: {overall_hit_rate:.2%}")
    
    return cache_tests


async def test_learning_adaptation():
    """Test learning system adaptation"""
    print("\n=== Testing Learning System Adaptation ===")
    
    engine = await get_predictive_engine()
    simulator = WorkflowSimulator()
    
    # Phase 1: Train with consistent patterns
    print("Phase 1: Training with consistent patterns...")
    
    accuracy_history = []
    
    for epoch in range(10):
        simulator.start_workflow('email_workflow')
        epoch_correct = 0
        epoch_total = 0
        
        while True:
            state, is_complete = simulator.get_next_state()
            if state is None:
                break
            
            # Make prediction
            predictions = engine.transition_matrix.get_predictions(state, top_k=1)
            
            # Get actual next state
            next_state, _ = simulator.get_next_state()
            
            if next_state and predictions:
                predicted = predictions[0][0]
                actual = next_state
                
                # Record prediction outcome
                engine.learning_system.record_prediction(
                    predicted, actual, predictions[0][2]
                )
                
                if predicted.app_state == actual.app_state:
                    epoch_correct += 1
                epoch_total += 1
            
            await engine.update_state(state)
            if next_state:
                await engine.update_state(next_state)
            
            if is_complete:
                break
        
        epoch_accuracy = epoch_correct / max(1, epoch_total)
        accuracy_history.append(epoch_accuracy)
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Accuracy = {epoch_accuracy:.2%}")
    
    # Phase 2: Introduce drift
    print("\nPhase 2: Introducing concept drift...")
    
    # Modify workflow to simulate drift
    modified_workflow = list(simulator.workflows['email_workflow'])
    modified_workflow[3] = ('chrome', 'gmail_compose', 'voice_input', 'morning', 'write_email')
    modified_workflow[4] = ('chrome', 'gmail_compose', 'ai_complete', 'morning', 'write_email')
    simulator.workflows['email_workflow_drift'] = modified_workflow
    
    for epoch in range(10, 15):
        simulator.start_workflow('email_workflow_drift')
        epoch_correct = 0
        epoch_total = 0
        
        while True:
            state, is_complete = simulator.get_next_state()
            if state is None:
                break
            
            predictions = engine.transition_matrix.get_predictions(state, top_k=1)
            next_state, _ = simulator.get_next_state()
            
            if next_state and predictions:
                predicted = predictions[0][0]
                actual = next_state
                
                engine.learning_system.record_prediction(
                    predicted, actual, predictions[0][2]
                )
                
                if predicted.app_state == actual.app_state:
                    epoch_correct += 1
                epoch_total += 1
            
            await engine.update_state(state)
            if next_state:
                await engine.update_state(next_state)
            
            if is_complete:
                break
        
        epoch_accuracy = epoch_correct / max(1, epoch_total)
        accuracy_history.append(epoch_accuracy)
        print(f"  Epoch {epoch}: Accuracy = {epoch_accuracy:.2%}")
    
    # Get learning report
    report = engine.learning_system.get_accuracy_report()
    
    print(f"\nLearning System Report:")
    print(f"  Overall accuracy: {report['overall_accuracy']:.2%}")
    print(f"  Confidence threshold: {report['confidence_threshold']:.3f}")
    print(f"  Recent trend: {report['recent_trend']}")
    
    return accuracy_history


async def test_vision_analyzer_integration():
    """Test integration with ClaudeVisionAnalyzer"""
    print("\n=== Testing Vision Analyzer Integration ===")
    
    # Initialize analyzer with predictive engine enabled
    analyzer = ClaudeVisionAnalyzer(
        api_key="test_key",  # Would use real key in production
        enable_realtime=False
    )
    
    # Enable predictive engine
    analyzer._predictive_engine_config['enabled'] = True
    analyzer._predictive_engine_config['enable_speculative'] = True
    
    # Simulate screenshot analysis workflow
    from PIL import Image
    dummy_image = Image.new('RGB', (800, 600), color='white')
    
    # Test queries that form a pattern
    test_queries = [
        ("Find the submit button", "chrome", "form_page"),
        ("Click on the blue submit button", "chrome", "form_page"),
        ("Locate the confirmation message", "chrome", "confirmation_page"),
        ("Find the submit button", "chrome", "form_page"),  # Repeat pattern
        ("Click on the blue submit button", "chrome", "form_page"),  # Should be predicted
    ]
    
    print("Simulating vision analysis workflow...")
    
    for i, (query, app_id, app_state) in enumerate(test_queries):
        print(f"\nQuery {i+1}: '{query}'")
        
        # Update context
        analyzer._context['app_id'] = app_id
        
        # In real usage, would call analyze_screenshot
        # For testing, simulate the state update
        if analyzer._predictive_engine_config['enabled']:
            try:
                engine = await analyzer.get_predictive_engine()
                if engine:
                    state = StateVector(
                        app_id=app_id,
                        app_state=app_state,
                        user_action='analyze',
                        time_context=analyzer._get_time_context(),
                        goal_context=query[:50],
                        metadata={'query': query}
                    )
                    
                    await engine.update_state(state)
                    
                    # Check predictions
                    predictions = engine.transition_matrix.get_predictions(state, top_k=3)
                    if predictions:
                        print(f"  Predictions:")
                        for next_state, prob, conf in predictions:
                            print(f"    - {next_state.goal_context} (prob: {prob:.3f}, conf: {conf:.3f})")
            
            except Exception as e:
                print(f"  Predictive engine error: {e}")
    
    # Get engine statistics
    if analyzer.predictive_engine:
        stats = analyzer.predictive_engine.get_statistics()
        print(f"\n=== Predictive Engine Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")


def visualize_results(accuracy_history: List[float], cache_tests: List[Dict],
                     queue_stats: List[Dict]):
    """Visualize test results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Predictive Pre-computation Engine Analysis', fontsize=16)
    
    # Accuracy over time
    ax = axes[0, 0]
    epochs = list(range(len(accuracy_history)))
    ax.plot(epochs, accuracy_history, 'b-', linewidth=2, marker='o')
    ax.axvline(x=10, color='r', linestyle='--', label='Drift introduced')
    ax.set_title('Learning Accuracy Over Time')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Prediction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cache hit rates
    ax = axes[0, 1]
    if cache_tests:
        states = [test['state'] for test in cache_tests]
        hit_rates = [test['hit_rate'] for test in cache_tests]
        bars = ax.bar(range(len(states)), hit_rates, color='green', alpha=0.7)
        ax.set_title('Cache Hit Rates by State')
        ax.set_xlabel('State')
        ax.set_ylabel('Hit Rate')
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, rate in zip(bars, hit_rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1%}', ha='center', va='bottom')
    
    # Queue statistics
    ax = axes[1, 0]
    if queue_stats:
        steps = [stat['step'] for stat in queue_stats]
        queued = [stat['queued'] for stat in queue_stats]
        active = [stat['active'] for stat in queue_stats]
        
        ax.plot(steps, queued, 'b-', label='Queued', marker='s')
        ax.plot(steps, active, 'r-', label='Active', marker='o')
        ax.set_title('Prediction Queue Activity')
        ax.set_xlabel('Workflow Step')
        ax.set_ylabel('Task Count')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # State transition heatmap
    ax = axes[1, 1]
    # Create sample transition matrix for visualization
    states = ['homepage', 'gmail', 'compose', 'send', 'confirm']
    transition_data = np.array([
        [0.1, 0.8, 0.0, 0.0, 0.1],
        [0.2, 0.2, 0.5, 0.0, 0.1],
        [0.0, 0.1, 0.3, 0.6, 0.0],
        [0.0, 0.1, 0.0, 0.1, 0.8],
        [0.7, 0.2, 0.0, 0.0, 0.1]
    ])
    
    im = ax.imshow(transition_data, cmap='YlOrRd')
    ax.set_xticks(range(len(states)))
    ax.set_yticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha='right')
    ax.set_yticklabels(states)
    ax.set_title('State Transition Probabilities')
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    
    # Add text annotations
    for i in range(len(states)):
        for j in range(len(states)):
            text = ax.text(j, i, f'{transition_data[i, j]:.2f}',
                          ha="center", va="center", color="black")
    
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('predictive_engine_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to predictive_engine_analysis.png")


async def main():
    """Main test runner"""
    print("=" * 60)
    print("JARVIS Vision - Predictive Pre-computation Engine Test Suite")
    print("=" * 60)
    
    # Run all tests
    accuracy = await test_state_transitions()
    queue_stats = await test_speculative_execution()
    cache_tests = await test_cache_performance()
    accuracy_history = await test_learning_adaptation()
    await test_vision_analyzer_integration()
    
    # Visualize results
    print("\n=== Generating Visualizations ===")
    visualize_results(accuracy_history, cache_tests, queue_stats)
    
    # Shutdown engine
    engine = await get_predictive_engine()
    await engine.shutdown()
    
    print("\n=== Test Suite Complete ===")


if __name__ == "__main__":
    asyncio.run(main())