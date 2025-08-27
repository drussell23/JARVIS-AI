#!/usr/bin/env python3
"""
Test script for JARVIS Vision System v2.0 Phase 3
Tests Transformer Routing, Continuous Learning, and <100ms latency
"""

import asyncio
import logging
import sys
import os
import time
import numpy as np
from datetime import datetime
from typing import Dict, List
from statistics import mean, median

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_transformer_routing_latency():
    """Test that transformer routing achieves <100ms latency"""
    print("\n=== Testing Transformer Router Latency (<100ms) ===\n")
    
    try:
        from vision.transformer_command_router import get_transformer_router
        router = get_transformer_router()
        print("✓ Transformer Router initialized")
        
        # Register test handlers
        async def fast_handler(cmd: str, ctx: Dict):
            # Simulate fast processing
            await asyncio.sleep(0.01)  # 10ms
            return f"Fast response for: {cmd}"
        
        async def slow_handler(cmd: str, ctx: Dict):
            # Simulate slower processing
            await asyncio.sleep(0.05)  # 50ms
            return f"Slow response for: {cmd}"
        
        # Discover handlers
        await router.discover_handler(fast_handler)
        await router.discover_handler(slow_handler)
        
        # Test commands
        test_commands = [
            "Can you see my screen?",
            "Describe what's on my display",
            "Analyze the current window",
            "What am I looking at?",
            "Check for errors on screen",
        ]
        
        latencies = []
        
        print("\nTesting routing latency...")
        print("-" * 50)
        
        for i, command in enumerate(test_commands * 4):  # Test each command 4 times
            start_time = time.perf_counter()
            
            result, route_info = await router.route_command(command, {
                'user': 'latency_test',
                'timestamp': datetime.now().isoformat()
            })
            
            latency = route_info['latency_ms']
            latencies.append(latency)
            
            # Print every 5th result
            if i % 5 == 0:
                print(f"Command {i+1}: {latency:.1f}ms - {'✓ PASS' if latency < 100 else '✗ FAIL'}")
        
        # Calculate statistics
        avg_latency = mean(latencies)
        median_latency = median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nLatency Statistics (n={len(latencies)}):")
        print(f"  Average: {avg_latency:.1f}ms")
        print(f"  Median: {median_latency:.1f}ms")
        print(f"  P95: {p95_latency:.1f}ms")
        print(f"  P99: {p99_latency:.1f}ms")
        print(f"  Under 100ms: {sum(1 for l in latencies if l < 100)}/{len(latencies)} ({100 * sum(1 for l in latencies if l < 100) / len(latencies):.1f}%)")
        
        # Get routing analytics
        analytics = router.get_routing_analytics()
        print(f"\nRouting Performance:")
        print(f"  Cache hit rate: {analytics['performance']['cache_hit_rate']:.1%}")
        print(f"  Handlers registered: {analytics['learning']['handlers_registered']}")
        
        return avg_latency < 100
        
    except Exception as e:
        print(f"✗ Error testing transformer routing: {e}")
        logger.error(f"Transformer routing test error: {e}", exc_info=True)
        return False


async def test_dynamic_handler_discovery():
    """Test dynamic handler discovery system"""
    print("\n=== Testing Dynamic Handler Discovery ===\n")
    
    try:
        from vision.transformer_command_router import get_transformer_router
        router = get_transformer_router()
        print("✓ Router initialized for handler discovery")
        
        # Create handlers with different patterns
        async def handle_screenshot_analysis(command: str, context: Dict):
            """Analyze screenshots and images from the screen"""
            return "Screenshot analysis result"
        
        async def handle_window_management(command: str, context: Dict):
            """Manage windows and application states
            Examples: minimize window, maximize screen, close app"""
            return "Window management result"
        
        async def handle_text_extraction(command: str, context: Dict):
            """Extract text from screen regions"""
            return "Text extraction result"
        
        # Discover handlers
        handlers = [
            handle_screenshot_analysis,
            handle_window_management,
            handle_text_extraction
        ]
        
        discovered_names = []
        for handler in handlers:
            name = await router.discover_handler(handler, auto_analyze=True)
            discovered_names.append(name)
            print(f"✓ Discovered handler: {name}")
        
        # Test routing to discovered handlers
        test_cases = [
            ("analyze this screenshot", "handle_screenshot_analysis"),
            ("minimize all windows", "handle_window_management"),
            ("extract text from screen", "handle_text_extraction"),
            ("close this application", "handle_window_management"),
        ]
        
        print("\nTesting routing to discovered handlers...")
        correct_routes = 0
        
        for command, expected_handler in test_cases:
            result, route_info = await router.route_command(command)
            actual_handler = route_info['handler']
            
            is_correct = actual_handler == expected_handler
            if is_correct:
                correct_routes += 1
            
            print(f"  '{command}' -> {actual_handler} {'✓' if is_correct else '✗ (expected ' + expected_handler + ')'}")
        
        accuracy = correct_routes / len(test_cases)
        print(f"\nDiscovery accuracy: {accuracy:.1%}")
        
        return accuracy > 0.5  # At least 50% correct for new handlers
        
    except Exception as e:
        print(f"✗ Error testing handler discovery: {e}")
        logger.error(f"Handler discovery test error: {e}", exc_info=True)
        return False


async def test_continuous_learning_pipeline():
    """Test continuous learning and model updates"""
    print("\n=== Testing Continuous Learning Pipeline ===\n")
    
    try:
        from vision.continuous_learning_pipeline import get_learning_pipeline
        pipeline = get_learning_pipeline()
        print("✓ Learning Pipeline initialized")
        
        # Simulate learning events
        print("\nSimulating command interactions...")
        
        # Successful commands
        for i in range(30):
            await pipeline.record_learning_event(
                event_type='command',
                data={
                    'command': f'test command {i}',
                    'handler': 'test_handler',
                    'success': True,
                    'latency_ms': np.random.normal(50, 10),
                    'confidence': np.random.uniform(0.7, 0.95),
                    'embedding': np.random.randn(768).tolist()
                },
                user_id='test_user'
            )
        
        # Failed commands
        for i in range(10):
            await pipeline.record_learning_event(
                event_type='command',
                data={
                    'command': f'failed command {i}',
                    'handler': 'test_handler',
                    'success': False,
                    'latency_ms': np.random.normal(80, 20),
                    'confidence': np.random.uniform(0.3, 0.6),
                    'error_type': 'handler_error'
                },
                user_id='test_user'
            )
        
        print("✓ Recorded 40 learning events")
        
        # Get learning status
        status = pipeline.get_learning_status()
        
        print(f"\nLearning Pipeline Status:")
        print(f"  Pipeline version: {status['pipeline_version']}")
        print(f"  Model version: {status['model_version']}")
        print(f"  Learning buffer: {status['learning_buffer_size']} events")
        print(f"  Feedback buffer: {status['feedback_buffer_size']} events")
        
        if status['current_performance']:
            print(f"  Success rate: {status['current_performance'].get('success_rate', 0):.1%}")
            print(f"  Avg latency: {status['current_performance'].get('avg_latency', 0):.1f}ms")
        
        # Test A/B testing capability
        print("\nTesting A/B test functionality...")
        
        # Simulate requests with model selection
        model_selections = {'production': 0, 'candidate': 0}
        
        for _ in range(100):
            model_version, _ = pipeline.select_model_for_request()
            model_selections[model_version] += 1
            
            # Record result
            pipeline.record_request_result(
                model_version=model_version,
                success=np.random.random() > 0.1,
                latency_ms=np.random.normal(60, 15)
            )
        
        print(f"  Model selections - Production: {model_selections['production']}, Candidate: {model_selections['candidate']}")
        print(f"  A/B test active: {status['ab_test_active']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing learning pipeline: {e}")
        logger.error(f"Learning pipeline test error: {e}", exc_info=True)
        return False


async def test_route_optimization():
    """Test route optimization based on performance"""
    print("\n=== Testing Route Optimization ===\n")
    
    try:
        from vision.transformer_command_router import get_transformer_router
        router = get_transformer_router()
        
        # Create handlers with different performance characteristics
        async def reliable_handler(cmd: str, ctx: Dict):
            await asyncio.sleep(0.02)
            return "Reliable result"
        
        async def unreliable_handler(cmd: str, ctx: Dict):
            # Fail 50% of the time
            if np.random.random() < 0.5:
                raise Exception("Simulated failure")
            return "Unreliable result"
        
        async def slow_handler(cmd: str, ctx: Dict):
            await asyncio.sleep(0.15)  # 150ms - too slow
            return "Slow result"
        
        # Register handlers
        await router.discover_handler(reliable_handler)
        await router.discover_handler(unreliable_handler)
        await router.discover_handler(slow_handler)
        
        print("✓ Registered handlers with different performance profiles")
        
        # Run multiple commands to build performance history
        print("\nBuilding performance history...")
        
        test_command = "optimize this route"
        results = {'reliable': 0, 'unreliable': 0, 'slow': 0}
        
        for i in range(20):
            try:
                result, route_info = await router.route_command(
                    test_command,
                    force_exploration=True if i < 5 else False  # Explore first 5
                )
                
                handler_name = route_info['handler']
                # Map handler name to expected format
                if 'reliable' in handler_name:
                    results['reliable'] += 1
                elif 'unreliable' in handler_name:
                    results['unreliable'] += 1
                elif 'slow' in handler_name:
                    results['slow'] += 1
                
            except Exception as e:
                logger.debug(f"Expected error during routing: {e}")
        
        print(f"\nRoute selections after optimization:")
        for handler, count in results.items():
            print(f"  {handler}: {count}/20 ({count/20:.1%})")
        
        # Check if optimization prefers reliable handler
        optimization_working = results['reliable'] > results['unreliable'] and \
                             results['reliable'] > results['slow']
        
        print(f"\n{'✓' if optimization_working else '✗'} Optimization {'is' if optimization_working else 'is not'} preferring reliable handler")
        
        # Get metrics
        analytics = router.get_routing_analytics()
        print("\nHandler Performance:")
        for handler_name, metrics in analytics['handlers'].items():
            if metrics['total_calls'] > 0:
                print(f"  {handler_name}:")
                print(f"    Success rate: {metrics['success_rate']:.1%}")
                print(f"    Avg latency: {metrics['avg_latency_ms']:.1f}ms")
                print(f"    Total calls: {metrics['total_calls']}")
        
        return optimization_working
        
    except Exception as e:
        print(f"✗ Error testing route optimization: {e}")
        logger.error(f"Route optimization test error: {e}", exc_info=True)
        return False


async def test_integrated_phase3_system():
    """Test the fully integrated Phase 3 system"""
    print("\n=== Testing Integrated Vision System v2.0 (Phase 3) ===\n")
    
    try:
        from vision.vision_system_v2 import get_vision_system_v2
        system = get_vision_system_v2()
        print("✓ Vision System v2.0 initialized with Phase 3 features")
        
        # Verify transformer routing is enabled
        if not system.use_transformer_routing:
            print("⚠️  Transformer routing is disabled, enabling it")
            system.use_transformer_routing = True
        
        # Test various commands with latency measurement
        test_scenarios = [
            {
                'command': 'Can you see my screen?',
                'context': {'user': 'phase3_test', 'require_fast': True}
            },
            {
                'command': 'Describe the current application window',
                'context': {'user': 'phase3_test', 'exploration': True}
            },
            {
                'command': 'What errors do you see on the display?',
                'context': {'user': 'phase3_test', 'confidence_threshold': 0.8}
            }
        ]
        
        print("\nTesting integrated system with transformer routing...")
        latencies = []
        
        for scenario in test_scenarios:
            start_time = time.perf_counter()
            
            response = await system.process_command(
                scenario['command'],
                scenario['context']
            )
            
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            latencies.append(elapsed_ms)
            
            print(f"\nCommand: '{scenario['command']}'")
            print(f"  Success: {response.success}")
            print(f"  Latency: {elapsed_ms:.1f}ms")
            print(f"  Intent: {response.intent_type}")
            print(f"  Confidence: {response.confidence:.2%}")
            
            if response.data and 'route_decision' in response.data:
                route = response.data['route_decision']
                if isinstance(route, dict) and 'latency_ms' in route:
                    print(f"  Router latency: {route['latency_ms']:.1f}ms")
                    print(f"  Cache hit: {route.get('cache_hit', False)}")
        
        # Get system statistics
        stats = await system.get_system_stats()
        
        print(f"\n{'='*50}")
        print("System Statistics (Phase 3):")
        print(f"  Version: {stats['version']}")
        print(f"  Phase: {stats['phase']}")
        print(f"  Total interactions: {stats['total_interactions']}")
        
        if 'transformer_routing' in stats:
            tr_stats = stats['transformer_routing']
            print(f"\nTransformer Routing:")
            print(f"  Enabled: {tr_stats['enabled']}")
            print(f"  Cache hit rate: {tr_stats['cache_hit_rate']:.1%}")
            print(f"  Avg latency: {tr_stats['avg_latency_ms']:.1f}ms")
        
        if 'continuous_learning' in stats:
            cl_stats = stats['continuous_learning']
            print(f"\nContinuous Learning:")
            print(f"  Pipeline version: {cl_stats['pipeline_version']}")
            print(f"  Model version: {cl_stats['model_version']}")
            print(f"  Learning buffer: {cl_stats['learning_buffer_size']}")
            print(f"  A/B test active: {cl_stats['ab_test_active']}")
        
        avg_system_latency = mean(latencies)
        print(f"\nAverage system latency: {avg_system_latency:.1f}ms")
        
        return avg_system_latency < 150  # System should be under 150ms total
        
    except Exception as e:
        print(f"✗ Error testing integrated system: {e}")
        logger.error(f"Integrated system test error: {e}", exc_info=True)
        return False


async def test_route_explanation_system():
    """Test route explanation for debugging"""
    print("\n=== Testing Route Explanation System ===\n")
    
    try:
        from vision.transformer_command_router import get_transformer_router
        router = get_transformer_router()
        
        # Create some routing history
        test_commands = [
            "Show me what's on screen",
            "Can you analyze this error message?",
            "Minimize all windows please"
        ]
        
        print("Creating routing history...")
        for cmd in test_commands:
            await router.route_command(cmd, {
                'user': 'debug_test',
                'session': 'test_session'
            })
        
        # Get route explanations
        explanations = router.export_route_explanations()
        
        print(f"\nRoute Explanations (showing last {len(explanations)}):")
        print("-" * 60)
        
        for exp in explanations[-3:]:  # Show last 3
            print(f"\nCommand: '{exp['command']}'")
            print(f"  Timestamp: {exp['timestamp']}")
            print(f"  Selected: {exp['selected_handler']} ({exp['confidence']:.1%} confidence)")
            alt_str = [f"{a['handler']} ({a['score']:.1%})" for a in exp['alternatives'][:2]]
            print(f"  Alternatives: {alt_str}")
            print(f"  Success: {exp['success']}")
            print(f"  Latency: {exp['execution_time_ms']:.1f}ms")
            print(f"  Explanation: {exp['explanation']}")
        
        return len(explanations) > 0
        
    except Exception as e:
        print(f"✗ Error testing route explanation: {e}")
        logger.error(f"Route explanation test error: {e}", exc_info=True)
        return False


async def main():
    """Run all Phase 3 tests"""
    print("\n" + "="*60)
    print("JARVIS Vision System v2.0 - Phase 3 Test Suite")
    print("Testing Transformer Routing & Continuous Learning")
    print("="*60)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   Some features may be limited")
    
    # Run tests
    results = []
    
    results.append(await test_transformer_routing_latency())
    results.append(await test_dynamic_handler_discovery())
    results.append(await test_continuous_learning_pipeline())
    results.append(await test_route_optimization())
    results.append(await test_route_explanation_system())
    results.append(await test_integrated_phase3_system())
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All Phase 3 tests passed!")
        print("  - Transformer routing achieves <100ms latency")
        print("  - Dynamic handler discovery is working")
        print("  - Continuous learning pipeline is operational")
        print("  - Route optimization based on performance")
        print("  - Debugging explanations available")
    else:
        print("\n✗ Some tests failed")
    
    # Cleanup
    print("\nShutting down components...")
    from vision.vision_system_v2 import get_vision_system_v2
    system = get_vision_system_v2()
    await system.shutdown()
    
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())