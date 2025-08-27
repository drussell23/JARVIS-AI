#!/usr/bin/env python3
"""
Test script for JARVIS Vision System v2.0 Phase 2
Tests Intelligent Response Generation and Neural Routing
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_neural_routing():
    """Test neural command routing replacing if/elif chains"""
    print("\n=== Testing Neural Command Router ===\n")
    
    try:
        from vision.neural_command_router import get_neural_router
        router = get_neural_router()
        print("✓ Neural Command Router initialized")
        
        # Define test handlers
        async def handle_screen_description(cmd: str, ctx: Dict):
            return f"Screen description for: {cmd}"
        
        async def handle_capability_check(cmd: str, ctx: Dict):
            return f"Yes, I can see your screen. Context: {ctx.get('user', 'Unknown')}"
        
        async def handle_error_analysis(cmd: str, ctx: Dict):
            return f"Analyzing errors in: {cmd}"
        
        # Register handlers
        router.register_handler(
            "screen_description",
            handle_screen_description,
            "Describes what's on the screen",
            ["describe my screen", "what do you see", "analyze display"]
        )
        
        router.register_handler(
            "capability_check",
            handle_capability_check,
            "Checks vision capabilities",
            ["can you see", "are you able to view", "do you see my screen"]
        )
        
        router.register_handler(
            "error_analysis",
            handle_error_analysis,
            "Analyzes errors on screen",
            ["check for errors", "find problems", "analyze issues"]
        )
        
        # Test routing
        test_commands = [
            ("Can you see my screen?", {'user': 'TestUser', 'confidence': 0.9}),
            ("Please describe what's on my display", {'user': 'TestUser'}),
            ("Check if there are any errors", {'urgency': 0.8}),
            ("What am I looking at?", {}),  # Ambiguous command
        ]
        
        print("\nTesting neural routing...")
        for command, context in test_commands:
            result, decision = await router.route_command(command, context)
            print(f"\nCommand: '{command}'")
            print(f"  Routed to: {decision.handler_name}")
            print(f"  Confidence: {decision.confidence:.2%}")
            print(f"  Result: {result}")
            
            if decision.alternative_routes:
                print(f"  Alternatives: {[f'{h}({c:.1%})' for h, c in decision.alternative_routes[:2]]}")
        
        # Show routing metrics
        metrics = router.get_routing_metrics()
        print(f"\nRouting Metrics:")
        print(f"  Total routes: {metrics['total_routes']}")
        print(f"  Success rate: {metrics['success_rate']:.1%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing neural router: {e}")
        logger.error(f"Neural router test error: {e}", exc_info=True)
        return False


async def test_dynamic_response_composer():
    """Test dynamic response composition"""
    print("\n=== Testing Dynamic Response Composer ===\n")
    
    try:
        from vision.dynamic_response_composer import get_response_composer, ResponseContext
        composer = get_response_composer()
        print("✓ Dynamic Response Composer initialized")
        
        # Test different styles
        test_cases = [
            # (content, context, expected_style)
            (
                "I can see a code editor with Python files open",
                ResponseContext(
                    intent_type="screen_analysis",
                    confidence=0.9,
                    user_name="John",
                    user_preferences={'style': 'concise'}
                ),
                "concise"
            ),
            (
                "There appears to be an error in your code",
                ResponseContext(
                    intent_type="error_detection",
                    confidence=0.7,
                    user_name="Sarah",
                    emotion_state="frustrated"
                ),
                "empathetic"
            ),
            (
                "Your screen shows a web browser with multiple tabs",
                ResponseContext(
                    intent_type="screen_description",
                    confidence=0.95,
                    user_preferences={'style': 'technical', 'format': 'markdown'}
                ),
                "technical"
            )
        ]
        
        print("\nTesting response composition with different styles...")
        for content, context, expected_style in test_cases:
            response = await composer.compose_response(content, context)
            
            print(f"\nOriginal: {content}")
            print(f"Context: {context.intent_type}, user={context.user_name}")
            print(f"Generated ({response.style}): {response.text}")
            print(f"Format: {response.format}")
            print(f"Confidence: {response.confidence:.2%}")
            
            if response.alternatives:
                print(f"Alternatives: {len(response.alternatives)} available")
        
        # Test learning from feedback
        composer.learn_from_feedback(
            response_text=response.text,
            was_effective=True,
            user_feedback="Great response!",
            user_name="John"
        )
        print("\n✓ Feedback recorded for learning")
        
        # Show metrics
        metrics = composer.get_metrics()
        print(f"\nComposer Metrics:")
        print(f"  Total responses: {metrics['total_responses_generated']}")
        print(f"  Training examples: {metrics['training_examples']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing response composer: {e}")
        logger.error(f"Response composer test error: {e}", exc_info=True)
        return False


async def test_personalization_engine():
    """Test user personalization"""
    print("\n=== Testing Personalization Engine ===\n")
    
    try:
        from vision.personalization_engine import get_personalization_engine
        personalization = get_personalization_engine()
        print("✓ Personalization Engine initialized")
        
        # Simulate different user interactions
        users = [
            {
                'id': 'concise_user',
                'messages': [
                    "screen?",
                    "errors?", 
                    "status"
                ]
            },
            {
                'id': 'verbose_user',
                'messages': [
                    "Could you please describe in detail what you can see on my screen right now?",
                    "I would appreciate if you could analyze the current state of my display",
                    "Would you be so kind as to check for any potential issues?"
                ]
            },
            {
                'id': 'technical_user',
                'messages': [
                    "Analyze viewport DOM elements",
                    "Check console.log output",
                    "Debug stack trace on screen"
                ]
            }
        ]
        
        print("\nAnalyzing user communication styles...")
        for user_data in users:
            user_id = user_data['id']
            print(f"\nUser: {user_id}")
            
            # Analyze each message
            for msg in user_data['messages']:
                analysis = await personalization.analyze_user_style(user_id, msg)
            
            # Get personalization parameters
            params = personalization.get_personalization_params(user_id)
            print(f"  Style: {params['style']}")
            print(f"  Tone: {params['tone']}")
            print(f"  Complexity: {params['complexity']:.2f}")
            print(f"  Target length: {params['target_length']:.0f} words")
            
            # Get insights
            insights = personalization.get_user_insights(user_id)
            print(f"  Interactions: {insights['behavior']['interaction_count']}")
        
        # Test preference learning
        personalization.update_satisfaction('verbose_user', 0.9, response_params={'style': 'verbose'})
        personalization.update_satisfaction('concise_user', 0.8, response_params={'style': 'concise'})
        
        print("\n✓ User preferences updated based on satisfaction")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing personalization: {e}")
        logger.error(f"Personalization test error: {e}", exc_info=True)
        return False


async def test_integrated_phase2_system():
    """Test integrated Vision System v2.0 with Phase 2 features"""
    print("\n=== Testing Integrated Vision System v2.0 (Phase 2) ===\n")
    
    try:
        from vision.vision_system_v2 import get_vision_system_v2
        system = get_vision_system_v2()
        print("✓ Vision System v2.0 initialized with Phase 2 features")
        
        # Test with different users and contexts
        test_scenarios = [
            {
                'user': 'tech_user',
                'user_name': 'Alice',
                'command': 'Can you analyze my IDE for syntax errors?',
                'context': {'confidence': 0.8, 'intent': 'error_detection'}
            },
            {
                'user': 'casual_user', 
                'user_name': 'Bob',
                'command': 'hey, what am i looking at?',
                'context': {'time_of_day': 'evening', 'emotion': 'relaxed'}
            },
            {
                'user': 'business_user',
                'user_name': 'Carol',
                'command': 'Please provide a detailed analysis of the current display',
                'context': {'urgency': 0.9, 'format_preference': 'formal'}
            }
        ]
        
        print("\nTesting personalized responses...")
        for scenario in test_scenarios:
            print(f"\n{'='*50}")
            print(f"User: {scenario['user_name']} ({scenario['user']})")
            print(f"Command: '{scenario['command']}'")
            
            # Process command
            response = await system.process_command(
                scenario['command'],
                scenario['context']
            )
            
            print(f"\nResponse:")
            print(f"  Message: {response.message[:100]}...")
            print(f"  Success: {response.success}")
            print(f"  Confidence: {response.confidence:.2%}")
            print(f"  Intent: {response.intent_type}")
            
            if response.data:
                if 'route_decision' in response.data:
                    route = response.data['route_decision']
                    print(f"  Neural Route: {route.get('handler_name')} ({route.get('confidence', 0):.1%})")
                if 'personalization' in response.data:
                    pers = response.data['personalization']
                    print(f"  Personalized: style={pers.get('style')}, tone={pers.get('tone')}")
        
        # Get system statistics
        stats = await system.get_system_stats()
        print(f"\n{'='*50}")
        print(f"System Statistics:")
        print(f"  Version: {stats['version']}")
        print(f"  Total interactions: {stats['total_interactions']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Claude available: {stats['claude_available']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing integrated system: {e}")
        logger.error(f"Integrated system test error: {e}", exc_info=True)
        return False


async def main():
    """Run all Phase 2 tests"""
    print("\n" + "="*60)
    print("JARVIS Vision System v2.0 - Phase 2 Test Suite")
    print("Testing Intelligent Response Generation & Neural Routing")
    print("="*60)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   Some features may be limited")
    
    # Run tests
    results = []
    
    results.append(await test_neural_routing())
    results.append(await test_dynamic_response_composer())
    results.append(await test_personalization_engine())
    results.append(await test_integrated_phase2_system())
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All Phase 2 tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())