#!/usr/bin/env python3
"""
Test script for JARVIS Vision System v2.0 Phase 1
Validates ML-based intent classification and semantic understanding
"""

import asyncio
import logging
import sys
import os
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init()

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_phase_1_p0_features():
    """Test Phase 1 P0 features: ML-based classification and semantic understanding"""
    print(f"\n{Fore.CYAN}=== Testing JARVIS Vision System v2.0 - Phase 1 P0 Features ==={Style.RESET_ALL}\n")
    
    try:
        # Import Vision System v2.0
        from vision.vision_system_v2 import get_vision_system_v2
        vision_system = get_vision_system_v2()
        print(f"{Fore.GREEN}✓ Vision System v2.0 initialized{Style.RESET_ALL}")
        
        # Test cases for dynamic intent recognition
        test_commands = [
            # Vision capability confirmation (no hardcoding)
            "Can you see my screen?",
            "Are you able to see what I'm looking at?",
            "Do you see my desktop?",
            
            # Screen description variations (learned patterns)
            "Describe what's on my screen",
            "Tell me what you see",
            "What am I looking at right now?",
            
            # Multi-language test (if supported)
            "¿Puedes ver mi pantalla?",  # Spanish
            "Pouvez-vous voir mon écran?",  # French
            
            # Ambiguous commands (should use confidence scoring)
            "Check this",
            "Look at that window",
            "Analyze the thing I'm working on"
        ]
        
        print(f"\n{Fore.YELLOW}Testing {len(test_commands)} different vision commands...{Style.RESET_ALL}\n")
        
        for i, command in enumerate(test_commands, 1):
            print(f"{Fore.BLUE}Test {i}: {command}{Style.RESET_ALL}")
            
            # Process command
            response = await vision_system.process_command(command)
            
            # Display results
            print(f"  Intent Type: {response.intent_type}")
            print(f"  Confidence: {response.confidence:.2%}")
            print(f"  Success: {response.success}")
            print(f"  Response: {response.message[:100]}..." if len(response.message) > 100 else f"  Response: {response.message}")
            
            if response.suggestions:
                print(f"  Suggestions: {', '.join(response.suggestions)}")
            
            print()
            
            # Small delay between tests
            await asyncio.sleep(0.5)
        
        # Test confidence threshold auto-tuning
        print(f"\n{Fore.YELLOW}Testing confidence threshold auto-tuning...{Style.RESET_ALL}")
        
        # Get ML classifier stats
        from vision.ml_intent_classifier import get_ml_intent_classifier
        classifier = get_ml_intent_classifier()
        
        initial_threshold = classifier.confidence_threshold
        print(f"Initial confidence threshold: {initial_threshold:.3f}")
        
        # Simulate some successful and failed interactions
        test_patterns = [
            ("describe my screen", "screen_analysis_request", True),
            ("can you see", "vision_capability_confirmation", True),
            ("xyz123 gibberish", "unknown", False),
            ("show me the screen", "screen_analysis_request", True)
        ]
        
        for command, intent, success in test_patterns:
            classifier.learn_from_interaction(command, intent, success)
        
        # Check if threshold adjusted
        new_threshold = classifier.confidence_threshold
        print(f"New confidence threshold: {new_threshold:.3f}")
        print(f"Threshold {'adjusted' if new_threshold != initial_threshold else 'unchanged'}")
        
        # Test real-time pattern learning
        print(f"\n{Fore.YELLOW}Testing real-time pattern learning...{Style.RESET_ALL}")
        
        # Check learned patterns
        pattern_stats = classifier.export_patterns_for_visualization()
        print(f"Total learned patterns: {pattern_stats['total_patterns']}")
        print("Learned intent types:")
        for intent, stats in pattern_stats['intents'].items():
            print(f"  - {intent}: {stats['pattern_count']} patterns, {stats['avg_confidence']:.2%} confidence")
        
        # Test semantic understanding
        print(f"\n{Fore.YELLOW}Testing semantic understanding engine...{Style.RESET_ALL}")
        
        from vision.semantic_understanding_engine import get_semantic_understanding_engine
        semantic_engine = get_semantic_understanding_engine()
        
        # Test context-aware understanding
        test_context = {
            'user_activity': 'coding',
            'active_windows': ['VS Code', 'Terminal'],
            'time_of_day': 'morning'
        }
        
        understanding = await semantic_engine.understand_intent(
            "Can you check if there are any errors on my screen?",
            test_context
        )
        
        print(f"Primary intent: {understanding.primary_intent}")
        print(f"Sub-intents: {understanding.sub_intents}")
        print(f"Language detected: {understanding.context.language}")
        print(f"Confidence: {understanding.confidence:.2%}")
        print(f"Ambiguity score: {understanding.ambiguity_score:.2%}")
        print(f"Clarification needed: {understanding.clarification_needed}")
        
        # Display system statistics
        print(f"\n{Fore.YELLOW}System Statistics:{Style.RESET_ALL}")
        stats = await vision_system.get_system_stats()
        
        print(f"Total interactions: {stats['total_interactions']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Confidence threshold: {stats['confidence_threshold']:.3f}")
        print(f"Learned patterns: {stats['learned_patterns']}")
        print(f"Vision capabilities: {stats['vision_capabilities']}")
        print(f"Claude API available: {stats['claude_available']}")
        
        print(f"\n{Fore.GREEN}✓ Phase 1 P0 features test completed successfully!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}✗ Error during testing: {e}{Style.RESET_ALL}")
        logger.error(f"Test error: {e}", exc_info=True)


async def test_integration_with_existing_system():
    """Test integration with existing JARVIS voice system"""
    print(f"\n{Fore.CYAN}=== Testing Integration with Existing System ==={Style.RESET_ALL}\n")
    
    try:
        # Test through intelligent command handler
        from voice.intelligent_command_handler import IntelligentCommandHandler
        handler = IntelligentCommandHandler()
        
        # Test vision command routing
        test_commands = [
            "Can you see my screen?",
            "Describe what's on my desktop"
        ]
        
        for command in test_commands:
            print(f"{Fore.BLUE}Testing: {command}{Style.RESET_ALL}")
            response, handler_type = await handler.handle_command(command)
            print(f"  Handler used: {handler_type}")
            print(f"  Response: {response[:100]}..." if len(response) > 100 else f"  Response: {response}")
            print()
        
        # Test through vision action handler
        from system_control.vision_action_handler import get_vision_action_handler
        vision_handler = get_vision_action_handler()
        
        # Check if v2.0 system is being used
        if hasattr(vision_handler, 'vision_system_v2') and vision_handler.vision_system_v2:
            print(f"{Fore.GREEN}✓ Vision Action Handler is using Vision System v2.0{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}! Vision Action Handler is using legacy system{Style.RESET_ALL}")
        
        # Test direct vision actions
        result = await vision_handler.describe_screen({'query': 'What applications are open?'})
        print(f"\nDirect action test:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Description: {result.description[:100]}...")
        
        print(f"\n{Fore.GREEN}✓ Integration test completed!{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"\n{Fore.RED}✗ Integration test error: {e}{Style.RESET_ALL}")
        logger.error(f"Integration test error: {e}", exc_info=True)


async def main():
    """Run all Phase 1 tests"""
    print(f"{Fore.MAGENTA}")
    print("=" * 60)
    print("JARVIS Vision System v2.0 - Phase 1 Test Suite")
    print("Testing ML-Based Intent Classification & Semantic Understanding")
    print("=" * 60)
    print(f"{Style.RESET_ALL}")
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{Fore.YELLOW}⚠️  Warning: ANTHROPIC_API_KEY not set - some features may be limited{Style.RESET_ALL}")
    
    # Run P0 feature tests
    await test_phase_1_p0_features()
    
    # Run integration tests
    await test_integration_with_existing_system()
    
    print(f"\n{Fore.MAGENTA}=== All Phase 1 tests completed ==={Style.RESET_ALL}\n")


if __name__ == "__main__":
    asyncio.run(main())