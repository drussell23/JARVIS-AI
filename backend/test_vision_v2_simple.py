#!/usr/bin/env python3
"""
Simple test script for JARVIS Vision System v2.0 Phase 1
Tests without external dependencies
"""

import asyncio
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_ml_intent_classifier():
    """Test ML-based intent classification"""
    print("\n=== Testing ML Intent Classifier ===\n")
    
    try:
        from vision.ml_intent_classifier import get_ml_intent_classifier
        classifier = get_ml_intent_classifier()
        print("✓ ML Intent Classifier initialized")
        
        # Test intent classification
        test_commands = [
            "Can you see my screen?",
            "Describe what's on my desktop",
            "What am I looking at?",
            "Check for errors"
        ]
        
        for command in test_commands:
            intent = classifier.classify_intent(command)
            print(f"\nCommand: {command}")
            print(f"  Intent Type: {intent.intent_type}")
            print(f"  Confidence: {intent.confidence:.2%}")
        
        # Test pattern learning
        print("\n--- Testing Pattern Learning ---")
        classifier.learn_from_interaction(
            "show me my screen", 
            "screen_analysis_request", 
            True
        )
        print("✓ Pattern learned successfully")
        
        # Check confidence threshold
        print(f"\nCurrent confidence threshold: {classifier.confidence_threshold:.3f}")
        
        # Export statistics
        stats = classifier.export_patterns_for_visualization()
        print(f"Total learned patterns: {stats['total_patterns']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing ML classifier: {e}")
        logger.error(f"ML classifier test error: {e}", exc_info=True)
        return False


async def test_semantic_understanding():
    """Test semantic understanding engine"""
    print("\n=== Testing Semantic Understanding Engine ===\n")
    
    try:
        from vision.semantic_understanding_engine import get_semantic_understanding_engine
        engine = get_semantic_understanding_engine()
        print("✓ Semantic Understanding Engine initialized")
        
        # Test understanding
        test_text = "Can you see what's happening on my screen?"
        understanding = await engine.understand_intent(test_text)
        
        print(f"\nAnalyzing: '{test_text}'")
        print(f"  Primary Intent: {understanding.primary_intent}")
        print(f"  Confidence: {understanding.confidence:.2%}")
        print(f"  Language: {understanding.context.language}")
        print(f"  Question Type: {understanding.context.question_type}")
        print(f"  Ambiguity Score: {understanding.ambiguity_score:.2%}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing semantic engine: {e}")
        logger.error(f"Semantic engine test error: {e}", exc_info=True)
        return False


async def test_dynamic_vision_engine():
    """Test dynamic vision engine"""
    print("\n=== Testing Dynamic Vision Engine ===\n")
    
    try:
        from vision.dynamic_vision_engine import get_dynamic_vision_engine
        engine = get_dynamic_vision_engine()
        print("✓ Dynamic Vision Engine initialized")
        
        # Check capabilities
        print(f"Discovered capabilities: {len(engine.capabilities)}")
        
        # Test vision command processing
        test_command = "describe what is on my screen"
        response, metadata = await engine.process_vision_command(test_command)
        
        print(f"\nProcessed command: '{test_command}'")
        print(f"  Success: {metadata.get('success', 'Unknown')}")
        print(f"  Response: {response[:100]}..." if len(response) > 100 else f"  Response: {response}")
        
        # Get statistics
        stats = engine.get_statistics()
        print(f"\nEngine Statistics:")
        print(f"  Total capabilities: {stats['total_capabilities']}")
        print(f"  Commands processed: {stats['total_commands_processed']}")
        print(f"  Learned patterns: {stats['learned_patterns']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing dynamic engine: {e}")
        logger.error(f"Dynamic engine test error: {e}", exc_info=True)
        return False


async def test_vision_system_v2():
    """Test integrated Vision System v2.0"""
    print("\n=== Testing Vision System v2.0 Integration ===\n")
    
    try:
        from vision.vision_system_v2 import get_vision_system_v2
        system = get_vision_system_v2()
        print("✓ Vision System v2.0 initialized")
        
        # Test command processing
        test_cases = [
            ("Can you see my screen?", "capability confirmation"),
            ("Describe my desktop", "screen description"),
            ("What's happening on my display?", "general query")
        ]
        
        for command, description in test_cases:
            print(f"\nTesting {description}: '{command}'")
            response = await system.process_command(command)
            
            print(f"  Success: {response.success}")
            print(f"  Intent: {response.intent_type}")
            print(f"  Confidence: {response.confidence:.2%}")
            print(f"  Message: {response.message[:80]}..." if len(response.message) > 80 else f"  Message: {response.message}")
        
        # Get system statistics
        stats = await system.get_system_stats()
        print(f"\nSystem Statistics:")
        print(f"  Version: {stats['version']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Confidence threshold: {stats['confidence_threshold']:.3f}")
        print(f"  Claude available: {stats['claude_available']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing Vision System v2.0: {e}")
        logger.error(f"Vision System v2.0 test error: {e}", exc_info=True)
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("JARVIS Vision System v2.0 - Phase 1 Test Suite")
    print("Testing ML-Based Components")
    print("="*60)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   Vision analysis features will be limited")
    
    # Run tests
    results = []
    
    results.append(await test_ml_intent_classifier())
    results.append(await test_semantic_understanding())
    results.append(await test_dynamic_vision_engine())
    results.append(await test_vision_system_v2())
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())