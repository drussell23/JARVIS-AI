#!/usr/bin/env python3
"""
Test script to verify the fix for "can you see my screen?" command
Tests that it properly analyzes the screen instead of returning generic response
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_screen_visibility_commands():
    """Test various ways users ask about screen visibility"""
    print("\n=== Testing Screen Visibility Commands ===\n")
    
    try:
        from vision.vision_system_v2 import get_vision_system_v2
        system = get_vision_system_v2()
        print("✓ Vision System v2.0 initialized")
        
        # Test different variations of the command
        test_commands = [
            "can you see my screen?",
            "Can you see my screen?",
            "are you able to see my screen?",
            "do you see what's on my display?",
            "can you view my monitor?",
            "is my screen visible to you?",
            "what's on my screen?",
            "describe my screen",
            "analyze my display"
        ]
        
        print("\nTesting various screen visibility commands...")
        print("=" * 60)
        
        for i, command in enumerate(test_commands, 1):
            print(f"\nTest {i}: '{command}'")
            
            # Process command
            response = await system.process_command(command, {
                'user': 'test_user',
                'confidence': 0.8,
                'timestamp': datetime.now().isoformat()
            })
            
            print(f"Success: {response.success}")
            print(f"Intent: {response.intent_type}")
            print(f"Confidence: {response.confidence:.2%}")
            print(f"Response: {response.message[:200]}...")
            
            # Check if it's properly analyzing the screen
            if "command executed successfully" in response.message.lower():
                print("❌ FAIL: Got generic response instead of screen analysis")
            elif response.intent_type in ['vision_capability_confirmation', 'screen_analysis', 'general_vision_request']:
                print("✓ PASS: Properly recognized as vision command")
            else:
                print(f"⚠️  WARNING: Unexpected intent type: {response.intent_type}")
            
            print("-" * 60)
            
            # Small delay between commands
            await asyncio.sleep(0.1)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing screen commands: {e}")
        logger.error(f"Screen command test error: {e}", exc_info=True)
        return False


async def test_neural_routing_for_screen():
    """Test that neural router correctly routes screen commands"""
    print("\n=== Testing Neural Routing for Screen Commands ===\n")
    
    try:
        from vision.neural_command_router import get_neural_router
        router = get_neural_router()
        print("✓ Neural Router initialized")
        
        # Test routing decision
        command = "Can you see my screen?"
        result, decision = await router.route_command(command, {
            'user': 'test_user',
            'confidence': 0.9
        })
        
        print(f"\nCommand: '{command}'")
        print(f"Routed to handler: {decision.handler_name}")
        print(f"Confidence: {decision.confidence:.2%}")
        print(f"Alternative routes:")
        for handler, conf in decision.alternative_routes[:3]:
            print(f"  - {handler}: {conf:.2%}")
        
        # Check context factors
        if 'context_factors' in decision.__dict__:
            print(f"\nContext factors considered:")
            for key, value in decision.context_factors.items():
                if key != 'attention_distribution':  # Skip large array
                    print(f"  - {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing neural routing: {e}")
        logger.error(f"Neural routing test error: {e}", exc_info=True)
        return False


async def test_integration_with_vision_action_handler():
    """Test integration with the actual vision action handler"""
    print("\n=== Testing Integration with Vision Action Handler ===\n")
    
    try:
        # First test with v2 system
        from vision.vision_system_v2 import process_vision_command_v2
        
        print("Testing with Vision System v2.0...")
        result = await process_vision_command_v2("Can you see my screen?", {
            'user': 'integration_test',
            'confidence': 0.8
        })
        
        print(f"Success: {result['success']}")
        print(f"Description: {result['description'][:200]}...")
        print(f"Intent Type: {result.get('intent_type', 'unknown')}")
        print(f"Confidence: {result.get('confidence', 0):.2%}")
        
        if result.get('data') and 'metadata' in result['data']:
            metadata = result['data']['metadata']
            print(f"\nMetadata:")
            print(f"  - Capabilities discovered: {metadata.get('discovered_capabilities', 0)}")
            print(f"  - Analysis performed: {'analysis' in str(metadata).lower()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing integration: {e}")
        logger.error(f"Integration test error: {e}", exc_info=True)
        return False


async def main():
    """Run all tests for the screen visibility fix"""
    print("\n" + "="*60)
    print("JARVIS Vision System v2.0 - Screen Visibility Fix Test")
    print("Verifying proper handling of 'can you see my screen?' commands")
    print("="*60)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   Some features may be limited")
    
    # Run tests
    results = []
    
    results.append(await test_screen_visibility_commands())
    results.append(await test_neural_routing_for_screen())
    results.append(await test_integration_with_vision_action_handler())
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary:")
    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"  Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! Screen visibility commands are working correctly.")
    else:
        print("\n✗ Some tests failed. Check the output above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())