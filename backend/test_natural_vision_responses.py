#!/usr/bin/env python3
"""
Test the new natural vision responses
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from vision.natural_responses import get_response_generator

def test_natural_responses():
    """Test various natural response scenarios."""
    gen = get_response_generator()
    
    print("Testing Natural Vision Confirmations:")
    print("=" * 50)
    
    # Test basic confirmation
    print("\n1. Basic confirmation (no context):")
    for _ in range(3):
        response = gen.generate_vision_confirmation()
        print(f"   - {response}")
    
    # Test with window count
    print("\n2. With multiple windows:")
    context = {'window_count': 5}
    for _ in range(3):
        response = gen.generate_vision_confirmation(context)
        print(f"   - {response}")
    
    # Test with active app
    print("\n3. With active application:")
    apps = ['Terminal', 'Chrome', 'Cursor', 'Slack']
    for app in apps:
        context = {'active_app': app}
        response = gen.generate_vision_confirmation(context)
        print(f"   - {response}")
    
    # Test with full context
    print("\n4. With full context:")
    context = {
        'window_count': 3,
        'active_app': 'Cursor'
    }
    for _ in range(3):
        response = gen.generate_vision_confirmation(context)
        print(f"   - {response}")
    
    # Test technical response simplification
    print("\n5. Technical response simplification:")
    technical = """Based on the screenshot, it appears that the system has successfully detected and described the screen. The output shows that the "describe_screen" intent has been executed successfully, indicating that the system can perceive the contents of your screen.

The screenshot appears to show a terminal interface with various logs and information related to a vision system. The system seems to be functioning properly and has registered various vision capabilities.

I don't see any obvious issues or errors in the output. The system seems to be working as expected and should be able to provide a comprehensive context about what's on your screen."""
    
    simplified = gen.simplify_technical_response(technical)
    print(f"\nOriginal ({len(technical)} chars):")
    print(technical[:200] + "...")
    print(f"\nSimplified ({len(simplified)} chars):")
    print(simplified)

async def test_integration():
    """Test integration with vision system."""
    try:
        from vision.vision_system_v2 import get_vision_system_v2
        
        print("\n\n6. Testing with actual Vision System:")
        print("=" * 50)
        
        vision = get_vision_system_v2()
        
        # Test "can you see my screen" command
        response = await vision.process_command("can you see my screen?")
        
        print(f"Success: {response.success}")
        print(f"Message: {response.message}")
        print(f"Intent: {response.intent_type}")
        print(f"Confidence: {response.confidence:.2f}")
        
    except Exception as e:
        print(f"Could not test integration: {e}")

if __name__ == "__main__":
    # Test natural responses
    test_natural_responses()
    
    # Test integration
    print("\n" + "=" * 50)
    asyncio.run(test_integration())