#!/usr/bin/env python3
"""Debug multi-space ValueError"""

import asyncio
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_multi_space_query():
    """Test the multi-space query that's failing"""
    from api.pure_vision_intelligence import PureVisionIntelligence
    import numpy as np
    
    # Mock Claude client
    class MockClaudeClient:
        async def analyze_image_with_prompt(self, image, prompt, max_tokens=500, **kwargs):
            return {
                'text': f"I can see Terminal is running on Desktop 2.",
                'detailed_description': "Terminal application detected"
            }
        
        async def analyze_multiple_images_with_prompt(self, images, prompt, max_tokens=1000, **kwargs):
            return {
                'text': f"Looking across your {len(images)} desktop spaces, I can see Terminal is running on Desktop 2. You have VS Code on Desktop 1, Chrome on Desktop 3, and Slack on Desktop 4.",
                'detailed_description': "Multi-space analysis complete"
            }
    
    try:
        # Create vision intelligence
        vision = PureVisionIntelligence(MockClaudeClient(), enable_multi_space=True)
        
        # Create mock screenshot
        screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Test the query that's failing
        print("Testing query: 'Where is Terminal?'")
        response = await vision.understand_and_respond(screenshot, "Where is Terminal?")
        print(f"Success! Response: {response}")
        
    except Exception as e:
        print(f"\nError occurred: {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        
        # Additional debugging
        if hasattr(e, '__traceback__'):
            tb = traceback.extract_tb(e.__traceback__)
            print("\nError location:")
            for frame in tb[-3:]:  # Last 3 frames
                print(f"  File: {frame.filename}")
                print(f"  Line {frame.lineno}: {frame.line}")
                print(f"  Function: {frame.name}")
                print()

if __name__ == "__main__":
    asyncio.run(test_multi_space_query())