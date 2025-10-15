#!/usr/bin/env python3
"""Test the multi-space fix"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.pure_vision_intelligence import PureVisionIntelligence
from api.vision_command_handler import VisionCommandHandler

async def test_fixed_flow():
    """Test the fixed multi-space flow"""
    print("🔧 Testing Fixed Multi-Space Flow")
    print("=" * 80)
    
    # Mock Claude client
    class MockClaudeClient:
        async def analyze_image_with_prompt(self, image, prompt, max_tokens=500):
            # Check if multi-space
            if isinstance(image, dict):
                return {
                    'content': f"Looking across {len(image)} desktop spaces:\n"
                              f"I can see Cursor IDE is open on Desktop 2 with your Python project. "
                              f"Desktop 1 has Chrome and Terminal running."
                }
            else:
                return {'content': 'I see a single desktop with Chrome open.'}
    
    # Test 1: Direct intelligence test
    print("\n1️⃣ Testing PureVisionIntelligence directly:")
    intelligence = PureVisionIntelligence(MockClaudeClient(), enable_multi_space=True)
    
    # Single screenshot
    single_response = await intelligence.understand_and_respond(
        "mock_screenshot", 
        "what do you see?"
    )
    print(f"Single space response: {single_response}")
    
    # Multi-space dict
    multi_screenshots = {1: "screen1", 2: "screen2", 3: "screen3"}
    multi_response = await intelligence.understand_and_respond(
        multi_screenshots,
        "can you see the Cursor IDE in the other desktop space?"
    )
    print(f"Multi-space response: {multi_response}")
    
    # Test 2: Vision command handler flow
    print("\n\n2️⃣ Testing VisionCommandHandler flow:")
    handler = VisionCommandHandler()
    
    # Mock the capture method
    async def mock_capture(multi_space=False, space_number=None):
        if multi_space:
            return {1: "screen1", 2: "screen2"}
        return "single_screen"
    
    handler._capture_screen = mock_capture
    
    # Initialize with mock client
    handler.intelligence = PureVisionIntelligence(MockClaudeClient(), enable_multi_space=True)
    
    # Test the query
    query = "can you see the Cursor IDE in the other desktop space?"
    try:
        result = await handler.handle_command(query)
        print(f"✅ Command handled successfully!")
        print(f"Response: {result.get('response', '')[:100]}...")
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_flow())