#!/usr/bin/env python3
"""
Test multi-space integration with JARVIS chatbot
"""

import asyncio
import sys
import os
import traceback
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def test_jarvis_multi_space():
    """Test JARVIS with multi-space query"""
    print("=== Testing JARVIS Multi-Space Integration ===\n")
    
    try:
        # Import the chatbot
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        
        # Create a mock API key for testing
        api_key = os.getenv('ANTHROPIC_API_KEY', 'test-key')
        
        # Create chatbot instance
        chatbot = ClaudeVisionChatbot(api_key=api_key, model='claude-3-5-sonnet-20241022')
        print("✓ ClaudeVisionChatbot created")
        
        # Test query
        test_query = "Where is the Terminal?"
        print(f"\nQuery: {test_query}")
        
        # Try to process with better error handling
        try:
            # Get a mock screenshot
            from PIL import Image
            import numpy as np
            
            # Create a simple test image
            screenshot = Image.new('RGB', (1920, 1080), color='white')
            
            # Process the query
            response = await chatbot.chat(test_query)
            print(f"\nResponse: {response}")
            
        except Exception as e:
            print(f"\n✗ Error during chat processing")
            print(f"  Error type: {type(e).__name__}")
            print(f"  Error message: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            
            # Check if it's related to multi-space
            print("\nChecking multi-space components...")
            
            # Check PureVisionIntelligence
            try:
                from api.pure_vision_intelligence import PureVisionIntelligence
                
                # Mock Claude client
                class MockClaude:
                    async def analyze_image_with_prompt(self, image, prompt, max_tokens):
                        return {'content': 'Test response'}
                
                intelligence = PureVisionIntelligence(MockClaude(), enable_multi_space=True)
                print(f"✓ PureVisionIntelligence multi_space_enabled: {intelligence.multi_space_enabled}")
                
                # Test if query triggers multi-space
                if intelligence._should_use_multi_space(test_query):
                    print("✓ Query correctly detected as multi-space")
                    
                    # Try to gather multi-space data
                    try:
                        window_data = await intelligence._gather_multi_space_data()
                        print(f"✓ Multi-space data gathered: {len(window_data.get('windows', []))} windows")
                    except Exception as e2:
                        print(f"✗ Error gathering multi-space data: {e2}")
                        traceback.print_exc()
                        
            except Exception as e3:
                print(f"✗ Error testing PureVisionIntelligence: {e3}")
                
    except Exception as e:
        print(f"✗ Failed to create chatbot: {e}")
        traceback.print_exc()

async def test_direct_vision_handler():
    """Test vision handler directly"""
    print("\n=== Testing Vision Handler Directly ===\n")
    
    try:
        from api.vision_command_handler import VisionCommandHandler
        
        handler = VisionCommandHandler()
        await handler.initialize_intelligence()
        
        print("✓ VisionCommandHandler initialized")
        
        # Test with multi-space query
        test_query = "Where is the Terminal?"
        
        # Create mock screenshot
        from PIL import Image
        screenshot = Image.new('RGB', (100, 100), color='white')
        
        response = await handler.intelligence.understand_and_respond(screenshot, test_query)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"✗ Vision handler test failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_jarvis_multi_space())
    asyncio.run(test_direct_vision_handler())