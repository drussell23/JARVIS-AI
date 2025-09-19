#!/usr/bin/env python3
"""
Test script for Multi-Space Capture Engine integration
Tests the enhanced PureVisionIntelligence with multi-space screenshot capture
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from api.pure_vision_intelligence import PureVisionIntelligence
from vision.multi_space_capture_engine import (
    MultiSpaceCaptureEngine, SpaceCaptureRequest, CaptureQuality
)
import numpy as np
from PIL import Image
import logging

# Mock Claude client for testing
class MockClaudeClient:
    """Mock Claude client that returns test responses"""
    
    async def send_message(self, model, messages, max_tokens=4096):
        # Return a mock response
        return type('Response', (), {
            'content': [type('Content', (), {
                'text': f"Mock response for: {messages[-1]['content'][-1]['text'][:100] if messages else 'No message'}"
            })()]
        })()
        
    def create_message(self, model, messages, max_tokens=4096):
        # Return a mock response
        return type('Response', (), {
            'content': [type('Content', (), {
                'text': f"Mock response for: {messages[-1]['content'][-1]['text'][:100] if messages else 'No message'}"
            })()]
        })()
    
    async def analyze_image_with_prompt(self, image, prompt, max_tokens=500, **kwargs):
        """Mock analyze_image_with_prompt method"""
        return {
            'text': f"Mock vision analysis: I see a desktop with multiple spaces. The query was about: {prompt[:100]}...",
            'detailed_description': "Mock detailed description of the screen content"
        }

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_multi_space_queries():
    """Test various multi-space queries"""
    
    print("\n🚀 Testing Multi-Space Capture Integration\n")
    
    # Initialize vision intelligence with mock client
    claude_client = MockClaudeClient()
    vision = PureVisionIntelligence(claude_client, enable_multi_space=True)
    
    # Test queries that should trigger multi-space capture
    test_queries = [
        "Show me all my workspaces",
        "What's on Desktop 2?",
        "Find Terminal across all spaces",
        "Which space has VSCode open?",
        "Show me everything I'm working on",
        "What applications are running on other desktops?"
    ]
    
    # Get current screenshot for comparison
    print("📸 Capturing current space screenshot...")
    current_screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        print("-" * 50)
        
        try:
            # Test the vision intelligence response
            response = await vision.understand_and_respond(
                current_screenshot,
                query
            )
            
            print(f"✅ Response: {response[:200]}...")
            
            # Check if multi-space was triggered
            if hasattr(vision, '_last_multi_space_context'):
                context = vision._last_multi_space_context
                if context:
                    print(f"📊 Multi-space context created:")
                    print(f"   - Total spaces: {context.total_spaces}")
                    print(f"   - Analyzed spaces: {context.analyzed_spaces}")
                    print(f"   - Total windows: {context.total_windows}")
                    print(f"   - Query type: {context.query_type}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

async def test_capture_engine_directly():
    """Test the capture engine directly"""
    
    print("\n\n🔧 Testing Capture Engine Directly\n")
    
    # Initialize capture engine
    engine = MultiSpaceCaptureEngine()
    
    # Enumerate available spaces
    print("📋 Enumerating spaces...")
    spaces = await engine.enumerate_spaces()
    print(f"Found {len(spaces)} spaces: {spaces}")
    
    # Get current space
    current = await engine.get_current_space()
    print(f"Current space: {current}")
    
    # Test capture request
    if len(spaces) > 1:
        print(f"\n🎯 Attempting to capture spaces {spaces[:2]}...")
        
        request = SpaceCaptureRequest(
            space_ids=spaces[:2],
            quality=CaptureQuality.FAST,
            use_cache=False,
            reason="test_capture",
            require_permission=False  # For testing
        )
        
        result = await engine.capture_all_spaces(request)
        
        print(f"\n📊 Capture Results:")
        print(f"   - Success: {result.success}")
        print(f"   - Duration: {result.total_duration:.2f}s")
        print(f"   - Screenshots captured: {len(result.screenshots)}")
        print(f"   - Errors: {result.errors}")
        print(f"   - Cache hits: {result.cache_hits}")
        print(f"   - New captures: {result.new_captures}")
        
        for space_id, metadata in result.metadata.items():
            print(f"\n   Space {space_id}:")
            print(f"     - Method: {metadata.capture_method.value}")
            print(f"     - Resolution: {metadata.resolution}")
            print(f"     - Apps: {', '.join(metadata.applications[:5])}")
            print(f"     - Windows: {metadata.window_count}")

async def test_permission_flow():
    """Test permission-based capture"""
    
    print("\n\n🔐 Testing Permission Flow\n")
    
    engine = MultiSpaceCaptureEngine()
    spaces = await engine.enumerate_spaces()
    
    if len(spaces) > 1:
        print(f"Requesting permission to capture space {spaces[1]}...")
        
        request = SpaceCaptureRequest(
            space_ids=[spaces[1]],
            quality=CaptureQuality.OPTIMIZED,
            reason="user_request",
            require_permission=True
        )
        
        # Note: This will actually request permission
        print("⚠️  Set JARVIS_AUTO_APPROVE_SPACE_SWITCH=true to auto-approve")
        
        result = await engine.capture_all_spaces(request)
        
        if result.success:
            print("✅ Permission granted and capture successful")
        else:
            print(f"❌ Capture failed: {result.errors}")

async def main():
    """Run all tests"""
    
    # Test 1: Multi-space queries through PureVisionIntelligence
    await test_multi_space_queries()
    
    # Test 2: Direct capture engine test
    await test_capture_engine_directly()
    
    # Test 3: Permission flow (optional)
    if '--with-permissions' in sys.argv:
        await test_permission_flow()
    else:
        print("\n💡 Tip: Run with --with-permissions to test permission flow")
    
    print("\n\n✅ All tests completed!")

if __name__ == "__main__":
    # Set environment for testing
    import os
    os.environ['JARVIS_AUTO_APPROVE_SPACE_SWITCH'] = 'true'
    
    asyncio.run(main())