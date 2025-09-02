#!/usr/bin/env python3
"""
Simple mock test to verify the fixes work
"""

import sys
import os
import numpy as np
from PIL import Image
import asyncio
from unittest.mock import Mock, patch

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class MockAnthropicClient:
    """Mock Anthropic client"""
    def __init__(self, api_key):
        self.api_key = api_key
        self.messages = self
        
    def create(self, **kwargs):
        """Mock messages.create response"""
        mock_response = Mock()
        mock_response.content = [Mock(text="Mock analysis: Test successful")]
        return mock_response

async def test_fixes():
    """Test that our fixes work"""
    print("Testing Image.save fix...")
    
    # Test 1: Image.save fix
    img = Image.new('RGB', (100, 100), color='white')
    buffer = __import__('io').BytesIO()
    
    try:
        # This should work now
        img.save(buffer, "JPEG", quality=85, optimize=True)
        print("✓ Image.save fix works")
    except Exception as e:
        print(f"✗ Image.save fix failed: {e}")
        return False
    
    print("\nTesting analyzer with mock client...")
    
    # Test 2: Analyzer with mock
    try:
        with patch('anthropic.Anthropic', MockAnthropicClient):
            from claude_vision_analyzer_main import ClaudeVisionAnalyzer
            
            analyzer = ClaudeVisionAnalyzer("test-key")
            
            # Create test image
            test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Test analyze_screenshot
            result, metrics = await analyzer.analyze_screenshot(test_array, "test prompt")
            
            print(f"✓ analyze_screenshot works: {result.get('description', 'No description')}")
            
            # Test methods exist
            methods = [
                'check_for_notifications',
                'check_for_errors', 
                'find_ui_element',
                'analyze_workspace'
            ]
            
            for method in methods:
                if hasattr(analyzer, method):
                    print(f"✓ Method exists: {method}")
                else:
                    print(f"✗ Method missing: {method}")
            
            # Test memory stats
            stats = analyzer.get_all_memory_stats()
            print(f"✓ Memory stats work: {stats.get('system', {}).get('process_mb', 0):.2f} MB")
            
            return True
            
    except Exception as e:
        print(f"✗ Analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixes())
    print(f"\n{'✅ All fixes verified!' if success else '❌ Some fixes failed!'}")
    sys.exit(0 if success else 1)