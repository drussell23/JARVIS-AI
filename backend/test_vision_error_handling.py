#!/usr/bin/env python3
"""Test Vision Error Handling - Ensure basic operations work flawlessly"""

import asyncio
import os
import sys
from unittest.mock import Mock, AsyncMock, patch
import logging

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_vision_error_handling():
    """Test error handling in vision operations"""
    print("🧪 Testing Vision Error Handling")
    print("=" * 80)
    
    # Test 1: Test capture_screen error handling
    print("\n1. Testing capture_screen error handling:")
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Create analyzer with mock API key
    analyzer = ClaudeVisionAnalyzer(api_key="test_key")
    
    # Test capture with all methods failing
    with patch.object(analyzer, '_capture_from_video_stream', side_effect=Exception("Video stream error")):
        with patch.object(analyzer, '_capture_macos_screencapture', side_effect=Exception("Screencapture error")):
            with patch.object(analyzer, '_capture_pil_imagegrab', side_effect=Exception("PIL error")):
                result = await analyzer.capture_screen()
                print(f"  ✓ All methods failed, returned: {result}")
                assert result is None, "Should return None when all methods fail"
    
    # Test 2: Test analyze_screenshot error handling
    print("\n2. Testing analyze_screenshot error handling:")
    
    # Test with no API key
    analyzer_no_key = ClaudeVisionAnalyzer(api_key="")
    analyzer_no_key.client = None
    
    # Mock image
    from PIL import Image
    test_image = Image.new('RGB', (100, 100), color='red')
    
    # Test API key error
    result, metrics = await analyzer_no_key.analyze_screenshot(test_image, "Test prompt")
    print(f"  ✓ No API key error handled:")
    print(f"    - Error type: {result.get('error_type')}")
    print(f"    - Description: {result.get('description')}")
    assert 'error' in result
    assert 'api' in result.get('description', '').lower() or 'key' in result.get('description', '').lower()
    
    # Test 3: Test vision command handler error handling
    print("\n3. Testing vision command handler error handling:")
    from api.vision_command_handler import VisionCommandHandler
    
    handler = VisionCommandHandler()
    
    # Test with no vision manager
    result = await handler.analyze_screen("Can you see my screen?")
    print(f"  ✓ No vision manager error handled:")
    print(f"    - Response: {result.get('response')}")
    assert result.get('handled') is True  # Vision command handler returns 'handled' not 'error'
    
    # Test 4: Test chatbot error handling
    print("\n4. Testing chatbot error handling:")
    from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
    
    chatbot = ClaudeVisionChatbot(api_key="test_key")
    chatbot._monitoring_active = True
    
    # Mock capture failure
    with patch.object(chatbot, 'capture_screenshot', side_effect=Exception("Permission denied")):
        response = await chatbot._analyze_current_screen("Can you see my terminal?")
        print(f"  ✓ Capture error handled:")
        print(f"    - Response: {response}")
        assert "error" in response.lower() or "permission" in response.lower()
    
    # Test 5: Test memory error handling
    print("\n5. Testing memory error handling:")
    
    # Create analyzer with memory safety enabled
    from vision.claude_vision_analyzer_main import VisionConfig
    config = VisionConfig(enable_memory_safety=True, reject_on_memory_pressure=True)
    analyzer_mem = ClaudeVisionAnalyzer(api_key="test_key", config=config)
    
    # Mock memory pressure
    with patch.object(analyzer_mem.memory_monitor, 'ensure_memory_available', return_value=False):
        try:
            result, metrics = await analyzer_mem.analyze_screenshot(test_image, "Test")
            print(f"  ✓ Memory error handled:")
            print(f"    - Error type: {result.get('error_type')}")
            print(f"    - Description: {result.get('description')}")
            assert result.get('error_type') == 'MemoryError' or 'memory' in result.get('description', '').lower()
        except MemoryError as e:
            print(f"  ✓ Memory error raised: {e}")
    
    print("\n✅ All error handling tests passed!")
    print("\nSummary:")
    print("- ✓ capture_screen handles all methods failing gracefully")
    print("- ✓ analyze_screenshot provides clear error messages")
    print("- ✓ Vision command handler checks for missing components")
    print("- ✓ Chatbot handles capture errors with user-friendly messages")
    print("- ✓ Memory errors are properly detected and reported")

if __name__ == "__main__":
    asyncio.run(test_vision_error_handling())