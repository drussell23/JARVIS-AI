#!/usr/bin/env python3
"""
Debug script to find the Image.save issue
"""
import sys
import os
import numpy as np
from PIL import Image
import logging

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_image_save():
    """Test basic image save operations"""
    # Create a test image
    test_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    pil_image = Image.fromarray(test_array)
    
    # Test normal save
    import io
    buffer = io.BytesIO()
    
    try:
        # This should work
        pil_image.save(buffer, format="PNG")
        logger.info("Basic save works")
    except Exception as e:
        logger.error(f"Basic save failed: {e}")
    
    # Now test with the analyzer
    try:
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        analyzer = ClaudeVisionAnalyzer("test-key")
        
        # Override PIL Image.save to trace calls
        original_save = Image.Image.save
        def traced_save(self, *args, **kwargs):
            logger.info(f"Image.save called with args: {args}, kwargs: {kwargs}")
            return original_save(self, *args, **kwargs)
        
        Image.Image.save = traced_save
        
        # Run a simple analysis
        import asyncio
        async def run_test():
            result = await analyzer.analyze_screenshot(test_array, "test prompt")
            return result
        
        result = asyncio.run(run_test())
        logger.info("Analysis completed")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original save
        if 'original_save' in locals():
            Image.Image.save = original_save

if __name__ == "__main__":
    test_image_save()