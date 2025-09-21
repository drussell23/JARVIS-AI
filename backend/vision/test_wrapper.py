#!/usr/bin/env python3
"""Test the vision analyzer wrapper"""

import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_wrapper():
    """Test the wrapped vision analyzer"""
    
    # Import the wrapper
    from claude_vision_analyzer import ClaudeVisionAnalyzer
    
    # Initialize
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY")
        return
    
    analyzer = ClaudeVisionAnalyzer(api_key)
    logger.info("✅ Analyzer initialized")
    
    # Test get_screen_context
    logger.info("\n🔍 Testing get_screen_context()...")
    result = await analyzer.get_screen_context()
    
    logger.info(f"Result type: {type(result)}")
    logger.info(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    
    if isinstance(result, dict) and 'description' in result:
        logger.info(f"\n✅ Screen context: {result['description'][:200]}...")
    else:
        logger.error(f"❌ Unexpected result: {result}")
    
    # Test with direct screenshot
    import subprocess
    import tempfile
    from PIL import Image
    import numpy as np
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        subprocess.run(['screencapture', '-x', tmp_path], check=True)
        image = Image.open(tmp_path)
        screenshot = np.array(image)
        
        logger.info(f"\n🔍 Testing analyze_screenshot()...")
        result = await analyzer.analyze_screenshot(
            screenshot,
            "What applications are visible on screen?"
        )
        
        logger.info(f"Result type: {type(result)}")
        if isinstance(result, dict) and 'description' in result:
            logger.info(f"✅ Analysis: {result['description'][:200]}...")
        
    finally:
        os.unlink(tmp_path)
    
    await analyzer.cleanup_all_components()
    logger.info("\n✅ All tests completed")

if __name__ == "__main__":
    asyncio.run(test_wrapper())