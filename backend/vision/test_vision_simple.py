#!/usr/bin/env python3
"""
Simple test to verify vision is working with actual screen capture
"""

import asyncio
import os
import subprocess
import tempfile
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def capture_and_test():
    """Capture screen and test vision analysis"""
    
    # Check API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        logger.error("Please set ANTHROPIC_API_KEY")
        return
    
    # Capture screenshot
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Capture using macOS command
        subprocess.run(['screencapture', '-x', tmp_path], check=True)
        
        # Load image
        image = Image.open(tmp_path)
        screenshot = np.array(image)
        logger.info(f"Captured screenshot: {screenshot.shape}")
        
        # Import and test analyzer
        from claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
        
        # Simple test
        logger.info("Analyzing screen...")
        result = await analyzer.analyze_screenshot(
            screenshot,
            "Describe what you see on this computer screen in detail. What applications, windows, or content is visible?"
        )
        
        logger.info(f"Raw result type: {type(result)}")
        logger.info(f"Raw result: {result}")
        
        # Handle different response formats
        if isinstance(result, dict):
            if 'description' in result:
                logger.info(f"\n✅ Analysis: {result['description']}")
            elif 'content' in result:
                logger.info(f"\n✅ Analysis: {result['content']}")
            else:
                logger.info(f"\n✅ Analysis: {result}")
        elif isinstance(result, str):
            logger.info(f"\n✅ Analysis: {result}")
        else:
            logger.info(f"\n❓ Unexpected result type: {type(result)}")
        
        # Cleanup
        await analyzer.cleanup_all_components()
        
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

if __name__ == "__main__":
    asyncio.run(capture_and_test())