#!/usr/bin/env python3
"""
Fix to ensure only clean text is sent as response, not the full result object
"""

import logging

logger = logging.getLogger(__name__)

# Monkey patch to intercept vision responses
_original_vision_handler_describe = None
_original_vision_analyzer_analyze = None

def patch_vision_responses():
    """Patch vision system to return clean text only"""
    global _original_vision_handler_describe, _original_vision_analyzer_analyze
    
    try:
        # Try to patch vision handler describe_screen
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        
        if hasattr(ClaudeVisionAnalyzer, 'analyze_image_async'):
            _original_vision_analyzer_analyze = ClaudeVisionAnalyzer.analyze_image_async
            
            async def clean_analyze_image_async(self, image, prompt, **kwargs):
                result = await _original_vision_analyzer_analyze(self, image, prompt, **kwargs)
                # If result has 'content' and 'raw_result', return just the content text
                if isinstance(result, dict) and 'content' in result:
                    return result['content']
                return result
                
            ClaudeVisionAnalyzer.analyze_image_async = clean_analyze_image_async
            logger.info("Patched ClaudeVisionAnalyzer.analyze_image_async to return clean text")
            
    except Exception as e:
        logger.warning(f"Could not patch vision responses: {e}")

# Apply the patch immediately
patch_vision_responses()