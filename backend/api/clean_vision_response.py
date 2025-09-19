#!/usr/bin/env python3
"""
Clean Vision Response Interceptor
Ensures only clean text is sent to frontend, not full vision result objects
"""

import logging
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)

def clean_vision_response(response: Any) -> str:
    """
    Extract clean text from various vision response formats.
    Handles cases where the full vision analyzer result is returned.
    """
    # If it's already a string, return it
    if isinstance(response, str):
        return response
    
    # If it's a dict, try to extract the text
    if isinstance(response, dict):
        # Check if this looks like a vision analyzer result
        if 'content' in response and 'raw_result' in response and 'success' in response:
            # This is the full vision analyzer result - extract just the content
            content = response.get('content', '')
            if isinstance(content, str):
                logger.info(f"[CLEAN VISION] Extracted content from vision analyzer result: {content[:100]}...")
                return content
            elif isinstance(content, dict) and 'content' in content:
                # Nested content
                return clean_vision_response(content)
        
        # Check other common response formats
        if 'response' in response:
            return clean_vision_response(response['response'])
        if 'text' in response:
            return clean_vision_response(response['text'])
        if 'content' in response:
            return clean_vision_response(response['content'])
        if 'description' in response:
            return clean_vision_response(response['description'])
        if 'message' in response:
            return clean_vision_response(response['message'])
        
        # If it has a raw_result with description, use that
        if 'raw_result' in response and isinstance(response['raw_result'], dict):
            if 'description' in response['raw_result']:
                logger.info("[CLEAN VISION] Extracting from raw_result.description")
                return response['raw_result']['description']
    
    # Try to convert to string as last resort
    result = str(response)
    
    # Don't return dict/object representations
    if result.startswith('{') and result.endswith('}'):
        logger.warning(f"[CLEAN VISION] Unable to extract clean text from response: {result[:200]}...")
        return "I processed your request, but encountered a formatting issue with my response."
    
    return result


def patch_jarvis_voice_api():
    """
    Patch the JARVIS Voice API to clean vision responses before sending to frontend.
    """
    try:
        import api.jarvis_voice_api as jarvis_voice_api
        
        # Store original send_json method
        if hasattr(jarvis_voice_api, '_original_websocket_send_json'):
            return  # Already patched
            
        # Patch WebSocket send_json to intercept vision responses
        original_send_json = None
        
        async def clean_send_json(websocket, data: Dict[str, Any]):
            """Intercept and clean vision responses before sending"""
            # Check if this is a response message
            if data.get('type') == 'response' and 'text' in data:
                # Clean the text field
                original_text = data['text']
                cleaned_text = clean_vision_response(original_text)
                
                if original_text != cleaned_text:
                    logger.info(f"[CLEAN VISION] Cleaned response before sending to frontend")
                    data['text'] = cleaned_text
            
            # Call original send_json
            await websocket.send_json(data)
        
        # Store the patch function for use
        jarvis_voice_api._clean_send_json = clean_send_json
        logger.info("[CLEAN VISION] Patched JARVIS Voice API for clean responses")
        
    except Exception as e:
        logger.error(f"[CLEAN VISION] Failed to patch JARVIS Voice API: {e}")


# Auto-patch on import
patch_jarvis_voice_api()