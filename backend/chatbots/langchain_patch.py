"""
Patch for LangChain to use quantized models automatically
This module patches the existing langchain_chatbot to use optimized quantized models
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)


def apply_quantized_patch():
    """Apply patch to use quantized models in LangChain"""
    
    # Check if we should use quantized models
    use_quantized = os.getenv("USE_QUANTIZED_MODELS", "false").lower() == "true"
    if not use_quantized:
        return
        
    try:
        # Import the optimized version
        from .optimized_langchain_chatbot import OptimizedLangChainChatbot
        
        # Replace the original LangChainChatbot
        sys.modules['backend.chatbots.langchain_chatbot'].LangChainChatbot = OptimizedLangChainChatbot
        
        # Also update the module-level flag
        sys.modules['backend.chatbots.langchain_chatbot'].LANGCHAIN_AVAILABLE = True
        
        logger.info("Applied quantized model patch to LangChain chatbot")
        
    except Exception as e:
        logger.warning(f"Could not apply quantized patch: {e}")


# Auto-apply patch on import
apply_quantized_patch()