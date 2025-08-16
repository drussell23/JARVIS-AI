#!/usr/bin/env python3
"""
Force JARVIS to use LangChain mode for math capabilities
"""

import os
import sys
import asyncio
import logging

# Set environment variables before importing
os.environ['PREFER_LANGCHAIN'] = '1'
os.environ['USE_QUANTIZED_MODELS'] = 'true'
os.environ['AUTO_PATCH_TRANSFORMERS'] = 'true'

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from chatbots.dynamic_chatbot import DynamicChatbot
from memory.memory_manager import M1MemoryManager

logging.basicConfig(level=logging.INFO)

async def test_math_with_langchain():
    """Test math capabilities with forced LangChain mode"""
    
    # Create memory manager
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Create chatbot with LangChain preference
    chatbot = DynamicChatbot(
        memory_manager,
        auto_switch=False,  # Disable auto-switching
        prefer_langchain=True
    )
    
    # Force LangChain mode
    print("🚀 Forcing LangChain mode...")
    await chatbot.force_mode("langchain")
    
    # Wait for initialization
    await asyncio.sleep(5)
    
    # Test math queries
    math_queries = [
        "What is 2 + 2?",
        "Calculate 15 * 8",
        "What is 100 divided by 4?",
        "What's 7 plus 13?",
        "Compute the square root of 144"
    ]
    
    print("\n🧮 Testing Math Capabilities in LangChain Mode")
    print("=" * 50)
    
    for query in math_queries:
        print(f"\n📝 Query: {query}")
        try:
            response = await chatbot.generate_response(query)
            print(f"✅ Answer: {response}")
            print(f"🤖 Mode: {chatbot.current_mode.value}")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Check capabilities
    caps = chatbot.get_capabilities()
    print(f"\n📊 Current Capabilities:")
    print(f"   - Mode: {caps.get('current_mode', 'unknown')}")
    print(f"   - LangChain Available: {caps.get('langchain_available', False)}")
    print(f"   - Has Tools: {caps.get('has_tools', False)}")
    
    # Cleanup
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()

if __name__ == "__main__":
    print("🤖 JARVIS Math Test - Forcing LangChain Mode")
    asyncio.run(test_math_with_langchain())