#!/usr/bin/env python3
"""
Quick test for LangChain integration
"""
import asyncio
import logging
from memory.memory_manager import M1MemoryManager, ComponentPriority
from chatbots.langchain_chatbot import LangChainChatbot, LANGCHAIN_AVAILABLE

logging.basicConfig(level=logging.INFO)

async def test_langchain():
    """Test LangChain directly"""
    print(f"LangChain Available: {LANGCHAIN_AVAILABLE}")
    
    if not LANGCHAIN_AVAILABLE:
        print("LangChain is not available. Check imports.")
        return
    
    # Create memory manager
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Create LangChain chatbot directly
    print("Creating LangChain chatbot...")
    chatbot = LangChainChatbot(
        memory_manager=memory_manager,
        use_local_models=True
    )
    
    # Wait for initialization
    print("Waiting for initialization...")
    await asyncio.sleep(5)
    
    # Test queries
    test_queries = [
        "Hello!",
        "What is 2 + 2?",
        "Calculate 15 * 23",
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        try:
            response = await chatbot.generate_response(query)
            print(f"JARVIS: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Check capabilities
    caps = chatbot.get_capabilities()
    print(f"\nCapabilities: {caps.get('langchain', {})}")
    
    # Cleanup
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(test_langchain())