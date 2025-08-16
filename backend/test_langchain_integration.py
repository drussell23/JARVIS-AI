#!/usr/bin/env python3
"""
Test LangChain Integration with JARVIS
Demonstrates the power of LangChain tools with memory management
"""

import asyncio
import logging
from memory.memory_manager import M1MemoryManager, ComponentPriority
from chatbots.dynamic_chatbot import DynamicChatbot

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_langchain_features():
    """Test LangChain-powered features"""
    
    print("🤖 JARVIS LangChain Integration Test")
    print("=" * 60)
    
    # Create memory manager
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Register required components
    memory_manager.register_component("intelligent_chatbot", ComponentPriority.HIGH, 1500)
    memory_manager.register_component("langchain_chatbot", ComponentPriority.HIGH, 4000)
    
    # Create dynamic chatbot with LangChain preference
    chatbot = DynamicChatbot(
        memory_manager=memory_manager,
        auto_switch=True,
        preserve_context=True,
        prefer_langchain=True
    )
    
    print("\n✅ JARVIS initialized with LangChain support")
    print("⏳ Waiting for components to load...\n")
    
    # Wait for initialization
    await asyncio.sleep(10)
    
    # Test queries that showcase LangChain capabilities
    test_scenarios = [
        {
            "category": "🧮 Mathematical Calculations",
            "queries": [
                "What is 2 + 2?",
                "Calculate 15 * 23 + 47",
                "What's the square root of 144?",
                "Calculate (100 * 5) / 25 + 10"
            ]
        },
        {
            "category": "🔍 Web Search & Current Information",
            "queries": [
                "Search for the latest AI news",
                "What's the current weather in San Francisco?",
                "Find information about quantum computing advances"
            ]
        },
        {
            "category": "📚 Knowledge & Information",
            "queries": [
                "Tell me about the history of artificial intelligence",
                "What is machine learning?",
                "Explain quantum entanglement in simple terms"
            ]
        },
        {
            "category": "💻 System Awareness",
            "queries": [
                "What's your current status?",
                "How much memory are you using?",
                "What tools do you have available?"
            ]
        },
        {
            "category": "🎯 Complex Reasoning",
            "queries": [
                "If I have 5 apples and give away 2, then buy 7 more, how many do I have?",
                "Compare the advantages of Python vs JavaScript for web development",
                "What would happen if we could travel faster than light?"
            ]
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['category']}")
        print("-" * 40)
        
        for query in scenario['queries']:
            print(f"\n👤 User: {query}")
            
            # Get response
            response = await chatbot.generate_response(query)
            print(f"🤖 JARVIS [{chatbot.current_mode.value}]: {response}")
            
            # Small delay between queries
            await asyncio.sleep(2)
    
    # Final report
    print("\n" + "=" * 60)
    print("📊 Final Report:")
    
    # Get capabilities
    capabilities = chatbot.get_capabilities()
    
    print(f"\nMode Statistics:")
    print(f"- Current Mode: {chatbot.current_mode.value}")
    print(f"- Mode Switches: {capabilities.get('mode_switches', 0)}")
    print(f"- Simple Responses: {capabilities.get('simple_responses', 0)}")
    print(f"- Intelligent Responses: {capabilities.get('intelligent_responses', 0)}")
    print(f"- LangChain Responses: {capabilities.get('langchain_responses', 0)}")
    
    if 'langchain' in capabilities:
        langchain_info = capabilities['langchain']
        print(f"\nLangChain Status:")
        print(f"- Enabled: {langchain_info.get('enabled', False)}")
        print(f"- Available Tools: {', '.join(langchain_info.get('tools', []))}")
        print(f"- LLM Type: {langchain_info.get('llm_type', 'Unknown')}")
    
    # Memory report
    memory_report = await memory_manager.get_memory_report()
    print(f"\nMemory Status:")
    print(f"- Current Usage: {memory_report['current_state']['percent_used']:.1f}%")
    print(f"- State: {memory_report['current_state']['state']}")
    
    # Cleanup
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()
    
    print("\n✅ Test completed successfully!")

async def test_mode_switching():
    """Test manual mode switching"""
    
    print("\n🔄 Testing Manual Mode Switching")
    print("=" * 60)
    
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Register required components for mode switching
    memory_manager.register_component("intelligent_chatbot", ComponentPriority.HIGH, 1500)
    memory_manager.register_component("langchain_chatbot", ComponentPriority.HIGH, 4000)
    
    chatbot = DynamicChatbot(
        memory_manager=memory_manager,
        auto_switch=False,  # Disable auto-switch for manual testing
        preserve_context=True,
        prefer_langchain=True
    )
    
    modes = ["simple", "intelligent", "langchain"]
    test_query = "Calculate 50 + 50 and tell me a fun fact about the number 100"
    
    for mode in modes:
        print(f"\n🔧 Forcing mode: {mode}")
        await chatbot.force_mode(mode)
        await asyncio.sleep(5)  # Wait for mode switch
        
        print(f"👤 User: {test_query}")
        response = await chatbot.generate_response(test_query)
        print(f"🤖 JARVIS [{mode}]: {response}")
        
        # Check capabilities
        caps = chatbot.get_capabilities()
        if mode == "langchain" and 'langchain' in caps:
            print(f"   Tools: {', '.join(caps['langchain'].get('tools', []))}")
    
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()

if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Full LangChain feature test")
    print("2. Mode switching test")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        asyncio.run(test_mode_switching())
    else:
        asyncio.run(test_langchain_features())