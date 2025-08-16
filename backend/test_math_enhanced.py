#!/usr/bin/env python3
"""
Test enhanced mathematical capabilities of JARVIS
"""
import asyncio
from memory.memory_manager import M1MemoryManager, ComponentPriority
from chatbots.langchain_chatbot import LangChainChatbot, CalculatorTool

async def test_calculator_tool():
    """Test the calculator tool directly"""
    print("Testing Calculator Tool")
    print("=" * 40)
    
    calc = CalculatorTool()
    
    test_cases = [
        "2+2",
        "what is 2+2",
        "15 * 23 + 47",
        "100 divided by 4",
        "5 plus 5 times 2",
        "10 squared",
        "2 to the power of 8",
        "What's 50 minus 30?",
        "calculate 1000 / 25"
    ]
    
    for query in test_cases:
        result = calc._run(query)
        print(f"Query: {query}")
        print(f"Result: {result}")
        print("-" * 40)

async def test_jarvis_math():
    """Test JARVIS with mathematical questions"""
    print("\nTesting JARVIS with Math Questions")
    print("=" * 40)
    
    # Initialize
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Register components
    memory_manager.register_component("langchain_chatbot", ComponentPriority.HIGH, 4000)
    
    # Create chatbot
    chatbot = LangChainChatbot(memory_manager)
    
    # Wait for initialization
    print("Initializing JARVIS...")
    await asyncio.sleep(5)
    
    # Test queries
    test_queries = [
        "What is 2+2?",
        "Calculate 15 * 23 + 47",
        "How much is 100 divided by 4?",
        "What's 5 plus 5 times 2?",
        "Compute the square root of 144",
        "What is 2 to the power of 8?",
        "Can you calculate 50% of 200?",
        "What's the sum of 1, 2, 3, 4, and 5?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        try:
            response = await chatbot.generate_response(query)
            print(f"JARVIS: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Cleanup
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()

if __name__ == "__main__":
    print("JARVIS Enhanced Math Test\n")
    
    # Test calculator directly
    asyncio.run(test_calculator_tool())
    
    # Test full JARVIS
    # asyncio.run(test_jarvis_math())
    
    print("\nTo test full JARVIS math capabilities, uncomment the test_jarvis_math() line")