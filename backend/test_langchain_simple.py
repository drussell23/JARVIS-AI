#!/usr/bin/env python3
"""
Simple test to verify LangChain integration is working
"""
import asyncio
from memory.memory_manager import M1MemoryManager, ComponentPriority
from chatbots.dynamic_chatbot import DynamicChatbot
from chatbots.langchain_chatbot import LANGCHAIN_AVAILABLE

async def test_langchain_simple():
    """Simple test of LangChain integration"""
    
    print("=" * 60)
    print("JARVIS LangChain Integration Test")
    print("=" * 60)
    
    # Check if LangChain is available
    print(f"\n✓ LangChain Available: {LANGCHAIN_AVAILABLE}")
    if not LANGCHAIN_AVAILABLE:
        print("✗ LangChain is not installed properly")
        return
    
    # Create memory manager
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Get memory status
    memory_snapshot = await memory_manager.get_memory_snapshot()
    print(f"✓ Memory Usage: {memory_snapshot.percent * 100:.1f}%")
    print(f"✓ Memory State: {memory_snapshot.state.value}")
    
    # Register components
    memory_manager.register_component("intelligent_chatbot", ComponentPriority.HIGH, 1500)
    memory_manager.register_component("langchain_chatbot", ComponentPriority.HIGH, 4000)
    
    # Create dynamic chatbot
    print("\n✓ Creating Dynamic Chatbot with LangChain support...")
    chatbot = DynamicChatbot(
        memory_manager=memory_manager,
        auto_switch=False,  # Manual control for testing
        preserve_context=True,
        prefer_langchain=True
    )
    
    # Check initial mode
    print(f"✓ Initial Mode: {chatbot.current_mode.value}")
    
    # Get capabilities
    capabilities = chatbot.get_capabilities()
    print(f"✓ Dynamic Mode: {capabilities.get('dynamic_mode', False)}")
    print(f"✓ LangChain Available in Capabilities: {capabilities.get('langchain_available', False)}")
    
    # Test simple response without LangChain
    print("\n--- Testing Simple Mode ---")
    response = await chatbot.generate_response("Hello JARVIS!")
    print(f"User: Hello JARVIS!")
    print(f"JARVIS [{chatbot.current_mode.value}]: {response}")
    
    # Try to switch to LangChain mode
    print("\n--- Attempting to switch to LangChain Mode ---")
    try:
        await chatbot.force_mode("langchain")
        print(f"✓ Successfully switched to: {chatbot.current_mode.value}")
    except Exception as e:
        print(f"✗ Failed to switch to LangChain: {e}")
    
    # Test math with calculator tool (if available)
    print("\n--- Testing Math Calculation ---")
    response = await chatbot.generate_response("What is 2 + 2?")
    print(f"User: What is 2 + 2?")
    print(f"JARVIS [{chatbot.current_mode.value}]: {response}")
    
    # Final capabilities check
    final_caps = chatbot.get_capabilities()
    print("\n--- Final Capabilities ---")
    print(f"✓ Current Mode: {chatbot.current_mode.value}")
    print(f"✓ Mode Switches: {final_caps.get('mode_switches', 0)}")
    if 'langchain' in final_caps:
        langchain_info = final_caps['langchain']
        print(f"✓ LangChain Enabled: {langchain_info.get('enabled', False)}")
        print(f"✓ LangChain Tools: {langchain_info.get('tools', [])}")
    
    # Cleanup
    print("\n✓ Cleaning up...")
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()
    
    print("\n✓ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_langchain_simple())