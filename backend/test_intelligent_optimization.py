#!/usr/bin/env python3
"""
Test the intelligent memory optimization system
"""
import asyncio
import psutil
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
from memory.memory_manager import M1MemoryManager
from chatbots.dynamic_chatbot import DynamicChatbot


async def test_intelligent_optimization():
    """Test the intelligent memory optimization"""
    print("=" * 60)
    print("Testing Intelligent Memory Optimization for JARVIS")
    print("=" * 60)
    
    # Create optimizer
    optimizer = IntelligentMemoryOptimizer()
    
    # Get initial memory status
    mem = psutil.virtual_memory()
    print(f"\nInitial Memory Status:")
    print(f"- Total: {mem.total / (1024**3):.1f} GB")
    print(f"- Used: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
    print(f"- Available: {mem.available / (1024**3):.1f} GB")
    
    # Get optimization suggestions
    print("\nGetting optimization suggestions...")
    suggestions = await optimizer.get_optimization_suggestions()
    if suggestions:
        print("\nSuggestions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
    else:
        print("No specific suggestions at this time.")
    
    # Check if optimization is needed
    if mem.percent <= 50:
        print(f"\n✅ Memory is already below 50% ({mem.percent}%) - LangChain ready!")
        return
    
    print(f"\n⚠️  Memory is at {mem.percent}% - need to optimize for LangChain (target: < 50%)")
    
    # Ask for confirmation
    print("\nWARNING: This will attempt to free memory by:")
    print("- Running garbage collection")
    print("- Clearing system caches (may require sudo)")
    print("- Killing helper processes")
    print("- Suspending background apps")
    print("- Optimizing browser memory")
    
    response = input("\nProceed with optimization? (y/n): ")
    if response.lower() != 'y':
        print("Optimization cancelled.")
        return
    
    # Run optimization
    print("\nStarting intelligent memory optimization...")
    success, report = await optimizer.optimize_for_langchain()
    
    # Display results
    print("\n" + "=" * 60)
    print("Optimization Report")
    print("=" * 60)
    print(f"Success: {'✅ Yes' if success else '❌ No'}")
    print(f"Initial Memory: {report['initial_percent']:.1f}%")
    print(f"Final Memory: {report['final_percent']:.1f}%")
    print(f"Memory Freed: {report['memory_freed_mb']:.0f} MB")
    
    if report['actions_taken']:
        print("\nActions Taken:")
        for action in report['actions_taken']:
            print(f"- {action['strategy']}: freed {action['freed_mb']:.0f} MB")
    
    print("\n" + "=" * 60)
    
    if success:
        print("🎉 Memory optimization successful! JARVIS can now use LangChain features.")
    else:
        print("⚠️  Could not free enough memory. Try closing more applications manually.")


async def test_dynamic_chatbot_with_optimization():
    """Test the DynamicChatbot with intelligent optimization"""
    print("\n" + "=" * 60)
    print("Testing DynamicChatbot with Intelligent Optimization")
    print("=" * 60)
    
    # Create memory manager
    memory_manager = M1MemoryManager()
    await memory_manager.start_monitoring()
    
    # Create dynamic chatbot
    chatbot = DynamicChatbot(
        memory_manager,
        auto_switch=True,
        prefer_langchain=True
    )
    
    print("\nTesting JARVIS with automatic memory optimization...")
    
    # Test queries
    test_queries = [
        "What is 2 + 2?",
        "Calculate 15 * 23 + 47",
        "What's the weather like in San Francisco?"
    ]
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = await chatbot.generate_response(query)
        print(f"JARVIS [{chatbot.current_mode.value}]: {response}")
        
        # Show optimization metrics
        if chatbot.metrics["intelligent_optimizations"] > 0:
            print(f"  (Optimizations attempted: {chatbot.metrics['intelligent_optimizations']}, "
                  f"Successful: {chatbot.metrics['optimization_successes']})")
        
        await asyncio.sleep(2)
    
    # Show final capabilities
    caps = chatbot.get_capabilities()
    print(f"\nFinal Mode: {caps['current_mode']}")
    print(f"Mode Switches: {caps['mode_switches']}")
    print(f"Intelligent Optimizations: {caps['intelligent_optimizations']}")
    print(f"Optimization Successes: {caps['optimization_successes']}")
    
    # Cleanup
    await chatbot.cleanup()
    await memory_manager.stop_monitoring()


if __name__ == "__main__":
    print("Choose test mode:")
    print("1. Test intelligent memory optimization only")
    print("2. Test DynamicChatbot with automatic optimization")
    
    choice = input("\nEnter choice (1 or 2): ")
    
    if choice == "1":
        asyncio.run(test_intelligent_optimization())
    elif choice == "2":
        asyncio.run(test_dynamic_chatbot_with_optimization())
    else:
        print("Invalid choice.")