#!/usr/bin/env python3
"""
Test memory efficiency of JARVIS components
"""

import psutil
import gc
import time
import asyncio
from pathlib import Path

def test_memory_loading():
    """Test how much memory each component uses"""
    
    print("🧪 Testing JARVIS Memory Efficiency")
    print("=" * 50)
    
    # Baseline memory
    baseline = psutil.virtual_memory().percent
    print(f"Baseline memory: {baseline:.1f}%")
    
    # Test 1: Import core modules
    print("\n1. Importing core modules...")
    start_mem = psutil.virtual_memory().percent
    
    try:
        from core.jarvis_core import JARVISCore
        from memory.memory_manager import M1MemoryManager
        from chatbots.dynamic_chatbot import DynamicChatbot
        
        end_mem = psutil.virtual_memory().percent
        print(f"   Memory change: {start_mem:.1f}% → {end_mem:.1f}% (+{end_mem-start_mem:.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return
    
    # Test 2: Create memory manager
    print("\n2. Creating memory manager...")
    start_mem = psutil.virtual_memory().percent
    
    try:
        memory_manager = M1MemoryManager()
        end_mem = psutil.virtual_memory().percent
        print(f"   Memory change: {start_mem:.1f}% → {end_mem:.1f}% (+{end_mem-start_mem:.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Memory manager creation failed: {e}")
        return
    
    # Test 3: Create dynamic chatbot
    print("\n3. Creating dynamic chatbot...")
    start_mem = psutil.virtual_memory().percent
    
    try:
        chatbot = DynamicChatbot(memory_manager=memory_manager)
        end_mem = psutil.virtual_memory().percent
        print(f"   Memory change: {start_mem:.1f}% → {end_mem:.1f}% (+{end_mem-start_mem:.1f}%)")
        
    except Exception as e:
        print(f"   ❌ Chatbot creation failed: {e}")
        return
    
    # Test 4: Check lazy loading
    print("\n4. Checking lazy loading...")
    print(f"   Loaded components: {chatbot._loaded_components}")
    print(f"   Current bot exists: {chatbot._current_bot is not None}")
    print(f"   Memory optimizer exists: {chatbot._memory_optimizer is not None}")
    
    # Test 5: Trigger first response (should lazy load)
    print("\n5. Testing first response (triggers lazy loading)...")
    start_mem = psutil.virtual_memory().percent
    
    async def test_response():
        response = await chatbot.generate_response("Hello")
        return response
    
    try:
        response = asyncio.run(test_response())
        end_mem = psutil.virtual_memory().percent
        print(f"   Memory change: {start_mem:.1f}% → {end_mem:.1f}% (+{end_mem-start_mem:.1f}%)")
        print(f"   Response: {response[:50]}...")
        print(f"   Loaded components: {chatbot._loaded_components}")
        
    except Exception as e:
        print(f"   ❌ Response generation failed: {e}")
    
    # Test 6: Force garbage collection
    print("\n6. Testing garbage collection...")
    start_mem = psutil.virtual_memory().percent
    
    # Delete objects
    del chatbot
    del memory_manager
    gc.collect()
    
    end_mem = psutil.virtual_memory().percent
    print(f"   Memory change: {start_mem:.1f}% → {end_mem:.1f}% ({end_mem-start_mem:+.1f}%)")
    
    # Final status
    final_mem = psutil.virtual_memory().percent
    print(f"\n📊 Final Results:")
    print(f"   Baseline: {baseline:.1f}%")
    print(f"   Final: {final_mem:.1f}%")
    print(f"   Net change: {final_mem-baseline:+.1f}%")
    
    if final_mem - baseline < 5:
        print("   ✅ Excellent memory efficiency!")
    elif final_mem - baseline < 10:
        print("   ⚠️  Good memory efficiency")
    else:
        print("   ❌ Poor memory efficiency - needs optimization")

def test_memory_progression():
    """Test memory usage as components are loaded"""
    print("\n\n🔄 Testing Memory Progression")
    print("=" * 50)
    
    async def test_progression():
        from memory.memory_manager import M1MemoryManager
        from chatbots.dynamic_chatbot import DynamicChatbot
        
        # Start monitoring
        memory_manager = M1MemoryManager()
        await memory_manager.start_monitoring()
        
        # Create chatbot
        chatbot = DynamicChatbot(
            memory_manager,
            auto_switch=False  # Manual control
        )
        
        print(f"Initial memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"Loaded components: {chatbot._loaded_components}")
        
        # Test simple mode
        print("\n1. Simple mode response:")
        response = await chatbot.generate_response("Hello")
        print(f"   Memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Components: {chatbot._loaded_components}")
        
        # Force upgrade to intelligent
        print("\n2. Upgrading to Intelligent mode:")
        await chatbot.force_mode("intelligent")
        print(f"   Memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   Components: {chatbot._loaded_components}")
        
        # Test intelligent mode
        response = await chatbot.generate_response("What is AI?")
        print(f"   Memory after response: {psutil.virtual_memory().percent:.1f}%")
        
        # Try upgrade to langchain
        print("\n3. Attempting LangChain upgrade:")
        try:
            await chatbot.force_mode("langchain")
            print(f"   Memory: {psutil.virtual_memory().percent:.1f}%")
            print(f"   Components: {chatbot._loaded_components}")
        except Exception as e:
            print(f"   Failed: {e}")
        
        # Cleanup
        await chatbot.cleanup()
        await memory_manager.stop_monitoring()
        print(f"\nFinal memory after cleanup: {psutil.virtual_memory().percent:.1f}%")
    
    asyncio.run(test_progression())

if __name__ == "__main__":
    print("Starting JARVIS Memory Efficiency Tests\n")
    
    # Test 1: Basic memory loading
    test_memory_loading()
    
    # Wait a bit
    time.sleep(2)
    
    # Test 2: Progressive loading
    test_memory_progression()
    
    print("\n✅ Tests complete!")