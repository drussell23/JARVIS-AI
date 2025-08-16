#!/usr/bin/env python3
"""
Test LangChain loading with proper memory management
"""

import asyncio
import psutil
import logging
from pathlib import Path
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_langchain_loading():
    """Test if LangChain can be loaded"""
    
    print("🧪 Testing LangChain Loading")
    print("=" * 50)
    
    # Check initial memory
    mem = psutil.virtual_memory()
    print(f"Initial memory: {mem.percent:.1f}%")
    
    if mem.percent > 70:
        print("⚠️  Memory too high for LangChain. Need to free memory first.")
        print("🔧 Run: ./emergency_memory_cleanup.sh")
        return False
    
    try:
        # Import components
        print("\n1. Importing components...")
        from memory.memory_manager import M1MemoryManager
        from chatbots.dynamic_chatbot import DynamicChatbot
        print("   ✓ Imports successful")
        
        # Create memory manager
        print("\n2. Creating memory manager...")
        memory_manager = M1MemoryManager()
        print("   ✓ Memory manager created")
        
        # Register components
        print("\n3. Registering components...")
        await memory_manager.register_intelligent_components()
        print("   ✓ Components registered")
        
        # Start monitoring
        print("\n4. Starting memory monitoring...")
        await memory_manager.start_monitoring()
        print("   ✓ Monitoring started")
        
        # Create chatbot
        print("\n5. Creating dynamic chatbot...")
        chatbot = DynamicChatbot(
            memory_manager=memory_manager,
            auto_switch=False,  # Manual control for testing
            prefer_langchain=True
        )
        print("   ✓ Chatbot created")
        print(f"   - Current mode: {chatbot.current_mode.value}")
        print(f"   - Loaded components: {chatbot._loaded_components}")
        
        # Test simple mode
        print("\n6. Testing simple mode...")
        response = await chatbot.generate_response("Hello")
        print(f"   ✓ Response: {response[:50]}...")
        print(f"   - Memory: {psutil.virtual_memory().percent:.1f}%")
        print(f"   - Components: {chatbot._loaded_components}")
        
        # Check if we can upgrade
        print("\n7. Checking upgrade possibilities...")
        current_mem = psutil.virtual_memory().percent
        print(f"   Current memory: {current_mem:.1f}%")
        
        can_upgrade = await chatbot._can_upgrade()
        print(f"   Can upgrade: {can_upgrade}")
        
        if current_mem < 65:
            print("   ✅ Memory suitable for Intelligent mode")
            
            # Try to upgrade to Intelligent
            print("\n8. Testing upgrade to Intelligent mode...")
            try:
                await chatbot.force_mode("intelligent")
                print("   ✓ Successfully upgraded to Intelligent mode")
                print(f"   - Memory: {psutil.virtual_memory().percent:.1f}%")
                print(f"   - Components: {chatbot._loaded_components}")
                
                # Test intelligent response
                response = await chatbot.generate_response("What is artificial intelligence?")
                print(f"   ✓ Response: {response[:50]}...")
                
            except Exception as e:
                print(f"   ❌ Failed to upgrade: {e}")
            
            if current_mem < 50:
                print("\n   ✅ Memory suitable for LangChain mode")
                
                # Try to upgrade to LangChain
                print("\n9. Testing upgrade to LangChain mode...")
                try:
                    await chatbot.force_mode("langchain")
                    print("   ✓ Successfully upgraded to LangChain mode!")
                    print(f"   - Memory: {psutil.virtual_memory().percent:.1f}%")
                    print(f"   - Components: {chatbot._loaded_components}")
                    
                    # Test LangChain response
                    response = await chatbot.generate_response("Tell me about quantum computing")
                    print(f"   ✓ Response: {response[:50]}...")
                    
                except Exception as e:
                    print(f"   ❌ Failed to upgrade to LangChain: {e}")
            else:
                print("   ⚠️  Need <50% memory for LangChain")
        else:
            print("   ❌ Memory too high for upgrades")
        
        # Show capabilities
        print("\n10. Current capabilities:")
        caps = chatbot.get_capabilities()
        print(f"   - Mode: {caps['current_mode']}")
        print(f"   - LangChain available: {caps['langchain_available']}")
        print(f"   - Mode switches: {caps['mode_switches']}")
        
        # Cleanup
        print("\n11. Cleaning up...")
        await chatbot.cleanup()
        await memory_manager.stop_monitoring()
        print("   ✓ Cleanup complete")
        
        # Final memory
        final_mem = psutil.virtual_memory().percent
        print(f"\nFinal memory: {final_mem:.1f}%")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_memory_progression():
    """Test memory usage as we progress through modes"""
    print("\n\n🔄 Testing Memory Progression Through Modes")
    print("=" * 50)
    
    memory_readings = []
    
    try:
        from memory.memory_manager import M1MemoryManager
        from chatbots.dynamic_chatbot import DynamicChatbot
        
        # Initial reading
        mem_start = psutil.virtual_memory().percent
        memory_readings.append(("Start", mem_start))
        
        # Create components
        memory_manager = M1MemoryManager()
        await memory_manager.register_intelligent_components()
        await memory_manager.start_monitoring()
        
        chatbot = DynamicChatbot(
            memory_manager=memory_manager,
            auto_switch=False,
            prefer_langchain=True
        )
        
        # Test each mode
        modes = ["simple", "intelligent", "langchain"]
        for mode in modes:
            print(f"\nTesting {mode} mode...")
            try:
                if mode != "simple":  # Already in simple mode
                    await chatbot.force_mode(mode)
                
                # Generate a response
                response = await chatbot.generate_response(f"Test in {mode} mode")
                
                # Record memory
                mem = psutil.virtual_memory().percent
                memory_readings.append((mode.capitalize(), mem))
                print(f"   Memory in {mode} mode: {mem:.1f}%")
                
            except Exception as e:
                print(f"   Failed to test {mode} mode: {e}")
                memory_readings.append((mode.capitalize(), psutil.virtual_memory().percent))
        
        # Cleanup
        await chatbot.cleanup()
        await memory_manager.stop_monitoring()
        
        # Show progression
        print("\n📊 Memory Progression:")
        prev_mem = mem_start
        for stage, mem in memory_readings:
            delta = mem - prev_mem
            print(f"   {stage:12} : {mem:5.1f}% ({delta:+.1f}%)")
            prev_mem = mem
        
    except Exception as e:
        print(f"Progression test failed: {e}")

if __name__ == "__main__":
    print("🚀 JARVIS LangChain Loading Test\n")
    
    # Check Python path
    print("Python path:")
    for p in sys.path:
        print(f"  {p}")
    print()
    
    # Run tests
    success = asyncio.run(test_langchain_loading())
    
    if success:
        # Run progression test if first test passed
        asyncio.run(test_memory_progression())
    
    print("\n💡 Tips:")
    print("1. If memory is too high, run: ./emergency_memory_cleanup.sh")
    print("2. Close unnecessary applications")
    print("3. Consider restarting your Mac if needed")
    print("4. Run with DEBUG logging: PYTHONPATH=. python -u test_langchain_loading.py")