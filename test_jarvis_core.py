#!/usr/bin/env python3
"""
Test script for the new JARVIS Core architecture
"""

import asyncio
import sys
sys.path.append('backend')

from core import JARVISAssistant


async def test_jarvis_core():
    """Test the JARVIS Core system"""
    print("🚀 JARVIS Core Test")
    print("=" * 50)
    
    # Initialize JARVIS
    print("Initializing JARVIS Core...")
    jarvis = JARVISAssistant()
    
    # Wait for initialization
    await asyncio.sleep(2)
    
    # Test queries of different complexities
    test_queries = [
        {
            "query": "Hello, how are you?",
            "expected_tier": "tiny",
            "description": "Simple greeting"
        },
        {
            "query": "Write a Python function to calculate fibonacci numbers",
            "expected_tier": "std",
            "description": "Code generation"
        },
        {
            "query": "Analyze the complexity of different sorting algorithms and compare their performance characteristics in detail",
            "expected_tier": "adv",
            "description": "Complex analysis"
        }
    ]
    
    print("\n📝 Running test queries...")
    for i, test in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: {test['description']} ---")
        print(f"Query: {test['query']}")
        
        # Get response with metadata
        response = await jarvis.chat_with_info(test['query'])
        
        if response["success"]:
            print(f"✅ Response: {response['response'][:100]}...")
            print(f"📊 Metadata:")
            print(f"   - Model Tier: {response['metadata']['model_tier']}")
            print(f"   - Task Type: {response['metadata']['task_analysis']['type']}")
            print(f"   - Complexity: {response['metadata']['task_analysis']['complexity']:.2f}")
            print(f"   - Memory State: {response['metadata']['memory_state']}")
            print(f"   - Processing Time: {response['metadata']['processing_time']:.2f}s")
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
    
    # Get system status
    print("\n📊 System Status:")
    status = jarvis.get_status()
    
    print(f"\n🔧 Core Stats:")
    print(f"   - Total Queries: {status['core']['total_queries']}")
    print(f"   - Model Switches: {status['core']['model_switches']}")
    print(f"   - Memory Optimizations: {status['core']['memory_optimizations']}")
    
    print(f"\n💾 Memory Status:")
    memory = status['memory']['current']
    print(f"   - Usage: {memory['percent_used']:.1f}%")
    print(f"   - Available: {memory['available_mb']:.0f}MB")
    print(f"   - Pressure: {memory['pressure']}")
    
    print(f"\n🤖 Loaded Models:")
    for model in status['models']['loaded_models']:
        print(f"   - {model['name']} ({model['tier']}): {model['size_gb']}GB")
    
    print("\n✅ Test complete!")
    

async def interactive_demo():
    """Interactive demo of JARVIS Core"""
    print("\n🤖 JARVIS Core Interactive Demo")
    print("=" * 50)
    print("Type 'quit' to exit, 'status' for system status, 'optimize' to optimize")
    print("=" * 50)
    
    jarvis = JARVISAssistant()
    await asyncio.sleep(2)
    
    while True:
        try:
            query = input("\n💬 You: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'status':
                status = jarvis.get_status()
                print(f"\n📊 System Status:")
                print(f"   Models loaded: {status['models']['loaded_count']}")
                print(f"   Memory usage: {status['memory']['current']['percent_used']:.1f}%")
                print(f"   Total queries: {status['core']['total_queries']}")
                continue
            elif query.lower() == 'optimize':
                print("🔧 Running optimization...")
                results = await jarvis.optimize()
                print(f"✅ Optimization complete: {results}")
                continue
            
            # Process query
            response = await jarvis.chat_with_info(query)
            
            print(f"\n🤖 JARVIS ({response['metadata']['model_tier']}): {response['response']}")
            print(f"   ⚡ {response['metadata']['processing_time']:.2f}s | "
                  f"📊 {response['metadata']['task_analysis']['type']} | "
                  f"💾 {response['metadata']['memory_state']}")
                  
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


async def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        await interactive_demo()
    else:
        await test_jarvis_core()


if __name__ == "__main__":
    asyncio.run(main())