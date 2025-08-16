#!/usr/bin/env python3
"""
Demo of JARVIS Core architecture (works without models)
"""

import asyncio
import sys
sys.path.append('backend')

from core import JARVISCore, ModelManager, MemoryController, TaskRouter
from core.model_manager import ModelTier
import psutil


async def demo_architecture():
    """Demonstrate the JARVIS Core architecture components"""
    print("🏗️  JARVIS Core Architecture Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1️⃣ Initializing Components...")
    model_manager = ModelManager()
    memory_controller = MemoryController()
    task_router = TaskRouter(model_manager, memory_controller)
    
    await memory_controller.start_monitoring()
    await asyncio.sleep(1)
    
    # Show memory status
    print("\n2️⃣ Memory Controller Demo")
    print("-" * 30)
    mem_stats = memory_controller.get_memory_stats()
    print(f"Current Memory Usage: {mem_stats['current']['percent_used']:.1f}%")
    print(f"Memory Pressure: {mem_stats['current']['pressure']}")
    print(f"Available: {mem_stats['current']['available_mb']:.0f}MB")
    print("\nTop Processes:")
    for proc in mem_stats['current']['top_processes'][:3]:
        print(f"  - {proc['name']}: {proc['memory_percent']:.1f}%")
    
    # Demonstrate task analysis
    print("\n3️⃣ Task Router Demo")
    print("-" * 30)
    
    test_queries = [
        "Hello, how are you?",
        "Write a Python function to sort a list",
        "Analyze the implications of quantum computing on cryptography"
    ]
    
    for query in test_queries:
        analysis = task_router.analyze_task(query)
        print(f"\nQuery: '{query}'")
        print(f"  Type: {analysis.task_type.value}")
        print(f"  Complexity: {analysis.complexity:.2f}")
        print(f"  Estimated tokens: {analysis.estimated_tokens}")
        print(f"  Reasoning: {analysis.reasoning}")
    
    # Show model manager capabilities
    print("\n4️⃣ Model Manager Demo")
    print("-" * 30)
    print("Model Tiers:")
    for tier, info in model_manager.models.items():
        print(f"\n{tier.value.upper()} - {info.name}:")
        print(f"  Size: {info.size_gb}GB")
        print(f"  Context: {info.context_size} tokens")
        print(f"  Capabilities: {', '.join(f'{k}={v:.1f}' for k, v in info.capabilities.items())}")
    
    # Demonstrate memory pressure handling
    print("\n5️⃣ Memory Pressure Simulation")
    print("-" * 30)
    
    # Check what would happen at different memory levels
    memory_scenarios = [
        ("Low pressure (30%)", 30),
        ("Moderate pressure (55%)", 55),
        ("High pressure (78%)", 78),
        ("Critical pressure (88%)", 88)
    ]
    
    for scenario, percent in memory_scenarios:
        pressure = memory_controller._calculate_pressure(percent)
        can_load, reason = memory_controller.should_load_model(2048)  # 2GB model
        print(f"\n{scenario}:")
        print(f"  Pressure level: {pressure.value}")
        print(f"  Can load 2GB model: {'Yes' if can_load else 'No'} - {reason}")
    
    # Show routing statistics
    print("\n6️⃣ Routing Statistics")
    print("-" * 30)
    routing_stats = task_router.get_routing_stats()
    print(f"Total requests: {routing_stats['total_requests']}")
    print(f"Average routing time: {routing_stats['average_routing_time']:.3f}s")
    
    # Demonstrate optimization suggestions
    print("\n7️⃣ System Optimization")
    print("-" * 30)
    
    # Memory optimization
    if mem_stats['current']['percent_used'] > 70:
        print("Memory optimization needed!")
        result = await memory_controller.optimize_memory()
        print(f"  Freed: {result['freed_mb']:.0f}MB")
        print(f"  Actions: {', '.join(result['actions'])}")
    else:
        print("Memory usage is optimal")
    
    # Stop monitoring
    await memory_controller.stop_monitoring()
    
    print("\n✅ Demo complete!")
    print("\n💡 To use with actual models:")
    print("   1. Run: python download_jarvis_models.py")
    print("   2. Run: python test_jarvis_core.py")


async def show_live_monitoring():
    """Show live memory monitoring"""
    print("\n📊 Live Memory Monitoring (10 seconds)")
    print("=" * 50)
    
    memory_controller = MemoryController(monitoring_interval=1.0)
    await memory_controller.start_monitoring()
    
    for i in range(10):
        await asyncio.sleep(1)
        stats = memory_controller.get_memory_stats()
        current = stats['current']
        print(f"\r[{i+1:2d}s] Memory: {current['percent_used']:5.1f}% | "
              f"Pressure: {current['pressure']:8s} | "
              f"Available: {current['available_mb']:6.0f}MB", end='', flush=True)
    
    print("\n")
    await memory_controller.stop_monitoring()


async def main():
    """Main entry point"""
    print("🤖 JARVIS Core Architecture Demo (No Models Required)")
    print("=" * 50)
    
    # Run architecture demo
    await demo_architecture()
    
    # Show live monitoring
    await show_live_monitoring()
    
    print("\n🎯 This demo shows the architecture working without models.")
    print("   To see it with actual language models, download them first:")
    print("   python download_jarvis_models.py")


if __name__ == "__main__":
    asyncio.run(main())