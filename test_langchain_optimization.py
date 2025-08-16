#!/usr/bin/env python3
"""
Test the enhanced memory optimization for enabling LangChain
Shows how JARVIS can intelligently close high-memory apps to enable advanced features
"""

import asyncio
import psutil
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
from memory.optimization_config import optimization_config


def show_memory_status():
    """Display current memory status"""
    mem = psutil.virtual_memory()
    print(f"\n{'='*60}")
    print(f"Current Memory Status")
    print(f"{'='*60}")
    print(f"Total: {mem.total / (1024**3):.1f} GB")
    print(f"Used: {mem.used / (1024**3):.1f} GB ({mem.percent:.1f}%)")
    print(f"Available: {mem.available / (1024**3):.1f} GB")
    
    if mem.percent < 50:
        print("✅ Memory OK for LangChain mode")
    elif mem.percent < 65:
        print("⚠️  Memory OK for Intelligent mode only")
    else:
        print("❌ Memory too high - limited features only")
    
    return mem.percent


def show_target_apps():
    """Show which apps would be targeted for closure"""
    print(f"\n{'='*60}")
    print(f"Target Applications for LangChain Optimization")
    print(f"{'='*60}")
    
    # Find running target apps
    target_apps = []
    total_memory_to_free = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
        try:
            name = proc.info['name']
            memory_percent = proc.info['memory_percent']
            
            if optimization_config.should_close_for_langchain(name, memory_percent):
                profile = optimization_config.get_app_profile(name)
                target_apps.append({
                    'name': name,
                    'memory_percent': memory_percent,
                    'priority': profile.priority if profile else 0,
                    'profile': profile
                })
                total_memory_to_free += memory_percent
        except:
            pass
    
    # Sort by priority and memory usage
    target_apps.sort(key=lambda x: (x['priority'], x['memory_percent']), reverse=True)
    
    if target_apps:
        print("\nApps that would be closed (in order):")
        for i, app in enumerate(target_apps, 1):
            profile = app['profile']
            print(f"{i}. {app['name']:30} {app['memory_percent']:5.1f}% "
                  f"[Priority: {app['priority']}, "
                  f"{'Graceful' if profile and profile.graceful_close else 'Force'} close]")
        
        print(f"\nTotal memory that could be freed: ~{total_memory_to_free:.1f}%")
        
        current_mem = psutil.virtual_memory().percent
        projected_mem = current_mem - total_memory_to_free
        print(f"Projected memory after optimization: ~{projected_mem:.1f}%")
        
        if projected_mem < 50:
            print("✅ This should enable LangChain mode!")
        else:
            print("⚠️  May need to close additional apps manually")
    else:
        print("No high-priority apps found to close.")
    
    return target_apps


async def test_optimization():
    """Test the memory optimization"""
    print("\n🧠 Testing Enhanced Memory Optimization for LangChain")
    
    # Show current status
    current_percent = show_memory_status()
    
    if current_percent < 50:
        print("\n✅ Memory is already optimal for LangChain!")
        return
    
    # Show what would be targeted
    target_apps = show_target_apps()
    
    if not target_apps:
        print("\n⚠️  No apps to automatically close. Manual intervention needed.")
        return
    
    # Ask for confirmation
    print(f"\n{'='*60}")
    print("Ready to optimize?")
    print(f"{'='*60}")
    response = input("\nProceed with optimization? (y/n): ")
    
    if response.lower() != 'y':
        print("Optimization cancelled.")
        return
    
    # Run optimization
    print("\n🚀 Starting intelligent memory optimization...")
    optimizer = IntelligentMemoryOptimizer()
    
    # Use aggressive mode if memory is very high
    aggressive = current_percent > 65
    if aggressive:
        print("Using aggressive mode due to high memory usage...")
    
    success, report = await optimizer.optimize_for_langchain(aggressive=aggressive)
    
    # Show results
    print(f"\n{'='*60}")
    print("Optimization Results")
    print(f"{'='*60}")
    print(f"Success: {'✅ Yes' if success else '❌ No'}")
    print(f"Initial Memory: {report['initial_percent']:.1f}%")
    print(f"Final Memory: {report['final_percent']:.1f}%")
    print(f"Memory Freed: {report['memory_freed_mb']:.0f} MB")
    
    if report['actions_taken']:
        print("\nActions taken:")
        for action in report['actions_taken']:
            print(f"  - {action['strategy']}: {action['freed_mb']:.0f} MB")
    
    # Final status
    print("\nFinal Status:")
    if report['final_percent'] < 50:
        print("🎉 LangChain mode is now available!")
        print("\nYou can now use:")
        print("  - Mathematical calculations")
        print("  - Web searches")
        print("  - Wikipedia lookups")
        print("  - Advanced reasoning")
    elif report['final_percent'] < 65:
        print("✅ Intelligent mode is available")
        print("❌ LangChain still requires more memory to be freed")
    else:
        print("⚠️  Memory still high - limited features only")


def show_config():
    """Show current optimization configuration"""
    print(f"\n{'='*60}")
    print("Optimization Configuration")
    print(f"{'='*60}")
    
    print(f"\nTarget memory for LangChain: < {optimization_config.target_memory_percent}%")
    
    print("\nHigh Priority Apps (will be closed first):")
    for app in optimization_config.high_priority_apps:
        print(f"  - {app.name} (priority: {app.priority}, min memory: {app.min_memory_percent}%)")
    
    print("\nSuspendable Apps:")
    for app in optimization_config.suspendable_apps:
        print(f"  - {app.name} (priority: {app.priority})")
    
    print(f"\nBrowser config: Keep {optimization_config.browser_config['max_tabs']} tabs, "
          f"close if > {optimization_config.browser_config['close_if_memory_percent']}%")


if __name__ == "__main__":
    print("🤖 JARVIS LangChain Memory Optimization Test")
    print("\nOptions:")
    print("1. Test optimization")
    print("2. Show configuration")
    print("3. Show current status only")
    
    choice = input("\nSelect option (1-3): ")
    
    if choice == "1":
        asyncio.run(test_optimization())
    elif choice == "2":
        show_config()
    elif choice == "3":
        show_memory_status()
        show_target_apps()
    else:
        print("Invalid choice")