#!/usr/bin/env python3
"""
Advanced Memory Optimization Script for JARVIS
Provides more aggressive optimization strategies to enable LangChain mode
"""

import asyncio
import sys
import os
import argparse
import subprocess
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
import psutil
from pathlib import Path
import json


def check_model_usage():
    """Check for large model files that might be causing memory issues"""
    common_model_paths = [
        Path.home() / ".cache" / "huggingface",
        Path.home() / ".cache" / "torch",
        Path.home() / ".llama",
        Path.home() / ".jarvis" / "models",
        Path("./models"),
        Path("./backend/models")
    ]
    
    large_files = []
    for path in common_model_paths:
        if path.exists():
            for file in path.rglob("*.bin"):
                if file.stat().st_size > 1e9:  # Files larger than 1GB
                    large_files.append((file, file.stat().st_size))
            for file in path.rglob("*.safetensors"):
                if file.stat().st_size > 1e9:
                    large_files.append((file, file.stat().st_size))
    
    if large_files:
        print("\n⚠️  WARNING: Large model files detected!")
        print("These unquantized models use excessive memory:")
        for file, size in sorted(large_files, key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {file.name}: {size / 1e9:.1f} GB")
        print("\n💡 SOLUTION: Use quantized models instead!")
        print("   Run: python setup_m1_optimized_llm.py")
        print("   This will set up efficient GGUF models that use 75% less RAM\n")


def check_llm_config():
    """Check if JARVIS is configured to use quantized models"""
    config_paths = [
        Path.home() / ".jarvis" / "llm_config.json",
        Path(".env.llm"),
        Path("backend/config/llm_config.json")
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            if config_path.suffix == '.json':
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                        if config.get('llm', {}).get('model_type') == 'gguf':
                            return True
                except:
                    pass
            else:
                # Check .env file
                try:
                    with open(config_path) as f:
                        content = f.read()
                        if 'USE_QUANTIZED_MODELS=true' in content:
                            return True
                except:
                    pass
    
    return False


def display_memory_status():
    """Display current memory status"""
    mem = psutil.virtual_memory()
    print(f"\n{'='*50}")
    print(f"Current Memory Status:")
    print(f"  Used: {mem.percent:.1f}%")
    print(f"  Available: {mem.available / (1024**3):.1f} GB")
    print(f"  Total: {mem.total / (1024**3):.1f} GB")
    print(f"{'='*50}\n")
    
    # Check for large model files
    check_model_usage()


async def run_optimization(aggressive=False, target_percent=45):
    """Run memory optimization"""
    optimizer = IntelligentMemoryOptimizer()
    optimizer.target_memory_percent = target_percent
    
    print(f"Starting {'AGGRESSIVE' if aggressive else 'standard'} memory optimization...")
    print(f"Target: {target_percent}% memory usage\n")
    
    # Get initial suggestions
    suggestions = await optimizer.get_optimization_suggestions()
    if suggestions:
        print("Suggested actions:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")
        print()
    
    # Run optimization
    success, report = await optimizer.optimize_for_langchain(aggressive=aggressive)
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Optimization Results:")
    print(f"  Initial Memory: {report['initial_percent']:.1f}%")
    print(f"  Final Memory: {report['final_percent']:.1f}%")
    print(f"  Memory Freed: {report['memory_freed_mb']:.0f} MB")
    print(f"  Success: {'YES' if success else 'NO'}")
    print(f"\nActions Taken:")
    for action in report['actions_taken']:
        print(f"  - {action['strategy']}: freed {action['freed_mb']:.0f} MB")
    print(f"{'='*50}\n")
    
    return success, report


async def interactive_optimization():
    """Interactive optimization with user choices"""
    print("\nJARVIS Memory Optimization Tool")
    print("================================\n")
    
    display_memory_status()
    
    mem = psutil.virtual_memory()
    if mem.percent <= 45:
        print("✅ Memory usage is already optimal for LangChain!")
        return
    
    # Check if using correct model format
    if check_llm_config():
        print("\n✅ Already using quantized models - good!")
    else:
        print("\n❌ Not using quantized models!")
        print("This is likely why you're having memory issues.")
        print("\n🚀 Quick Fix Available!")
        print("Run: python setup_m1_optimized_llm.py")
        fix_now = input("\nSet up optimized models now? (Y/n): ").strip().lower()
        if fix_now != 'n':
            subprocess.run([sys.executable, "setup_m1_optimized_llm.py"])
            return
    
    print("\nMemory optimization is needed to enable LangChain features.")
    print("\nOptions:")
    print("1. Standard optimization (safe, may not free enough)")
    print("2. Aggressive optimization (closes apps, more effective)")
    print("3. Custom optimization (you choose what to close)")
    print("4. Fix model setup (use quantized models)")
    print("5. Exit")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == '1':
        await run_optimization(aggressive=False)
    elif choice == '2':
        print("\n⚠️  WARNING: Aggressive mode will close applications!")
        confirm = input("Continue? (y/N): ").strip().lower()
        if confirm == 'y':
            await run_optimization(aggressive=True)
    elif choice == '3':
        await custom_optimization()
    elif choice == '4':
        print("\nSetting up optimized models...")
        subprocess.run([sys.executable, "setup_m1_optimized_llm.py"])
        return
    elif choice == '5':
        print("Exiting...")
        return
    
    # Check if we should try again
    mem = psutil.virtual_memory()
    if mem.percent > 45:
        print(f"\n⚠️  Memory usage still at {mem.percent:.1f}%")
        retry = input("Try more aggressive optimization? (y/N): ").strip().lower()
        if retry == 'y':
            await run_optimization(aggressive=True, target_percent=40)


async def custom_optimization():
    """Let user choose which apps to close"""
    print("\nAnalyzing running applications...")
    
    # Get high memory apps
    apps = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'memory_info']):
        try:
            if proc.info['memory_percent'] > 1.0:
                apps.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'memory_percent': proc.info['memory_percent'],
                    'memory_mb': proc.info['memory_info'].rss / (1024 * 1024)
                })
        except:
            continue
    
    # Sort by memory usage
    apps.sort(key=lambda x: x['memory_mb'], reverse=True)
    
    print("\nHigh Memory Applications:")
    for i, app in enumerate(apps[:15], 1):
        print(f"{i:2d}. {app['name']:<30} {app['memory_percent']:5.1f}% ({app['memory_mb']:.0f} MB)")
    
    print("\nEnter app numbers to close (comma-separated), or 'q' to quit:")
    choice = input("> ").strip()
    
    if choice.lower() == 'q':
        return
    
    # Parse selections
    try:
        selections = [int(x.strip()) - 1 for x in choice.split(',')]
        
        # Close selected apps
        for idx in selections:
            if 0 <= idx < len(apps):
                app = apps[idx]
                try:
                    proc = psutil.Process(app['pid'])
                    proc.terminate()
                    print(f"✓ Closed {app['name']}")
                except:
                    print(f"✗ Failed to close {app['name']}")
        
        # Show new memory status
        await asyncio.sleep(2)
        display_memory_status()
        
    except ValueError:
        print("Invalid selection")


def main():
    parser = argparse.ArgumentParser(description='JARVIS Memory Optimization Tool')
    parser.add_argument('--aggressive', '-a', action='store_true',
                        help='Use aggressive optimization (closes apps)')
    parser.add_argument('--target', '-t', type=int, default=45,
                        help='Target memory percentage (default: 45)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive mode with choices')
    parser.add_argument('--api', action='store_true',
                        help='Use API endpoint instead of direct optimization')
    
    args = parser.parse_args()
    
    if args.api:
        # Use the API endpoint
        import requests
        print("Using JARVIS API for optimization...")
        try:
            payload = {"aggressive": args.aggressive} if args.aggressive else {}
            response = requests.post("http://localhost:8000/chat/optimize-memory", json=payload)
            if response.status_code == 200:
                result = response.json()
                print(f"\nOptimization {'succeeded' if result['success'] else 'failed'}")
                print(f"Memory: {result['initial_memory_percent']:.1f}% → {result['final_memory_percent']:.1f}%")
                print(f"Freed: {result['memory_freed_mb']:.0f} MB")
            else:
                print(f"API error: {response.status_code}")
        except Exception as e:
            print(f"Failed to connect to API: {e}")
    elif args.interactive:
        asyncio.run(interactive_optimization())
    else:
        asyncio.run(run_optimization(aggressive=args.aggressive, target_percent=args.target))


if __name__ == "__main__":
    main()