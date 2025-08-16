#!/usr/bin/env python3
"""
Quick memory optimization script for JARVIS
Checks memory status and offers optimization options
"""

import psutil
import subprocess
import sys
import time
import platform
import os

# Add backend to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_memory_status():
    """Print current memory status"""
    mem = psutil.virtual_memory()
    
    print(f"\n{Colors.HEADER}{'='*60}")
    print(f"{Colors.BOLD}JARVIS Memory Status{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*60}{Colors.ENDC}\n")
    
    print(f"Total Memory: {mem.total / (1024**3):.1f} GB")
    print(f"Used Memory: {mem.used / (1024**3):.1f} GB ({mem.percent:.1f}%)")
    print(f"Available Memory: {mem.available / (1024**3):.1f} GB")
    print(f"Free Memory: {mem.free / (1024**3):.1f} GB")
    
    print(f"\n{Colors.CYAN}JARVIS Mode Requirements:{Colors.ENDC}")
    print(f"🤖 LangChain Mode: < 50% (need < {mem.total * 0.5 / (1024**3):.1f} GB used)")
    print(f"🧠 Intelligent Mode: < 65% (need < {mem.total * 0.65 / (1024**3):.1f} GB used)")
    print(f"💬 Simple Mode: > 80% (forced when > {mem.total * 0.8 / (1024**3):.1f} GB used)")
    
    # Current status
    print(f"\n{Colors.BOLD}Current Status:{Colors.ENDC}")
    if mem.percent < 50:
        print(f"{Colors.GREEN}✅ Memory OK for LangChain mode!{Colors.ENDC}")
        return "langchain"
    elif mem.percent < 65:
        print(f"{Colors.YELLOW}⚠️  Memory OK for Intelligent mode only{Colors.ENDC}")
        return "intelligent"
    elif mem.percent < 80:
        print(f"{Colors.YELLOW}⚠️  Memory OK for Intelligent mode (limited){Colors.ENDC}")
        return "intelligent_limited"
    else:
        print(f"{Colors.FAIL}❌ Memory too high - stuck in Simple mode{Colors.ENDC}")
        return "simple"


def get_top_processes():
    """Get top memory consuming processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'memory_percent']):
        try:
            pinfo = proc.info
            if pinfo['memory_percent'] > 1:  # Only show processes using > 1%
                processes.append(pinfo)
        except:
            pass
    
    # Sort by memory usage
    processes.sort(key=lambda x: x['memory_percent'], reverse=True)
    
    if processes:
        print(f"\n{Colors.CYAN}Top Memory Users:{Colors.ENDC}")
        for i, proc in enumerate(processes[:10]):
            print(f"{i+1}. {proc['name'][:30]:30} {proc['memory_percent']:.1f}%")


def basic_optimization():
    """Run basic memory optimization"""
    print(f"\n{Colors.BLUE}Running basic memory optimization...{Colors.ENDC}")
    
    freed_total = 0
    
    # 1. Python garbage collection
    import gc
    before = psutil.Process().memory_info().rss / (1024 * 1024)
    gc.collect(2)
    after = psutil.Process().memory_info().rss / (1024 * 1024)
    freed = max(0, before - after)
    if freed > 0:
        print(f"  ✓ Garbage collection freed {freed:.0f} MB")
        freed_total += freed
    
    # 2. Kill helper processes (macOS)
    if platform.system() == "Darwin":
        helpers = [
            ("Cursor Helper", "Cursor Helper"),
            ("Chrome Helper", "Chrome Helper"),
            ("Code Helper", "Code Helper"),
        ]
        
        for name, pattern in helpers:
            try:
                result = subprocess.run(
                    ["pkill", "-f", pattern],
                    capture_output=True
                )
                if result.returncode == 0:
                    print(f"  ✓ Killed {name} processes")
                    freed_total += 100  # Estimate
            except:
                pass
    
    # 3. Clear caches
    if platform.system() == "Darwin":
        try:
            subprocess.run(["purge"], capture_output=True, timeout=5)
            print(f"  ✓ Cleared system caches")
            freed_total += 200  # Estimate
        except:
            pass
    
    return freed_total


def intelligent_optimization():
    """Run intelligent memory optimization"""
    print(f"\n{Colors.BLUE}Running intelligent memory optimization...{Colors.ENDC}")
    print(f"{Colors.YELLOW}This will attempt more aggressive optimizations.{Colors.ENDC}")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        return 0
    
    try:
        # Import and run the intelligent optimizer
        from memory.intelligent_memory_optimizer import IntelligentMemoryOptimizer
        import asyncio
        
        optimizer = IntelligentMemoryOptimizer()
        
        async def run_optimization():
            success, report = await optimizer.optimize_for_langchain()
            return success, report
        
        print(f"\n{Colors.CYAN}Starting intelligent optimization...{Colors.ENDC}")
        success, report = asyncio.run(run_optimization())
        
        if success:
            print(f"\n{Colors.GREEN}✅ Optimization successful!{Colors.ENDC}")
        else:
            print(f"\n{Colors.YELLOW}⚠️  Optimization completed but target not reached{Colors.ENDC}")
        
        print(f"Memory freed: {report['memory_freed_mb']:.0f} MB")
        print(f"Final memory: {report['final_percent']:.1f}%")
        
        if report['actions_taken']:
            print(f"\n{Colors.CYAN}Actions taken:{Colors.ENDC}")
            for action in report['actions_taken']:
                print(f"  - {action['strategy']}: {action['freed_mb']:.0f} MB")
        
        return report['memory_freed_mb']
        
    except ImportError:
        print(f"{Colors.FAIL}❌ Intelligent optimizer not available{Colors.ENDC}")
        print("Make sure you're running from the project root directory")
        return 0
    except Exception as e:
        print(f"{Colors.FAIL}❌ Optimization failed: {e}{Colors.ENDC}")
        return 0


def main():
    """Main entry point"""
    print(f"{Colors.BOLD}🧠 JARVIS Memory Optimization Tool{Colors.ENDC}")
    
    # Show current status
    mode = print_memory_status()
    
    # Show top processes
    get_top_processes()
    
    # If memory is already good, exit
    if mode == "langchain":
        print(f"\n{Colors.GREEN}Memory is already optimized for all features!{Colors.ENDC}")
        sys.exit(0)
    
    # Offer optimization options
    print(f"\n{Colors.CYAN}Optimization Options:{Colors.ENDC}")
    print("1. Basic optimization (safe, quick)")
    print("2. Intelligent optimization (aggressive, effective)")
    print("3. Manual suggestions only")
    print("4. Exit")
    
    choice = input("\nSelect option (1-4): ")
    
    if choice == "1":
        freed = basic_optimization()
        if freed > 0:
            print(f"\n{Colors.GREEN}Freed approximately {freed:.0f} MB{Colors.ENDC}")
            time.sleep(2)
            print("\nChecking new memory status...")
            print_memory_status()
        else:
            print(f"\n{Colors.YELLOW}No significant memory freed{Colors.ENDC}")
            
    elif choice == "2":
        freed = intelligent_optimization()
        if freed > 0:
            time.sleep(2)
            print("\nChecking new memory status...")
            print_memory_status()
            
    elif choice == "3":
        print(f"\n{Colors.CYAN}Manual Optimization Suggestions:{Colors.ENDC}")
        print("1. Close browser tabs you're not using")
        print("2. Quit heavy applications (IDEs, Docker, etc.)")
        print("3. Close Slack, Discord, and other chat apps")
        print("4. Restart your computer if memory usage remains high")
        print("\nFor automatic optimization, run this script again and choose option 1 or 2")
        
    elif choice == "4":
        print("Exiting...")
        sys.exit(0)
    else:
        print("Invalid choice")
    
    # Final tip
    mem = psutil.virtual_memory()
    if mem.percent > 50:
        print(f"\n{Colors.CYAN}Tip: For best JARVIS performance, keep memory below 50%{Colors.ENDC}")


if __name__ == "__main__":
    main()