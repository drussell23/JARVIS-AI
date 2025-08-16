#!/usr/bin/env python3
"""
Quick memory usage checker for JARVIS
"""
import psutil
import subprocess

def get_memory_info():
    """Get detailed memory information"""
    mem = psutil.virtual_memory()
    
    print("=" * 60)
    print("JARVIS Memory Status")
    print("=" * 60)
    print(f"Total Memory: {mem.total / (1024**3):.1f} GB")
    print(f"Used Memory: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
    print(f"Available Memory: {mem.available / (1024**3):.1f} GB")
    print(f"Free Memory: {mem.free / (1024**3):.1f} GB")
    print("=" * 60)
    
    # JARVIS mode thresholds
    print("\nJARVIS Mode Requirements:")
    print(f"🤖 LangChain Mode: < 50% (need < {mem.total * 0.5 / (1024**3):.1f} GB used)")
    print(f"🧠 Intelligent Mode: < 65% (need < {mem.total * 0.65 / (1024**3):.1f} GB used)")
    print(f"💬 Simple Mode: > 80% (forced when > {mem.total * 0.8 / (1024**3):.1f} GB used)")
    print("=" * 60)
    
    # Current status
    if mem.percent < 50:
        print("✅ Memory OK for LangChain mode!")
    elif mem.percent < 65:
        print("⚠️  Memory OK for Intelligent mode only")
    elif mem.percent < 80:
        print("⚠️  Memory OK for Intelligent mode (limited)")
    else:
        print("❌ Memory too high - stuck in Simple mode")
    
    # Top memory users
    print("\nTop Memory Users:")
    try:
        # Get top processes by memory
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
        
        for i, proc in enumerate(processes[:10]):
            print(f"{i+1}. {proc['name'][:30]:30} {proc['memory_percent']:.1f}%")
    except:
        print("Could not get process list")

if __name__ == "__main__":
    get_memory_info()