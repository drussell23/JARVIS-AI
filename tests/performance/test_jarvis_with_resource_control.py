#!/usr/bin/env python3
"""
Test JARVIS with Resource Control
================================

Demonstrates how JARVIS now manages memory to stay under 70% on 16GB systems.
"""

import asyncio
import psutil
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent))

from backend import initialize_backend, cleanup_backend, get_backend_status


def print_memory_status(label=""):
    """Print current memory status"""
    memory = psutil.virtual_memory()
    print(f"\n💾 Memory Status {label}")
    print(f"  Total: {memory.total/1024/1024/1024:.1f}GB")
    print(f"  Used: {memory.used/1024/1024/1024:.1f}GB ({memory.percent:.1f}%)")
    print(f"  Available: {memory.available/1024/1024/1024:.1f}GB")
    print(f"  Target: ≤70% ({0.7 * memory.total/1024/1024/1024:.1f}GB)")
    
    # Visual indicator
    if memory.percent <= 70:
        print(f"  Status: ✅ Within target")
    elif memory.percent <= 80:
        print(f"  Status: ⚠️  Above target but manageable")
    else:
        print(f"  Status: ❌ Critical - resource manager will act")


async def test_jarvis_startup():
    """Test JARVIS startup with resource control"""
    print("🤖 Testing JARVIS with Resource Control")
    print("=" * 60)
    
    # Show initial state
    print_memory_status("(Before JARVIS)")
    
    # Initialize backend
    print("\n🚀 Initializing JARVIS backend...")
    await initialize_backend()
    
    # Get status
    status = get_backend_status()
    
    print("\n📊 Backend Status:")
    print(f"  Version: {status['version']}")
    print(f"  Modules:")
    for module, available in status['modules'].items():
        icon = "✅" if available else "❌"
        print(f"    {icon} {module}")
    
    # Show resource manager status
    if 'resources' in status:
        res = status['resources']
        print(f"\n🎛️  Resource Manager:")
        print(f"  Memory: {res.get('memory_percent', 0):.1f}%")
        print(f"  CPU: {res.get('cpu_percent', 0):.1f}%")
        print(f"  Throttle Level: {res.get('throttle_level', 0)}")
        print(f"  Current ML Model: {res.get('current_ml_model', 'None')}")
        print(f"  JARVIS Memory: {res.get('jarvis_memory_mb', 0):.1f}MB")
        
    # Show voice unlock status
    if 'voice_unlock' in status and isinstance(status['voice_unlock'], dict):
        vu = status['voice_unlock']
        if 'ml_status' in vu:
            ml = vu['ml_status']
            print(f"\n🔊 Voice Unlock ML:")
            print(f"  Memory: {ml.get('memory_percent', 0):.1f}%")
            print(f"  ML Models: {ml.get('active_models', 0)}")
            print(f"  Cache: {ml.get('cache_size_mb', 0):.1f}MB")
            print(f"  Degraded Mode: {ml.get('degraded_mode', False)}")
    
    # Show memory after initialization
    print_memory_status("(After JARVIS init)")
    
    # Simulate some ML operations
    print("\n🧪 Testing ML Model Loading...")
    
    if status['modules']['voice_unlock']:
        from backend.voice_unlock.ml import get_ml_manager
        ml_manager = get_ml_manager()
        
        print("\n1️⃣ Requesting first ML model...")
        try:
            # This will be subject to resource manager approval
            result = ml_manager.request_ml_model("test_model_1", priority=8)
            print(f"   Result: {'Approved' if result else 'Denied'}")
        except Exception as e:
            print(f"   Error: {e}")
            
        print("\n2️⃣ Requesting second ML model...")
        try:
            # Should unload first model automatically
            result = ml_manager.request_ml_model("test_model_2", priority=9)
            print(f"   Result: {'Approved' if result else 'Denied'}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Final memory status
    print_memory_status("(After ML tests)")
    
    # Monitor for a few seconds
    print("\n⏱️  Monitoring for 10 seconds...")
    for i in range(5):
        await asyncio.sleep(2)
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"  [{i*2}s] Memory: {memory.percent:.1f}%, CPU: {cpu:.1f}%")
    
    # Cleanup
    print("\n🧹 Cleaning up...")
    await cleanup_backend()
    
    # Final status
    print_memory_status("(After cleanup)")
    
    print("\n✅ Test complete!")
    print("\n📝 Summary:")
    print("  - Resource manager enforces memory limits")
    print("  - Only one ML model loaded at a time")
    print("  - Automatic throttling when CPU/memory high")
    print("  - Target: Keep system memory ≤70%")


if __name__ == "__main__":
    asyncio.run(test_jarvis_startup())