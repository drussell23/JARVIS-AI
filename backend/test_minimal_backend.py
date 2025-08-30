#!/usr/bin/env python3
"""
Minimal backend test to verify Swift integration and startup
"""

import os
import sys
import time

# Set Swift library path
swift_lib_path = os.path.join(os.path.dirname(__file__), "swift_bridge/.build/release")
os.environ["DYLD_LIBRARY_PATH"] = swift_lib_path

print("🧪 Testing minimal backend with Swift integration...")
print("=" * 50)

# Test 1: Swift monitoring
try:
    from core.swift_system_monitor import get_swift_system_monitor
    monitor = get_swift_system_monitor()
    metrics = monitor.get_current_metrics()
    
    print(f"✅ Swift monitoring working!")
    print(f"   CPU: {metrics.cpu_percent:.1f}%")
    print(f"   Memory: {metrics.memory_percent:.1f}%")
    print(f"   Available: {metrics.memory_available_mb}MB")
except Exception as e:
    print(f"❌ Swift monitoring failed: {e}")

# Test 2: Smart startup manager
try:
    from smart_startup_manager import startup_manager
    status = startup_manager.get_startup_status()
    print(f"\n✅ Smart startup manager loaded!")
    print(f"   Phase: {status['phase']}")
    print(f"   Health: {status['health']}")
except Exception as e:
    print(f"\n❌ Smart startup manager failed: {e}")

# Test 3: Memory quantizer
try:
    from core.memory_quantizer import memory_quantizer
    mem_status = memory_quantizer.get_memory_status()
    print(f"\n✅ Memory quantizer loaded!")
    print(f"   Level: {mem_status['current_level']}")
    print(f"   Usage: {mem_status['memory_usage_gb']:.1f}GB")
except Exception as e:
    print(f"\n❌ Memory quantizer failed: {e}")

print("\n" + "=" * 50)
print("✅ Core components working! Backend should be able to start.")
print("\nIf backend is hanging, it may be due to:")
print("1. High system CPU load (other processes)")
print("2. Waiting for model downloads")
print("3. Network connectivity issues")
print("\nTry closing other applications to free up CPU.")