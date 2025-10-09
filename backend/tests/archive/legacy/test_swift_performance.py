#!/usr/bin/env python3
"""
Test Swift Performance Bridge Integration
"""

import os
import sys
import time

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_swift_system_monitor():
    """Test Swift system monitoring integration"""
    print("🧪 Testing Swift System Monitor...")
    print("=" * 50)
    
    try:
        from core.swift_system_monitor import get_swift_system_monitor, SWIFT_MONITORING_AVAILABLE
        
        print(f"Swift monitoring available: {SWIFT_MONITORING_AVAILABLE}")
        
        if not SWIFT_MONITORING_AVAILABLE:
            print("❌ Swift performance bridge not available")
            return False
        
        # Get monitor instance
        monitor = get_swift_system_monitor()
        print(f"Monitor enabled: {monitor.enabled}")
        
        # Get current metrics
        print("\n📊 Getting system metrics...")
        metrics = monitor.get_current_metrics()
        
        print(f"CPU: {metrics.cpu_percent:.1f}%")
        print(f"Memory: {metrics.memory_used_mb}MB / {metrics.memory_total_mb}MB ({metrics.memory_percent:.1f}%)")
        print(f"Available: {metrics.memory_available_mb}MB")
        print(f"Pressure: {metrics.memory_pressure}")
        print(f"JARVIS Memory: {metrics.jarvis_memory_mb}MB")
        
        # Test performance
        print("\n⚡ Testing performance overhead...")
        start = time.time()
        for _ in range(100):
            _ = monitor.get_current_metrics()
        elapsed = time.time() - start
        avg_ms = (elapsed / 100) * 1000
        
        print(f"Average time per call: {avg_ms:.2f}ms")
        
        # Get performance stats
        stats = monitor.get_performance_stats()
        print(f"\nMonitoring stats: {stats}")
        
        print("\n✅ Swift system monitor test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Swift system monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_swift_audio_processor():
    """Test Swift audio processor integration"""
    print("\n🎤 Testing Swift Audio Processor...")
    print("=" * 50)
    
    try:
        from core.swift_audio_integration import SwiftAudioBridge, test_swift_audio
        
        # Run test
        success = test_swift_audio()
        
        if success:
            print("✅ Swift audio processor test passed!")
        else:
            print("❌ Swift audio processor test failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Swift audio processor test failed: {e}")
        return False

def test_swift_vision_processor():
    """Test Swift vision processor integration"""
    print("\n👁️ Testing Swift Vision Processor...")
    print("=" * 50)
    
    try:
        from core.swift_vision_integration import SwiftVisionBridge
        
        # Create bridge
        bridge = SwiftVisionBridge()
        
        if bridge.available:
            print("✅ Swift vision processor available")
            
            # Test with a small test image
            import numpy as np
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            
            # Process image
            result = bridge.process_image(test_image)
            print(f"Processing result: {result}")
            
            print("✅ Swift vision processor test passed!")
            return True
        else:
            print("❌ Swift vision processor not available")
            return False
            
    except Exception as e:
        print(f"❌ Swift vision processor test failed: {e}")
        return False

def main():
    """Run all Swift performance tests"""
    print("🚀 JARVIS Swift Performance Bridge Test Suite")
    print("=" * 60)
    
    # Set library path
    lib_path = os.path.join(os.path.dirname(__file__), "swift_bridge/.build/release")
    os.environ["DYLD_LIBRARY_PATH"] = lib_path
    print(f"Library path: {lib_path}")
    
    # Check if library exists
    lib_file = os.path.join(lib_path, "libPerformanceCore.dylib")
    if os.path.exists(lib_file):
        print(f"✅ Found library: {lib_file}")
    else:
        print(f"❌ Library not found: {lib_file}")
        return
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_swift_system_monitor():
        tests_passed += 1
    
    if test_swift_audio_processor():
        tests_passed += 1
        
    if test_swift_vision_processor():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"📊 Test Summary: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All Swift performance tests passed!")
    elif tests_passed > 0:
        print("⚠️  Some tests failed")
    else:
        print("❌ All tests failed")

if __name__ == "__main__":
    main()