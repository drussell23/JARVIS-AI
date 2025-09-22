#!/usr/bin/env python3
"""
Test Resource Management System
==============================

Demonstrates strict resource control for JARVIS on 16GB systems.
"""

import time
import psutil
from backend.resource_manager import get_resource_manager, throttled_operation
from backend.voice_unlock.ml import get_ml_manager


def print_system_status():
    """Print current system resource status"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    
    print(f"\n📊 System Resources:")
    print(f"  Memory: {memory.percent:.1f}% ({memory.used/1024/1024/1024:.1f}GB / {memory.total/1024/1024/1024:.1f}GB)")
    print(f"  Available: {memory.available/1024/1024/1024:.1f}GB")
    print(f"  CPU: {cpu:.1f}%")


def test_resource_management():
    """Test the resource management system"""
    print("🧪 Testing JARVIS Resource Management")
    print("=" * 60)
    
    # Show initial system state
    print_system_status()
    
    # Get resource manager
    rm = get_resource_manager()
    
    print("\n🚀 Starting Resource Manager...")
    status = rm.get_status()
    print(f"  Throttle Level: {status.get('throttle_level', 0)}")
    print(f"  Current ML Model: {status.get('current_ml_model', 'None')}")
    
    # Test ML model loading with resource constraints
    print("\n📦 Testing ML Model Loading...")
    
    # Simulate model loading requests
    models = ['voice_auth_1', 'voice_auth_2', 'speaker_verify']
    
    for i, model_id in enumerate(models):
        print(f"\n🔄 Requesting model: {model_id}")
        
        # Check if we can load the model
        approved = rm.request_ml_model(model_id, priority=10-i)
        
        if approved:
            print(f"  ✅ Approved: {model_id}")
        else:
            print(f"  ❌ Denied: {model_id} (memory pressure)")
            
        # Show current status
        status = rm.get_status()
        print(f"  Memory: {status.get('memory_percent', 0):.1f}%")
        print(f"  Throttle: {status.get('throttle_level', 0)}")
        print(f"  Loaded: {status.get('current_ml_model', 'None')}")
        
        time.sleep(2)  # Simulate some work
        
    # Test throttling
    print("\n⏱️  Testing Throttling...")
    
    @throttled_operation
    def simulated_operation(name):
        print(f"  Executing: {name}")
        time.sleep(0.5)
        
    for i in range(3):
        print(f"\nOperation {i+1}:")
        start = time.time()
        simulated_operation(f"Task_{i+1}")
        elapsed = time.time() - start
        print(f"  Time: {elapsed:.2f}s (includes throttle delay)")
        
    # Test prediction
    print("\n🔮 Testing Model Prediction...")
    predicted = rm.predict_next_model()
    print(f"  Predicted next model: {predicted}")
    
    # Show final status
    print("\n📈 Final Status:")
    print_system_status()
    
    final_status = rm.get_status()
    print(f"\n  JARVIS Memory: {final_status.get('jarvis_memory_mb', 0):.1f}MB")
    print(f"  Active Services: {final_status.get('active_services', {})}")
    
    # Test emergency cleanup
    print("\n🚨 Testing Emergency Cleanup (if memory > 85%)...")
    current_memory = psutil.virtual_memory().percent
    if current_memory > 85:
        print("  Triggering emergency cleanup...")
        rm.enforce_memory_limit()
    else:
        print(f"  Memory at {current_memory:.1f}% - no emergency action needed")
    
    # Stop monitoring
    print("\n🛑 Stopping Resource Manager...")
    rm.stop_monitoring()
    
    print("\n✅ Resource management test complete!")


if __name__ == "__main__":
    test_resource_management()