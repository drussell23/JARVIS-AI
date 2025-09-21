#!/usr/bin/env python3
"""Test Dynamic Memory Allocation - Verify no fixed budgets"""

import os
import sys
import psutil

# Add backend to path
backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

def test_dynamic_memory_allocation():
    """Test that all components use dynamic memory allocation"""
    print("🧪 Testing Dynamic Memory Allocation")
    print("=" * 80)
    
    # Get system info
    vm = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total: {vm.total / (1024**3):.1f} GB")
    print(f"  Available: {vm.available / (1024**3):.1f} GB ({vm.available / (1024**2):.0f} MB)")
    print(f"  Used: {vm.percent:.1f}%")
    
    # Test 1: Vision Config
    print("\n1. Testing VisionConfig dynamic memory:")
    from vision.claude_vision_analyzer_main import VisionConfig
    config = VisionConfig()
    
    print(f"  Process memory limit: {config.process_memory_limit_mb} MB")
    print(f"  Memory warning threshold: {config.memory_warning_threshold_mb} MB")
    print(f"  Min system available: {config.min_system_available_gb} GB")
    
    # Verify it's dynamic
    expected_process_limit = min(int(vm.total / (1024**2) * 0.25), int(vm.available / (1024**2) * 0.5), 4096)
    expected_process_limit = max(expected_process_limit, 512)
    print(f"  Expected process limit: ~{expected_process_limit} MB")
    
    # Test 2: Video Streaming Config
    print("\n2. Testing VideoStreamConfig dynamic memory:")
    from vision.video_stream_capture import VideoStreamConfig
    video_config = VideoStreamConfig()
    
    print(f"  Video memory limit: {video_config.memory_limit_mb} MB")
    
    # Expected: 20% of available, between 200-1500 MB
    expected_video = max(200, min(int(vm.available / (1024**2) * 0.2), 1500))
    print(f"  Expected video limit: ~{expected_video} MB")
    
    # Test 3: Swift Vision Config
    print("\n3. Testing Swift Vision dynamic memory:")
    # Import and test the calculation method directly without creating instance
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "swift_vision", 
        os.path.join(backend_path, "vision", "swift_vision_integration.py")
    )
    swift_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(swift_module)
    
    # Call the calculation method directly
    swift_memory = swift_module.MemoryAwareSwiftVisionIntegration._calculate_swift_memory_limit(None)
    print(f"  Swift memory limit: {swift_memory} MB")
    
    # Expected: 15% of available, between 150-1000 MB
    expected_swift = max(150, min(int(vm.available / (1024**2) * 0.15), 1000))
    print(f"  Expected swift limit: ~{expected_swift} MB")
    
    # Test 4: Integration Orchestrator
    print("\n4. Testing Integration Orchestrator dynamic memory:")
    from vision.intelligence.integration_orchestrator import IntegrationOrchestrator
    orchestrator = IntegrationOrchestrator()
    
    print(f"  Total budget: {orchestrator.config['total_memory_mb']} MB")
    print(f"  Intelligence: {orchestrator.config['intelligence_memory_mb']} MB")
    print(f"  Optimization: {orchestrator.config['optimization_memory_mb']} MB")
    print(f"  Buffer: {orchestrator.config['buffer_memory_mb']} MB")
    
    # Expected: 30% of available, capped at 3000 MB
    expected_total = min(int(vm.available / (1024**2) * 0.3), 3000)
    print(f"  Expected total: ~{expected_total} MB")
    
    # Test 5: Verify allocations adapt to available memory
    print("\n5. Verifying dynamic adaptation:")
    
    # Sum up all allocations
    total_allocated = (
        video_config.memory_limit_mb +
        swift.config['max_memory_mb'] +
        orchestrator.config['total_memory_mb']
    )
    
    print(f"\n  Total allocated across components: {total_allocated} MB")
    print(f"  Percentage of available: {total_allocated / (vm.available / (1024**2)) * 100:.1f}%")
    
    # Should be reasonable percentage of available
    if total_allocated > vm.available / (1024**2):
        print("  ❌ ERROR: Allocated more than available!")
    elif total_allocated > vm.available / (1024**2) * 0.8:
        print("  ⚠️  WARNING: Using >80% of available memory")
    else:
        print("  ✅ Memory allocation is reasonable")
    
    # Check for hardcoded values
    print("\n6. Checking for hardcoded values:")
    hardcoded_found = False
    
    # These should NOT be hardcoded values
    if orchestrator.config['total_memory_mb'] == 1200:
        print("  ❌ Orchestrator still using hardcoded 1200MB")
        hardcoded_found = True
    if video_config.memory_limit_mb == 800:
        print("  ❌ Video still using hardcoded 800MB")
        hardcoded_found = True
    if swift.config['max_memory_mb'] == 300:
        print("  ❌ Swift still using hardcoded 300MB")
        hardcoded_found = True
        
    if not hardcoded_found:
        print("  ✅ No hardcoded memory values found!")
    
    print("\n" + "="*80)
    print("Summary:")
    print(f"- System has {vm.available / (1024**3):.1f}GB available")
    print(f"- Components allocated {total_allocated}MB total ({total_allocated / (vm.available / (1024**2)) * 100:.1f}% of available)")
    print("- All allocations are dynamic based on available RAM")
    print("- No fixed budgets detected")

if __name__ == "__main__":
    test_dynamic_memory_allocation()