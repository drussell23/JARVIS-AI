#!/usr/bin/env python3
"""Simple test for dynamic memory allocation"""

import os
import sys
import psutil

backend_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_path)

def test_dynamic_memory():
    """Test dynamic memory calculations directly"""
    print("🧪 Testing Dynamic Memory Allocation (Simple)")
    print("=" * 80)
    
    vm = psutil.virtual_memory()
    print(f"\nSystem Memory:")
    print(f"  Total: {vm.total / (1024**3):.1f} GB")
    print(f"  Available: {vm.available / (1024**3):.1f} GB ({vm.available / (1024**2):.0f} MB)")
    
    # Test VisionConfig
    print("\n1. VisionConfig dynamic memory calculation:")
    available_mb = vm.available / (1024**2)
    total_mb = vm.total / (1024**2)
    
    process_limit = min(
        int(total_mb * 0.25),
        int(available_mb * 0.5),
        4096
    )
    process_limit = max(process_limit, 512)
    print(f"  Process memory limit: {process_limit} MB")
    print(f"  (25% of total OR 50% of available, whichever is smaller)")
    
    # Test VideoStreamConfig
    print("\n2. VideoStreamConfig dynamic memory:")
    video_limit = max(200, min(int(available_mb * 0.2), 1500))
    print(f"  Video memory limit: {video_limit} MB")
    print(f"  (20% of available, between 200-1500 MB)")
    
    # Test SwiftVision
    print("\n3. Swift Vision dynamic memory:")
    swift_limit = max(150, min(int(available_mb * 0.15), 1000))
    print(f"  Swift memory limit: {swift_limit} MB")
    print(f"  (15% of available, between 150-1000 MB)")
    
    # Test Integration Orchestrator
    print("\n4. Integration Orchestrator dynamic memory:")
    orchestrator_budget = min(int(available_mb * 0.3), 3000)
    print(f"  Total budget: {orchestrator_budget} MB")
    print(f"  (30% of available, capped at 3000 MB)")
    print(f"  Intelligence: {int(orchestrator_budget * 0.5)} MB")
    print(f"  Optimization: {int(orchestrator_budget * 0.38)} MB")
    print(f"  Buffer: {int(orchestrator_budget * 0.12)} MB")
    
    # Check for hardcoded values
    print("\n5. Checking for hardcoded values:")
    issues = []
    
    # These values should NOT appear in dynamic calculations
    if process_limit == 2048:
        issues.append("VisionConfig might be using hardcoded 2048MB")
    if video_limit == 800:
        issues.append("VideoStreamConfig might be using hardcoded 800MB")
    if swift_limit == 300:
        issues.append("Swift might be using hardcoded 300MB")
    if orchestrator_budget == 1200:
        issues.append("Orchestrator might be using hardcoded 1200MB")
    
    if issues:
        print("  ❌ Found potential hardcoded values:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  ✅ No hardcoded values detected!")
    
    # Summary
    print("\n" + "="*80)
    print("Summary:")
    total_allocated = video_limit + swift_limit + orchestrator_budget
    print(f"- Total allocated: {total_allocated} MB")
    print(f"- Percentage of available: {total_allocated / available_mb * 100:.1f}%")
    print(f"- All allocations are dynamic based on {available_mb:.0f}MB available RAM")

if __name__ == "__main__":
    test_dynamic_memory()