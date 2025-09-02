#!/usr/bin/env python3
"""
Automated test runner for Claude Vision Analyzer
Runs all test suites with memory safety enabled
"""

import subprocess
import sys
import os
import time
from datetime import datetime

# Set memory safety environment variables
os.environ.update({
    'VISION_MEMORY_SAFETY': 'true',
    'VISION_PROCESS_LIMIT_MB': '2048',
    'VISION_MIN_SYSTEM_RAM_GB': '2.0',
    'VISION_REJECT_ON_MEMORY': 'true',
    'VISION_MAX_CONCURRENT': '10',
    'VISION_CACHE_SIZE_MB': '100',
    'VISION_CACHE_ENTRIES': '50',
    'VISION_MEMORY_THRESHOLD': '60'
})

# Test suites to run
test_suites = [
    {
        'name': 'Enhanced Vision Integration Tests',
        'file': 'test_enhanced_vision_integration.py',
        'description': 'Tests all enhanced features and API integration'
    },
    {
        'name': 'Memory Usage Analysis',
        'file': 'test_memory_quick.py',
        'description': 'Quick memory usage analysis and leak detection'
    },
    {
        'name': 'Real-World Scenarios',
        'file': 'test_real_world_scenarios.py',
        'description': 'Practical use case tests and performance benchmarks'
    }
]

def print_header(text):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"{text.center(60)}")
    print('='*60)

def run_test(test_info):
    """Run a single test suite"""
    print(f"\n🧪 Running: {test_info['name']}")
    print(f"   {test_info['description']}")
    print(f"   File: {test_info['file']}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, test_info['file']],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.1f}s)")
            return True, duration
        else:
            print(f"❌ FAILED ({duration:.1f}s)")
            print("\nError output:")
            if result.stderr:
                print(result.stderr[-1000:])  # Last 1000 chars of error
            return False, duration
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT (exceeded 5 minutes)")
        return False, 300
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, 0

def main():
    """Main test runner"""
    print_header("Claude Vision Analyzer Test Suite")
    print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Platform: macOS")
    print(f"🧠 Memory Safety: ENABLED")
    
    print("\n📋 Test Configuration:")
    print(f"   Process Limit: {os.environ['VISION_PROCESS_LIMIT_MB']}MB")
    print(f"   Min System RAM: {os.environ['VISION_MIN_SYSTEM_RAM_GB']}GB")
    print(f"   Max Concurrent: {os.environ['VISION_MAX_CONCURRENT']}")
    print(f"   Cache Size: {os.environ['VISION_CACHE_SIZE_MB']}MB")
    
    print_header("Running Tests")
    
    results = []
    total_time = 0
    
    for test in test_suites:
        passed, duration = run_test(test)
        results.append((test['name'], passed, duration))
        total_time += duration
        
        if not passed and '--continue-on-failure' not in sys.argv:
            print("\n⚠️  Test failed. Use --continue-on-failure to run all tests.")
            break
    
    print_header("Test Results Summary")
    
    passed_count = sum(1 for _, passed, _ in results if passed)
    total_count = len(results)
    
    print(f"\n📊 Results: {passed_count}/{total_count} passed")
    print(f"⏱️  Total time: {total_time:.1f}s")
    
    print("\n📋 Detailed Results:")
    for name, passed, duration in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {status} - {name} ({duration:.1f}s)")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! The vision analyzer is ready for production.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())