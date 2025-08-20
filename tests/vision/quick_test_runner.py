#!/usr/bin/env python3
"""Quick test runner for debugging"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test_functional import FunctionalTestSuite
from test_utils import generate_test_report, print_test_summary

async def run_quick_tests():
    """Run a quick subset of tests"""
    print("\n🚀 JARVIS QUICK TEST RUNNER")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run functional tests only
    print("\nRunning Functional Tests...")
    suite = FunctionalTestSuite()
    
    # Run just a few tests
    await suite.test_single_window_analysis()
    await suite.test_privacy_controls()
    
    # Generate report
    report = generate_test_report(suite.results, "quick_test_report.json")
    print_test_summary(report)
    
    # Check success
    success = report["summary"]["pass_rate"] >= 0.8
    
    print("\n" + "="*60)
    if success:
        print("✅ Quick tests PASSED!")
    else:
        print("❌ Quick tests FAILED!")
    print("="*60)
    
    return success

if __name__ == "__main__":
    success = asyncio.run(run_quick_tests())
    sys.exit(0 if success else 1)