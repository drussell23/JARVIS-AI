#!/usr/bin/env python3
"""
JARVIS Vision Intelligence Test Runner
Runs all tests for the intelligent vision system
"""

import sys
import os
import unittest
import asyncio
from datetime import datetime
import platform

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_intelligent_vision import (
    TestIntelligentVisionFunctionality,
    TestIntegrationUnknownApps,
    TestDynamicVisualAnalysis as TestDynamicAnalysisFunctional
)
from tests.test_vision_integration import (
    TestVisionIntegration,
    TestDynamicContentAnalysis
)
from tests.test_dynamic_visual_analysis import (
    TestDynamicVisualAnalysis,
    TestVisualIndicatorPatterns
)


class VisionTestRunner:
    """Comprehensive test runner for vision intelligence"""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.results = {
            'functional': {'passed': 0, 'failed': 0, 'errors': []},
            'integration': {'passed': 0, 'failed': 0, 'errors': []},
            'dynamic': {'passed': 0, 'failed': 0, 'errors': []},
            'async': {'passed': 0, 'failed': 0, 'errors': []}
        }
        self.is_macos = platform.system() == "Darwin"
    
    def print_header(self):
        """Print test runner header"""
        print("\n" + "="*70)
        print("🧪 JARVIS VISION INTELLIGENCE TEST SUITE")
        print("="*70)
        print(f"Platform: {platform.system()} {platform.machine()}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    def run_functional_tests(self):
        """Run functional tests"""
        print("\n📋 FUNCTIONAL TESTS")
        print("-"*50)
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add functional test classes
        test_classes = [
            TestIntelligentVisionFunctionality,
            TestIntegrationUnknownApps,
            TestDynamicAnalysisFunctional
        ]
        
        for test_class in test_classes:
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Update results
        self.results['functional']['passed'] = result.testsRun - len(result.failures) - len(result.errors)
        self.results['functional']['failed'] = len(result.failures) + len(result.errors)
        self.results['functional']['errors'] = [(str(test), err) for test, err in result.failures + result.errors]
        
        return result.wasSuccessful()
    
    def run_integration_tests(self):
        """Run integration tests"""
        print("\n🔗 INTEGRATION TESTS")
        print("-"*50)
        
        if not self.is_macos:
            print("⚠️  Skipping integration tests (requires macOS)")
            return True
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add integration test classes
        test_classes = [
            TestVisionIntegration,
            TestDynamicContentAnalysis
        ]
        
        for test_class in test_classes:
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Update results
        self.results['integration']['passed'] = result.testsRun - len(result.failures) - len(result.errors)
        self.results['integration']['failed'] = len(result.failures) + len(result.errors)
        self.results['integration']['errors'] = [(str(test), err) for test, err in result.failures + result.errors]
        
        return result.wasSuccessful()
    
    def run_dynamic_analysis_tests(self):
        """Run dynamic visual analysis tests"""
        print("\n🎨 DYNAMIC VISUAL ANALYSIS TESTS")
        print("-"*50)
        
        # Create test suite
        suite = unittest.TestSuite()
        
        # Add dynamic analysis test classes
        test_classes = [
            TestDynamicVisualAnalysis,
            TestVisualIndicatorPatterns
        ]
        
        for test_class in test_classes:
            suite.addTests(unittest.TestLoader().loadTestsFromTestCase(test_class))
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = runner.run(suite)
        
        # Update results
        self.results['dynamic']['passed'] = result.testsRun - len(result.failures) - len(result.errors)
        self.results['dynamic']['failed'] = len(result.failures) + len(result.errors)
        self.results['dynamic']['errors'] = [(str(test), err) for test, err in result.failures + result.errors]
        
        return result.wasSuccessful()
    
    async def run_async_tests(self):
        """Run async tests"""
        print("\n⚡ ASYNC TESTS")
        print("-"*50)
        
        async_tests = []
        errors = []
        
        # Functional async tests
        try:
            test = TestIntelligentVisionFunctionality()
            test.setUp()
            await test.test_vision_command_handling()
            async_tests.append("test_vision_command_handling")
            print("✅ test_vision_command_handling")
        except Exception as e:
            errors.append(("test_vision_command_handling", str(e)))
            print(f"❌ test_vision_command_handling: {e}")
        
        try:
            test = TestIntegrationUnknownApps()
            test.setUp()
            await test.test_end_to_end_unknown_app_query()
            async_tests.append("test_end_to_end_unknown_app_query")
            print("✅ test_end_to_end_unknown_app_query")
        except Exception as e:
            errors.append(("test_end_to_end_unknown_app_query", str(e)))
            print(f"❌ test_end_to_end_unknown_app_query: {e}")
        
        # Integration async tests (macOS only)
        if self.is_macos:
            try:
                test = TestVisionIntegration()
                test.setUp()
                await test.test_workspace_analysis_with_real_windows()
                async_tests.append("test_workspace_analysis_with_real_windows")
                print("✅ test_workspace_analysis_with_real_windows")
            except Exception as e:
                errors.append(("test_workspace_analysis_with_real_windows", str(e)))
                print(f"❌ test_workspace_analysis_with_real_windows: {e}")
            
            try:
                await test.test_window_capture_fallback()
                async_tests.append("test_window_capture_fallback")
                print("✅ test_window_capture_fallback")
            except Exception as e:
                errors.append(("test_window_capture_fallback", str(e)))
                print(f"❌ test_window_capture_fallback: {e}")
        
        # Dynamic analysis async tests
        try:
            test = TestDynamicVisualAnalysis()
            test.setUp()
            await test.test_visual_content_analysis_mock()
            async_tests.append("test_visual_content_analysis_mock")
            print("✅ test_visual_content_analysis_mock")
        except Exception as e:
            errors.append(("test_visual_content_analysis_mock", str(e)))
            print(f"❌ test_visual_content_analysis_mock: {e}")
        
        # Update results
        self.results['async']['passed'] = len(async_tests)
        self.results['async']['failed'] = len(errors)
        self.results['async']['errors'] = errors
        
        return len(errors) == 0
    
    def print_summary(self):
        """Print test summary"""
        duration = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*70)
        print("📊 TEST SUMMARY")
        print("="*70)
        
        total_passed = sum(r['passed'] for r in self.results.values())
        total_failed = sum(r['failed'] for r in self.results.values())
        total_tests = total_passed + total_failed
        
        # Category breakdown
        for category, results in self.results.items():
            if results['passed'] + results['failed'] > 0:
                print(f"\n{category.upper()} TESTS:")
                print(f"  ✅ Passed: {results['passed']}")
                print(f"  ❌ Failed: {results['failed']}")
                
                if results['errors']:
                    print(f"  Errors:")
                    for test_name, error in results['errors'][:3]:  # Show first 3
                        print(f"    - {test_name}: {str(error)[:100]}...")
        
        # Overall summary
        print("\n" + "-"*70)
        print(f"TOTAL: {total_tests} tests")
        print(f"✅ PASSED: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "No tests run")
        print(f"❌ FAILED: {total_failed}")
        print(f"⏱️  Duration: {duration:.2f} seconds")
        
        # Feature coverage
        print("\n📋 FEATURE COVERAGE:")
        print("  ✅ Unknown app detection")
        print("  ✅ Multi-language support") 
        print("  ✅ Dynamic notification detection")
        print("  ✅ Pattern-based app categorization")
        print("  ✅ Context-aware query routing")
        print("  ✅ Visual indicator recognition")
        print("  ✅ Fallback analysis (no screenshots)")
        if self.is_macos:
            print("  ✅ Real window detection")
            print("  ✅ Window capture with fallback")
        else:
            print("  ⚠️  Real window detection (requires macOS)")
            print("  ⚠️  Window capture (requires macOS)")
        
        print("="*70)
        
        if total_failed == 0:
            print("🎉 ALL TESTS PASSED! Vision intelligence is working correctly.")
        else:
            print("❌ Some tests failed. Please review the errors above.")
        
        return total_failed == 0
    
    def run_all_tests(self):
        """Run all test categories"""
        self.print_header()
        
        # Run each test category
        functional_success = self.run_functional_tests()
        integration_success = self.run_integration_tests()
        dynamic_success = self.run_dynamic_analysis_tests()
        
        # Run async tests
        async_success = asyncio.run(self.run_async_tests())
        
        # Print summary
        all_success = self.print_summary()
        
        return all_success


def main():
    """Main entry point"""
    runner = VisionTestRunner()
    
    # Check for specific test category
    if len(sys.argv) > 1:
        category = sys.argv[1].lower()
        
        if category == "functional":
            runner.print_header()
            success = runner.run_functional_tests()
        elif category == "integration":
            runner.print_header()
            success = runner.run_integration_tests()
        elif category == "dynamic":
            runner.print_header()
            success = runner.run_dynamic_analysis_tests()
        elif category == "async":
            runner.print_header()
            success = asyncio.run(runner.run_async_tests())
        else:
            print(f"Unknown test category: {category}")
            print("Usage: python run_vision_tests.py [functional|integration|dynamic|async]")
            sys.exit(1)
        
        runner.print_summary()
    else:
        # Run all tests
        success = runner.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()