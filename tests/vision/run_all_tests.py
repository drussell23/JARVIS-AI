#!/usr/bin/env python3
"""
Comprehensive Test Runner for JARVIS Multi-Window Intelligence
Runs all test suites and generates consolidated reports
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from test_functional import FunctionalTestSuite
from test_performance import PerformanceTestSuite
from test_integration import IntegrationTestSuite
from test_utils import TestResult, generate_test_report, print_test_summary


class ComprehensiveTestRunner:
    """Runs all test suites and generates comprehensive reports"""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.suite_results: Dict[str, Dict[str, Any]] = {}
        self.start_time = None
        self.end_time = None
        
        # Create reports directory
        self.reports_dir = Path("test_reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    async def run_all_suites(self) -> Dict[str, Any]:
        """Run all test suites"""
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("🚀 JARVIS COMPREHENSIVE TEST SUITE")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run each suite
        await self.run_functional_tests()
        await self.run_performance_tests()
        await self.run_integration_tests()
        
        self.end_time = time.time()
        
        # Generate consolidated report
        consolidated_report = self.generate_consolidated_report()
        
        # Print summary
        self.print_consolidated_summary(consolidated_report)
        
        return consolidated_report
    
    async def run_functional_tests(self):
        """Run functional test suite"""
        print("\n" + "-"*60)
        print("Running Functional Tests...")
        print("-"*60)
        
        try:
            suite = FunctionalTestSuite()
            results = await suite.run_all_tests()
            self.results.extend(results)
            
            # Generate suite report
            report = generate_test_report(
                results, 
                str(self.reports_dir / "functional_test_report.json")
            )
            self.suite_results["functional"] = report
            
            print(f"\n✅ Functional tests completed: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
            
        except Exception as e:
            print(f"\n❌ Functional test suite failed: {e}")
            self.suite_results["functional"] = {"error": str(e)}
    
    async def run_performance_tests(self):
        """Run performance test suite"""
        print("\n" + "-"*60)
        print("Running Performance Tests...")
        print("-"*60)
        
        try:
            suite = PerformanceTestSuite()
            results = await suite.run_all_tests()
            self.results.extend(results)
            
            # Generate suite report
            report = generate_test_report(
                results,
                str(self.reports_dir / "performance_test_report.json")
            )
            self.suite_results["performance"] = report
            
            print(f"\n✅ Performance tests completed: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
            
        except Exception as e:
            print(f"\n❌ Performance test suite failed: {e}")
            self.suite_results["performance"] = {"error": str(e)}
    
    async def run_integration_tests(self):
        """Run integration test suite"""
        print("\n" + "-"*60)
        print("Running Integration Tests...")
        print("-"*60)
        
        try:
            suite = IntegrationTestSuite()
            results = await suite.run_all_tests()
            self.results.extend(results)
            
            # Generate suite report
            report = generate_test_report(
                results,
                str(self.reports_dir / "integration_test_report.json")
            )
            self.suite_results["integration"] = report
            
            print(f"\n✅ Integration tests completed: {report['summary']['passed']}/{report['summary']['total_tests']} passed")
            
        except Exception as e:
            print(f"\n❌ Integration test suite failed: {e}")
            self.suite_results["integration"] = {"error": str(e)}
    
    def generate_consolidated_report(self) -> Dict[str, Any]:
        """Generate consolidated report from all suites"""
        total_duration = (self.end_time - self.start_time) if self.end_time else 0
        
        # Calculate overall stats
        total_tests = len(self.results)
        total_passed = sum(1 for r in self.results if r.passed)
        total_failed = total_tests - total_passed
        
        # Group by category
        by_category = {}
        for result in self.results:
            category = result.category
            if category not in by_category:
                by_category[category] = {"passed": 0, "failed": 0, "tests": []}
            
            if result.passed:
                by_category[category]["passed"] += 1
            else:
                by_category[category]["failed"] += 1
            by_category[category]["tests"].append({
                "name": result.test_name,
                "passed": result.passed,
                "duration_ms": result.duration_ms,
                "error": result.error_message
            })
        
        # Extract key metrics
        response_times = [r.duration_ms for r in self.results if r.category == "performance"]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Check success criteria
        success_criteria = self.evaluate_success_criteria()
        
        consolidated = {
            "summary": {
                "total_tests": total_tests,
                "passed": total_passed,
                "failed": total_failed,
                "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
                "duration_seconds": total_duration,
                "timestamp": datetime.now().isoformat()
            },
            "by_suite": self.suite_results,
            "by_category": by_category,
            "key_metrics": {
                "avg_response_time_ms": avg_response_time,
                "test_coverage": self.calculate_test_coverage(),
                "p0_bugs": self.count_p0_bugs(),
                "api_cost_compliance": self.check_api_cost_compliance()
            },
            "success_criteria": success_criteria,
            "recommendations": self.generate_recommendations()
        }
        
        # Save consolidated report
        report_path = self.reports_dir / "consolidated_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(consolidated, f, indent=2)
        
        return consolidated
    
    def evaluate_success_criteria(self) -> Dict[str, bool]:
        """Evaluate success criteria for launch"""
        criteria = {
            "90_percent_test_coverage": self.calculate_test_coverage() >= 90,
            "zero_p0_bugs": self.count_p0_bugs() == 0,
            "response_time_under_3s": self.check_response_time_compliance(),
            "api_cost_under_5_cents": self.check_api_cost_compliance(),
            "all_integration_tests_pass": self.check_integration_tests()
        }
        
        criteria["ready_for_launch"] = all(criteria.values())
        
        return criteria
    
    def calculate_test_coverage(self) -> float:
        """Calculate test coverage percentage"""
        # Count test coverage based on feature areas tested
        feature_areas = {
            "single_window": False,
            "multi_window": False,
            "privacy": False,
            "meeting": False,
            "workflow": False,
            "performance": False,
            "integration": False,
            "error_handling": False
        }
        
        for result in self.results:
            if "single_window" in result.test_name:
                feature_areas["single_window"] = True
            elif "multi_window" in result.test_name:
                feature_areas["multi_window"] = True
            elif "privacy" in result.test_name:
                feature_areas["privacy"] = True
            elif "meeting" in result.test_name:
                feature_areas["meeting"] = True
            elif "workflow" in result.test_name:
                feature_areas["workflow"] = True
            elif result.category == "performance":
                feature_areas["performance"] = True
            elif result.category == "integration":
                feature_areas["integration"] = True
            elif "error" in result.test_name:
                feature_areas["error_handling"] = True
        
        covered = sum(1 for v in feature_areas.values() if v)
        total = len(feature_areas)
        
        return (covered / total) * 100
    
    def count_p0_bugs(self) -> int:
        """Count P0 (critical) bugs"""
        # P0 bugs are failures in basic functionality
        p0_tests = [
            "single_window_basic",
            "multi_window_What am",
            "claude_api_success",
            "macos_window_detection",
            "e2e_developer_error_workflow"
        ]
        
        p0_failures = sum(
            1 for r in self.results 
            if any(p0 in r.test_name for p0 in p0_tests) and not r.passed
        )
        
        return p0_failures
    
    def check_response_time_compliance(self) -> bool:
        """Check if 95% of queries are under 3 seconds"""
        response_times = [
            r.duration_ms for r in self.results 
            if r.duration_ms > 0 and r.category != "performance"  # Exclude perf tests
        ]
        
        if not response_times:
            return True
        
        # Calculate 95th percentile
        sorted_times = sorted(response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p95_time = sorted_times[p95_index] if p95_index < len(sorted_times) else sorted_times[-1]
        
        return p95_time < 3000  # 3 seconds
    
    def check_api_cost_compliance(self) -> bool:
        """Check if API costs are under $0.05 for 90% of requests"""
        cost_tests = [r for r in self.results if "api_cost" in r.test_name]
        
        if not cost_tests:
            return True
        
        compliant = sum(1 for r in cost_tests if r.passed)
        total = len(cost_tests)
        
        return (compliant / total) >= 0.9 if total > 0 else True
    
    def check_integration_tests(self) -> bool:
        """Check if all integration tests pass"""
        integration_tests = [r for r in self.results if r.category == "integration"]
        
        if not integration_tests:
            return False
        
        return all(r.passed for r in integration_tests)
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for consistent failures
        failed_tests = [r for r in self.results if not r.passed]
        
        if failed_tests:
            # Group failures by type
            failure_types = {}
            for failure in failed_tests:
                failure_type = failure.test_name.split('_')[0]
                if failure_type not in failure_types:
                    failure_types[failure_type] = 0
                failure_types[failure_type] += 1
            
            for ftype, count in failure_types.items():
                if count > 2:
                    recommendations.append(f"Multiple failures in {ftype} tests - investigate {ftype} functionality")
        
        # Performance recommendations
        perf_tests = [r for r in self.results if r.category == "performance"]
        if perf_tests:
            slow_tests = [r for r in perf_tests if r.duration_ms > 2000]
            if len(slow_tests) > len(perf_tests) * 0.2:
                recommendations.append("Performance optimization needed - 20% of operations exceed 2 seconds")
        
        # API cost recommendations
        cost_tests = [r for r in self.results if "api_cost" in r.test_name and r.metrics]
        if cost_tests:
            high_cost = [r for r in cost_tests if r.metrics.get("cost", 0) > 0.04]
            if high_cost:
                recommendations.append(f"Optimize API usage - {len(high_cost)} queries approach cost limit")
        
        if not recommendations:
            recommendations.append("All systems performing within acceptable parameters")
        
        return recommendations
    
    def print_consolidated_summary(self, report: Dict[str, Any]):
        """Print consolidated test summary"""
        print("\n" + "="*60)
        print("📊 CONSOLIDATED TEST RESULTS")
        print("="*60)
        
        summary = report["summary"]
        print(f"\nTotal Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
        print(f"Failed: {summary['failed']}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")
        
        print("\nBy Category:")
        for category, stats in report["by_category"].items():
            total = stats["passed"] + stats["failed"]
            pass_rate = stats["passed"] / total if total > 0 else 0
            print(f"  {category}: {stats['passed']}/{total} ({pass_rate:.1%})")
        
        print("\n🎯 Success Criteria:")
        criteria = report["success_criteria"]
        for criterion, met in criteria.items():
            if criterion != "ready_for_launch":
                status = "✅ PASS" if met else "❌ FAIL"
                print(f"  {criterion}: {status}")
        
        print("\n🚀 Launch Readiness:")
        if criteria["ready_for_launch"]:
            print("  ✅ READY FOR LAUNCH - All criteria met!")
        else:
            print("  ❌ NOT READY - Some criteria not met")
        
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
        
        print("\n📁 Reports saved to: test_reports/")
        print("="*60)


async def main():
    """Run comprehensive test suite"""
    runner = ComprehensiveTestRunner()
    report = await runner.run_all_suites()
    
    # Return success status
    return report["success_criteria"]["ready_for_launch"]


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)