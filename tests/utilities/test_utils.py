#!/usr/bin/env python3
"""
Test Utilities and Fixtures for JARVIS Testing
Provides common test data and helper functions
"""

import time
import psutil
import json
import asyncio
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from pathlib import Path

from backend.vision.window_detector import WindowInfo


@dataclass
class TestResult:
    """Result of a single test"""
    test_name: str
    category: str  # functional, performance, integration
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a test run"""
    response_time_ms: float
    cpu_percent: float
    memory_mb: float
    api_calls: int = 0
    api_cost: float = 0.0
    windows_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        self.start_time = None
        self.start_cpu = None
        self.start_memory = None
        self.process = psutil.Process()
    
    def start(self):
        """Start monitoring"""
        self.start_time = time.time()
        self.start_cpu = self.process.cpu_percent()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
    
    def stop(self) -> PerformanceMetrics:
        """Stop monitoring and return metrics"""
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000
        
        # Get final measurements
        cpu_percent = self.process.cpu_percent()
        memory_mb = self.process.memory_info().rss / 1024 / 1024
        
        return PerformanceMetrics(
            response_time_ms=duration_ms,
            cpu_percent=cpu_percent,
            memory_mb=memory_mb - self.start_memory  # Delta
        )


class WindowFixtures:
    """Generate test window configurations"""
    
    @staticmethod
    def single_window() -> List[WindowInfo]:
        """Single window test case"""
        return [
            WindowInfo(
                window_id=1,
                app_name="Visual Studio Code",
                window_title="test.py — JARVIS-AI-Agent",
                is_focused=True,
                bounds={"x": 0, "y": 0, "width": 1920, "height": 1080},
                layer=0,
                is_visible=True,
                process_id=1234
            )
        ]
    
    @staticmethod
    def development_setup() -> List[WindowInfo]:
        """Typical development setup"""
        return [
            WindowInfo(
                window_id=1,
                app_name="Visual Studio Code",
                window_title="workspace_analyzer.py — JARVIS-AI-Agent",
                is_focused=True,
                bounds={"x": 0, "y": 0, "width": 1200, "height": 800},
                layer=0,
                is_visible=True,
                process_id=1234
            ),
            WindowInfo(
                window_id=2,
                app_name="Terminal",
                window_title="~/Documents/repos/JARVIS-AI-Agent",
                is_focused=False,
                bounds={"x": 1200, "y": 0, "width": 720, "height": 400},
                layer=0,
                is_visible=True,
                process_id=1235
            ),
            WindowInfo(
                window_id=3,
                app_name="Chrome",
                window_title="Python asyncio documentation - Google Chrome",
                is_focused=False,
                bounds={"x": 1200, "y": 400, "width": 720, "height": 400},
                layer=0,
                is_visible=True,
                process_id=1236
            )
        ]
    
    @staticmethod
    def meeting_setup() -> List[WindowInfo]:
        """Meeting preparation setup"""
        return [
            WindowInfo(
                window_id=1,
                app_name="Zoom",
                window_title="Zoom Meeting",
                is_focused=True,
                bounds={"x": 0, "y": 0, "width": 1920, "height": 1080},
                layer=0,
                is_visible=True,
                process_id=2001
            ),
            WindowInfo(
                window_id=2,
                app_name="Calendar",
                window_title="Calendar",
                is_focused=False,
                bounds={"x": 100, "y": 100, "width": 800, "height": 600},
                layer=0,
                is_visible=True,
                process_id=2002
            ),
            WindowInfo(
                window_id=3,
                app_name="Notes",
                window_title="Meeting Notes - Project Update",
                is_focused=False,
                bounds={"x": 900, "y": 100, "width": 600, "height": 600},
                layer=0,
                is_visible=True,
                process_id=2003
            ),
            WindowInfo(
                window_id=4,
                app_name="1Password",
                window_title="1Password 7 - Personal",
                is_focused=False,
                bounds={"x": 200, "y": 200, "width": 400, "height": 300},
                layer=0,
                is_visible=True,
                process_id=2004
            )
        ]
    
    @staticmethod
    def communication_heavy() -> List[WindowInfo]:
        """Multiple communication apps"""
        return [
            WindowInfo(
                window_id=1,
                app_name="Discord",
                window_title="Discord - #general",
                is_focused=False,
                bounds={"x": 0, "y": 0, "width": 600, "height": 800},
                layer=0,
                is_visible=True,
                process_id=3001
            ),
            WindowInfo(
                window_id=2,
                app_name="Slack",
                window_title="Slack - Workspace (2)",
                is_focused=False,
                bounds={"x": 600, "y": 0, "width": 600, "height": 800},
                layer=0,
                is_visible=True,
                process_id=3002
            ),
            WindowInfo(
                window_id=3,
                app_name="Messages",
                window_title="Messages",
                is_focused=False,
                bounds={"x": 1200, "y": 0, "width": 400, "height": 800},
                layer=0,
                is_visible=True,
                process_id=3003
            ),
            WindowInfo(
                window_id=4,
                app_name="Mail",
                window_title="Mail - Inbox (5)",
                is_focused=True,
                bounds={"x": 0, "y": 0, "width": 1600, "height": 900},
                layer=1,
                is_visible=True,
                process_id=3004
            )
        ]
    
    @staticmethod
    def edge_case_many_windows(count: int = 25) -> List[WindowInfo]:
        """Generate many windows for edge case testing"""
        windows = []
        apps = ["Chrome", "Safari", "Terminal", "VS Code", "Finder", "Preview", "Notes"]
        
        for i in range(count):
            app = apps[i % len(apps)]
            windows.append(WindowInfo(
                window_id=i + 1,
                app_name=app,
                window_title=f"{app} - Window {i + 1}",
                is_focused=(i == 0),
                bounds={
                    "x": (i * 50) % 1600,
                    "y": (i * 30) % 900,
                    "width": 400,
                    "height": 300
                },
                layer=i // 10,
                is_visible=(i < 20),  # Some hidden
                process_id=4000 + i
            ))
        
        return windows
    
    @staticmethod
    def privacy_sensitive() -> List[WindowInfo]:
        """Windows with sensitive content"""
        return [
            WindowInfo(
                window_id=1,
                app_name="1Password",
                window_title="1Password 7 - Personal Vault",
                is_focused=False,
                bounds={"x": 0, "y": 0, "width": 600, "height": 400},
                layer=0,
                is_visible=True,
                process_id=5001
            ),
            WindowInfo(
                window_id=2,
                app_name="Chrome",
                window_title="Bank of America - Account Summary",
                is_focused=False,
                bounds={"x": 600, "y": 0, "width": 800, "height": 600},
                layer=0,
                is_visible=True,
                process_id=5002
            ),
            WindowInfo(
                window_id=3,
                app_name="Preview",
                window_title="tax_return_2024.pdf",
                is_focused=False,
                bounds={"x": 0, "y": 400, "width": 600, "height": 400},
                layer=0,
                is_visible=True,
                process_id=5003
            ),
            WindowInfo(
                window_id=4,
                app_name="TextEdit",
                window_title="passwords.txt",
                is_focused=True,
                bounds={"x": 600, "y": 600, "width": 400, "height": 300},
                layer=0,
                is_visible=True,
                process_id=5004
            )
        ]


class QueryFixtures:
    """Test queries for different scenarios"""
    
    FUNCTIONAL_QUERIES = [
        # General queries
        ("What am I working on?", "current_work"),
        ("What's on my screen?", "workspace_overview"),
        ("Describe my workspace", "workspace_overview"),
        
        # Specific queries
        ("Do I have any messages?", "messages"),
        ("Are there any errors?", "errors"),
        ("Show me documentation", "documentation"),
        
        # Window queries
        ("What windows are open?", "windows"),
        ("Show me Chrome windows", "specific_app"),
        ("What's in my terminal?", "specific_app"),
        
        # Advanced queries
        ("Optimize my workspace", "optimize"),
        ("Prepare for meeting", "meeting"),
        ("What's my usual workflow?", "workflow"),
        ("Set privacy mode to meeting", "privacy"),
    ]
    
    EDGE_CASE_QUERIES = [
        ("", "empty"),  # Empty query
        ("Show me everything" * 100, "very_long"),  # Very long query
        ("何か開いてる？", "non_english"),  # Non-English
        ("SHOW ME ERRORS!!!", "shouting"),  # All caps with punctuation
        ("sh0w m3 w1nd0ws", "leetspeak"),  # Numbers in text
    ]


class TestTimer:
    """Context manager for timing operations"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        
    @property
    def duration_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class MockAPITracker:
    """Track API calls and costs for testing"""
    
    def __init__(self):
        self.calls = []
        self.total_cost = 0.0
        
    def track_call(self, endpoint: str, tokens: int, cost: float):
        """Track an API call"""
        self.calls.append({
            "endpoint": endpoint,
            "tokens": tokens,
            "cost": cost,
            "timestamp": datetime.now()
        })
        self.total_cost += cost
        
    def reset(self):
        """Reset tracking"""
        self.calls.clear()
        self.total_cost = 0.0
        
    def get_stats(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        return {
            "total_calls": len(self.calls),
            "total_cost": self.total_cost,
            "avg_cost_per_call": self.total_cost / len(self.calls) if self.calls else 0.0,
            "calls_by_endpoint": self._group_by_endpoint()
        }
        
    def _group_by_endpoint(self) -> Dict[str, int]:
        """Group calls by endpoint"""
        counts = {}
        for call in self.calls:
            endpoint = call["endpoint"]
            counts[endpoint] = counts.get(endpoint, 0) + 1
        return counts


async def run_test_with_timeout(test_func: Callable, timeout_seconds: float = 10.0) -> TestResult:
    """Run a test function with timeout"""
    try:
        result = await asyncio.wait_for(test_func(), timeout=timeout_seconds)
        return result
    except asyncio.TimeoutError:
        return TestResult(
            test_name=test_func.__name__,
            category="unknown",
            passed=False,
            duration_ms=timeout_seconds * 1000,
            error_message=f"Test timed out after {timeout_seconds} seconds"
        )


def generate_test_report(results: List[TestResult], output_file: Optional[str] = None) -> Dict[str, Any]:
    """Generate a test report from results"""
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    
    # Group by category
    by_category = {}
    for result in results:
        if result.category not in by_category:
            by_category[result.category] = {"passed": 0, "failed": 0, "tests": []}
        
        if result.passed:
            by_category[result.category]["passed"] += 1
        else:
            by_category[result.category]["failed"] += 1
        
        # Convert TestResult to dict for JSON serialization
        test_dict = {
            "name": result.test_name,
            "passed": result.passed,
            "duration_ms": result.duration_ms,
            "error": result.error_message,
            "metrics": result.metrics
        }
        by_category[result.category]["tests"].append(test_dict)
    
    # Calculate performance metrics
    response_times = [r.duration_ms for r in results]
    p95_response_time = np.percentile(response_times, 95) if response_times else 0
    
    report = {
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0,
            "execution_time_ms": sum(response_times),
            "timestamp": datetime.now().isoformat()
        },
        "by_category": by_category,
        "performance": {
            "avg_response_time_ms": np.mean(response_times) if response_times else 0,
            "p95_response_time_ms": p95_response_time,
            "max_response_time_ms": max(response_times) if response_times else 0,
            "min_response_time_ms": min(response_times) if response_times else 0
        },
        "failures": [
            {
                "test": r.test_name,
                "category": r.category,
                "error": r.error_message
            }
            for r in results if not r.passed
        ]
    }
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


def print_test_summary(report: Dict[str, Any]):
    """Print a formatted test summary"""
    print("\n" + "="*60)
    print("TEST EXECUTION SUMMARY")
    print("="*60)
    
    summary = report["summary"]
    print(f"\nTotal Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ({summary['pass_rate']:.1%})")
    print(f"Failed: {summary['failed']}")
    print(f"Total Time: {summary['execution_time_ms']:.0f}ms")
    
    print("\nBy Category:")
    for category, stats in report["by_category"].items():
        print(f"  {category}: {stats['passed']}/{stats['passed'] + stats['failed']} passed")
    
    perf = report["performance"]
    print(f"\nPerformance:")
    print(f"  Average Response: {perf['avg_response_time_ms']:.0f}ms")
    print(f"  P95 Response: {perf['p95_response_time_ms']:.0f}ms")
    print(f"  Max Response: {perf['max_response_time_ms']:.0f}ms")
    
    if report["failures"]:
        print(f"\nFailures ({len(report['failures'])}):")
        for failure in report["failures"][:5]:  # Show first 5
            print(f"  • {failure['test']} ({failure['category']}): {failure['error']}")
        if len(report["failures"]) > 5:
            print(f"  ... and {len(report['failures']) - 5} more")
    
    print("\n" + "="*60)