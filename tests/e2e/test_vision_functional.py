#!/usr/bin/env python3
"""
Functional Testing Suite for JARVIS Multi-Window Intelligence
Tests all functional requirements from the PRD
"""

import asyncio
import sys
import os
from typing import List, Dict, Any

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.vision.jarvis_workspace_integration import JARVISWorkspaceIntelligence
from backend.vision.window_detector import WindowDetector
from backend.vision.privacy_controls import PrivacyControlSystem
from backend.vision.meeting_preparation import MeetingPreparationSystem
from backend.vision.workflow_learning import WorkflowLearningSystem

from test_utils import (
    TestResult, WindowFixtures, QueryFixtures, TestTimer,
    run_test_with_timeout, generate_test_report, print_test_summary
)


class FunctionalTestSuite:
    """Functional testing for JARVIS workspace intelligence"""
    
    def __init__(self):
        self.jarvis = JARVISWorkspaceIntelligence()
        self.results: List[TestResult] = []
        
    async def run_all_tests(self) -> List[TestResult]:
        """Run all functional tests"""
        print("\n🧪 FUNCTIONAL TEST SUITE")
        print("="*60)
        
        # Test categories
        await self.test_single_window_analysis()
        await self.test_multi_window_analysis()
        await self.test_edge_cases()
        await self.test_privacy_controls()
        await self.test_query_types()
        await self.test_meeting_preparation()
        await self.test_workflow_learning()
        
        return self.results
    
    async def test_single_window_analysis(self):
        """Test single window analysis functionality"""
        print("\n📝 Testing Single Window Analysis...")
        
        # Mock single window
        self.jarvis.window_detector.get_all_windows = lambda: WindowFixtures.single_window()
        
        with TestTimer() as timer:
            try:
                response = await self.jarvis.handle_workspace_command("What am I working on?")
                
                # Verify response
                passed = all([
                    isinstance(response, str),
                    len(response) > 0,
                    "Visual Studio Code" in response or "working" in response.lower()
                ])
                
                self.results.append(TestResult(
                    test_name="single_window_basic",
                    category="functional",
                    passed=passed,
                    duration_ms=timer.duration_ms,
                    error_message=None if passed else "Response validation failed"
                ))
                
            except Exception as e:
                self.results.append(TestResult(
                    test_name="single_window_basic",
                    category="functional",
                    passed=False,
                    duration_ms=timer.duration_ms,
                    error_message=str(e)
                ))
    
    async def test_multi_window_analysis(self):
        """Test multi-window analysis (2-10 windows)"""
        print("\n📝 Testing Multi-Window Analysis...")
        
        # Test with development setup (3 windows)
        self.jarvis.window_detector.get_all_windows = lambda: WindowFixtures.development_setup()
        
        test_cases = [
            ("What am I working on?", ["Visual Studio Code", "Terminal"]),
            ("Do I have any documentation open?", ["Chrome", "Python"]),
            ("What's on my screen?", ["Visual Studio Code", "Terminal", "Chrome"])
        ]
        
        for query, expected_mentions in test_cases:
            with TestTimer() as timer:
                try:
                    response = await self.jarvis.handle_workspace_command(query)
                    
                    # Check if expected apps are mentioned
                    passed = any(app.lower() in response.lower() for app in expected_mentions)
                    
                    self.results.append(TestResult(
                        test_name=f"multi_window_{query[:20]}",
                        category="functional",
                        passed=passed,
                        duration_ms=timer.duration_ms,
                        metrics={"windows_count": 3, "query": query}
                    ))
                    
                except Exception as e:
                    self.results.append(TestResult(
                        test_name=f"multi_window_{query[:20]}",
                        category="functional",
                        passed=False,
                        duration_ms=timer.duration_ms,
                        error_message=str(e)
                    ))
    
    async def test_edge_cases(self):
        """Test edge cases (20+ windows, minimized, full-screen)"""
        print("\n📝 Testing Edge Cases...")
        
        # Test with many windows
        self.jarvis.window_detector.get_all_windows = lambda: WindowFixtures.edge_case_many_windows(25)
        
        edge_cases = [
            ("many_windows", "What's on my screen?", 25),
            ("empty_query", "", 25),
            ("very_long_query", "Show me " * 50 + "everything", 25)
        ]
        
        for test_name, query, window_count in edge_cases:
            with TestTimer() as timer:
                try:
                    response = await self.jarvis.handle_workspace_command(query)
                    
                    # Should handle gracefully
                    passed = isinstance(response, str) and len(response) > 0
                    
                    self.results.append(TestResult(
                        test_name=f"edge_case_{test_name}",
                        category="functional",
                        passed=passed,
                        duration_ms=timer.duration_ms,
                        metrics={"windows_count": window_count}
                    ))
                    
                except Exception as e:
                    # Some edge cases might be expected to fail gracefully
                    self.results.append(TestResult(
                        test_name=f"edge_case_{test_name}",
                        category="functional",
                        passed=False,
                        duration_ms=timer.duration_ms,
                        error_message=str(e)
                    ))
    
    async def test_privacy_controls(self):
        """Test privacy controls and blacklist enforcement"""
        print("\n📝 Testing Privacy Controls...")
        
        # Set up privacy sensitive windows
        self.jarvis.window_detector.get_all_windows = lambda: WindowFixtures.privacy_sensitive()
        
        # Test privacy mode activation
        privacy_tests = [
            ("Set privacy mode to meeting", "meeting", "meeting privacy mode"),
            ("Private mode", "private", "maximum privacy"),
            ("Check privacy", "normal", "privacy mode")
        ]
        
        for query, expected_mode, expected_response in privacy_tests:
            with TestTimer() as timer:
                try:
                    response = await self.jarvis.handle_workspace_command(query)
                    
                    # Check response mentions privacy
                    passed = "privacy" in response.lower()
                    
                    # Verify mode if applicable
                    if expected_mode != "normal":
                        current_mode = self.jarvis.privacy_controls.current_mode
                        passed = passed and (current_mode == expected_mode or expected_response in response.lower())
                    
                    self.results.append(TestResult(
                        test_name=f"privacy_{expected_mode}",
                        category="functional",
                        passed=passed,
                        duration_ms=timer.duration_ms,
                        metrics={"mode": expected_mode}
                    ))
                    
                except Exception as e:
                    self.results.append(TestResult(
                        test_name=f"privacy_{expected_mode}",
                        category="functional",
                        passed=False,
                        duration_ms=timer.duration_ms,
                        error_message=str(e)
                    ))
        
        # Test sensitive window detection
        with TestTimer() as timer:
            try:
                # Check if sensitive windows are detected
                sensitive = self.jarvis.privacy_controls.detect_sensitive_content(
                    WindowFixtures.privacy_sensitive()
                )
                
                # Should detect at least password manager and bank
                passed = len(sensitive) >= 2
                
                self.results.append(TestResult(
                    test_name="privacy_sensitive_detection",
                    category="functional",
                    passed=passed,
                    duration_ms=timer.duration_ms,
                    metrics={"sensitive_count": len(sensitive)}
                ))
                
            except Exception as e:
                self.results.append(TestResult(
                    test_name="privacy_sensitive_detection",
                    category="functional",
                    passed=False,
                    duration_ms=timer.duration_ms,
                    error_message=str(e)
                ))
    
    async def test_query_types(self):
        """Test different query types"""
        print("\n📝 Testing Query Types...")
        
        # Use communication setup for message queries
        self.jarvis.window_detector.get_all_windows = lambda: WindowFixtures.communication_heavy()
        
        for query, query_type in QueryFixtures.FUNCTIONAL_QUERIES:
            with TestTimer() as timer:
                try:
                    response = await self.jarvis.handle_workspace_command(query)
                    
                    # Basic validation
                    passed = isinstance(response, str) and len(response) > 0
                    
                    # Type-specific validation
                    if query_type == "messages" and passed:
                        passed = any(app in response for app in ["Discord", "Slack", "Messages", "Mail"])
                    elif query_type == "optimize" and passed:
                        passed = any(word in response.lower() for word in ["layout", "arrange", "productivity"])
                    
                    self.results.append(TestResult(
                        test_name=f"query_type_{query_type}",
                        category="functional",
                        passed=passed,
                        duration_ms=timer.duration_ms,
                        metrics={"query": query, "type": query_type}
                    ))
                    
                except Exception as e:
                    self.results.append(TestResult(
                        test_name=f"query_type_{query_type}",
                        category="functional",
                        passed=False,
                        duration_ms=timer.duration_ms,
                        error_message=str(e)
                    ))
    
    async def test_meeting_preparation(self):
        """Test meeting preparation functionality"""
        print("\n📝 Testing Meeting Preparation...")
        
        # Set up meeting windows
        self.jarvis.window_detector.get_all_windows = lambda: WindowFixtures.meeting_setup()
        
        with TestTimer() as timer:
            try:
                response = await self.jarvis.handle_workspace_command("Prepare for meeting")
                
                # Should mention Zoom and sensitive windows
                passed = all([
                    "Zoom" in response,
                    "sensitive" in response.lower() or "1Password" in response
                ])
                
                self.results.append(TestResult(
                    test_name="meeting_preparation",
                    category="functional",
                    passed=passed,
                    duration_ms=timer.duration_ms,
                    metrics={"response_length": len(response)}
                ))
                
            except Exception as e:
                self.results.append(TestResult(
                    test_name="meeting_preparation",
                    category="functional",
                    passed=False,
                    duration_ms=timer.duration_ms,
                    error_message=str(e)
                ))
        
        # Test meeting layout
        with TestTimer() as timer:
            try:
                meeting_system = MeetingPreparationSystem()
                context, alerts = meeting_system.analyze_meeting_preparation(
                    WindowFixtures.meeting_setup()
                )
                
                layout = meeting_system.get_meeting_layout(context)
                
                passed = layout is not None and layout.layout_type in [
                    "presentation_mode", "collaboration_mode", "meeting_focus"
                ]
                
                self.results.append(TestResult(
                    test_name="meeting_layout_generation",
                    category="functional",
                    passed=passed,
                    duration_ms=timer.duration_ms,
                    metrics={"layout_type": layout.layout_type if layout else None}
                ))
                
            except Exception as e:
                self.results.append(TestResult(
                    test_name="meeting_layout_generation",
                    category="functional",
                    passed=False,
                    duration_ms=timer.duration_ms,
                    error_message=str(e)
                ))
    
    async def test_workflow_learning(self):
        """Test workflow learning functionality"""
        print("\n📝 Testing Workflow Learning...")
        
        # Test workflow recording
        with TestTimer() as timer:
            try:
                workflow_system = WorkflowLearningSystem()
                
                # Record a few states
                for _ in range(3):
                    workflow_system.record_window_state(WindowFixtures.development_setup())
                
                # Get predictions
                predictions = workflow_system.predict_workflow(WindowFixtures.single_window())
                
                # Should work without errors (might not have predictions yet)
                passed = True  # If no exception, it's working
                
                self.results.append(TestResult(
                    test_name="workflow_recording",
                    category="functional",
                    passed=passed,
                    duration_ms=timer.duration_ms,
                    metrics={"predictions_count": len(predictions)}
                ))
                
            except Exception as e:
                self.results.append(TestResult(
                    test_name="workflow_recording",
                    category="functional",
                    passed=False,
                    duration_ms=timer.duration_ms,
                    error_message=str(e)
                ))
        
        # Test workflow insights
        with TestTimer() as timer:
            try:
                response = await self.jarvis.handle_workspace_command("What's my usual workflow?")
                
                # Should provide some response about workflow
                passed = "workflow" in response.lower() or "pattern" in response.lower()
                
                self.results.append(TestResult(
                    test_name="workflow_insights_query",
                    category="functional",
                    passed=passed,
                    duration_ms=timer.duration_ms
                ))
                
            except Exception as e:
                self.results.append(TestResult(
                    test_name="workflow_insights_query",
                    category="functional",
                    passed=False,
                    duration_ms=timer.duration_ms,
                    error_message=str(e)
                ))


async def main():
    """Run functional test suite"""
    suite = FunctionalTestSuite()
    results = await suite.run_all_tests()
    
    # Generate report
    report = generate_test_report(results, "functional_test_report.json")
    print_test_summary(report)
    
    # Check success criteria
    success_criteria = {
        "coverage": len(results) >= 20,  # At least 20 tests
        "pass_rate": report["summary"]["pass_rate"] >= 0.9,  # 90% pass rate
        "no_critical_failures": all(r.passed for r in results if "basic" in r.test_name)
    }
    
    print("\n✅ Success Criteria:")
    for criterion, met in success_criteria.items():
        status = "PASS" if met else "FAIL"
        print(f"  {criterion}: {status}")
    
    return all(success_criteria.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)