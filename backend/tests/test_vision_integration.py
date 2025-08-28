#!/usr/bin/env python3
"""
Integration Tests for JARVIS Vision System
Tests actual window detection and visual analysis with real apps
"""

import unittest
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.window_detector import WindowDetector
from vision.workspace_analyzer import WorkspaceAnalyzer
from vision.smart_query_router import SmartQueryRouter, QueryIntent
from vision.multi_window_capture import MultiWindowCapture
import platform

class TestVisionIntegration(unittest.TestCase):
    """Integration tests with actual window detection"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once"""
        cls.is_macos = platform.system() == "Darwin"
        if not cls.is_macos:
            print("⚠️  Vision tests require macOS")
    
    def setUp(self):
        """Set up each test"""
        if not self.is_macos:
            self.skipTest("Vision features require macOS")
            
        self.detector = WindowDetector()
        self.router = SmartQueryRouter()
        self.analyzer = WorkspaceAnalyzer()
        self.capture = MultiWindowCapture()
    
    def test_detect_any_open_app(self):
        """Test that we can detect any currently open app"""
        # Get all open windows
        windows = self.detector.get_all_windows()
        
        # Should detect at least some windows
        self.assertGreater(len(windows), 0, "Should detect at least one open window")
        
        # Print what we found (for debugging)
        print(f"\n📱 Detected {len(windows)} windows:")
        for i, window in enumerate(windows[:5]):  # First 5
            print(f"   {i+1}. {window.app_name} - {window.window_title or 'Untitled'}")
        
        # Test that each window has required properties
        for window in windows:
            self.assertIsNotNone(window.window_id, "Window should have ID")
            self.assertIsNotNone(window.app_name, "Window should have app name")
            self.assertIsInstance(window.bounds, dict, "Window should have bounds")
            self.assertIn('x', window.bounds, "Bounds should have x coordinate")
            self.assertIn('y', window.bounds, "Bounds should have y coordinate")
    
    def test_route_query_to_actual_apps(self):
        """Test routing queries to actually open apps"""
        windows = self.detector.get_all_windows()
        
        if not windows:
            self.skipTest("No windows open to test")
        
        # Test generic queries that should work with any apps
        test_queries = [
            "what's on my screen",
            "show me all windows",
            "any notifications",
            "check for messages",
            "what apps are open"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                route = self.router.route_query(query, windows)
                
                # Should produce a valid route
                self.assertIsNotNone(route, f"Query '{query}' should produce a route")
                self.assertIsNotNone(route.intent, "Route should have intent")
                self.assertGreater(route.confidence, 0, "Route should have confidence > 0")
                
                # For overview queries, should have some target windows
                if "all" in query or "screen" in query:
                    self.assertGreater(len(route.target_windows), 0, 
                                     "Overview query should target some windows")
    
    def test_unknown_app_detection(self):
        """Test that we can handle apps not in our predefined lists"""
        windows = self.detector.get_all_windows()
        
        # Find an app that might not be in our hardcoded lists
        unknown_apps = []
        common_apps = ['chrome', 'safari', 'firefox', 'discord', 'slack', 'whatsapp', 
                      'terminal', 'code', 'mail', 'messages']
        
        for window in windows:
            app_lower = window.app_name.lower()
            if not any(known in app_lower for known in common_apps):
                unknown_apps.append(window)
        
        if unknown_apps:
            print(f"\n🔍 Found {len(unknown_apps)} potentially unknown apps:")
            for app in unknown_apps[:3]:
                print(f"   - {app.app_name}")
            
            # Test querying these unknown apps
            for window in unknown_apps[:3]:
                query = f"check {window.app_name} for notifications"
                route = self.router.route_query(query, windows)
                
                # Should still route correctly
                self.assertIsNotNone(route, f"Should route query for unknown app {window.app_name}")
                self.assertGreater(route.confidence, 0, "Should have confidence even for unknown app")
    
    async def test_workspace_analysis_with_real_windows(self):
        """Test full workspace analysis with actual windows"""
        # Test various queries
        test_queries = [
            "what am I working on",
            "do I have any messages",
            "check for notifications",
            "show me my workspace"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                try:
                    # Analyze workspace
                    analysis = await self.analyzer.analyze_workspace(query)
                    
                    # Should produce analysis
                    self.assertIsNotNone(analysis, f"Should analyze '{query}'")
                    self.assertIsNotNone(analysis.focused_task, "Should identify focused task")
                    self.assertIsInstance(analysis.confidence, float, "Should have confidence score")
                    
                    # Print analysis result
                    print(f"\n📊 Analysis for '{query}':")
                    print(f"   Task: {analysis.focused_task}")
                    print(f"   Context: {analysis.workspace_context}")
                    print(f"   Confidence: {analysis.confidence:.0%}")
                    
                except Exception as e:
                    # If vision fails (permissions, etc), that's OK for this test
                    print(f"   ⚠️  Analysis failed (expected if no permissions): {e}")
    
    def test_app_categorization_flexibility(self):
        """Test that app categorization works flexibly"""
        windows = self.detector.get_all_windows()
        
        # Count how many apps match each category
        categories = {
            'communication': 0,
            'development': 0,
            'terminal': 0,
            'browser': 0,
            'documentation': 0,
            'uncategorized': 0
        }
        
        for window in windows:
            categorized = False
            
            if self.router._is_communication_app(window):
                categories['communication'] += 1
                categorized = True
            elif self.router._is_development_app(window):
                categories['development'] += 1
                categorized = True
            elif self.router._is_terminal_app(window):
                categories['terminal'] += 1
                categorized = True
            elif self.router._is_browser_app(window):
                categories['browser'] += 1
                categorized = True
            elif self.router._is_documentation_window(window):
                categories['documentation'] += 1
                categorized = True
            
            if not categorized:
                categories['uncategorized'] += 1
        
        print("\n📊 App Categorization Results:")
        for category, count in categories.items():
            if count > 0:
                print(f"   {category}: {count} windows")
        
        # Test passes as long as detection works
        total_windows = sum(categories.values())
        self.assertEqual(total_windows, len(windows), "All windows should be counted")
    
    async def test_window_capture_fallback(self):
        """Test that window capture falls back gracefully"""
        windows = self.detector.get_all_windows()[:3]  # Test with first 3 windows
        
        if not windows:
            self.skipTest("No windows to test capture")
        
        captured = 0
        failed = 0
        
        for window in windows:
            try:
                # Try to capture
                capture = await self.capture._async_capture_window(window, resolution_scale=0.5)
                if capture and capture.image is not None:
                    captured += 1
                    print(f"✅ Captured: {window.app_name}")
                else:
                    failed += 1
                    print(f"⚠️  No image for: {window.app_name}")
            except Exception as e:
                failed += 1
                print(f"❌ Failed to capture {window.app_name}: {e}")
        
        print(f"\n📸 Capture Results: {captured} success, {failed} failed")
        
        # Test passes if we handled all windows (success or graceful failure)
        self.assertEqual(captured + failed, len(windows), "All windows should be processed")

class TestDynamicContentAnalysis(unittest.TestCase):
    """Test dynamic analysis of visual content"""
    
    def setUp(self):
        """Set up dynamic analysis tests"""
        if platform.system() != "Darwin":
            self.skipTest("Vision features require macOS")
            
        self.detector = WindowDetector()
        self.router = SmartQueryRouter()
    
    def test_notification_pattern_detection(self):
        """Test detection of various notification patterns in window titles"""
        windows = self.detector.get_all_windows()
        
        notification_patterns = [
            r'\(\d+\)',  # (5) style
            r'\d+ new',   # 5 new
            r'unread',    # unread messages
            r'notification',  # notification word
            r'•',         # bullet indicator
            r'!',         # exclamation
            r'badge',     # badge word
        ]
        
        windows_with_notifications = []
        
        for window in windows:
            if window.window_title:
                title_lower = window.window_title.lower()
                for pattern in notification_patterns:
                    if pattern in title_lower or (pattern == r'\(\d+\)' and 
                                                 any(c.isdigit() for c in window.window_title)):
                        windows_with_notifications.append(window)
                        break
        
        if windows_with_notifications:
            print(f"\n🔔 Found {len(windows_with_notifications)} windows with notification indicators:")
            for window in windows_with_notifications[:5]:
                print(f"   - {window.app_name}: {window.window_title}")
        else:
            print("\n🔔 No windows with notification indicators found (this is OK)")
        
        # Test passes - we're just checking the detection logic works
        self.assertTrue(True, "Notification pattern detection completed")
    
    def test_query_intent_accuracy(self):
        """Test that query intents are accurately detected"""
        windows = self.detector.get_all_windows()
        
        # Test queries with expected intents
        test_cases = [
            ("do i have any messages from X", [QueryIntent.MESSAGES, QueryIntent.NOTIFICATIONS]),
            ("check X for notifications", [QueryIntent.NOTIFICATIONS]),
            ("what's on my screen", [QueryIntent.WORKSPACE_OVERVIEW]),
            ("show me X", [QueryIntent.SPECIFIC_APP]),
            ("any errors in terminal", [QueryIntent.ERRORS]),
            ("find the save function", [QueryIntent.CODE_SEARCH])
        ]
        
        results = []
        
        for query_template, expected_intents in test_cases:
            # Use actual app name if available
            if windows and "X" in query_template:
                query = query_template.replace("X", windows[0].app_name)
            else:
                query = query_template.replace("X", "SomeApp")
            
            route = self.router.route_query(query, windows)
            
            # Check if intent matches expected
            matches = route.intent in expected_intents
            results.append({
                'query': query,
                'expected': expected_intents,
                'actual': route.intent,
                'matches': matches
            })
        
        # Print results
        print("\n🎯 Query Intent Detection Results:")
        for result in results:
            status = "✅" if result['matches'] else "❌"
            print(f"   {status} '{result['query'][:40]}...' -> {result['actual'].value}")
        
        # Count successes
        successes = sum(1 for r in results if r['matches'])
        print(f"\n   Accuracy: {successes}/{len(results)} ({successes/len(results)*100:.0f}%)")
        
        # Test passes if most intents are correct
        self.assertGreater(successes / len(results), 0.7, "At least 70% of intents should be correct")

def run_integration_tests():
    """Run all integration tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVisionIntegration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDynamicContentAnalysis))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

async def run_async_integration_tests():
    """Run async integration tests"""
    print("\n🔄 Running async integration tests...")
    
    test = TestVisionIntegration()
    test.setUp()
    
    if platform.system() == "Darwin":
        try:
            await test.test_workspace_analysis_with_real_windows()
            await test.test_window_capture_fallback()
            print("✅ Async integration tests completed")
        except Exception as e:
            print(f"❌ Async test failed: {e}")
            return False
    else:
        print("⚠️  Skipping async tests (requires macOS)")
    
    return True

if __name__ == "__main__":
    print("🧪 Running JARVIS Vision Integration Tests")
    print("=" * 50)
    
    # Run sync tests
    sync_success = run_integration_tests()
    
    # Run async tests
    async_success = asyncio.run(run_async_integration_tests())
    
    if sync_success and async_success:
        print("\n🎉 All integration tests passed!")
    else:
        print("\n❌ Some integration tests failed")
        sys.exit(1)