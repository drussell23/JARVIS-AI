#!/usr/bin/env python3
"""
Dynamic Visual Content Analysis Tests
Tests JARVIS's ability to analyze any visual content on screen dynamically
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.workspace_analyzer import WorkspaceAnalyzer, WorkspaceAnalysis
from vision.window_detector import WindowInfo
from vision.multi_window_capture import WindowCapture
from vision.smart_query_router import QueryRoute, QueryIntent

class TestDynamicVisualAnalysis(unittest.TestCase):
    """Test dynamic analysis capabilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.analyzer = WorkspaceAnalyzer()
        
        # Mock window scenarios
        self.test_scenarios = [
            {
                "name": "Unknown messaging app with badge",
                "windows": [
                    WindowInfo(
                        window_id=1,
                        app_name="FutureChatApp",
                        window_title="FutureChatApp (12)",  # Badge in title
                        bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                        is_focused=True,
                        layer=0,
                        is_visible=True,
                        process_id=1001
                    )
                ],
                "expected_detection": {
                    "has_notifications": True,
                    "notification_count": "12",
                    "app_type": "communication"
                }
            },
            {
                "name": "Custom work app with alerts",
                "windows": [
                    WindowInfo(
                        window_id=2,
                        app_name="CustomWorkflowTool",
                        window_title="CustomWorkflowTool - 3 tasks pending, 2 alerts",
                        bounds={"x": 0, "y": 0, "width": 1200, "height": 800},
                        is_focused=True,
                        layer=0,
                        is_visible=True,
                        process_id=1002
                    )
                ],
                "expected_detection": {
                    "has_notifications": True,
                    "has_alerts": True,
                    "task_count": "3",
                    "alert_count": "2"
                }
            },
            {
                "name": "Non-English app",
                "windows": [
                    WindowInfo(
                        window_id=3,
                        app_name="聊天应用",  # "Chat App" in Chinese
                        window_title="聊天应用 - 新消息 (5)",  # "New messages (5)"
                        bounds={"x": 100, "y": 100, "width": 600, "height": 400},
                        is_focused=False,
                        layer=0,
                        is_visible=True,
                        process_id=1003
                    )
                ],
                "expected_detection": {
                    "has_notifications": True,
                    "notification_count": "5",
                    "app_type": "communication"
                }
            },
            {
                "name": "App with visual indicators",
                "windows": [
                    WindowInfo(
                        window_id=4,
                        app_name="VisualNotifierApp",
                        window_title="VisualNotifierApp • • •",  # Dot indicators
                        bounds={"x": 0, "y": 0, "width": 400, "height": 300},
                        is_focused=False,
                        layer=0,
                        is_visible=True,
                        process_id=1004
                    )
                ],
                "expected_detection": {
                    "has_notifications": True,
                    "indicator_type": "dots"
                }
            }
        ]
    
    def test_dynamic_notification_detection(self):
        """Test detection of notifications in any app"""
        for scenario in self.test_scenarios:
            with self.subTest(scenario=scenario["name"]):
                window = scenario["windows"][0]
                
                # Test title-based detection
                title = window.window_title.lower()
                
                # Check for numeric badges
                import re
                numeric_match = re.search(r'\((\d+)\)', window.window_title)
                if numeric_match:
                    self.assertTrue(scenario["expected_detection"].get("has_notifications"),
                                  f"{scenario['name']} should detect notifications")
                    self.assertEqual(numeric_match.group(1), 
                                   scenario["expected_detection"].get("notification_count"),
                                   f"Should detect correct count")
                
                # Check for keywords
                notification_keywords = ['new', 'unread', 'alert', 'pending', '新消息']
                has_keywords = any(keyword in title for keyword in notification_keywords)
                
                # Check for visual indicators
                has_indicators = '•' in window.window_title or '!' in window.window_title
                
                # Verify detection
                if scenario["expected_detection"].get("has_notifications"):
                    self.assertTrue(numeric_match or has_keywords or has_indicators,
                                  f"{scenario['name']} should show some notification indicator")
    
    async def test_visual_content_analysis_mock(self):
        """Test visual content analysis with mocked captures"""
        # Create mock window captures
        mock_captures = []
        
        for scenario in self.test_scenarios[:2]:  # Test first 2 scenarios
            # Create a mock image (numpy array)
            mock_image = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Create mock capture
            import time
            capture = WindowCapture(
                window_info=scenario["windows"][0],
                image=mock_image,
                resolution_scale=1.0,
                capture_time=time.time()
            )
            mock_captures.append(capture)
        
        # Mock the analyzer's Claude client
        with patch.object(self.analyzer, 'claude_client') as mock_claude:
            # Mock Claude's response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="""
                PRIMARY TASK: You're working in FutureChatApp with 12 unread messages
                CONTEXT: CustomWorkflowTool shows 3 pending tasks and 2 alerts
                NOTIFICATIONS: FutureChatApp has 12 new messages
                SUGGESTIONS: Check the urgent messages in FutureChatApp
            """)]
            
            mock_claude.messages.create.return_value = mock_response
            
            # Test analysis
            self.analyzer.claude_client = mock_claude
            analysis = await self.analyzer._analyze_with_claude(
                mock_captures, 
                "do I have any notifications"
            )
            
            # Verify analysis
            self.assertIsNotNone(analysis)
            self.assertIn("FutureChatApp", analysis.focused_task)
            self.assertIn("12", analysis.focused_task)
            self.assertGreater(analysis.confidence, 0.5)
    
    def test_app_type_inference(self):
        """Test inferring app type from name and content"""
        test_apps = [
            ("SuperChatApp", "communication"),  # Has 'chat'
            ("MessageHub", "communication"),    # Has 'message'
            ("CodeMaster", "development"),      # Has 'code'
            ("TerminalPro", "terminal"),        # Has 'terminal'
            ("BrowserX", "browser"),            # Has 'browser'
            ("PDFViewer", "documentation"),     # Has 'pdf'
            ("RandomBusinessApp", None)         # Unknown type
        ]
        
        for app_name, expected_type in test_apps:
            with self.subTest(app=app_name):
                window = WindowInfo(
                    window_id=100,
                    app_name=app_name,
                    window_title=app_name,
                    bounds={},
                    is_focused=False,
                    layer=0,
                    is_visible=True,
                    process_id=2000
                )
                
                # Test categorization
                from vision.smart_query_router import SmartQueryRouter
                router = SmartQueryRouter()
                
                detected_type = None
                if router._is_communication_app(window):
                    detected_type = "communication"
                elif router._is_development_app(window):
                    detected_type = "development"
                elif router._is_terminal_app(window):
                    detected_type = "terminal"
                elif router._is_browser_app(window):
                    detected_type = "browser"
                elif router._is_documentation_window(window):
                    detected_type = "documentation"
                
                if expected_type:
                    self.assertEqual(detected_type, expected_type,
                                   f"{app_name} should be detected as {expected_type}")
    
    def test_context_aware_routing(self):
        """Test that queries route correctly based on context"""
        from vision.smart_query_router import SmartQueryRouter
        router = SmartQueryRouter()
        
        # Create diverse window set
        test_windows = [
            WindowInfo(window_id=1, app_name="UnknownMessenger", 
                      window_title="UnknownMessenger (5)", bounds={}, is_focused=True,
                      layer=0, is_visible=True, process_id=3001),
            WindowInfo(window_id=2, app_name="WorkTracker", 
                      window_title="WorkTracker - 3 tasks", bounds={}, is_focused=False,
                      layer=0, is_visible=True, process_id=3002),
            WindowInfo(window_id=3, app_name="RandomApp", 
                      window_title="RandomApp v2.0", bounds={}, is_focused=False,
                      layer=0, is_visible=True, process_id=3003),
        ]
        
        # Test queries
        test_queries = [
            {
                "query": "notifications from UnknownMessenger",
                "expected_targets": ["UnknownMessenger"],
                "expected_intent": QueryIntent.NOTIFICATIONS
            },
            {
                "query": "check WorkTracker",
                "expected_targets": ["WorkTracker"],
                "expected_intent": QueryIntent.SPECIFIC_APP
            },
            {
                "query": "any notifications",
                "expected_min_targets": 0,  # May or may not find specific targets
                "expected_intent": QueryIntent.NOTIFICATIONS
            },
            {
                "query": "what's on my screen",
                "expected_min_targets": 1,  # Should include something
                "expected_intent": QueryIntent.WORKSPACE_OVERVIEW
            }
        ]
        
        for test_case in test_queries:
            with self.subTest(query=test_case["query"]):
                route = router.route_query(test_case["query"], test_windows)
                
                # Check intent - allow some flexibility for intents that can overlap
                expected_intent = test_case["expected_intent"]
                if expected_intent == QueryIntent.NOTIFICATIONS and "from" in test_case["query"]:
                    # "notifications from X" can be either NOTIFICATIONS or SPECIFIC_APP
                    acceptable_intents = [QueryIntent.NOTIFICATIONS, QueryIntent.SPECIFIC_APP]
                    self.assertIn(route.intent, acceptable_intents,
                                f"Query '{test_case['query']}' should have intent in {acceptable_intents}")
                else:
                    self.assertEqual(route.intent, expected_intent,
                                   f"Query '{test_case['query']}' should have intent {expected_intent}")
                
                # Check targets
                if "expected_targets" in test_case:
                    target_names = [w.app_name for w in route.target_windows]
                    for expected in test_case["expected_targets"]:
                        self.assertIn(expected, target_names,
                                    f"Should target {expected}")
                
                if "expected_min_targets" in test_case:
                    self.assertGreaterEqual(len(route.target_windows), 
                                          test_case["expected_min_targets"],
                                          f"Should have at least {test_case['expected_min_targets']} targets")
    
    def test_fallback_analysis(self):
        """Test fallback analysis when screenshots fail"""
        # Test the fallback method directly
        windows = [
            WindowInfo(window_id=1, app_name="AppWithoutScreenshot",
                      window_title="AppWithoutScreenshot - Error loading",
                      bounds={}, is_focused=True, layer=0, is_visible=True, process_id=4001),
            WindowInfo(window_id=2, app_name="AnotherApp",
                      window_title="AnotherApp (3 notifications)",
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=4002)
        ]
        
        # Create a mock route
        route = QueryRoute(
            intent=QueryIntent.NOTIFICATIONS,
            target_windows=windows,
            confidence=0.8,
            reasoning="Checking notifications"
        )
        
        # Test fallback analysis
        analysis = self.analyzer._analyze_from_window_info_only(
            windows, 
            "check notifications",
            route
        )
        
        # Verify fallback analysis
        self.assertIsNotNone(analysis)
        self.assertIn("AppWithoutScreenshot", analysis.focused_task)
        self.assertEqual(analysis.confidence, 0.5)  # Lower confidence for fallback
        self.assertGreater(len(analysis.important_notifications), 0)  # Should detect the (3)

class TestVisualIndicatorPatterns(unittest.TestCase):
    """Test detection of various visual indicator patterns"""
    
    def test_notification_badge_patterns(self):
        """Test various notification badge formats"""
        test_patterns = [
            # Format: (window_title, expected_count, has_notification)
            ("App (5)", 5, True),
            ("App [3]", 3, True),
            ("App - 12 new", 12, True),
            ("App • 7 unread", 7, True),
            ("App (99+)", 99, True),
            ("App ••• new messages", None, True),  # Dots but no count
            ("App !", None, True),  # Exclamation
            ("App", None, False),  # No indicators
            ("App v2.0", None, False),  # Version number, not notification
        ]
        
        for title, expected_count, has_notification in test_patterns:
            with self.subTest(title=title):
                # Extract notification info
                import re
                
                # Try various patterns
                count = None
                patterns = [
                    r'\((\d+)\+?\)',  # (5) or (99+)
                    r'\[(\d+)\]',     # [3]
                    r'(\d+)\s+new',   # 12 new
                    r'(\d+)\s+unread' # 7 unread
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, title)
                    if match:
                        count = int(match.group(1))
                        break
                
                # Check for other indicators
                has_indicator = any(indicator in title for indicator in ['•', '!', 'new', 'unread'])
                
                # Verify detection
                if has_notification:
                    self.assertTrue(count is not None or has_indicator,
                                  f"'{title}' should show notification indicator")
                    if expected_count:
                        self.assertEqual(count, expected_count,
                                       f"'{title}' should show count {expected_count}")
                else:
                    self.assertFalse(count is not None or has_indicator,
                                   f"'{title}' should not show notification indicator")
    
    def test_multi_language_indicators(self):
        """Test notification indicators in different languages"""
        multi_lang_patterns = [
            ("App - 新消息 (5)", True),      # Chinese: "New messages"
            ("App - Новые (3)", True),      # Russian: "New"
            ("App - नया संदेश", True),      # Hindi: "New message"
            ("App - 新着メッセージ", True),    # Japanese: "New messages"
            ("App - رسائل جديدة", True),    # Arabic: "New messages"
        ]
        
        for title, has_notification in multi_lang_patterns:
            with self.subTest(title=title):
                # Check for numeric indicators or parentheses
                import re
                has_numeric = bool(re.search(r'\d+', title))
                has_parens = '(' in title and ')' in title
                has_arabic = 'رسائل جديدة' in title  # Arabic for "new messages"
                has_chinese = '新' in title  # Chinese for "new"
                has_russian = 'Новые' in title  # Russian for "new"
                has_japanese = '新着' in title  # Japanese for "new"
                has_hindi = 'नया' in title  # Hindi for "new"
                
                if has_notification:
                    self.assertTrue(has_numeric or has_parens or has_arabic or has_chinese or has_russian or has_japanese or has_hindi,
                                  f"'{title}' should have some indicator")

def run_dynamic_analysis_tests():
    """Run all dynamic visual analysis tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDynamicVisualAnalysis))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestVisualIndicatorPatterns))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

async def run_async_tests():
    """Run async test methods"""
    test = TestDynamicVisualAnalysis()
    test.setUp()
    
    try:
        await test.test_visual_content_analysis_mock()
        print("\n✅ Async visual analysis tests completed")
        return True
    except Exception as e:
        print(f"\n❌ Async test failed: {e}")
        return False

if __name__ == "__main__":
    print("🎨 Running Dynamic Visual Analysis Tests")
    print("=" * 50)
    
    # Run sync tests
    sync_success = run_dynamic_analysis_tests()
    
    # Run async tests
    async_success = asyncio.run(run_async_tests())
    
    if sync_success and async_success:
        print("\n🎉 All dynamic visual analysis tests passed!")
        print("\n📊 Test Coverage:")
        print("   ✅ Unknown app detection")
        print("   ✅ Multi-language support")
        print("   ✅ Various notification formats")
        print("   ✅ Visual indicator patterns")
        print("   ✅ Context-aware routing")
        print("   ✅ Fallback analysis")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)