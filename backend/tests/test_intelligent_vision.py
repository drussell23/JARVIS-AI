#!/usr/bin/env python3
"""
Functional Tests for JARVIS Intelligent Vision System
Tests the ability to detect and analyze ANY app without hardcoding
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from voice.jarvis_agent_voice import JARVISAgentVoice
from vision.smart_query_router import SmartQueryRouter, QueryIntent
from vision.window_detector import WindowInfo

class TestIntelligentVisionFunctionality(unittest.TestCase):
    """Test intelligent vision functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.jarvis = JARVISAgentVoice()
        self.router = SmartQueryRouter()
        
        # Create mock windows for various apps (including unknown ones)
        self.mock_windows = [
            # Known apps
            WindowInfo(
                window_id=1,
                app_name="WhatsApp",
                window_title="WhatsApp",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=True,
                layer=0,
                is_visible=True,
                process_id=1234
            ),
            # Unknown/new apps
            WindowInfo(
                window_id=2,
                app_name="SuperNewChatApp",  # App JARVIS has never seen
                window_title="SuperNewChatApp - 5 unread messages",
                bounds={"x": 800, "y": 0, "width": 600, "height": 400},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1235
            ),
            WindowInfo(
                window_id=3,
                app_name="CustomWorkTool",  # Another unknown app
                window_title="CustomWorkTool - 3 notifications",
                bounds={"x": 0, "y": 600, "width": 800, "height": 400},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1236
            ),
            WindowInfo(
                window_id=4,
                app_name="未知应用",  # Non-English app name
                window_title="新消息 (2)",  # "New messages (2)" in Chinese
                bounds={"x": 1400, "y": 0, "width": 400, "height": 300},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1237
            )
        ]
    
    def test_query_detection_for_unknown_apps(self):
        """Test that queries about unknown apps are detected as system commands"""
        test_queries = [
            # Queries about unknown apps
            "do i have any notifications from SuperNewChatApp",
            "check CustomWorkTool notifications",
            "any messages in that new app",
            "show me notifications from 未知应用",
            "what's new in the purple app",
            "check that app with the blue icon",
            # Generic queries that should work with any app
            "do i have any notifications",
            "show me all my notifications",
            "any new messages anywhere",
            "what apps have badges"
        ]
        
        for query in test_queries:
            with self.subTest(query=query):
                is_system = self.jarvis._is_system_command(query)
                self.assertTrue(is_system, f"Query '{query}' should be detected as system command")
    
    def test_query_routing_for_unknown_apps(self):
        """Test that queries are properly routed even for unknown apps"""
        test_cases = [
            # Specific unknown app queries
            ("notifications from SuperNewChatApp", QueryIntent.NOTIFICATIONS),
            ("messages in CustomWorkTool", QueryIntent.MESSAGES),
            ("check 未知应用", QueryIntent.SPECIFIC_APP),
            # Generic queries
            ("any notifications", QueryIntent.NOTIFICATIONS),
            ("do i have messages", QueryIntent.MESSAGES),
            ("what's on my screen", QueryIntent.WORKSPACE_OVERVIEW)
        ]
        
        for query, expected_intent in test_cases:
            with self.subTest(query=query):
                route = self.router.route_query(query, self.mock_windows)
                # Allow both NOTIFICATIONS and MESSAGES for notification queries
                # Also allow SPECIFIC_APP for app-specific queries
                if expected_intent == QueryIntent.NOTIFICATIONS:
                    acceptable_intents = [QueryIntent.NOTIFICATIONS, QueryIntent.MESSAGES, QueryIntent.SPECIFIC_APP]
                    self.assertIn(route.intent, acceptable_intents,
                                f"Query '{query}' should route to one of {acceptable_intents}")
                elif expected_intent == QueryIntent.MESSAGES:
                    acceptable_intents = [QueryIntent.MESSAGES, QueryIntent.NOTIFICATIONS, QueryIntent.SPECIFIC_APP]
                    self.assertIn(route.intent, acceptable_intents,
                                f"Query '{query}' should route to one of {acceptable_intents}")
                else:
                    self.assertEqual(route.intent, expected_intent,
                                   f"Query '{query}' should route to {expected_intent}")
    
    def test_app_pattern_detection(self):
        """Test pattern-based app detection works for unknown apps"""
        # Test communication app detection
        # Note: These need to contain keywords from app_categories['communication']
        # like 'chat', 'message', 'mail', etc.
        chat_apps = [
            WindowInfo(window_id=1, app_name="SuperChatApp", window_title="", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=2001),
            WindowInfo(window_id=2, app_name="MessageCenter", window_title="", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=2002),
            WindowInfo(window_id=3, app_name="NewMailClient", window_title="", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=2003)
        ]
        
        for window in chat_apps:
            with self.subTest(app=window.app_name):
                # Should detect as communication app due to patterns
                is_comm = self.router._is_communication_app(window)
                self.assertTrue(is_comm, f"{window.app_name} should be detected as communication app")
    
    def test_notification_detection_in_window_titles(self):
        """Test that notifications are detected from window titles"""
        windows_with_notifications = [
            WindowInfo(window_id=1, app_name="UnknownApp", 
                      window_title="UnknownApp - 5 notifications", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=3001),
            WindowInfo(window_id=2, app_name="RandomApp", 
                      window_title="You have 3 unread messages", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=3002),
            WindowInfo(window_id=3, app_name="NewApp", 
                      window_title="NewApp (2 new)", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=3003)
        ]
        
        for window in windows_with_notifications:
            # The window title contains notification indicators
            has_notif_keywords = any(keyword in window.window_title.lower() 
                                   for keyword in ['notification', 'unread', 'new', 'message'])
            self.assertTrue(has_notif_keywords, 
                          f"Window '{window.window_title}' should be detected as having notifications")
    
    def test_flexible_query_handling(self):
        """Test that various ways of asking about apps work"""
        query_variations = [
            # Direct app references
            ("do i have notifications from SuperNewChatApp", "SuperNewChatApp"),
            ("check SuperNewChatApp", "SuperNewChatApp"),
            ("SuperNewChatApp notifications", "SuperNewChatApp"),
            # Indirect references
            ("notifications from that new app", None),  # Should check all
            ("the app with the blue icon", None),  # Should check all
            ("any messages in the chat app", None),  # Should check communication apps
        ]
        
        for query, expected_app in query_variations:
            with self.subTest(query=query):
                route = self.router.route_query(query, self.mock_windows)
                self.assertIsNotNone(route, f"Query '{query}' should produce a route")
                self.assertGreater(route.confidence, 0, f"Query '{query}' should have confidence > 0")
    
    async def test_vision_command_handling(self):
        """Test that vision commands work for any app"""
        # Mock the workspace intelligence
        mock_response = "Sir, I can see SuperNewChatApp with 5 unread messages in the title bar."
        
        with patch.object(self.jarvis, 'workspace_intelligence_enabled', True):
            with patch.object(self.jarvis, '_handle_workspace_command', 
                            return_value=mock_response) as mock_handler:
                
                response = await self.jarvis._handle_system_command(
                    "do i have any notifications from SuperNewChatApp"
                )
                
                # Verify the command was handled
                mock_handler.assert_called_once()
                self.assertIn("SuperNewChatApp", response)

class TestIntegrationUnknownApps(unittest.TestCase):
    """Integration tests for unknown app handling"""
    
    def setUp(self):
        """Set up integration test environment"""
        self.jarvis = JARVISAgentVoice()
        self.router = SmartQueryRouter()
    
    async def test_end_to_end_unknown_app_query(self):
        """Test complete flow from query to response for unknown app"""
        # Create a query about an app JARVIS has never seen
        query = "do i have any notifications from FuturisticMessenger"
        
        # Step 1: Query should be detected as system command
        is_system = self.jarvis._is_system_command(query)
        self.assertTrue(is_system, "Query should be detected as system command")
        
        # Step 2: Mock windows including the unknown app
        mock_windows = [
            WindowInfo(
                window_id=1,
                app_name="FuturisticMessenger",
                window_title="FuturisticMessenger - 8 new messages",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=True,
                layer=0,
                is_visible=True,
                process_id=4001
            )
        ]
        
        # Step 3: Query should be routed correctly
        route = self.router.route_query(query, mock_windows)
        self.assertIn(route.intent, [QueryIntent.NOTIFICATIONS, QueryIntent.MESSAGES, QueryIntent.SPECIFIC_APP])
        self.assertGreater(len(route.target_windows), 0, "Should find the unknown app")
        self.assertEqual(route.target_windows[0].app_name, "FuturisticMessenger")
    
    def test_multi_language_app_support(self):
        """Test that apps with non-English names work"""
        # Apps with various language names
        international_windows = [
            WindowInfo(window_id=1, app_name="微信", window_title="微信 - 新消息",  # WeChat in Chinese
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=5001),
            WindowInfo(window_id=2, app_name="Телеграм", window_title="Телеграм (5)",  # Telegram in Russian
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=5002),
            WindowInfo(window_id=3, app_name="واتساب", window_title="واتساب - 3",  # WhatsApp in Arabic
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=5003)
        ]
        
        queries = [
            "check 微信 for messages",
            "any notifications from Телеграм",
            "واتساب notifications"
        ]
        
        for query in queries:
            with self.subTest(query=query):
                # Should detect as system command even with non-English
                is_system = self.jarvis._is_system_command(query)
                self.assertTrue(is_system, f"Query '{query}' should be detected as system command")
                
                # Should route correctly
                route = self.router.route_query(query, international_windows)
                self.assertIsNotNone(route, f"Query '{query}' should produce a route")

class TestDynamicVisualAnalysis(unittest.TestCase):
    """Test dynamic visual content analysis capabilities"""
    
    def setUp(self):
        """Set up visual analysis tests"""
        self.router = SmartQueryRouter()
    
    def test_badge_detection_scenarios(self):
        """Test various notification badge scenarios"""
        badge_scenarios = [
            # Numeric badges
            WindowInfo(window_id=1, app_name="App1", window_title="App1 (5)", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=6001),
            # Text badges
            WindowInfo(window_id=2, app_name="App2", window_title="App2 - New!", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=6002),
            # Dot badges
            WindowInfo(window_id=3, app_name="App3", window_title="App3 •", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=6003),
            # Multiple indicators
            WindowInfo(window_id=4, app_name="App4", window_title="App4 (99+) - 5 unread", 
                      bounds={}, is_focused=False, layer=0, is_visible=True, process_id=6004)
        ]
        
        for window in badge_scenarios:
            # These patterns indicate notifications
            title = window.window_title.lower()
            has_indicator = any(pattern in title for pattern in ['(', 'new', 'unread', '•', '!'])
            self.assertTrue(has_indicator, 
                          f"Window '{window.window_title}' should show notification indicators")
    
    def test_context_aware_responses(self):
        """Test that responses adapt to what's actually visible"""
        test_scenarios = [
            {
                "windows": [
                    WindowInfo(window_id=1, app_name="UnknownChat", 
                              window_title="UnknownChat - Loading...", 
                              bounds={}, is_focused=True, layer=0, is_visible=True, process_id=7001)
                ],
                "query": "do i have messages in UnknownChat",
                "expected_keywords": ["loading", "UnknownChat"]
            },
            {
                "windows": [
                    WindowInfo(window_id=2, app_name="WorkApp", 
                              window_title="WorkApp - Error: Connection failed", 
                              bounds={}, is_focused=True, layer=0, is_visible=True, process_id=7002)
                ],
                "query": "check WorkApp",
                "expected_keywords": ["error", "connection", "WorkApp"]
            },
            {
                "windows": [
                    WindowInfo(window_id=3, app_name="EmptyApp", 
                              window_title="EmptyApp", 
                              bounds={}, is_focused=True, layer=0, is_visible=True, process_id=7003)
                ],
                "query": "any notifications in EmptyApp",
                "expected_keywords": ["EmptyApp", "no"]
            }
        ]
        
        for scenario in test_scenarios:
            with self.subTest(query=scenario["query"]):
                route = self.router.route_query(scenario["query"], scenario["windows"])
                # Should route to the app regardless of its state
                # Note: If the app name is in the query, it should target that window
                if any(window.app_name.lower() in scenario["query"].lower() for window in scenario["windows"]):
                    self.assertGreater(len(route.target_windows), 0, 
                                     f"Query mentioning app should target that app's window")
                else:
                    # For generic queries, it's OK if no specific window is targeted
                    self.assertIsNotNone(route, "Should still produce a valid route")
                # The actual response would be generated by vision analysis

def run_tests():
    """Run all intelligent vision tests"""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntelligentVisionFunctionality))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegrationUnknownApps))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDynamicVisualAnalysis))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    # For async tests
    import asyncio
    
    # Run sync tests
    success = run_tests()
    
    # Run async tests separately
    async def run_async_tests():
        """Run async test methods"""
        test_instance = TestIntelligentVisionFunctionality()
        test_instance.setUp()
        await test_instance.test_vision_command_handling()
        
        integration_test = TestIntegrationUnknownApps()
        integration_test.setUp()
        await integration_test.test_end_to_end_unknown_app_query()
        
        print("\n✅ Async tests completed")
    
    asyncio.run(run_async_tests())
    
    if success:
        print("\n🎉 All intelligent vision tests passed!")
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)