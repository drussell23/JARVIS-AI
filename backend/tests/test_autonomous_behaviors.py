#!/usr/bin/env python3
"""
Comprehensive tests for autonomous behaviors
Tests various scenarios including edge cases
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomy.autonomous_behaviors import (
    MessageHandler, MeetingHandler, WorkspaceOrganizer, 
    SecurityHandler, AutonomousBehaviorManager
)
from autonomy.autonomous_decision_engine import ActionPriority, ActionCategory
from vision.window_detector import WindowInfo


class TestMessageHandler:
    """Test message handling behaviors"""
    
    @pytest.fixture
    def handler(self):
        return MessageHandler()
    
    @pytest.fixture
    def mock_windows(self):
        """Create various test windows"""
        return {
            "automated": WindowInfo(
                window_id=1,
                app_name="Email",
                window_title="Newsletter: Daily Digest",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1001
            ),
            "meeting": WindowInfo(
                window_id=2,
                app_name="Slack",
                window_title="Reminder: Team Standup starts in 5 minutes",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1002
            ),
            "urgent": WindowInfo(
                window_id=3,
                app_name="Messages",
                window_title="URGENT: Production server down!",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1003
            ),
            "security": WindowInfo(
                window_id=4,
                app_name="Mail",
                window_title="Security Alert: Unusual login attempt detected",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1004
            ),
            "normal": WindowInfo(
                window_id=5,
                app_name="Slack",
                window_title="Hey, how's the project going?",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1005
            )
        }
    
    @pytest.mark.asyncio
    async def test_automated_notification_handling(self, handler, mock_windows):
        """Test handling of automated notifications"""
        window = mock_windows["automated"]
        
        # Mock vision extraction
        handler._extract_message_content = AsyncMock(
            return_value="Newsletter: Daily Digest\nAutomated message - do not reply"
        )
        
        action = await handler.handle_routine_message(window)
        
        assert action is not None
        assert action.action_type == "dismiss_notification"
        assert action.priority == ActionPriority.LOW
        assert action.confidence >= 0.9
        assert "automated" in action.reasoning.lower()
    
    @pytest.mark.asyncio
    async def test_meeting_reminder_handling(self, handler, mock_windows):
        """Test handling of meeting reminders"""
        window = mock_windows["meeting"]
        
        handler._extract_message_content = AsyncMock(
            return_value="Team Standup starts in 5 minutes\nZoom link: https://zoom.us/..."
        )
        
        action = await handler.handle_routine_message(window)
        
        assert action is not None
        assert action.action_type == "prepare_for_meeting"
        assert action.priority == ActionPriority.HIGH
        assert action.category == ActionCategory.CALENDAR
        assert "meeting" in action.reasoning.lower()
        
        # Check meeting info extraction
        meeting_info = action.params.get("meeting_info", {})
        assert meeting_info.get("platform") == "Zoom"
        assert "5 minutes" in str(meeting_info.get("time", ""))
    
    @pytest.mark.asyncio
    async def test_urgent_message_handling(self, handler, mock_windows):
        """Test handling of urgent messages"""
        window = mock_windows["urgent"]
        
        handler._extract_message_content = AsyncMock(
            return_value="URGENT: Production server down! Need immediate attention!"
        )
        
        action = await handler.handle_routine_message(window)
        
        assert action is not None
        assert action.action_type == "highlight_urgent_message"
        assert action.priority == ActionPriority.HIGH
        assert action.params.get("urgency_level") == "critical"
    
    @pytest.mark.asyncio
    async def test_security_alert_handling(self, handler, mock_windows):
        """Test handling of security alerts"""
        window = mock_windows["security"]
        
        handler._extract_message_content = AsyncMock(
            return_value="Security Alert: Unusual login attempt from unknown location"
        )
        
        action = await handler.handle_routine_message(window)
        
        assert action is not None
        assert action.action_type == "security_alert"
        assert action.priority == ActionPriority.CRITICAL
        assert action.category == ActionCategory.SECURITY
        assert action.params.get("alert_type") == "access_attempt"
    
    @pytest.mark.asyncio
    async def test_normal_message_handling(self, handler, mock_windows):
        """Test handling of normal messages"""
        window = mock_windows["normal"]
        
        handler._extract_message_content = AsyncMock(
            return_value="Hey, how's the project going? Let me know when you have time."
        )
        
        action = await handler.handle_routine_message(window)
        
        assert action is not None
        assert action.action_type == "queue_for_review"
        assert action.priority == ActionPriority.MEDIUM
        assert "queued" in action.reasoning.lower()
    
    def test_pattern_detection(self, handler):
        """Test various pattern detection methods"""
        # Test automated patterns
        assert handler._is_automated_notification("Automated newsletter")
        assert handler._is_automated_notification("noreply@company.com")
        assert handler._is_automated_notification("Daily digest summary")
        assert not handler._is_automated_notification("Hey John, check this out")
        
        # Test meeting patterns
        assert handler._is_meeting_reminder("Meeting starts in 10 minutes")
        assert handler._is_meeting_reminder("Zoom standup at 2pm")
        assert handler._is_meeting_reminder("Calendar reminder: Team sync")
        assert not handler._is_meeting_reminder("Let's meet for coffee")
        
        # Test urgent patterns
        assert handler._is_urgent_message("URGENT: Server down")
        assert handler._is_urgent_message("Critical issue - need help ASAP")
        assert handler._is_urgent_message("Deadline today!")
        assert not handler._is_urgent_message("No rush, whenever you can")
        
        # Test security patterns
        assert handler._is_security_alert("Security alert: suspicious login")
        assert handler._is_security_alert("Password change required")
        assert handler._is_security_alert("2FA verification needed")
        assert not handler._is_security_alert("Secure connection established")


class TestMeetingHandler:
    """Test meeting preparation behaviors"""
    
    @pytest.fixture
    def handler(self):
        return MeetingHandler()
    
    @pytest.fixture
    def test_windows(self):
        """Create test windows for meeting scenarios"""
        return [
            WindowInfo(
                window_id=1,
                app_name="1Password",
                window_title="1Password - Vault",
                bounds={"x": 0, "y": 0, "width": 400, "height": 300},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=2001
            ),
            WindowInfo(
                window_id=2,
                app_name="Banking App",
                window_title="Chase Banking - Account Summary",
                bounds={"x": 400, "y": 0, "width": 600, "height": 500},
                is_focused=True,
                layer=0,
                is_visible=True,
                process_id=2002
            ),
            WindowInfo(
                window_id=3,
                app_name="Discord",
                window_title="Discord - Gaming Server",
                bounds={"x": 0, "y": 300, "width": 400, "height": 400},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=2003
            ),
            WindowInfo(
                window_id=4,
                app_name="Spotify",
                window_title="Spotify - Now Playing",
                bounds={"x": 1000, "y": 0, "width": 300, "height": 400},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=2004
            )
        ]
    
    @pytest.mark.asyncio
    async def test_meeting_preparation(self, handler, test_windows):
        """Test comprehensive meeting preparation"""
        meeting_info = {
            "title": "Team Standup",
            "time": datetime.now() + timedelta(minutes=5),
            "platform": "Zoom"
        }
        
        actions = await handler.prepare_for_meeting(meeting_info, test_windows)
        
        # Should hide sensitive windows
        sensitive_actions = [a for a in actions if a.action_type == "minimize_window"]
        assert len(sensitive_actions) >= 2  # 1Password and Banking
        
        # Should open meeting platform
        open_actions = [a for a in actions if a.action_type == "open_application"]
        assert len(open_actions) == 1
        assert open_actions[0].target == "Zoom"
        
        # Should mute distracting apps
        mute_actions = [a for a in actions if a.action_type == "mute_notifications"]
        assert len(mute_actions) >= 2  # Discord and Spotify
    
    def test_sensitive_window_detection(self, handler, test_windows):
        """Test detection of sensitive windows"""
        assert handler._is_sensitive_window(test_windows[0])  # 1Password
        assert handler._is_sensitive_window(test_windows[1])  # Banking
        assert not handler._is_sensitive_window(test_windows[2])  # Discord
        assert not handler._is_sensitive_window(test_windows[3])  # Spotify
        
        # Test pattern-based detection
        medical_window = WindowInfo(
            window_id=5,
            app_name="Chrome",
            window_title="MyHealth - Medical Records",
            bounds={"x": 0, "y": 0, "width": 800, "height": 600},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=2005
        )
        assert handler._is_sensitive_window(medical_window)
    
    def test_distracting_app_detection(self, handler, test_windows):
        """Test detection of distracting apps"""
        distracting = handler._find_distracting_apps(test_windows)
        
        assert "Discord" in distracting
        assert "Spotify" in distracting
        assert "1Password" not in distracting
        assert len(distracting) == 2


class TestWorkspaceOrganizer:
    """Test workspace organization behaviors"""
    
    @pytest.fixture
    def organizer(self):
        return WorkspaceOrganizer()
    
    @pytest.fixture
    def cluttered_workspace(self):
        """Create a cluttered workspace with many windows"""
        windows = []
        # Create overlapping windows
        for i in range(15):
            windows.append(WindowInfo(
                window_id=i,
                app_name=f"App{i}",
                window_title=f"Window {i}",
                bounds={"x": i * 50, "y": i * 30, "width": 600, "height": 400},
                is_focused=(i == 0),
                layer=0,
                is_visible=True,
                process_id=3000 + i
            ))
        
        # Add some duplicate windows
        windows.append(WindowInfo(
            window_id=100,
            app_name="Chrome",
            window_title="untitled",
            bounds={"x": 100, "y": 100, "width": 800, "height": 600},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=3100
        ))
        
        windows.append(WindowInfo(
            window_id=101,
            app_name="Chrome",
            window_title="untitled",
            bounds={"x": 150, "y": 150, "width": 800, "height": 600},
            is_focused=False,
            layer=0,
            is_visible=True,
            process_id=3101
        ))
        
        return windows
    
    @pytest.mark.asyncio
    async def test_workspace_organization(self, organizer, cluttered_workspace):
        """Test workspace organization suggestions"""
        actions = await organizer.analyze_and_organize(
            cluttered_workspace, 
            user_state="available"
        )
        
        # Should suggest arranging windows due to overlap
        arrange_actions = [a for a in actions if a.action_type == "arrange_windows"]
        assert len(arrange_actions) > 0
        
        # Should suggest closing duplicates
        close_actions = [a for a in actions if a.action_type == "close_duplicate"]
        assert len(close_actions) > 0
        
        # Should suggest window reduction
        reduction_actions = [a for a in actions if a.action_type == "close_window"]
        assert len(reduction_actions) > 0
    
    @pytest.mark.asyncio
    async def test_focus_mode_suggestion(self, organizer):
        """Test focus mode suggestions"""
        # Create a cluttered workspace during focused work
        windows = []
        for i in range(20):
            windows.append(WindowInfo(
                window_id=i,
                app_name=["VSCode", "Chrome", "Slack", "Discord", "Spotify"][i % 5],
                window_title=f"Window {i}",
                bounds={"x": i * 40, "y": i * 30, "width": 600, "height": 400},
                is_focused=(i == 0),
                layer=0,
                is_visible=True,
                process_id=4000 + i
            ))
        
        actions = await organizer.analyze_and_organize(windows, user_state="focused")
        
        # Should suggest focus mode
        focus_actions = [a for a in actions if a.action_type == "enable_focus_mode"]
        assert len(focus_actions) > 0
        
        # Should identify primary work apps
        if focus_actions:
            keep_apps = focus_actions[0].params.get("keep_apps", [])
            assert "VSCode" in keep_apps
            assert "Chrome" in keep_apps
    
    def test_window_overlap_detection(self, organizer):
        """Test window overlap detection"""
        # Non-overlapping windows
        w1 = WindowInfo(
            window_id=1, app_name="App1", window_title="Window 1",
            bounds={"x": 0, "y": 0, "width": 400, "height": 300},
            is_focused=False, layer=0, is_visible=True, process_id=5001
        )
        w2 = WindowInfo(
            window_id=2, app_name="App2", window_title="Window 2",
            bounds={"x": 500, "y": 0, "width": 400, "height": 300},
            is_focused=False, layer=0, is_visible=True, process_id=5002
        )
        
        assert not organizer._windows_overlap(w1, w2)
        
        # Overlapping windows
        w3 = WindowInfo(
            window_id=3, app_name="App3", window_title="Window 3",
            bounds={"x": 200, "y": 100, "width": 400, "height": 300},
            is_focused=False, layer=0, is_visible=True, process_id=5003
        )
        
        assert organizer._windows_overlap(w1, w3)
    
    def test_window_grouping(self, organizer):
        """Test window grouping by context"""
        windows = [
            WindowInfo(
                window_id=1, app_name="Visual Studio Code", 
                window_title="project.py", bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=True, layer=0, is_visible=True, process_id=6001
            ),
            WindowInfo(
                window_id=2, app_name="Terminal", 
                window_title="bash", bounds={"x": 800, "y": 0, "width": 400, "height": 600},
                is_focused=False, layer=0, is_visible=True, process_id=6002
            ),
            WindowInfo(
                window_id=3, app_name="Chrome", 
                window_title="Documentation", bounds={"x": 0, "y": 600, "width": 1200, "height": 400},
                is_focused=False, layer=0, is_visible=True, process_id=6003
            ),
            WindowInfo(
                window_id=4, app_name="Slack", 
                window_title="Team Chat", bounds={"x": 1200, "y": 0, "width": 400, "height": 1000},
                is_focused=False, layer=0, is_visible=True, process_id=6004
            )
        ]
        
        groups = organizer._group_windows_by_context(windows)
        
        assert "development" in groups
        assert "terminal" in groups
        assert "browser" in groups
        assert "communication" in groups
        
        assert len(groups["development"]) == 1
        assert groups["development"][0].app_name == "Visual Studio Code"


class TestSecurityHandler:
    """Test security handling behaviors"""
    
    @pytest.fixture
    def handler(self):
        return SecurityHandler()
    
    @pytest.mark.asyncio
    async def test_suspicious_login_handling(self, handler):
        """Test handling of suspicious login attempts"""
        context = {
            "source": "Gmail",
            "location": "Unknown",
            "time": datetime.now()
        }
        
        actions = await handler.handle_security_event("suspicious_login", context)
        
        # Should lock sensitive apps
        lock_actions = [a for a in actions if a.action_type == "lock_sensitive_apps"]
        assert len(lock_actions) == 1
        assert lock_actions[0].priority == ActionPriority.CRITICAL
        
        # Should notify user
        notify_actions = [a for a in actions if a.action_type == "security_notification"]
        assert len(notify_actions) == 1
        assert notify_actions[0].params.get("severity") == "high"
    
    @pytest.mark.asyncio
    async def test_password_manager_during_meeting(self, handler):
        """Test password manager visibility during meetings"""
        context = {
            "in_meeting": True,
            "current_windows": [
                WindowInfo(
                    window_id=1, app_name="1Password", 
                    window_title="Vault", bounds={"x": 0, "y": 0, "width": 400, "height": 600},
                    is_focused=True, layer=0, is_visible=True, process_id=7001
                ),
                WindowInfo(
                    window_id=2, app_name="Zoom", 
                    window_title="Team Meeting", bounds={"x": 400, "y": 0, "width": 800, "height": 600},
                    is_focused=False, layer=0, is_visible=True, process_id=7002
                )
            ]
        }
        
        actions = await handler.handle_security_event("password_manager_open", context)
        
        # Should hide password manager
        hide_actions = [a for a in actions if a.action_type == "hide_window"]
        assert len(hide_actions) == 1
        assert hide_actions[0].target == "1Password"
        assert hide_actions[0].priority == ActionPriority.CRITICAL
    
    @pytest.mark.asyncio
    async def test_sensitive_data_exposure(self, handler):
        """Test handling of exposed sensitive data"""
        context = {
            "app_name": "TextEdit",
            "window_id": 123,
            "data_type": "password"
        }
        
        actions = await handler.handle_security_event("sensitive_data_exposed", context)
        
        # Should blur sensitive content
        blur_actions = [a for a in actions if a.action_type == "blur_sensitive_content"]
        assert len(blur_actions) == 1
        assert blur_actions[0].params.get("content_type") == "password"
    
    @pytest.mark.asyncio
    async def test_phishing_detection(self, handler):
        """Test phishing site detection and blocking"""
        context = {
            "alert_type": "phishing",
            "url": "http://suspicious-site.com",
            "browser": "Chrome"
        }
        
        actions = await handler.handle_security_event("security_alert", context)
        
        # Should block phishing site
        block_actions = [a for a in actions if a.action_type == "block_phishing_site"]
        assert len(block_actions) == 1
        assert block_actions[0].params.get("url") == "http://suspicious-site.com"


class TestAutonomousBehaviorManager:
    """Test the complete behavior management system"""
    
    @pytest.fixture
    def manager(self):
        return AutonomousBehaviorManager()
    
    @pytest.mark.asyncio
    async def test_complete_workspace_processing(self, manager):
        """Test processing a complete workspace state"""
        # Create a complex workspace
        windows = [
            # Messages
            WindowInfo(
                window_id=1, app_name="Slack", 
                window_title="Slack (3 new messages)", 
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False, layer=0, is_visible=True, process_id=8001
            ),
            # Meeting reminder
            WindowInfo(
                window_id=2, app_name="Calendar", 
                window_title="Meeting starts in 5 minutes", 
                bounds={"x": 800, "y": 0, "width": 400, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=8002
            ),
            # Sensitive app
            WindowInfo(
                window_id=3, app_name="1Password", 
                window_title="Vault", 
                bounds={"x": 0, "y": 600, "width": 400, "height": 400},
                is_focused=False, layer=0, is_visible=True, process_id=8003
            ),
            # Work apps
            WindowInfo(
                window_id=4, app_name="VSCode", 
                window_title="main.py", 
                bounds={"x": 400, "y": 300, "width": 800, "height": 700},
                is_focused=True, layer=0, is_visible=True, process_id=8004
            )
        ]
        
        workspace_state = {
            "window_count": len(windows),
            "user_state": "available",
            "in_meeting": False,
            "last_organized": None
        }
        
        # Mock vision analyzer
        with patch.object(manager.message_handler, '_extract_message_content', 
                         new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = "Team update: 3 new messages in #general"
            
            actions = await manager.process_workspace_state(workspace_state, windows)
        
        # Should generate multiple actions
        assert len(actions) > 0
        
        # Check action diversity
        action_types = {action.action_type for action in actions}
        
        # Should handle messages
        assert any(t in action_types for t in ["queue_for_review", "dismiss_notification"])
        
        # Should prepare for meeting
        assert "prepare_for_meeting" in action_types or "minimize_window" in action_types
    
    @pytest.mark.asyncio
    async def test_action_prioritization(self, manager):
        """Test that actions are properly prioritized"""
        # Create windows that will generate different priority actions
        windows = [
            # Security alert (CRITICAL)
            WindowInfo(
                window_id=1, app_name="Mail", 
                window_title="Security Alert: Suspicious login detected", 
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False, layer=0, is_visible=True, process_id=9001
            ),
            # Meeting (HIGH)
            WindowInfo(
                window_id=2, app_name="Calendar", 
                window_title="Meeting starts in 2 minutes", 
                bounds={"x": 0, "y": 0, "width": 400, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=9002
            ),
            # Normal message (MEDIUM)
            WindowInfo(
                window_id=3, app_name="Slack", 
                window_title="Hey, got a minute?", 
                bounds={"x": 0, "y": 0, "width": 600, "height": 400},
                is_focused=False, layer=0, is_visible=True, process_id=9003
            )
        ]
        
        workspace_state = {"window_count": 3, "user_state": "available"}
        
        with patch.object(manager.message_handler, '_extract_message_content', 
                         new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = [
                "Security Alert: Suspicious login from unknown location",
                "Team Standup starts in 2 minutes - Zoom link included",
                "Hey, got a minute to discuss the project?"
            ]
            
            actions = await manager.process_workspace_state(workspace_state, windows)
        
        # Actions should be prioritized by importance
        if actions:
            # First action should be security-related
            assert actions[0].priority == ActionPriority.CRITICAL
            
            # Check that high priority actions come before medium/low
            priorities = [a.priority for a in actions]
            critical_idx = [i for i, p in enumerate(priorities) if p == ActionPriority.CRITICAL]
            high_idx = [i for i, p in enumerate(priorities) if p == ActionPriority.HIGH]
            medium_idx = [i for i, p in enumerate(priorities) if p == ActionPriority.MEDIUM]
            
            # Critical should come first
            if critical_idx and high_idx:
                assert min(critical_idx) < min(high_idx)
            if critical_idx and medium_idx:
                assert min(critical_idx) < min(medium_idx)


@pytest.mark.asyncio
async def test_edge_case_empty_workspace():
    """Test behavior with empty workspace"""
    manager = AutonomousBehaviorManager()
    
    actions = await manager.process_workspace_state({}, [])
    
    # Should handle gracefully
    assert actions == []


@pytest.mark.asyncio
async def test_edge_case_massive_workspace():
    """Test behavior with extremely cluttered workspace"""
    manager = AutonomousBehaviorManager()
    
    # Create 100 windows
    windows = []
    for i in range(100):
        windows.append(WindowInfo(
            window_id=i,
            app_name=f"App{i % 10}",
            window_title=f"Window {i}",
            bounds={"x": (i % 10) * 100, "y": (i // 10) * 50, "width": 400, "height": 300},
            is_focused=(i == 0),
            layer=0,
            is_visible=True,
            process_id=10000 + i
        ))
    
    workspace_state = {
        "window_count": len(windows),
        "user_state": "available"
    }
    
    actions = await manager.process_workspace_state(workspace_state, windows)
    
    # Should generate organization actions
    assert len(actions) > 0
    
    # Should limit actions to reasonable number
    assert len(actions) <= 10  # As defined in _prioritize_actions


@pytest.mark.asyncio
async def test_edge_case_rapid_changes():
    """Test behavior with rapidly changing windows"""
    manager = AutonomousBehaviorManager()
    
    # Simulate rapid window changes
    for i in range(5):
        windows = [
            WindowInfo(
                window_id=j,
                app_name="RapidApp",
                window_title=f"Notification {i}-{j}",
                bounds={"x": j * 100, "y": 0, "width": 300, "height": 200},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=11000 + j
            ) for j in range(3)
        ]
        
        workspace_state = {"window_count": 3, "user_state": "available"}
        
        actions = await manager.process_workspace_state(workspace_state, windows)
        
        # Should handle each state independently
        assert isinstance(actions, list)
        
        # Brief delay to simulate rapid changes
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])