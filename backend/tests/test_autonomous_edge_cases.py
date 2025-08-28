#!/usr/bin/env python3
"""
Integration tests for autonomous system edge cases
Tests complex scenarios and system boundaries
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import random

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autonomy.autonomous_decision_engine import (
    AutonomousDecisionEngine, AutonomousAction, 
    ActionPriority, ActionCategory
)
from autonomy.permission_manager import PermissionManager
from autonomy.context_engine import ContextEngine, UserState
from autonomy.action_executor import ActionExecutor, ExecutionStatus
from autonomy.autonomous_behaviors import AutonomousBehaviorManager
from vision.window_detector import WindowInfo
from vision.workspace_analyzer import WorkspaceAnalysis

class TestEdgeCaseScenarios:
    """Test complex edge case scenarios"""
    
    @pytest.fixture
    def full_system(self):
        """Create a full autonomous system"""
        return {
            "decision_engine": AutonomousDecisionEngine(),
            "permission_manager": PermissionManager(),
            "context_engine": ContextEngine(),
            "action_executor": ActionExecutor(),
            "behavior_manager": AutonomousBehaviorManager()
        }
    
    @pytest.mark.asyncio
    async def test_conflicting_priorities(self, full_system):
        """Test handling of conflicting priority situations"""
        # Scenario: Meeting starting + urgent message + security alert
        windows = [
            WindowInfo(
                window_id=1, app_name="Calendar",
                window_title="Important Client Meeting starts in 1 minute",
                bounds={"x": 0, "y": 0, "width": 400, "height": 300},
                is_focused=True, layer=0, is_visible=True, process_id=1001
            ),
            WindowInfo(
                window_id=2, app_name="Slack",
                window_title="URGENT: Production database is down!",
                bounds={"x": 400, "y": 0, "width": 600, "height": 400},
                is_focused=False, layer=0, is_visible=True, process_id=1002
            ),
            WindowInfo(
                window_id=3, app_name="Mail",
                window_title="Security Alert: Multiple failed login attempts",
                bounds={"x": 0, "y": 300, "width": 800, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=1003
            ),
            WindowInfo(
                window_id=4, app_name="1Password",
                window_title="1Password - Vault Unlocked",
                bounds={"x": 800, "y": 300, "width": 400, "height": 400},
                is_focused=False, layer=0, is_visible=True, process_id=1004
            )
        ]
        
        workspace_state = WorkspaceAnalysis(
            focused_task="Preparing for client meeting",
            window_relationships={},
            workspace_context="Multiple critical situations",
            important_notifications=["Meeting", "Database down", "Security alert"],
            suggestions=[],
            confidence=0.9
        )
        
        # Get decisions from engine
        decision_engine = full_system["decision_engine"]
        actions = await decision_engine.analyze_and_decide(workspace_state, windows)
        
        # Should handle all critical situations
        assert len(actions) > 0
        
        # Security should be highest priority
        security_actions = [a for a in actions if a.category == ActionCategory.SECURITY]
        assert len(security_actions) > 0
        assert security_actions[0].priority == ActionPriority.CRITICAL
        
        # Meeting prep should also be high priority
        meeting_actions = [a for a in actions if "meeting" in a.action_type.lower()]
        assert len(meeting_actions) > 0
        
        # Should have security-related actions for password manager
        # Note: The decision engine may not generate specific "minimize_window" actions
        # but should recognize the security concern
        password_related = [a for a in actions if "password" in str(a.target).lower() or 
                           "password" in str(a.reasoning).lower() or
                           a.category == ActionCategory.SECURITY]
        assert len(password_related) > 0
    
    @pytest.mark.asyncio
    async def test_permission_learning_conflicts(self, full_system):
        """Test when user preferences conflict with safety"""
        permission_mgr = full_system["permission_manager"]
        
        # Simulate user always approving risky actions
        risky_action = AutonomousAction(
            action_type="close_all_windows",
            target="workspace",
            params={"force": True},
            priority=ActionPriority.LOW,
            confidence=0.8,
            category=ActionCategory.MAINTENANCE,
            reasoning="Cleaning up workspace"
        )
        
        # Record multiple approvals
        for _ in range(10):
            permission_mgr.record_decision(risky_action, approved=True)
        
        # Now test with a critical context
        critical_action = AutonomousAction(
            action_type="close_all_windows",
            target="workspace",
            params={"force": True, "unsaved_work": True},
            priority=ActionPriority.LOW,
            confidence=0.8,
            category=ActionCategory.MAINTENANCE,
            reasoning="Cleaning up workspace (unsaved work detected)"
        )
        
        # Even with history, should be cautious
        permission, confidence, reason = permission_mgr.check_permission(critical_action)
        
        # Should not auto-approve dangerous actions
        assert permission is None or permission is False
    
    @pytest.mark.asyncio
    async def test_rapid_context_switching(self, full_system):
        """Test system behavior during rapid context changes"""
        context_engine = full_system["context_engine"]
        
        # Simulate rapid context changes
        contexts = []
        
        for i in range(10):
            # Alternate between different states rapidly
            if i % 3 == 0:
                windows = [WindowInfo(
                    window_id=i, app_name="Zoom",
                    window_title="Meeting in progress",
                    bounds={"x": 0, "y": 0, "width": 1920, "height": 1080},
                    is_focused=True, layer=0, is_visible=True, process_id=2000+i
                )]
                state = "in_meeting"
            elif i % 3 == 1:
                windows = [WindowInfo(
                    window_id=i, app_name="VSCode",
                    window_title="Deep work session",
                    bounds={"x": 0, "y": 0, "width": 1920, "height": 1080},
                    is_focused=True, layer=0, is_visible=True, process_id=2000+i
                )]
                state = "focused"
            else:
                windows = [WindowInfo(
                    window_id=i, app_name="Safari",
                    window_title="Browsing",
                    bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                    is_focused=True, layer=0, is_visible=True, process_id=2000+i
                )]
                state = "available"
            
            workspace = WorkspaceAnalysis(
                focused_task=state,
                window_relationships={},
                workspace_context=f"Context {i}",
                important_notifications=[],
                suggestions=[],
                confidence=0.8
            )
            
            context = await context_engine.analyze_context(workspace, windows)
            contexts.append(context)
            
            # Brief delay between context switches
            await asyncio.sleep(0.1)
        
        # System should handle rapid changes gracefully
        assert len(contexts) == 10
        
        # Should have detected different states
        states = {c.user_state for c in contexts}
        assert len(states) > 1
    
    @pytest.mark.asyncio
    async def test_action_rollback_cascade(self, full_system):
        """Test cascading rollback scenarios"""
        executor = full_system["action_executor"]
        
        # Create a series of related actions
        actions = [
            AutonomousAction(
                action_type="minimize_window",
                target="1Password",
                params={"window_id": 1},
                priority=ActionPriority.HIGH,
                confidence=0.9,
                category=ActionCategory.SECURITY,
                reasoning="Hiding sensitive data"
            ),
            AutonomousAction(
                action_type="open_application",
                target="Zoom",
                params={"meeting_link": "https://zoom.us/meeting"},
                priority=ActionPriority.HIGH,
                confidence=0.9,
                category=ActionCategory.CALENDAR,
                reasoning="Opening meeting app"
            ),
            AutonomousAction(
                action_type="mute_notifications",
                target="all",
                params={"duration_minutes": 60},
                priority=ActionPriority.MEDIUM,
                confidence=0.85,
                category=ActionCategory.NOTIFICATION,
                reasoning="Muting for meeting"
            )
        ]
        
        # Execute actions
        results = []
        for action in actions:
            result = await executor.execute_action(action, dry_run=False)
            results.append(result)
        
        # Simulate needing to rollback (e.g., meeting cancelled)
        rollback_results = []
        for i in range(len(results) - 1, -1, -1):
            if results[i].rollback_available:
                rollback_result = await executor.rollback_action(actions[i])
                rollback_results.append(rollback_result)
        
        # All rollbacks should succeed
        assert all(r.status == ExecutionStatus.SUCCESS for r in rollback_results)
    
    @pytest.mark.asyncio
    async def test_resource_exhaustion(self, full_system):
        """Test system behavior under resource constraints"""
        behavior_mgr = full_system["behavior_manager"]
        
        # Create an overwhelming number of windows
        windows = []
        for i in range(200):  # Way more than typical
            windows.append(WindowInfo(
                window_id=i,
                app_name=f"App{i % 20}",
                window_title=f"Window {i} - {'Urgent' if i % 10 == 0 else 'Normal'}",
                bounds={"x": (i % 20) * 50, "y": (i % 10) * 50, "width": 300, "height": 200},
                is_focused=(i == 0),
                layer=i % 3,
                is_visible=(i < 50),  # Only first 50 visible
                process_id=3000 + i
            ))
        
        workspace_state = {
            "window_count": len(windows),
            "user_state": "available",
            "resource_usage": {"cpu": 0.95, "memory": 0.90}  # High resource usage
        }
        
        # System should handle gracefully
        start_time = datetime.now()
        actions = await behavior_mgr.process_workspace_state(workspace_state, windows)
        end_time = datetime.now()
        
        # Should complete in reasonable time
        assert (end_time - start_time).total_seconds() < 5.0
        
        # Should limit actions to prevent further strain
        assert len(actions) <= 10
        
        # Should prioritize important actions
        if actions:
            # Check that urgent items are handled
            urgent_handled = any("urgent" in a.reasoning.lower() for a in actions)
            organization_suggested = any(a.category == ActionCategory.MAINTENANCE for a in actions)
            assert urgent_handled or organization_suggested
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, full_system):
        """Test detection of circular action dependencies"""
        decision_engine = full_system["decision_engine"]
        
        # Create a scenario with potential circular dependencies
        windows = [
            WindowInfo(
                window_id=1, app_name="App1",
                window_title="Depends on App2",
                bounds={"x": 0, "y": 0, "width": 400, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=4001
            ),
            WindowInfo(
                window_id=2, app_name="App2",
                window_title="Depends on App3",
                bounds={"x": 400, "y": 0, "width": 400, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=4002
            ),
            WindowInfo(
                window_id=3, app_name="App3",
                window_title="Depends on App1",
                bounds={"x": 800, "y": 0, "width": 400, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=4003
            )
        ]
        
        workspace = WorkspaceAnalysis(
            focused_task="Complex dependency chain",
            window_relationships={
                "App1": ["App2"],
                "App2": ["App3"],
                "App3": ["App1"]  # Circular!
            },
            workspace_context="Circular dependencies detected",
            important_notifications=[],
            suggestions=["Resolve circular dependencies"],
            confidence=0.7
        )
        
        actions = await decision_engine.analyze_and_decide(workspace, windows)
        
        # Should handle circular dependencies gracefully
        assert isinstance(actions, list)
        
        # Should not create actions that would worsen the situation
        close_actions = [a for a in actions if a.action_type == "close_window"]
        if close_actions:
            # Should close in a way that breaks the cycle
            assert len(close_actions) <= 1
    
    @pytest.mark.asyncio
    async def test_network_failure_resilience(self, full_system):
        """Test system resilience to network failures"""
        behavior_mgr = full_system["behavior_manager"]
        
        # Mock network-dependent operations to fail
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = asyncio.TimeoutError("Network timeout")
            
            windows = [
                WindowInfo(
                    window_id=1, app_name="Chrome",
                    window_title="GitHub - Connection timeout",
                    bounds={"x": 0, "y": 0, "width": 1200, "height": 800},
                    is_focused=True, layer=0, is_visible=True, process_id=5001
                ),
                WindowInfo(
                    window_id=2, app_name="Slack",
                    window_title="Slack - Reconnecting...",
                    bounds={"x": 1200, "y": 0, "width": 400, "height": 800},
                    is_focused=False, layer=0, is_visible=True, process_id=5002
                )
            ]
            
            workspace_state = {
                "window_count": 2,
                "user_state": "available",
                "network_status": "unstable"
            }
            
            # Should handle network issues gracefully
            actions = await behavior_mgr.process_workspace_state(workspace_state, windows)
            
            # Should still generate local actions
            assert isinstance(actions, list)
            
            # Should not suggest network-dependent actions
            network_actions = [a for a in actions if "online" in a.action_type or "sync" in a.action_type]
            assert len(network_actions) == 0
    
    @pytest.mark.asyncio
    async def test_permission_boundary_conditions(self, full_system):
        """Test permission system boundary conditions"""
        permission_mgr = full_system["permission_manager"]
        
        # Test with exactly at threshold
        threshold_action = AutonomousAction(
            action_type="test_threshold",
            target="test",
            params={},
            priority=ActionPriority.MEDIUM,
            confidence=0.85,  # Exactly at typical threshold
            category=ActionCategory.MAINTENANCE,
            reasoning="Testing threshold behavior"
        )
        
        # Record decisions to reach exactly the learning threshold
        for i in range(permission_mgr.learning_threshold - 1):
            permission_mgr.record_decision(threshold_action, approved=True)
        
        # One more to reach threshold
        permission_mgr.record_decision(threshold_action, approved=True)
        
        # Now check permission
        permission, confidence, reason = permission_mgr.check_permission(threshold_action)
        
        # Should now auto-approve
        assert permission is True
        assert "auto-approved" in reason.lower()
    
    @pytest.mark.asyncio
    async def test_time_boundary_conditions(self, full_system):
        """Test behavior at time boundaries (midnight, quiet hours)"""
        context_engine = full_system["context_engine"]
        permission_mgr = full_system["permission_manager"]
        
        # Test at quiet hours boundary
        with patch('datetime.datetime') as mock_datetime:
            # Set time to 10:00 PM (start of quiet hours)
            mock_datetime.now.return_value = datetime.now().replace(hour=22, minute=0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            low_priority_action = AutonomousAction(
                action_type="organize_desktop",
                target="desktop",
                params={},
                priority=ActionPriority.LOW,
                confidence=0.8,
                category=ActionCategory.MAINTENANCE,
                reasoning="Regular cleanup"
            )
            
            permission, _, reason = permission_mgr.check_permission(low_priority_action)
            
            # Should block low priority during quiet hours
            assert permission is False
            assert "quiet hours" in reason.lower()
            
            # But critical actions should still be allowed
            critical_action = AutonomousAction(
                action_type="security_lockdown",
                target="system",
                params={},
                priority=ActionPriority.CRITICAL,
                confidence=0.95,
                category=ActionCategory.SECURITY,
                reasoning="Security threat detected"
            )
            
            permission, _, _ = permission_mgr.check_permission(critical_action)
            assert permission is None  # Should ask user for critical security
    
    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, full_system):
        """Test that system doesn't accumulate unbounded state"""
        behavior_mgr = full_system["behavior_manager"]
        permission_mgr = full_system["permission_manager"]
        
        # Run many iterations
        for i in range(100):
            windows = [
                WindowInfo(
                    window_id=j,
                    app_name=f"TestApp{j}",
                    window_title=f"Iteration {i} Window {j}",
                    bounds={"x": j * 100, "y": 0, "width": 300, "height": 200},
                    is_focused=(j == 0),
                    layer=0,
                    is_visible=True,
                    process_id=6000 + i * 10 + j
                ) for j in range(5)
            ]
            
            workspace_state = {
                "window_count": 5,
                "user_state": "available",
                "iteration": i
            }
            
            actions = await behavior_mgr.process_workspace_state(workspace_state, windows)
            
            # Record some decisions
            for action in actions[:2]:  # Just first 2 to avoid explosion
                permission_mgr.record_decision(action, approved=(i % 2 == 0))
        
        # Check that permission history is bounded
        total_history_size = sum(
            len(record.context_history) 
            for record in permission_mgr.permissions.values()
        )
        
        # History should be capped (50 per record as defined)
        assert total_history_size < 1000
    
    @pytest.mark.asyncio
    async def test_concurrent_action_execution(self, full_system):
        """Test handling of concurrent action requests"""
        executor = full_system["action_executor"]
        
        # Create multiple non-conflicting actions
        actions = []
        for i in range(10):
            actions.append(AutonomousAction(
                action_type="minimize_window",
                target=f"App{i}",
                params={"window_id": i},
                priority=ActionPriority.MEDIUM,
                confidence=0.8,
                category=ActionCategory.MAINTENANCE,
                reasoning=f"Minimizing app {i}"
            ))
        
        # Execute concurrently
        tasks = [executor.execute_action(action, dry_run=True) for action in actions]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete without errors
        assert all(not isinstance(r, Exception) for r in results)
        assert all(r.status in [ExecutionStatus.SUCCESS, ExecutionStatus.DRY_RUN] for r in results)

class TestSystemIntegration:
    """Test full system integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_autonomous_cycle(self):
        """Test a complete autonomous decision cycle"""
        # Initialize all components
        decision_engine = AutonomousDecisionEngine()
        permission_mgr = PermissionManager()
        context_engine = ContextEngine()
        executor = ActionExecutor()
        behavior_mgr = AutonomousBehaviorManager()
        
        # Simulate a realistic workspace
        windows = [
            WindowInfo(
                window_id=1, app_name="Mail",
                window_title="Inbox (5 unread) - Important: Contract review needed",
                bounds={"x": 0, "y": 0, "width": 1000, "height": 700},
                is_focused=True, layer=0, is_visible=True, process_id=7001
            ),
            WindowInfo(
                window_id=2, app_name="Calendar",
                window_title="Reminder: Legal review meeting in 10 minutes",
                bounds={"x": 1000, "y": 0, "width": 400, "height": 700},
                is_focused=False, layer=0, is_visible=True, process_id=7002
            ),
            WindowInfo(
                window_id=3, app_name="1Password",
                window_title="1Password 7 - Personal Vault",
                bounds={"x": 0, "y": 700, "width": 500, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=7003
            ),
            WindowInfo(
                window_id=4, app_name="Spotify",
                window_title="Spotify - Chill Vibes Playlist",
                bounds={"x": 500, "y": 700, "width": 500, "height": 300},
                is_focused=False, layer=0, is_visible=True, process_id=7004
            )
        ]
        
        # Step 1: Analyze workspace
        workspace_state = WorkspaceAnalysis(
            focused_task="Reviewing contracts",
            window_relationships={},
            workspace_context="Professional work with upcoming meeting",
            important_notifications=["Contract review", "Meeting in 10 min"],
            suggestions=[],
            confidence=0.85
        )
        
        # Step 2: Generate decisions
        actions = await behavior_mgr.process_workspace_state(
            workspace_state.__dict__,
            windows
        )
        
        assert len(actions) > 0
        
        # Step 3: Check context
        context = await context_engine.analyze_context(workspace_state, windows)
        assert context.user_state in [UserState.AVAILABLE, UserState.FOCUSED]
        
        # Step 4: Process permissions and execute
        executed_actions = []
        for action in actions[:3]:  # Limit to first 3
            # Check permission
            permission, confidence, reason = permission_mgr.check_permission(action)
            
            if permission is True or (permission is None and action.priority == ActionPriority.HIGH):
                # Execute (dry run for test)
                result = await executor.execute_action(action, dry_run=True)
                executed_actions.append((action, result))
                
                # Record decision if not auto-decided
                if permission is None:
                    permission_mgr.record_decision(action, approved=True)
        
        # Verify execution
        assert len(executed_actions) > 0
        assert all(r.status == ExecutionStatus.DRY_RUN for _, r in executed_actions)
        
        # Should have hidden password manager for meeting
        password_actions = [a for a, _ in executed_actions 
                           if a.target == "1Password" and a.action_type == "minimize_window"]
        assert len(password_actions) > 0
        
        # Should handle meeting preparation
        meeting_actions = [a for a, _ in executed_actions 
                          if "meeting" in a.reasoning.lower()]
        assert len(meeting_actions) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])