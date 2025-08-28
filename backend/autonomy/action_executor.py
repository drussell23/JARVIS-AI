#!/usr/bin/env python3
"""
Action Executor for JARVIS Autonomous System
Executes autonomous actions with safety and rollback capabilities
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from system_control.macos_controller import MacOSController
from .autonomous_decision_engine import AutonomousAction, ActionCategory, ActionPriority

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Status of action execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"

@dataclass
class ExecutionResult:
    """Result of action execution"""
    action: AutonomousAction
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime]
    result_data: Optional[Dict[str, Any]]
    error: Optional[str]
    rollback_available: bool
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get execution time in seconds"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action': self.action.to_dict(),
            'status': self.status.value,
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'execution_time': self.execution_time,
            'result_data': self.result_data,
            'error': self.error,
            'rollback_available': self.rollback_available
        }

class ActionExecutor:
    """Executes autonomous actions with safety mechanisms"""
    
    def __init__(self):
        self.macos_controller = MacOSController()
        self.execution_history = []
        self.rollback_stack = []
        
        # Action handlers mapped by action type
        self.action_handlers = {
            'handle_notifications': self._handle_notifications,
            'prepare_meeting': self._prepare_meeting,
            'organize_workspace': self._organize_workspace,
            'security_alert': self._handle_security_alert,
            'respond_message': self._respond_to_message,
            'cleanup_workspace': self._cleanup_workspace,
            'minimize_distractions': self._minimize_distractions,
            'routine_automation': self._execute_routine,
            'handle_urgent_item': self._handle_urgent_item
        }
        
        # Safety limits
        self.limits = {
            'max_windows_close': 5,      # Max windows to close at once
            'max_apps_launch': 3,        # Max apps to launch at once
            'max_notifications': 10,     # Max notifications to handle at once
            'execution_timeout': 30      # Seconds before timeout
        }
        
        # Dry run mode for testing
        self.dry_run = False
    
    async def execute_action(self, action: AutonomousAction, 
                           dry_run: bool = False) -> ExecutionResult:
        """Execute an autonomous action with safety checks"""
        self.dry_run = dry_run
        
        # Create execution result
        result = ExecutionResult(
            action=action,
            status=ExecutionStatus.PENDING,
            started_at=datetime.now(),
            completed_at=None,
            result_data=None,
            error=None,
            rollback_available=False
        )
        
        try:
            # Pre-execution safety checks
            safety_check = await self._safety_check(action)
            if not safety_check['safe']:
                result.status = ExecutionStatus.FAILED
                result.error = f"Safety check failed: {safety_check['reason']}"
                result.completed_at = datetime.now()
                return result
            
            # Get appropriate handler
            handler = self.action_handlers.get(action.action_type)
            if not handler:
                result.status = ExecutionStatus.FAILED
                result.error = f"No handler for action type: {action.action_type}"
                result.completed_at = datetime.now()
                return result
            
            # Execute with timeout
            result.status = ExecutionStatus.EXECUTING
            logger.info(f"Executing action: {action.action_type} on {action.target}")
            
            execution_result = await asyncio.wait_for(
                handler(action),
                timeout=self.limits['execution_timeout']
            )
            
            # Update result
            result.status = ExecutionStatus.SUCCESS
            result.result_data = execution_result
            result.rollback_available = execution_result.get('rollback_available', False)
            
            # Store rollback info if available
            if result.rollback_available:
                self.rollback_stack.append({
                    'action': action,
                    'rollback_data': execution_result.get('rollback_data'),
                    'timestamp': datetime.now()
                })
            
        except asyncio.TimeoutError:
            result.status = ExecutionStatus.FAILED
            result.error = "Execution timeout"
            logger.error(f"Action timeout: {action.action_type}")
            
        except Exception as e:
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            logger.error(f"Action failed: {action.action_type} - {e}")
        
        finally:
            result.completed_at = datetime.now()
            self._record_execution(result)
        
        return result
    
    async def _safety_check(self, action: AutonomousAction) -> Dict[str, Any]:
        """Perform safety checks before execution"""
        # Check action limits
        if action.action_type == 'cleanup_workspace':
            window_count = len(action.params.get('window_ids', []))
            if window_count > self.limits['max_windows_close']:
                return {
                    'safe': False,
                    'reason': f"Attempting to close {window_count} windows, limit is {self.limits['max_windows_close']}"
                }
        
        # Check recent failures
        recent_failures = self._get_recent_failures(action.action_type)
        if len(recent_failures) >= 3:
            return {
                'safe': False,
                'reason': f"Action type {action.action_type} has failed {len(recent_failures)} times recently"
            }
        
        # Check system state
        if action.category == ActionCategory.SECURITY:
            # Extra checks for security actions
            if not await self._verify_security_context():
                return {
                    'safe': False,
                    'reason': "Security context verification failed"
                }
        
        return {'safe': True, 'reason': None}
    
    async def _handle_notifications(self, action: AutonomousAction) -> Dict[str, Any]:
        """Handle notification management"""
        app = action.params['app']
        count = action.params.get('count', 0)
        window_id = action.params.get('window_id')
        
        logger.info(f"Handling {count} notifications in {app}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would handle {count} notifications in {app}",
                'rollback_available': False
            }
        
        # Focus the application
        await self.macos_controller.focus_application(app)
        await asyncio.sleep(0.5)
        
        # Mark notifications as read (app-specific logic)
        if app.lower() in ['discord', 'slack']:
            # Keyboard shortcut to mark as read
            await self.macos_controller.send_keystroke("shift+cmd+a")  # Mark all as read
        
        return {
            'success': True,
            'notifications_handled': count,
            'app': app,
            'rollback_available': False
        }
    
    async def _prepare_meeting(self, action: AutonomousAction) -> Dict[str, Any]:
        """Prepare workspace for meeting"""
        meeting_info = action.params.get('meeting_info', '')
        minutes_until = action.params.get('minutes_until', 5)
        
        logger.info(f"Preparing for meeting in {minutes_until} minutes")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would prepare for meeting: {meeting_info}",
                'rollback_available': True
            }
        
        # Store current state for rollback
        current_state = await self._capture_workspace_state()
        
        # Hide sensitive windows
        sensitive_apps = ['1Password', 'Banking', 'Terminal']
        for app in sensitive_apps:
            try:
                await self.macos_controller.hide_application(app)
            except:
                pass  # App might not be open
        
        # Open meeting app if needed
        if 'zoom' in meeting_info.lower():
            await self.macos_controller.open_application('Zoom')
        
        # Mute notifications
        await self.macos_controller.toggle_do_not_disturb(True)
        
        return {
            'success': True,
            'actions_taken': ['hid_sensitive_windows', 'opened_meeting_app', 'enabled_dnd'],
            'rollback_available': True,
            'rollback_data': current_state
        }
    
    async def _organize_workspace(self, action: AutonomousAction) -> Dict[str, Any]:
        """Organize windows for better productivity"""
        task = action.params.get('task', '')
        arrangement = action.params.get('window_arrangement', {})
        
        logger.info(f"Organizing workspace for: {task}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would organize workspace for {task}",
                'rollback_available': True
            }
        
        # Store current state
        current_state = await self._capture_workspace_state()
        
        # Implement window arrangement
        organized_count = 0
        
        # Focus primary windows
        for window_id in arrangement.get('primary_focus', []):
            # In production, use actual window manipulation
            logger.info(f"Would focus window {window_id}")
            organized_count += 1
        
        # Minimize distractions
        for window_id in arrangement.get('minimize', []):
            logger.info(f"Would minimize window {window_id}")
            organized_count += 1
        
        return {
            'success': True,
            'windows_organized': organized_count,
            'task': task,
            'rollback_available': True,
            'rollback_data': current_state
        }
    
    async def _handle_security_alert(self, action: AutonomousAction) -> Dict[str, Any]:
        """Handle security-related actions"""
        app = action.params['app']
        concern_type = action.params['concern_type']
        
        logger.warning(f"Security alert: {concern_type} in {app}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would handle security concern in {app}",
                'rollback_available': False
            }
        
        # Take immediate action based on concern type
        if concern_type == 'sensitive_content':
            # Hide the application immediately
            await self.macos_controller.hide_application(app)
            
            # Take screenshot for audit
            screenshot_path = await self.macos_controller.take_screenshot(
                f"security_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            
            return {
                'success': True,
                'action_taken': 'hid_application',
                'screenshot': screenshot_path,
                'rollback_available': True,
                'rollback_data': {'app': app, 'was_visible': True}
            }
        
        return {
            'success': False,
            'error': f"Unknown concern type: {concern_type}",
            'rollback_available': False
        }
    
    async def _respond_to_message(self, action: AutonomousAction) -> Dict[str, Any]:
        """Respond to messages intelligently"""
        app = action.params['app']
        message_preview = action.params.get('message_preview', '')
        suggested_response = action.params.get('suggested_response', '')
        
        logger.info(f"Responding to message in {app}")
        
        # For safety, we don't actually send messages automatically
        # Instead, we prepare the response for user confirmation
        
        return {
            'success': True,
            'action': 'prepared_response',
            'app': app,
            'suggested_response': suggested_response,
            'requires_confirmation': True,
            'rollback_available': False
        }
    
    async def _cleanup_workspace(self, action: AutonomousAction) -> Dict[str, Any]:
        """Clean up workspace by closing unnecessary windows"""
        cleanup_type = action.params.get('type', 'general')
        window_ids = action.params.get('window_ids', [])
        
        logger.info(f"Cleaning up workspace: {cleanup_type}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would close {len(window_ids)} windows",
                'rollback_available': True
            }
        
        # Store current state
        current_state = await self._capture_workspace_state()
        
        # Close windows (with safety limit)
        closed_count = 0
        for window_id in window_ids[:self.limits['max_windows_close']]:
            # In production, implement actual window closing
            logger.info(f"Would close window {window_id}")
            closed_count += 1
        
        return {
            'success': True,
            'windows_closed': closed_count,
            'cleanup_type': cleanup_type,
            'rollback_available': True,
            'rollback_data': current_state
        }
    
    async def _minimize_distractions(self, action: AutonomousAction) -> Dict[str, Any]:
        """Minimize distracting applications"""
        distraction_apps = action.params.get('distraction_apps', [])
        focus_task = action.params.get('focus_task', '')
        
        logger.info(f"Minimizing distractions for: {focus_task}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would minimize {len(distraction_apps)} distracting apps",
                'rollback_available': True
            }
        
        minimized = []
        for app in distraction_apps:
            try:
                await self.macos_controller.hide_application(app)
                minimized.append(app)
            except Exception as e:
                logger.warning(f"Could not minimize {app}: {e}")
        
        # Enable focus mode
        await self.macos_controller.toggle_do_not_disturb(True)
        
        return {
            'success': True,
            'apps_minimized': minimized,
            'focus_enabled': True,
            'rollback_available': True,
            'rollback_data': {'apps': minimized, 'dnd_was_on': False}
        }
    
    async def _execute_routine(self, action: AutonomousAction) -> Dict[str, Any]:
        """Execute a learned routine"""
        routine_name = action.params.get('routine_name', '')
        expected_apps = action.params.get('expected_apps', [])
        
        logger.info(f"Executing routine: {routine_name}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would execute {routine_name} routine",
                'rollback_available': False
            }
        
        # Open expected apps
        opened_apps = []
        for app in expected_apps[:self.limits['max_apps_launch']]:
            try:
                await self.macos_controller.open_application(app)
                opened_apps.append(app)
                await asyncio.sleep(1)  # Give apps time to launch
            except Exception as e:
                logger.warning(f"Could not open {app}: {e}")
        
        return {
            'success': True,
            'routine': routine_name,
            'apps_opened': opened_apps,
            'rollback_available': False
        }
    
    async def _handle_urgent_item(self, action: AutonomousAction) -> Dict[str, Any]:
        """Handle urgent items that need attention"""
        app = action.params['app']
        urgency_score = action.params.get('urgency_score', 0)
        title = action.params.get('title', '')
        
        logger.info(f"Handling urgent item in {app}: {title}")
        
        if self.dry_run:
            return {
                'success': True,
                'message': f"Would handle urgent item in {app}",
                'rollback_available': False
            }
        
        # Focus the application
        await self.macos_controller.focus_application(app)
        
        # If very urgent, also notify
        if urgency_score > 0.8:
            await self.macos_controller.show_notification(
                title=f"Urgent: {app}",
                message=title[:100]  # Truncate long titles
            )
        
        return {
            'success': True,
            'app_focused': app,
            'urgency_score': urgency_score,
            'notification_shown': urgency_score > 0.8,
            'rollback_available': False
        }
    
    async def _capture_workspace_state(self) -> Dict[str, Any]:
        """Capture current workspace state for rollback"""
        # In production, capture actual window positions, app states, etc.
        return {
            'timestamp': datetime.now().isoformat(),
            'open_apps': [],  # Would list actual open apps
            'window_positions': {},  # Would capture positions
            'dnd_enabled': False  # Would check actual DND state
        }
    
    async def _verify_security_context(self) -> bool:
        """Verify it's safe to perform security actions"""
        # Additional security checks
        # In production, verify user presence, check for anomalies, etc.
        return True
    
    def _get_recent_failures(self, action_type: str, hours: int = 1) -> List[ExecutionResult]:
        """Get recent failures for an action type"""
        cutoff = datetime.now().timestamp() - (hours * 3600)
        
        return [
            result for result in self.execution_history
            if result.action.action_type == action_type
            and result.status == ExecutionStatus.FAILED
            and result.started_at.timestamp() > cutoff
        ]
    
    def _record_execution(self, result: ExecutionResult):
        """Record execution result for analysis"""
        self.execution_history.append(result)
        
        # Keep only recent history (last 1000 executions)
        if len(self.execution_history) > 1000:
            self.execution_history = self.execution_history[-1000:]
        
        # Log result
        if result.status == ExecutionStatus.SUCCESS:
            logger.info(f"Action completed: {result.action.action_type} in {result.execution_time:.1f}s")
        else:
            logger.error(f"Action failed: {result.action.action_type} - {result.error}")
    
    async def rollback_last_action(self) -> bool:
        """Rollback the last rollback-able action"""
        if not self.rollback_stack:
            logger.warning("No actions to rollback")
            return False
        
        rollback_info = self.rollback_stack.pop()
        action = rollback_info['action']
        rollback_data = rollback_info['rollback_data']
        
        logger.info(f"Rolling back: {action.action_type}")
        
        try:
            # Implement rollback based on action type
            if action.action_type == 'prepare_meeting':
                # Restore hidden windows, disable DND, etc.
                await self.macos_controller.toggle_do_not_disturb(False)
                # Would restore window states from rollback_data
                
            elif action.action_type == 'minimize_distractions':
                # Restore minimized apps
                for app in rollback_data.get('apps', []):
                    await self.macos_controller.show_application(app)
                if not rollback_data.get('dnd_was_on', False):
                    await self.macos_controller.toggle_do_not_disturb(False)
            
            # Add more rollback implementations as needed
            
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        total = len(self.execution_history)
        if total == 0:
            return {'total_executions': 0}
        
        success_count = sum(1 for r in self.execution_history if r.status == ExecutionStatus.SUCCESS)
        failed_count = sum(1 for r in self.execution_history if r.status == ExecutionStatus.FAILED)
        
        avg_execution_time = sum(
            r.execution_time for r in self.execution_history 
            if r.execution_time is not None
        ) / max(1, sum(1 for r in self.execution_history if r.execution_time is not None))
        
        # Group by action type
        action_stats = {}
        for result in self.execution_history:
            action_type = result.action.action_type
            if action_type not in action_stats:
                action_stats[action_type] = {'success': 0, 'failed': 0}
            
            if result.status == ExecutionStatus.SUCCESS:
                action_stats[action_type]['success'] += 1
            else:
                action_stats[action_type]['failed'] += 1
        
        return {
            'total_executions': total,
            'success_count': success_count,
            'failed_count': failed_count,
            'success_rate': success_count / total,
            'average_execution_time': avg_execution_time,
            'action_stats': action_stats,
            'rollback_available': len(self.rollback_stack)
        }

async def test_action_executor():
    """Test the action executor"""
    executor = ActionExecutor()
    
    # Create test action
    from .autonomous_decision_engine import AutonomousAction, ActionCategory, ActionPriority
    
    test_action = AutonomousAction(
        action_type='handle_notifications',
        target='Discord',
        params={
            'app': 'Discord',
            'count': 5,
            'window_id': 123
        },
        priority=ActionPriority.MEDIUM,
        confidence=0.8,
        category=ActionCategory.NOTIFICATION,
        reasoning="5 unread messages in Discord"
    )
    
    print("🚀 Action Executor Test")
    print("=" * 50)
    
    # Execute in dry run mode
    result = await executor.execute_action(test_action, dry_run=True)
    
    print(f"\nAction: {test_action.action_type}")
    print(f"Target: {test_action.target}")
    print(f"Status: {result.status.value}")
    print(f"Execution Time: {result.execution_time:.2f}s" if result.execution_time else "N/A")
    
    if result.result_data:
        print(f"Result: {result.result_data}")
    
    if result.error:
        print(f"Error: {result.error}")
    
    # Test another action
    meeting_action = AutonomousAction(
        action_type='prepare_meeting',
        target='calendar',
        params={
            'meeting_info': 'Team Standup in 5 minutes',
            'minutes_until': 5
        },
        priority=ActionPriority.HIGH,
        confidence=0.9,
        category=ActionCategory.CALENDAR,
        reasoning="Meeting starting soon"
    )
    
    result2 = await executor.execute_action(meeting_action, dry_run=True)
    
    print(f"\n\nAction 2: {meeting_action.action_type}")
    print(f"Status: {result2.status.value}")
    print(f"Rollback Available: {result2.rollback_available}")
    
    # Show stats
    stats = executor.get_execution_stats()
    print(f"\n📊 Execution Statistics:")
    print(f"Total Executions: {stats['total_executions']}")
    print(f"Success Rate: {stats.get('success_rate', 0):.1%}")

if __name__ == "__main__":
    asyncio.run(test_action_executor())