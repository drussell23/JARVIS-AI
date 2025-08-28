#!/usr/bin/env python3
"""
JARVIS Workspace Intelligence Integration
Connects multi-window awareness to JARVIS voice commands
"""

import asyncio
import logging
import re
from typing import Optional, Dict, Any, List
from datetime import datetime

from .workspace_analyzer import WorkspaceAnalyzer, WorkspaceAnalysis
from .window_detector import WindowDetector
from .proactive_insights import ProactiveInsights, Insight
from .workspace_optimizer import WorkspaceOptimizer, WorkspaceOptimization
from .meeting_preparation import MeetingPreparationSystem, MeetingContext, MeetingAlert
from .workflow_learning import WorkflowLearningSystem, WorkflowPrediction
from .privacy_controls import PrivacyControlSystem

# Import autonomous components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from autonomy.autonomous_decision_engine import AutonomousDecisionEngine, AutonomousAction
from autonomy.permission_manager import PermissionManager
from autonomy.context_engine import ContextEngine
from autonomy.action_executor import ActionExecutor

logger = logging.getLogger(__name__)

class JARVISWorkspaceIntelligence:
    """Integrates workspace awareness into JARVIS"""
    
    def __init__(self):
        self.workspace_analyzer = WorkspaceAnalyzer()
        self.window_detector = WindowDetector()
        self.proactive_insights = ProactiveInsights()
        self.workspace_optimizer = WorkspaceOptimizer()
        self.meeting_system = MeetingPreparationSystem()
        self.workflow_learning = WorkflowLearningSystem()
        self.privacy_controls = PrivacyControlSystem()
        self.last_analysis = None
        self.monitoring_active = False
        self.pending_insights: List[Insight] = []
        
        # Initialize autonomous components
        self.autonomous_engine = AutonomousDecisionEngine()
        self.permission_manager = PermissionManager()
        self.context_engine = ContextEngine()
        self.action_executor = ActionExecutor()
        self.autonomous_enabled = False
        self.autonomous_task = None
        
        # Query patterns for workspace intelligence
        self.workspace_patterns = {
            'current_work': [
                "what am i working on",
                "what am i doing",
                "what's my current task",
                "describe my work",
                "analyze my workspace"
            ],
            'messages': [
                "do i have messages",
                "any messages",
                "check messages",
                "unread messages",
                "check discord",
                "check slack"
            ],
            'errors': [
                "any errors",
                "show errors",
                "terminal errors",
                "what's broken",
                "debug help"
            ],
            'windows': [
                "what windows are open",
                "list windows",
                "show applications",
                "what apps are running"
            ],
            'context': [
                "what's on my screen",
                "describe my workspace",
                "workspace summary",
                "what can you see"
            ],
            'optimize': [
                "optimize my workspace",
                "improve my setup",
                "organize windows",
                "workspace suggestions",
                "productivity tips"
            ],
            'meeting': [
                "prepare for meeting",
                "meeting setup",
                "hide sensitive windows",
                "screen sharing mode",
                "meeting layout"
            ],
            'privacy': [
                "privacy mode",
                "hide windows",
                "sensitive content",
                "private mode",
                "privacy settings"
            ],
            'workflow': [
                "workflow suggestions",
                "what should i open",
                "missing windows",
                "usual setup",
                "workflow patterns"
            ]
        }
    
    async def handle_workspace_command(self, command: str) -> str:
        """Handle workspace-aware commands"""
        command_lower = command.lower()
        
        # Store the command for use in formatting
        self._last_command = command
        
        # Check for autonomous commands first
        if any(word in command_lower for word in ['autonomous', 'rollback', 'permission stats']):
            return await self.handle_autonomous_command(command)
        
        # Determine query type
        query_type = self._determine_query_type(command_lower)
        
        # Record window state for workflow learning
        self.workflow_learning.record_window_state()
        
        # Analyze workspace with appropriate context
        try:
            # Apply privacy filtering
            all_windows = self.window_detector.get_all_windows()
            filtered_windows, blocked = self.privacy_controls.filter_windows(all_windows)
            
            # Different handling based on query type
            if query_type == 'meeting':
                return await self._format_meeting_response()
            elif query_type == 'privacy':
                return self._format_privacy_response(command_lower)
            elif query_type == 'workflow':
                return self._format_workflow_response()
            
            # For other queries, use filtered windows
            analysis = await self.workspace_analyzer.analyze_workspace(command)
            self.last_analysis = analysis
            
            # Format response based on query type
            if query_type == 'current_work':
                return self._format_work_response(analysis)
            elif query_type == 'messages':
                return self._format_messages_response(analysis)
            elif query_type == 'errors':
                return self._format_errors_response(analysis)
            elif query_type == 'windows':
                return self._format_windows_response()
            elif query_type == 'optimize':
                return await self._format_optimization_response()
            else:
                return self._format_general_response(analysis)
                
        except Exception as e:
            logger.error(f"Workspace analysis error: {e}")
            return "I'm having trouble analyzing your workspace at the moment, sir."
    
    def _determine_query_type(self, command: str) -> str:
        """Determine the type of workspace query"""
        for query_type, patterns in self.workspace_patterns.items():
            if any(pattern in command for pattern in patterns):
                return query_type
        return 'context'
    
    def _format_work_response(self, analysis: WorkspaceAnalysis) -> str:
        """Format response about current work"""
        # Clean up the focused task string
        task = analysis.focused_task
        
        # Remove any numbered list formatting
        task = re.sub(r'^\d+[\.\)]\s*', '', task)
        
        # Remove any section markers that might have leaked through
        task = task.replace("PRIMARY TASK:", "").replace("1. PRIMARY TASK:", "")
        task = task.replace("CONTEXT:", "").replace("2. CONTEXT:", "")
        task = task.replace("NOTIFICATIONS:", "").replace("3. NOTIFICATIONS:", "")
        task = task.replace("SUGGESTIONS:", "").replace("4. SUGGESTIONS:", "")
        
        # Remove verbose prefixes
        task = task.replace("Based on the information provided,", "")
        task = task.replace("Looking at your workspace,", "")
        task = task.replace("I can see that", "")
        task = task.strip()
        
        # If task is empty or generic, get focused window info directly
        if not task or "working on" not in task.lower():
            windows = self.window_detector.get_all_windows()
            focused_window = next((w for w in windows if w.is_focused), None)
            
            if focused_window:
                if focused_window.window_title:
                    task = f"You're working in {focused_window.app_name} on {focused_window.window_title}"
                else:
                    task = f"You're focused on {focused_window.app_name}"
        
        # Ensure task starts with "You're" for consistency
        if task and not task.startswith("You're") and not task.startswith("Working"):
            if "working" in task.lower():
                task = task
            else:
                task = f"You're {task}"
        
        # Build response
        if task:
            response = f"Sir, {task}."
        else:
            response = "Sir, analyzing your workspace."
        
        # Add window relationship info if available
        if analysis.window_relationships and "supporting_apps" in analysis.window_relationships:
            apps_info = analysis.window_relationships["supporting_apps"][0]
            # Only add if it's concise
            if len(apps_info) < 30:
                response += f" {apps_info}."
        
        # Only add critical notifications
        if analysis.important_notifications:
            notification = analysis.important_notifications[0]
            # Clean up notification text
            notification = notification.replace("NOTIFICATIONS:", "").strip()
            if notification and any(word in notification.lower() for word in ['error', 'failed', 'warning', 'urgent']):
                response += f" Alert: {notification}"
        
        return response.strip()
    
    def _format_messages_response(self, analysis: WorkspaceAnalysis) -> str:
        """Format response about messages"""
        # Look for communication apps in the analysis
        workspace = analysis.workspace_context.lower()
        
        messages_found = []
        
        if 'discord' in workspace:
            messages_found.append("Discord")
        if 'slack' in workspace:
            messages_found.append("Slack")
        if 'messages' in workspace:
            messages_found.append("Messages")
        if 'mail' in workspace:
            messages_found.append("Mail")
        
        if messages_found:
            # Check for unread indicators
            unread_count = sum(1 for n in analysis.important_notifications 
                             if 'unread' in n.lower() or 'message' in n.lower())
            
            if unread_count > 0:
                response = f"Sir, you have unread messages in {', '.join(messages_found)}."
            else:
                response = f"Sir, {', '.join(messages_found)} {'is' if len(messages_found) == 1 else 'are'} open but no new messages."
        else:
            response = "Sir, no communication apps are currently open."
        
        return response
    
    def _format_errors_response(self, analysis: WorkspaceAnalysis) -> str:
        """Format response about errors"""
        # Check for error-related content
        error_indicators = ['error', 'failed', 'exception', 'warning', 'bug']
        
        # Check in workspace context and notifications
        errors_found = any(indicator in analysis.workspace_context.lower() for indicator in error_indicators)
        
        error_notifications = [n for n in analysis.important_notifications 
                             if any(ind in n.lower() for ind in error_indicators)]
        
        if errors_found or error_notifications:
            if error_notifications:
                response = f"Sir, {error_notifications[0]}."
            else:
                response = "Sir, I detect error indicators in your workspace."
        else:
            response = "Sir, no errors detected in your current workspace."
        
        return response
    
    def _format_windows_response(self) -> str:
        """Format response listing open windows"""
        windows = self.window_detector.get_all_windows()
        
        if not windows:
            return "Sir, no windows detected."
        
        # Group by application
        apps = {}
        for window in windows:
            if window.app_name not in apps:
                apps[window.app_name] = 0
            apps[window.app_name] += 1
        
        # Get focused window
        focused = next((w for w in windows if w.is_focused), None)
        
        # Build concise response
        response = f"Sir, {len(windows)} windows across {len(apps)} applications."
        
        if focused:
            title_part = f": {focused.window_title}" if focused.window_title else ""
            response += f" Focused on {focused.app_name}{title_part}."
        
        return response
    
    def _format_general_response(self, analysis: WorkspaceAnalysis) -> str:
        """Format general workspace response"""
        # For "What's on my screen?" we want to list all windows
        command = getattr(self, '_last_command', '').lower()
        
        if "what's on my screen" in command or "what is on my screen" in command:
            # Get fresh window data for comprehensive list
            windows = self.window_detector.get_all_windows()
            
            if not windows:
                return "Sir, no windows detected on your screen."
            
            # Group by application
            apps = {}
            focused_app = None
            for window in windows:
                if window.app_name not in apps:
                    apps[window.app_name] = []
                apps[window.app_name].append(window)
                if window.is_focused:
                    focused_app = window.app_name
            
            # Build response listing all apps
            response = f"Sir, you have {len(windows)} windows open: "
            app_list = list(apps.keys())
            
            # Put focused app first
            if focused_app and focused_app in app_list:
                app_list.remove(focused_app)
                app_list.insert(0, f"{focused_app} (focused)")
            
            response += ", ".join(app_list[:8])  # List up to 8 apps
            if len(apps) > 8:
                response += f", and {len(apps) - 8} more"
            response += "."
            
            return response
        
        # For other general queries, use the analyzed task
        task = analysis.focused_task
        
        # Remove any section markers and numbered lists
        task = re.sub(r'^\d+[\.\)]\s*', '', task)
        task = task.replace("PRIMARY TASK:", "").replace("1. PRIMARY TASK:", "")
        task = task.replace("CONTEXT:", "").replace("2. CONTEXT:", "")
        task = task.replace("NOTIFICATIONS:", "").replace("3. NOTIFICATIONS:", "")
        task = task.replace("SUGGESTIONS:", "").replace("4. SUGGESTIONS:", "")
        task = task.strip()
        
        if task:
            response = f"Sir, {task}."
        else:
            response = "Sir, analyzing your workspace."
        
        # Add supporting apps if available
        if analysis.window_relationships and "supporting_apps" in analysis.window_relationships:
            apps_info = analysis.window_relationships["supporting_apps"][0]
            if len(apps_info) < 30:
                response += f" {apps_info}."
        
        # Only add critical notifications
        if analysis.important_notifications:
            notification = analysis.important_notifications[0]
            notification = notification.replace("NOTIFICATIONS:", "").strip()
            if notification and any(word in notification.lower() for word in ['error', 'failed', 'warning', 'urgent']):
                response += f" Alert: {notification}"
        
        return response.strip()
    
    async def _format_optimization_response(self) -> str:
        """Format workspace optimization suggestions"""
        optimization = self.workspace_optimizer.analyze_workspace()
        return optimization.to_jarvis_message()
    
    async def _format_meeting_response(self) -> str:
        """Format meeting preparation response"""
        context, alerts = self.meeting_system.analyze_meeting_preparation()
        
        response_parts = []
        
        # Check meeting readiness
        if context.meeting_app:
            response_parts.append(f"Sir, {context.meeting_app.app_name} is ready for your meeting")
        else:
            response_parts.append("Sir, no meeting application detected")
        
        # Report alerts
        if alerts:
            high_priority = [a for a in alerts if a.severity == 'high']
            if high_priority:
                alert = high_priority[0]
                response_parts.append(alert.message)
                if alert.suggestions:
                    response_parts.append(alert.suggestions[0])
        
        # Check sensitive windows
        if context.sensitive_windows:
            response_parts.append(f"I've detected {len(context.sensitive_windows)} sensitive windows that should be hidden")
        
        # Suggest layout
        if context.meeting_app:
            layout = self.meeting_system.get_meeting_layout(context)
            if layout:
                response_parts.append(f"I can arrange your windows in {layout.layout_type} mode")
        
        return ". ".join(response_parts)
    
    def _format_privacy_response(self, command: str) -> str:
        """Format privacy control response"""
        # Check for specific privacy commands
        if "meeting" in command and "mode" in command:
            success = self.privacy_controls.set_privacy_mode('meeting')
            if success:
                return "Sir, I've activated meeting privacy mode. Sensitive windows will be hidden during screen sharing"
        
        elif "private" in command or "privacy mode" in command:
            success = self.privacy_controls.set_privacy_mode('private')
            if success:
                return "Sir, maximum privacy mode activated. All windows are excluded from analysis"
        
        elif "normal" in command:
            success = self.privacy_controls.set_privacy_mode('normal')
            if success:
                return "Sir, privacy mode set to normal. Standard protections are active"
        
        # Generate privacy report
        report = self.privacy_controls.generate_privacy_report()
        
        response_parts = [
            f"Sir, privacy mode is set to '{report['current_mode']}'",
            f"{report['statistics']['blocked_windows']} windows are currently blocked"
        ]
        
        if report['statistics']['sensitive_windows'] > 0:
            response_parts.append(f"{report['statistics']['sensitive_windows']} windows contain sensitive content")
        
        if report['recommendations']:
            response_parts.append(report['recommendations'][0])
        
        return ". ".join(response_parts)
    
    def _format_workflow_response(self) -> str:
        """Format workflow learning response"""
        predictions = self.workflow_learning.predict_workflow()
        
        if not predictions:
            insights = self.workflow_learning.get_workflow_insights()
            if insights['total_sessions'] < 5:
                return "Sir, I'm still learning your workflow patterns. I need more data to make suggestions"
            else:
                return "Sir, your current setup matches your usual workflow"
        
        # Get top prediction
        top_prediction = predictions[0]
        
        if top_prediction.prediction_type == 'missing_window':
            apps = ", ".join(top_prediction.suggested_apps[:2])
            return f"Sir, {top_prediction.description}. You usually have {apps} open at this time"
        
        elif top_prediction.prediction_type == 'likely_relationship':
            return f"Sir, {top_prediction.description}"
        
        elif top_prediction.prediction_type == 'workflow_suggestion':
            return f"Sir, {top_prediction.description}"
        
        return "Sir, I've analyzed your workflow patterns and everything looks optimal"
    
    def get_pending_insights(self) -> List[str]:
        """Get any pending proactive insights as JARVIS messages"""
        messages = []
        for insight in self.pending_insights:
            messages.append(insight.to_jarvis_message())
        
        # Clear pending insights after retrieving
        self.pending_insights = []
        return messages
    
    async def start_monitoring(self):
        """Start monitoring workspace for proactive insights"""
        self.monitoring_active = True
        
        # Start proactive insights monitoring in background
        asyncio.create_task(self._monitor_insights())
        
        # Start autonomous monitoring if enabled
        if self.autonomous_enabled:
            await self.start_autonomous_mode()
        
        async def monitor_callback(changes):
            """Handle workspace changes"""
            if changes['focus_changed'] and changes['current_focus']:
                logger.info(f"Focus changed to: {changes['current_focus'].app_name}")
                
                # Could trigger proactive analysis here
                # For example, if switching to Slack, check for important messages
                
        # Start monitoring
        await self.window_detector.monitor_windows(monitor_callback)
    
    async def _monitor_insights(self):
        """Background task to monitor for proactive insights"""
        try:
            async for insight in self.proactive_insights.start_monitoring():
                if self.monitoring_active:
                    self.pending_insights.append(insight)
                    logger.info(f"New proactive insight: {insight.insight_type}")
                else:
                    break
        except Exception as e:
            logger.error(f"Error in proactive monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop workspace monitoring"""
        self.monitoring_active = False
        
        # Stop autonomous monitoring
        if self.autonomous_task:
            self.autonomous_task.cancel()
            self.autonomous_task = None
    
    # ========== AUTONOMOUS SYSTEM METHODS ==========
    
    async def enable_autonomous_mode(self, user_confirmation: bool = True):
        """Enable JARVIS autonomous mode"""
        logger.info("Enabling JARVIS autonomous mode")
        
        self.autonomous_enabled = True
        
        if user_confirmation:
            # In production, get actual user confirmation
            response = "Sir, I'll now operate autonomously. I'll handle routine tasks, "
            response += "organize your workspace, and manage notifications intelligently. "
            response += "I'll always ask permission for important actions. Shall I proceed?"
            logger.info(response)
        
        # Start autonomous monitoring if monitoring is active
        if self.monitoring_active:
            await self.start_autonomous_mode()
        
        return "Autonomous mode activated. I'll handle routine tasks while you focus on what matters."
    
    async def disable_autonomous_mode(self):
        """Disable JARVIS autonomous mode"""
        logger.info("Disabling JARVIS autonomous mode")
        
        self.autonomous_enabled = False
        
        # Cancel autonomous task
        if self.autonomous_task:
            self.autonomous_task.cancel()
            self.autonomous_task = None
        
        return "Autonomous mode disabled. I'll wait for your explicit commands."
    
    async def start_autonomous_mode(self):
        """Start autonomous monitoring and decision-making"""
        if not self.autonomous_enabled:
            return
        
        logger.info("Starting autonomous monitoring")
        
        # Cancel existing task if any
        if self.autonomous_task:
            self.autonomous_task.cancel()
        
        # Start new autonomous monitoring task
        self.autonomous_task = asyncio.create_task(self._autonomous_monitor_loop())
    
    async def _autonomous_monitor_loop(self):
        """Main autonomous monitoring loop"""
        logger.info("Autonomous monitor loop started")
        
        while self.autonomous_enabled and self.monitoring_active:
            try:
                # Get current workspace state
                windows = self.window_detector.get_all_windows()
                workspace_state = await self.workspace_analyzer.analyze_workspace("")
                
                # Analyze context
                context = await self.context_engine.analyze_context(workspace_state, windows)
                
                # Make autonomous decisions
                actions = await self.autonomous_engine.analyze_and_decide(workspace_state, windows)
                
                # Process each action
                for action in actions:
                    await self._process_autonomous_action(action, context)
                
                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                logger.info("Autonomous monitoring cancelled")
                break
            except Exception as e:
                logger.error(f"Error in autonomous monitoring: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def _process_autonomous_action(self, action: AutonomousAction, context):
        """Process a single autonomous action"""
        try:
            # Check if we should act now based on context
            should_act, timing_reason = self.context_engine.should_act_now(action, context)
            
            if not should_act:
                logger.info(f"Delaying action {action.action_type}: {timing_reason}")
                # Could queue for later or skip
                return
            
            # Check permissions
            permission, confidence, permission_reason = self.permission_manager.check_permission(action)
            
            if permission is True:
                # Auto-approved - execute
                logger.info(f"Auto-executing {action.action_type}: {permission_reason}")
                await self._execute_autonomous_action(action)
                
            elif permission is False:
                # Auto-denied - skip
                logger.info(f"Auto-denied {action.action_type}: {permission_reason}")
                
            else:
                # Need user permission
                if action.requires_permission or confidence < 0.7:
                    await self._request_user_permission(action, confidence, permission_reason)
                else:
                    # High confidence, just notify
                    await self._notify_autonomous_action(action)
                    await self._execute_autonomous_action(action)
            
        except Exception as e:
            logger.error(f"Error processing autonomous action: {e}")
    
    async def _execute_autonomous_action(self, action: AutonomousAction):
        """Execute an autonomous action"""
        try:
            # Execute the action
            result = await self.action_executor.execute_action(action, dry_run=False)
            
            # Learn from result
            success = result.status.value == "success"
            self.autonomous_engine.learn_from_feedback(action, success)
            
            if success:
                logger.info(f"Successfully executed {action.action_type}")
                # Could notify user of successful action
            else:
                logger.error(f"Failed to execute {action.action_type}: {result.error}")
                
        except Exception as e:
            logger.error(f"Error executing autonomous action: {e}")
    
    async def _request_user_permission(self, action: AutonomousAction, confidence: float, reason: str):
        """Request user permission for an action"""
        # In production, this would integrate with the UI/voice system
        message = f"Sir, may I {action.action_type.replace('_', ' ')} for {action.target}? "
        message += f"Confidence: {confidence:.0%}. {reason}"
        
        logger.info(f"Permission request: {message}")
        
        # For now, simulate approval
        # In production, wait for actual user response
        approved = True  # Would get actual user input
        
        # Record decision for learning
        self.permission_manager.record_decision(action, approved)
        
        if approved:
            await self._execute_autonomous_action(action)
    
    async def _notify_autonomous_action(self, action: AutonomousAction):
        """Notify user of autonomous action being taken"""
        message = f"Sir, I'm {action.action_type.replace('_', ' ')} for {action.target}. "
        message += action.reasoning
        
        logger.info(f"Autonomous action notification: {message}")
        
        # In production, this would show a notification or speak to user
        # Could use self.workspace_analyzer.claude_client to generate natural language
    
    async def handle_autonomous_command(self, command: str) -> str:
        """Handle commands related to autonomous operation"""
        command_lower = command.lower()
        
        if "enable autonomous" in command_lower or "activate autonomous" in command_lower:
            return await self.enable_autonomous_mode()
            
        elif "disable autonomous" in command_lower or "stop autonomous" in command_lower:
            return await self.disable_autonomous_mode()
            
        elif "autonomous status" in command_lower:
            if self.autonomous_enabled:
                stats = self.action_executor.get_execution_stats()
                return f"Autonomous mode is active. {stats['total_executions']} actions taken, {stats.get('success_rate', 0):.0%} success rate."
            else:
                return "Autonomous mode is currently disabled."
                
        elif "rollback" in command_lower:
            if await self.action_executor.rollback_last_action():
                return "Sir, I've rolled back the last action."
            else:
                return "No actions available to rollback."
                
        elif "permission" in command_lower and "stats" in command_lower:
            stats = self.permission_manager.get_permission_stats()
            return f"Permission stats: {stats['total_decisions']} decisions recorded, {len(stats['auto_approval_candidates'])} actions ready for automation."
            
        else:
            return "I can enable/disable autonomous mode, show status, rollback actions, or show permission statistics."

# Integration function for JARVIS voice system
async def process_workspace_query(query: str) -> str:
    """Process workspace-aware queries for JARVIS"""
    workspace_intel = JARVISWorkspaceIntelligence()
    return await workspace_intel.handle_workspace_command(query)

async def test_jarvis_workspace():
    """Test JARVIS workspace integration"""
    print("🎯 Testing JARVIS Workspace Intelligence")
    print("=" * 50)
    
    workspace_intel = JARVISWorkspaceIntelligence()
    
    # Test various commands
    test_commands = [
        "Hey JARVIS, what am I working on?",
        "Do I have any messages?",
        "What windows are open?",
        "Are there any errors I should look at?",
        "Describe my current workspace",
        "What's on my screen right now?",
        "Optimize my workspace",
        "Give me productivity tips"
    ]
    
    for command in test_commands:
        print(f"\n🎤 User: {command}")
        response = await workspace_intel.handle_workspace_command(command)
        print(f"🤖 JARVIS: {response}")
        print("-" * 50)
        
        # Small delay between commands
        await asyncio.sleep(1)
    
    # Test proactive insights
    print("\n🔔 Testing Proactive Insights (10 seconds)...")
    await workspace_intel.start_monitoring()
    
    # Wait for some insights
    await asyncio.sleep(10)
    
    insights = workspace_intel.get_pending_insights()
    if insights:
        print(f"\n📢 Proactive Insights Generated:")
        for insight in insights:
            print(f"   • {insight}")
    else:
        print("\n   No proactive insights generated in test period")
    
    workspace_intel.stop_monitoring()
    
    print("\n✅ JARVIS workspace intelligence test complete!")

if __name__ == "__main__":
    asyncio.run(test_jarvis_workspace())