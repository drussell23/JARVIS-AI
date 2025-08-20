#!/usr/bin/env python3
"""
JARVIS Workspace Intelligence Integration
Connects multi-window awareness to JARVIS voice commands
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from .workspace_analyzer import WorkspaceAnalyzer, WorkspaceAnalysis
from .window_detector import WindowDetector
from .proactive_insights import ProactiveInsights, Insight
from .workspace_optimizer import WorkspaceOptimizer, WorkspaceOptimization

logger = logging.getLogger(__name__)


class JARVISWorkspaceIntelligence:
    """Integrates workspace awareness into JARVIS"""
    
    def __init__(self):
        self.workspace_analyzer = WorkspaceAnalyzer()
        self.window_detector = WindowDetector()
        self.proactive_insights = ProactiveInsights()
        self.workspace_optimizer = WorkspaceOptimizer()
        self.last_analysis = None
        self.monitoring_active = False
        self.pending_insights: List[Insight] = []
        
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
            ]
        }
    
    async def handle_workspace_command(self, command: str) -> str:
        """Handle workspace-aware commands"""
        command_lower = command.lower()
        
        # Store the command for use in formatting
        self._last_command = command
        
        # Determine query type
        query_type = self._determine_query_type(command_lower)
        
        # Analyze workspace with appropriate context
        try:
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
        import re
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
        import re
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