#!/usr/bin/env python3
"""
Proactive Insights for JARVIS Multi-Window Intelligence
Surfaces important information across windows without being asked
"""

import asyncio
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import re

from .window_detector import WindowDetector, WindowInfo
from .window_relationship_detector import WindowRelationshipDetector
from .smart_query_router import SmartQueryRouter
from .multi_window_capture import MultiWindowCapture

logger = logging.getLogger(__name__)


@dataclass
class Insight:
    """Represents a proactive insight"""
    insight_type: str  # 'new_message', 'error_detected', 'doc_suggestion', etc.
    priority: str  # 'high', 'medium', 'low'
    title: str
    description: str
    source_windows: List[WindowInfo]
    timestamp: datetime
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def to_jarvis_message(self) -> str:
        """Convert insight to JARVIS voice message"""
        if self.insight_type == 'new_message':
            app_name = self.source_windows[0].app_name if self.source_windows else "an app"
            return f"Sir, you have new messages in {app_name}"
        elif self.insight_type == 'error_detected':
            return f"Sir, I've detected errors in your {self.title}"
        elif self.insight_type == 'doc_suggestion':
            return f"Sir, I found relevant documentation for {self.title}"
        elif self.insight_type == 'workspace_alert':
            return f"Sir, {self.description}"
        else:
            return f"Sir, {self.description}"


@dataclass
class WindowState:
    """Tracks state of a window for change detection"""
    window_info: WindowInfo
    last_title: str
    last_checked: datetime
    title_history: List[Tuple[datetime, str]] = field(default_factory=list)
    has_new_content: bool = False


class ProactiveInsights:
    """Generates proactive insights from workspace activity"""
    
    def __init__(self):
        self.window_detector = WindowDetector()
        self.relationship_detector = WindowRelationshipDetector()
        self.query_router = SmartQueryRouter()
        self.capture_system = MultiWindowCapture()
        
        # Window state tracking
        self.window_states: Dict[int, WindowState] = {}
        self.last_full_scan = datetime.now()
        
        # Insight generation settings
        self.check_interval = 5.0  # seconds between checks
        self.insight_cooldown = 60.0  # seconds before repeating similar insights
        self.recent_insights: List[Insight] = []
        
        # Pattern matchers for different insight types
        self.message_indicators = [
            r'\(\d+\)',  # (1) unread count
            r'•',  # bullet indicating unread
            r'unread',
            r'new message',
            r'mentioned you',
            r'@\w+',  # mentions
        ]
        
        self.error_indicators = [
            r'error',
            r'exception',
            r'failed',
            r'warning',
            r'traceback',
            r'undefined',
            r'null pointer',
            r'segmentation fault',
            r'compilation failed',
        ]
        
        self.documentation_triggers = [
            r'how to',
            r'what is',
            r'undefined',
            r'cannot find',
            r'no such',
            r'unknown',
        ]
        
        # Track user's current context
        self.current_focus_context = None
        self.context_start_time = None
    
    async def start_monitoring(self) -> None:
        """Start the proactive monitoring loop"""
        logger.info("Starting proactive insights monitoring")
        
        while True:
            try:
                # Generate insights
                insights = await self.scan_for_insights()
                
                # Process new insights
                for insight in insights:
                    if self._should_surface_insight(insight):
                        self.recent_insights.append(insight)
                        logger.info(f"New insight: {insight.insight_type} - {insight.title}")
                        # In real integration, this would trigger JARVIS to speak
                        yield insight
                
                # Clean old insights
                self._clean_old_insights()
                
            except Exception as e:
                logger.error(f"Error in proactive monitoring: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def scan_for_insights(self) -> List[Insight]:
        """Scan workspace for new insights"""
        insights = []
        
        # Get current windows
        current_windows = self.window_detector.get_all_windows()
        
        # Update window states and detect changes
        window_changes = self._update_window_states(current_windows)
        
        # Detect current user context
        self._update_user_context(current_windows)
        
        # Check for new messages
        message_insights = self._check_for_messages(current_windows, window_changes)
        insights.extend(message_insights)
        
        # Check for errors
        error_insights = self._check_for_errors(current_windows, window_changes)
        insights.extend(error_insights)
        
        # Check for documentation opportunities
        doc_insights = await self._check_documentation_opportunities(current_windows)
        insights.extend(doc_insights)
        
        # Check for workspace issues
        workspace_insights = self._check_workspace_health(current_windows)
        insights.extend(workspace_insights)
        
        return insights
    
    def _update_window_states(self, current_windows: List[WindowInfo]) -> Dict[int, str]:
        """Update window states and return changed windows"""
        changes = {}
        current_time = datetime.now()
        
        # Update existing windows
        for window in current_windows:
            if window.window_id in self.window_states:
                state = self.window_states[window.window_id]
                
                # Check for title change
                if window.window_title != state.last_title:
                    changes[window.window_id] = window.window_title
                    state.title_history.append((current_time, window.window_title))
                    state.last_title = window.window_title
                    state.has_new_content = True
                
                state.last_checked = current_time
                state.window_info = window
            else:
                # New window
                self.window_states[window.window_id] = WindowState(
                    window_info=window,
                    last_title=window.window_title or "",
                    last_checked=current_time,
                    title_history=[(current_time, window.window_title or "")]
                )
        
        # Remove closed windows
        current_ids = {w.window_id for w in current_windows}
        closed_ids = set(self.window_states.keys()) - current_ids
        for window_id in closed_ids:
            del self.window_states[window_id]
        
        return changes
    
    def _update_user_context(self, windows: List[WindowInfo]) -> None:
        """Update understanding of user's current context"""
        focused_window = next((w for w in windows if w.is_focused), None)
        
        if focused_window:
            # Check if context has changed
            new_context = self._determine_context(focused_window)
            
            if new_context != self.current_focus_context:
                self.current_focus_context = new_context
                self.context_start_time = datetime.now()
                logger.debug(f"User context changed to: {new_context}")
    
    def _determine_context(self, window: WindowInfo) -> str:
        """Determine user's current context from focused window"""
        if any(ide in window.app_name for ide in ['Visual Studio Code', 'Cursor', 'Xcode']):
            return "coding"
        elif any(comm in window.app_name for comm in ['Discord', 'Slack', 'Messages']):
            return "communicating"
        elif any(browser in window.app_name for browser in ['Chrome', 'Safari', 'Firefox']):
            if window.window_title and 'docs' in window.window_title.lower():
                return "researching"
            else:
                return "browsing"
        elif 'Terminal' in window.app_name:
            return "terminal_work"
        else:
            return "other"
    
    def _check_for_messages(self, windows: List[WindowInfo], 
                          changes: Dict[int, str]) -> List[Insight]:
        """Check for new messages across communication apps"""
        insights = []
        
        # Only check if user is NOT in communication context
        if self.current_focus_context == "communicating":
            return insights
        
        for window in windows:
            # Check communication apps
            if self._is_communication_window(window):
                # Check for unread indicators in title
                if window.window_title:
                    for pattern in self.message_indicators:
                        if re.search(pattern, window.window_title, re.IGNORECASE):
                            # Check if this is new (window changed)
                            if window.window_id in changes:
                                insights.append(Insight(
                                    insight_type="new_message",
                                    priority="medium" if self.current_focus_context == "coding" else "low",
                                    title=f"New messages in {window.app_name}",
                                    description=f"Unread messages detected in {window.app_name}",
                                    source_windows=[window],
                                    timestamp=datetime.now(),
                                    metadata={"app": window.app_name, "pattern": pattern}
                                ))
                                break
        
        return insights
    
    def _check_for_errors(self, windows: List[WindowInfo], 
                         changes: Dict[int, str]) -> List[Insight]:
        """Check for errors in terminals and development tools"""
        insights = []
        
        # Focus on terminal and IDE windows
        for window in windows:
            if self._is_development_window(window) or self._is_terminal_window(window):
                # Only check recently changed windows
                if window.window_id in changes and window.window_title:
                    title_lower = window.window_title.lower()
                    
                    for pattern in self.error_indicators:
                        if re.search(pattern, title_lower):
                            # Higher priority if user is coding
                            priority = "high" if self.current_focus_context == "coding" else "medium"
                            
                            insights.append(Insight(
                                insight_type="error_detected",
                                priority=priority,
                                title=f"{window.app_name}",
                                description=f"Error detected in {window.app_name}: {pattern} found",
                                source_windows=[window],
                                timestamp=datetime.now(),
                                metadata={"error_type": pattern, "window_title": window.window_title}
                            ))
                            break
        
        return insights
    
    async def _check_documentation_opportunities(self, windows: List[WindowInfo]) -> List[Insight]:
        """Suggest relevant documentation based on current work"""
        insights = []
        
        # Only suggest docs if user is coding or in terminal
        if self.current_focus_context not in ["coding", "terminal_work"]:
            return insights
        
        # Get focused window
        focused_window = next((w for w in windows if w.is_focused), None)
        if not focused_window or not focused_window.window_title:
            return insights
        
        # Check for documentation triggers in title
        title_lower = focused_window.window_title.lower()
        
        for trigger in self.documentation_triggers:
            if re.search(trigger, title_lower):
                # Look for open documentation that might help
                doc_windows = [w for w in windows if self._is_documentation_window(w)]
                
                if doc_windows:
                    # Find related documentation
                    related_docs = self._find_related_documentation(focused_window, doc_windows)
                    
                    if related_docs:
                        insights.append(Insight(
                            insight_type="doc_suggestion",
                            priority="low",
                            title=focused_window.window_title.split(' - ')[0],
                            description=f"Found {len(related_docs)} relevant documentation windows",
                            source_windows=related_docs,
                            timestamp=datetime.now(),
                            metadata={"trigger": trigger, "doc_count": len(related_docs)}
                        ))
                
                break
        
        return insights
    
    def _check_workspace_health(self, windows: List[WindowInfo]) -> List[Insight]:
        """Check for workspace issues and optimization opportunities"""
        insights = []
        
        # Check for too many windows
        if len(windows) > 30:
            insights.append(Insight(
                insight_type="workspace_alert",
                priority="low",
                title="Workspace Overload",
                description=f"You have {len(windows)} windows open. Consider closing unused windows",
                source_windows=[],
                timestamp=datetime.now(),
                metadata={"window_count": len(windows)}
            ))
        
        # Check for duplicate apps
        app_counts = defaultdict(int)
        for window in windows:
            app_counts[window.app_name] += 1
        
        for app, count in app_counts.items():
            if count > 5:  # Many windows of same app
                insights.append(Insight(
                    insight_type="workspace_alert",
                    priority="low",
                    title=f"Multiple {app} Windows",
                    description=f"You have {count} {app} windows open",
                    source_windows=[w for w in windows if w.app_name == app][:3],
                    timestamp=datetime.now(),
                    metadata={"app": app, "count": count}
                ))
        
        return insights
    
    def _should_surface_insight(self, insight: Insight) -> bool:
        """Determine if an insight should be surfaced to the user"""
        # Check priority vs context
        if insight.priority == "low" and self.current_focus_context == "coding":
            return False  # Don't interrupt coding with low priority
        
        # Check for similar recent insights
        cutoff_time = datetime.now() - timedelta(seconds=self.insight_cooldown)
        
        for recent in self.recent_insights:
            if recent.timestamp > cutoff_time:
                # Check if similar
                if (recent.insight_type == insight.insight_type and
                    recent.title == insight.title):
                    return False  # Too similar to recent insight
        
        return True
    
    def _clean_old_insights(self) -> None:
        """Remove old insights from history"""
        cutoff_time = datetime.now() - timedelta(minutes=10)
        self.recent_insights = [i for i in self.recent_insights 
                               if i.timestamp > cutoff_time]
    
    def _is_communication_window(self, window: WindowInfo) -> bool:
        """Check if window is a communication app"""
        comm_apps = ['Discord', 'Slack', 'Messages', 'Mail', 'WhatsApp', 'Telegram']
        return any(app in window.app_name for app in comm_apps)
    
    def _is_development_window(self, window: WindowInfo) -> bool:
        """Check if window is a development tool"""
        dev_apps = ['Visual Studio Code', 'Cursor', 'Xcode', 'IntelliJ', 'PyCharm']
        return any(app in window.app_name for app in dev_apps)
    
    def _is_terminal_window(self, window: WindowInfo) -> bool:
        """Check if window is a terminal"""
        return 'Terminal' in window.app_name or 'iTerm' in window.app_name
    
    def _is_documentation_window(self, window: WindowInfo) -> bool:
        """Check if window contains documentation"""
        if window.window_title:
            title_lower = window.window_title.lower()
            doc_keywords = ['docs', 'documentation', 'api', 'reference', 'guide', 
                           'stackoverflow', 'github']
            return any(keyword in title_lower for keyword in doc_keywords)
        return False
    
    def _find_related_documentation(self, focus_window: WindowInfo, 
                                   doc_windows: List[WindowInfo]) -> List[WindowInfo]:
        """Find documentation windows related to current focus"""
        related = []
        
        if not focus_window.window_title:
            return related
        
        # Extract keywords from focused window
        focus_words = set(re.findall(r'\w{3,}', focus_window.window_title.lower()))
        
        # Find docs with matching keywords
        for doc in doc_windows:
            if doc.window_title:
                doc_words = set(re.findall(r'\w{3,}', doc.window_title.lower()))
                
                # Calculate overlap
                common_words = focus_words & doc_words
                if len(common_words) >= 2:  # At least 2 common words
                    related.append(doc)
        
        return related[:3]  # Return top 3


async def test_proactive_insights():
    """Test proactive insights system"""
    print("🧠 Testing Proactive Insights")
    print("=" * 50)
    
    insights_engine = ProactiveInsights()
    
    # Run for 30 seconds
    print("\nMonitoring workspace for insights...")
    print("(Simulating 30 seconds of monitoring)")
    
    start_time = datetime.now()
    insight_count = 0
    
    async for insight in insights_engine.start_monitoring():
        insight_count += 1
        print(f"\n🔔 New Insight #{insight_count}")
        print(f"   Type: {insight.insight_type}")
        print(f"   Priority: {insight.priority}")
        print(f"   Message: {insight.to_jarvis_message()}")
        print(f"   Details: {insight.description}")
        
        # Stop after 30 seconds
        if (datetime.now() - start_time).seconds > 30:
            break
    
    print(f"\n✅ Generated {insight_count} insights in 30 seconds")


if __name__ == "__main__":
    asyncio.run(test_proactive_insights())