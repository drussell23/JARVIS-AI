#!/usr/bin/env python3
"""
Meeting Preparation System for JARVIS Multi-Window Intelligence
Helps users prepare for meetings by analyzing windows and detecting conflicts
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

from .window_detector import WindowDetector, WindowInfo
from .window_relationship_detector import WindowRelationshipDetector
from .workspace_optimizer import WorkspaceOptimizer, WindowLayout

logger = logging.getLogger(__name__)

@dataclass
class MeetingContext:
    """Context about an upcoming meeting"""
    meeting_app: Optional[WindowInfo] = None  # Zoom, Teams, etc.
    calendar_app: Optional[WindowInfo] = None  # Calendar, Fantastical
    notes_apps: List[WindowInfo] = field(default_factory=list)  # Notes, Notion, etc.
    document_windows: List[WindowInfo] = field(default_factory=list)  # Related docs
    presentation_windows: List[WindowInfo] = field(default_factory=list)  # Slides, etc.
    sensitive_windows: List[WindowInfo] = field(default_factory=list)  # To hide
    
    @property
    def is_meeting_ready(self) -> bool:
        """Check if basic meeting requirements are met"""
        return self.meeting_app is not None or self.calendar_app is not None
    
    @property
    def has_materials(self) -> bool:
        """Check if user has any meeting materials open"""
        return bool(self.notes_apps or self.document_windows or self.presentation_windows)

@dataclass
class MeetingAlert:
    """Alert about meeting preparation"""
    alert_type: str  # 'missing_app', 'sensitive_windows', 'no_materials', 'conflict'
    severity: str  # 'high', 'medium', 'low'
    message: str
    affected_windows: List[WindowInfo] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

class MeetingPreparationSystem:
    """Helps users prepare for meetings with intelligent window analysis"""
    
    def __init__(self):
        self.window_detector = WindowDetector()
        self.relationship_detector = WindowRelationshipDetector()
        self.workspace_optimizer = WorkspaceOptimizer()
        
        # Meeting-related applications
        self.meeting_apps = {
            'Zoom', 'Microsoft Teams', 'Google Meet', 'Webex', 
            'Skype', 'GoToMeeting', 'Discord'
        }
        
        self.calendar_apps = {
            'Calendar', 'Fantastical', 'BusyCal', 'Outlook', 
            'Google Calendar'
        }
        
        self.notes_apps = {
            'Notes', 'Notion', 'Obsidian', 'Bear', 'Evernote',
            'OneNote', 'Roam Research', 'Craft'
        }
        
        self.presentation_apps = {
            'Keynote', 'PowerPoint', 'Google Slides', 'Prezi'
        }
        
        # Sensitive content patterns
        self.sensitive_patterns = [
            r'password', r'secret', r'private', r'confidential',
            r'salary', r'compensation', r'performance review',
            r'personal', r'medical', r'financial', r'banking',
            r'tax', r'invoice', r'contract', r'legal'
        ]
        
        # Apps that are typically sensitive
        self.sensitive_apps = {
            '1Password', 'LastPass', 'Bitwarden', 'KeePass',
            'Messages', 'WhatsApp', 'Signal', 'Telegram',
            'Mail', 'Slack'  # Personal communications
        }
    
    def analyze_meeting_preparation(self, windows: List[WindowInfo] = None) -> Tuple[MeetingContext, List[MeetingAlert]]:
        """Analyze workspace for meeting preparation"""
        if windows is None:
            windows = self.window_detector.get_all_windows()
        
        # Build meeting context
        context = self._build_meeting_context(windows)
        
        # Generate alerts
        alerts = self._generate_meeting_alerts(context, windows)
        
        return context, alerts
    
    def _build_meeting_context(self, windows: List[WindowInfo]) -> MeetingContext:
        """Build context about meeting-related windows"""
        context = MeetingContext()
        
        for window in windows:
            # Check for meeting apps
            if any(app in window.app_name for app in self.meeting_apps):
                if not context.meeting_app or window.is_focused:
                    context.meeting_app = window
            
            # Check for calendar apps
            elif any(app in window.app_name for app in self.calendar_apps):
                context.calendar_app = window
            
            # Check for notes apps
            elif any(app in window.app_name for app in self.notes_apps):
                context.notes_apps.append(window)
            
            # Check for presentation apps
            elif any(app in window.app_name for app in self.presentation_apps):
                context.presentation_windows.append(window)
            
            # Check for documents
            elif self._is_document_window(window):
                # Check if it's meeting-related
                if self._is_meeting_related(window):
                    context.document_windows.append(window)
            
            # Check for sensitive windows
            if self._is_sensitive_window(window):
                context.sensitive_windows.append(window)
        
        return context
    
    def _generate_meeting_alerts(self, context: MeetingContext, 
                               all_windows: List[WindowInfo]) -> List[MeetingAlert]:
        """Generate alerts based on meeting context"""
        alerts = []
        
        # Check if meeting app is missing
        if not context.meeting_app:
            # But check if there are meeting-related windows
            if context.calendar_app or context.notes_apps or context.presentation_windows:
                alerts.append(MeetingAlert(
                    alert_type='missing_app',
                    severity='high',
                    message="Meeting materials detected but no meeting app is open",
                    suggestions=[
                        "Open Zoom, Teams, or your meeting application",
                        "Check your calendar for the meeting link"
                    ]
                ))
        
        # Check for sensitive windows that should be hidden
        if context.sensitive_windows and context.meeting_app:
            alerts.append(MeetingAlert(
                alert_type='sensitive_windows',
                severity='high',
                message=f"Detected {len(context.sensitive_windows)} sensitive windows that should be hidden",
                affected_windows=context.sensitive_windows,
                suggestions=[
                    "Hide or minimize sensitive windows before screen sharing",
                    "Use 'Hide sensitive windows' command"
                ]
            ))
        
        # Check if user has no materials ready
        if context.meeting_app and not context.has_materials:
            alerts.append(MeetingAlert(
                alert_type='no_materials',
                severity='medium',
                message="No meeting materials detected",
                suggestions=[
                    "Open meeting notes or agenda",
                    "Prepare any documents you need to share"
                ]
            ))
        
        # Check for conflicting applications
        conflicts = self._check_for_conflicts(context, all_windows)
        if conflicts:
            alerts.append(MeetingAlert(
                alert_type='conflict',
                severity='medium',
                message="Potentially distracting applications are open",
                affected_windows=conflicts,
                suggestions=[
                    "Close entertainment or social media apps",
                    "Minimize distracting windows"
                ]
            ))
        
        return alerts
    
    def get_meeting_layout(self, context: MeetingContext) -> Optional[WindowLayout]:
        """Get optimal layout for meeting"""
        if not context.meeting_app:
            return None
        
        # Determine layout based on what's available
        if context.presentation_windows:
            return self._calculate_presentation_layout(context)
        elif context.notes_apps and context.document_windows:
            return self._calculate_collaboration_layout(context)
        else:
            return self._calculate_basic_meeting_layout(context)
    
    def _calculate_presentation_layout(self, context: MeetingContext) -> WindowLayout:
        """Layout for presenting: Meeting app + Presentation + Notes"""
        positions = {}
        
        # Get screen dimensions from workspace optimizer
        screen_width = self.workspace_optimizer.screen_width
        screen_height = self.workspace_optimizer.screen_height
        
        # Meeting app: Top left corner (smaller)
        if context.meeting_app:
            positions[context.meeting_app.window_id] = {
                'x': 0,
                'y': 50,
                'width': screen_width // 3,
                'height': screen_height // 3
            }
        
        # Presentation: Center/right (large)
        if context.presentation_windows:
            pres = context.presentation_windows[0]
            positions[pres.window_id] = {
                'x': screen_width // 3,
                'y': 50,
                'width': (screen_width * 2) // 3,
                'height': (screen_height * 2) // 3
            }
        
        # Notes: Bottom left
        if context.notes_apps:
            notes = context.notes_apps[0]
            positions[notes.window_id] = {
                'x': 0,
                'y': 50 + screen_height // 3,
                'width': screen_width // 3,
                'height': (screen_height * 2) // 3 - 50
            }
        
        return WindowLayout(
            layout_type='presentation_mode',
            positions=positions,
            description="Presentation mode: Meeting + Slides + Notes",
            benefit="Optimized for screen sharing with notes visible",
            confidence=0.95
        )
    
    def _calculate_collaboration_layout(self, context: MeetingContext) -> WindowLayout:
        """Layout for collaboration: Meeting + Notes + Documents"""
        positions = {}
        
        screen_width = self.workspace_optimizer.screen_width
        screen_height = self.workspace_optimizer.screen_height
        
        # Meeting app: Left half
        if context.meeting_app:
            positions[context.meeting_app.window_id] = {
                'x': 0,
                'y': 50,
                'width': screen_width // 2,
                'height': screen_height - 50
            }
        
        # Notes: Top right
        if context.notes_apps:
            notes = context.notes_apps[0]
            positions[notes.window_id] = {
                'x': screen_width // 2,
                'y': 50,
                'width': screen_width // 2,
                'height': (screen_height - 50) // 2
            }
        
        # Document: Bottom right
        if context.document_windows:
            doc = context.document_windows[0]
            positions[doc.window_id] = {
                'x': screen_width // 2,
                'y': 50 + (screen_height - 50) // 2,
                'width': screen_width // 2,
                'height': (screen_height - 50) // 2
            }
        
        return WindowLayout(
            layout_type='collaboration_mode',
            positions=positions,
            description="Collaboration mode: Meeting + Notes + Docs",
            benefit="Easy reference to materials during discussion",
            confidence=0.9
        )
    
    def _calculate_basic_meeting_layout(self, context: MeetingContext) -> WindowLayout:
        """Basic meeting layout: Meeting app centered"""
        positions = {}
        
        screen_width = self.workspace_optimizer.screen_width
        screen_height = self.workspace_optimizer.screen_height
        
        # Meeting app: Centered and large
        if context.meeting_app:
            margin = 100
            positions[context.meeting_app.window_id] = {
                'x': margin,
                'y': margin,
                'width': screen_width - (2 * margin),
                'height': screen_height - (2 * margin)
            }
        
        return WindowLayout(
            layout_type='meeting_focus',
            positions=positions,
            description="Meeting focus: Centered meeting window",
            benefit="Distraction-free meeting view",
            confidence=0.85
        )
    
    def hide_sensitive_windows(self, windows: List[WindowInfo] = None) -> List[WindowInfo]:
        """Hide or minimize sensitive windows"""
        if windows is None:
            windows = self.window_detector.get_all_windows()
        
        hidden_windows = []
        
        for window in windows:
            if self._is_sensitive_window(window) and window.is_visible:
                # In a real implementation, this would use macOS APIs to hide
                # For now, we just track which windows should be hidden
                hidden_windows.append(window)
                logger.info(f"Would hide sensitive window: {window.app_name} - {window.window_title}")
        
        return hidden_windows
    
    def _is_sensitive_window(self, window: WindowInfo) -> bool:
        """Check if window contains sensitive content"""
        # Check if it's a sensitive app
        if any(app in window.app_name for app in self.sensitive_apps):
            return True
        
        # Check window title for sensitive patterns
        if window.window_title:
            title_lower = window.window_title.lower()
            for pattern in self.sensitive_patterns:
                if re.search(pattern, title_lower):
                    return True
        
        return False
    
    def _is_document_window(self, window: WindowInfo) -> bool:
        """Check if window is a document"""
        doc_apps = {'Preview', 'Adobe', 'Word', 'Pages', 'Google Docs', 'TextEdit'}
        browser_apps = {'Chrome', 'Safari', 'Firefox'}
        
        # Check document apps
        if any(app in window.app_name for app in doc_apps):
            return True
        
        # Check browsers with document-like content
        if any(app in window.app_name for app in browser_apps):
            if window.window_title and any(keyword in window.window_title.lower() 
                                          for keyword in ['docs', 'slides', 'sheet', 'pdf']):
                return True
        
        return False
    
    def _is_meeting_related(self, window: WindowInfo) -> bool:
        """Check if window is related to meetings"""
        if not window.window_title:
            return False
        
        title_lower = window.window_title.lower()
        meeting_keywords = [
            'meeting', 'agenda', 'minutes', 'presentation',
            'slides', 'schedule', 'calendar', 'invite'
        ]
        
        return any(keyword in title_lower for keyword in meeting_keywords)
    
    def _check_for_conflicts(self, context: MeetingContext, 
                           all_windows: List[WindowInfo]) -> List[WindowInfo]:
        """Check for windows that might conflict with meetings"""
        conflicts = []
        
        distracting_apps = {
            'YouTube', 'Netflix', 'Spotify', 'Music', 'TV',
            'Twitter', 'Facebook', 'Instagram', 'TikTok',
            'Reddit', 'Discord'  # Discord only if not the meeting app
        }
        
        for window in all_windows:
            # Skip if it's the meeting app itself
            if context.meeting_app and window.window_id == context.meeting_app.window_id:
                continue
            
            # Check for distracting apps
            if any(app in window.app_name for app in distracting_apps):
                # Special case: Discord might be the meeting platform
                if 'Discord' in window.app_name and context.meeting_app and 'Discord' in context.meeting_app.app_name:
                    continue
                conflicts.append(window)
        
        return conflicts

async def test_meeting_preparation():
    """Test meeting preparation system"""
    print("📅 Testing Meeting Preparation System")
    print("=" * 50)
    
    meeting_system = MeetingPreparationSystem()
    
    # Analyze current workspace
    print("\n🔍 Analyzing workspace for meeting preparation...")
    context, alerts = meeting_system.analyze_meeting_preparation()
    
    print(f"\n📊 Meeting Context:")
    print(f"   Meeting App: {context.meeting_app.app_name if context.meeting_app else 'None'}")
    print(f"   Calendar App: {context.calendar_app.app_name if context.calendar_app else 'None'}")
    print(f"   Notes Apps: {len(context.notes_apps)}")
    print(f"   Documents: {len(context.document_windows)}")
    print(f"   Presentations: {len(context.presentation_windows)}")
    print(f"   Sensitive Windows: {len(context.sensitive_windows)}")
    
    if alerts:
        print(f"\n⚠️  Meeting Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"\n   Alert: {alert.message}")
            print(f"   Type: {alert.alert_type} | Severity: {alert.severity}")
            if alert.suggestions:
                print(f"   Suggestions:")
                for suggestion in alert.suggestions:
                    print(f"     • {suggestion}")
    
    # Test layout generation
    if context.meeting_app:
        print(f"\n🏗️  Generating meeting layout...")
        layout = meeting_system.get_meeting_layout(context)
        if layout:
            print(f"   Layout: {layout.layout_type}")
            print(f"   Description: {layout.description}")
            print(f"   Benefit: {layout.benefit}")
    
    # Test sensitive window detection
    if context.sensitive_windows:
        print(f"\n🔒 Sensitive windows to hide:")
        for window in context.sensitive_windows[:3]:
            print(f"   • {window.app_name} - {window.window_title or 'Untitled'}")
    
    print("\n✅ Meeting preparation test complete!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_meeting_preparation())