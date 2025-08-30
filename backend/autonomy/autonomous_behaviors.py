#!/usr/bin/env python3
"""
Autonomous Behaviors for JARVIS
Implements specific behavior patterns for different scenarios
"""

import re
import os
import logging
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from vision.window_detector import WindowInfo
try:
    from vision.claude_vision_analyzer import ClaudeVisionAnalyzer
except ImportError:
    ClaudeVisionAnalyzer = None
    
from .autonomous_decision_engine import AutonomousAction, ActionPriority, ActionCategory

logger = logging.getLogger(__name__)

class MessageHandler:
    """Autonomous message handling behaviors"""
    
    def __init__(self):
        # Initialize vision analyzer with API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key and ClaudeVisionAnalyzer is not None:
            try:
                self.vision_analyzer = ClaudeVisionAnalyzer(api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize ClaudeVisionAnalyzer: {e}")
                self.vision_analyzer = None
        else:
            logger.warning("ANTHROPIC_API_KEY not set or ClaudeVisionAnalyzer not available - vision analysis will be limited")
            self.vision_analyzer = None
        
        # Patterns for different message types
        self.automated_patterns = [
            r"automated|automatic|bot|system notification|digest|summary",
            r"noreply|no-reply|do not reply|automated message",
            r"scheduled|recurring|daily|weekly|monthly",
            r"newsletter|subscription|marketing"
        ]
        
        self.meeting_patterns = [
            r"meeting in \d+ minutes?|starts? (at|in)|scheduled for",
            r"standup|stand-up|sync|call|conference",
            r"zoom|teams|google meet|hangout|webex|meeting link",
            r"calendar reminder|event reminder"
        ]
        
        self.urgent_patterns = [
            r"urgent|asap|emergency|critical|immediate",
            r"important|priority|deadline|due (today|soon)",
            r"action required|response needed|waiting for",
            r"blocker|blocked|issue|problem"
        ]
        
        self.security_patterns = [
            r"security alert|suspicious|unauthorized|breach",
            r"password|credential|authentication|2fa|two.?factor",
            r"login attempt|access denied|verification",
            r"fraud|scam|phishing|malware"
        ]
    
    async def handle_routine_message(self, message_window: WindowInfo) -> Optional[AutonomousAction]:
        """Determine action for routine messages"""
        try:
            # Extract message content using vision
            content = await self._extract_message_content(message_window)
            
            # Classify message
            if self._is_automated_notification(content):
                return AutonomousAction(
                    action_type="dismiss_notification",
                    target=message_window.app_name,
                    params={
                        "window_id": message_window.window_id,
                        "reason": "Automated notification"
                    },
                    priority=ActionPriority.LOW,
                    confidence=0.95,
                    category=ActionCategory.NOTIFICATION,
                    reasoning="Routine automated message that doesn't require attention"
                )
            
            if self._is_meeting_reminder(content):
                meeting_info = self._extract_meeting_info(content)
                return AutonomousAction(
                    action_type="prepare_for_meeting",
                    target="workspace",
                    params={
                        "meeting_info": meeting_info,
                        "window_id": message_window.window_id
                    },
                    priority=ActionPriority.HIGH,
                    confidence=0.9,
                    category=ActionCategory.CALENDAR,
                    reasoning=f"Meeting starting soon: {meeting_info.get('title', 'Unknown')}"
                )
            
            if self._is_urgent_message(content):
                return AutonomousAction(
                    action_type="highlight_urgent_message",
                    target=message_window.app_name,
                    params={
                        "window_id": message_window.window_id,
                        "urgency_level": self._get_urgency_level(content)
                    },
                    priority=ActionPriority.HIGH,
                    confidence=0.85,
                    category=ActionCategory.COMMUNICATION,
                    reasoning="Urgent message requiring attention"
                )
            
            if self._is_security_alert(content):
                return AutonomousAction(
                    action_type="security_alert",
                    target="security",
                    params={
                        "window_id": message_window.window_id,
                        "alert_type": self._get_security_type(content),
                        "app": message_window.app_name
                    },
                    priority=ActionPriority.CRITICAL,
                    confidence=0.95,
                    category=ActionCategory.SECURITY,
                    reasoning="Security-related message requiring immediate attention"
                )
            
            # Default action for unclassified messages
            return AutonomousAction(
                action_type="queue_for_review",
                target=message_window.app_name,
                params={
                    "window_id": message_window.window_id,
                    "message_preview": content[:100]
                },
                priority=ActionPriority.MEDIUM,
                confidence=0.7,
                category=ActionCategory.COMMUNICATION,
                reasoning="Message queued for later review"
            )
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            return None
    
    async def _extract_message_content(self, window: WindowInfo) -> str:
        """Extract message content from window using vision"""
        try:
            # Use vision analyzer to extract text if available
            if self.vision_analyzer:
                # For now, use window title as we don't have direct screenshot access
                # TODO: Integrate with screen capture system to get window screenshot
                # analysis = await self.vision_analyzer.analyze_screenshot(screenshot, "Extract text content")
                # return analysis.get('text_content', window.window_title)
                return window.window_title
            else:
                # Fallback to window title when vision analyzer not available
                return window.window_title
        except Exception as e:
            logger.debug(f"Vision analysis failed: {e}")
            # Fallback to window title
            return window.window_title
    
    def _is_automated_notification(self, content: str) -> bool:
        """Check if message is an automated notification"""
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in self.automated_patterns)
    
    def _is_meeting_reminder(self, content: str) -> bool:
        """Check if message is a meeting reminder"""
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in self.meeting_patterns)
    
    def _is_urgent_message(self, content: str) -> bool:
        """Check if message is urgent"""
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in self.urgent_patterns)
    
    def _is_security_alert(self, content: str) -> bool:
        """Check if message is security-related"""
        content_lower = content.lower()
        return any(re.search(pattern, content_lower) for pattern in self.security_patterns)
    
    def _extract_meeting_info(self, content: str) -> Dict[str, Any]:
        """Extract meeting information from content"""
        info = {"title": "Meeting", "time": None, "platform": None}
        
        # Extract time
        time_match = re.search(r"in (\d+) minutes?|at (\d{1,2}:\d{2})", content, re.IGNORECASE)
        if time_match:
            if time_match.group(1):
                minutes = int(time_match.group(1))
                info["time"] = f"in {minutes} minutes"  # Keep as string for compatibility
            else:
                info["time"] = time_match.group(2)
        
        # Extract platform
        for platform in ["zoom", "teams", "meet", "hangout", "webex"]:
            if platform in content.lower():
                info["platform"] = platform.capitalize()
                break
        
        # Extract title (first line or subject)
        lines = content.split('\n')
        if lines:
            info["title"] = lines[0].strip()[:50]
        
        return info
    
    def _get_urgency_level(self, content: str) -> str:
        """Determine urgency level from content"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["emergency", "critical", "asap", "immediate"]):
            return "critical"
        elif any(word in content_lower for word in ["urgent", "important", "deadline"]):
            return "critical"  # "urgent" should also be critical priority
        else:
            return "medium"
    
    def _get_security_type(self, content: str) -> str:
        """Determine security alert type"""
        content_lower = content.lower()
        
        if "password" in content_lower or "credential" in content_lower:
            return "authentication"
        elif "login" in content_lower or "access" in content_lower:
            return "access_attempt"
        elif "fraud" in content_lower or "scam" in content_lower:
            return "fraud_alert"
        else:
            return "general_security"

class MeetingHandler:
    """Autonomous meeting preparation behaviors"""
    
    def __init__(self):
        self.preparation_actions = []
        self.sensitive_apps = [
            "1Password", "Bitwarden", "LastPass", "KeePass",
            "Banking", "PayPal", "Venmo", "CashApp",
            "Personal", "Private", "Confidential"
        ]
    
    async def prepare_for_meeting(self, meeting_info: Dict[str, Any], 
                                  current_windows: List[WindowInfo]) -> List[AutonomousAction]:
        """Generate actions to prepare workspace for meeting"""
        actions = []
        
        # Hide sensitive windows
        for window in current_windows:
            if self._is_sensitive_window(window):
                actions.append(AutonomousAction(
                    action_type="minimize_window",
                    target=window.app_name,
                    params={"window_id": window.window_id},
                    priority=ActionPriority.HIGH,
                    confidence=0.95,
                    category=ActionCategory.SECURITY,
                    reasoning=f"Hiding sensitive app before meeting: {window.app_name}"
                ))
        
        # Open meeting platform
        platform = meeting_info.get("platform")
        if platform:
            actions.append(AutonomousAction(
                action_type="open_application",
                target=platform,
                params={"meeting_link": meeting_info.get("link")},
                priority=ActionPriority.HIGH,
                confidence=0.9,
                category=ActionCategory.CALENDAR,
                reasoning=f"Opening {platform} for upcoming meeting"
            ))
        
        # Mute distracting apps
        distracting_apps = self._find_distracting_apps(current_windows)
        for app in distracting_apps:
            actions.append(AutonomousAction(
                action_type="mute_notifications",
                target=app,
                params={"duration_minutes": 60},
                priority=ActionPriority.MEDIUM,
                confidence=0.85,
                category=ActionCategory.NOTIFICATION,
                reasoning=f"Muting {app} notifications during meeting"
            ))
        
        # Clean up desktop if needed
        if self._desktop_needs_cleanup(current_windows):
            actions.append(AutonomousAction(
                action_type="organize_desktop",
                target="desktop",
                params={"style": "meeting_ready"},
                priority=ActionPriority.MEDIUM,
                confidence=0.8,
                category=ActionCategory.MAINTENANCE,
                reasoning="Organizing desktop for professional appearance"
            ))
        
        return actions
    
    def _is_sensitive_window(self, window: WindowInfo) -> bool:
        """Check if window contains sensitive content"""
        window_text = f"{window.app_name} {window.window_title}".lower()
        
        # Check against sensitive apps
        for sensitive in self.sensitive_apps:
            if sensitive.lower() in window_text:
                return True
        
        # Check for sensitive patterns in title
        sensitive_patterns = [
            r"password|credential|secret",
            r"bank|finance|money|payment",
            r"personal|private|confidential",
            r"medical|health|prescription"
        ]
        
        return any(re.search(pattern, window_text) for pattern in sensitive_patterns)
    
    def _find_distracting_apps(self, windows: List[WindowInfo]) -> List[str]:
        """Find apps that might be distracting during meetings"""
        distracting = []
        distracting_patterns = [
            "Discord", "Slack", "Messages", "WhatsApp", "Telegram",
            "Twitter", "Facebook", "Instagram", "TikTok",
            "YouTube", "Netflix", "Spotify", "Music"
        ]
        
        seen_apps = set()
        for window in windows:
            for pattern in distracting_patterns:
                if pattern.lower() in window.app_name.lower() and window.app_name not in seen_apps:
                    distracting.append(window.app_name)
                    seen_apps.add(window.app_name)
                    break
        
        return distracting
    
    def _desktop_needs_cleanup(self, windows: List[WindowInfo]) -> bool:
        """Check if desktop needs organization"""
        # Simple heuristic: too many visible windows
        visible_windows = [w for w in windows if w.is_visible]
        return len(visible_windows) > 10

class WorkspaceOrganizer:
    """Autonomous workspace organization behaviors"""
    
    def __init__(self):
        self.project_patterns = {}
        self.window_groups = defaultdict(list)
    
    async def analyze_and_organize(self, windows: List[WindowInfo], 
                                   user_state: str) -> List[AutonomousAction]:
        """Analyze workspace and suggest organization actions"""
        actions = []
        
        # Group windows by project/context
        window_groups = self._group_windows_by_context(windows)
        
        # Check for inefficient layouts
        if self._has_overlapping_windows(windows):
            actions.append(AutonomousAction(
                action_type="arrange_windows",
                target="workspace",
                params={
                    "layout": "tiled",
                    "groups": window_groups
                },
                priority=ActionPriority.LOW,
                confidence=0.8,
                category=ActionCategory.MAINTENANCE,
                reasoning="Detected overlapping windows - suggesting tiled layout"
            ))
        
        # Check for too many windows
        if len([w for w in windows if w.is_visible]) > 15:
            actions.extend(self._suggest_window_reduction(windows))
        
        # Check for duplicate windows
        duplicates = self._find_duplicate_windows(windows)
        for dup in duplicates:
            actions.append(AutonomousAction(
                action_type="close_duplicate",
                target=dup.app_name,
                params={"window_id": dup.window_id},
                priority=ActionPriority.LOW,
                confidence=0.9,
                category=ActionCategory.MAINTENANCE,
                reasoning=f"Found duplicate {dup.app_name} window"
            ))
        
        # Suggest focus mode if too cluttered
        if self._workspace_is_cluttered(windows) and user_state == "focused":
            actions.append(AutonomousAction(
                action_type="enable_focus_mode",
                target="workspace",
                params={
                    "keep_apps": self._identify_primary_work_apps(windows),
                    "minimize_others": True
                },
                priority=ActionPriority.MEDIUM,
                confidence=0.85,
                category=ActionCategory.MAINTENANCE,
                reasoning="Suggesting focus mode to reduce distractions"
            ))
        
        return actions
    
    def _group_windows_by_context(self, windows: List[WindowInfo]) -> Dict[str, List[WindowInfo]]:
        """Group windows by project or context"""
        groups = defaultdict(list)
        
        # Simple grouping by app type and content
        for window in windows:
            if "code" in window.app_name.lower() or "ide" in window.window_title.lower():
                groups["development"].append(window)
            elif any(term in window.app_name.lower() for term in ["chrome", "safari", "firefox"]):
                groups["browser"].append(window)
            elif any(term in window.app_name.lower() for term in ["terminal", "iterm", "console"]):
                groups["terminal"].append(window)
            elif any(term in window.app_name.lower() for term in ["slack", "discord", "messages"]):
                groups["communication"].append(window)
            else:
                groups["other"].append(window)
        
        return dict(groups)
    
    def _has_overlapping_windows(self, windows: List[WindowInfo]) -> bool:
        """Check if windows are overlapping significantly"""
        visible_windows = [w for w in windows if w.is_visible]
        
        # Simple overlap detection
        for i, w1 in enumerate(visible_windows):
            for w2 in visible_windows[i+1:]:
                if self._windows_overlap(w1, w2):
                    return True
        
        return False
    
    def _windows_overlap(self, w1: WindowInfo, w2: WindowInfo) -> bool:
        """Check if two windows overlap"""
        # Get bounds
        b1 = w1.bounds
        b2 = w2.bounds
        
        # Check for overlap
        return not (
            b1["x"] + b1["width"] <= b2["x"] or
            b2["x"] + b2["width"] <= b1["x"] or
            b1["y"] + b1["height"] <= b2["y"] or
            b2["y"] + b2["height"] <= b1["y"]
        )
    
    def _suggest_window_reduction(self, windows: List[WindowInfo]) -> List[AutonomousAction]:
        """Suggest which windows to close or minimize"""
        actions = []
        
        # Find inactive windows (simplified logic)
        for window in windows:
            if window.is_visible and "untitled" in window.window_title.lower():
                actions.append(AutonomousAction(
                    action_type="close_window",
                    target=window.app_name,
                    params={"window_id": window.window_id},
                    priority=ActionPriority.LOW,
                    confidence=0.7,
                    category=ActionCategory.MAINTENANCE,
                    reasoning="Closing untitled/empty window"
                ))
        
        return actions
    
    def _find_duplicate_windows(self, windows: List[WindowInfo]) -> List[WindowInfo]:
        """Find duplicate windows of the same app"""
        duplicates = []
        app_windows = defaultdict(list)
        
        for window in windows:
            app_windows[window.app_name].append(window)
        
        for app, wins in app_windows.items():
            if len(wins) > 1:
                # Keep the focused one or the first one
                focused = [w for w in wins if w.is_focused]
                if focused:
                    keep = focused[0]
                else:
                    keep = wins[0]
                
                for w in wins:
                    if w.window_id != keep.window_id and w.window_title == keep.window_title:
                        duplicates.append(w)
        
        return duplicates
    
    def _workspace_is_cluttered(self, windows: List[WindowInfo]) -> bool:
        """Check if workspace is too cluttered"""
        visible_count = len([w for w in windows if w.is_visible])
        
        # Consider cluttered if:
        # - More than 10 visible windows
        # - Multiple overlapping windows
        # - Multiple apps with multiple windows
        
        if visible_count > 10:
            return True
        
        if self._has_overlapping_windows(windows):
            return True
        
        return False
    
    def _identify_primary_work_apps(self, windows: List[WindowInfo]) -> List[str]:
        """Identify primary work applications"""
        work_apps = []
        work_patterns = [
            "code", "visual studio", "xcode", "android studio",
            "terminal", "iterm", "console",
            "chrome", "safari", "firefox",
            "notes", "notion", "obsidian"
        ]
        
        seen = set()
        for window in windows:
            if window.is_focused:
                work_apps.append(window.app_name)
                seen.add(window.app_name)
                continue
            
            for pattern in work_patterns:
                if pattern in window.app_name.lower() and window.app_name not in seen:
                    work_apps.append(window.app_name)
                    seen.add(window.app_name)
                    break
        
        return work_apps[:5]  # Limit to 5 primary apps

class SecurityHandler:
    """Autonomous security-related behaviors"""
    
    def __init__(self):
        self.security_apps = ["1Password", "Bitwarden", "LastPass", "Keychain"]
        self.suspicious_patterns = [
            r"unauthorized access|failed login|suspicious activity",
            r"verify your account|confirm your identity",
            r"unusual activity|security alert",
            r"password expired|change password"
        ]
    
    async def handle_security_event(self, event_type: str, 
                                   context: Dict[str, Any]) -> List[AutonomousAction]:
        """Handle security-related events"""
        actions = []
        
        if event_type == "suspicious_login":
            actions.extend(self._handle_suspicious_login(context))
        elif event_type == "password_manager_open":
            actions.extend(self._handle_password_manager(context))
        elif event_type == "sensitive_data_exposed":
            actions.extend(self._handle_sensitive_data(context))
        elif event_type == "security_alert":
            actions.extend(self._handle_security_alert(context))
        
        return actions
    
    def _handle_suspicious_login(self, context: Dict[str, Any]) -> List[AutonomousAction]:
        """Handle suspicious login attempts"""
        actions = []
        
        # Lock sensitive apps
        actions.append(AutonomousAction(
            action_type="lock_sensitive_apps",
            target="security",
            params={"reason": "Suspicious login detected"},
            priority=ActionPriority.CRITICAL,
            confidence=0.95,
            category=ActionCategory.SECURITY,
            reasoning="Protecting sensitive data due to suspicious activity"
        ))
        
        # Notify user
        actions.append(AutonomousAction(
            action_type="security_notification",
            target="user",
            params={
                "message": "Suspicious login attempt detected",
                "severity": "high"
            },
            priority=ActionPriority.CRITICAL,
            confidence=0.95,
            category=ActionCategory.SECURITY,
            reasoning="Alerting user to potential security threat"
        ))
        
        return actions
    
    def _handle_password_manager(self, context: Dict[str, Any]) -> List[AutonomousAction]:
        """Handle password manager visibility"""
        windows = context.get("current_windows", [])
        actions = []
        
        # Check if in meeting or screen sharing
        if context.get("in_meeting", False) or context.get("screen_sharing", False):
            for window in windows:
                if any(app in window.app_name for app in self.security_apps):
                    actions.append(AutonomousAction(
                        action_type="hide_window",
                        target=window.app_name,
                        params={"window_id": window.window_id},
                        priority=ActionPriority.CRITICAL,
                        confidence=0.95,
                        category=ActionCategory.SECURITY,
                        reasoning="Hiding password manager during screen sharing"
                    ))
        
        return actions
    
    def _handle_sensitive_data(self, context: Dict[str, Any]) -> List[AutonomousAction]:
        """Handle exposed sensitive data"""
        actions = []
        
        # Immediately blur or hide
        actions.append(AutonomousAction(
            action_type="blur_sensitive_content",
            target=context.get("app_name", "unknown"),
            params={
                "window_id": context.get("window_id"),
                "content_type": context.get("data_type", "unknown")
            },
            priority=ActionPriority.CRITICAL,
            confidence=0.95,
            category=ActionCategory.SECURITY,
            reasoning="Protecting exposed sensitive data"
        ))
        
        return actions
    
    def _handle_security_alert(self, context: Dict[str, Any]) -> List[AutonomousAction]:
        """Handle security alerts"""
        alert_type = context.get("alert_type", "unknown")
        actions = []
        
        if alert_type == "phishing":
            actions.append(AutonomousAction(
                action_type="block_phishing_site",
                target="browser",
                params={"url": context.get("url")},
                priority=ActionPriority.CRITICAL,
                confidence=0.9,
                category=ActionCategory.SECURITY,
                reasoning="Blocking potential phishing attempt"
            ))
        
        return actions

class AutonomousBehaviorManager:
    """Manages all autonomous behaviors (Singleton)"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AutonomousBehaviorManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if AutonomousBehaviorManager._initialized:
            return
            
        AutonomousBehaviorManager._initialized = True
        
        self.message_handler = MessageHandler()
        self.meeting_handler = MeetingHandler()
        self.workspace_organizer = WorkspaceOrganizer()
        self.security_handler = SecurityHandler()
        
        logger.info("Autonomous Behavior Manager initialized")
    
    async def process_workspace_state(self, workspace_state: Dict[str, Any], 
                                     windows: List[WindowInfo]) -> List[AutonomousAction]:
        """Process current workspace state and generate actions"""
        all_actions = []
        
        try:
            # Handle messages
            message_windows = [w for w in windows if self._has_messages(w)]
            for window in message_windows:
                action = await self.message_handler.handle_routine_message(window)
                if action:
                    all_actions.append(action)
            
            # Check for meetings
            meeting_info = self._check_for_meetings(windows)
            if meeting_info:
                meeting_actions = await self.meeting_handler.prepare_for_meeting(
                    meeting_info, windows
                )
                all_actions.extend(meeting_actions)
            
            # Workspace organization
            if self._should_organize_workspace(workspace_state):
                org_actions = await self.workspace_organizer.analyze_and_organize(
                    windows, workspace_state.get("user_state", "available")
                )
                all_actions.extend(org_actions)
            
            # Security checks
            security_events = self._check_security_events(windows, workspace_state)
            for event in security_events:
                security_actions = await self.security_handler.handle_security_event(
                    event["type"], event["context"]
                )
                all_actions.extend(security_actions)
            
            # Deduplicate and prioritize actions
            all_actions = self._prioritize_actions(all_actions)
            
        except Exception as e:
            logger.error(f"Error processing workspace state: {e}")
        
        return all_actions
    
    def _has_messages(self, window: WindowInfo) -> bool:
        """Check if window has unread messages or important message content"""
        # Standard unread indicators
        unread_indicators = [
            r"\(\d+\)",  # (5) format
            r"\d+ new",  # 5 new messages
            r"unread",
            r"notification"
        ]
        
        # Important message types that should be processed
        important_indicators = [
            r"urgent|emergency|critical|asap|important",
            r"security alert|suspicious|unauthorized|breach",
            r"meeting in \d+ minutes?|starts? (at|in)|scheduled for",
            r"automated|automatic|bot|digest|newsletter"
        ]
        
        title_lower = window.window_title.lower()
        
        # Check for unread indicators
        has_unread = any(re.search(pattern, title_lower) for pattern in unread_indicators)
        
        # Check for important message content in typical communication apps
        is_communication_app = any(app in window.app_name.lower() for app in 
                                 ["mail", "email", "slack", "messages", "calendar", "teams", "discord"])
        has_important_content = any(re.search(pattern, title_lower) for pattern in important_indicators)
        
        return has_unread or (is_communication_app and has_important_content)
    
    def _check_for_meetings(self, windows: List[WindowInfo]) -> Optional[Dict[str, Any]]:
        """Check for upcoming meetings"""
        for window in windows:
            if "calendar" in window.app_name.lower() or "meeting" in window.window_title.lower():
                # Simple pattern matching for meeting times
                match = re.search(r"(\d+) minutes?|starts at (\d{1,2}:\d{2})", 
                                window.window_title, re.IGNORECASE)
                if match:
                    return {
                        "title": window.window_title,
                        "time": match.group(0),
                        "window": window
                    }
        return None
    
    def _should_organize_workspace(self, workspace_state: Dict[str, Any]) -> bool:
        """Determine if workspace needs organization"""
        # Check various indicators
        window_count = workspace_state.get("window_count", 0)
        last_organized = workspace_state.get("last_organized")
        user_state = workspace_state.get("user_state", "available")
        
        # Don't organize during focused work
        if user_state == "focused":
            return False
        
        # Organize if too many windows
        if window_count > 15:
            return True
        
        # Organize if not done recently (simplified check)
        if last_organized is None:
            return True
        
        return False
    
    def _check_security_events(self, windows: List[WindowInfo], 
                              workspace_state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for security-related events"""
        events = []
        
        # Check for password managers during meetings
        if workspace_state.get("in_meeting", False):
            for window in windows:
                if any(app in window.app_name for app in ["1Password", "Bitwarden", "LastPass"]):
                    events.append({
                        "type": "password_manager_open",
                        "context": {
                            "current_windows": windows,
                            "in_meeting": True
                        }
                    })
        
        # Check for security alerts in window titles
        for window in windows:
            if re.search(r"security alert|suspicious|unauthorized", 
                        window.window_title, re.IGNORECASE):
                events.append({
                    "type": "security_alert",
                    "context": {
                        "window": window,
                        "alert_type": "suspicious_activity"
                    }
                })
        
        return events
    
    def _prioritize_actions(self, actions: List[AutonomousAction]) -> List[AutonomousAction]:
        """Prioritize and deduplicate actions"""
        # Sort by priority (ascending - lower values first) and confidence (descending - higher confidence first)
        actions.sort(key=lambda a: (a.priority.value, -a.confidence))
        
        # Simple deduplication - keep first occurrence of each action type per target
        seen = set()
        deduped = []
        
        for action in actions:
            key = (action.action_type, action.target)
            if key not in seen:
                seen.add(key)
                deduped.append(action)
        
        # Limit to reasonable number of actions
        return deduped[:10]

def test_autonomous_behaviors():
    """Test the autonomous behaviors"""
    import asyncio
    
    async def run_test():
        manager = AutonomousBehaviorManager()
        
        # Test windows
        test_windows = [
            WindowInfo(
                window_id=1,
                app_name="Slack",
                window_title="Slack (5 new messages)",
                bounds={"x": 0, "y": 0, "width": 800, "height": 600},
                is_focused=False,
                layer=0,
                is_visible=True,
                process_id=1001
            ),
            WindowInfo(
                window_id=2,
                app_name="Calendar",
                window_title="Team Standup starts in 5 minutes",
                bounds={"x": 800, "y": 0, "width": 600, "height": 400},
                is_focused=True,
                layer=0,
                is_visible=True,
                process_id=1002
            ),
            WindowInfo(
                window_id=3,
                app_name="1Password",
                window_title="1Password - Vault",
                bounds={"x": 0, "y": 400, "width": 400, "height": 300},
                is_focused=False,
                layer=1,
                is_visible=True,
                process_id=1003
            )
        ]
        
        # Test workspace state
        workspace_state = {
            "window_count": len(test_windows),
            "user_state": "available",
            "in_meeting": False,
            "last_organized": None
        }
        
        # Process and get actions
        actions = await manager.process_workspace_state(workspace_state, test_windows)
        
        print("\n" + "="*60)
        print("AUTONOMOUS BEHAVIOR TEST RESULTS")
        print("="*60)
        print(f"\nGenerated {len(actions)} autonomous actions:\n")
        
        for i, action in enumerate(actions, 1):
            print(f"{i}. {action.action_type}")
            print(f"   Target: {action.target}")
            print(f"   Priority: {action.priority.name}")
            print(f"   Confidence: {action.confidence:.0%}")
            print(f"   Reasoning: {action.reasoning}")
            print()
    
    asyncio.run(run_test())

if __name__ == "__main__":
    test_autonomous_behaviors()