#!/usr/bin/env python3
"""
Enhanced Workspace Monitoring for JARVIS
Provides advanced detection of notifications, UI elements, and actionable items
"""

import re
import asyncio
import logging
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import cv2
import numpy as np

from .window_detector import WindowDetector, WindowInfo
from .workspace_analyzer import WorkspaceAnalyzer

logger = logging.getLogger(__name__)

class NotificationDetector:
    """Detects and classifies notifications across applications"""
    
    def __init__(self):
        # Notification patterns for different apps
        self.notification_patterns = {
            'badge': [
                r'\((\d+)\)',           # (5) style
                r'\[(\d+)\]',           # [3] style
                r'●\s*(\d+)',           # • 5 style
                r'(\d+)\s+new',         # 5 new
                r'(\d+)\s+unread',      # 3 unread
            ],
            'urgent': [
                'urgent', 'critical', 'important', 'asap', 'emergency',
                'deadline', 'overdue', 'action required', 'attention'
            ],
            'meeting': [
                'meeting', 'standup', 'sync', 'call', 'conference',
                'zoom', 'teams', 'meet', 'hangout', 'webex'
            ],
            'message': [
                'message', 'chat', 'dm', 'reply', 'mention',
                'comment', 'notification', 'alert'
            ]
        }
        
    def detect_notifications(self, windows: List[WindowInfo]) -> Dict[str, List[Dict[str, Any]]]:
        """Detect notifications across all windows"""
        notifications = {
            'messages': [],
            'meetings': [],
            'alerts': [],
            'badges': []
        }
        
        for window in windows:
            # Check window title for notifications
            if not window.window_title:
                continue
                
            title_lower = window.window_title.lower()
            
            # Check for badge-style notifications
            for pattern in self.notification_patterns['badge']:
                match = re.search(pattern, window.window_title)
                if match:
                    count = int(match.group(1)) if match.groups() else 0
                    notifications['badges'].append({
                        'app': window.app_name,
                        'count': count,
                        'window_id': window.window_id,
                        'type': 'badge'
                    })
                    
            # Check for urgent notifications
            if any(urgent in title_lower for urgent in self.notification_patterns['urgent']):
                notifications['alerts'].append({
                    'app': window.app_name,
                    'title': window.window_title,
                    'window_id': window.window_id,
                    'type': 'urgent',
                    'priority': 'high'
                })
                
            # Check for meeting notifications
            if any(meeting in title_lower for meeting in self.notification_patterns['meeting']):
                # Extract meeting time if present
                time_match = re.search(r'(\d+)\s*min', title_lower)
                minutes = int(time_match.group(1)) if time_match else None
                
                notifications['meetings'].append({
                    'app': window.app_name,
                    'title': window.window_title,
                    'window_id': window.window_id,
                    'type': 'meeting',
                    'minutes_until': minutes
                })
                
            # Check for message notifications
            if any(msg in title_lower for msg in self.notification_patterns['message']):
                notifications['messages'].append({
                    'app': window.app_name,
                    'title': window.window_title,
                    'window_id': window.window_id,
                    'type': 'message'
                })
                
        return notifications

class ApplicationStateTracker:
    """Tracks application states and detects changes"""
    
    def __init__(self):
        self.app_states = {}
        self.state_history = defaultdict(list)
        self.change_callbacks = []
        
    def update_state(self, windows: List[WindowInfo]) -> List[Dict[str, Any]]:
        """Update application states and detect changes"""
        changes = []
        current_apps = {}
        
        # Build current state
        for window in windows:
            app_name = window.app_name
            if app_name not in current_apps:
                current_apps[app_name] = {
                    'windows': [],
                    'focused': False,
                    'visible_count': 0
                }
            
            current_apps[app_name]['windows'].append(window)
            if window.is_focused:
                current_apps[app_name]['focused'] = True
            if window.is_visible:
                current_apps[app_name]['visible_count'] += 1
                
        # Detect changes
        for app_name, state in current_apps.items():
            if app_name not in self.app_states:
                # New app opened
                changes.append({
                    'type': 'app_opened',
                    'app': app_name,
                    'window_count': len(state['windows']),
                    'timestamp': datetime.now()
                })
            else:
                # Check for changes
                old_state = self.app_states[app_name]
                if len(state['windows']) > len(old_state['windows']):
                    changes.append({
                        'type': 'window_opened',
                        'app': app_name,
                        'new_count': len(state['windows']),
                        'timestamp': datetime.now()
                    })
                elif state['focused'] and not old_state['focused']:
                    changes.append({
                        'type': 'app_focused',
                        'app': app_name,
                        'timestamp': datetime.now()
                    })
                    
        # Check for closed apps
        for app_name in self.app_states:
            if app_name not in current_apps:
                changes.append({
                    'type': 'app_closed',
                    'app': app_name,
                    'timestamp': datetime.now()
                })
                
        # Update states
        self.app_states = current_apps
        
        # Record history
        for change in changes:
            self.state_history[change['app']].append(change)
            
        return changes
        
    def get_app_activity(self, app_name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get activity statistics for an app"""
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        
        events = [e for e in self.state_history.get(app_name, []) 
                  if e['timestamp'] > cutoff]
        
        return {
            'app': app_name,
            'open_count': sum(1 for e in events if e['type'] == 'app_opened'),
            'focus_count': sum(1 for e in events if e['type'] == 'app_focused'),
            'window_opened': sum(1 for e in events if e['type'] == 'window_opened'),
            'last_active': max((e['timestamp'] for e in events), default=None)
        }

class UIElementDetector:
    """Detects actionable UI elements in windows"""
    
    def __init__(self):
        self.actionable_patterns = {
            'buttons': [
                r'(?i)\b(ok|cancel|submit|save|delete|confirm|accept|reject)\b',
                r'(?i)\b(yes|no|continue|proceed|abort)\b'
            ],
            'alerts': [
                r'(?i)\b(error|warning|alert|notice|attention)\b',
                r'(?i)\b(failed|failure|problem|issue)\b'
            ],
            'prompts': [
                r'(?i)\b(enter|input|type|select|choose)\b',
                r'(?i)\b(password|username|email|name)\b'
            ]
        }
        
    def detect_actionable_elements(self, window: WindowInfo) -> List[Dict[str, Any]]:
        """Detect actionable UI elements in a window"""
        elements = []
        
        if not window.window_title:
            return elements
            
        title_lower = window.window_title.lower()
        
        # Check for buttons
        for pattern in self.actionable_patterns['buttons']:
            if re.search(pattern, title_lower):
                elements.append({
                    'type': 'button',
                    'window_id': window.window_id,
                    'app': window.app_name,
                    'context': window.window_title,
                    'action_required': True
                })
                break
                
        # Check for alerts
        for pattern in self.actionable_patterns['alerts']:
            if re.search(pattern, title_lower):
                elements.append({
                    'type': 'alert',
                    'window_id': window.window_id,
                    'app': window.app_name,
                    'context': window.window_title,
                    'priority': 'high'
                })
                break
                
        # Check for prompts
        for pattern in self.actionable_patterns['prompts']:
            if re.search(pattern, title_lower):
                elements.append({
                    'type': 'prompt',
                    'window_id': window.window_id,
                    'app': window.app_name,
                    'context': window.window_title,
                    'needs_input': True
                })
                break
                
        return elements

class ClipboardMonitor:
    """Monitors clipboard for relevant content"""
    
    def __init__(self):
        self.last_content = None
        self.content_history = []
        self.max_history = 10
        
    async def check_clipboard(self) -> Optional[Dict[str, Any]]:
        """Check clipboard for new content"""
        try:
            import AppKit
            
            pasteboard = AppKit.NSPasteboard.generalPasteboard()
            content = pasteboard.stringForType_(AppKit.NSPasteboardTypeString)
            
            if content and content != self.last_content:
                self.last_content = content
                
                # Analyze content
                content_info = {
                    'text': content[:200],  # Limit for privacy
                    'length': len(content),
                    'timestamp': datetime.now(),
                    'type': self._classify_content(content)
                }
                
                # Add to history
                self.content_history.append(content_info)
                if len(self.content_history) > self.max_history:
                    self.content_history.pop(0)
                    
                return content_info
                
        except Exception as e:
            logger.error(f"Clipboard monitoring error: {e}")
            
        return None
        
    def _classify_content(self, content: str) -> str:
        """Classify clipboard content type"""
        if re.match(r'^https?://', content):
            return 'url'
        elif re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', content):
            return 'email'
        elif re.match(r'^[\d\s\-\+\(\)]+$', content) and len(content) > 7:
            return 'phone'
        elif len(content.split()) > 10:
            return 'text_long'
        else:
            return 'text_short'

class EnhancedWorkspaceMonitor:
    """Enhanced workspace monitoring with all capabilities"""
    
    def __init__(self):
        self.window_detector = WindowDetector()
        self.workspace_analyzer = WorkspaceAnalyzer()
        self.notification_detector = NotificationDetector()
        self.state_tracker = ApplicationStateTracker()
        self.ui_detector = UIElementDetector()
        self.clipboard_monitor = ClipboardMonitor()
        
        self.monitoring_active = False
        self.update_callbacks = []
        
    async def get_complete_workspace_state(self) -> Dict[str, Any]:
        """Get complete workspace state with all monitoring data"""
        
        # Get windows
        windows = self.window_detector.get_all_windows()
        
        # Analyze workspace
        workspace_analysis = await self.workspace_analyzer.analyze_workspace()
        
        # Detect notifications
        notifications = self.notification_detector.detect_notifications(windows)
        
        # Track state changes
        state_changes = self.state_tracker.update_state(windows)
        
        # Detect UI elements
        ui_elements = []
        for window in windows:
            elements = self.ui_detector.detect_actionable_elements(window)
            ui_elements.extend(elements)
            
        # Check clipboard
        clipboard_content = await self.clipboard_monitor.check_clipboard()
        
        # Build complete state
        return {
            'timestamp': datetime.now().isoformat(),
            'windows': windows,
            'analysis': workspace_analysis,
            'notifications': notifications,
            'state_changes': state_changes,
            'ui_elements': ui_elements,
            'clipboard': clipboard_content,
            'stats': {
                'window_count': len(windows),
                'notification_count': sum(len(n) for n in notifications.values()),
                'focused_app': next((w.app_name for w in windows if w.is_focused), None),
                'actionable_items': len(ui_elements)
            }
        }
        
    async def start_monitoring(self, interval: float = 2.0):
        """Start continuous monitoring"""
        self.monitoring_active = True
        
        while self.monitoring_active:
            try:
                state = await self.get_complete_workspace_state()
                
                # Notify callbacks
                for callback in self.update_callbacks:
                    await callback(state)
                    
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(interval)
                
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        
    def add_update_callback(self, callback):
        """Add callback for workspace updates"""
        self.update_callbacks.append(callback)

async def test_enhanced_monitoring():
    """Test enhanced monitoring capabilities"""
    monitor = EnhancedWorkspaceMonitor()
    
    print("🔍 Enhanced Workspace Monitoring Test")
    print("=" * 50)
    
    # Get one complete state
    state = await monitor.get_complete_workspace_state()
    
    print(f"\n📊 Workspace Stats:")
    print(f"   Windows: {state['stats']['window_count']}")
    print(f"   Notifications: {state['stats']['notification_count']}")
    print(f"   Focused App: {state['stats']['focused_app']}")
    print(f"   Actionable Items: {state['stats']['actionable_items']}")
    
    # Show notifications
    if any(state['notifications'].values()):
        print(f"\n🔔 Notifications Detected:")
        for category, items in state['notifications'].items():
            if items:
                print(f"   {category.title()}: {len(items)}")
                for item in items[:2]:  # Show first 2
                    print(f"      - {item['app']}: {item.get('title', item.get('count', 'N/A'))}")
    
    # Show UI elements
    if state['ui_elements']:
        print(f"\n🎯 Actionable UI Elements:")
        for element in state['ui_elements'][:3]:
            print(f"   - {element['type']} in {element['app']}: {element['context']}")
    
    # Show clipboard
    if state['clipboard']:
        print(f"\n📋 Clipboard: {state['clipboard']['type']} ({state['clipboard']['length']} chars)")
    
    print("\n✅ Enhanced monitoring test complete!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_monitoring())