#!/usr/bin/env python3
"""
Window Analysis Module for JARVIS Vision System
Analyzes and categorizes application windows and their content
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from collections import defaultdict

from .window_detector import WindowDetector, WindowInfo
from .ocr_processor import OCRProcessor, OCRResult
from .screen_capture_module import ScreenCaptureModule, ScreenCapture

logger = logging.getLogger(__name__)

class ApplicationCategory(Enum):
    """Categories of applications"""
    BROWSER = "browser"
    COMMUNICATION = "communication"
    DEVELOPMENT = "development"
    PRODUCTIVITY = "productivity"
    MEDIA = "media"
    SYSTEM = "system"
    UTILITY = "utility"
    UNKNOWN = "unknown"

class WindowState(Enum):
    """States a window can be in"""
    ACTIVE = "active"
    IDLE = "idle"
    WAITING = "waiting"
    ERROR = "error"
    LOADING = "loading"

@dataclass
class WindowContent:
    """Analyzed content of a window"""
    window_id: int
    app_name: str
    category: ApplicationCategory
    state: WindowState
    title_elements: List[str]
    action_items: List[Dict[str, Any]]
    notifications: List[Dict[str, Any]]
    key_information: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def has_urgent_items(self) -> bool:
        """Check if window has urgent items requiring attention"""
        return any(n.get('urgent', False) for n in self.notifications)
        
    @property
    def action_count(self) -> int:
        """Number of actionable items in window"""
        return len(self.action_items)

@dataclass
class WorkspaceLayout:
    """Analyzed workspace layout"""
    primary_app: Optional[str] = None
    layout_type: str = "single"  # single, split, grid
    window_arrangement: Dict[str, List[WindowInfo]] = field(default_factory=dict)
    screen_utilization: float = 0.0
    overlap_detected: bool = False

class WindowAnalyzer:
    """Analyzes windows to understand their content and purpose"""
    
    def __init__(self):
        self.window_detector = WindowDetector()
        self.ocr_processor = OCRProcessor()
        self.screen_capture = ScreenCaptureModule()
        
        # Application categorization rules
        self.app_categories = {
            ApplicationCategory.BROWSER: [
                'chrome', 'safari', 'firefox', 'edge', 'opera', 'brave'
            ],
            ApplicationCategory.COMMUNICATION: [
                'slack', 'teams', 'zoom', 'discord', 'messages', 'mail', 
                'outlook', 'skype', 'telegram', 'whatsapp'
            ],
            ApplicationCategory.DEVELOPMENT: [
                'code', 'vscode', 'sublime', 'atom', 'intellij', 'xcode', 
                'terminal', 'iterm', 'docker', 'postman'
            ],
            ApplicationCategory.PRODUCTIVITY: [
                'notion', 'obsidian', 'word', 'excel', 'powerpoint', 
                'pages', 'numbers', 'keynote', 'google docs'
            ],
            ApplicationCategory.MEDIA: [
                'spotify', 'music', 'vlc', 'quicktime', 'photos', 
                'preview', 'photoshop', 'figma'
            ],
            ApplicationCategory.SYSTEM: [
                'finder', 'activity monitor', 'system preferences', 
                'settings', 'installer'
            ]
        }
        
        # Window state detection patterns
        self.state_patterns = {
            WindowState.LOADING: [
                'loading', 'please wait', 'processing', 'connecting'
            ],
            WindowState.ERROR: [
                'error', 'failed', 'cannot', 'unable', 'exception'
            ],
            WindowState.WAITING: [
                'waiting', 'pending', 'paused', 'stopped'
            ]
        }
        
        # Notification patterns
        self.notification_patterns = {
            'message': r'\((\d+)\)|(\d+)\s+(new|unread)\s+(message|notification)',
            'update': r'update\s+available|new\s+version|upgrade',
            'alert': r'alert|warning|attention|important',
            'reminder': r'reminder|due|deadline|scheduled'
        }
        
    def categorize_application(self, app_name: str) -> ApplicationCategory:
        """Categorize an application based on its name"""
        app_lower = app_name.lower()
        
        for category, apps in self.app_categories.items():
            if any(app in app_lower for app in apps):
                return category
                
        return ApplicationCategory.UNKNOWN
        
    def detect_window_state(self, window_title: str, ocr_text: str = "") -> WindowState:
        """Detect the current state of a window"""
        combined_text = f"{window_title} {ocr_text}".lower()
        
        for state, patterns in self.state_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                return state
                
        return WindowState.ACTIVE
        
    async def analyze_window(self, window: WindowInfo, 
                           capture: Optional[ScreenCapture] = None) -> WindowContent:
        """Analyze a single window's content"""
        # Get window screenshot if not provided
        if not capture:
            capture = self.screen_capture.capture_screen(
                region=(window.x, window.y, window.width, window.height)
            )
            
        # Initialize content
        content = WindowContent(
            window_id=window.window_id,
            app_name=window.app_name,
            category=self.categorize_application(window.app_name),
            state=WindowState.ACTIVE,
            title_elements=[],
            action_items=[],
            notifications=[],
            key_information={}
        )
        
        # Perform OCR if we have a capture
        if capture:
            ocr_result = await self.ocr_processor.process_image(capture.image)
            
            # Extract structured data
            structured = self.ocr_processor.extract_structured_data(ocr_result)
            
            # Update content based on OCR
            content.title_elements = structured.get('titles', [])
            
            # Detect state from OCR text
            content.state = self.detect_window_state(
                window.window_title, 
                ocr_result.full_text
            )
            
            # Extract action items
            for button in structured.get('buttons', []):
                content.action_items.append({
                    'type': 'button',
                    'text': button['text'],
                    'location': button['location'],
                    'clickable': True
                })
                
            # Detect notifications
            content.notifications = self._detect_notifications(
                window.window_title,
                ocr_result.full_text
            )
            
            # Extract key information based on app category
            content.key_information = self._extract_key_info(
                content.category,
                structured,
                ocr_result
            )
        else:
            # Basic analysis from window title only
            content.state = self.detect_window_state(window.window_title)
            content.notifications = self._detect_notifications(window.window_title)
            
        return content
        
    def _detect_notifications(self, window_title: str, 
                            ocr_text: str = "") -> List[Dict[str, Any]]:
        """Detect notifications in window"""
        notifications = []
        combined_text = f"{window_title} {ocr_text}"
        
        for notif_type, pattern in self.notification_patterns.items():
            matches = re.finditer(pattern, combined_text, re.IGNORECASE)
            for match in matches:
                notification = {
                    'type': notif_type,
                    'match': match.group(),
                    'urgent': notif_type in ['alert', 'reminder']
                }
                
                # Extract count if available
                if match.groups():
                    try:
                        count = int(match.group(1) or match.group(2))
                        notification['count'] = count
                    except:
                        pass
                        
                notifications.append(notification)
                
        return notifications
        
    def _extract_key_info(self, category: ApplicationCategory, 
                         structured: Dict[str, Any],
                         ocr_result: OCRResult) -> Dict[str, Any]:
        """Extract key information based on app category"""
        key_info = {}
        
        if category == ApplicationCategory.BROWSER:
            # Extract URL, page title, etc.
            key_info['urls'] = structured.get('urls', [])
            key_info['page_title'] = structured['titles'][0] if structured['titles'] else None
            
        elif category == ApplicationCategory.COMMUNICATION:
            # Extract sender, message preview, etc.
            key_info['has_messages'] = bool(structured.get('numbers', []))
            key_info['email_addresses'] = structured.get('emails', [])
            
            # Look for sender patterns
            for region in ocr_result.regions:
                if region.area_type == 'label' and 'from' in region.text.lower():
                    key_info['sender'] = region.text
                    
        elif category == ApplicationCategory.DEVELOPMENT:
            # Extract file names, error messages, etc.
            key_info['file_paths'] = []
            for text in ocr_result.full_text.split('\n'):
                if '/' in text or '\\' in text:
                    key_info['file_paths'].append(text.strip())
                    
            # Look for error patterns
            error_patterns = ['error:', 'exception:', 'failed:']
            for pattern in error_patterns:
                if pattern in ocr_result.full_text.lower():
                    key_info['has_errors'] = True
                    break
                    
        return key_info
        
    async def analyze_workspace(self, windows: Optional[List[WindowInfo]] = None) -> Dict[str, Any]:
        """Analyze the entire workspace"""
        if not windows:
            windows = self.window_detector.get_all_windows()
            
        # Analyze layout
        layout = self._analyze_layout(windows)
        
        # Analyze each visible window
        window_analyses = []
        for window in windows:
            if window.is_visible:
                analysis = await self.analyze_window(window)
                window_analyses.append(analysis)
                
        # Summarize workspace
        summary = self._summarize_workspace(window_analyses, layout)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'layout': layout,
            'windows': window_analyses,
            'summary': summary
        }
        
    def _analyze_layout(self, windows: List[WindowInfo]) -> WorkspaceLayout:
        """Analyze the layout of windows"""
        layout = WorkspaceLayout()
        
        if not windows:
            return layout
            
        # Find primary (focused) app
        focused = next((w for w in windows if w.is_focused), None)
        if focused:
            layout.primary_app = focused.app_name
            
        # Group windows by screen position
        quadrants = defaultdict(list)
        for window in windows:
            if window.is_visible:
                # Determine quadrant
                quad_x = 'left' if window.x < 960 else 'right'
                quad_y = 'top' if window.y < 540 else 'bottom'
                quadrant = f"{quad_y}_{quad_x}"
                quadrants[quadrant].append(window)
                
        # Determine layout type
        if len(quadrants) >= 3:
            layout.layout_type = 'grid'
        elif len(quadrants) == 2:
            layout.layout_type = 'split'
        else:
            layout.layout_type = 'single'
            
        # Check for overlapping windows
        layout.overlap_detected = self._check_overlap(windows)
        
        # Calculate screen utilization
        if windows:
            total_area = sum(w.width * w.height for w in windows if w.is_visible)
            screen_area = 1920 * 1080  # Assuming standard resolution
            layout.screen_utilization = min(total_area / screen_area, 1.0)
            
        return layout
        
    def _check_overlap(self, windows: List[WindowInfo]) -> bool:
        """Check if windows overlap significantly"""
        visible_windows = [w for w in windows if w.is_visible]
        
        for i, w1 in enumerate(visible_windows):
            for w2 in visible_windows[i+1:]:
                # Check if windows overlap
                if (w1.x < w2.x + w2.width and
                    w1.x + w1.width > w2.x and
                    w1.y < w2.y + w2.height and
                    w1.y + w1.height > w2.y):
                    
                    # Calculate overlap area
                    overlap_x = min(w1.x + w1.width, w2.x + w2.width) - max(w1.x, w2.x)
                    overlap_y = min(w1.y + w1.height, w2.y + w2.height) - max(w1.y, w2.y)
                    overlap_area = overlap_x * overlap_y
                    
                    # Check if overlap is significant (>20% of smaller window)
                    smaller_area = min(w1.width * w1.height, w2.width * w2.height)
                    if overlap_area > smaller_area * 0.2:
                        return True
                        
        return False
        
    def _summarize_workspace(self, analyses: List[WindowContent], 
                           layout: WorkspaceLayout) -> Dict[str, Any]:
        """Create a summary of the workspace state"""
        summary = {
            'total_windows': len(analyses),
            'layout_type': layout.layout_type,
            'primary_app': layout.primary_app,
            'categories': defaultdict(int),
            'states': defaultdict(int),
            'urgent_count': 0,
            'total_actions': 0,
            'total_notifications': 0
        }
        
        for analysis in analyses:
            summary['categories'][analysis.category.value] += 1
            summary['states'][analysis.state.value] += 1
            summary['total_actions'] += analysis.action_count
            summary['total_notifications'] += len(analysis.notifications)
            if analysis.has_urgent_items:
                summary['urgent_count'] += 1
                
        return dict(summary)
        
    def get_actionable_windows(self, analyses: List[WindowContent]) -> List[WindowContent]:
        """Get windows that require user action"""
        actionable = []
        
        for analysis in analyses:
            # Check if window needs attention
            if (analysis.has_urgent_items or 
                analysis.action_count > 0 or
                analysis.state in [WindowState.ERROR, WindowState.WAITING]):
                actionable.append(analysis)
                
        # Sort by priority
        actionable.sort(key=lambda x: (
            not x.has_urgent_items,  # Urgent first
            x.state != WindowState.ERROR,  # Errors second
            -x.action_count  # More actions = higher priority
        ))
        
        return actionable

async def test_window_analyzer():
    """Test window analysis functionality"""
    print("🪟 Testing Window Analyzer")
    print("=" * 50)
    
    analyzer = WindowAnalyzer()
    
    # Get current windows
    windows = analyzer.window_detector.get_all_windows()
    print(f"\n📊 Found {len(windows)} windows")
    
    if windows:
        # Analyze first few windows
        print("\n🔍 Analyzing windows...")
        
        for window in windows[:3]:
            if window.is_visible:
                print(f"\n   Analyzing: {window.app_name} - {window.window_title}")
                
                analysis = await analyzer.analyze_window(window)
                
                print(f"   Category: {analysis.category.value}")
                print(f"   State: {analysis.state.value}")
                print(f"   Actions: {analysis.action_count}")
                print(f"   Notifications: {len(analysis.notifications)}")
                
                if analysis.notifications:
                    print("   Notifications:")
                    for notif in analysis.notifications:
                        print(f"     - {notif['type']}: {notif.get('match', 'N/A')}")
                        
        # Analyze complete workspace
        print("\n🏢 Analyzing complete workspace...")
        workspace = await analyzer.analyze_workspace(windows)
        
        print(f"\n📋 Workspace Summary:")
        summary = workspace['summary']
        print(f"   Layout: {summary['layout_type']}")
        print(f"   Primary App: {summary['primary_app']}")
        print(f"   Total Windows: {summary['total_windows']}")
        print(f"   Urgent Items: {summary['urgent_count']}")
        print(f"   Total Actions: {summary['total_actions']}")
        
        print("\n   App Categories:")
        for cat, count in summary['categories'].items():
            print(f"     {cat}: {count}")
            
    print("\n✅ Window analyzer test complete!")

if __name__ == "__main__":
    asyncio.run(test_window_analyzer())