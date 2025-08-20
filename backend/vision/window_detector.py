#!/usr/bin/env python3
"""
Window Detection System for JARVIS Multi-Window Awareness
Detects and tracks all open windows on macOS
"""

import Quartz
import time
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WindowInfo:
    """Information about a single window"""
    window_id: int
    app_name: str
    window_title: str
    is_focused: bool
    bounds: Dict[str, int]  # x, y, width, height
    layer: int
    is_visible: bool
    process_id: int


class WindowDetector:
    """Detects and tracks all open windows on macOS"""
    
    def __init__(self):
        self.last_update = 0
        self.windows_cache = []
        self.focused_window_id = None
        self.update_interval = 0.1  # 100ms update interval
        
    def get_all_windows(self) -> List[WindowInfo]:
        """Get information about all open windows"""
        windows = []
        
        # Get window list from Quartz
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll | 
            Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID
        )
        
        if not window_list:
            logger.warning("No windows found")
            return windows
        
        # Get the frontmost application
        frontmost_app = Quartz.NSWorkspace.sharedWorkspace().frontmostApplication()
        frontmost_pid = frontmost_app.processIdentifier() if frontmost_app else None
        
        for window in window_list:
            # Extract window information
            window_id = window.get('kCGWindowNumber', 0)
            app_name = window.get('kCGWindowOwnerName', 'Unknown')
            window_title = window.get('kCGWindowName', '')
            process_id = window.get('kCGWindowOwnerPID', 0)
            layer = window.get('kCGWindowLayer', 0)
            alpha = window.get('kCGWindowAlpha', 0)
            
            # Skip invisible windows
            if alpha == 0:
                continue
            
            # Skip system UI elements
            if app_name in ['Window Server', 'SystemUIServer', 'Dock', 
                           'Control Center', 'Notification Center']:
                continue
                
            # Get window bounds
            bounds_dict = window.get('kCGWindowBounds', {})
            bounds = {
                'x': int(bounds_dict.get('X', 0)),
                'y': int(bounds_dict.get('Y', 0)),
                'width': int(bounds_dict.get('Width', 0)),
                'height': int(bounds_dict.get('Height', 0))
            }
            
            # Skip very small windows (likely UI elements)
            if bounds['width'] < 100 or bounds['height'] < 100:
                continue
            
            # Skip windows that are off-screen
            if bounds['x'] < -1000 or bounds['y'] < -1000:
                continue
            
            # Determine if window is focused
            is_focused = (process_id == frontmost_pid and layer == 0)
            
            # Track focused window
            if is_focused:
                self.focused_window_id = window_id
            
            window_info = WindowInfo(
                window_id=window_id,
                app_name=app_name,
                window_title=window_title,
                is_focused=is_focused,
                bounds=bounds,
                layer=layer,
                is_visible=alpha > 0,
                process_id=process_id
            )
            
            windows.append(window_info)
        
        # Sort windows by layer (frontmost first)
        windows.sort(key=lambda w: (w.layer, -w.bounds['width'] * w.bounds['height']))
        
        self.windows_cache = windows
        self.last_update = time.time()
        
        return windows
    
    def get_focused_window(self) -> Optional[WindowInfo]:
        """Get the currently focused window"""
        windows = self.get_all_windows()
        for window in windows:
            if window.is_focused:
                return window
        return None
    
    def get_windows_by_app(self, app_name: str) -> List[WindowInfo]:
        """Get all windows for a specific application"""
        windows = self.get_all_windows()
        return [w for w in windows if app_name.lower() in w.app_name.lower()]
    
    def detect_window_changes(self, previous_windows: List[WindowInfo]) -> Dict[str, Any]:
        """Detect changes between window states"""
        current_windows = self.get_all_windows()
        
        # Create window ID sets for comparison
        prev_ids = {w.window_id for w in previous_windows}
        curr_ids = {w.window_id for w in current_windows}
        
        # Detect changes
        opened_windows = curr_ids - prev_ids
        closed_windows = prev_ids - curr_ids
        
        # Detect focus changes
        prev_focused = next((w for w in previous_windows if w.is_focused), None)
        curr_focused = next((w for w in current_windows if w.is_focused), None)
        
        focus_changed = (
            (prev_focused.window_id if prev_focused else None) != 
            (curr_focused.window_id if curr_focused else None)
        )
        
        return {
            'opened': list(opened_windows),
            'closed': list(closed_windows),
            'focus_changed': focus_changed,
            'current_focus': curr_focused,
            'total_windows': len(current_windows)
        }
    
    async def monitor_windows(self, callback=None):
        """Monitor windows for changes in real-time"""
        previous_windows = []
        
        while True:
            try:
                changes = self.detect_window_changes(previous_windows)
                
                if changes['opened'] or changes['closed'] or changes['focus_changed']:
                    logger.info(f"Window changes detected: {changes}")
                    
                    if callback:
                        await callback(changes)
                
                previous_windows = self.windows_cache.copy()
                
            except Exception as e:
                logger.error(f"Error monitoring windows: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    def get_workspace_summary(self) -> Dict[str, Any]:
        """Get a summary of the current workspace"""
        windows = self.get_all_windows()
        
        # Group windows by application
        apps = {}
        for window in windows:
            if window.app_name not in apps:
                apps[window.app_name] = []
            apps[window.app_name].append(window)
        
        # Identify common productivity apps
        productivity_apps = {
            'development': ['Visual Studio Code', 'Xcode', 'Terminal', 'iTerm'],
            'browser': ['Chrome', 'Safari', 'Firefox', 'Edge'],
            'communication': ['Discord', 'Slack', 'Messages', 'Mail'],
            'documentation': ['Preview', 'Notion', 'Obsidian', 'Notes']
        }
        
        # Categorize windows
        categorized = {category: [] for category in productivity_apps}
        uncategorized = []
        
        for app_name, window_list in apps.items():
            categorized_flag = False
            for category, app_keywords in productivity_apps.items():
                if any(keyword.lower() in app_name.lower() for keyword in app_keywords):
                    categorized[category].extend(window_list)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                uncategorized.extend(window_list)
        
        return {
            'total_windows': len(windows),
            'focused_window': self.get_focused_window(),
            'applications': list(apps.keys()),
            'categorized_windows': {
                cat: len(windows) for cat, windows in categorized.items() if windows
            },
            'window_details': windows[:10]  # Top 10 windows by relevance
        }


# Utility functions for external use
def list_all_windows() -> List[Dict[str, Any]]:
    """Simple function to list all windows"""
    detector = WindowDetector()
    windows = detector.get_all_windows()
    
    return [{
        'app': w.app_name,
        'title': w.window_title,
        'focused': w.is_focused,
        'size': f"{w.bounds['width']}x{w.bounds['height']}",
        'position': f"({w.bounds['x']}, {w.bounds['y']})"
    } for w in windows]


def get_focused_app() -> Optional[str]:
    """Get the name of the currently focused application"""
    detector = WindowDetector()
    focused = detector.get_focused_window()
    return focused.app_name if focused else None


async def test_window_detection():
    """Test the window detection system"""
    print("🔍 Testing Window Detection System")
    print("=" * 50)
    
    detector = WindowDetector()
    
    # Test 1: List all windows
    print("\n1️⃣ All Open Windows:")
    windows = detector.get_all_windows()
    for i, window in enumerate(windows[:10]):  # Show first 10
        focus_indicator = "🎯 " if window.is_focused else "   "
        print(f"{focus_indicator}{i+1}. {window.app_name}: {window.window_title or 'Untitled'}")
        print(f"      Size: {window.bounds['width']}x{window.bounds['height']}")
    
    if len(windows) > 10:
        print(f"   ... and {len(windows) - 10} more windows")
    
    # Test 2: Focused window
    print("\n2️⃣ Currently Focused Window:")
    focused = detector.get_focused_window()
    if focused:
        print(f"   App: {focused.app_name}")
        print(f"   Title: {focused.window_title}")
        print(f"   Size: {focused.bounds['width']}x{focused.bounds['height']}")
    
    # Test 3: Workspace summary
    print("\n3️⃣ Workspace Summary:")
    summary = detector.get_workspace_summary()
    print(f"   Total Windows: {summary['total_windows']}")
    print(f"   Applications: {', '.join(summary['applications'][:5])}")
    
    if summary['categorized_windows']:
        print("\n   Window Categories:")
        for category, count in summary['categorized_windows'].items():
            print(f"   - {category.capitalize()}: {count} windows")
    
    # Test 4: Monitor for changes (5 seconds)
    print("\n4️⃣ Monitoring for window changes (5 seconds)...")
    print("   Try switching windows or opening/closing apps")
    
    async def change_callback(changes):
        if changes['focus_changed']:
            focus = changes['current_focus']
            if focus:
                print(f"   → Focus changed to: {focus.app_name}")
        if changes['opened']:
            print(f"   → New windows opened: {changes['opened']}")
        if changes['closed']:
            print(f"   → Windows closed: {changes['closed']}")
    
    # Monitor for 5 seconds
    monitor_task = asyncio.create_task(detector.monitor_windows(change_callback))
    await asyncio.sleep(5)
    monitor_task.cancel()
    
    print("\n✅ Window detection test complete!")


if __name__ == "__main__":
    asyncio.run(test_window_detection())