#!/usr/bin/env python3
"""
Multi-Space Window Detector for JARVIS
Enhanced window detection with space awareness and metadata intelligence
"""

import Quartz
import AppKit
import time
import asyncio
import subprocess
import json
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class SpaceWindowVisibility(Enum):
    """Window visibility states across spaces"""

    VISIBLE_CURRENT_SPACE = "visible_current"
    VISIBLE_OTHER_SPACE = "visible_other"
    MINIMIZED = "minimized"
    HIDDEN = "hidden"
    FULLSCREEN = "fullscreen"


@dataclass
class SpaceInfo:
    """Information about a desktop space"""

    space_id: int
    space_uuid: str
    display_id: int
    is_current: bool
    window_count: int
    last_accessed: Optional[datetime] = None
    cached_screenshot: Optional[Any] = None
    screenshot_timestamp: Optional[datetime] = None


@dataclass
class EnhancedWindowInfo:
    """Enhanced window information with space awareness"""

    # Basic info
    window_id: int
    app_name: str
    window_title: str
    process_id: int
    
    # Space info
    space_id: Optional[int] = None
    space_uuid: Optional[str] = None
    visibility: SpaceWindowVisibility = SpaceWindowVisibility.HIDDEN
    
    # Position and state
    bounds: Dict[str, int] = field(default_factory=dict)  # x, y, width, height
    is_minimized: bool = False
    is_fullscreen: bool = False
    is_focused: bool = False
    layer: int = 0
    alpha: float = 1.0
    
    # Enhanced metadata
    document_path: Optional[str] = None
    document_modified: Optional[datetime] = None
    last_activated: Optional[datetime] = None
    
    # Workspace context
    related_windows: List[int] = field(default_factory=list)
    workspace_role: Optional[str] = None  # "main", "reference", "tool", etc.
    
    def to_context_string(self) -> str:
        """Generate natural language context about the window"""
        context_parts = [f"{self.app_name}"]
        
        if self.window_title:
            context_parts.append(f'"{self.window_title}"')
            
        if self.document_path:
            context_parts.append(f"editing {self.document_path}")
            
        if self.is_fullscreen:
            context_parts.append("in fullscreen")
        elif self.is_minimized:
            context_parts.append("minimized")
            
        return " ".join(context_parts)


class MultiSpaceWindowDetector:
    """Enhanced window detector with full multi-space awareness"""
    
    def __init__(self):
        self.last_update = 0
        self.update_interval = 0.5
        
        # Caches
        self.windows_cache: Dict[int, EnhancedWindowInfo] = {}
        self.spaces_cache: Dict[int, SpaceInfo] = {}
        self.space_windows_map: Dict[int, Set[int]] = defaultdict(set)
        
        # State tracking
        self.current_space_id: Optional[int] = None
        self.current_space_uuid: Optional[str] = None
        
        # Screenshot cache
        self.screenshot_cache: Dict[int, Tuple[Any, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
    def get_all_windows_across_spaces(self) -> Dict[str, Any]:
        """Get comprehensive window information across all spaces"""
        # Get current space info first
        self._update_current_space()

        # Get all windows with enhanced metadata
        windows = self._get_enhanced_windows()

        # Get space information
        spaces = self._get_space_info()

        # Map windows to spaces
        self._map_windows_to_spaces(windows, spaces)

        # Enhance spaces with workspace names and application info
        enhanced_spaces = []
        spaces_dict = {}  # Also keep dict format for compatibility

        for space in spaces:
            space_id = space.space_id

            # Get windows for this space
            space_window_ids = self.space_windows_map.get(space_id, set())
            space_windows = [w for w in windows if w.window_id in space_window_ids]

            # Get applications on this space
            applications = list(
                set(
                    w.app_name
                    for w in space_windows
                    if w.app_name and w.app_name != "Unknown"
                )
            )

            # Determine primary app (most windows or first)
            app_counts = {}
            for w in space_windows:
                if w.app_name:
                    app_counts[w.app_name] = app_counts.get(w.app_name, 0) + 1
            primary_app = (
                max(app_counts.keys(), key=app_counts.get) if app_counts else None
            )

            # Determine workspace name based on applications
            workspace_name = self._determine_workspace_name(
                primary_app, applications, space_windows
            )

            # Create enhanced space info dict
            space_info = {
                "space_id": space_id,
                "space_name": workspace_name,
                "primary_app": primary_app,
                "applications": applications,
                "window_count": len(space_window_ids),
                "is_current": space.is_current,
                "windows": [self._window_to_dict(w) for w in space_windows],
            }

            # Add to both list and dict
            enhanced_spaces.append(space_info)
            spaces_dict[space_id] = space_info

            logger.info(
                f"[MULTI-SPACE] Space {space_id}: '{workspace_name}' with apps {applications[:3] if applications else []}"
            )

        # Build comprehensive result with both list (for compatibility) and dict
        result = {
            "current_space": {
                "id": self.current_space_id,
                "uuid": self.current_space_uuid,
                "window_count": len(
                    self.space_windows_map.get(self.current_space_id, set())
                ),
            },
            "spaces": spaces_dict,  # Dict format for workspace name processing
            "spaces_list": enhanced_spaces,  # List format for compatibility
            "windows": windows,
            "space_window_map": {
                space_id: list(window_ids)
                for space_id, window_ids in self.space_windows_map.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Update caches
        self.windows_cache = {w.window_id: w for w in windows}
        self.last_update = time.time()

        return result

    def _determine_workspace_name(
        self, primary_app: str, applications: List[str], windows: List
    ) -> str:
        """Determine workspace name from applications and windows"""
        # Check for JARVIS first
        for w in windows:
            if (
                "jarvis" in w.window_title.lower()
                or "j.a.r.v.i.s" in w.window_title.lower()
            ):
                return "J.A.R.V.I.S. interface"

        # If Chrome is primary, it might be JARVIS
        if primary_app and "chrome" in primary_app.lower():
            return "J.A.R.V.I.S. interface"

        # Use primary app as workspace name if meaningful
        if primary_app:
            app_mappings = {
                "cursor": "Cursor",
                "code": "Code",
                "visual studio code": "Code",
                "terminal": "Terminal",
                "iterm": "Terminal",
                "messages": "Messages",
                "slack": "Slack",
                "discord": "Discord",
                "safari": "Safari",
                "finder": "Finder",
            }

            for key, value in app_mappings.items():
                if key in primary_app.lower():
                    return value

            return primary_app

        # Default fallback
        return f"Desktop {windows[0].space_id if windows and hasattr(windows[0], 'space_id') else ''}"

    def _window_to_dict(self, window: EnhancedWindowInfo) -> Dict:
        """Convert window object to dictionary"""
        return {
            "kCGWindowNumber": window.window_id,
            "kCGWindowOwnerName": window.app_name,
            "kCGWindowName": window.window_title,
            "kCGWindowOwnerPID": window.process_id,
            "kCGWindowLayer": window.layer,
            "kCGWindowAlpha": window.alpha,
        }
        
    def _get_enhanced_windows(self) -> List[EnhancedWindowInfo]:
        """Get all windows with enhanced metadata"""
        windows = []
        
        # Get window list with all options for maximum info
        window_list = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionAll | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        
        if not window_list:
            logger.warning("No windows found")
            return windows
            
        # Get the frontmost application for focus detection
        workspace = AppKit.NSWorkspace.sharedWorkspace()
        frontmost_app = workspace.frontmostApplication()
        frontmost_pid = frontmost_app.processIdentifier() if frontmost_app else None
        
        # Process each window
        for window_dict in window_list:
            window = self._process_window_dict(window_dict, frontmost_pid)
            if window:
                # Try to get additional metadata
                self._enhance_window_metadata(window)
                windows.append(window)
                
        return windows
        
    def _process_window_dict(
        self, window_dict: Dict, frontmost_pid: int
    ) -> Optional[EnhancedWindowInfo]:
        """Process raw window dictionary into EnhancedWindowInfo"""
        # Extract basic info
        window_id = window_dict.get("kCGWindowNumber", 0)
        app_name = window_dict.get("kCGWindowOwnerName", "Unknown")
        window_title = window_dict.get("kCGWindowName", "")
        process_id = window_dict.get("kCGWindowOwnerPID", 0)
        layer = window_dict.get("kCGWindowLayer", 0)
        alpha = window_dict.get("kCGWindowAlpha", 0)
        
        # Skip system UI elements
        if app_name in [
            "Window Server",
            "SystemUIServer",
            "Dock",
            "Control Center",
            "Notification Center",
            "Wallpaper",
        ]:
            return None
            
        # Skip invisible windows
        if alpha == 0:
            return None
            
        # Get window bounds
        bounds_dict = window_dict.get("kCGWindowBounds", {})
        bounds = {
            "x": int(bounds_dict.get("X", 0)),
            "y": int(bounds_dict.get("Y", 0)),
            "width": int(bounds_dict.get("Width", 0)),
            "height": int(bounds_dict.get("Height", 0)),
        }
        
        # Skip tiny windows
        if bounds["width"] < 100 or bounds["height"] < 100:
            return None
            
        # Determine visibility and state
        is_minimized = window_dict.get("kCGWindowIsOnscreen", True) == False
        is_focused = process_id == frontmost_pid and layer == 0
        
        # Check if fullscreen (heuristic based on screen size)
        screen_height = AppKit.NSScreen.mainScreen().frame().size.height
        screen_width = AppKit.NSScreen.mainScreen().frame().size.width
        is_fullscreen = (
            bounds["width"] >= screen_width * 0.95
            and bounds["height"] >= screen_height * 0.95
        )
        
        # Determine visibility
        if is_minimized:
            visibility = SpaceWindowVisibility.MINIMIZED
        elif is_fullscreen:
            visibility = SpaceWindowVisibility.FULLSCREEN
        elif window_dict.get("kCGWindowIsOnscreen", False):
            visibility = SpaceWindowVisibility.VISIBLE_CURRENT_SPACE
        else:
            visibility = SpaceWindowVisibility.VISIBLE_OTHER_SPACE
            
        return EnhancedWindowInfo(
            window_id=window_id,
            app_name=app_name,
            window_title=window_title,
            process_id=process_id,
            bounds=bounds,
            is_minimized=is_minimized,
            is_fullscreen=is_fullscreen,
            is_focused=is_focused,
            layer=layer,
            alpha=alpha,
            visibility=visibility,
        )
        
    def _enhance_window_metadata(self, window: EnhancedWindowInfo):
        """Enhance window with additional metadata"""
        # Try to extract document path from window title
        if "—" in window.window_title:
            # Common pattern: "filename — application"
            parts = window.window_title.split("—")
            potential_path = parts[0].strip()
            
            # Check if it looks like a file path
            if "/" in potential_path or "." in potential_path:
                window.document_path = potential_path
                
        # Determine workspace role based on app and position
        if window.app_name in ["Visual Studio Code", "Xcode", "IntelliJ IDEA"]:
            window.workspace_role = "main"
        elif window.app_name in ["Terminal", "iTerm2"]:
            window.workspace_role = "tool"
        elif window.app_name in ["Safari", "Chrome", "Firefox"]:
            window.workspace_role = "reference"
            
    def _update_current_space(self):
        """Update current space information"""
        # Try to get current space using various methods
        try:
            # Method 1: Use accessibility API to get space info
            # This is a simplified version - in production you'd use more robust methods
            workspace = AppKit.NSWorkspace.sharedWorkspace()
            
            # Try to determine current space (this is a placeholder)
            # In reality, you'd need to use private APIs or other methods
            if not self.current_space_id:
                self.current_space_id = 1  # Default to space 1
                
        except Exception as e:
            logger.error(f"Failed to update current space: {e}")
            self.current_space_id = 1
            
    def _get_space_info(self) -> List[SpaceInfo]:
        """Get information about all spaces"""
        spaces = []
        
        # This is a simplified implementation
        # In production, you'd use more sophisticated methods to detect spaces
        
        # For now, infer spaces from window positions and visibility
        # Windows off-screen or not visible are likely on other spaces
        estimated_spaces = self._estimate_spaces_from_windows()
        
        for space_id in estimated_spaces:
            space = SpaceInfo(
                space_id=space_id,
                space_uuid=f"space-{space_id}",
                display_id=0,  # Main display
                is_current=(space_id == self.current_space_id),
                window_count=len(self.space_windows_map.get(space_id, set())),
            )
            
            # Check cache for screenshot
            if space_id in self.screenshot_cache:
                screenshot, timestamp = self.screenshot_cache[space_id]
                if datetime.now() - timestamp < self.cache_ttl:
                    space.cached_screenshot = screenshot
                    space.screenshot_timestamp = timestamp
                    
            spaces.append(space)
            
        return spaces
        
    def _estimate_spaces_from_windows(self) -> Set[int]:
        """Estimate number of spaces from window visibility"""
        # Start with current space
        spaces = {self.current_space_id}
        
        # Try Objective-C space detection first
        try:
            from .objc_space_detector import get_objc_space_detector

            objc_detector = get_objc_space_detector()
            if objc_detector.is_available():
                objc_spaces = objc_detector.enumerate_all_spaces()

                if objc_spaces:
                    detected_spaces = set()
                    for space_info in objc_spaces:
                        space_id = space_info.get("space_id", 1)
                        detected_spaces.add(space_id)

                    spaces.update(detected_spaces)
                    logger.info(
                        f"[SPACE_DETECTION] Detected {len(detected_spaces)} spaces via Objective-C: {sorted(detected_spaces)}"
                    )
                    return spaces

        except Exception as e:
            logger.warning(f"[SPACE_DETECTION] Objective-C detection failed: {e}")

        # Fallback: Use Core Graphics API to get all windows (including off-screen)
        try:
            import Quartz

            # Get all windows using Core Graphics
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID
            )

            # Analyze window positions and visibility
            visible_windows = []
            offscreen_windows = []

            for window_info in window_list:
                window_id = window_info.get("kCGWindowNumber")
                bounds = window_info.get("kCGWindowBounds", {})
                layer = window_info.get("kCGWindowLayer", 0)
                owner_name = window_info.get("kCGWindowOwnerName", "Unknown")

                # Skip system windows and very small windows
                if layer != 0 or owner_name in ["Window Server", "Dock", "MenuMeters"]:
                    continue

                x = bounds.get("X", 0)
                y = bounds.get("Y", 0)
                width = bounds.get("Width", 0)
                height = bounds.get("Height", 0)

                # Skip tiny windows
                if width < 100 or height < 100:
                    continue

                # Determine if window is visible on current screen
                screen_bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
                screen_x = screen_bounds.origin.x
                screen_y = screen_bounds.origin.y
                screen_width = screen_bounds.size.width
                screen_height = screen_bounds.size.height

                # Check if window is on current screen
                is_on_current_screen = (
                    x >= screen_x
                    and y >= screen_y
                    and x + width <= screen_x + screen_width
                    and y + height <= screen_y + screen_height
                )

                if is_on_current_screen:
                    visible_windows.append((x, y, width, height, owner_name))
                else:
                    offscreen_windows.append((x, y, width, height, owner_name))

            # If we have offscreen windows, estimate additional spaces
            if offscreen_windows:
                # Group offscreen windows by position
                position_groups = {}
                for x, y, w, h, app in offscreen_windows:
                    # Simple grouping by x-coordinate ranges
                    if x < -1000:  # Far left
                        group = 1
                    elif x < 0:  # Left edge
                        group = 2
                    elif x > screen_width:  # Right edge
                        group = 3
                    elif x > screen_width + 1000:  # Far right
                        group = 4
                    else:
                        group = 5  # Other positions

                    if group not in position_groups:
                        position_groups[group] = []
                    position_groups[group].append((x, y, w, h, app))

                # Add spaces for each group
                for group_id in position_groups.keys():
                    spaces.add(group_id)

                logger.info(
                    f"[SPACE_DETECTION] Detected {len(spaces)} spaces via Core Graphics: {sorted(spaces)}"
                )
                logger.info(
                    f"[SPACE_DETECTION] Visible windows: {len(visible_windows)}, Offscreen windows: {len(offscreen_windows)}"
                )
                return spaces

        except Exception as e:
            logger.warning(f"[SPACE_DETECTION] Core Graphics detection failed: {e}")

        # Fallback: Use window-based detection
        try:
            # Get all windows and analyze their positions
            windows = self._get_all_windows()

            # Group windows by approximate position clusters
            position_clusters = set()
            for window in windows:
                if window.bounds and window.bounds.get("x") is not None:
                    # Simple clustering based on x-coordinate ranges
                    x_pos = window.bounds["x"]
                    if x_pos < 100:  # Left edge
                        position_clusters.add(1)
                    elif x_pos < 2000:  # Middle area
                        position_clusters.add(2)
                    else:  # Right edge
                        position_clusters.add(3)

            if position_clusters:
                spaces.update(position_clusters)
                logger.info(
                    f"[SPACE_DETECTION] Detected spaces via window positions: {sorted(spaces)}"
                )
                return spaces

        except Exception as e:
            logger.warning(f"[SPACE_DETECTION] Window-based detection failed: {e}")

        # Final fallback: Use a reasonable default based on common setups
        logger.warning("[SPACE_DETECTION] Using fallback space detection")
        return {1, 2, 3, 4, 5, 6, 7}  # Allow for more spaces

    def _map_windows_to_spaces(
        self, windows: List[EnhancedWindowInfo], spaces: List[SpaceInfo]
    ):
        """Map windows to their respective spaces"""
        # Clear existing mapping
        self.space_windows_map.clear()
        
        for window in windows:
            # Determine space based on visibility
            if window.visibility == SpaceWindowVisibility.VISIBLE_CURRENT_SPACE:
                window.space_id = self.current_space_id
                self.space_windows_map[self.current_space_id].add(window.window_id)
                
            elif window.visibility == SpaceWindowVisibility.VISIBLE_OTHER_SPACE:
                # Try to determine which other space
                # This is a heuristic - in production you'd use actual detection
                window.space_id = self._estimate_window_space(window)
                self.space_windows_map[window.space_id].add(window.window_id)
                
    def _estimate_window_space(self, window: EnhancedWindowInfo) -> int:
        """Estimate which space a window is on"""
        # Try to use AppleScript to get the actual space for this window
        try:
            import subprocess

            # Use AppleScript to find which space a window is on
            script = f"""
            tell application "System Events"
                set targetApp to "{window.app_name}"
                set targetTitle to "{window.window_title}"
                
                try
                    set appProcess to first process whose name is targetApp
                    set targetWindow to first window of appProcess whose title contains targetTitle
                    
                    -- Try to determine space (this is limited by macOS security)
                    return 1  -- Default to space 1 for now
                on error
                    return 1
                end try
            end tell
            """

            result = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=3
            )

            if result.returncode == 0 and result.stdout.strip().isdigit():
                space_id = int(result.stdout.strip())
                logger.debug(
                    f"[WINDOW_SPACE] Window '{window.window_title}' detected on space {space_id}"
                )
                return space_id

        except Exception as e:
            logger.debug(
                f"[WINDOW_SPACE] AppleScript detection failed for window '{window.window_title}': {e}"
            )

        # Fallback: Use window position and visibility to estimate space
        if window.visibility == SpaceWindowVisibility.VISIBLE_CURRENT_SPACE:
            return self.current_space_id or 1

        elif window.visibility == SpaceWindowVisibility.VISIBLE_OTHER_SPACE:
            # Use window bounds to estimate space
            if window.bounds and window.bounds.get("x") is not None:
                x_pos = window.bounds["x"]
                # Simple heuristic based on position
                if x_pos < 0:  # Off-screen left
                    return 2
                elif x_pos > 2000:  # Off-screen right
                    return 3
                else:
                    return 2  # Default to space 2 for other spaces

        # Final fallback: Distribute based on app type
        if window.app_name in ["Visual Studio Code", "Cursor", "Terminal"]:
            return 2  # Development space
        elif window.app_name in ["Slack", "Messages", "Discord"]:
            return 3  # Communication space
        elif window.app_name in ["Finder"]:
            return 1  # Default space
        else:
            return 2  # Default to space 2 for unknown apps
            
    def find_window(self, query: str) -> List[EnhancedWindowInfo]:
        """Find windows matching a query"""
        query_lower = query.lower()
        matches = []
        
        # Update if cache is stale
        if time.time() - self.last_update > self.update_interval:
            self.get_all_windows_across_spaces()
            
        for window in self.windows_cache.values():
            if (
                query_lower in window.app_name.lower()
                or query_lower in window.window_title.lower()
                or (
                    window.document_path and query_lower in window.document_path.lower()
                )
            ):
                matches.append(window)
                
        return matches
        
    def get_space_summary(self, space_id: int) -> Dict[str, Any]:
        """Get a summary of what's on a specific space"""
        # Update cache if needed
        if time.time() - self.last_update > self.update_interval:
            self.get_all_windows_across_spaces()
            
        windows_on_space = [
            self.windows_cache[wid] 
            for wid in self.space_windows_map.get(space_id, set())
            if wid in self.windows_cache
        ]
        
        # Group by application
        apps = defaultdict(list)
        for window in windows_on_space:
            apps[window.app_name].append(window)
            
        # Build summary
        summary = {
            "space_id": space_id,
            "is_current": space_id == self.current_space_id,
            "window_count": len(windows_on_space),
            "applications": {
                app: [w.to_context_string() for w in windows]
                for app, windows in apps.items()
            },
            "has_cached_screenshot": space_id in self.screenshot_cache,
        }
        
        return summary
        
    async def capture_space_with_switch(self, space_id: int) -> Optional[Any]:
        """Capture a space by switching to it (requires user permission)"""
        if space_id == self.current_space_id:
            # Already on this space, capture normally
            from PIL import ImageGrab

            screenshot = ImageGrab.grab()
            self.screenshot_cache[space_id] = (screenshot, datetime.now())
            return screenshot
            
        # Switch to space using AppleScript
        script = f"""
        tell application "System Events"
            key code 18 using {{control down}}
        end tell
        """
        
        try:
            # Execute space switch
            subprocess.run(["osascript", "-e", script], check=True)
            
            # Wait for switch animation
            await asyncio.sleep(0.5)
            
            # Capture screenshot
            from PIL import ImageGrab

            screenshot = ImageGrab.grab()
            
            # Cache it
            self.screenshot_cache[space_id] = (screenshot, datetime.now())
            
            # Switch back
            subprocess.run(["osascript", "-e", script.replace("18", "19")], check=True)
            
            return screenshot
            
        except Exception as e:
            logger.error(f"Failed to capture space {space_id}: {e}")
            return None
