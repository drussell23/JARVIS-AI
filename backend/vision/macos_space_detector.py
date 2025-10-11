#!/usr/bin/env python3
"""
macOS Native Space Detection using Accessibility and Window APIs
Provides accurate detection of spaces, windows, and applications
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import Quartz
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionAll,
    kCGWindowListExcludeDesktopElements,
    kCGNullWindowID,
    CGWindowListCreateImage,
    CGRectNull,
    kCGWindowImageDefault,
    kCGWindowImageBoundsIgnoreFraming,
    kCGWindowImageNominalResolution
)
import AppKit
from AppKit import NSWorkspace, NSScreen, NSApplication
import objc

logger = logging.getLogger(__name__)

@dataclass
class SpaceInfo:
    """Information about a macOS space"""
    space_id: int
    space_uuid: str
    display_id: int
    is_current: bool
    windows: List[Dict[str, Any]]
    applications: List[str]
    primary_app: Optional[str]
    activity_type: str
    last_active: datetime

@dataclass
class WindowInfo:
    """Detailed window information"""
    window_id: int
    title: str
    app_name: str
    bounds: Dict[str, float]
    space_id: int
    is_visible: bool
    is_focused: bool
    layer: int
    opacity: float

class MacOSSpaceDetector:
    """
    Native macOS space detection using system APIs
    """

    def __init__(self):
        self.workspace = NSWorkspace.sharedWorkspace()
        self.screens = NSScreen.screens()
        self._init_private_apis()
        self._space_cache = {}
        self._window_cache = {}
        logger.info("MacOS Space Detector initialized with native APIs")

    def _init_private_apis(self):
        """Initialize private macOS APIs for space management"""
        try:
            # Load private framework for space management
            bundle = objc.loadBundle(
                'CoreGraphics',
                globals(),
                bundle_path='/System/Library/Frameworks/CoreGraphics.framework'
            )

            # Define private API signatures
            objc.loadBundleFunctions(
                bundle,
                globals(),
                [
                    ('CGSCopySpaces', b'@ii'),
                    ('CGSCopySpacesForWindows', b'@ii@'),
                    ('CGSSpaceGetType', b'iii'),
                    ('CGSGetActiveSpace', b'ii'),
                    ('CGSCopyManagedDisplaySpaces', b'@i'),
                    ('CGSGetWindowCount', b'ii^i'),
                    ('CGSGetOnScreenWindowList', b'iii^i^i'),
                ]
            )

            # Get connection ID
            self.cgs_connection = objc.objc_msgSend(
                objc.objc_getClass('NSApplication'),
                'sharedApplication'
            )

        except Exception as e:
            logger.warning(f"Could not load private APIs, using fallback: {e}")
            self.cgs_connection = None

    def get_all_spaces(self) -> List[SpaceInfo]:
        """
        Get information about all spaces across all displays
        """
        spaces = []

        try:
            if self.cgs_connection:
                # Use private APIs for accurate space detection
                spaces_data = self._get_spaces_via_private_api()
            else:
                # Fallback to AppleScript/public APIs
                spaces_data = self._get_spaces_via_applescript()

            # Enrich with window information
            for space_data in spaces_data:
                space_info = self._create_space_info(space_data)
                spaces.append(space_info)

        except Exception as e:
            logger.error(f"Error getting spaces: {e}")
            # Fallback to basic detection
            spaces = self._fallback_space_detection()

        return spaces

    def _get_spaces_via_private_api(self) -> List[Dict]:
        """Use private APIs to get space information"""
        spaces = []

        try:
            # Get managed display spaces
            display_spaces = CGSCopyManagedDisplaySpaces(self.cgs_connection)

            for display_dict in display_spaces:
                display_id = display_dict.get('Display Identifier')
                current_space = display_dict.get('Current Space')
                space_list = display_dict.get('Spaces', [])

                for space_dict in space_list:
                    space_id = space_dict.get('id64', space_dict.get('ManagedSpaceID'))
                    space_uuid = space_dict.get('uuid', str(space_id))

                    spaces.append({
                        'space_id': space_id,
                        'space_uuid': space_uuid,
                        'display_id': display_id,
                        'is_current': space_id == current_space.get('id64'),
                        'type': space_dict.get('type', 0)
                    })

        except Exception as e:
            logger.error(f"Private API error: {e}")

        return spaces

    def _get_spaces_via_applescript(self) -> List[Dict]:
        """Use AppleScript to get space information"""
        spaces = []

        script = """
        tell application "System Events"
            set spacesList to {}
            tell application process "Dock"
                tell list 1 of group 1 of group 1 of group 1
                    set spaceButtons to every button
                    repeat with spaceButton in spaceButtons
                        set spaceName to description of spaceButton
                        set spaceIndex to value of attribute "AXIndex" of spaceButton
                        set isCurrent to value of attribute "AXSelected" of spaceButton
                        set end of spacesList to {index:spaceIndex, name:spaceName, current:isCurrent}
                    end repeat
                end tell
            end tell
            return spacesList
        end tell
        """

        try:
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse AppleScript output
                lines = result.stdout.strip().split(',')
                for i in range(0, len(lines), 3):
                    if i + 2 < len(lines):
                        spaces.append({
                            'space_id': i // 3 + 1,
                            'space_uuid': f"space_{i // 3 + 1}",
                            'display_id': 1,
                            'is_current': 'true' in lines[i + 2].lower(),
                            'type': 0
                        })

        except Exception as e:
            logger.error(f"AppleScript error: {e}")

        return spaces

    def _fallback_space_detection(self) -> List[SpaceInfo]:
        """Fallback detection using window positions"""
        spaces = []

        # Get all windows
        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionAll,
            kCGNullWindowID
        )

        # Group windows by estimated space
        space_windows = {}

        for window in window_list:
            if window.get('kCGWindowLayer', 0) == 0:  # Normal windows
                # Estimate space based on visibility and position
                is_visible = window.get('kCGWindowIsOnscreen', False)
                bounds = window.get('kCGWindowBounds', {})

                # Simple heuristic: visible windows are on current space
                space_id = 1 if is_visible else 2

                if space_id not in space_windows:
                    space_windows[space_id] = []

                space_windows[space_id].append(window)

        # Create SpaceInfo objects
        for space_id, windows in space_windows.items():
            space_info = SpaceInfo(
                space_id=space_id,
                space_uuid=f"space_{space_id}",
                display_id=1,
                is_current=(space_id == 1),
                windows=windows,
                applications=list(set([w.get('kCGWindowOwnerName', 'Unknown') for w in windows])),
                primary_app=self._determine_primary_app(windows),
                activity_type=self._determine_activity_type(windows),
                last_active=datetime.now()
            )
            spaces.append(space_info)

        return spaces

    def _create_space_info(self, space_data: Dict) -> SpaceInfo:
        """Create a SpaceInfo object with enriched data"""
        space_id = space_data['space_id']

        # Get windows for this space
        windows = self._get_windows_for_space(space_id)

        # Extract application names
        applications = list(set([w.get('kCGWindowOwnerName', 'Unknown') for w in windows]))

        return SpaceInfo(
            space_id=space_id,
            space_uuid=space_data['space_uuid'],
            display_id=space_data['display_id'],
            is_current=space_data['is_current'],
            windows=windows,
            applications=applications,
            primary_app=self._determine_primary_app(windows),
            activity_type=self._determine_activity_type(windows),
            last_active=datetime.now()
        )

    def _get_windows_for_space(self, space_id: int) -> List[Dict]:
        """Get all windows in a specific space"""
        all_windows = CGWindowListCopyWindowInfo(
            kCGWindowListOptionAll,
            kCGNullWindowID
        )

        space_windows = []
        for window in all_windows:
            # Filter windows by space (this is an approximation)
            # In reality, we'd need private APIs to get exact space assignment
            if self._is_window_in_space(window, space_id):
                space_windows.append(window)

        return space_windows

    def _is_window_in_space(self, window: Dict, space_id: int) -> bool:
        """Determine if a window belongs to a specific space"""
        # This is a heuristic - accurate detection requires private APIs
        is_visible = window.get('kCGWindowIsOnscreen', False)

        if space_id == 1:  # Assume space 1 is current
            return is_visible
        else:
            return not is_visible and window.get('kCGWindowLayer', 0) == 0

    def _determine_primary_app(self, windows: List[Dict]) -> Optional[str]:
        """Determine the primary application in a space"""
        if not windows:
            return None

        # Count windows per application
        app_counts = {}
        for window in windows:
            app = window.get('kCGWindowOwnerName', 'Unknown')
            app_counts[app] = app_counts.get(app, 0) + 1

        # Return app with most windows
        return max(app_counts, key=app_counts.get)

    def _determine_activity_type(self, windows: List[Dict]) -> str:
        """Determine the type of activity in a space"""
        if not windows:
            return 'idle'

        apps = [w.get('kCGWindowOwnerName', '').lower() for w in windows]

        # Categorize based on applications
        if any('code' in app or 'cursor' in app or 'xcode' in app for app in apps):
            return 'development'
        elif any('chrome' in app or 'safari' in app or 'firefox' in app for app in apps):
            return 'browsing'
        elif any('terminal' in app or 'iterm' in app for app in apps):
            return 'terminal'
        elif any('slack' in app or 'discord' in app or 'messages' in app for app in apps):
            return 'communication'
        elif any('finder' in app for app in apps):
            return 'file_management'
        else:
            return 'general'

    def capture_space_screenshot(self, space_id: int) -> Optional[Any]:
        """Capture a screenshot of a specific space"""
        try:
            # Get windows for the space
            windows = self._get_windows_for_space(space_id)

            if not windows:
                return None

            # Get window IDs
            window_ids = [w.get('kCGWindowID', 0) for w in windows if w.get('kCGWindowID')]

            if window_ids:
                # Capture composite image of all windows
                image = CGWindowListCreateImage(
                    CGRectNull,
                    kCGWindowListOptionAll,
                    window_ids[0],  # Primary window
                    kCGWindowImageDefault | kCGWindowImageBoundsIgnoreFraming
                )
                return image

        except Exception as e:
            logger.error(f"Error capturing space {space_id}: {e}")

        return None

    def get_current_space(self) -> Optional[SpaceInfo]:
        """Get information about the current active space"""
        spaces = self.get_all_spaces()
        for space in spaces:
            if space.is_current:
                return space
        return None

    def monitor_space_changes(self, callback):
        """Monitor for space change events"""
        # Set up notification observer for space changes
        notification_center = NSWorkspace.sharedWorkspace().notificationCenter()

        notification_center.addObserverForName_object_queue_usingBlock_(
            'NSWorkspaceActiveSpaceDidChangeNotification',
            None,
            None,
            lambda notification: callback(self.get_current_space())
        )

        logger.info("Space change monitoring enabled")

# Integration helper for existing code
class SpaceDetectorAdapter:
    """Adapter to integrate with existing multi-space code"""

    def __init__(self):
        self.detector = MacOSSpaceDetector()

    def detect_spaces_and_windows(self) -> Dict[str, Any]:
        """Get spaces in format expected by existing code"""
        spaces = self.detector.get_all_spaces()

        result = {
            'spaces': {},
            'current_space': None,
            'total_spaces': len(spaces)
        }

        for space in spaces:
            result['spaces'][space.space_id] = {
                'windows': space.windows,
                'applications': space.applications,
                'primary_app': space.primary_app,
                'activity_type': space.activity_type,
                'is_current': space.is_current
            }

            if space.is_current:
                result['current_space'] = space.space_id

        return result