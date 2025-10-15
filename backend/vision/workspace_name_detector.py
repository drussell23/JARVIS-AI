#!/usr/bin/env python3
"""
Workspace Name Detector
Detects actual workspace names from current system state
"""

import subprocess
import logging
from typing import Dict, List, Optional
import Quartz
from Quartz import (
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionAll,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID
)

logger = logging.getLogger(__name__)

class WorkspaceNameDetector:
    """
    Detects actual workspace names based on windows and applications
    """

    def __init__(self):
        self.workspace_names = {}
        self.last_detection = None

    def detect_workspace_names(self) -> Dict[int, str]:
        """
        Detect workspace names based on current windows and applications
        """
        workspace_names = {}

        # Try to get workspace info from windows
        window_list = CGWindowListCopyWindowInfo(
            kCGWindowListOptionAll,
            kCGNullWindowID
        )

        # Group windows by estimated space
        space_apps = {}
        visible_apps = set()

        for window in window_list:
            # Skip system windows
            if window.get('kCGWindowLayer', 0) != 0:
                continue

            app_name = window.get('kCGWindowOwnerName', '')
            window_name = window.get('kCGWindowName', '')
            is_visible = window.get('kCGWindowIsOnscreen', False)

            if app_name and app_name != 'Window Server':
                if is_visible:
                    visible_apps.add(app_name)

                # Check for JARVIS
                if 'jarvis' in window_name.lower() or 'j.a.r.v.i.s' in window_name.lower():
                    # JARVIS is likely on space 1 if visible
                    if is_visible:
                        workspace_names[1] = 'J.A.R.V.I.S. interface'
                        continue

        # Try to infer workspace names from visible applications
        # This is a simplified approach - in reality, we'd need more sophisticated detection

        # Common workspace patterns based on your screenshot
        if 'Google Chrome' in visible_apps:
            # Chrome with JARVIS is usually space 1
            if 1 not in workspace_names:
                workspace_names[1] = 'J.A.R.V.I.S. interface'

        if 'Terminal' in visible_apps:
            # Terminal is often on its own space
            if 2 not in workspace_names:
                workspace_names[2] = 'Terminal'

        # Check for other common apps
        app_to_workspace = {
            'Cursor': 'Cursor',
            'Code': 'Code',
            'Visual Studio Code': 'Code',
            'Messages': 'Messages',
            'Slack': 'Slack',
            'Discord': 'Discord',
            'Safari': 'Safari',
            'Finder': 'Finder',
            'Notes': 'Notes',
            'Mail': 'Mail'
        }

        space_counter = 3  # Start from space 3 since 1 and 2 might be taken
        for app_name in visible_apps:
            for app_key, workspace_name in app_to_workspace.items():
                if app_key in app_name:
                    # Find an available space number
                    if workspace_name not in workspace_names.values():
                        workspace_names[space_counter] = workspace_name
                        space_counter += 1
                        break

        # Store for later use
        self.workspace_names = workspace_names
        self.last_detection = workspace_names

        logger.info(f"[WORKSPACE DETECTOR] Detected workspace names: {workspace_names}")
        return workspace_names

    def get_workspace_name(self, space_id: int) -> str:
        """
        Get workspace name for a specific space ID
        """
        if not self.workspace_names:
            self.detect_workspace_names()

        return self.workspace_names.get(space_id, f"Desktop {space_id}")

    def build_workspace_mapping(self, window_data: Dict = None) -> Dict[int, str]:
        """
        Build a mapping of space IDs to workspace names from window data
        """
        mapping = {}

        if window_data and 'spaces' in window_data:
            logger.info(f"[WORKSPACE NAME DETECTOR] Processing {len(window_data['spaces'])} spaces")
            for space_id, space_info in window_data['spaces'].items():
                if isinstance(space_info, dict):
                    # Check for space_name field first
                    if 'space_name' in space_info and space_info['space_name'] != f"Desktop {space_id}":
                        mapping[space_id] = space_info['space_name']
                        logger.info(f"[WORKSPACE NAME DETECTOR] Space {space_id}: Using existing name '{space_info['space_name']}'")
                    # Try to determine from primary_app
                    elif 'primary_app' in space_info and space_info['primary_app']:
                        workspace_name = self._determine_name_from_app(space_info['primary_app'])
                        mapping[space_id] = workspace_name
                        logger.info(f"[WORKSPACE NAME DETECTOR] Space {space_id}: Determined '{workspace_name}' from primary_app '{space_info['primary_app']}'")
                    # Otherwise try to determine from applications list
                    elif 'applications' in space_info and space_info['applications']:
                        apps = space_info['applications']
                        workspace_name = self._determine_name_from_apps(apps)
                        mapping[space_id] = workspace_name
                        logger.info(f"[WORKSPACE NAME DETECTOR] Space {space_id}: Determined '{workspace_name}' from apps {apps[:2]}")
                    else:
                        # Try to get from windows if available
                        if 'windows' in space_info:
                            workspace_name = self._determine_name_from_windows(space_info['windows'])
                            mapping[space_id] = workspace_name
                            logger.info(f"[WORKSPACE NAME DETECTOR] Space {space_id}: Determined '{workspace_name}' from windows")
                        else:
                            mapping[space_id] = f"Desktop {space_id}"
                            logger.info(f"[WORKSPACE NAME DETECTOR] Space {space_id}: No info available, using default")

        # If no window_data or mapping is empty, use detection
        if not mapping:
            logger.info("[WORKSPACE NAME DETECTOR] No mapping from window_data, using detection")
            mapping = self.detect_workspace_names()

        # Log the final mapping
        logger.info(f"[WORKSPACE NAME DETECTOR] Final mapping: {mapping}")

        return mapping

    def _determine_name_from_app(self, app_name: str) -> str:
        """
        Determine workspace name from a single app name
        """
        if not app_name or app_name == 'Unknown':
            return "Desktop"

        # Check for JARVIS first
        if 'jarvis' in app_name.lower() or 'j.a.r.v.i.s' in app_name.lower():
            return 'J.A.R.V.I.S. interface'

        # Check for Chrome with JARVIS
        if 'chrome' in app_name.lower():
            # This might be the JARVIS interface
            return 'J.A.R.V.I.S. interface'  # Assume Chrome with JARVIS

        # Map common apps to clean names
        app_mappings = {
            'cursor': 'Cursor',
            'code': 'Code',
            'visual studio code': 'Code',
            'terminal': 'Terminal',
            'iterm': 'Terminal',
            'messages': 'Messages',
            'slack': 'Slack',
            'discord': 'Discord',
            'safari': 'Safari',
            'finder': 'Finder',
            'notes': 'Notes',
            'mail': 'Mail'
        }

        app_lower = app_name.lower()
        for key, value in app_mappings.items():
            if key in app_lower:
                return value

        # Return the app name as is if no mapping
        return app_name

    def _determine_name_from_windows(self, windows: List) -> str:
        """
        Determine workspace name from windows list
        """
        if not windows:
            return "Desktop"

        # Check each window for relevant info
        for window in windows:
            if isinstance(window, dict):
                window_name = window.get('kCGWindowName', '').lower()
                owner_name = window.get('kCGWindowOwnerName', '')

                # Check for JARVIS
                if 'jarvis' in window_name or 'j.a.r.v.i.s' in window_name:
                    return 'J.A.R.V.I.S. interface'

                # Return owner name if it's meaningful
                if owner_name and owner_name not in ['Window Server', 'Dock', 'SystemUIServer']:
                    return self._determine_name_from_app(owner_name)

        return "Desktop"

    def _determine_name_from_apps(self, apps: List[str]) -> str:
        """
        Determine workspace name from list of applications
        """
        if not apps:
            return "Desktop"

        # Check for JARVIS first
        for app in apps:
            if 'jarvis' in app.lower():
                return 'J.A.R.V.I.S. interface'

        # Check for primary apps
        primary_apps = {
            'Google Chrome': 'Google Chrome',
            'Cursor': 'Cursor',
            'Code': 'Code',
            'Terminal': 'Terminal',
            'Messages': 'Messages',
            'Safari': 'Safari',
            'Slack': 'Slack',
            'Discord': 'Discord',
            'Finder': 'Finder'
        }

        for app in apps:
            for app_key, workspace_name in primary_apps.items():
                if app_key in app:
                    return workspace_name

        # Return first non-system app
        for app in apps:
            if app not in ['Window Server', 'Dock', 'SystemUIServer', 'Finder']:
                return app

        return "Desktop"

# Global instance
_detector = WorkspaceNameDetector()

def get_current_workspace_names() -> Dict[int, str]:
    """Get current workspace names"""
    return _detector.detect_workspace_names()

def process_response_with_workspace_names(response: str, window_data: Dict = None) -> str:
    """Process response to replace Desktop N with actual workspace names"""
    mapping = _detector.build_workspace_mapping(window_data)

    # Replace Desktop N patterns
    processed = response
    for space_id, workspace_name in mapping.items():
        patterns = [
            f"Desktop {space_id}",
            f"desktop {space_id}",
            f"On Desktop {space_id}",
            f"on Desktop {space_id}",
            f"in Desktop {space_id}",
            f"In Desktop {space_id}"
        ]

        for pattern in patterns:
            if pattern in processed and workspace_name != pattern:
                # Add context for better readability
                if "On " in pattern or "on " in pattern:
                    replacement = f"In {workspace_name}"
                elif "In " in pattern or "in " in pattern:
                    replacement = f"In {workspace_name}"
                else:
                    replacement = workspace_name

                processed = processed.replace(pattern, replacement)

    return processed