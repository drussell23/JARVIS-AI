#!/usr/bin/env python3
"""
Workspace Name Processor
Dynamically replaces generic "Desktop N" references with actual workspace names
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class WorkspaceNameProcessor:
    """
    Processes responses to replace generic desktop references with actual workspace names
    """

    def __init__(self):
        self.workspace_mapping = {}
        self.last_detected_names = {}

    def update_workspace_names(self, spaces_data: Dict) -> None:
        """
        Update the mapping of space IDs to actual workspace names
        """
        self.workspace_mapping = {}

        for space_id, space_info in spaces_data.items():
            if isinstance(space_info, dict):
                # Try to get the workspace name from various sources
                workspace_name = self._determine_workspace_name(space_id, space_info)
                self.workspace_mapping[space_id] = workspace_name
                self.last_detected_names[space_id] = workspace_name

    def _determine_workspace_name(self, space_id: int, space_info: Dict) -> str:
        """
        Determine the actual workspace name from space information
        """
        # First check if space_name is directly provided
        if 'space_name' in space_info:
            return space_info['space_name']

        # Check primary app
        primary_app = space_info.get('primary_app', '')
        if primary_app:
            # Special cases
            if 'jarvis' in primary_app.lower():
                return 'J.A.R.V.I.S. interface'

            # Common app mappings
            app_names = {
                'Google Chrome': 'Google Chrome',
                'Safari': 'Safari',
                'Code': 'Code',
                'Cursor': 'Cursor',
                'Terminal': 'Terminal',
                'iTerm2': 'iTerm',
                'Finder': 'Finder',
                'Slack': 'Slack',
                'Discord': 'Discord',
                'Messages': 'Messages'
            }

            for app_key, app_name in app_names.items():
                if app_key.lower() in primary_app.lower():
                    return app_name

            return primary_app

        # Check applications list
        apps = space_info.get('applications', [])
        if apps:
            # Check for JARVIS first
            for app in apps:
                if 'jarvis' in app.lower():
                    return 'J.A.R.V.I.S. interface'

            # Return first meaningful app
            for app in apps:
                if app and app != 'Unknown':
                    return app

        # Check windows for context
        windows = space_info.get('windows', [])
        for window in windows:
            if isinstance(window, dict):
                window_name = window.get('kCGWindowName', '').lower()
                owner_name = window.get('kCGWindowOwnerName', '')

                if 'jarvis' in window_name or 'j.a.r.v.i.s' in window_name:
                    return 'J.A.R.V.I.S. interface'
                elif owner_name:
                    return owner_name

        # Default fallback
        return f"Desktop {space_id}"

    def process_response(self, response: str, spaces_data: Dict = None) -> str:
        """
        Process a response to replace generic desktop references with workspace names
        """
        if spaces_data:
            self.update_workspace_names(spaces_data)

        # If no mapping available, return original
        if not self.workspace_mapping and not self.last_detected_names:
            return response

        # Use last detected names if no new data provided
        mapping = self.workspace_mapping if self.workspace_mapping else self.last_detected_names

        # Replace patterns like "Desktop 1", "Desktop 2", etc.
        processed = response

        # Sort by desktop number in reverse to avoid replacing "Desktop 1" in "Desktop 10"
        desktop_patterns = []
        for space_id in sorted(mapping.keys(), reverse=True):
            if isinstance(space_id, int):
                desktop_patterns.append((f"Desktop {space_id}", mapping[space_id]))
                desktop_patterns.append((f"desktop {space_id}", mapping[space_id]))
                desktop_patterns.append((f"Desktop{space_id}", mapping[space_id]))
                desktop_patterns.append((f"desktop{space_id}", mapping[space_id]))

        # Apply replacements
        for pattern, replacement in desktop_patterns:
            # Don't replace if the replacement is the same pattern
            if pattern.lower() != replacement.lower():
                processed = processed.replace(pattern, replacement)

        # Also replace "On Desktop N" patterns
        for space_id, name in mapping.items():
            if isinstance(space_id, int):
                patterns = [
                    f"On Desktop {space_id}",
                    f"on Desktop {space_id}",
                    f"in Desktop {space_id}",
                    f"at Desktop {space_id}"
                ]
                for pattern in patterns:
                    if pattern.lower() not in name.lower():
                        processed = processed.replace(pattern, f"in {name}")

        return processed

    def get_workspace_name(self, space_id: int) -> str:
        """
        Get the workspace name for a specific space ID
        """
        if space_id in self.workspace_mapping:
            return self.workspace_mapping[space_id]
        elif space_id in self.last_detected_names:
            return self.last_detected_names[space_id]
        else:
            return f"Desktop {space_id}"

# Global instance for easy access
_processor = WorkspaceNameProcessor()

def process_jarvis_response(response: str, spaces_data: Dict = None) -> str:
    """
    Global function to process JARVIS responses
    """
    return _processor.process_response(response, spaces_data)

def update_workspace_names(spaces_data: Dict) -> None:
    """
    Global function to update workspace names
    """
    _processor.update_workspace_names(spaces_data)

def get_workspace_name(space_id: int) -> str:
    """
    Global function to get a workspace name
    """
    return _processor.get_workspace_name(space_id)