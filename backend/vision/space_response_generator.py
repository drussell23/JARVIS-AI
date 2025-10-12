#!/usr/bin/env python3
"""
Space Response Generator - Natural language response generation for workspace intelligence
Transforms workspace analysis into conversational JARVIS responses
"""

import logging
import re
from typing import List, Optional, Dict, Any

from .workspace_analyzer import (
    WorkspaceAnalysis,
    SpaceSummary,
    ActivityType,
    ActivityPattern
)
from .yabai_space_detector import YabaiStatus

logger = logging.getLogger(__name__)


class SpaceResponseGenerator:
    """
    Generate natural language responses for workspace intelligence

    Features:
    - JARVIS voice and personality
    - Adaptive detail level based on context
    - Intelligent summarization
    - Contextual insights and suggestions
    """

    # JARVIS personality phrases
    GREETINGS = [
        "Sir, let me give you an overview",
        "Sir, here's what's happening",
        "Allow me to report",
        "Sir, I've analyzed your workspace"
    ]

    TRANSITION_PHRASES = [
        "In addition",
        "Additionally",
        "I also notice",
        "Furthermore",
        "It appears"
    ]

    ACTIVITY_DESCRIPTIONS = {
        ActivityType.CODING: [
            "working on code",
            "developing",
            "coding",
            "programming"
        ],
        ActivityType.BROWSING: [
            "browsing the web",
            "researching online",
            "web browsing"
        ],
        ActivityType.COMMUNICATION: [
            "communicating",
            "in conversation",
            "messaging"
        ],
        ActivityType.DESIGN: [
            "designing",
            "working on design",
            "creating visuals"
        ],
        ActivityType.TERMINAL: [
            "working in the terminal",
            "running commands",
            "terminal session active"
        ],
        ActivityType.DOCUMENTATION: [
            "writing documentation",
            "taking notes",
            "documenting"
        ],
        ActivityType.MEDIA: [
            "consuming media",
            "listening to music",
            "watching content"
        ],
        ActivityType.PRODUCTIVITY: [
            "working on productivity tasks",
            "organizing",
            "planning"
        ]
    }

    def __init__(self, use_sir_prefix: bool = True):
        """
        Initialize response generator

        Args:
            use_sir_prefix: Whether to use "Sir" in responses
        """
        self.use_sir_prefix = use_sir_prefix

    def generate_overview_response(
        self,
        analysis: WorkspaceAnalysis,
        include_details: bool = True
    ) -> str:
        """
        Generate complete workspace overview response

        Args:
            analysis: Workspace analysis
            include_details: Include per-space details

        Returns:
            Natural language response
        """
        logger.info("[RESPONSE] Generating workspace overview")

        parts = []

        # Opening
        parts.append(self._generate_opening(analysis))

        # High-level summary
        parts.append(self._generate_summary(analysis))

        # Space-by-space breakdown (if requested and meaningful)
        if include_details and analysis.active_spaces > 0:
            space_details = self._generate_space_details(analysis)
            if space_details:
                parts.append(space_details)

        # Insights and patterns
        insights = self._generate_insights(analysis)
        if insights:
            parts.append(insights)

        # Combine all parts
        response = "\n\n".join(filter(None, parts))

        logger.info(f"[RESPONSE] ✅ Generated {len(response)} character response")
        return response

    def generate_space_specific_response(
        self,
        analysis: WorkspaceAnalysis,
        space_id: int
    ) -> str:
        """
        Generate response for a specific space

        Args:
            analysis: Workspace analysis
            space_id: Space ID to describe

        Returns:
            Natural language response
        """
        space = analysis.get_space_by_id(space_id)

        if not space:
            return f"Sir, I don't have information about Space {space_id}."

        if not space.is_active:
            return f"Sir, {space.space_name} appears to be empty at the moment."

        parts = []

        # Opening
        parts.append(f"Sir, here's what's happening in {space.space_name}:")

        # Applications
        if space.applications:
            if len(space.applications) == 1:
                parts.append(f"You have {space.applications[0]} open.")
            else:
                apps_list = self._format_list(space.applications[:3])
                others = len(space.applications) - 3
                if others > 0:
                    apps_list += f" and {others} other{'s' if others > 1 else ''}"
                parts.append(f"You have {apps_list} active.")

        # Activity
        activity_desc = self._describe_activity(space.primary_activity)
        if activity_desc:
            parts.append(activity_desc)

        # Window titles (if meaningful)
        meaningful_titles = [
            t for t in space.window_titles
            if t and len(t) > 5 and not t.startswith('Window')
        ][:3]

        if meaningful_titles:
            parts.append("Working on: " + ", ".join(f'"{t}"' for t in meaningful_titles))

        return "\n".join(parts)

    def generate_yabai_installation_response(self, status: YabaiStatus) -> str:
        """
        Generate response for Yabai installation/setup

        Args:
            status: Current Yabai status

        Returns:
            Installation guidance response
        """
        if status == YabaiStatus.NOT_INSTALLED:
            return """Sir, for enhanced multi-space intelligence, I recommend installing Yabai.

**Installation via Homebrew:**
```
brew install koekeishiya/formulae/yabai
```

After installation, I'll be able to provide detailed workspace analysis across all your desktop spaces."""

        elif status == YabaiStatus.NO_PERMISSIONS:
            return """Sir, Yabai requires Accessibility permissions to function properly.

**To grant permissions:**
1. Open System Preferences → Security & Privacy → Privacy
2. Click "Accessibility" in the left sidebar
3. Click the lock icon and authenticate
4. Add Yabai to the list and check the box

Once permissions are granted, I'll have full workspace visibility."""

        elif status == YabaiStatus.ERROR:
            return """Sir, I'm experiencing difficulty accessing Yabai at the moment.

I'll continue using basic workspace detection. If you'd like enhanced multi-space intelligence, please ensure Yabai is properly installed and configured."""

        return "Sir, Yabai integration is operational."

    def _generate_opening(self, analysis: WorkspaceAnalysis) -> str:
        """Generate opening statement"""
        import random

        greeting = random.choice(self.GREETINGS)

        if analysis.active_spaces == 0:
            return f"{greeting}, your workspace appears to be clear at the moment."

        # Check if working on a specific project
        if analysis.detected_project:
            return f"{greeting} of your work on {analysis.detected_project}."

        return f"{greeting} of your workspace activity across your {analysis.total_spaces} desktop spaces:"

    def _generate_summary(self, analysis: WorkspaceAnalysis) -> str:
        """Generate high-level summary"""
        if analysis.active_spaces == 0:
            return "All spaces are currently empty."

        parts = []

        # Basic stats
        if analysis.active_spaces == 1:
            parts.append(f"You're currently active in 1 space")
        else:
            parts.append(f"You're active across {analysis.active_spaces} spaces")

        # Application count
        if analysis.unique_applications == 1:
            parts.append("with 1 application running")
        else:
            parts.append(f"with {analysis.unique_applications} applications running")

        # Overall activity
        if analysis.overall_activity and analysis.overall_activity.confidence > 0.6:
            activity_desc = self._get_activity_description(analysis.overall_activity.activity_type)
            parts.append(f"primarily focused on {activity_desc}")

        return " ".join(parts) + "."

    def _generate_space_details(self, analysis: WorkspaceAnalysis) -> str:
        """Generate per-space details"""
        details = []

        for space in sorted(analysis.space_summaries, key=lambda s: s.space_id):
            if not space.is_active:
                continue  # Skip empty spaces

            # Format space header
            space_header = f"**{space.space_name}**"
            if space.is_current:
                space_header += " (Current)"
            space_header += ":"

            # Describe activity
            if space.window_count == 1:
                activity = f"{space.applications[0]}"
            elif space.applications:
                activity = f"{self._format_list(space.applications[:2])}"
                if len(space.applications) > 2:
                    activity += f" and {len(space.applications) - 2} other app{'s' if len(space.applications) > 2 else ''}"
            else:
                continue  # Skip if no meaningful data

            # Add context from activity detection
            context = self._get_space_context(space)
            if context:
                activity += f" - {context}"

            details.append(f"{space_header} {activity}")

        if not details:
            return ""

        return "\n".join(details)

    def _generate_insights(self, analysis: WorkspaceAnalysis) -> str:
        """Generate insights and observations"""
        if analysis.active_spaces == 0:
            return ""

        insights = []

        # Project detection
        if analysis.detected_project:
            insights.append(
                f"You appear to be working on the {analysis.detected_project} project."
            )

        # Focus pattern
        if analysis.focus_pattern:
            insights.append(analysis.focus_pattern + ".")

        # Workspace organization observation
        if analysis.active_spaces >= 4 and analysis.unique_applications >= 6:
            insights.append(
                "You have a well-organized multi-space workflow with different contexts separated."
            )

        if not insights:
            return ""

        return "\n".join(insights)

    def _describe_activity(self, activity: ActivityPattern) -> str:
        """Describe an activity pattern"""
        if activity.confidence < 0.3:
            return ""

        desc = self._get_activity_description(activity.activity_type)

        if activity.project_name:
            return f"You're {desc} on {activity.project_name}."
        elif activity.primary_app:
            return f"You're {desc} in {activity.primary_app}."
        else:
            return f"You're {desc}."

    def _get_activity_description(self, activity_type: ActivityType) -> str:
        """Get description for activity type"""
        import random

        descriptions = self.ACTIVITY_DESCRIPTIONS.get(
            activity_type,
            ["working"]
        )
        return random.choice(descriptions)

    def _get_space_context(self, space: SpaceSummary) -> str:
        """Get contextual description for a space"""
        activity = space.primary_activity

        if activity.confidence < 0.4:
            return ""

        # Try to extract meaningful context from window titles
        if space.window_titles:
            # Look for file names
            for title in space.window_titles[:3]:
                # Extract filename
                filename_match = re.search(r'([^\\/]+\.[a-z]{2,4})', title, re.IGNORECASE)
                if filename_match:
                    return f"working on {filename_match.group(1)}"

                # Extract project hints
                if '/' in title or '\\' in title:
                    path_parts = re.split(r'[\\/]', title)
                    meaningful_parts = [
                        p for p in path_parts
                        if p and p not in {'Users', 'Documents', 'Desktop', 'home'}
                    ]
                    if meaningful_parts:
                        return f"in {meaningful_parts[-1]}"

        # Fallback to activity description
        return self._get_activity_description(activity.activity_type)

    def _format_list(self, items: List[str], max_items: int = 3) -> str:
        """Format list with proper grammar"""
        items = items[:max_items]

        if not items:
            return ""
        elif len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return ", ".join(items[:-1]) + f", and {items[-1]}"

    def generate_app_location_response(
        self,
        analysis: WorkspaceAnalysis,
        app_name: str
    ) -> str:
        """
        Generate response about where an app is located

        Args:
            analysis: Workspace analysis
            app_name: App to locate

        Returns:
            Natural language response
        """
        found_spaces = []

        for space in analysis.space_summaries:
            if app_name.lower() in [a.lower() for a in space.applications]:
                found_spaces.append(space)

        if not found_spaces:
            return f"Sir, I don't see {app_name} running in any of your spaces at the moment."

        if len(found_spaces) == 1:
            space = found_spaces[0]
            windows_with_app = [
                w for w in space.window_titles
                if app_name.lower() in w.lower()
            ]

            response = f"Sir, {app_name} is running in {space.space_name}."

            if windows_with_app:
                response += "\n\nOpen windows:"
                for title in windows_with_app[:3]:
                    response += f"\n  • {title}"

            return response
        else:
            spaces_list = self._format_list([s.space_name for s in found_spaces])
            return f"Sir, {app_name} is running in multiple spaces: {spaces_list}."
