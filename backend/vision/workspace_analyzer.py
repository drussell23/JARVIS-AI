#!/usr/bin/env python3
"""
Workspace Analyzer - Intelligent activity detection and pattern recognition
Analyzes workspace data to understand what the user is working on
"""

import re
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any
from enum import Enum

from .yabai_space_detector import SpaceInfo, WindowInfo

logger = logging.getLogger(__name__)


class ActivityType(Enum):
    """Types of workspace activities"""
    CODING = "coding"
    BROWSING = "browsing"
    COMMUNICATION = "communication"
    DESIGN = "design"
    TERMINAL = "terminal"
    DOCUMENTATION = "documentation"
    MEDIA = "media"
    PRODUCTIVITY = "productivity"
    GAMING = "gaming"
    SYSTEM = "system"
    UNKNOWN = "unknown"


@dataclass
class ActivityPattern:
    """Detected activity pattern"""
    activity_type: ActivityType
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    primary_app: Optional[str] = None
    project_name: Optional[str] = None

    @property
    def description(self) -> str:
        """Human-readable activity description"""
        if self.project_name:
            return f"{self.activity_type.value} on {self.project_name}"
        elif self.primary_app:
            return f"{self.activity_type.value} with {self.primary_app}"
        else:
            return self.activity_type.value


@dataclass
class SpaceSummary:
    """Summary of a single space's activity"""
    space_id: int
    space_name: str
    window_count: int
    applications: List[str]
    primary_activity: ActivityPattern
    window_titles: List[str]
    is_active: bool = False
    is_current: bool = False
    display: int = 1

    @property
    def activity_description(self) -> str:
        """Get concise activity description"""
        if self.window_count == 0:
            return "Empty"
        return self.primary_activity.description


@dataclass
class WorkspaceAnalysis:
    """Complete workspace analysis"""
    total_spaces: int
    active_spaces: int
    total_windows: int
    unique_applications: int
    space_summaries: List[SpaceSummary]
    detected_project: Optional[str] = None
    overall_activity: Optional[ActivityPattern] = None
    focus_pattern: Optional[str] = None

    def get_space_by_id(self, space_id: int) -> Optional[SpaceSummary]:
        """Get space summary by ID"""
        for summary in self.space_summaries:
            if summary.space_id == space_id:
                return summary
        return None


class WorkspaceAnalyzer:
    """
    Intelligent workspace activity analyzer

    Features:
    - Dynamic activity detection (no hardcoding)
    - Project name extraction from file paths and titles
    - Pattern recognition across spaces
    - Focus detection (deep work vs context switching)
    - Multi-display awareness
    """

    # Activity detection rules (extensible, not hardcoded)
    ACTIVITY_RULES = {
        ActivityType.CODING: {
            'apps': {
                'Visual Studio Code', 'Code', 'VSCode', 'Cursor', 'Xcode',
                'IntelliJ IDEA', 'PyCharm', 'WebStorm', 'Sublime Text',
                'Atom', 'Vim', 'Emacs', 'Android Studio', 'Eclipse'
            },
            'title_patterns': [
                r'\.py\b', r'\.js\b', r'\.ts\b', r'\.go\b', r'\.rs\b',
                r'\.java\b', r'\.cpp\b', r'\.c\b', r'\.swift\b', r'\.kt\b',
                r'\.rb\b', r'\.php\b', r'\.jsx\b', r'\.tsx\b', r'\.vue\b',
                r'\.html\b', r'\.css\b', r'\.scss\b', r'\.sql\b'
            ],
            'weight': 2.0
        },
        ActivityType.TERMINAL: {
            'apps': {
                'Terminal', 'iTerm2', 'iTerm', 'Alacritty', 'Kitty',
                'Hyper', 'Warp', 'WezTerm'
            },
            'title_patterns': [r'bash', r'zsh', r'fish', r'ssh', r'vim', r'nano'],
            'weight': 1.5
        },
        ActivityType.BROWSING: {
            'apps': {
                'Safari', 'Google Chrome', 'Chrome', 'Firefox', 'Brave Browser',
                'Microsoft Edge', 'Opera', 'Arc', 'Vivaldi'
            },
            'title_patterns': [r'http', r'www\.', r'\.com', r'\.org', r'\.io'],
            'weight': 1.0
        },
        ActivityType.COMMUNICATION: {
            'apps': {
                'Slack', 'Discord', 'Microsoft Teams', 'Zoom', 'Messages',
                'Mail', 'Outlook', 'Telegram', 'WhatsApp', 'Signal',
                'Skype', 'FaceTime', 'Gmail'
            },
            'title_patterns': [r'@', r'Direct Message', r'Meeting', r'Inbox'],
            'weight': 1.5
        },
        ActivityType.DESIGN: {
            'apps': {
                'Figma', 'Sketch', 'Adobe Photoshop', 'Adobe Illustrator',
                'Adobe XD', 'Affinity Designer', 'Blender', 'Maya',
                'Cinema 4D', 'Canva'
            },
            'title_patterns': [r'\.fig\b', r'\.sketch\b', r'\.psd\b', r'\.ai\b'],
            'weight': 2.0
        },
        ActivityType.DOCUMENTATION: {
            'apps': {
                'Notion', 'Obsidian', 'Bear', 'Notes', 'OneNote',
                'Evernote', 'Google Docs', 'Microsoft Word', 'Pages',
                'Typora', 'MarkText'
            },
            'title_patterns': [r'\.md\b', r'\.txt\b', r'\.doc\b', r'README', r'notes'],
            'weight': 1.5
        },
        ActivityType.MEDIA: {
            'apps': {
                'Spotify', 'Apple Music', 'VLC', 'QuickTime Player',
                'Final Cut Pro', 'Adobe Premiere', 'iMovie', 'YouTube Music',
                'iTunes', 'Music'
            },
            'title_patterns': [r'\.mp4\b', r'\.mov\b', r'\.mp3\b', r'Playing', r'Paused'],
            'weight': 0.5
        },
        ActivityType.PRODUCTIVITY: {
            'apps': {
                'Excel', 'Microsoft Excel', 'Numbers', 'Google Sheets',
                'PowerPoint', 'Keynote', 'Calendar', 'Reminders',
                'Things', 'Todoist', 'OmniFocus'
            },
            'title_patterns': [r'\.xlsx\b', r'\.csv\b', r'\.pptx\b', r'Calendar'],
            'weight': 1.5
        }
    }

    def __init__(self):
        """Initialize workspace analyzer"""
        pass

    def analyze(
        self,
        spaces: List[SpaceInfo],
        windows: List[WindowInfo]
    ) -> WorkspaceAnalysis:
        """
        Analyze complete workspace

        Args:
            spaces: List of space information
            windows: List of window information

        Returns:
            Complete workspace analysis
        """
        logger.info(f"[ANALYZER] Analyzing {len(spaces)} spaces with {len(windows)} windows")

        # Build space summaries
        space_summaries = []
        all_apps = set()

        for space in spaces:
            # Get windows in this space
            space_windows = [w for w in windows if w.space_index == space.index]

            # Extract app names
            apps = [w.app_name for w in space_windows if w.app_name != 'Unknown']
            all_apps.update(apps)

            # Detect primary activity
            primary_activity = self._detect_activity(space_windows)

            # Create summary
            summary = SpaceSummary(
                space_id=space.index,
                space_name=space.display_name,
                window_count=len(space_windows),
                applications=apps,
                primary_activity=primary_activity,
                window_titles=[w.title for w in space_windows if w.title],
                is_active=len(space_windows) > 0,
                is_current=space.is_focused,
                display=space.display
            )

            space_summaries.append(summary)

        # Detect overall project
        detected_project = self._detect_project(windows)

        # Detect overall activity pattern
        overall_activity = self._detect_overall_activity(space_summaries)

        # Analyze focus pattern
        focus_pattern = self._analyze_focus_pattern(space_summaries)

        analysis = WorkspaceAnalysis(
            total_spaces=len(spaces),
            active_spaces=len([s for s in space_summaries if s.is_active]),
            total_windows=len(windows),
            unique_applications=len(all_apps),
            space_summaries=space_summaries,
            detected_project=detected_project,
            overall_activity=overall_activity,
            focus_pattern=focus_pattern
        )

        logger.info(
            f"[ANALYZER] ✅ Analysis complete: "
            f"{analysis.active_spaces}/{analysis.total_spaces} active spaces, "
            f"{analysis.unique_applications} unique apps, "
            f"project: {detected_project or 'None'}"
        )

        return analysis

    def _detect_activity(self, windows: List[WindowInfo]) -> ActivityPattern:
        """
        Detect primary activity from windows

        Args:
            windows: Windows in the space

        Returns:
            Detected activity pattern
        """
        if not windows:
            return ActivityPattern(
                activity_type=ActivityType.UNKNOWN,
                confidence=1.0,
                evidence=["No windows"],
                primary_app=None
            )

        # Score each activity type
        scores = defaultdict(float)
        evidence_map = defaultdict(list)

        for window in windows:
            # Filter out tiny or minimized windows
            if not window.is_substantial or window.is_minimized:
                continue

            app_name = window.app_name
            title = window.title.lower()

            # Score based on rules
            for activity_type, rules in self.ACTIVITY_RULES.items():
                score = 0.0
                evidence = []

                # Check app match
                if app_name in rules['apps']:
                    score += 1.0 * rules['weight']
                    evidence.append(f"App: {app_name}")

                # Check title patterns
                for pattern in rules['title_patterns']:
                    if re.search(pattern, title, re.IGNORECASE):
                        score += 0.5 * rules['weight']
                        evidence.append(f"Title pattern: {pattern}")

                if score > 0:
                    scores[activity_type] += score
                    evidence_map[activity_type].extend(evidence)

        # Get top activity
        if scores:
            top_activity = max(scores.items(), key=lambda x: x[1])
            activity_type, score = top_activity

            # Calculate confidence (normalize by window count)
            confidence = min(1.0, score / (len(windows) * 2))

            # Get primary app
            primary_app = Counter(
                w.app_name for w in windows if w.is_substantial
            ).most_common(1)[0][0] if windows else None

            return ActivityPattern(
                activity_type=activity_type,
                confidence=confidence,
                evidence=list(set(evidence_map[activity_type]))[:5],
                primary_app=primary_app
            )

        # Fallback to most common app
        if windows:
            most_common_app = Counter(
                w.app_name for w in windows if w.is_substantial
            ).most_common(1)

            if most_common_app:
                return ActivityPattern(
                    activity_type=ActivityType.UNKNOWN,
                    confidence=0.5,
                    evidence=[f"Primary app: {most_common_app[0][0]}"],
                    primary_app=most_common_app[0][0]
                )

        return ActivityPattern(
            activity_type=ActivityType.UNKNOWN,
            confidence=0.0,
            evidence=["No significant windows"],
            primary_app=None
        )

    def _detect_project(self, windows: List[WindowInfo]) -> Optional[str]:
        """
        Detect project name from window titles and paths

        Args:
            windows: All windows

        Returns:
            Detected project name or None
        """
        # Common project indicators
        project_patterns = [
            r'/([^/]+)\.git',  # Git repo name
            r'~/([^/]+)/',  # Home directory project
            r'/repos/([^/]+)',  # repos folder
            r'/projects/([^/]+)',  # projects folder
            r'/workspace/([^/]+)',  # workspace folder
            r'/([A-Z][A-Za-z0-9_-]+)\s*[-–—]',  # Project name before dash
        ]

        project_candidates = []

        for window in windows:
            title = window.title

            for pattern in project_patterns:
                match = re.search(pattern, title, re.IGNORECASE)
                if match:
                    project_name = match.group(1)
                    # Filter out common false positives
                    if project_name not in {'Documents', 'Desktop', 'Downloads', 'tmp', 'temp'}:
                        project_candidates.append(project_name)

        # Get most common project
        if project_candidates:
            most_common = Counter(project_candidates).most_common(1)[0]
            if most_common[1] >= 2:  # At least 2 mentions
                logger.info(f"[ANALYZER] 🎯 Detected project: {most_common[0]}")
                return most_common[0]

        return None

    def _detect_overall_activity(
        self,
        space_summaries: List[SpaceSummary]
    ) -> Optional[ActivityPattern]:
        """Detect overall workspace activity"""
        # Aggregate activities from active spaces
        activity_scores = defaultdict(float)

        for summary in space_summaries:
            if summary.is_active:
                activity_scores[summary.primary_activity.activity_type] += (
                    summary.primary_activity.confidence * summary.window_count
                )

        if not activity_scores:
            return None

        # Get dominant activity
        top_activity = max(activity_scores.items(), key=lambda x: x[1])
        activity_type, score = top_activity

        # Calculate overall confidence
        total_windows = sum(s.window_count for s in space_summaries if s.is_active)
        confidence = min(1.0, score / total_windows) if total_windows > 0 else 0.0

        return ActivityPattern(
            activity_type=activity_type,
            confidence=confidence,
            evidence=[f"Dominant across {len([s for s in space_summaries if s.is_active])} spaces"]
        )

    def _analyze_focus_pattern(self, space_summaries: List[SpaceSummary]) -> str:
        """
        Analyze focus pattern (deep work vs context switching)

        Returns:
            Description of focus pattern
        """
        active_spaces = [s for s in space_summaries if s.is_active]

        if len(active_spaces) <= 1:
            return "Focused work session"
        elif len(active_spaces) <= 3:
            return "Multi-tasking across a few contexts"
        elif len(active_spaces) <= 5:
            return "Context switching between several projects"
        else:
            return "High context switching - many active workspaces"
